# %%
import torch
import json
import numpy as np
import typing as T
import pytorch_lightning as pl

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from ..dataset import AudioFeaturesDataset
from ..featurizer import AudioFeaturizer
from .base import CADPredictor, Array, FilePath
from config import settings
from utils.annotations import NULL_LABEL, get_power_mask


class CADPredictorLSTM(CADPredictor):
    def __init__(self, weights_file, params_file: FilePath):
        super().__init__(weights_file, params_file)
        self._silence_th = settings.AUDIO.SILENCE_TH_DB
        self._hop_frames = settings.LSTM.PREDICT_HOP_FRAMES

    def load_from_params(self, weights_file: FilePath, params_file: FilePath):
        params = json.load(open(params_file, "r"))
        lstm_params = params["lstm_params"]
        audio_lstm = AudioClassifier.load_model(weights_file, AudioLSTM(**lstm_params))
        audio_lstm.eval()

        # Loaded instance
        self.featurizer = AudioFeaturizer(cache=True, verbose=True)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(params["label_classes"])
        self._model = audio_lstm
        self._params = params
        self._null_label_idx = self._label_encoder.transform([NULL_LABEL])[0]

    def predict(self, audio: Array) -> T.Tuple[np.array, np.array, np.array]:
        dset = AudioFeaturesDataset(
            audio,
            None,  # No labels
            self.featurizer,
            self._label_encoder,
            frames_per_sample=self._params["frames_per_sample"],
            frames_hop_size=self._hop_frames,
            frames_label=self._params["frames_label"],
        )
        audio = dset.get_active_audio()  # Cut to actually used audio (sample size)

        # Apply train mean/std to normalize test data as well
        train_mean = torch.tensor(self._params["normalization"]["train_mean"])
        train_std = torch.tensor(self._params["normalization"]["train_std"])
        _ = dset.normalize_features(train_mean, train_std)

        # Predictions without removing silences
        dloader = DataLoader(dset, self._params["batch_size"], shuffle=False)
        y_preds = []
        y_pred_scores = []
        with torch.no_grad():
            for wavs, feats, _ in dloader:
                log_softmax = self._model(feats)
                preds = torch.argmax(log_softmax, dim=1)  # label == position
                y_pred_scores += torch.exp(
                    log_softmax[np.arange(len(preds)), preds]
                ).tolist()
                y_preds += list(preds)
        y_preds = np.array(y_preds)
        y_pred_scores = np.array(y_pred_scores)

        # Null silences
        mask_power = get_power_mask(
            audio,
            self._silence_th,
            win_len_s=dset.samples_win_s,
            hop_len_s=dset.samples_hop_s,
        )
        assert (
            abs(len(y_preds) - mask_power.data.shape[0]) <= 1
        ), f"Error: {len(y_preds)=} | {mask_power.data.shape=}"
        mask_power_ = mask_power.data.squeeze()[: len(y_preds)]
        y_preds[~(mask_power_.astype(bool))] = self._null_label_idx
        y_pred_labels = self._label_encoder.inverse_transform(y_preds)

        # Create time axis and return
        t_preds = np.linspace(0, dset.active_duration - dset.samples_win_s, len(dset))
        return t_preds, y_pred_labels, y_pred_scores


class AudioClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_function, labels, label_weights):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.labels = labels
        self.label_weights = label_weights
        self.init_lr = 1e-3

    @staticmethod
    def load_model(checkpoint_path, model):
        checkpoint = torch.load(checkpoint_path)
        model_weights = checkpoint["state_dict"]
        # update keys by dropping `model.` (used in AudioClassifier)
        for key in list(model_weights):
            if "model." not in key:
                del model_weights[key]  # discard audio_classifier.loss_function
            else:
                model_weights[key.replace("model.", "")] = model_weights.pop(key)
        model.load_state_dict(model_weights)
        return model

    def get_accuracy(self, out_logits, target):
        self.label_weights = self.label_weights.to(self.device)
        preds = torch.argmax(out_logits, dim=1)
        correct = (preds == target).float()
        weighted_correct = self.label_weights[target] * correct
        return torch.mean(correct).item(), torch.mean(weighted_correct).item()

    def training_step(self, batch, batch_idx):
        wavs, feats, labels = batch
        outs = self.model(feats)
        loss = self.loss_function(outs, labels)
        self.log("train_loss", loss)  # not accumulated
        return loss

    def validation_step(self, batch, batch_idx):
        wavs, feats, labels = batch
        outs = self.model(feats)
        loss = self.loss_function(outs, labels)
        self.log("validation_loss", loss, prog_bar=True)  # auto accumulated
        accuracy, weighted_acc = self.get_accuracy(outs, labels)
        self.log("val_acc", accuracy, prog_bar=True)
        self.log("val_acc_weighted", weighted_acc)
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])

    def test_step(self, batch, batch_idx):
        wavs, feats, labels = batch
        outs = self.model(feats)
        loss = self.loss_function(outs, labels)
        self.log("test_loss", loss)  # auto accumulated
        accuracy, weighted_acc = self.get_accuracy(outs, labels)
        self.log("test_acc", accuracy)
        self.log("test_acc_weighted", weighted_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=5
                ),
                "min_lr": 1e-5,
                "monitor": "validation_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }


class AudioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(AudioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Using log to combine with NLLLoss

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # (N,L,H) -> get H for last sequence element (t=L)
        return self.log_softmax(out)
