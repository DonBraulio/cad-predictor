# %%
import torch
import json
import numpy as np
import typing as T

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pathlib import Path
from rich import print
from speechbrain.lobes.features import (
    ContextWindow,
)

from ..featurizer import AudioFeaturizer
from .base import CADPredictor, Array, FilePath
from config import settings
from utils.annotations import NULL_LABEL, get_power_mask


class CADPredictorXGB(CADPredictor):
    def __init__(self, weights_file: FilePath, params_file: FilePath):
        super().__init__(weights_file, params_file)
        self._silence_th = settings.AUDIO.SILENCE_TH_DB

    def load_from_params(self, weights_file: FilePath, params_file: FilePath):
        params = json.load(open(params_file, "r"))
        context_lr = params["context_window"]
        # Load pretrained model
        self._model = xgb.Booster({"nthread": 1})
        self._model.load_model(weights_file)
        self._feature_names = self._model.feature_names
        self.featurizer = AudioFeaturizer(verbose=True)
        self._compute_cw = ContextWindow(
            left_frames=context_lr, right_frames=context_lr
        )
        self._hop_s = self.featurizer.hop_length_ms / 1000
        self._win_s = self.featurizer.win_length_ms / 1000
        self._fs = self.featurizer.fs
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(params["label_classes"])

    def predict(self, audio: Array) -> T.Tuple[np.array, np.array, np.array]:
        features_c, _, _ = self.featurizer(audio)

        # Context window (add past and ahead frames)
        with torch.no_grad():
            features = (
                self._compute_cw(torch.tensor(features_c).unsqueeze(0))
                .squeeze()
                .numpy()
            )

        y_preds = self._model.predict(
            xgb.DMatrix(features, feature_names=self._feature_names)
        ).astype(int)
        y_preds = np.array(y_preds)
        y_pred_scores = np.ones_like(y_preds)  # No scores for xgboost

        # Null silences
        mask_power = get_power_mask(
            audio, self._silence_th, win_len_s=self._win_s, hop_len_s=self._hop_s
        )
        assert (
            abs(len(y_preds) - mask_power.data.shape[0]) <= 1
        ), f"Error: {len(y_preds)=} | {mask_power.data.shape=}"
        y_pred_labels = self._label_encoder.inverse_transform(y_preds)
        mask_power_ = mask_power.data.squeeze()[: len(y_preds)]
        y_pred_labels[~(mask_power_.astype(bool))] = NULL_LABEL

        # Create time axis and return
        t_preds = np.linspace(0, len(audio) / self._fs, len(y_preds))
        return t_preds, y_pred_labels, y_pred_scores
