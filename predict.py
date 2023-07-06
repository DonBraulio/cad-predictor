#!/usr/bin/env python
# %%
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from pyannote.core import Annotation

from config import settings
from cad.models.base import CADPredictor
from utils.metrics import win_density
from utils.data_load import load_audio
from utils.system import cmd_mp4_to_wav, run_command
from utils.annotations import (
    NULL_LABEL,
    discretize_annotations,
    load_annotations_csv,
    remap_labels_df,
)


# Command line params
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument(
        "--weights", default="lstm.ckpt", help="File with the model weights."
    )
    parser.add_argument(
        "--params", default="lstm.json", help="File with the model JSON params."
    )
    parser.add_argument(
        "--labels", default="", help="File with manual ground-truth CAD labels."
    )
    args = parser.parse_args()
    input_path = args.filename
    labels_path = args.labels
    model_weights = args.weights
    model_params = args.params
except:  # noqa. Notebook mode
    input_path = "sample.mp4"
    labels_path = ""
    model_weights = "lstm.ckpt"
    model_params = "lstm.json"
input_path = Path(input_path)

# Config params
use_labels = settings.CLASSIFICATION.USE_LABELS
remap = settings.CLASSIFICATION.REMAP
labels = settings.CLASSIFICATION.LABELS
win_density_s = settings.OUTPUTS.DENSITY_WINDOW_S

# Convert mp4 -> wav if needed
audio_path = input_path.with_suffix(".wav")
if input_path.suffix == ".mp4":
    cmd = cmd_mp4_to_wav(
        input_path,
        audio_path,
        sample_rate=settings.AUDIO.SAMPLE_RATE,
    )
    print("[green]Extracting WAV from MP4 file using ffmpeg...[/green]")
    print(f"[yellow]{' '.join(cmd)}[/yellow]")

    ret_code, err_msg = run_command(cmd)

    if ret_code == 0:
        print(f"[green]Created WAV: [/green][yellow]{audio_path}[/yellow]")
    else:
        print(err_msg)
        print(f"[red]Check msg above, process ended with error code: {ret_code}[/red]")
        raise ValueError(f"Error code: {ret_code}")
elif input_path.suffix != ".wav":
    raise ValueError(f"Only WAV or MP4 files are supported. Got: {input_path.suffix}")


# Instantiate model
print(f"Loading model params: {model_params}")
model_paths = {
    "lstm": "cad.models.lstm_classifier.CADPredictorLSTM",
    "xgboost": "cad.models.xgboost_classifier.CADPredictorXGB",
    "diarization": "cad.models.pyannote_diarization.CADPredictorDiarization",
}
params = json.load(open(model_params, "r"))
model_path = model_paths[params["model_type"]]
cad_model = CADPredictor.create_instance(model_path, model_weights, model_params)

# Load input audio and annotations (optional)
audio_t = load_audio(audio_path).squeeze()
if labels_path and Path(labels_path).exists():
    annotations = load_annotations_csv(labels_path, remap=remap)
else:
    annotations = Annotation()

# Run prediction
print(f"Running CAD on {audio_path}...")
pred_t, pred_labels, pred_score = cad_model.predict(audio_t)

# Create output CSV
hop_s = pred_t[1] - pred_t[0]
ref_labels = discretize_annotations(
    annotations,
    t_step=hop_s,
    use_labels=use_labels,
    n_samples=len(pred_labels),
)
df_predictions = pd.DataFrame(
    {
        "t": pred_t,
        "prediction": pred_labels,
        "reference": ref_labels,
        "score": pred_score,
    }
)
df_predictions = remap_labels_df(df_predictions, remap, col="reference", inplace=False)
remap_labels_df(df_predictions, remap, col="prediction", inplace=True)

df_predictions.round(2).to_csv(f"output_{audio_path.stem}.csv")


# %%
# Create output image (density function)
def get_density(df_preds, label, remove_null=False):
    hop_s = df_preds.iloc[1]["t"] - df_preds.iloc[0]["t"]
    win_len = round(win_density_s // hop_s)
    print(f"{win_len=}")
    if remove_null:
        df_preds = df_preds[
            (df_preds["reference"] != NULL_LABEL)
            & (df_preds["prediction"] != NULL_LABEL)
        ]
    label_mask_ref = df_preds["reference"] == label
    label_mask_pred = df_preds["prediction"] == label
    return win_density(label_mask_ref, win_size=win_len), win_density(
        label_mask_pred, win_size=win_len
    )


def show_predictions(df_preds, remove_nulls=False):
    sns.set_palette("muted")
    fig, axes = plt.subplots(nrows=3, figsize=(8, 4.5), sharex=True)
    colors = sns.color_palette()
    for idx, label in enumerate(labels):
        ax = axes[idx]
        density_ref, density_pred = get_density(
            df_preds, label, remove_null=remove_nulls
        )
        t = np.arange(len(density_pred)) * win_density_s
        color = colors[idx]
        ax.plot(
            t, density_ref, label=f"{label} (ref)", alpha=1, linestyle="-", color=color
        )
        ax.plot(
            t,
            density_pred,
            label=f"{label} (pred)",
            alpha=0.8,
            linestyle=":",
            color=color,
        )
        ax.set_ylabel("Density")
        ax.set_ylim([-0.05, 1.05])
        ax.grid()
        ax.legend(loc="upper right")
    ax.set_xlabel("t (s)")


show_predictions(df_predictions)
plt.savefig(f"out_density_{audio_path.stem}.pdf")
plt.savefig(f"out_density_{audio_path.stem}.png")

# %%
