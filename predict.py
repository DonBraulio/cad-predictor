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
    save_annotations_csv,
    remap_labels_df,
    list_to_annotation,
)

# %%
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
    parser.add_argument(
        "--create_video", action="store_true", help="Create output video"
    )
    args = parser.parse_args()
    input_video = args.filename
    labels_path = args.labels
    model_weights = args.weights
    model_params = args.params
    create_video = args.create_video
except:  # noqa. Notebook mode
    input_video = "sample.mp4"
    labels_path = ""
    model_weights = "lstm.ckpt"
    model_params = "lstm.json"
    create_video = True
input_video = Path(input_video)

# Config params
use_labels = settings.CLASSIFICATION.USE_LABELS
remap = settings.CLASSIFICATION.REMAP
labels = settings.CLASSIFICATION.LABELS
win_density_s = settings.OUTPUTS.DENSITY_WINDOW_S

# Convert mp4 -> wav if needed
audio_path = input_video.with_suffix(".wav")
if input_video.suffix == ".mp4":
    cmd = cmd_mp4_to_wav(
        input_video,
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
elif input_video.suffix != ".wav":
    raise ValueError(f"Only WAV or MP4 files are supported. Got: {input_video.suffix}")


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

# %%
# Smooth predictions by using a moving window and majority voting
s_labels = df_predictions["prediction"].reset_index(drop=True)
win_s = 1.0
win_len = int(win_s / hop_s)
s_smoothed = s_labels.groupby(s_labels.index // win_len).apply(
    lambda group: group.mode().iloc[0]
)
hop_smoothed = win_len * hop_s  # may differ from win_s

# No smoothing
anns = list_to_annotation(df_predictions["prediction"], hop_s)

# %%
# Convert to annotations and save CSV
anns_smoothed = list_to_annotation(s_smoothed, hop_smoothed)
out_filepath = f"out_{audio_path.stem}.csv"
save_annotations_csv(
    anns_smoothed, out_filepath, sep=",", header=["t_start", "t_end", "cad"]
)


# %%
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


def show_predictions(t_density, label_densities):
    sns.set_palette("muted")
    fig, axes = plt.subplots(nrows=3, figsize=(8, 4.5), sharex=True)
    colors = sns.color_palette()
    for idx, label in enumerate(labels):
        density_ref, density_pred = label_densities[label]
        ax = axes[idx]
        color = colors[idx]
        ax.plot(
            t_density,
            density_ref,
            label=f"{label} (ref)",
            alpha=1,
            linestyle="-",
            color=color,
        )
        ax.plot(
            t_density,
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


label_densities = {}
for label in labels:
    d_ref, d_pred = get_density(df_predictions, label, remove_null=False)
    label_densities[label] = (d_ref, d_pred)
    density_len = len(d_ref)

t_density = np.linspace(start=0, stop=df_predictions.iloc[-1]["t"], num=density_len)
show_predictions(t_density, label_densities)
plt.savefig(f"out_density_{audio_path.stem}.pdf")
plt.savefig(f"out_density_{audio_path.stem}.png")

# %%
if create_video:
    from utils.video import create_video

    out_video = f"out_{input_video.stem}.mp4"
    print("[red]Creating output video with predictions...[/red]")
    create_video(
        anns_smoothed, label_densities, t_density, audio_t, input_video, out_video
    )
