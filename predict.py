#!/usr/bin/env python
import json
import argparse
from pathlib import Path

import pandas as pd
from rich import print
from pyannote.core import Annotation

from config import settings
from cad.models.base import CADPredictor
from utils.data_load import load_audio
from utils.system import cmd_mp4_to_wav, run_command
from utils.annotations import (
    discretize_annotations,
    load_annotations_csv,
    remap_labels_df,
)


# Command line params
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
input_path = Path(args.filename)
labels_path = args.labels
model_weights = args.weights
model_params = args.params

# Config params
use_labels = settings.CLASSIFICATION.USE_LABELS
remap = settings.CLASSIFICATION.REMAP

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
df_remap = remap_labels_df(df_predictions, remap, col="reference", inplace=False)
remap_labels_df(df_remap, remap, col="prediction", inplace=True)

df_remap.to_csv(f"output_{audio_path.stem}.csv")
