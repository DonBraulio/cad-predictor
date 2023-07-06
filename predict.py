#!/usr/bin/env python
import argparse
from pathlib import Path

from rich import print

from config import settings
from utils.system import cmd_mp4_to_wav, run_command


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument(
    "--weights", default="lstm.ckpt", help="File with the model weights."
)
parser.add_argument("--model", default="LSTM", help="Model name. E.g: LSTM, XGBoost")
args = parser.parse_args()

input_path = Path(args.filename)
model_weights = args.weights
model_name = args.model

audio_path = input_path.with_suffix(".wav")

# mp4 -> wav if needed
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
