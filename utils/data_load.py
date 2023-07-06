import torch
import pandas as pd

from rich import print
from typing import Dict
from pyannote.audio.core.io import Audio
from pyannote.core import Annotation, Segment

from utils.annotations import filter_labels_interval
from config import settings

fs = settings.AUDIO.SAMPLE_RATE


def load_audio(audio_path, verbose=False):
    print(f"Reading: {audio_path}")
    audio_l = Audio(sample_rate=fs, mono=True)
    t_audio, f_fs = audio_l(audio_path)
    assert fs == f_fs
    scale_factor = 1.0 / t_audio.max()
    t_audio = t_audio * scale_factor
    if verbose:
        print(f"Normalizing with factor: {scale_factor:.2f}")
    return t_audio


def concat_audio_and_annotations(df_labels, audio_list, groups, splits, verbose=False):
    df = df_labels[df_labels["group"].isin(groups) & df_labels["split"].isin(splits)]
    concat_audio_l = []
    concat_anns = Annotation()
    t_offset = 0.0
    for n_audio in df["n_audio"].unique():
        # Duration of the actual WAV audio
        audio_chunk = audio_list[n_audio]
        audio_duration = len(audio_chunk.squeeze()) / fs
        df_audio = df[df["n_audio"] == n_audio]
        group = list(df_audio["group"].unique())
        split = list(df_audio["split"].unique())
        input_key = list(df_audio["input_key"].unique())
        assert len(input_key) == 1, f"{len(input_key)=}"
        assert len(group) == 1, f"{len(group)=}"
        assert len(split) == 1, f"{len(split)=}"
        first_label = df_audio["start"].min()
        last_label = df_audio["end"].max()
        if verbose:
            print(
                f"Adding {input_key} {group=} {split=}"
                f" / labels: ({first_label:.1f}, {last_label:.1f})s"
                f" / {t_offset=:.1f}s {audio_duration=:.1f}s"
            )

        # Append annotations, with corresponding offset from previous audios
        for row in df_audio.itertuples():
            time_limit = audio_duration + 1.0  # Margin to warning
            if row.start > time_limit or row.end > time_limit:
                print(
                    f"[yellow]WARNING: Label {row.label} out of bounds.[/yellow]"
                    f" {audio_duration=:.1f}s, {row.start=:.1f}, {row.end=:.1f}"
                )
            start = min(audio_duration, row.start)
            end = min(audio_duration, row.end)
            concat_anns[Segment(start + t_offset, end + t_offset)] = row.label

        # Append audio
        concat_audio_l.append(audio_chunk)
        t_offset += audio_duration
    return torch.concat(concat_audio_l), concat_anns


def split_labels_df(df: pd.DataFrame, split_times: Dict) -> Dict[int, pd.DataFrame]:
    # Split labeled time into n_splits
    split_labels_dfs = {}
    for n_split, (start, end) in split_times.items():
        df_split = filter_labels_interval(df, start, end)
        df_split[["start", "end"]] -= start  # Time relative to split start
        df_split["offset"] = start  # So the original timestamps can be recovered
        split_labels_dfs[n_split] = df_split
    return split_labels_dfs


def split_audio(t_audio: torch.Tensor, split_times: Dict):
    t_audio = t_audio.squeeze()
    start = 0
    split_audios = {}
    for n_split, (start, end) in split_times.items():
        split_audios[n_split] = t_audio[int(start * fs) : int(end * fs)]
        start = end
    return split_audios
