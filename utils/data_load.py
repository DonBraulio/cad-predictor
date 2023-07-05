import torch
import pandas as pd

from rich import print
from pathlib import Path
from typing import Dict, List
from pyannote.audio.core.io import Audio
from pyannote.core import Annotation, Segment

from utils.annotations import (
    str_to_seconds,
    load_annotations_csv,
    load_labels_df,
    filter_labels_interval,
)
from diarization.dataset import SegmentsDataset

from config import settings

fs = settings.AUDIO.SAMPLE_RATE
win_sec = settings.EMBEDDINGS.WIN_SECS
hop_sec = settings.EMBEDDINGS.HOP_SECS
min_seg_length = settings.EMBEDDINGS.MIN_SEG_LENGTH
use_labels = settings.CLASSIFICATION.USE_LABELS


def load_project_data(return_data_map=False):
    """
    Load all train/test data
    """
    test_size = settings.GENERAL.TEST_SIZE
    rows_list = []
    audio_list = []
    data_map = {}
    for input_key, input_params in settings.INPUTS.items():

        # Load audio and labels
        file_name = input_params.NAME
        path_labels = f"{Path(settings.LABELS.SAVE_DIR) / file_name}.txt"
        path_audio = f"{(Path(settings.AUDIO.SAVE_DIR) / file_name)}.wav"
        df_labels = load_labels_df(path_labels).sort_values(by=["start"])
        audio = load_audio(path_audio)

        # Split into n_splits
        split_times = get_split_limits(audio, df_labels, input_params.SPLITS, test_size)
        split_labels = split_labels_df(df_labels, split_times)
        split_audios = split_audio(audio, split_times)
        data_map[input_key] = split_times

        for n_split in input_params.SPLITS:
            n_audio = len(audio_list)
            audio_list.append(split_audios[n_split])
            for label_row in split_labels[n_split].to_dict(orient="records"):
                label_row["input_key"] = input_key
                label_row["split"] = n_split
                label_row["group"] = input_params.group
                label_row["n_audio"] = n_audio
                rows_list.append(label_row)  # also contains: start, end, label
    df_labels = pd.DataFrame(rows_list).sort_values(by=["n_audio", "start"])

    if return_data_map:
        return df_labels, audio_list, data_map
    else:
        return df_labels, audio_list


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


def get_split_limits(
    audio: torch.Tensor,
    df_labels: pd.DataFrame,
    splits_order: List,
    test_size: float = 0.5,
):
    """
    Calculate time boundaries (in seconds) for each split, in the original audio
    e.g: {0: (100.0, 200.0), 1: (0, 20), 2: (20, 40), 3: (40, 60) ...}
    Split sizes are calculated using the "labeled_duration",
    and except for test split 0 (with size test_size), all other splits are equal size.
    splits_order=[1, 0, 3, 4, 5]  means that test_split is the second audio chunk
    """
    labels_start, labels_end = df_labels["start"].min(), df_labels["end"].max()
    audio_duration = len(audio.squeeze()) / fs
    if labels_end > audio_duration + 1.0:
        print(f"WARN: Labels out of bounds ({audio_duration=:.1f}, {labels_end=:.1f})")
    end_time = min(audio_duration, labels_end)  # Ignore labels out of bounds
    labeled_duration = end_time - labels_start

    # Calculate train/test splits duration
    test_duration = labeled_duration * test_size
    n_train_splits = len(splits_order) - 1  # don't count test split 0
    train_splits_duration = (labeled_duration - test_duration) / n_train_splits

    # Iterate through chunks and assign them to split numbers, with start/end times
    split_start = 0.0  # First chunk always starts at time 0 (even if not labeled)
    split_boundaries = {}
    for position, n_split in enumerate(splits_order):
        split_duration = test_duration if n_split == 0 else train_splits_duration
        if position == 0:  # first chunk is a bit larger if labels_start != 0
            split_duration += labels_start
        split_end = split_start + split_duration
        split_boundaries[n_split] = (split_start, split_end)
        split_start = split_end  # Next split
    return split_boundaries


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


def load_dev_audio(file_name):
    labels_t0 = 0

    # Ignore differences between boy, girl, chorus, etc
    labels_remap = settings.CLASSIFICATION.REMAP

    input_file = (Path(settings.GENERAL.DEV_DIR) / file_name).with_suffix(".wav")
    annotations_file = (Path(settings.GENERAL.DEV_DIR) / file_name).with_suffix(".txt")

    print(f"Loading audio file {file_name}. Annotations: {annotations_file.exists()}")

    audio_l = Audio(sample_rate=fs, mono=True)
    t_audio, f_fs = audio_l(input_file)
    assert fs == f_fs
    annotations = load_annotations_csv(
        annotations_file, t0=labels_t0, remap=labels_remap
    )
    return t_audio, annotations, input_file, annotations_file


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


def load_audio_and_annotations(audio_id, labels_remap=None):
    input_params = settings.INPUTS[audio_id]
    file_name = input_params.NAME

    input_file = (Path(settings.AUDIO.SAVE_DIR) / file_name).with_suffix(".wav")
    anns_file = (Path(settings.LABELS.SAVE_DIR) / file_name).with_suffix(".txt")

    print(f"Loading audio file for {audio_id}. Annotations: {anns_file.exists()}")

    t_audio = load_audio(input_file)

    annotations = load_annotations_csv(anns_file, remap=labels_remap)
    return t_audio, annotations, input_file, anns_file


def load_input(input_key, t_start, t_end=0, labels_remap=None):
    i_audio, i_anns, _, _ = load_audio_and_annotations(input_key, labels_remap)
    i_audio = i_audio.squeeze()  # Just in case

    # Interval to use (in seconds and samples)
    total_length = len(i_audio) / fs
    t_end = t_end or total_length
    end_idx = int(t_end * fs)
    start_idx = int(t_start * fs)
    print(f"Loading {input_key}: ({t_start=:.2f}s, {t_end=:.2f}s)/{total_length=:.2f}s")

    anns = Annotation()
    # Create new annotations for train/test, shifted in time (previous audios)
    for seg, _, label in i_anns.itertracks(yield_label=True):
        if (
            seg.start > t_start and seg.start < t_end  # label starts inside interval
        ) or (
            seg.end > t_start and seg.end < t_end  # label ends inside interval
        ):
            seg_offset = Segment(
                max(seg.start - t_start, 0), min(seg.end - t_start, t_end)
            )
            anns[seg_offset] = label
    return i_audio[start_idx:end_idx], anns


def load_inputs(input_keys_splits: Dict[str, float], labels_remap=None):
    """Load audios concatenated as train/test, with their corresponding
    annotations, shifted in time accordingly.

    Args:
        input_keys (Dict[str, float]): file keys to load with their
        train split time in seconds. E.g: {"eng1": 1000, "pc8": 500}

    Returns:
        (train_audios: torch.Tensor,
        test_audios: torch.Tensor,
        train_annotations: Annotation,
        test_annotations: Annotation,
        input_offsets: dict{key: (train_start, test_start)})
    """

    train_audio_l = []
    test_audio_l = []
    train_anns = Annotation()
    test_anns = Annotation()
    train_offset = 0.0
    test_offset = 0.0
    input_offsets = {}  # map with time offsets for each input (train/test)

    # Iterate over each input audio and its train/test split time
    for input_key, split_t in input_keys_splits.items():
        i_audio, i_anns, _, _ = load_audio_and_annotations(input_key, labels_remap)
        i_audio = i_audio.squeeze()  # Just in case
        split_idx = int(split_t * fs)

        # Append train and test audios to a list
        # split_t=0           --> all test, no train
        # split_t >= duration --> all train, no test
        train_duration = test_duration = 0.0
        if split_idx > 0:  # add audio to train
            train_audio_l.append(i_audio[:split_idx])
            train_duration = train_audio_l[-1].shape[0] / fs
        if split_idx < i_audio.shape[0]:  # add audio to test
            test_audio_l.append(i_audio[split_idx:])
            test_duration = test_audio_l[-1].shape[0] / fs

        input_offsets[input_key] = (
            train_offset,  # train start
            train_offset + train_duration,  # train end
            test_offset,  # test start
            test_offset + test_duration,  # test end
        )

        print(
            f"Using {input_key}: {train_duration:.1f}s->train | {test_duration:.1f}->test"
        )

        # Create new annotations for train/test, shifted in time (previous audios)
        for seg, _, label in i_anns.itertracks(yield_label=True):
            if seg.end < split_t:  # (0, split_t)->Train
                seg_offset = Segment(seg.start + train_offset, seg.end + train_offset)
                train_anns[seg_offset] = label
            else:  # (split_t, end)->Test
                seg_offset = Segment(
                    max(0, seg.start - split_t) + test_offset,
                    seg.end - split_t + test_offset,
                )
                test_anns[seg_offset] = label

        # Add offsets for next train/test audios to append
        train_offset += train_duration
        test_offset += test_duration

    # Concatenate all audios from list into a single tensor
    train_audio = torch.concat(train_audio_l)
    test_audio = torch.concat(test_audio_l)

    return train_audio, test_audio, train_anns, test_anns, input_offsets


def load_datasets(train_test_per_id, only_test=None):
    """
    train_test_per_id is a dictionary with the split fraction per audio
    example: {"pc1": 0.5, "pc2": 0.5, "eng1": 0.5, "eng2": 0}
    only_test is a single id like "eng3", returned separately as test_2
    """
    test_audio_l = []
    train_audio_l = []
    train_anns = Annotation()
    test_anns = Annotation()
    train_offset = 0.0
    test_offset = 0.0

    labels_remap = settings.CLASSIFICATION.REMAP

    # Append other train audios
    for audio_id, split_frac in train_test_per_id.items():
        i_audio, i_anns, _, _ = load_audio_and_annotations(audio_id, labels_remap)
        split_idx = int(i_audio.shape[1] * split_frac)
        split_t = split_idx / fs

        train_audio_l.append(i_audio[:, :split_idx])
        test_audio_l.append(i_audio[:, split_idx:])

        for seg, _, label in i_anns.itertracks(yield_label=True):
            if seg.end < split_t:  # (0, split_t)->Train
                seg_offset = Segment(seg.start + train_offset, seg.end + train_offset)
                train_anns[seg_offset] = label
            else:  # (split_t, end)->Test
                seg_offset = Segment(
                    max(0, seg.start - split_t) + test_offset,
                    seg.end - split_t + test_offset,
                )
                test_anns[seg_offset] = label

        # Add offsets for next train/test audios to append
        train_offset += split_idx / fs
        test_offset += (i_audio.shape[1] - split_idx) / fs

    train_audio = torch.concat(train_audio_l, dim=1)
    test_audio = torch.concat(test_audio_l, dim=1)

    # Create datasets for train/test
    print("[yellow]Creating TRAIN dataset...[/yellow]")
    train_dset = SegmentsDataset(
        train_audio, train_anns, win_sec, hop_sec, fs, min_seg_length, use_labels
    )
    print("[yellow]Creating TEST 1 dataset...[/yellow]")
    test_dset = SegmentsDataset(
        test_audio, test_anns, win_sec, hop_sec, fs, min_seg_length, use_labels
    )
    test_only_dset = None
    if only_test:
        print("[yellow]Creating TEST 2 dataset...[/yellow]")
        i_audio, i_anns, _, _ = load_audio_and_annotations(only_test, labels_remap)
        test_only_dset = SegmentsDataset(
            i_audio, i_anns, win_sec, hop_sec, fs, min_seg_length, use_labels
        )
    return train_dset, test_dset, test_only_dset
