import numpy as np
import pandas as pd

from pathlib import Path
from typing import Mapping, Union, List, Optional, Iterable, Dict

from pyannote.core import Segment, Annotation, Timeline, SlidingWindowFeature

from .features import s_power_db

NULL_LABEL = "-"


def get_label_weights(annotations: Annotation, use_labels: List = None):
    if not use_labels:
        use_labels = annotations.labels()
    use_labels = set(use_labels) - set([NULL_LABEL])
    # Calculate label weights, based on the inverse of their duration
    label_durations = {label: annotations.label_duration(label) for label in use_labels}
    total_duration = sum(label_durations.values())
    label_weights = {
        label: total_duration / duration for label, duration in label_durations.items()
    }
    weight_sum = sum(label_weights.values())
    # Return normalized weights (add up to 1)
    return {label: weight / weight_sum for label, weight in label_weights.items()}


def list_to_annotation(
    labels: Iterable, hop_s: float, ignore_label=NULL_LABEL
) -> Annotation:
    """
    Convert discrete labels to annotation
    hop_s: hop length in seconds
    """
    result = Annotation()
    t = 0.0
    segment_ini = 0.0
    latest_label = ignore_label
    for label in labels:
        if label != latest_label:
            if latest_label != ignore_label:  # end new segment
                result[Segment(segment_ini, t)] = latest_label
            if label != ignore_label:  # start new segment
                segment_ini = t
        latest_label = label
        t += hop_s
    if latest_label != ignore_label:  # end last segment
        result[Segment(segment_ini, t)] = latest_label
    return result


def get_power_mask(audio, power_th_db, win_len_s=0.5, hop_len_s=0):
    power_db = s_power_db(audio, win_len_s=win_len_s, hop_len_s=hop_len_s)
    boolean_swf = power_db > power_th_db
    return boolean_swf


def remove_silence_labels(audio, annotations, power_th_db, win_len_s=0.5, hop_len_s=0):
    power_timeline = mask_to_timeline(
        get_power_mask(audio, power_th_db, win_len_s, hop_len_s=hop_len_s)
    )
    return mask_annotations(annotations, power_timeline)


def mask_to_timeline(binary_signal: SlidingWindowFeature) -> Timeline:
    # binary_signal.data = binary_signal.data.astype(np.uint8)  # Make repr_feature work

    mask_timeline = Timeline()
    for seg, value in binary_signal:
        if value:
            mask_timeline.add(seg)
    mask_timeline = mask_timeline.support()  # glue contiguous segments
    # Return both formats: SlidingWindowFeature, Timeline
    return mask_timeline


def mask_annotations(annotations: Annotation, mask: Timeline) -> Annotation:
    # Crop
    cropped = annotations.crop(mask, mode="intersection")

    # print new durations
    for label in cropped.labels():
        old_duration = annotations.label_timeline(label).duration()
        new_duration = cropped.label_timeline(label).duration()
        print(f"{label}: {old_duration:.1f}s --> {new_duration:.1f}s")
    return cropped


def clean_annotations(
    annotations: Annotation, use_labels: Optional[List] = None
) -> Annotation:
    """
    Remove unused labels and potential overlaps
    """
    new_annotations = Annotation()
    for label in annotations.labels():
        if use_labels is None or label in use_labels:
            for seg in annotations.label_support(label):
                new_annotations[seg] = label
    return new_annotations


def shift_annotations(annotations: Annotation, t_shift: float):
    shifted_anns = Annotation()
    for win, _, label in annotations.itertracks(yield_label=True):
        shifted_anns[Segment(win.start + t_shift, win.end + t_shift)] = label
    return shifted_anns


def filter_annotations(annotations: Annotation, t_start: float, t_end: float):
    filtered_anns = Annotation()
    for win, _, label in annotations.itertracks(yield_label=True):
        if win.end > t_start and win.start < t_end:
            start = max(t_start, win.start)
            end = min(t_end, win.end)
            filtered_anns[Segment(start, end)] = label
    return filtered_anns


def segment_overlap(seg1, seg2):
    "Calculate overlap time over seg1 duration"
    max_ini = max(seg1.start, seg2.start)
    min_end = min(seg1.end, seg2.end)
    return max(0, min_end - max_ini) / seg1.duration


def load_labels_df(file_path: Union[str, Path]) -> pd.DataFrame:
    separator = " " if ".lab" in str(file_path) else "\t"
    df = pd.read_csv(
        str(file_path),
        sep=separator,
        names=["start", "end", "label"],
        dtype={"start": float, "end": float},
    )
    return df


def save_labels_df(df: pd.DataFrame, file_path: Union[str, Path]):
    df[["start", "end", "label"]].to_csv(
        str(file_path), sep="\t", index=False, header=False
    )


def load_annotations_csv(
    file_path: Union[str, Path],
    t0: Union[str, float, None] = None,
    remap: Mapping[str, str] = None,
) -> Annotation:
    if t0 is None:
        t0 = 0.0
    elif type(t0) is str:
        t0 = str_to_seconds(t0)

    # Detect delimiter " " or "\t" automatically
    with open(file_path) as f:
        sep = " " if f.readline().count(" ") == 2 else "\t"

    # Read CSV using detected separator
    df = pd.read_csv(str(file_path), sep=sep, names=["start", "end", "label"])
    df.loc[:, ["start", "end"]] = df[["start", "end"]].astype(float) - t0
    remap_labels_df(df, remap, inplace=True)
    return df_to_annotation(df)


def save_annotations_csv(annotation: Annotation, file_path: Union[Path, str]):
    with open(file_path, "w") as f:
        for segment, _, label in annotation.itertracks(yield_label=True):
            f.write(
                f"{segment.start:.3f} {segment.start + segment.duration:.3f} {label}\n"
            )


def df_to_annotation(df: pd.DataFrame, track: str = "_") -> Annotation:
    ann = Annotation()
    for _, s in df.iterrows():
        ann[Segment(start=s["start"], end=s["end"]), track] = str(s["label"])
    return ann


def annotation_to_df(ann: Annotation) -> pd.DataFrame:
    rows = []
    for segment, _, label in ann.itertracks(yield_label=True):
        rows.append({"label": label, "start": segment.start, "end": segment.end})
    return pd.DataFrame(rows)


def filter_labels_interval(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    df = df[(df["end"] > start) & (df["start"] < end)].copy()
    df["start"].clip(lower=start, inplace=True)
    df["end"].clip(upper=end, inplace=True)
    return df


def remap_labels_array(labels: np.ndarray, remap: Dict) -> np.ndarray:
    # Make sure that the target dtype has enough length for the target values (e.g: <U2)
    remap_dtype = np.array(remap.values()).dtype
    labels = labels.copy().astype(remap_dtype)  # Avoid changing by reference
    for label_origin, label_target in remap.items():
        labels[labels == label_origin] = label_target
    return labels


def remap_labels_df(
    df: pd.DataFrame, remap: Mapping, col: str = "label", inplace: bool = True
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    if remap is not None:
        for k in remap:
            df.loc[df[col] == k, col] = remap[k]
    return df


def discretize_annotations(
    annotations: Annotation,
    t_step: float,
    n_samples: int = 0,
    null_label: str = NULL_LABEL,
    use_labels: Optional[List] = None,
) -> np.ndarray:
    annotations = clean_annotations(annotations, use_labels=use_labels)
    if not n_samples:
        end_time = annotations.get_timeline(copy=False).extent().end
        n_samples = round(end_time / t_step)
    annotations_discrete = np.full(n_samples, null_label, dtype=str)
    for win, _, label in annotations.itertracks(yield_label=True):
        start = int(win.start / t_step)
        if start >= n_samples:
            continue
        end = min(int(win.end / t_step), n_samples)
        annotations_discrete[start:end] = label
    return annotations_discrete


def convert_df_samples_to_segments(df_samples, col_time, col_label, cols_aggregate):
    """
    Inverse of discretize_annotations to convert dataframe format:
    't,label,*cols_aggregate' -> 'start,end,label,*cols_aggregate'
    """
    dfs = df_samples.copy().sort_values(by=col_time)
    # Create a unique id per segment (intervals without label change)
    dfs["seg_id"] = (dfs[col_label] != dfs[col_label].shift(1)).cumsum()
    # Group by segment and calculate start/end times (min/max)
    df_segments = dfs.groupby("seg_id").agg(
        {"t": ["min", "max"], col_label: "first", **cols_aggregate}
    )
    # Remove/rename multi-index columns
    col_names = ["start", "end", "label"] + list(cols_aggregate.keys())
    df_segments.columns = df_segments.columns.map(
        {old_name: col_names[idx] for idx, old_name in enumerate(df_segments.columns)}
    )
    # Add each segment's duration to "end" column
    if len(df_samples) > 1:
        hop_s = df_samples[col_time].iloc[1] - df_samples[col_time].iloc[0]
        df_segments["end"] += hop_s
    else:
        print(f"WARNING: Creating null segment (start=end), {len(df_samples)=}")
    return df_segments.reset_index(drop=True)
