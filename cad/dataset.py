import numpy as np
import typing as T

from rich import print
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
from pyannote.core import Annotation
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

from utils.annotations import discretize_annotations
from cad.featurizer import AudioFeaturizer, feat_to_str


class AudioFeaturesDataset(Dataset):
    def __init__(
        self,
        audio: Tensor,
        annotations: T.Optional[Annotation],
        featurizer: AudioFeaturizer,
        label_encoder: LabelEncoder,
        frames_per_sample: int,
        frames_hop_size: int = 0,
        frames_label: int = 0,
        use_labels: T.List = None,
        require_normalization: bool = True,
    ):
        self.audio = audio.squeeze()
        self.featurizer = featurizer
        self.use_labels = use_labels

        print("Dataset: Extracting features...")
        self.features, self.waveforms, feature_map = featurizer(self.audio)
        self.fs = featurizer.fs
        self.total_frames = self.features.shape[0]

        # Featurizer STFT windows
        # NOTE: these are audio discretized samples, not dataset samples
        hop_length_s = featurizer.hop_length_samples / featurizer.fs
        win_length_s = featurizer.win_length_samples / featurizer.fs

        # Number of frames (STFT windows) to form one single sample
        self.frames_per_sample = int(frames_per_sample)
        self.frames_hop_size = int(frames_hop_size or frames_per_sample)
        self.frames_label = int(frames_label or frames_per_sample)

        self.samples_hop_s = self.frames_hop_size * hop_length_s
        self.samples_win_s = (self.frames_per_sample - 1) * hop_length_s + win_length_s

        # Length of the dataset
        self.total_samples = (
            self.total_frames - self.frames_per_sample
        ) // self.frames_hop_size + 1
        self.active_duration = (
            self.total_samples - 1
        ) * self.samples_hop_s + self.samples_win_s

        # Actual duration in seconds, after considering entire windows
        print(f"Samples of {self.samples_win_s:.4f}s ({self.frames_per_sample} frames)")
        frames_overlap = self.frames_hop_size < self.frames_per_sample
        print(f"Samples hop: {self.samples_hop_s:.4f}s | Overlapping: {frames_overlap}")

        if annotations is None:
            annotations = Annotation()
        self.labels_t = discretize_annotations(
            annotations,
            t_step=hop_length_s,
            n_samples=self.total_frames,
            use_labels=use_labels,
        )

        self.label_encoder = label_encoder
        try:
            self.labels = self.label_encoder.transform(self.labels_t)
            print("LabelEncoder was already fitted")
        except NotFittedError:
            print("Fitting LabelEncoder")
            self.labels = self.label_encoder.fit_transform(self.labels_t)
        self._cached_labels = -1 * np.ones(self.total_samples, dtype=np.int_)

        assert self.labels.shape[0] == self.features.shape[0]
        print(f"Feature map: {feat_to_str(feature_map)}")
        self._ignore_normalization = not require_normalization
        self._normalized = False
        self._iter_idx = 0

    def get_active_audio(self):
        # This should not need round() unless for some machine error
        self.active_len_fs = round(self.active_duration * self.fs)
        assert self.active_len_fs <= len(self.audio)
        return self.audio[: self.active_len_fs]

    def _get_sample_label(self, sample_idx, w_start, w_end):
        # Cached labels: important speedup
        if self._cached_labels[sample_idx] != -1:
            return self._cached_labels[sample_idx]

        # Label for the whole sequence: majority in last frames (frames_label)
        label_start = max(w_start, w_end - self.frames_label)
        seq_labels = self.labels[label_start:w_end]
        vals, counts = np.unique(seq_labels, return_counts=True)
        res = vals[np.argmax(counts)]
        self._cached_labels[sample_idx] = res
        return res

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        result = self[self._iter_idx]
        self._iter_idx += 1
        return result

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> T.Tuple[Tensor, Tensor, np.array]:
        """
        Get a sequence of consecutive audio frames
        with length defined by frames_per_sample.
        Label given by the most frequent one along the sequence.
        """
        if index >= len(self):
            raise KeyError(f"Length: {len(self)} | {index=}")
        assert self._normalized or self._ignore_normalization

        w_start = index * self.frames_hop_size
        w_end = w_start + self.frames_per_sample
        # detach to avoid memory issues when an item is moved to GPU
        return (
            self.waveforms[w_start:w_end].detach(),
            self.features[w_start:w_end].detach(),
            self._get_sample_label(index, w_start, w_end),
        )

    def normalize_features(self, mean: Tensor = None, std: Tensor = None):
        if mean is None:
            mean = self.features.mean(axis=0)
        if std is None:
            std = self.features.std(axis=0)
        if not self._normalized:
            self.features -= mean
            self.features /= std
            self._normalized = True
        else:
            print("Dataset was already normalized, ignoring...")
        return mean, std


def get_count_per_label(dset):
    count_per_label = {}
    for s in dset:
        label = s[-1]  # Last element in the tuple should be always label
        try:
            count_per_label[label] += 1
        except KeyError:
            count_per_label[label] = 1
    return count_per_label, sum(count_per_label.values())


def print_dataset_info(dset, label_encoder):
    count_per_label, total = get_count_per_label(dset)
    for idx, label in enumerate(label_encoder.classes_):
        count = count_per_label[idx] if idx in count_per_label else 0
        percent = 100 * count / total
        print(f"{label} ({idx}): {count} samples ({percent:.2f}%)")
    return count_per_label, total


def split_train_val(dset, seq_len, seq_hop, val_frac=0.2):
    """
    Split given dataset into train+validation subsets.
    """
    seq_hop = seq_hop or seq_len  # Default: no overlap
    if seq_hop >= seq_len:  # eazy peazy: no overlapping between samples
        train_dataset, val_dataset = random_split(dset, [1 - val_frac, val_frac])
    else:
        """
        If consecutive samples overlap (e.g: dset[0] and dset[1] share (seq_len - seq_hop)
        samples), everything is more complicated.
        This function must reserve val_frac of the audio duration for validation, but:
        -> validation samples must not overlap, so val_idxs must be separated at least
        by seq_len samples.
        -> train samples can overlap between them, but not with validation samples,
        so when one val_idx is removed from train, also remove up to val_idx+seq_len
        """
        # First of all, set dset length to multiple of seq_len
        dset_idxs = np.arange(len(dset))  # all possible overlapping sample indices

        # Step size needed to avoid overlapping between samples
        gap_no_overlap = int(np.ceil(seq_len / seq_hop))
        idxs_no_overlap = dset_idxs[::gap_no_overlap]  # idxs without overlap
        # Select validation indices as val_frac of those without overlapping
        val_idxs = np.random.choice(
            idxs_no_overlap, int(val_frac * len(idxs_no_overlap))
        )
        # Then remove from train all indices that overlap with any validation sample
        for val_idx in val_idxs:
            # Each validation sample overlaps with all samples in [idx : idx+seq_len]
            overlap_range = np.arange(
                val_idx, min(val_idx + gap_no_overlap, len(dset_idxs))
            )
            dset_idxs[overlap_range] = -1  # Flag as unavailable for train_idxs
        train_idxs = dset_idxs[dset_idxs != -1]

        train_dataset = Subset(dset, train_idxs)
        val_dataset = Subset(dset, val_idxs)
    return train_dataset, val_dataset
