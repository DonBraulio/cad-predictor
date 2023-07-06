import json
import time
import torch
import numpy as np
import typing as T
import torch.nn.functional as F

from pathlib import Path
from enum import Enum
from torch import Tensor
from speechbrain.lobes.features import (
    STFT,
    Filterbank,
    DCT,
    spectral_magnitude,
)
from librosa.feature import (
    spectral_centroid,
    spectral_flatness,
    spectral_bandwidth,
    spectral_contrast,
    rms,
)
from utils.features import rms_to_db
from config import settings


class FeatName(Enum):
    MFCC = "MFCC"
    SPEC_FLAT = "Spec.Flatness"
    SPEC_MEAN = "Spec.Mean"
    SPEC_BW = "Spec.Bandwidth"
    SPEC_CONTRAST = "Spec.Contrast"
    PWR_DB = "RMS(dB)"


def feat_to_str(feats: T.List[FeatName]) -> T.List[str]:
    return [f.name for f in feats]


def str_to_feat(names: T.List[str]) -> T.List[FeatName]:
    return [FeatName[f] for f in names]


class AudioFeaturizer:
    def __init__(self, verbose=False, cache=True):
        self.verbose = verbose
        self.fs = settings.AUDIO.SAMPLE_RATE

        hop_length_ms = settings.FEATURIZER.HOP_LEN_MS
        win_length_ms = settings.FEATURIZER.WIN_LEN_MS
        f_min = settings.FEATURIZER.F_MIN
        f_max = settings.FEATURIZER.F_MAX
        n_mels = settings.FEATURIZER.N_MELS
        n_mfcc = settings.FEATURIZER.N_MFCC

        # Sliding window parameters
        win_length_s = win_length_ms * 1e-3
        hop_length_s = hop_length_ms * 1e-3
        self.win_length_samples = int(self.fs * win_length_s)
        self.hop_length_samples = int(self.fs * hop_length_s)
        # Actual lengths after considering fs resolution
        self.win_length_ms = 1000 * self.win_length_samples // self.fs
        self.hop_length_ms = 1000 * self.hop_length_samples // self.fs

        # Time domain -> time/freq domain (STFT)
        self.compute_STFT = STFT(
            sample_rate=self.fs,
            n_fft=self.win_length_samples,
            win_length=win_length_ms,
            hop_length=hop_length_ms,
            center=False,
        )

        # magnitude(STFT) -> Mel filterbank
        self.compute_fbanks = Filterbank(
            sample_rate=self.fs,
            n_fft=self.win_length_samples,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            log_mel=True,
            filter_shape="triangular",
        )

        # Mel -> Cosine transform
        self.compute_dct = DCT(input_size=n_mels, n_out=n_mfcc)

        # Used for log operations
        self.eps = 1e-6
        self._feature_map = None

        # Cache initialization
        self.setup_cache(cache)

    def info(self, msg):
        if self.verbose:
            print(msg)

    def setup_cache(self, enable=True):
        self.cache = enable and settings.FEATURIZER.CACHE_ENABLED
        self.cache_dir = Path(settings.FEATURIZER.CACHE_DIR)
        self.cache_expire_seconds = 3600 * settings.FEATURIZER.CACHE_EXPIRE_HOURS
        self.cache_max_size = settings.FEATURIZER.CACHE_MAX_SIZE
        if self.cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._clean_cache()

    def clear_cache(self):
        n_audios = len(list(self.cache_dir.glob("*_map.json")))
        print(f"Clearing all cached features ({n_audios}) from: {self.cache_dir}")
        all_files = list(self.cache_dir.iterdir())
        for f in all_files:
            f.unlink()
        print(f"Deleted a total of {len(all_files)} files.")

    def _save_in_cache(self, audio, all_features, feature_map):
        timestamp_ms = int(time.time() * 1000)
        # Save audio signature
        np.save(
            self.cache_dir / f"{timestamp_ms}_signature.npy",
            self._audio_signature(audio),
        )
        # Save audio features
        np.save(
            self.cache_dir / f"{timestamp_ms}_features.npy",
            np.array(all_features),
        )
        # Save feature map
        json.dump(
            feat_to_str(feature_map),
            open(self.cache_dir / f"{timestamp_ms}_map.json", "w"),
        )
        # Clean cache in case it has become too big
        self._clean_cache()

    def _clean_cache(self):
        other_suffixes = ("_signature.npy", "_features.npy")
        # Iterate <time>_map.json files, and delete also their numpy counterparts
        for idx, f in enumerate(
            sorted(self.cache_dir.glob("*_map.json"))  # sorted: newest to oldest
        ):
            if idx < self.cache_max_size * (1 + len(other_suffixes)):
                try:
                    # File names: <timestamp>_map.json
                    f_time = f.stem.replace("_map.json", "")
                    f_time_s = int(f_time) / 1000  # ms -> seconds
                    f_life_time_s = time.time() - f_time_s
                    if not (0 <= f_life_time_s < self.cache_expire_seconds):
                        f.unlink()
                        for suffix in other_suffixes:
                            f_other = f.with_name(f"{f_time}{suffix}")
                            f_other.unlink(missing_ok=True)
                except ValueError:
                    pass

    def _find_in_cache(self, audio):
        # Compare audio signature to previously saved audios
        # Upon signature match, load precalculated features
        # Assumes previous _clean_cache()
        audio_sig = self._audio_signature(audio)
        for f in self.cache_dir.glob("*_signature.npy"):
            cached_sig = np.load(f)
            if self._check_signatures(audio_sig, cached_sig):
                # Return precalculated features and feature map
                fname = f.name.replace("_signature.npy", "")
                return (
                    np.load(self.cache_dir / f"{fname}_features.npy"),
                    str_to_feat(
                        json.load(open(self.cache_dir / f"{fname}_map.json", "r"))
                    ),
                )
        return None, None

    def _audio_signature(self, audio):
        # sum samples each second to create representative but smaller signature
        step = 16000
        audio = np.array(audio)
        pad_len = step - (len(audio) % step)
        reshaped = np.pad(audio, (0, pad_len)).reshape(-1, step)
        return reshaped.sum(axis=1)

    def _check_signatures(self, sig_1, sig_2):
        # Check two signatures created with _cache_signature
        if len(sig_1) != len(sig_2):
            return False
        abs_error = np.sum(np.abs(sig_1 - sig_2)) / len(sig_1)
        return abs_error < 1e-6

    def _concat_features(self, feature_list):
        feature_map = []
        for feat, name in feature_list:
            if len(feat.shape) == 2:
                feature_map += [name] * feat.shape[1]
            else:
                feature_map.append(name)
        features_concat = np.concatenate([feat for feat, _ in feature_list], axis=1)
        self._feature_map = np.array(feature_map)
        return features_concat, feature_map

    def select_feature(self, features_concat: Tensor, feature_name: FeatName) -> Tensor:
        """
        Filter from a tensor features_concat[samples, features]
        the features corresponding to the position of
        feature_name (one of AudioFeaturizer.FT_...)
        """
        assert self._feature_map is not None, "AudioFeaturizer was never used"
        return features_concat[:, self._feature_map == feature_name]

    def extract_spectrum(self, audio: Tensor) -> T.Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            audio = audio.squeeze().unsqueeze(0)  # predictability :)
            # STFT magnitude
            spec_magnitude = spectral_magnitude(self.compute_STFT(audio))
            # Mel
            mel = self.compute_fbanks(spec_magnitude)
            # DCT
            mfcc = self.compute_dct(mel).squeeze()
            self.info(f"Feature: {mfcc.shape=}")
        return spec_magnitude, mel, mfcc

    def extract_spectral_stats(
        self, spec_magnitude: torch.Tensor
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        spec_magnitude should be in the format returned by extract_mfcc above
        """
        # Tensor -> librosa format
        S_np = spec_magnitude.squeeze().numpy().T  # numpy

        # Spectral flatness
        spec_flat = np.log10(spectral_flatness(S=S_np) + self.eps).T
        self.info(f"Feature: {spec_flat.shape=}")

        # Spectral centroid
        spec_mean = spectral_centroid(S=S_np).T
        self.info(f"Feature: {spec_mean.shape=}")

        # Spectral bandwidth (logarithmic)
        spec_bw = spectral_bandwidth(S=S_np).T
        self.info(f"Feature: {spec_bw.shape=}")

        # Spectral contrast
        spec_contrast = spectral_contrast(S=S_np).T
        self.info(f"Feature: {spec_contrast.shape=}")

        # Signal RMS power (db)
        power_db = rms_to_db(rms(S=S_np, frame_length=self.win_length_samples)).T
        self.info(f"Feature: {power_db.shape=}")
        return spec_flat, spec_mean, spec_bw, spec_contrast, power_db

    def __call__(self, audio: Tensor) -> T.Tuple[Tensor, Tensor, T.List]:
        """
        Feature Extraction
        """
        all_features = None

        # Cached precalculated features
        if self.cache:
            all_features, feature_map = self._find_in_cache(audio)
            if all_features is not None and self.verbose:
                print("Features found and loaded from cache")

        # Calculate features and save cache
        if all_features is None:
            print("Extracting features...")
            S, mel, mfcc = self.extract_spectrum(audio)
            (
                spec_flat,
                spec_mean,
                spec_bw,
                spec_contrast,
                power_db,
            ) = self.extract_spectral_stats(S)
            all_features, feature_map = self._concat_features(
                (
                    (mfcc.numpy(), FeatName.MFCC),
                    (spec_flat, FeatName.SPEC_FLAT),
                    (spec_mean, FeatName.SPEC_MEAN),
                    (spec_bw, FeatName.SPEC_BW),
                    (spec_contrast, FeatName.SPEC_CONTRAST),
                    (power_db, FeatName.PWR_DB),
                )
            )
            self.info(f"All features concatenated: shape={all_features.shape}")

            if self.cache:  # Save cache
                self._save_in_cache(audio, all_features, feature_map)

        # Audio chunks corresponding to each feature vector
        waveforms = audio.squeeze().unfold(
            0, self.win_length_samples, self.hop_length_samples
        )
        assert waveforms.shape[0] == all_features.shape[0], "Oops"
        return Tensor(all_features), waveforms, feature_map
