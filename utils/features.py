import torch
import numpy as np

from time import time
from librosa.feature import rms
from torch import functional as F
from pyannote.core import (
    SlidingWindow,
    SlidingWindowFeature,
)
from einops import rearrange
from scipy.signal import medfilt
from sklearn import manifold


POWER_REF_LEVEL = 20e-6  # audition threshold
POWER_REF_DB = 20 * np.log10(POWER_REF_LEVEL)


def s_power_db(audio, win_len_s=0.5, hop_len_s=0, fs=16000):
    hop_len_s = hop_len_s or win_len_s
    win_len = int(win_len_s * fs)
    hop_len = int(hop_len_s * fs)
    return SlidingWindowFeature(
        data=rms_to_db(
            rms(audio, frame_length=win_len, hop_length=hop_len, center=False),
        ).reshape(-1, 1),
        sliding_window=SlidingWindow(duration=win_len_s, step=hop_len_s),
    )


def rms_to_db(rms_values):
    return 20 * np.log10(
        np.maximum(
            POWER_REF_LEVEL,
            rms_values,
        )
        / POWER_REF_LEVEL  # divide over ref_level to get positive numbers
    )


def moving_max_pooling(X: np.ndarray, n=5):
    return rearrange(
        F.max_pool1d(
            rearrange(torch.Tensor(X), "n c -> 1 c n"),
            n,
            stride=1,
            padding=n // 2,
        ),
        "1 c n -> n c",
    ).numpy()


def moving_avg_filter(X: np.ndarray, n_avg=5):
    # Moving average (separate channel for each coefficient)
    return rearrange(
        F.avg_pool1d(
            rearrange(torch.Tensor(X), "n c -> 1 c n"),
            n_avg,
            stride=1,
            padding=n_avg // 2,
            count_include_pad=False,
        ),
        "1 c n -> n c",
    ).numpy()


def moving_median_filter(X: np.ndarray, n=5):
    if len(X.shape) == 2:
        X_m = np.zeros_like(X)
        for c in range(X.shape[1]):
            X_m[:, c] = moving_median_filter(X[:, c])
        return X_m
    elif len(X.shape) != 1:
        raise ValueError("Shape should be (n,) or (n,c)")
    return medfilt(X, kernel_size=n)


def reduce_dimensionality(embeddings_ss, method="tsne", metric="cosine"):
    """
    Reduce embeddings dimensionality to 2D
    """
    t0 = time()
    if method == "tsne":
        trans_data = (
            manifold.TSNE(n_components=2, perplexity=50, metric=metric, random_state=0)
            .fit_transform(embeddings_ss)
            .T
        )
    elif method == "mds":
        trans_data = (
            manifold.MDS(2, metric=metric, max_iter=100, n_init=1)
            .fit_transform(embeddings_ss)
            .T
        )
    # Doesn't support cosine distance?
    # elif method == "lle":
    #     trans_data = (
    #         manifold.LocallyLinearEmbedding(
    #             n_neighbors=30, n_components=2, method="standard"  # ltsa
    #         )
    #         .fit_transform(embeddings_ss)
    #         .T
    #     )
    elif method == "isomap":
        trans_data = (
            manifold.Isomap(n_neighbors=30, n_components=2, metric=metric)
            .fit_transform(embeddings_ss)
            .T
        )
    # Doesn't support cosine distance?
    # elif method == "spectral":
    #     trans_data = (
    #         manifold.SpectralEmbedding(n_neighbors=30, n_components=2)
    #         .fit_transform(embeddings_ss)
    #         .T
    #     )
    print(f"Dim reduction ({method}/{metric}) time: {(time() - t0):.2g} sec")
    return trans_data
