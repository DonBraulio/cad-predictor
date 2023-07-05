import torch
import numpy as np
from typing import Union
from pyannote.core import Segment
from utils.features import s_power_db


def crop_audio(
    audio: Union[torch.Tensor, np.ndarray], segment: Segment, fs: int
) -> Union[torch.Tensor, np.ndarray]:
    start_idx = int(segment.start * fs)
    end_idx = int(segment.end * fs)
    return (
        audio[start_idx:end_idx]
        if len(audio.shape) == 1
        else audio[:, start_idx:end_idx]
    )


def find_sample_above_threshold(audio, power_th_db=80, win_len_s=2, mode="min"):
    # find minimum power example above threshold
    power_db = s_power_db(audio, win_len_s=win_len_s)
    power_db.data[power_db < power_th_db] = np.nan
    if mode == "min":
        win_n = np.nanargmin(power_db.data)
    else:
        win_n = np.nanargmax(power_db.data)
    return power_db.sliding_window[win_n]
