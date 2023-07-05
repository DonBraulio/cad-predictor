import pandas as pd
import numpy as np

from typing import Union


def win_density(boolean_array, win_size=10):
    """
    Given a boolean array, return number of True values in each
    window of size win_size
    """
    # Accumulate true values, sample every win_size, get difference
    return np.diff(boolean_array.cumsum()[::win_size]) / win_size


def calculate_metrics(
    predictions: Union[pd.Series, np.ndarray],
    references: Union[pd.Series, np.ndarray],
    label: Union[str, int],
    win_density_len: int,
):
    metrics = {}
    pred_mask = predictions == label
    ref_mask = references == label
    total_correct = (pred_mask & ref_mask).sum()

    # Classification precision-recall per label
    precision = total_correct / pred_mask.sum()
    recall = total_correct / ref_mask.sum()
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = precision * recall

    # Time-density curve based metrics
    pred_density = win_density(pred_mask, win_size=win_density_len)
    ref_density = win_density(ref_mask, win_size=win_density_len)
    error_abs = abs(pred_density - ref_density)

    # Mean Absolute Error
    mae = error_abs.mean()
    metrics["mae"] = mae

    metrics["corr"] = np.corrcoef(ref_density, pred_density)[0][1]

    # DEPRECATED: All these are too complex or not stable to null values
    # Relative errors (scale-independent)
    # eps = 1e-6  # add to avoid null denominators
    # metrics["smape_orig"] = (
    #     100 * (error_abs / (abs(pred_density) + abs(ref_density) + eps)).mean()
    # )
    # # SMAPE but taking mean as denominator
    # metrics["smape"] = (
    #     100 * (error_abs / (abs(pred_density) + abs(ref_density) + eps).mean()).mean()
    # )
    # metrics["wmape"] = 100 * (error_abs / (abs(ref_density).mean() + eps)).mean()

    # # Taken from:
    # # https://www.sciencedirect.com/science/article/pii/S0169207016000121
    # # Adjusted to be in range 0-100 (instead of angles)
    # angle_to_percentage = 100 / np.pi / 2
    # metrics["maape"] = (
    #     np.arctan(error_abs / (abs(ref_density) + eps)).mean() * angle_to_percentage
    # )

    # # This is an invention mixing wmape and maape, which makes a lot of sense :)
    # metrics["wmaape"] = (
    #     np.arctan(error_abs / (abs(ref_density).mean() + eps)).mean()
    #     * angle_to_percentage
    # )
    return metrics
