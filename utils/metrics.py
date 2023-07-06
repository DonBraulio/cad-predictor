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


def get_confusion_probs(df_preds, labels):
    confusion_matrix_recall = np.empty((len(labels), len(labels)))
    confusion_matrix_precision = np.empty((len(labels), len(labels)))
    for ref_idx, ref_label in enumerate(labels):
        for pred_idx, pred_label in enumerate(labels):
            # Predicted confusion matrix
            ref_mask = df_preds["reference"] == ref_label
            pred_mask = df_preds["prediction"] == pred_label

            # Count how many items we've in this cell (ref=ref_label, pred=pred_label)
            hits_cell = ref_mask & pred_mask

            # Bayes calculation for recall:
            # P(pred=label | ref=label)
            #   = P(pred=label & ref=label) / P(ref=label)
            recall = hits_cell.sum() / ref_mask.sum()

            # Bayes calculation for precision:
            # P(ref=label | pred=label)
            #   = P(pred=label & ref=label) / P(pred=label)
            precision = hits_cell.sum() / pred_mask.sum()

            confusion_matrix_recall[ref_idx][pred_idx] = recall
            confusion_matrix_precision[ref_idx][pred_idx] = precision
    return confusion_matrix_recall, confusion_matrix_precision


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
