import numpy as np
import typing as T
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from matplotlib.axes import Axes
from plotly import graph_objects as go
from pyannote.core import Segment, notebook

from pyannote.metrics.diarization import DiarizationErrorRate
from config import settings
from utils.metrics import IoMin, win_density
from utils.vscode_audio import display_segment
from utils.annotations import NULL_LABEL

fs = settings.AUDIO.SAMPLE_RATE
light_colors = settings.STYLES.LIGHT


def plot_density_correlation(ref_density, pred_density, label):
    plt.figure(figsize=(4, 4))
    plt.scatter(ref_density, pred_density, marker=".", alpha=0.7)
    plt.xlabel("Reference density")
    plt.ylabel("Predicted density")
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Show correlation coefficient in title
    corr_model = np.corrcoef(ref_density, pred_density)[0][1]
    plt.title(f"Correlation $\\rho_{{RP}}={corr_model:.2f}$ ({label=})")


def plot_density_time(ref_density, pred_density, label, win_density_s):
    sns.set_palette("muted")
    t = np.arange(len(pred_density)) * win_density_s
    plt.figure(figsize=(8, 4))
    plt.plot(t, ref_density, label="Reference", alpha=1)
    plt.plot(t, pred_density, linestyle=":", label="Prediction")
    plt.xlabel("t (s)")
    plt.ylabel("Density estimation")
    plt.title(f"Density estimation for label {label}")
    plt.legend(loc="upper right")


def color_cv2(rgb_255):
    """
    Convert RGB iterable->BGR tuple for opencv objects
    """
    return tuple(rgb_255[::-1])


def color_pyplot(rgb_255):
    """
    Convert RGB 0-255 -> RGB 0-1 for matplotlib
    """
    return np.array(rgb_255) / 255
