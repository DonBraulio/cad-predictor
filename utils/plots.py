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


def view_segment(start_s, end_s, audio, annotations):
    """
    Display all annotations or features of a segment,
    and the audio widget
    """
    if type(annotations) is not list:
        annotations = [annotations]
    sample_seg = Segment(start_s, end_s)
    fig, ax = plt.subplots(nrows=len(annotations), sharex=True, figsize=(10, 6))
    if type(ax) is Axes:
        ax = [ax]
    notebook.crop = sample_seg
    # Using testcomp_anns instead of test_anns since they're filtered
    for idx, ann in enumerate(annotations):
        notebook.plot_annotation(ann, ax[idx])
    display_segment(audio, sample_seg, fs)


def plot_xy(xy_points, point_labels, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    unique_labels = np.unique(point_labels)

    for label in unique_labels:
        label_mask = point_labels == label
        ax.plot(
            xy_points[0][label_mask],
            xy_points[1][label_mask],
            linestyle="",
            label=label,
            marker=".",
            alpha=0.4,
        )
    ax.set_title(title)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.legend(loc="upper right")


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


def plot_confusion(
    r_labels, c_labels, matrix, r_name="", c_name="", title="", save_file=""
):
    fig, ax = plt.subplots(figsize=(9, 9))
    f = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
    )
    c_range = range(len(c_labels))
    r_range = range(len(r_labels))
    for y in r_range:
        for x in c_range:
            ax.text(x, y, f"{matrix[y, x]:.4f}", va="center", ha="center")
    ax.set_xticks(ticks=c_range, labels=c_labels, rotation=90)
    ax.set_yticks(ticks=r_range, labels=r_labels)
    ax.set_title(title)
    ax.set_xlabel(c_name)
    ax.set_ylabel(r_name)
    fig.colorbar(f)
    if save_file:
        fig.savefig(save_file)


def show_confusion_der(
    ref_annotations, pred_annotations, use_labels=None, title="", save_file=None
):
    if use_labels is not None:
        ref_annotations = ref_annotations.subset(use_labels)
        pred_annotations = pred_annotations.subset(use_labels)
    metric = DiarizationErrorRate(skip_overlap=False)
    der = metric(ref_annotations, pred_annotations, detailed=True)
    print(f"DER: {der['diarization error rate']}")
    print(der)
    miss = der["missed detection"] / der["total"]
    fa = der["false alarm"] / der["total"]
    conf = der["confusion"] / der["total"]
    correct = der["correct"] / der["total"]
    der = der["diarization error rate"]

    confusion = IoMin(pred_annotations, ref_annotations)
    # Calculate Intersection over min duration
    title += f"\n{der=:.2f} | {correct=:.2f} | {conf=:.2f} | {miss=:.2f} | {fa=:.2f}"
    plot_confusion(
        pred_annotations.labels(),
        ref_annotations.labels(),
        confusion,
        "Predicted label",
        "True label",
        title,
        save_file=save_file,
    )


def draw_matches(neigh_labels, a_labels):
    # Alternative view by labeled samples, no time or segments
    matches = (neigh_labels == a_labels.reshape(-1, 1)).T
    plt.figure(figsize=(0.3 * matches.shape[1], 0.3 + 0.3 * matches.shape[0]))
    colors = np.array([[255, 91, 66], [148, 255, 66]])  # Red, green
    matches_colored = colors[matches.flatten().astype(int)].reshape(
        matches.shape + (3,)
    )
    plt.imshow(
        matches_colored,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn",
    )
    _ = plt.xticks(ticks=range(len(a_labels)), labels=a_labels, rotation=90)
    plt.yticks(range(0, matches.shape[0]))


def plot_density_interactive(
    preds_array,
    labels_array,
    hop_s,
    win_size,
    labels_l=None,
    environ_array=None,
    visible=None,
    title=None,
    savefig=None,
):
    assert len(preds_array) == len(
        labels_array
    ), f"{len(preds_array)=}, {len(labels_array)=}"
    if visible is None:
        visible = []
    if environ_array is None:
        environ_array = []
    else:
        assert len(environ_array) == len(
            labels_array
        ), f"{len(preds_array)=}, {len(labels_array)=}"
    if labels_l is None:
        labels_l = set(np.unique(labels_array)) | set(np.unique(preds_array))
    fig = go.Figure()
    t_density = None
    for env_label in set(np.unique(environ_array)) - set([NULL_LABEL]):
        trace_density = win_density(environ_array == env_label, win_size=win_size)
        n_wins = len(trace_density)
        t_density = np.linspace(start=0, stop=n_wins * hop_s * win_size, num=n_wins)
        fig.add_trace(
            go.Scatter(
                x=t_density,
                y=trace_density,
                mode="lines",
                name=f"Environment ({env_label})",
                line=dict(color="green", width=1),
                visible="legendonly",
            )
        )
    for idx, label in enumerate(labels_l):
        preds_density = win_density(preds_array == label, win_size=win_size)
        ref_density = win_density(labels_array == label, win_size=win_size)
        if t_density is None:
            n_wins = len(preds_density)
            t_density = np.linspace(start=0, stop=n_wins * hop_s * win_size, num=n_wins)
        # Assuming len(preds_array) == len(labels_array)
        color = f"rgb{tuple(light_colors.label_colors[label])}"
        fig.add_trace(
            go.Scatter(
                x=t_density,
                y=preds_density,
                mode="lines",
                name=f"{label} (pred)",
                line=dict(color=color, width=1),
                visible=True if label in visible else "legendonly",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_density,
                y=ref_density,
                mode="lines",
                name=f"{label} (ref)",
                line=dict(color=color, width=0.5, dash="dot"),
                visible=True if label in visible else "legendonly",
            )
        )
    if savefig and ".png" not in str(savefig):
        # Not included in png-only images
        fig.update_xaxes(
            rangeslider_visible=True,
        )
    fig.update_layout(
        title=title,
        autosize=False,
        width=800,
        height=400,
        margin=dict(l=10, r=10, b=20, t=20, pad=2),
        paper_bgcolor="white",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    )
    if savefig:
        savefig = str(savefig)
        if ".html" in savefig:
            fig.write_html(savefig)
        elif ".png" in savefig:
            fig.write_image(savefig)
        else:  # No extension provided: save both
            fig.write_image(f"{savefig}.png")
            fig.write_html(f"{savefig}.html")
    return fig


def plot_density_multiple(
    arrays_list: T.List[np.ndarray],  # List of arrays with predictions like 'p', 'm'
    styles_list: T.List[str],  # Same size as above, style for each array ('solid')
    legends_list: T.List[str],  # Same size as above, legend for each array
    show_labels: T.List[str],  # List of labels to show ('p',  'm')
    hop_s: float,
    win_size,
    title=None,
    savefig=None,
    slider=False,
):
    fig = go.Figure()
    t_density = None
    for label in show_labels:
        color = f"rgb{tuple(light_colors.label_colors[label])}"
        for idx, labels_array in enumerate(arrays_list):
            label_density = win_density(labels_array == label, win_size=win_size)
            if t_density is None:
                n_wins = len(label_density)
                t_density = np.linspace(
                    start=0, stop=n_wins * hop_s * win_size, num=n_wins
                )
            fig.add_trace(
                go.Scatter(
                    x=t_density,
                    y=label_density,
                    mode="lines",
                    name=f"{legends_list[idx] or ''} {label}",
                    # Dash values: solid, dot, dash, longdash, dashdot, longdashdot
                    line=dict(color=color, width=1, dash=styles_list[idx]),
                    visible=True,
                    showlegend=legends_list[idx] is not None,
                )
            )
    fig.update_xaxes(
        rangeslider_visible=slider,
    )
    fig.update_layout(
        title=title,
        autosize=False,
        width=800,
        height=400,
        margin=dict(l=10, r=10, b=20, t=20, pad=2),
        paper_bgcolor="white",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h"),
    )
    if savefig:
        savefig = str(savefig)
        if ".html" in savefig:
            fig.write_html(savefig)
        elif ".png" in savefig or ".svg" in savefig or ".pdf" in savefig:
            fig.write_image(savefig)
        else:  # No extension provided: save png and html
            fig.write_image(f"{savefig}.png")
            fig.write_html(f"{savefig}.html")
    return fig
