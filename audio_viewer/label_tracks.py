# %%
import matplotlib.pyplot as plt

from typing import List, Mapping, Optional
from pyannote.core import Annotation

from utils.plots import color_pyplot

from .base_tracks import BaseOutlineTrack, BaseZoomedTrack


class MixinLabelsTrack:
    """
    Helper functions for label tracks (could be zoomed or shadowed)
    """

    def __init__(
        self,
        annotations: Annotation,
        label_colors: Mapping = None,
        labels_order: List = None,
        ref_annotations: Optional[Annotation] = None,
    ):
        self.annotations = annotations
        self.ref_annotations = ref_annotations

        # Colors: {label1: (R,G,B), label2: (R,G,B), default: (R, G, B)}
        self._colors = label_colors if label_colors else {}
        if "default" not in self._colors:
            self._colors["default"] = (100, 100, 100)

        # Label order (bottom to top)
        # Convert order=[a, b, c], existing=[b, d, a, f] --->  [a, b, d, f]
        self._order = labels_order if labels_order else []
        existing_labels = set(annotations.labels())
        if ref_annotations is not None:
            existing_labels |= set(ref_annotations.labels())
        self._order = [l for l in self._order if l in existing_labels]
        self._order += [l for l in existing_labels if l not in self._order]

        # Configureable params
        self._show_legend = True
        self._show_time = False
        self._name = ""

    def show_legend(self, show=True):
        self._show_legend = show

    def show_time(self, show=True):
        self._show_time = show

    def set_name(self, name=""):
        self._name = name

    def draw_labels(self, ax, y=1, linewidth=5.0, margin=0.05):
        label_handles = []
        label_names = []
        y_min = y - margin
        for label in self._order:
            timeline = self.annotations.label_timeline(label)
            color = color_pyplot(self._colors.get(label, self._colors["default"]))
            if self.ref_annotations is not None:
                for seg in self.ref_annotations.label_timeline(label):
                    l = ax.hlines(
                        y,
                        xmin=seg.start,
                        xmax=seg.end,
                        color=color,
                        alpha=0.8,
                        linewidth=linewidth * 0.5,
                        linestyles="dotted",
                    )
            for seg in timeline:
                l = ax.hlines(
                    y,
                    xmin=seg.start,
                    xmax=seg.end,
                    color=color,
                    alpha=0.8,
                    linewidth=linewidth,
                )
            y += margin
            label_handles.append(l)
            label_names.append(label)
        if self._show_legend:
            y += 6 * margin  # Make room for legend
            ax.legend(
                label_handles,
                label_names,
                loc="upper left",
                ncol=len(label_names),
            )
        ax.set_ylim(y_min, y)
        if not self._show_time:
            ax.get_xaxis().set_visible(False)
        if self._name:
            ax.set_ylabel(self._name)
        # These two combined will show track name, but not tick values
        ax.get_yaxis().set_visible(True)
        plt.setp(ax.get_yticklabels(), visible=False)


class LabelsZoomedTrack(BaseZoomedTrack, MixinLabelsTrack):
    """
    Zoomed Labels, showing only one the current segment
    """

    def __init__(
        self,
        annotations: Annotation,
        label_colors: Mapping = None,
        labels_order: List = None,
        ref_annotations: Optional[Annotation] = None,
    ):
        BaseZoomedTrack.__init__(self)
        MixinLabelsTrack.__init__(
            self, annotations, label_colors, labels_order, ref_annotations
        )

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_labels(ax)


class LabelsOutlineTrack(BaseOutlineTrack, MixinLabelsTrack):
    """
    Zoomed Labels, showing only one the current segment
    """

    def __init__(
        self,
        annotations: Annotation,
        label_colors: Mapping = None,
        labels_order: List = None,
        ref_annotations: Optional[Annotation] = None,
    ):
        BaseOutlineTrack.__init__(self)
        MixinLabelsTrack.__init__(
            self, annotations, label_colors, labels_order, ref_annotations
        )

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_labels(ax)
