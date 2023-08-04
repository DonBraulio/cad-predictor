import numpy as np
import matplotlib.pyplot as plt
from typing import List, Mapping

from .base_tracks import BaseOutlineTrack, BaseZoomedTrack


class MixinScalarTrack:
    """
    Helper functions for plotting scalar value tracks (could be zoomed or outline)
    """

    def __init__(
        self,
        t_list: List[np.ndarray],
        y_list: List[np.ndarray],
        styles_list: Mapping[str, str],
    ):
        # Configureable params
        self._show_legend = True
        self._show_time = False
        self._name = None
        self._t_list = t_list
        self._y_list = y_list
        self._styles_list = styles_list

    def show_legend(self, show=True):
        self._show_legend = show

    def show_time(self, show=True):
        self._show_time = show

    def set_name(self, name=""):
        self._name = name

    def draw_labels(self, ax):
        for idx, t_axis in enumerate(self._t_list):
            ax.plot(t_axis, self._y_list[idx], **self._styles_list[idx])
        if self._show_legend:
            ax.legend(
                loc="upper left",
                ncol=len(self._t_list),
            )
        if not self._show_time:
            ax.get_xaxis().set_visible(False)
        if self._name:
            ax.set_ylabel(self._name)
        # These two combined will show track name, but not tick values
        ax.get_yaxis().set_visible(True)
        plt.setp(ax.get_yticklabels(), visible=False)


class ScalarZoomedTrack(BaseZoomedTrack, MixinScalarTrack):
    """
    Zoomed scalar value, showing only one the current segment
    """

    def __init__(
        self,
        t_list: List[np.ndarray],
        y_list: List[np.ndarray],
        styles_list: Mapping[str, str],
    ):
        BaseZoomedTrack.__init__(self)
        MixinScalarTrack.__init__(self, t_list, y_list, styles_list)

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_labels(ax)


class ScalarOutlineTrack(BaseOutlineTrack, MixinScalarTrack):
    """
    Outline of scalar value, showing overview of the entire audio
    """

    def __init__(
        self,
        t_list: List[np.ndarray],
        y_list: List[np.ndarray],
        styles_list: Mapping[str, str],
    ):
        BaseOutlineTrack.__init__(self)
        MixinScalarTrack.__init__(self, t_list, y_list, styles_list)

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_labels(ax)
