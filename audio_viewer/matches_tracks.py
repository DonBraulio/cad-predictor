# %%
from audio_viewer.base_tracks import BaseOutlineTrack, BaseZoomedTrack
from utils.plots import color_pyplot


class MixinMatchesTrack:
    """
    Helper functions for matching tracks (could be zoomed or shadowed)
    """

    def __init__(self, labels, segments, neigh_labels, neigh_distances):
        assert len(labels) == neigh_labels.shape[0]
        assert neigh_labels.shape == neigh_distances.shape

        self.labels = labels  # length n
        self.segments = segments  # length n
        self.neigh_labels = neigh_labels  # nxk
        self.neigh_distances = neigh_distances  # nxk
        self.k = neigh_labels.shape[1]  # number of neighbors

        self.color_error = color_pyplot([255, 91, 66])  # Red
        self.color_match = color_pyplot([148, 255, 66])  # Green
        # Configureable params
        self._show_time = False
        self._name = f"Distances (k={self.k})"

    def show_time(self, show=True):
        self._show_time = show

    def set_name(self, name=""):
        self._name = name

    def draw_matches(self, ax, linewidth=2.0):
        for sample_idx, label in enumerate(self.labels):
            seg = self.segments[sample_idx]
            for n, neighbor_dist in enumerate(self.neigh_distances[sample_idx, :]):
                is_match = self.neigh_labels[sample_idx, n] == label
                ax.hlines(
                    neighbor_dist,
                    xmin=seg.start,
                    xmax=seg.end,
                    color=self.color_match if is_match else self.color_error,
                    alpha=0.3,
                    linewidth=linewidth,
                )

        ax.set_ylabel(self._name)
        if not self._show_time:
            ax.get_xaxis().set_visible(False)


class MatchesZoomedTrack(BaseZoomedTrack, MixinMatchesTrack):
    """
    Zoomed Matches, showing only one the current segment
    """

    def __init__(self, labels, segments, neigh_labels, neigh_distances):
        BaseZoomedTrack.__init__(self)
        MixinMatchesTrack.__init__(
            self, labels, segments, neigh_labels, neigh_distances
        )

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_matches(ax)


class MatchesOutlineTrack(BaseOutlineTrack, MixinMatchesTrack):
    """
    Outline of Matches, showing the whole audio track duration
    """

    def __init__(self, labels, segments, neigh_labels, neigh_distances):
        BaseOutlineTrack.__init__(self)
        MixinMatchesTrack.__init__(
            self, labels, segments, neigh_labels, neigh_distances
        )

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_matches(ax)
