from utils.plots import color_pyplot
from .base_tracks import BaseOutlineTrack, BaseZoomedTrack


class MixinWaveformTrack:
    def __init__(self, signal_color=(0, 255, 0)):
        self.signal_color = color_pyplot(signal_color)

    def draw_waveform(self, audio, ticks, ax):
        _ = ax.plot(ticks, audio, color=self.signal_color)
        ax.get_yaxis().set_visible(False)
        ax.grid(False)


class WaveformOutlineTrack(BaseOutlineTrack, MixinWaveformTrack):
    """
    Visualize all audio waveform, darkening current focus
    """

    def __init__(self, signal_color=(0, 255, 0)):
        MixinWaveformTrack.__init__(self, signal_color)
        BaseOutlineTrack.__init__(self)

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_waveform(audio, ticks, ax)


class WaveformZoomedTrack(BaseZoomedTrack, MixinWaveformTrack):
    """
    Zoomed audio waveform, showing only one the current segment
    """

    def __init__(self, signal_color=(0, 255, 0)):
        MixinWaveformTrack.__init__(self, signal_color)
        BaseZoomedTrack.__init__(self)

    def init_plot(self, audio, ticks, ax, fig):
        self.draw_waveform(audio, ticks, ax)
