import cv2
import numpy as np

from abc import abstractmethod
from utils.plots import color_cv2


class BaseViewerTrack:
    def __init__(self, cursor_color=(255, 50, 50), cursor_width=2):
        self.ax = None
        self.fig = None
        self.audio = None
        self.ticks = None
        self.cursor_color = color_cv2(cursor_color)
        self.cursor_width = cursor_width

    def setup(self, audio, ticks, ax, fig):
        self.audio = audio
        self.ticks = ticks
        self.ax = ax
        self.fig = fig
        (_, self.px_top), (_, self.px_bottom) = self.get_rect_px()

    def draw_cursor(self, time, frame):
        current_x, _ = self.get_coord_px(x_value=time)
        frame = cv2.line(
            frame,
            (current_x, self.px_top),
            (current_x, self.px_bottom),
            self.cursor_color,
            self.cursor_width,
        )
        return frame

    def get_coord_px(self, x_value=0.0, y_value=0.0):
        x_px, y_px = self.ax.transData.transform((x_value, y_value))
        _, fig_height = self.fig.canvas.get_width_height()
        return int(x_px), int(fig_height - y_px)

    def get_rect_px(self):
        """
        Return pixel positions of the rectangle for this track as:
         ((left, top), (right, bottom))
        """
        xlim_start, xlim_end = self.ax.get_xlim()
        ylim_bottom, ylim_top = self.ax.get_ylim()
        return (
            self.get_coord_px(xlim_start, ylim_top),
            self.get_coord_px(xlim_end, ylim_bottom),
        )

    @abstractmethod
    def on_segment_change(self, start_idx: int, end_idx: int) -> None:
        """
        Called only once when current segment window moves.
        Allows to re-draw the plot.
        Receives audio sample number idxs, to index
        self.audio or self.ticks
        """
        return NotImplemented

    @abstractmethod
    def on_segment_redraw(
        self, start_idx: int, end_idx: int, frame: np.ndarray
    ) -> np.ndarray:
        """
        Called only once when current segment window moves, after
        on_segment_change() is called and after the plot gets re-drawn.
        Allows to draw over the frame using opencv (e.g: segment shadow).
        """
        return NotImplemented

    @abstractmethod
    def on_time_change(self, time: float, frame: np.ndarray) -> np.ndarray:
        """
        Called on every iteration when time changes.
        Allows to draw over the frame using opencv (e.g: time cursor).
        """
        return NotImplemented


class BaseOutlineTrack(BaseViewerTrack):
    """
    Visualize all signal interval, darkening current focus
    """

    def __init__(
        self, cursor_color=(255, 50, 50), rect_color=(255, 255, 255), rect_alpha=0.2
    ):
        super().__init__(cursor_color, cursor_width=2)
        self.rect_color = color_cv2(rect_color)
        self.rect_alpha = rect_alpha

    def setup(self, audio, ticks, ax, fig):
        super().setup(audio, ticks, ax, fig)
        # Keep track of last updated segment
        self.t_start = 0.0
        self.t_end = 0.0
        self.init_plot(audio, ticks, ax, fig)

        # Make sure tracks are aligned even if they're not the same length
        ax.set_xlim(ticks.min(), ticks.max())

    @abstractmethod
    def init_plot(self, audio, ticks, ax, fig):
        """
        Create some kind of plot using self.ax
        """
        raise NotImplemented

    def on_segment_change(self, start_idx, end_idx):
        # No need to change anything in the plot here
        pass

    def on_segment_redraw(self, start_idx, end_idx, frame):
        # Draw segment shadow using opencv
        start_t, end_t = self.ticks[start_idx], self.ticks[end_idx]
        px_start, _ = self.get_coord_px(start_t)
        px_end, _ = self.get_coord_px(end_t)
        # filled rectangle
        frame_rect = cv2.rectangle(
            frame.copy(),
            (px_start, self.px_top),
            (px_end, self.px_bottom),
            self.rect_color,
            -1,
        )
        frame = cv2.addWeighted(
            frame, (1 - self.rect_alpha), frame_rect, self.rect_alpha, 0
        )
        return frame

    def on_time_change(self, time: float, frame: np.ndarray):
        frame = self.draw_cursor(time, frame)
        return frame


class BaseZoomedTrack(BaseViewerTrack):
    """
    Zoomed waveform, showing only one the current segment
    """

    def __init__(self, cursor_color=(255, 50, 50)):
        super().__init__(cursor_color=cursor_color, cursor_width=4)

    def setup(self, audio, ticks, ax, fig):
        super().setup(audio, ticks, ax, fig)
        self.init_plot(audio, ticks, ax, fig)

    @abstractmethod
    def init_plot(self, audio, ticks, ax, fig):
        """
        Create some kind of plot using self.ax
        """
        raise NotImplemented

    def on_segment_change(self, seg_start, seg_end):
        self.ax.set_xlim(self.ticks[seg_start], self.ticks[seg_end])

    def on_segment_redraw(
        self, start_idx: int, end_idx: int, frame: np.ndarray
    ) -> np.ndarray:
        return frame

    def on_time_change(self, time: float, frame: np.ndarray):
        frame = self.draw_cursor(time, frame)
        return frame
