import cv2
import builtins
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from typing import List, Union
from rich import print, progress
from collections.abc import Iterable

from .base_tracks import BaseViewerTrack
from utils.system import cmd_combine_videos, cmd_encode_audio_video, run_command

# Make sure that DVC pipeline runs the same as notebook
print("[red]AudioViewer: Setting matplotlib backend_inline[/red]")
matplotlib.use("module://matplotlib_inline.backend_inline")


class AudioViewer:
    def __init__(
        self,
        audio: np.ndarray,
        fs: int,
        segment_length_s: float = 10,
        fig_width: int = 16,
        fig_height: int = 0,
        mpl_style: str = "classic",
    ):
        self.audio = audio
        self.fs = fs
        self.ts = 1.0 / fs
        self.duration = self.ts * len(audio)
        self.segment_idx = -1  # trigger draw at first iteration
        self.segment_length_n = int(segment_length_s * self.fs)
        self.duration_n = int(self.duration * self.fs)

        # use 4:3 video aspect ratio
        if not fig_height:
            fig_height = int((3 / 4) * fig_width)
        self.figsize = (fig_width, fig_height)

        self.fig = None
        self.axes = None
        self._tracks: List[BaseViewerTrack] = []
        self._height_ratios: List[float] = []

        self.t_start = 0
        self.ticks = np.arange(self.t_start, self.duration, self.ts)
        self.time = 0
        self.mpl_style = mpl_style

        # Keep plot as a frame to draw using opencv, updated on segment changes
        self.segment_frame = None

    def add_track(self, track: BaseViewerTrack, height_ratio: float = 1.0):
        self._tracks.append(track)
        self._height_ratios.append(height_ratio)

    def setup(self):
        # Create plot and axes
        with plt.style.context([self.mpl_style]):
            self.fig, self.axes = plt.subplots(
                nrows=len(self._tracks),
                figsize=self.figsize,
                gridspec_kw={"height_ratios": self._height_ratios},
            )
            self.fig.tight_layout(pad=1.5)
            if not isinstance(self.axes, Iterable):
                self.axes = [self.axes]  # when nrows=1, subplots doesn't return list

            # Initialize tracks
            for idx, ax in enumerate(self.axes):
                self._tracks[idx].setup(self.audio, self.ticks, ax, self.fig)

    def update(self, time: float) -> np.ndarray:
        """
        Update frame image using 2 criteria:
         - When the time segment moves, re-draw the whole plot using matplotlib
         - Otherwise, only draw over frame using opencv
        """
        if self.fig is None:
            self.setup()

        self.time = time
        time_n = int(time * self.fs)
        new_segment_idx = int(time_n // self.segment_length_n)

        # Redraw the whole plot only when time segment moves
        if self.segment_frame is None or self.segment_idx != new_segment_idx:
            self.segment_idx = new_segment_idx
            # These are audio sample numbers
            seg_start_n = self.segment_idx * self.segment_length_n
            seg_end_n = min(seg_start_n + self.segment_length_n, self.duration_n) - 1

            # Update plot: allowed only once per segment
            for track in self._tracks:
                track.on_segment_change(seg_start_n, seg_end_n)

            # Redraw plot and extract opencv frame
            frame = self._redraw_plot()

            # Tracks may also draw over frame using opencv once per segment
            for track in self._tracks:
                frame = track.on_segment_redraw(seg_start_n, seg_end_n, frame)

            # Keep this frame to draw over when segment doesn't change
            self.segment_frame = frame

        # Tracks can draw over segment frame everytime
        frame = self.segment_frame.copy()
        for track in self._tracks:
            frame = track.on_time_change(time, frame)
        return frame

    def _redraw_plot(self):
        self.fig.canvas.draw()
        return cv2.cvtColor(
            np.asarray(self.fig.canvas.renderer.buffer_rgba()),
            cv2.COLOR_RGBA2BGR,
        )

    def create_video(self, file_name: Union[Path, str], fps=5):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Ensure frame is initialized
        if self.segment_frame is None:
            self.update(0.0)
        frame_shape = self.segment_frame.shape[1::-1]

        print(f"Creating video with shape: {frame_shape}")
        video = cv2.VideoWriter(str(file_name), fourcc, fps, frameSize=frame_shape)

        # Iterate over time and create frames
        t_frames = np.arange(0, self.duration, 1 / fps)
        times = [time()]
        for t in progress.track(t_frames, description="Generating video..."):
            frame = self.update(time=t)
            video.write(frame)
            times.append(time())
        video.release()

        # Benchmarking
        times = np.diff(times)
        print(
            f"[green]------- Video generated @ {fps}fps -------[/green]\n"
            f"Avg speed: {1/times.mean():.2f}fps"
            f" | total: {times.sum():.1f}s"
            f" | video: {self.duration}s ({self.duration / times.sum():.1f}X)"
        )

    @staticmethod
    def encode_video(
        video_file: Union[Path, str],
        audio_file: Union[Path, str],
        encoded_file: Union[Path, str],
        audio_offset: float = 0.0,
    ):
        """
        Use encoding=copy to avoid quality loss (bigger files)
        """
        cmd_encode = cmd_encode_audio_video(
            video_file, audio_file, encoded_file, audio_offset=audio_offset
        )
        # Don't use rich.print because there might be []
        builtins.print(" ".join(cmd_encode))
        exit_code, error_msg = run_command(cmd_encode)
        if exit_code:
            print(error_msg)
            raise ValueError("An error ocurred during video encoding")

    @staticmethod
    def combine_videos(
        top_video: Union[Path, str],
        bottom_video: Union[Path, str],
        output_file: Union[Path, str],
        out_width: int = 800,
    ):
        cmd_encode = cmd_combine_videos(
            top_video, bottom_video, output_file, out_width=out_width
        )
        # Don't use rich.print because there might be []
        builtins.print(" ".join(cmd_encode))
        exit_code, error_msg = run_command(cmd_encode, exec_timeout=600)
        if exit_code:
            print(error_msg)
            raise ValueError("An error ocurred during video encoding")
