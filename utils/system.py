import subprocess

from typing import Union
from rich import print
from pathlib import Path


def cmd_mp4_to_wav(
    input_file: Union[Path, str],
    output_file: Union[Path, str],
    start: Union[str, int] = "00:00:00",
    end: Union[str, int, None] = None,
    sample_rate: int = 16000,
):
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_file), "-c:a", "pcm_s16le"]
    if sample_rate:
        ffmpeg_cmd += ["-ar", str(sample_rate)]
    if start:
        ffmpeg_cmd += ["-ss", str(start)]
    if end:
        ffmpeg_cmd += ["-to", str(end)]
    ffmpeg_cmd += [str(output_file)]
    return ffmpeg_cmd


def cmd_encode_audio_video(
    video_file: Union[Path, str],
    audio_file: Union[Path, str],
    output_file: Union[Path, str],
    video_encoding: str = "h264",
    audio_encoding: str = "aac",
    audio_offset: float = 0.0,
):
    """
    Use encoding=copy to avoid quality loss (bigger files)
    """
    ffmpeg_cmd = ["ffmpeg", "-y"]
    ffmpeg_cmd += ["-ss", "0.0"]  # No offset for video
    ffmpeg_cmd += ["-i", str(video_file)]
    ffmpeg_cmd += ["-ss", f"{audio_offset:.1f}"]
    ffmpeg_cmd += ["-i", str(audio_file)]
    ffmpeg_cmd += ["-c:v", video_encoding]
    ffmpeg_cmd += ["-c:a", audio_encoding]
    ffmpeg_cmd += [str(output_file)]
    return ffmpeg_cmd


def cmd_combine_videos(
    top_video_file: Union[Path, str],
    bottom_video_file: Union[Path, str],
    output_file: Union[Path, str],
    out_width: int = 800,
    video_encoding: str = "h264",
    audio_encoding: str = "aac",
    fps: int = 24,
):
    ffmpeg_cmd = ["ffmpeg", "-y"]
    ffmpeg_cmd += ["-i", str(top_video_file)]
    ffmpeg_cmd += ["-i", str(bottom_video_file)]
    # Rescale to the same width so they can be v-stacked
    # Automatic height to keep aspect ratio and must be multiple of 2 (codec requires)
    ffmpeg_cmd += [
        "-filter_complex",
        f"[0:v]scale={out_width}:-1,setsar=1,"
        f"scale={out_width}:trunc(ow/a/2)*2,fps={fps}[top];"
        f"[1:v]scale={out_width}:-1,setsar=1,"
        f"scale={out_width}:trunc(ow/a/2)*2,fps={fps}[bottom];"
        f"[top][bottom]vstack=inputs=2",
    ]
    ffmpeg_cmd += ["-c:v", video_encoding]
    ffmpeg_cmd += ["-c:a", audio_encoding]
    ffmpeg_cmd += [str(output_file)]
    return ffmpeg_cmd


def run_command(cmd: str, exec_timeout: int = 120):
    with subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,  # ffmpeg only uses stderr
        encoding="utf-8",
    ) as proc:
        if proc is None:
            raise ValueError(f"Could not exec command: {cmd}")
        try:
            ret_code = proc.wait(timeout=exec_timeout)  # raises timeout unless finished
            error_msg = proc.stderr.read()
        except subprocess.TimeoutExpired:
            print("[red]Timeout, killing ffmpeg process...[/red]")
            proc.kill()
            (out, err) = proc.communicate()
            err += "\n\n TIMEOUT: PROCESS KILLED"
            raise TimeoutError(f"Execution time exceeded {exec_timeout}s")
    return ret_code, error_msg
