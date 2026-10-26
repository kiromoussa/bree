"""annotated.mp4 writers: OpenCV mp4v (always works) or ffmpeg h264_nvenc (GPU encode, tiny CPU cost).

Frames reaching either writer are already head-pixelated (bree.alerts.annotate.draw_frame); nothing here sees an
unblurred frame.
"""
from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np

from bree.perf import ffmpeg_encoder_works


class FfmpegWriter:
    """Pipes raw BGR frames to `ffmpeg -c:v h264_nvenc`. Same write/release as cv2.VideoWriter."""

    def __init__(self, path: str, fps: float, size: tuple[int, int], codec: str = "h264_nvenc"):
        w, h = size
        self.size = (w, h)
        self.path = path
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.4f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", codec]
        cmd += ["-preset", "p3", "-rc", "vbr", "-cq", "27", "-b:v", "0"] if codec.endswith("nvenc") else []
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.dead = False

    def write(self, img: np.ndarray) -> None:
        if self.dead:
            return
        try:
            self.proc.stdin.write(np.ascontiguousarray(img).data)
        except (BrokenPipeError, OSError):
            self.dead = True
            print(f"warning: ffmpeg encoder stopped; {self.path} is incomplete", file=sys.stderr)

    def release(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()


def open_video_writer(path: str, fps: float, size: tuple[int, int], nvenc: bool = False):
    """NVENC writer when asked for and this machine encodes with it, else cv2 mp4v (the old behaviour)."""
    if nvenc and ffmpeg_encoder_works("h264_nvenc"):
        return FfmpegWriter(path, fps, size)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
