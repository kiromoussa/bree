"""Video input: files, webcams, and RTSP/HTTP streams, as one iterator of frames."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    t: float              # seconds since stream start (file: frame index / fps; live: wall clock)
    image: np.ndarray     # BGR
    wall: float           # time.time() when the frame was read (latency measurement)
    camera: str | None = None   # set by the pipeline when several cameras feed one store


class VideoSource:
    """`src` is a file path, a webcam index ("0"), or an rtsp:// / http(s):// URL."""

    def __init__(self, src: str | int, max_frames: int | None = None, reconnect: bool = True):
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        is_url = isinstance(self.src, str) and self.src.startswith(("rtsp://", "http://", "https://"))
        self.live = isinstance(self.src, int) or is_url
        self.max_frames = max_frames
        self.reconnect = reconnect and self.live
        self.cap = self._open()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 and fps < 240 else 15.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise IOError(f"cannot open video source {self.src!r}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        i, t0, failures = 0, time.time(), 0
        while self.max_frames is None or i < self.max_frames:
            ok, img = self.cap.read()
            if not ok:
                if not self.reconnect or failures >= 5:
                    break
                failures += 1          # RTSP hiccup: back off and reopen
                time.sleep(min(2 ** failures, 10))
                self.cap.release()
                self.cap = self._open()
                continue
            failures = 0
            now = time.time()
            t = (now - t0) if self.live else i / self.fps
            yield Frame(i, t, img, now)
            i += 1
        self.cap.release()
