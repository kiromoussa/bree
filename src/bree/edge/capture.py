"""Two-speed capture for a camera node.

Every tick the source hands back one low-res frame (for the trigger) and one full-res frame (kept
in a short ring buffer in RAM, not encoded). When the trigger fires, the ring becomes the pre-roll of
a burst, frames keep being added while a zone is active and for `post_roll_s` after, and the burst is
JPEG-encoded and handed to the uplink. Each frame carries camera id, monotonic time and zone ids.

Sources: `Picamera2Source` (the real Pi camera, imported only on a Pi) and `FileSource` (a video
file or a webcam, through the pipeline's own VideoSource) so the whole node runs on a laptop.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import cv2
import numpy as np


@dataclass
class NodeFrame:
    t: float            # monotonic seconds (Pi: sensor timestamp; file: frame index / fps)
    lores: np.ndarray   # BGR, low resolution
    main: Any           # full-res frame in whatever form the source's encode() takes


class FrameSource(Protocol):
    size: tuple[int, int]     # full-res (w, h)
    fps: float

    def frames(self) -> Iterator[NodeFrame]: ...
    def encode(self, main: Any, quality: int) -> bytes: ...   # one JPEG


def _jpeg(bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


class FileSource:
    """Mock camera: a video file (time = frame index / fps, as fast as it decodes) or a webcam index
    (time = time.monotonic()). Frames are thinned to about `fps`."""

    def __init__(self, src: str | int, fps: float = 10.0, lores_width: int = 320, max_frames: int | None = None):
        from bree.ingest.source import VideoSource
        self._src = VideoSource(src, max_frames=max_frames)
        self.live = self._src.live            # live = frame times are time.monotonic()
        self.step = max(1, round(self._src.fps / fps)) if not self._src.live else 1
        self.fps = self._src.fps / self.step if not self._src.live else fps
        self.size = (self._src.width, self._src.height)
        w = min(lores_width, self.size[0])
        self._lores = (w, max(2, round(self.size[1] * w / self.size[0])))

    def frames(self) -> Iterator[NodeFrame]:
        last = -math.inf
        for f in self._src:
            if self._src.live:
                t = time.monotonic()
                if t - last < 1.0 / self.fps:
                    continue
                last = t
            elif f.index % self.step:
                continue
            else:
                t = f.t
            yield NodeFrame(t, cv2.resize(f.image, self._lores, interpolation=cv2.INTER_AREA), f.image)

    def encode(self, main: np.ndarray, quality: int) -> bytes:
        return _jpeg(main, quality)


class Picamera2Source:
    """Camera Module 3 through picamera2: a full-res `main` stream and a small `lores` stream from the
    same sensor frame (the ISP scales it, so the low-res frame costs the CPU nothing). Both are YUV420:
    4.5 MB per 2304x1296 frame in the ring instead of 9.0 MB as RGB. NOT RUN YET: no Pi was available when
    this was written; the calls follow the picamera2 manual. Expect to adjust on first boot."""

    def __init__(self, size: tuple[int, int] = (2304, 1296), fps: float = 10.0, lores_width: int = 320):
        from picamera2 import Picamera2   # only exists on Raspberry Pi OS
        self.live = True
        self.size, self.fps = tuple(size), fps
        lw = lores_width - lores_width % 32                      # stream widths must align
        self._lores = (lw, int(size[1] * lw / size[0]) // 2 * 2)
        self.cam = Picamera2()
        us = int(1e6 / fps)
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": self.size, "format": "YUV420"}, lores={"size": self._lores, "format": "YUV420"},
            controls={"FrameDurationLimits": (us, us)}, buffer_count=4))
        self.cam.start()

    def frames(self) -> Iterator[NodeFrame]:
        while True:
            req = self.cam.capture_request()
            try:
                # SensorTimestamp is nanoseconds since boot, the same clock as time.monotonic() on a Pi
                # that never suspends.
                t = req.get_metadata()["SensorTimestamp"] / 1e9
                lores = cv2.cvtColor(req.make_array("lores"), cv2.COLOR_YUV2BGR_I420)[:self._lores[1], :self._lores[0]]
                main = req.make_array("main")                    # a copy: the request goes back to the camera
            finally:
                req.release()
            yield NodeFrame(t, lores, main)

    def encode(self, main: np.ndarray, quality: int) -> bytes:
        # Software JPEG (the Pi's hardware MJPEG encoder stops at 1080p). Called from the node's encode thread.
        return _jpeg(cv2.cvtColor(main, cv2.COLOR_YUV2BGR_I420)[:self.size[1], :self.size[0]], quality)


@dataclass
class BurstFrame:
    t: float
    zones: list[str]     # zones active at this frame ([] in pre-roll and post-roll)
    phase: str           # pre | during | post
    main: Any


@dataclass
class Burst:
    camera_id: str
    trigger_id: int      # one per trigger; a long trigger is cut into parts that share it
    part: int
    t_trigger: float
    zones: list[str]     # every zone active during the burst
    frames: list[BurstFrame] = field(default_factory=list)


class BurstRecorder:
    """Ring buffer plus burst assembly. push() once per captured frame; it returns finished bursts."""

    def __init__(self, camera_id: str, fps: float, pre_roll_s: float = 2.0, post_roll_s: float = 1.5,
                 max_burst_s: float = 6.0):
        self.camera_id, self.post_roll_s, self.max_burst_s = camera_id, post_roll_s, max_burst_s
        self.ring: deque[tuple[float, Any]] = deque(maxlen=max(1, math.ceil(pre_roll_s * fps)))
        self.cur: Burst | None = None
        self._last_active = 0.0
        self._n = 0

    def push(self, t: float, main: Any, active: set[str]) -> list[Burst]:
        zones = sorted(active)
        if self.cur is None:
            if not active:
                self.ring.append((t, main))
                return []
            self._n += 1
            self.cur = Burst(self.camera_id, self._n, 0, t, [],
                             [BurstFrame(rt, [], "pre", rm) for rt, rm in self.ring])
            self.ring.clear()
        b = self.cur
        if active:
            self._last_active = t
            b.zones = sorted(set(b.zones) | active)
        b.frames.append(BurstFrame(t, zones, "during" if active else "post", main))
        if t - self._last_active >= self.post_roll_s:
            self.cur = None
            return [b]
        if t - b.frames[0].t >= self.max_burst_s:      # long visit: ship what we have, keep recording
            self.cur = Burst(self.camera_id, b.trigger_id, b.part + 1, b.t_trigger, zones)
            return [b]
        return []

    def flush(self) -> list[Burst]:
        b, self.cur = self.cur, None
        return [b] if b and b.frames else []
