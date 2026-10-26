"""Where onboarding gets frames from: recorded footage on disk, or a live DVR through the Imou platform.

Both give, per channel, a handful of frames spread over time (full size) and the stream's resolution.
Live discovery opens each online channel's sub stream one at a time for a few seconds. That consumes the
platform's metered live flow, so it is bounded (`seconds` per channel, one channel at once) and never
retries a stub: a channel whose stream is a stub is reported as unreachable with the reason.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ChannelSample:
    channel: int
    frames: list[np.ndarray] = field(default_factory=list)
    times: list[float] = field(default_factory=list)       # seconds from the first sampled frame (footage: position in file)
    size: tuple[int, int] = (0, 0)                          # (w, h) of the source stream
    fps: float = 0.0
    online: bool = True
    error: str | None = None                                # why no frames: offline / stub / unreadable
    source: str = ""                                        # file names or "live sub stream" (no URLs, no secrets)


def _segments(chdir: Path, min_bytes: int = 1_000_000) -> list[Path]:
    return sorted(p for p in chdir.glob("*.mp4") if p.stat().st_size > min_bytes)   # smaller = stub / partial


def sample_file(path: Path, n: int, skip_s: float = 3.0) -> tuple[list[np.ndarray], list[float], float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], [], 0.0, (0, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total / fps if total > 0 else 0.0
    skip_s = min(skip_s, dur * 0.1)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames, times = [], []
    for i in range(n):
        t = skip_s + (max(dur - 2 * skip_s, 0)) * (i + 0.5) / n if dur else 0
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if ok:
            frames.append(img)
            times.append(t)
    cap.release()
    return frames, times, fps, size


class FolderDiscovery:
    """`root` holds chNN/ folders of .mp4 segments (the layout bree.ingest.record writes)."""

    def __init__(self, root: str | Path, per_segment: int = 3, max_segments: int = 4, min_bytes: int = 1_000_000):
        self.root, self.per_segment, self.max_segments, self.min_bytes = Path(root), per_segment, max_segments, min_bytes

    def channels(self) -> list[int]:
        return sorted(int(m.group(1)) for p in self.root.glob("ch*") if p.is_dir() and (m := re.fullmatch(r"ch(\d+)", p.name)) and _segments(p, self.min_bytes))

    def sample(self, ch: int) -> ChannelSample:
        segs = _segments(self.root / f"ch{ch:02d}", self.min_bytes)
        if len(segs) > self.max_segments:   # spread over the recording, not just the first segments
            idx = np.linspace(0, len(segs) - 1, self.max_segments).round().astype(int)
            segs = [segs[i] for i in idx]
        out = ChannelSample(ch, source=",".join(s.name for s in segs))
        for s in segs:
            fr, ts, fps, size = sample_file(s, self.per_segment)
            out.frames += fr
            out.times += ts
            out.fps, out.size = fps or out.fps, size if size[0] else out.size
        if not out.frames:
            out.error = "no readable footage"
        return out


class ImouDiscovery:
    """Live DVR: deviceOnline for the channel list, then a short sub-stream sample per online channel."""

    def __init__(self, client, serial: str, seconds: float = 20.0, n_frames: int = 6, stream: str = "sub",
                 clock=time.time, sleep=time.sleep):
        self.client, self.serial, self.seconds, self.n, self.stream = client, serial, seconds, n_frames, stream
        self._status = None
        self._clock, self._sleep = clock, sleep

    def status(self):
        if self._status is None:
            self._status = self.client.device_online(self.serial)
        return self._status

    def channels(self) -> list[int]:
        st = self.status()
        if not st.online:
            from bree.ingest.imou import DeviceOfflineError
            raise DeviceOfflineError(f"device {self.serial} is offline")
        return sorted(st.channels)

    def sample(self, ch: int) -> ChannelSample:
        from bree.ingest.imou import ImouError, ImouSource, scrub
        out = ChannelSample(ch, source=f"live {self.stream} stream")
        if not self.status().channels.get(ch, True):
            out.online, out.error = False, "channel offline"
            return out
        try:
            src = ImouSource(f"imou://{self.serial}/{ch}?stream={self.stream}", client=self.client, max_stub_retries=1,
                             min_run_s=min(8.0, self.seconds / 2), check_online=False, max_wait_s=15)
        except ImouError as e:
            out.error = scrub(e)
            return out
        out.size, out.fps = (src.width, src.height), src.fps
        t0 = time.time()
        gap = max(src.fps * self.seconds / self.n, 1)
        try:
            for i, f in enumerate(src):
                if i % int(gap) == 0:
                    out.frames.append(f.image)
                    out.times.append(f.t)
                if len(out.frames) >= self.n or f.t > self.seconds:
                    break
        except ImouError as e:
            out.error = scrub(e)
        finally:
            src.release()
        if not out.frames and not out.error:
            out.error = "no frames"
        return out
