"""Online ghost suppression: learn static low-confidence person boxes while the camera runs.

`ghost.find_ghosts` learns ghosts from a calibration clip, but real ghosts come and go (a lit display above the counter
fired in about 1% of one 5 minute segment and in 28% of the next). This keeps a small set of clusters of recent low
confidence person boxes. A cluster that persists in the same place for `min_s` seconds, never gets confident, and
has a low mean confidence is flagged as a ghost and its detections are dropped from then on. One confident detection
(`release_conf`) in a flagged cluster unflags it, so a real clerk who stands in a flagged place for long comes back.
Cost: the first `min_s` of a ghost still leaks through (the track gate, bree.perception.gate, absorbs most of it), and a
person who really stands still at low detector confidence for longer than `min_s` is lost until the detector becomes
confident, so use it on cameras where staff are seen clearly or accept that trade.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Cluster:
    cx: float
    cy: float
    first: float
    last: float
    n: int
    sum_conf: float
    max_conf: float
    flagged: bool = False


class AdaptiveGhostMask:
    def __init__(self, fps: float, radius: float = 0.03, min_s: float = 8.0, gap_s: float = 6.0, max_mean_conf: float = 0.5,
                 max_peak_conf: float = 0.6, release_conf: float = 0.65, min_fill: float = 0.4):
        self.fps, self.radius, self.min_s, self.gap_s = fps, radius, min_s, gap_s
        self.max_mean, self.max_peak, self.release, self.min_fill = max_mean_conf, max_peak_conf, release_conf, min_fill
        self.clusters: list[_Cluster] = []
        self.frame = 0

    def _flag(self, c: _Cluster) -> bool:
        dur = c.last - c.first
        return (dur >= self.min_s and c.n >= self.min_fill * dur * self.fps and c.sum_conf / c.n <= self.max_mean
                and c.max_conf <= self.max_peak)

    def update(self, boxes: np.ndarray, confs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Keep mask for this frame's person boxes (xyxy px). Call once per frame, also with no boxes."""
        t = self.frame / self.fps
        self.frame += 1
        keep = np.ones(len(boxes), bool)
        h, w = shape[:2]
        for i in range(len(boxes)):
            cx, cy = (boxes[i, 0] + boxes[i,2]) / 2 / w, (boxes[i, 1] + boxes[i, 3]) / 2 / h
            near = [c for c in self.clusters if np.hypot(c.cx - cx, c.cy - cy) <= self.radius]
            c = min(near, key=lambda c: np.hypot(c.cx - cx, c.cy - cy)) if near else None
            if c is not None and not c.flagged and t - c.last > self.gap_s:      # a stale run: start over
                self.clusters.remove(c)
                c = None
            if c is None:
                c = _Cluster(cx, cy, t, t, 0, 0.0, 0.0)
                self.clusters.append(c)
            if c.flagged and confs[i] >= self.release:                          # a real person stands here
                self.clusters.remove(c)
                continue
            c.last, c.n, c.sum_conf, c.max_conf = t, c.n + 1, c.sum_conf + float(confs[i]), max(c.max_conf, float(confs[i]))
            if not c.flagged and self._flag(c):
                c.flagged = True
            if c.flagged:
                keep[i] = False
        self.clusters = [c for c in self.clusters if c.flagged or t - c.last <= self.gap_s]
        return keep
