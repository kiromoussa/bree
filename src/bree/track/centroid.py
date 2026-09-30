"""Centroid tracker for small, fast-moving objects (products in a hand).

ByteTrack associates by box overlap (IoU). A 15-20 px product carried in a hand
moves about its own size per frame, so consecutive boxes barely overlap and
new tracks never get confirmed (seen on the toy clips: the product in hand was
never tracked). Here we match by centre distance instead, gated relative to the
object's size, within the same category.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Track:
    id: int
    category: str
    center: np.ndarray
    velocity: np.ndarray
    size: float
    missed: int = 0


class CentroidTracker:
    def __init__(self, max_missed: int = 5, gate_sizes: float = 3.0, min_gate_px: float = 30.0,
                 id_start: int = 100000):
        self.tracks: list[_Track] = []
        self.max_missed = max_missed
        self.gate_sizes, self.min_gate_px = gate_sizes, min_gate_px
        self.next_id = id_start

    def update(self, boxes: np.ndarray, labels: list[str]) -> list[int]:
        """Returns a track id per input box (same order)."""
        centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]) \
            if len(boxes) else np.zeros((0, 2))
        sizes = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0)

        # All candidate (distance, track, detection) pairs within the gate, greedily matched.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            pred = tr.center + tr.velocity
            gate = max(self.min_gate_px, self.gate_sizes * tr.size) * (1 + 0.5 * tr.missed)
            for di in range(len(boxes)):
                if labels[di] != tr.category:
                    continue
                d = float(np.linalg.norm(centers[di] - pred))
                if d <= gate:
                    pairs.append((d, ti, di))
        pairs.sort()
        ids = [-1] * len(boxes)
        used_t = set()
        for d, ti, di in pairs:
            if ti in used_t or ids[di] != -1:
                continue
            tr = self.tracks[ti]
            new_c = centers[di]
            tr.velocity = 0.5 * tr.velocity + 0.5 * (new_c - tr.center) / (1 + tr.missed)
            tr.center, tr.size, tr.missed = new_c, float(sizes[di]), 0
            ids[di] = tr.id
            used_t.add(ti)
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for di in range(len(boxes)):
            if ids[di] == -1:
                tr = _Track(self.next_id, labels[di], centers[di], np.zeros(2), float(sizes[di]))
                self.next_id += 1
                self.tracks.append(tr)
                ids[di] = tr.id
        return ids
