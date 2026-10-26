"""Learn where the detector keeps seeing a "person" that never moves: a poster, a dark display, a rack edge, a reflection.

The pipeline baselines on this store found such ghosts on every aisle camera (47 percent of one camera's person boxes sat on one
spot). A real person who stands still is confident and sooner or later moves, so a cell is masked only when boxes fall in it in
at least `min_share` of the sampled frames AND their mean confidence is under `max_conf`. The mask is learned from the footage
itself, per camera, and is returned as polygons in video pixels for `FlowConfig.ignore` (judged by the foot point).
"""
from __future__ import annotations

import numpy as np


def learn_static_mask(frames: list[dict], size: tuple[int, int], cell_px: int = 64, min_share: float = 0.5,
                      max_conf: float = 0.65, pad_cells: int = 1) -> list[list[list[float]]]:
    """frames: detection frames (bree.intel.detections); size: (width, height) of the video. Returns rectangles as polygons."""
    w, h = size
    cols, rows = max(w // cell_px, 1), max(h // cell_px, 1)
    hits = np.zeros((rows, cols))
    conf = np.zeros((rows, cols))
    for f in frames:
        seen = set()
        for p in f["persons"]:
            b = p["bbox"]
            c, r = min(int((b[0] + b[2]) / 2 / w * cols), cols - 1), min(max(int(b[3] / h * rows), 0), rows - 1)
            if (r, c) in seen:
                continue
            seen.add((r, c))
            hits[r, c] += 1
            conf[r, c] += p["conf"]
    n = max(len(frames), 1)
    mask = (hits / n >= min_share) & (conf / np.maximum(hits, 1) < max_conf)
    out = []
    for r, c in zip(*np.nonzero(mask)):
        x0, x1 = max(c - pad_cells, 0) * w / cols, min(c + 1 + pad_cells, cols) * w / cols
        y0, y1 = max(r - pad_cells, 0) * h / rows, min(r + 1 + pad_cells, rows) * h / rows
        out.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return out
