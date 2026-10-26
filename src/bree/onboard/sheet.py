"""Contact sheet with pixelated heads, for choosing roles and drawing zones by eye. Local use only.

Pixelation is best effort: the top part of every detected person box is blocked out (heads), so a person
the detector missed stays visible. The sheet and the stills are therefore still sensitive images: they are written
to the private station folder (mode 700) and must not be committed, uploaded or pasted into documents.
"""
from __future__ import annotations

import cv2
import numpy as np


def pixelate_people(img: np.ndarray, people_boxes, head_frac: float = 0.32, block: int = 14) -> np.ndarray:
    """People boxes are normalised (x0, y0, x1, y1). Pixelates the top head_frac of each box (a bit wider than the box)."""
    out = img.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in people_boxes:
        bw, bh = (x1 - x0), (y1 - y0)
        hx0, hx1 = int(max(0, (x0 - 0.15 * bw)) * w), int(min(1, (x1 + 0.15 * bw)) * w)
        hy0, hy1 = int(max(0, y0 - 0.05 * bh) * h), int(min(1, y0 + head_frac * bh) * h)
        if hx1 - hx0 < 2 or hy1 - hy0 < 2:
            continue
        roi = out[hy0:hy1, hx0:hx1]
        bs = max(2, int(block * w / 1280))
        small = cv2.resize(roi, (max(1, roi.shape[1] // bs), max(1, roi.shape[0] // bs)), interpolation=cv2.INTER_AREA)
        out[hy0:hy1, hx0:hx1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


def contact_sheet(tiles: list[tuple[str, np.ndarray]], cols: int = 3, tile_w: int = 560) -> np.ndarray:
    """tiles: (caption, image). Captions are drawn on a dark band under each tile."""
    th = int(tile_w * 9 / 16)
    cells = []
    for cap, img in tiles:
        h, w = img.shape[:2]
        s = min(tile_w / w, th / h)
        r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cell = np.zeros((th + 30, tile_w, 3), np.uint8)
        cell[:r.shape[0], :r.shape[1]] = r
        cv2.putText(cell, cap, (6, th + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)
