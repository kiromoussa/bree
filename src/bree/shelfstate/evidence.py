"""Evidence pictures for a shelf-state event: the place before, the place after, and the whole view with the box.

People are pixelated whole (not only heads): any person box in the frame is mosaicked before anything is saved.
The 'before' picture comes from the reference, which is built from person-free cells, so it normally has no one in it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def pixelate_people(img: np.ndarray, persons: np.ndarray, grow: float = 0.15, block: int = 8) -> np.ndarray:
    """Copy of img with every normalised person box (x0, y0, x1, y1, ...) replaced by a coarse mosaic."""
    out = img.copy()
    h, w = img.shape[:2]
    for x0, y0, x1, y1, *_ in persons:
        bw, bh = x1 - x0, y1 - y0
        a, b = int(max(0, (x0 - grow * bw) * w)), int(max(0, (y0 - grow * bh) * h))
        c, d = int(min(w, (x1 + grow * bw) * w)) + 1, int(min(h, (y1 + grow * bh) * h)) + 1
        if c - a < 2 or d - b < 2:
            continue
        roi = out[b:d, a:c]
        small = cv2.resize(roi, (max(1, (c - a) // block), max(1, (d - b) // block)), interpolation=cv2.INTER_AREA)
        out[b:d, a:c] = cv2.resize(small, (c - a, d - b), interpolation=cv2.INTER_NEAREST)
    return out


def crop_box(img: np.ndarray, box, margin: float = 0.35, min_side: int = 64, up: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = max((x1 - x0) * margin, min_side / 2 / w), max((y1 - y0) * margin, min_side / 2 / h)
    a, b = int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h))
    c, d = int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h))
    return cv2.resize(img[b:d, a:c], None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)


def write_evidence(out_dir: str | Path, stem: str, before: np.ndarray, after: np.ndarray, box, persons: np.ndarray) -> dict:
    """Save before / after crops and the context view. Returns {"before", "after", "context"} file names (relative)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    before, after = pixelate_people(before, persons), pixelate_people(after, persons)
    h, w = after.shape[:2]
    ctx = after.copy()
    cv2.rectangle(ctx, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (0, 255, 255), 2)
    names = {"before": f"{stem}_before.jpg", "after": f"{stem}_after.jpg", "context": f"{stem}_ctx.jpg"}
    cv2.imwrite(str(d / names["before"]), crop_box(before, box), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(d / names["after"]), crop_box(after, box), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(d / names["context"]), ctx, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return names
