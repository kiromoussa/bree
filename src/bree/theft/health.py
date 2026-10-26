"""Per-frame image health features for tamper detection (bree.theft.tamper) and the perception cache.

One cheap pass on a downscaled grey frame: brightness, contrast, sharpness (variance of the Laplacian),
fraction of dark / blown-out pixels, and a 32x18 grey thumbnail (too small to show a face or a person; it is
what scene-change and "camera moved" are measured against). Nothing here keeps a full frame.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURES = ("mean", "std", "lap", "dark", "bright", "edge", "diff")
THUMB = (32, 18)           # (w, h)
ANALYSIS_W = 320           # frames are shrunk to this width before measuring (sharpness is relative to it)


def grey_small(img: np.ndarray, width: int = ANALYSIS_W) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small


def frame_features(grey: np.ndarray, prev_thumb: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """(feature dict, thumbnail uint8 (18, 32)) for one analysis-size grey frame. `diff` is the mean absolute
    difference to `prev_thumb` (0..255), 0.0 when there is none."""
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    thumb = cv2.resize(grey, THUMB, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(grey, 60, 140)
    f = {
        "mean": float(grey.mean()), "std": float(grey.std()), "lap": float(lap.var()),
        "dark": float((grey < 20).mean()), "bright": float((grey > 235).mean()),
        "edge": float((edges > 0).mean()),
        "diff": 0.0 if prev_thumb is None else float(np.abs(thumb.astype(np.float32) - prev_thumb.astype(np.float32)).mean()),
    }
    return f, thumb
