"""Empty-store background of a camera: the per-pixel median of frames spread over the recorded segments.

People move, the store does not, so the median over a few dozen frames from different minutes is the scene without
them (floor tiles, tape, shelf ends). The calibration helper (bree.fusion.calibrate) proposes floor tiles from it
and draws its overlays on it. Local data: it is a picture of the store, kept with the footage, never in git.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def median_background(segments: list[Path], per_segment: int = 6, max_w: int = 2560) -> np.ndarray:
    import cv2
    frames = []
    for seg in segments:
        cap = cv2.VideoCapture(str(seg))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(int(n * 0.05), int(n * 0.95), per_segment).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if ok:
                if im.shape[1] > max_w:
                    im = cv2.resize(im, (max_w, int(im.shape[0] * max_w / im.shape[1])), interpolation=cv2.INTER_AREA)
                frames.append(im)
        cap.release()
    if not frames:
        raise ValueError("no frames could be read")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
