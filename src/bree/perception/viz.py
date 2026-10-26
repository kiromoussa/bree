"""Draw labels (green) and detections (red, with conf) on a cached labelled frame for eyeballing. Local use only:
the output contains people, so it is written under <data>/perception/viz, never into a repo."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bree.perception import frames as fr
from bree.perception.labels import DATA_DIR, LabelledFrame
from bree.perception.scoring import load_dets


def draw(lf: LabelledFrame, config: str, conf: float = 0.3, width: int = 1280, data_dir: Path = DATA_DIR,
         dets: dict | None = None):
    im = fr.load(lf, data_dir)
    if im is None:
        return None
    h, w = im.shape[:2]
    for x, y in lf.centres:
        cv2.circle(im, (int(x * w), int(y * h)), 14, (0, 255, 0), 4)
    d = (dets or load_dets(config, lf.channel, data_dir)).get(lf.key, {"boxes": [], "confs": []})
    for b, c in zip(d["boxes"], d["confs"]):
        if c >= conf:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
            cv2.putText(im, f"{c:.2f}", (int(b[0]), max(20, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return cv2.resize(im, (width, int(h * width / w)))
