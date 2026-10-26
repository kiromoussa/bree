"""Score stored detections (bree.perception.sweep) against the labels, optionally through a profile's filters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bree.perception.evalcounts import FrameScore, match_frame, summarize
from bree.perception.labels import DATA_DIR, LabelledFrame, load_labels


def load_dets(config: str, channel: int, data_dir: Path = DATA_DIR) -> dict:
    p = data_dir / "perception" / "dets" / config / f"ch{channel:02d}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def score(config: str, channel: int, split: str, conf: float = 0.3, filt=None, data_dir: Path = DATA_DIR,
          dets: dict | None = None, labels: list[LabelledFrame] | None = None) -> dict:
    """`filt(boxes, confs, shape) -> keep mask` applies profile filters (ghost mask, min height ...)."""
    dets = dets if dets is not None else load_dets(config, channel, data_dir)
    out = []
    for lf in (labels if labels is not None else load_labels(channel, data_dir)):
        if lf.split != split or lf.key not in dets:
            continue
        d = dets[lf.key]
        b, c = np.array(d["boxes"], float).reshape(-1, 4), np.array(d["confs"], float)
        k = c >= conf
        b, c = b[k], c[k]
        if filt is not None and len(b):
            m = filt(b, c, tuple(d["shape"]))
            b, c = b[m], c[m]
        out.append(match_frame(lf.centres, b, tuple(d["shape"])))
    return summarize(out)


def f1(s: dict) -> float:
    r, p = s.get("recall"), s.get("precision")
    if not r or not p:
        return 0.0
    return 2 * r * p / (r + p)
