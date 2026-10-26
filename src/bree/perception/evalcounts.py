"""Score person detections against the agent-made count labels (REAL-MODEL-LABELLED, not ground truth).

Per frame: exact count agreement, absolute count error, presence (anyone / nobody) agreement, and greedy
position matching (label centre vs detection box centre, normalised distance) giving TP / FP / FN. Because the
labels are rough (+-0.03 to +-0.1) the position threshold is generous (0.10) and the count metrics matter more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameScore:
    n_label: int
    n_pred: int
    tp: int


def match_frame(label_centres, boxes: np.ndarray, shape: tuple[int, int], thresh: float = 0.10) -> FrameScore:
    """Greedy nearest-first matching in normalised coordinates."""
    h, w = shape[:2]
    pc = [((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h) for b in boxes]
    pairs = sorted((np.hypot(l[0] - p[0], l[1] - p[1]), i, j)
                   for i, l in enumerate(label_centres) for j, p in enumerate(pc))
    ul, up, tp = set(), set(), 0
    for d, i, j in pairs:
        if d > thresh:
            break
        if i in ul or j in up:
            continue
        ul.add(i); up.add(j); tp += 1
    return FrameScore(len(label_centres), len(boxes), tp)


def summarize(scores: list[FrameScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {"frames": 0}
    nl, npred, tp = (sum(getattr(s, k) for s in scores) for k in ("n_label", "n_pred", "tp"))
    return {
        "frames": n, "people_labelled": nl, "people_predicted": npred,
        "recall": round(tp / nl, 3) if nl else None,
        "precision": round(tp / npred, 3) if npred else None,
        "exact_count": round(sum(s.n_label == s.n_pred for s in scores) / n, 3),
        "mae": round(sum(abs(s.n_label - s.n_pred) for s in scores) / n, 3),
        "presence_agree": round(sum((s.n_label > 0) == (s.n_pred > 0) for s in scores) / n, 3),
        "false_alarm_frames": sum(s.n_label == 0 and s.n_pred > 0 for s in scores),
        "empty_label_frames": sum(s.n_label == 0 for s in scores),
    }
