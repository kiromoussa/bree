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
    """Greedy matching in normalised coordinates. Labels are rough person centres (usually the torso), so a label
    matches a box that contains it (grown by 10% of its size) or lies within `thresh` of the box centre; the
    cheapest pairs, by distance relative to box size, are taken first."""
    h, w = shape[:2]
    pairs = []
    for i, (lx, ly) in enumerate(label_centres):
        for j, b in enumerate(boxes):
            bw, bh = (b[2] - b[0]) / w, (b[3] - b[1]) / h
            cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
            inside = abs(lx - cx) <= 0.55 * bw and abs(ly - cy) <= 0.55 * bh
            d = np.hypot(lx - cx, ly - cy)
            if inside or d <= thresh:
                pairs.append((np.hypot((lx - cx) / max(bw, 1e-3), (ly - cy) / max(bh, 1e-3)), i, j))
    ul, up, tp = set(), set(), 0
    for _, i, j in sorted(pairs):
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
