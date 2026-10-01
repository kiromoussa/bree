"""Per-frame perception output, independent of which detector produced it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint indices used by the event rules.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
HEAD_KPTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)


@dataclass
class PersonObs:
    track_id: int
    bbox: tuple[float, float, float, float]      # x1, y1, x2, y2
    conf: float = 1.0
    keypoints: np.ndarray | None = None          # (17, 3): x, y, visibility/conf

    def kpt(self, idx: int, min_conf: float = 0.3) -> tuple[float, float] | None:
        if self.keypoints is None:
            return None
        x, y, c = self.keypoints[idx]
        return (float(x), float(y)) if c >= min_conf else None

    def wrists(self, min_conf: float = 0.3) -> list[tuple[str, tuple[float, float]]]:
        out = []
        for name, idx in (("left", L_WRIST), ("right", R_WRIST)):
            p = self.kpt(idx, min_conf)
            if p is not None:
                out.append((name, p))
        return out

    def hands(self, min_conf: float = 0.3, extend: float = 0.0) -> list[tuple[str, tuple[float, float]]]:
        """Wrists pushed `extend` forearm lengths past the wrist (wrist + extend * (wrist - elbow)): the fingertips
        reach deeper into a shelf than the wrist. extend=0 (or no elbow) is the wrist itself."""
        out = []
        for name, w, e in (("left", L_WRIST, L_ELBOW), ("right", R_WRIST, R_ELBOW)):
            p = self.kpt(w, min_conf)
            if p is None:
                continue
            q = self.kpt(e, min_conf) if extend else None
            out.append((name, (p[0] + extend * (p[0] - q[0]), p[1] + extend * (p[1] - q[1])) if q else p))
        return out

    def foot_point(self, mode: str = "bottom") -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        if mode == "center":
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return ((x1 + x2) / 2, y2)

    def torso_box(self, margin: float = 0.25, min_conf: float = 0.3) -> tuple[float, float, float, float]:
        """Shoulders-to-hips box (pockets, waistband, bag at the hip), expanded by margin.
        Falls back to the middle of the person box when keypoints are missing."""
        pts = [self.kpt(i, min_conf) for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
        pts = [p for p in pts if p is not None]
        if len(pts) >= 3:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        else:
            bx1, by1, bx2, by2 = self.bbox
            h = by2 - by1
            x1, x2, y1, y2 = bx1, bx2, by1 + 0.2 * h, by1 + 0.65 * h
        w, h = x2 - x1, y2 - y1
        mx, my = margin * max(w, 10), margin * max(h, 10)
        return (x1 - mx, y1 - my, x2 + mx, y2 + my)


@dataclass
class ProductObs:
    track_id: int
    bbox: tuple[float, float, float, float]
    category: str
    conf: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameObs:
    frame: int
    t: float
    persons: list[PersonObs] = field(default_factory=list)
    products: list[ProductObs] = field(default_factory=list)
