"""Toy perception backend for the rendered 2D toy clips (see bree/sim/toy_render.py).

It detects people, hands, and products from PIXELS by colour (it does not read
the renderer's ground truth), then synthesises COCO-17 keypoints from each
person's blob geometry plus the detected hand blobs, the way a pose model would
report them. This lets the toy clips exercise the real tracker, event engine,
ledger, and alert path. It is TOY DATA ONLY: it says nothing about how YOLO
performs on real store footage.
"""
from __future__ import annotations

import cv2
import numpy as np

from bree.detect.base import Detections

# BGR palette shared with the renderer.
PERSON_COLORS = [(170, 60, 120), (120, 40, 40), (40, 70, 110), (100, 100, 20)]
HAND_COLOR = (0, 140, 255)
PRODUCT_COLORS = {
    "soda_bottle": (30, 30, 230),
    "energy_drink": (230, 230, 30),
    "water_bottle": (230, 140, 60),
    "candy_bar": (20, 220, 230),
    "chips_bag": (40, 200, 40),
    "jerky": (60, 90, 160),
    "lighter": (230, 40, 230),
}
TOL = 38


PALETTE = PERSON_COLORS + [HAND_COLOR] + list(PRODUCT_COLORS.values())


def _build_lut(tol: int = TOL) -> np.ndarray:
    """5-bit-per-channel colour cube -> palette label (0 = background, k = PALETTE[k-1])."""
    q = (np.arange(32) * 8 + 4).astype(np.int16)
    b, g, r = np.meshgrid(q, q, q, indexing="ij")
    cube = np.stack([b, g, r], axis=-1).reshape(-1, 3)
    lut = np.zeros(len(cube), np.uint8)
    best = np.full(len(cube), tol, np.int16)
    for k, color in enumerate(PALETTE, start=1):
        d = np.abs(cube - np.array(color, np.int16)).max(axis=1)
        hit = d < best
        lut[hit], best[hit] = k, d[hit]
    return lut


_LUT = _build_lut()


def _labels(img: np.ndarray) -> np.ndarray:
    """Per-pixel palette label in one vectorised pass."""
    q = (img >> 3).astype(np.int32)
    return _LUT[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]


def _blobs(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(x, y, x + w, y + h, a) for x, y, w, h, a in stats[1:] if a >= min_area]


def synth_keypoints(box, hands: list[tuple[float, float]]) -> np.ndarray:
    """COCO-17 keypoints from a person blob box (head on top) and hand positions."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    k = np.zeros((17, 3), np.float32)
    head_y = y1 + 0.09 * h
    k[0] = (cx, head_y, 0.9)                                  # nose
    k[1], k[2] = (cx - 4, head_y - 3, 0.9), (cx + 4, head_y - 3, 0.9)   # eyes
    k[3], k[4] = (cx - 9, head_y, 0.9), (cx + 9, head_y, 0.9)           # ears
    sh_y, hip_y = y1 + 0.24 * h, y1 + 0.62 * h
    k[5], k[6] = (x1 + 0.1 * w, sh_y, 0.9), (x2 - 0.1 * w, sh_y, 0.9)   # shoulders (L, R in image)
    k[11], k[12] = (x1 + 0.2 * w, hip_y, 0.9), (x2 - 0.2 * w, hip_y, 0.9)
    k[13], k[14] = (x1 + 0.25 * w, y1 + 0.8 * h, 0.9), (x2 - 0.25 * w, y1 + 0.8 * h, 0.9)
    k[15], k[16] = (x1 + 0.25 * w, y2, 0.9), (x2 - 0.25 * w, y2, 0.9)
    hands = sorted(hands)[:2]
    if len(hands) == 1:   # one visible hand: assign to the nearer side
        side = 9 if hands[0][0] < cx else 10
        k[side] = (*hands[0], 0.9)
    elif len(hands) == 2:
        k[9], k[10] = (*hands[0], 0.9), (*hands[1], 0.9)
    for wrist, shoulder, elbow in ((9, 5, 7), (10, 6, 8)):
        if k[wrist, 2] > 0:
            k[elbow] = ((k[wrist, 0] + k[shoulder, 0]) / 2, (k[wrist, 1] + k[shoulder, 1]) / 2, 0.9)
    return k


class ToyBackend:
    name = "toy"

    def __init__(self, min_person_area: int = 1500, min_product_area: int = 60, hand_reach_px: float = 110):
        self.min_person_area = min_person_area
        self.min_product_area = min_product_area
        self.hand_reach_px = hand_reach_px

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]:
        lab = _labels(image)
        mask = lambda color: (lab == PALETTE.index(color) + 1).astype(np.uint8)
        people = []
        for color in PERSON_COLORS:
            people += [b[:4] for b in _blobs(mask(color), self.min_person_area)]
        hands = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2, _ in _blobs(mask(HAND_COLOR), 40)]

        # Each hand belongs to the nearest person whose (expanded) box it is near.
        assigned: list[list[tuple[float, float]]] = [[] for _ in people]
        for hx, hy in hands:
            best, best_d = None, self.hand_reach_px
            for i, (x1, y1, x2, y2) in enumerate(people):
                dx = max(x1 - hx, 0, hx - x2)
                dy = max(y1 - hy, 0, hy - y2)
                d = float(np.hypot(dx, dy))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                assigned[best].append((hx, hy))

        if people:
            pb = np.array(people, np.float32)
            kp = np.stack([synth_keypoints(b, h) for b, h in zip(people, assigned)])
            persons = Detections(pb, np.full(len(pb), 0.9, np.float32), ["person"] * len(pb), kp)
        else:
            persons = Detections.empty(with_kpts=True)

        boxes, labels = [], []
        for cat, color in PRODUCT_COLORS.items():
            for x1, y1, x2, y2, _ in _blobs(mask(color), self.min_product_area):
                boxes.append((x1, y1, x2, y2))
                labels.append(cat)
        products = (Detections(np.array(boxes, np.float32), np.full(len(boxes), 0.85, np.float32), labels)
                    if boxes else Detections.empty())
        return persons, products
