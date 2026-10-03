"""Drawing: zones, person boxes + IDs + skeletons, products, recent events.

Privacy: `blur_heads` is applied to every frame we write to disk (annotated
video and alert clips). Heads are located from pose keypoints (nose/eyes/ears),
falling back to the top of the person box. We never crop or store faces.
"""
from __future__ import annotations

import cv2
import numpy as np

from bree.events.observations import HEAD_KPTS, PersonObs, ProductObs
from bree.events.types import Event
from bree.events.zones import StoreConfig

ZONE_COLORS = {"shelf": (80, 180, 80), "cooler": (200, 160, 40), "register": (60, 200, 230), "exit": (60, 60, 220),
               "entrance": (220, 60, 160)}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
EVENT_COLORS = {"pick": (0, 200, 0), "put_back": (200, 200, 0), "conceal": (0, 0, 255),
                "pay": (0, 200, 255), "exit": (255, 0, 255), "enter": (200, 200, 200)}


def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def head_box(p: PersonObs, kpt_conf: float = 0.3) -> tuple[int, int, int, int]:
    pts = [p.kpt(i, kpt_conf) for i in HEAD_KPTS]
    pts = [q for q in pts if q is not None]
    x1, y1, x2, y2 = p.bbox
    if len(pts) >= 2:
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        r = max(max(xs) - min(xs), max(ys) - min(ys), 0.12 * (x2 - x1)) * 1.1 + 6
    else:
        cx, cy, r = (x1 + x2) / 2, y1 + 0.1 * (y2 - y1), 0.18 * (y2 - y1) / 2 + 6
    return int(cx - r), int(cy - r), int(cx + r), int(cy + r)


def blur_heads(img: np.ndarray, persons: list[PersonObs]) -> np.ndarray:
    h, w = img.shape[:2]
    for p in persons:
        x1, y1, x2, y2 = head_box(p)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 > 2 and y2 - y1 > 2:
            # Pixelate: 5x5 blocks carry no identifiable facial detail.
            roi = img[y1:y2, x1:x2]
            small = cv2.resize(roi, (5, 5), interpolation=cv2.INTER_AREA)
            img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return img


def draw_zones(img: np.ndarray, store: StoreConfig, scale: float = 1.0) -> None:
    overlay = img.copy()
    for z in store.zones:
        pts = (z.polygon * scale).astype(np.int32)
        cv2.fillPoly(overlay, [pts], ZONE_COLORS[z.kind])
        cv2.putText(img, z.name, tuple(pts[0] + [4, 16]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ZONE_COLORS[z.kind], 1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)


def draw_frame(img: np.ndarray, store: StoreConfig | None, persons: list[PersonObs],
               products: list[ProductObs], recent: list[Event], baskets: dict[int, list[str]] | None = None,
               fps: float | None = None, blur: bool = True) -> np.ndarray:
    out = img.copy()
    if blur:
        blur_heads(out, persons)
    if store is not None:
        draw_zones(out, store)
    for p in persons:
        c = _color(p.track_id)
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"ID {p.track_id}"
        if baskets and baskets.get(p.track_id):
            label += " [" + ",".join(baskets[p.track_id]) + "]"
        cv2.putText(out, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if p.keypoints is not None:
            k = p.keypoints
            for a, b in SKELETON:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    cv2.line(out, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), c, 2)
            for j in (9, 10):
                if k[j, 2] > 0.3:
                    cv2.circle(out, (int(k[j, 0]), int(k[j, 1])), 5, (255, 255, 255), -1)
    for pr in products:
        x1, y1, x2, y2 = map(int, pr.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.putText(out, f"{pr.category}#{pr.track_id}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    y = 20
    for ev in recent[-6:]:
        txt = f"{ev.t:6.1f}s  P{ev.person_id} {ev.type.value} {ev.item or ''} {ev.zone or ''}"
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EVENT_COLORS.get(ev.type.value, (255, 255, 255)), 2)
        y += 18
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (out.shape[1] - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
