"""Pictures of flow results drawn on a person-free background: track trails, zone outlines, heatmaps. The background is
the median of frames sampled across a segment (people who move disappear), so these images show the layout, not people.
Check any background before it goes into docs: a person who stood still all segment survives the median."""
from __future__ import annotations

import cv2
import numpy as np

COLORS = [(66, 135, 245), (245, 166, 35), (80, 200, 120), (230, 90, 140), (150, 110, 230), (60, 200, 220), (200, 200, 60)]


def median_background(video: str, n: int = 40, size=(1280, 720)) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            frames.append(cv2.resize(im, size))
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8)


def draw_layout(img: np.ndarray, layout, scale: float = 1.0) -> np.ndarray:
    out = img.copy()
    for z in layout.zones:
        p = (z.polygon * scale).astype(np.int32)
        cv2.polylines(out, [p], True, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, z.name, tuple(p[0] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    a, b = (layout.door.a * scale).astype(int), (layout.door.b * scale).astype(int)
    cv2.line(out, tuple(a), tuple(b), (0, 0, 255), 2, cv2.LINE_AA)
    return out


def draw_tracks(img: np.ndarray, tracks, scale: float = 1.0, labels: dict | None = None) -> np.ndarray:
    out = img.copy()
    for k, tr in enumerate(tracks):
        col = COLORS[tr.id % len(COLORS)]
        pts = np.array([[p[1], p[2]] for p in tr.path]) * scale
        cv2.polylines(out, [pts.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        cv2.circle(out, tuple(pts[0].astype(int)), 5, col, -1)
        cv2.rectangle(out, tuple((pts[-1] - 5).astype(int)), tuple((pts[-1] + 5).astype(int)), col, 2)
        txt = f"{tr.id}" + (f" {labels[tr.id]}" if labels and tr.id in labels else "")
        cv2.putText(out, txt, tuple((pts[0] + [6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def heatmap_overlay(img: np.ndarray, grid: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend a (rows, cols) seconds-per-cell grid over the image (log-compressed so a long stand does not hide the walks)."""
    g = np.log1p(grid.astype(np.float32))
    if g.max() > 0:
        g = g / g.max()
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    g = np.clip(g / max(g.max(), 1e-6), 0, 1)
    col = cv2.applyColorMap((cv2.resize(g, (img.shape[1], img.shape[0])) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    mask = cv2.resize((g > 0.03).astype(np.float32), (img.shape[1], img.shape[0]))[..., None]
    return (img * (1 - alpha * mask) + col * alpha * mask).astype(np.uint8)
