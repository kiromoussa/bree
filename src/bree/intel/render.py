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


def pixelate_heads(img: np.ndarray, boxes, frac: float = 0.22, block: int = 10) -> np.ndarray:
    """Pixelate the top `frac` of each person box (the head) so a debug image can be looked at or kept without faces."""
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        h = y1 - y0
        ya, yb = int(max(y0 - 0.03 * h, 0)), int(min(y0 + frac * h, img.shape[0]))
        xa, xb = int(max(x0, 0)), int(min(x1, img.shape[1]))
        if yb - ya < 2 or xb - xa < 2:
            continue
        roi = out[ya:yb, xa:xb]
        small = cv2.resize(roi, (max((xb - xa) // block, 1), max((yb - ya) // block, 1)), interpolation=cv2.INTER_LINEAR)
        out[ya:yb, xa:xb] = cv2.resize(small, (xb - xa, yb - ya), interpolation=cv2.INTER_NEAREST)
    return out


def annotated_frames(video: str, dets: list[dict], frame_log: list, layout, times: list[float], size=(1280, 720),
                     pixelate: bool = True) -> list[np.ndarray]:
    """Frames of `video` at the sampled moments nearest `times`, with zones, track ids and roles drawn (heads pixelated)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ft = np.array([t for t, _ in frame_log])
    out = []
    for t in times:
        i = int(np.argmin(abs(ft - t)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(dets[i]["frame"])))
        ok, im = cap.read()
        if not ok:
            continue
        sx, sy = size[0] / im.shape[1], size[1] / im.shape[0]
        boxes = [[p["bbox"][0] * sx, p["bbox"][1] * sy, p["bbox"][2] * sx, p["bbox"][3] * sy] for p in dets[i]["persons"]]
        im = cv2.resize(im, size)
        if pixelate:
            im = pixelate_heads(im, boxes)
        scale = size[0] / layout.size[0]
        im = draw_layout(im, layout, scale)
        for p, b in zip(dets[i]["persons"], boxes):
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 1)
        for tid, role, x, y in frame_log[i][1]:
            col = (60, 200, 255) if role == "staff" else (80, 220, 80)
            cv2.circle(im, (int(x * scale), int(y * scale)), 7, col, -1)
            cv2.putText(im, f"{tid} {role[:3]}", (int(x * scale) + 8, int(y * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(im, f"t={dets[i]['t']:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        out.append(im)
    cap.release()
    return out


def plan_view(grid: np.ndarray, layout, scale: int = 6) -> np.ndarray | None:
    """Heat grid (image cells, seconds) re-projected onto the layout's schematic floor plan (plan units, not metres) with the
    zone outlines. The plan homography is four hand-picked floor points: treat positions as schematic, see docs/ultra/intel.md."""
    if layout.homography is None:
        return None
    pw, ph = layout.plan_size
    rows, cols = grid.shape
    k = layout.size[0] / (cols * 8.0)                 # grid -> 8 px per cell -> layout pixels
    up = cv2.resize(grid.astype(np.float32), (cols * 8, rows * 8), interpolation=cv2.INTER_LINEAR)
    m = np.diag([scale, scale, 1.0]) @ layout.homography @ np.diag([k, k, 1.0])
    acc = cv2.warpPerspective(up, m, (int(pw * scale), int(ph * scale)), flags=cv2.INTER_LINEAR)
    ph_img = (int(ph) * scale, int(pw) * scale)
    acc = cv2.resize(acc, (ph_img[1] // scale, ph_img[0] // scale), interpolation=cv2.INTER_AREA)
    img = np.full((int(ph) * scale, int(pw) * scale, 3), 245, np.uint8)
    img = heatmap_overlay(img, acc, alpha=0.9)
    for z in layout.zones:
        pts = [layout.to_plan(x, y) for x, y in z.polygon]
        if all(pts):
            cv2.polylines(img, [(np.array(pts) * scale).astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(img, z.name, tuple((np.array(pts[0]) * scale + [3, 12]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return img
