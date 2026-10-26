"""Sliced (tiled) person detection for small people in a large frame.

A 2560x1440 frame letterboxed to 640 shrinks a person 40 px tall at the far counter to 10 px. Running the detector
on overlapping windows at the model's native scale keeps them large. Windows come from a spec (grid + optional
regions of interest); the results are merged with a box NMS that also drops a box lying mostly inside a larger one
(a person cut by a window edge shows up as a partial box next to the whole one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SliceSpec:
    cols: int = 1                 # grid of overlapping windows; 1x1 with full_frame=True is the plain detector
    rows: int = 1
    overlap: float = 0.2          # fraction of a window shared with its neighbour
    full_frame: bool = True       # also run the whole frame (large and near people, no edge cuts)
    rois: list[list[float]] = field(default_factory=list)   # extra windows [x0, y0, x1, y1] normalised 0..1
    window_imgsz: int | None = None   # input size for windows; None = the detector's imgsz

    @property
    def n_windows(self) -> int:
        return (self.cols * self.rows if self.cols * self.rows > 1 else 0) + int(self.full_frame) + len(self.rois)


def grid_windows(w: int, h: int, cols: int, rows: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """cols x rows windows covering the frame, each overlapping its neighbours by `overlap` of its own size."""
    def axis(n: int, size: int) -> list[tuple[int, int]]:
        if n <= 1:
            return [(0, size)]
        win = size / (n - (n - 1) * overlap)       # n windows of this length, stride win*(1-overlap), cover `size`
        stride = win * (1 - overlap)
        return [(int(round(i * stride)), min(size, int(round(i * stride + win)))) for i in range(n)]
    return [(x0, y0, x1, y1) for y0, y1 in axis(rows, h) for x0, x1 in axis(cols, w)]


def plan_windows(shape: tuple[int, int], spec: SliceSpec) -> list[tuple[int, int, int, int]]:
    """Pixel windows (x0, y0, x1, y1) for an image of shape (h, w[, c])."""
    h, w = shape[:2]
    wins: list[tuple[int, int, int, int]] = []
    if spec.full_frame:
        wins.append((0, 0, w, h))
    if spec.cols * spec.rows > 1:
        wins += grid_windows(w, h, spec.cols, spec.rows, spec.overlap)
    for x0, y0, x1, y1 in spec.rois:
        wins.append((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return wins


def _areas(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)


def merge_boxes(boxes: np.ndarray, confs: np.ndarray, iou: float = 0.55, ios: float = 0.8) -> np.ndarray:
    """Indices kept after merging overlapping window detections. A box is dropped when a higher-scoring box overlaps
    it with IoU >= iou, or covers >= ios of the smaller box and the smaller box is not the more confident one."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    order = np.argsort(-confs)
    area = _areas(boxes)
    keep: list[int] = []
    for i in order:
        drop = False
        for j in keep:
            ix = max(0.0, min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])) * \
                 max(0.0, min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1]))
            union = area[i] + area[j] - ix
            if ix / max(union, 1e-9) >= iou or ix / max(min(area[i], area[j]), 1e-9) >= ios:
                drop = True
                break
        if not drop:
            keep.append(int(i))
    return np.array(sorted(keep), int)


def touches_cut(box: np.ndarray, win: tuple[int, int, int, int], frame: tuple[int, int], margin: int = 4) -> bool:
    """True if the box touches a window edge that is NOT the frame edge: the person is probably cut by the window."""
    x0, y0, x1, y1 = win
    h, w = frame
    return bool((box[0] <= x0 + margin and x0 > 0) or (box[1] <= y0 + margin and y0 > 0)
                or (box[2] >= x1 - margin and x1 < w) or (box[3] >= y1 - margin and y1 < h))


def sliced_predict(model, image: np.ndarray, spec: SliceSpec, imgsz: int, conf: float, class_ids: list[int],
                   device: str = "cpu", iou_merge: float = 0.55, half: bool = False):
    """Run `model` (an Ultralytics YOLO) over the windows and return merged (boxes xyxy, confs, cls) in image pixels.
    All windows go in one batched predict call per input size."""
    wins = plan_windows(image.shape, spec)
    all_b, all_c, all_k, all_w = [], [], [], []
    groups: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for n, win in enumerate(wins):
        full = win == (0, 0, image.shape[1], image.shape[0])
        groups.setdefault(imgsz if full or spec.window_imgsz is None else spec.window_imgsz, []).append((n, win))
    for size, items in groups.items():
        crops = [np.ascontiguousarray(image[y0:y1, x0:x1]) for _, (x0, y0, x1, y1) in items]
        res = model.predict(crops, imgsz=size, conf=conf, device=device, classes=class_ids, verbose=False, half=half)
        for (n, (x0, y0, x1, y1)), r in zip(items, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            full = (x0, y0, x1, y1) == (0, 0, image.shape[1], image.shape[0])
            c = r.boxes.conf.cpu().numpy()
            k = r.boxes.cls.cpu().numpy().astype(int)
            if not full:       # drop window-edge cuts; the neighbouring window or the full frame has the whole person
                ok = np.array([not touches_cut(bb, (x0, y0, x1, y1), image.shape[:2]) for bb in b], bool)
                b, c, k = b[ok], c[ok], k[ok]
            all_b.append(b); all_c.append(c); all_k.append(k)
    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    b, c, k = np.concatenate(all_b), np.concatenate(all_c), np.concatenate(all_k)
    keep = merge_boxes(b, c, iou_merge)
    return b[keep], c[keep], k[keep]
