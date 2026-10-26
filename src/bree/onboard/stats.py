"""Per-channel scene statistics for role suggestion: image statistics plus COCO detection statistics.

Everything here is a plain number that `bree.onboard.roles` turns into a role with its reasons. No learned
classifier, no labels: the numbers are inspectable in onboard_report.json. COCO detectors know people,
vehicles, bottles, screens, keyboards and fridges and nothing store-specific, so the statistics describe
the scene, they do not read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")
POS_GEAR = ("keyboard", "laptop", "tv", "cell phone", "mouse", "remote")     # screens, keyboards, card terminals
DRINKS = ("bottle", "cup", "wine glass", "can")
FRIDGE = ("refrigerator",)
SHELF_GOODS = DRINKS + ("book", "vase", "bowl", "banana", "apple", "orange", "sandwich", "donut", "cake")
DETECT_WIDTH = 1280


@dataclass
class Det:
    name: str
    conf: float
    box: tuple[float, float, float, float]       # normalised x0, y0, x1, y1


class YoloScene:
    """COCO objects (yolo26 detect weights) on frames resized to DETECT_WIDTH wide. Lazy import: torch only on use."""

    def __init__(self, weights: str | None = None, conf: float = 0.25, imgsz: int = 960, device: str | None = None):
        from bree.cli import _weights
        from bree.edge.ort import load_yolo
        from bree.hw import detect_hardware
        hw = detect_hardware()
        self.model = load_yolo(weights or _weights(hw.detect_model), "detect", imgsz)
        self.conf, self.imgsz, self.device = conf, imgsz, device or hw.device

    def __call__(self, img: np.ndarray) -> list[Det]:
        h, w = img.shape[:2]
        if w > DETECT_WIDTH:
            img = cv2.resize(img, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)), interpolation=cv2.INTER_AREA)
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        hh, ww = img.shape[:2]
        out = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            out.append(Det(r.names[k], float(c), (b[0] / ww, b[1] / hh, b[2] / ww, b[3] / hh)))
        return out


def image_stats(frames: list[np.ndarray]) -> dict:
    """Brightness, colour, sharpness and region statistics averaged over the sampled frames."""
    rows = []
    for img in frames:
        h, w = img.shape[:2]
        s = cv2.resize(img, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        top = hsv[: int(hsv.shape[0] * 0.35)]
        hue, sat, val = top[..., 0].astype(int), top[..., 1].astype(int), top[..., 2].astype(int)
        sky = ((hue >= 95) & (hue <= 130) & (sat >= 25) & (val >= 140)).mean()
        green = ((hsv[..., 0] >= 35) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 50)).mean()
        edges = cv2.Canny(gray, 80, 160)
        rows.append({
            "brightness": gray.mean() / 255, "contrast": gray.std() / 255, "saturation": hsv[..., 1].mean() / 255,
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "edge_density": float((edges > 0).mean()),
            "dark_frac": float((gray < 30).mean()), "bright_frac": float((gray > 235).mean()),
            "sky_frac": float(sky), "green_frac": float(green),
        })
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def _area(b) -> float:
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def detection_stats(per_frame: list[list[Det]]) -> dict:
    n = max(len(per_frame), 1)
    out: dict = {"n_frames": len(per_frame)}

    def group(names, key, min_conf=0.25):
        sel = [[d for d in f if d.name in names and d.conf >= min_conf] for f in per_frame]
        out[f"{key}_frac"] = sum(1 for f in sel if f) / n                     # share of frames with at least one
        out[f"{key}_mean_count"] = sum(len(f) for f in sel) / n
        boxes = [d.box for f in sel for d in f]
        out[f"{key}_mean_area"] = float(np.mean([_area(b) for b in boxes])) if boxes else 0.0
        out[f"{key}_mean_cy"] = float(np.mean([(b[1] + b[3]) / 2 for b in boxes])) if boxes else None
        return sel
    persons = group(("person",), "person", 0.4)
    group(VEHICLES, "vehicle")
    group(POS_GEAR, "posgear")
    group(DRINKS, "drinks")
    group(FRIDGE, "fridge")
    group(SHELF_GOODS, "goods")
    pb = [d.box for f in persons for d in f]
    out["person_mean_height"] = float(np.mean([b[3] - b[1] for b in pb])) if pb else 0.0
    out["person_edge_frac"] = float(np.mean([b[3] > 0.97 or b[0] < 0.02 or b[2] > 0.98 or b[1] < 0.02 for b in pb])) if pb else 0.0
    out["person_feet_cy"] = float(np.mean([b[3] for b in pb])) if pb else None
    # persistence: is one object type fixed in the same place across frames (screen, terminal, bottle on a shelf)?
    for key, names in (("posgear", POS_GEAR), ("drinks", DRINKS)):
        out[f"{key}_static"] = static_fraction(per_frame, names)
    return out


def static_fraction(per_frame: list[list[Det]], names, iou_min: float = 0.5) -> float:
    """Share of detections of `names` that reappear at the same place (IoU >= iou_min) in at least half of the
    other frames: fixed furniture and stock rather than things that move through."""
    items = [[d.box for d in f if d.name in names and d.conf >= 0.25] for f in per_frame]
    total = sum(len(f) for f in items)
    if total == 0 or len(items) < 3:
        return 0.0
    static = 0
    for i, f in enumerate(items):
        for b in f:
            hits = sum(any(_iou(b, c) >= iou_min for c in g) for j, g in enumerate(items) if j != i)
            if hits >= (len(items) - 1) / 2:
                static += 1
    return static / total


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = _area(a) + _area(b) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class ChannelStats:
    channel: int
    image: dict = field(default_factory=dict)
    det: dict = field(default_factory=dict)
    size: tuple[int, int] = (0, 0)
    n_frames: int = 0
    detections: list[list[Det]] = field(default_factory=list, repr=False)   # per sampled frame (for zone drafts)

    def flat(self) -> dict:
        return {**self.image, **self.det}


def compute_stats(sample, scene) -> ChannelStats:
    """`scene`: callable frame -> list[Det] (YoloScene, or a fake in tests)."""
    st = ChannelStats(sample.channel, size=sample.size, n_frames=len(sample.frames))
    if not sample.frames:
        return st
    st.image = image_stats(sample.frames)
    st.detections = [scene(f) for f in sample.frames]
    st.det = detection_stats(st.detections)
    return st
