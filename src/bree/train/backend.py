"""Optional product backend: the SKU detector trained on SIMULATED store frames (bree.train.train).

People and pose come from the normal YOLO backend (COCO weights). Products come from the SKU detector,
run on 640 px tiles at the camera's native resolution, because a 4MP frame squeezed to 640 px turns a
25 px can into a 6 px one. In the pipeline only tiles around people are run (`roi="people"`): the event
engine uses products in or near a hand, not the stock on the shelf.

SIM-TRAINED. The classes are the simulator's invented SKUs; on real footage this detects nothing useful
until it is fine-tuned on real labelled frames (scripts/train/finetune_real.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bree.detect.base import Detections
from bree.train.dataset import OVERLAP, TILE, tile_origins

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS = ROOT / "data" / "synth" / "weights" / "sim_sku.pt"


def sku_weights(path: str | Path | None = None) -> Path:
    """`path` or BREE_SKU_WEIGHTS: a file, or the name of a version in data/synth/weights (sim_sku = the first
    detector, items only; sim_sku_hands_v2 = items + the hand class, see README.md)."""
    p = Path(path or os.environ.get("BREE_SKU_WEIGHTS") or DEFAULT_WEIGHTS)
    if not p.exists() and (DEFAULT_WEIGHTS.parent / f"{p.name}.pt").exists():
        p = DEFAULT_WEIGHTS.parent / f"{p.name}.pt"
    if not p.exists():
        raise FileNotFoundError(f"SKU detector weights not found: {p}. Train them (make sku-train) or set BREE_SKU_WEIGHTS.")
    return p


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])     # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def merge_tiles(boxes: np.ndarray, confs: np.ndarray, cls: np.ndarray, cut: np.ndarray, iou: float = 0.5, inside: float = 0.7):
    """One list from overlapping tiles: best-first, drop a box that overlaps a kept one (IoU, any class: one
    item has one SKU), or that was cut by a tile edge and lies mostly inside a kept box of the same class.
    ponytail: an item larger than the tile overlap that no tile holds whole can stay as two partial boxes;
    add a downscaled full-frame pass if close-up items matter."""
    order = np.argsort(-confs)
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in order:
        if keep:
            k = np.array(keep)
            if (_iou_matrix(boxes[i:i + 1], boxes[k])[0] >= iou).any():
                continue
            if cut[i]:
                ix = (np.minimum(boxes[i, 2], boxes[k, 2]) - np.maximum(boxes[i, 0], boxes[k, 0])).clip(0) * \
                     (np.minimum(boxes[i, 3], boxes[k, 3]) - np.maximum(boxes[i, 1], boxes[k, 1])).clip(0)
                if ((ix >= inside * area[i]) & (cls[k] == cls[i])).any():
                    continue
        keep.append(int(i))
    keep = np.array(keep, int)
    return boxes[keep], confs[keep], cls[keep]


class SkuDetector:
    """Tiled SKU detection on one frame. Returns (boxes xyxy in frame pixels, confs, class ids), items only.
    Weights with a "hand" class (sim_sku_hands_v2): `detect` also returns the hand boxes; no person box is needed."""

    def __init__(self, weights: str | Path | None = None, device: str = "cpu", conf: float = 0.25, batch: int = 16,
                 tile: int = TILE, overlap: int = OVERLAP):
        from ultralytics import YOLO
        self.model = YOLO(str(sku_weights(weights)))
        self.names = self.model.names
        self.hand_cls = next((i for i, n in self.names.items() if n == "hand"), None)
        self.device, self.conf, self.batch, self.tile, self.overlap = device, conf, batch, tile, overlap

    def tiles(self, h: int, w: int, rois: np.ndarray | None = None) -> list[tuple[int, int]]:
        out = [(x, y) for y in tile_origins(h, self.tile, self.overlap) for x in tile_origins(w, self.tile, self.overlap)]
        if rois is None:
            return out
        return [(x, y) for x, y in out if any(x < r[2] and x + self.tile > r[0] and y < r[3] and y + self.tile > r[1] for r in rois)]

    def __call__(self, image: np.ndarray, rois: np.ndarray | None = None):
        return self.detect(image, rois)[0]

    def detect(self, image: np.ndarray, rois: np.ndarray | None = None, second_look: int = 0):
        """((item boxes, confs, class ids), (hand boxes, confs)). rois: only tiles touching these boxes are run and only
        detections centred inside one are kept (None = whole frame). Hands are empty for weights without the class.
        second_look N: one more tile centred on each of the N most confident hands, so an item in a hand that the grid
        cut in two is seen whole once."""
        h, w = image.shape[:2]
        raw = self._run(image, self.tiles(h, w, rois))
        out = self._merge(raw, rois)
        if second_look and len(out[1][0]):
            hb = out[1][0][np.argsort(-out[1][1])[:second_look]]
            at = [(int(np.clip((b[0] + b[2]) / 2 - self.tile / 2, 0, max(w - self.tile, 0))), int(np.clip((b[1] + b[3]) / 2 - self.tile / 2, 0, max(h - self.tile, 0)))) for b in hb]
            out = self._merge([np.concatenate(pair) for pair in zip(raw, self._run(image, at))], rois)
        return out

    def _run(self, image: np.ndarray, origins: list[tuple[int, int]]):
        """Raw boxes of these tiles in frame pixels: boxes, confs, class ids, cut by a tile edge."""
        h, w = image.shape[:2]
        B, C, K, CUT = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.float32)], [np.zeros(0, int)], [np.zeros(0, bool)]
        for i in range(0, len(origins), self.batch):
            chunk = origins[i:i + self.batch]
            res = self.model.predict([image[y:y + self.tile, x:x + self.tile] for x, y in chunk], imgsz=self.tile, conf=self.conf,
                                     device=self.device, verbose=False, max_det=300)
            for (x, y), r in zip(chunk, res):
                if r.boxes is None or not len(r.boxes):
                    continue
                b = r.boxes.xyxy.cpu().numpy()
                th, tw = r.orig_shape
                # touching a tile edge that is not the frame edge: the item continues in the next tile
                cut = ((b[:, 0] < 2) & (x > 0)) | ((b[:, 1] < 2) & (y > 0)) | ((b[:, 2] > tw - 2) & (x + tw < w)) | ((b[:, 3] > th - 2) & (y + th < h))
                small = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) < self.overlap     # a small cut box is whole in the neighbour tile
                ok = ~(cut & small)
                B.append(b[ok] + np.array([x, y, x, y], np.float32))
                C.append(r.boxes.conf.cpu().numpy()[ok])
                K.append(r.boxes.cls.cpu().numpy().astype(int)[ok])
                CUT.append(cut[ok])
        return np.concatenate(B), np.concatenate(C), np.concatenate(K), np.concatenate(CUT)

    def _merge(self, raw, rois: np.ndarray | None):
        B, C, K, CUT = raw
        none = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))
        out = []
        for sel in (K != self.hand_cls, K == self.hand_cls):     # a hand overlaps the item it holds: merged apart
            if not sel.any():
                out.append(none)
                continue
            boxes, confs, cls = merge_tiles(B[sel], C[sel], K[sel], CUT[sel])
            if rois is not None and len(boxes):
                cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
                near = np.zeros(len(boxes), bool)
                for r in rois:
                    near |= (cx >= r[0]) & (cx <= r[2]) & (cy >= r[1]) & (cy <= r[3])
                boxes, confs, cls = boxes[near], confs[near], cls[near]
            out.append((boxes, confs, cls))
        return out[0], out[1][:2]


class SimSkuBackend:
    """PerceptionBackend: people + pose from `base` (YoloBackend with products off), products from SkuDetector."""
    name = "yolo+sim_sku"

    def __init__(self, base, sku: SkuDetector, class_map: dict[str, str] | None = None, roi: str = "people", margin: float = 0.25):
        self.base, self.sku, self.roi, self.margin = base, sku, roi, margin
        self.last_hands = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        self.class_map = class_map or {}      # detector class (SKU id) -> ledger category; empty = keep the SKU id

    @property
    def last_bags(self):
        return getattr(self.base, "last_bags", None)

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]:
        persons, _ = self.base(image)
        rois = None
        if self.roi == "people":
            if not len(persons):
                return persons, Detections.empty()
            b = persons.boxes
            mx, my = (b[:, 2] - b[:, 0]) * self.margin, (b[:, 3] - b[:, 1]) * self.margin
            rois = np.stack([b[:, 0] - mx, b[:, 1] - my, b[:, 2] + mx, b[:, 3] + my], 1)
        (boxes, confs, cls), self.last_hands = self.sku.detect(image, rois)     # last_hands: (boxes, confs), empty for weights without the class
        names = [self.sku.names[int(c)] for c in cls]
        ok = np.array([not self.class_map or n in self.class_map for n in names], bool)
        return persons, Detections(boxes[ok], confs[ok], [self.class_map.get(n, n) for n, k in zip(names, ok) if k])


def make_sim_backend(store, imgsz: int | None = None, runtime: str = "pytorch", weights: str | Path | None = None,
                     roi: str | None = None, conf: float = 0.25) -> SimSkuBackend:
    """Same construction as bree.cli.make_backend("yolo", ...), with the product half swapped.
    roi None: BREE_SKU_ROI (people | full), default people. full = every tile of the frame, no person box needed."""
    roi = roi or os.environ.get("BREE_SKU_ROI", "people")
    from bree.cli import _weights
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    hw = detect_hardware()
    # BREE_PERSON_CONF: person confidence threshold (YoloBackend default 0.3). The simulator's figures seen close up
    # from a rail camera score around that threshold with COCO weights, so sim runs may want to lower it.
    base = YoloBackend(_weights(hw.pose_model), _weights(hw.detect_model), {}, device=hw.device, imgsz=imgsz or hw.imgsz,
                       person_conf=float(os.environ.get("BREE_PERSON_CONF", 0.3)), products=False, runtime=runtime)
    return SimSkuBackend(base, SkuDetector(weights, device=hw.device, conf=conf), store.product_classes, roi=roi)
