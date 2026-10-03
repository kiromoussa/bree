"""YOLO perception backend.

One detector pass finds people AND product-like objects; then a pose model runs
on each person crop (top-down pose, see bree/pose/yolo_pose.py).
"""
from __future__ import annotations

import numpy as np

from bree.detect.base import Detections
from bree.pose.yolo_pose import YoloPose


class YoloBackend:
    name = "yolo"

    def __init__(self, pose_weights: str, detect_weights: str, class_map: dict[str, str],
                 device: str = "cpu", imgsz: int = 640, pose_crop: int = 160,
                 person_conf: float = 0.3, product_conf: float = 0.25, products: bool = True,
                 runtime: str = "pytorch", providers: list | None = None, dedupe_inside: float = 0.85):
        from bree.edge.ort import load_yolo
        self.det = load_yolo(detect_weights, "detect", imgsz, runtime, providers)
        self.dedupe_inside = dedupe_inside   # drop a person box this much inside a larger one (0 = off)
        self.pose = YoloPose(pose_weights, device, crop_size=pose_crop, runtime=runtime, providers=providers)
        if runtime == "onnx":
            device = "cpu"   # Ultralytics pre/post-processing; ONNX Runtime picks the accelerator (bree.edge.ort)
        self.class_map = class_map if products else {}   # detector class name -> item category
        names = self.det.names
        self.person_id = next(i for i, n in names.items() if n == "person")
        self.product_ids = [i for i, n in names.items() if n in self.class_map]
        # Carried objects: a re-ID cue only (bree/track/reid.py), never products for the event engine.
        self.bag_ids = [i for i, n in names.items() if n in ("backpack", "handbag", "suitcase")
                        and n not in self.class_map]
        self.last_bags = None
        self.device, self.imgsz = device, imgsz
        self.person_conf, self.product_conf = person_conf, product_conf

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]:
        r = self.det.predict(image, imgsz=self.imgsz, conf=min(self.person_conf, self.product_conf),
                             device=self.device, classes=[self.person_id] + self.product_ids + self.bag_ids,
                             verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)

        is_person = (cls == self.person_id) & (confs >= self.person_conf)
        is_bag = np.isin(cls, self.bag_ids)
        self.last_bags = boxes[is_bag & (confs >= self.product_conf)]
        is_product = (cls != self.person_id) & ~is_bag & (confs >= self.product_conf)
        if self.dedupe_inside:
            is_person &= ~contained(boxes, is_person, self.dedupe_inside)
        pb = boxes[is_person]
        persons = Detections(pb, confs[is_person], ["person"] * len(pb), self.pose(image, pb))
        products = Detections(boxes[is_product], confs[is_product],
                              [self.class_map[r.names[c]] for c in cls[is_product]])
        return persons, products


def contained(boxes: np.ndarray, mask: np.ndarray, frac: float) -> np.ndarray:
    """True for masked boxes that lie >= frac inside a larger masked box: duplicate detections of one person
    (a partial-body box inside the full one), which would otherwise start a second track."""
    out = np.zeros(len(boxes), bool)
    idx = np.flatnonzero(mask)
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    for i in idx:
        for j in idx:
            if j != i and area[j] > area[i]:
                ix = (min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])).clip(0) * \
                     (min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])).clip(0)
                if ix >= frac * max(area[i], 1.0):
                    out[i] = True
                    break
    return out
