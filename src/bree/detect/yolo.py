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
                 person_conf: float = 0.3, product_conf: float = 0.25, products: bool = True):
        from ultralytics import YOLO
        self.det = YOLO(detect_weights, task="detect")
        self.pose = YoloPose(pose_weights, device, crop_size=pose_crop)
        self.class_map = class_map if products else {}   # detector class name -> item category
        names = self.det.names
        self.person_id = next(i for i, n in names.items() if n == "person")
        self.product_ids = [i for i, n in names.items() if n in self.class_map]
        self.device, self.imgsz = device, imgsz
        self.person_conf, self.product_conf = person_conf, product_conf

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]:
        r = self.det.predict(image, imgsz=self.imgsz, conf=min(self.person_conf, self.product_conf),
                             device=self.device, classes=[self.person_id] + self.product_ids,
                             verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)

        is_person = (cls == self.person_id) & (confs >= self.person_conf)
        is_product = (cls != self.person_id) & (confs >= self.product_conf)
        pb = boxes[is_person]
        persons = Detections(pb, confs[is_person], ["person"] * len(pb), self.pose(image, pb))
        products = Detections(boxes[is_product], confs[is_product],
                              [self.class_map[r.names[c]] for c in cls[is_product]])
        return persons, products
