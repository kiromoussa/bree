"""Multi-object tracking with ByteTrack (Ultralytics implementation).

We call Ultralytics' BYTETracker directly on plain detection arrays, so the same
tracker serves the YOLO backend and the toy backend. Persons and products get
separate tracker instances (separate ID spaces).
"""
from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from bree.detect.base import Detections
from bree.events.observations import PersonObs, ProductObs


def _bytetrack(fps: float, buffer_s: float) -> BYTETracker:
    cfg = dict(YAML.load(check_yaml("bytetrack.yaml")))
    cfg["track_buffer"] = max(1, int(round(buffer_s * fps)))  # frames a lost track survives
    return BYTETracker(IterableSimpleNamespace(**cfg))


class Tracker:
    def __init__(self, fps: float, person_buffer_s: float = 2.0, product_buffer_s: float = 0.5):
        self.persons = _bytetrack(fps, person_buffer_s)
        self.products = _bytetrack(fps, product_buffer_s)
        self.product_classes: dict[str, int] = {}

    def _run(self, tracker: BYTETracker, det: Detections, shape, cls_ids: np.ndarray) -> np.ndarray:
        if len(det) == 0:
            arr = np.zeros((0, 6), np.float32)
        else:
            arr = np.concatenate([det.boxes, det.confs[:, None], cls_ids[:, None]], axis=1).astype(np.float32)
        return tracker.update(Boxes(arr, shape))   # rows: x1 y1 x2 y2 track_id score cls det_idx

    def update(self, persons: Detections, products: Detections, shape) -> tuple[list[PersonObs], list[ProductObs]]:
        out_p = []
        for row in self._run(self.persons, persons, shape, np.zeros(len(persons))):
            idx = int(row[7])
            kp = persons.keypoints[idx] if persons.keypoints is not None else None
            out_p.append(PersonObs(int(row[4]), tuple(map(float, persons.boxes[idx])), float(row[5]), kp))
        ids = np.array([self.product_classes.setdefault(l, len(self.product_classes)) for l in products.labels])
        out_i = []
        for row in self._run(self.products, products, shape, ids):
            idx = int(row[7])
            out_i.append(ProductObs(int(row[4]), tuple(map(float, products.boxes[idx])),
                                    products.labels[idx], float(row[5])))
        return out_p, out_i
