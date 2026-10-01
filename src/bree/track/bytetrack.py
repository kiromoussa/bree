"""Multi-object tracking.

People: ByteTrack (Ultralytics' BYTETracker, called directly on detection
arrays so the YOLO and toy backends share it).
Products: centroid tracker (bree/track/centroid.py), because IoU association
fails for small objects moving in a hand.
"""
from __future__ import annotations

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from bree.detect.base import Detections
from bree.events.observations import PersonObs, ProductObs
from bree.track.centroid import CentroidTracker


def _bytetrack(fps: float, buffer_s: float, overrides: dict | None = None) -> BYTETracker:
    cfg = dict(YAML.load(check_yaml("bytetrack.yaml")))
    cfg["track_buffer"] = max(1, int(round(buffer_s * fps)))  # frames a lost track survives
    cfg.update(overrides or {})                                 # e.g. new_track_thresh, match_thresh
    return BYTETracker(IterableSimpleNamespace(**cfg))


class Tracker:
    def __init__(self, fps: float, person_buffer_s: float = 2.0, product_buffer_s: float = 0.35,
                 bytetrack: dict | None = None):
        self.persons = _bytetrack(fps, person_buffer_s, bytetrack)
        self.products = CentroidTracker(max_missed=max(1, int(round(product_buffer_s * fps))))

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
        ids = self.products.update(products.boxes, products.labels)
        out_i = [ProductObs(tid, tuple(map(float, products.boxes[i])), products.labels[i], float(products.confs[i]))
                 for i, tid in enumerate(ids)]
        return out_p, out_i
