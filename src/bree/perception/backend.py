"""A perception backend driven by a PerceptionProfile (same interface as bree.detect.yolo.YoloBackend).

Differences from the stock backend: sliced inference for small people, profile ignore regions / size / edge filters,
optional pose, and `make_tracker` so the pipeline builds the profile's tracker (ByteTrack overrides plus a TrackGate
that withholds a track from the event engine until it has lived `min_track_age_s`).
"""
from __future__ import annotations

import numpy as np

from bree.detect.base import Detections
from bree.detect.yolo import YoloBackend, contained
from bree.perception.profile import PerceptionProfile, fixture_mask, keep_mask
from bree.perception.sliced import sliced_predict


class ProfiledBackend(YoloBackend):
    name = "yolo_profiled"

    def __init__(self, profile: PerceptionProfile, class_map: dict[str, str] | None = None, device: str = "cpu",
                 products: bool = False, runtime: str = "pytorch"):
        from bree.perception.sweep import resolve_weights as _weights
        self.profile = profile
        super().__init__(_weights(profile.pose_weights), _weights(profile.weights), class_map or {}, device=device,
                         imgsz=profile.imgsz, pose_crop=profile.pose_crop, person_conf=profile.person_conf,
                         product_conf=profile.product_conf, products=products, runtime=runtime,
                         dedupe_inside=profile.dedupe_inside)
        self.use_pose = profile.pose
        self.adaptive = None
        self._fps = 30.0

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]:
        p = self.profile
        cls_ids = [self.person_id] + self.product_ids + self.bag_ids
        boxes, confs, cls = sliced_predict(self.det, image, p.slices, self.imgsz, min(p.person_conf, p.product_conf),
                                           cls_ids, device=self.device)
        is_person = (cls == self.person_id) & (confs >= p.person_conf)
        is_bag = np.isin(cls, self.bag_ids)
        self.last_bags = boxes[is_bag & (confs >= p.product_conf)]
        is_product = (cls != self.person_id) & ~is_bag & (confs >= p.product_conf)
        ok = np.zeros(len(boxes), bool)
        ok[is_person] = keep_mask(p, boxes[is_person], confs[is_person], image.shape[:2])
        is_person = ok
        if p.adaptive_ghost_s > 0:
            if self.adaptive is None:
                from bree.perception.adaptive import AdaptiveGhostMask
                self.adaptive = AdaptiveGhostMask(self._fps, min_s=p.adaptive_ghost_s)
            idx = np.where(is_person)[0]
            is_person[idx[~self.adaptive.update(boxes[idx], confs[idx], image.shape[:2])]] = False
        if self.dedupe_inside:
            is_person &= ~contained(boxes, is_person, self.dedupe_inside)
        is_product &= fixture_mask(p, boxes, image.shape[:2])       # a card terminal in view all day is not a pick
        pb = boxes[is_person]
        kp = self.pose(image, pb) if self.use_pose else None
        persons = Detections(pb, confs[is_person], ["person"] * len(pb), kp)
        products = Detections(boxes[is_product], confs[is_product], [self.class_map[self.det.names[c]] for c in cls[is_product]])
        return persons, products

    def make_tracker(self, fps: float):
        self._fps = fps
        from bree.perception.gate import GatedTracker
        return GatedTracker(fps, self.profile)
