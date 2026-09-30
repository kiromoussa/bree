"""Detector output shared by every backend (YOLO, toy)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray                      # (N, 4) x1, y1, x2, y2 in pixels
    confs: np.ndarray                      # (N,)
    labels: list[str]                      # class name per box
    keypoints: np.ndarray | None = None    # (N, 17, 3) COCO keypoints (persons only)

    @staticmethod
    def empty(with_kpts: bool = False) -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [],
                          np.zeros((0, 17, 3), np.float32) if with_kpts else None)

    def __len__(self) -> int:
        return len(self.boxes)


class PerceptionBackend(Protocol):
    """Anything that turns an image into (persons with keypoints, products)."""
    name: str

    def __call__(self, image: np.ndarray) -> tuple[Detections, Detections]: ...
