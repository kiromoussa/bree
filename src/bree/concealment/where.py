""""Where the hand went": a small classifier on a crop around a hand box of an item camera.

Four classes, from two questions: is the hand at a shelf or counter (reaching, taking, putting back, paying) or at the
person's own body; and is an item visible in it. Trained on rendered frames of TRAIN-range seeds only
(scripts/conceal/train_where.py, SIMULATED: one store model, block-shaped hands, invented package art).
The simulator has no baskets, carts or bags, so there is no "near basket" class: see the fragment.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLASSES = ("shelf_empty", "shelf_item", "body_empty", "body_item")
SIZE, CONTEXT = 64, 3.0        # crop side in pixels; crop side in hand-box sizes
WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "conceal_where.pt"


def crop(image: np.ndarray, box, context: float = CONTEXT, size: int = SIZE, shift=(0.0, 0.0)) -> np.ndarray:
    """Square BGR crop centred on the box (plus shift, in box sizes), side = context x the larger box side; grey outside the picture."""
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    s = max(x1 - x0, y1 - y0, 8.0)
    cx, cy, h = (x0 + x1) / 2 + shift[0] * s, (y0 + y1) / 2 + shift[1] * s, context * s / 2
    M = np.array([[size / (2 * h), 0, -(cx - h) * size / (2 * h)], [0, size / (2 * h), -(cy - h) * size / (2 * h)]])
    return cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_AREA if 2 * h > size else cv2.INTER_LINEAR, borderValue=(114, 114, 114))


def net():
    import torch.nn as nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, len(CLASSES)))


class Where:
    """probs = Where()(image, hand boxes) -> (N, 4) in CLASSES order; at_shelf = probs[:, :2].sum(1); item = probs[:, 1] + probs[:, 3]."""

    def __init__(self, weights: Path | None = None, device: str = "cpu"):
        import torch
        self.torch, self.device = torch, device
        self.model = net()
        self.model.load_state_dict(torch.load(weights or WEIGHTS, map_location="cpu"))
        self.model.eval().to(device)

    def __call__(self, image: np.ndarray, boxes) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, len(CLASSES)), np.float32)
        x = np.stack([crop(image, b) for b in boxes])[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        with self.torch.no_grad():
            return self.torch.softmax(self.model(self.torch.from_numpy(x).to(self.device)), 1).cpu().numpy()
