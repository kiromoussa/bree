"""COCO-17 body keypoints with a YOLO pose model, run top-down on person crops.

Why crops: on CCTV-sized people (~60-120 px tall) the full-frame nano pose model
misses most of them (0 of 7 people on OpenCV's vtest.avi at 640 px), while the
nano detector finds them all. Running pose on each detected person's crop,
resized to ~160 px, recovers keypoints for every one of them. See DECISIONS.md.
"""
from __future__ import annotations

import numpy as np


class YoloPose:
    def __init__(self, weights: str, device: str = "cpu", crop_size: int = 160, conf: float = 0.1,
                 pad_x: float = 0.25, pad_y: float = 0.10):
        from ultralytics import YOLO
        self.model = YOLO(weights, task="pose")
        self.device, self.crop_size, self.conf = device, crop_size, conf
        self.pad_x, self.pad_y = pad_x, pad_y

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """boxes: (N, 4) person boxes -> (N, 17, 3) keypoints (x, y, conf) in image coords.
        People where pose fails get all-zero keypoints (conf 0)."""
        n = len(boxes)
        out = np.zeros((n, 17, 3), np.float32)
        if n == 0:
            return out
        H, W = image.shape[:2]
        crops, origins = [], []
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            X1, Y1 = int(max(0, x1 - self.pad_x * w)), int(max(0, y1 - self.pad_y * h))
            X2, Y2 = int(min(W, x2 + self.pad_x * w)), int(min(H, y2 + self.pad_y * h))
            crops.append(image[Y1:Y2, X1:X2])
            origins.append((X1, Y1, x1 - X1, y1 - Y1, x2 - X1, y2 - Y1))
        results = self.model.predict(crops, imgsz=self.crop_size, conf=self.conf, device=self.device,
                                     verbose=False)
        for i, (r, (X1, Y1, bx1, by1, bx2, by2)) in enumerate(zip(results, origins)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            # The crop may contain bystanders: keep the pose whose box best matches the target person.
            pb = r.boxes.xyxy.cpu().numpy()
            ix1, iy1 = np.maximum(pb[:, 0], bx1), np.maximum(pb[:, 1], by1)
            ix2, iy2 = np.minimum(pb[:, 2], bx2), np.minimum(pb[:, 3], by2)
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            union = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) + (bx2 - bx1) * (by2 - by1) - inter
            j = int(np.argmax(inter / np.maximum(union, 1e-6)))
            kp = r.keypoints.data[j].cpu().numpy().astype(np.float32)
            kp[:, 0] += X1
            kp[:, 1] += Y1
            out[i] = kp
        return out
