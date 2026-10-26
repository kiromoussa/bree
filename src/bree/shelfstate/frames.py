"""Frame sampling and the person gate for shelf-state monitoring.

Shelf state changes slowly (minutes), so the monitor reads the stream at about 1 frame per second at working size
(640 px wide) and never needs the full-resolution stream. `sample_frames` decodes with ffmpeg and yields
(t_seconds_from_start, small BGR frame, detector-size BGR frame); the detector-size frame is only there so a person
detector can see far-field people, the state model uses the small one. `PersonGate` turns person boxes into a cell
mask: cells a person covers (or is about to cover) are "unknown", never "changed".
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

WORK_W, WORK_H = 640, 360
DET_W, DET_H = 1280, 720


def probe_size(path: str | Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def sample_frames(path: str | Path, fps: float = 1.0, det_size: tuple[int, int] | None = (DET_W, DET_H),
                  work_size: tuple[int, int] = (WORK_W, WORK_H), start: float = 0.0, duration: float | None = None):
    """Yield (t, work_frame, det_frame or None). t is seconds from the start of the file. Frames are BGR uint8."""
    w, h = det_size or work_size
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-vf", f"fps={fps},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, size = 0, w * h * 3
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            big = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            if det_size:
                import cv2
                small = cv2.resize(big, work_size, interpolation=cv2.INTER_AREA)
                yield start + n / fps, small, big
            else:
                yield start + n / fps, big.copy(), None
            n += 1
    finally:
        p.kill()
        p.wait()


class PersonDetector:
    """Person boxes (normalised x0, y0, x1, y1) from the repo's YOLO weights. Low confidence on purpose: for a gate a
    false person only hides a patch of shelf for a few seconds, a missed one is a false shelf event."""

    def __init__(self, weights: str = "models/yolo26s.pt", conf: float = 0.2, imgsz: int = 1280, device: str | None = None):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=[0], device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5))
        h, w = frame.shape[:2]
        b = r.boxes.xyxy.cpu().numpy() / np.array([w, h, w, h])
        return np.concatenate([b, r.boxes.conf.cpu().numpy()[:, None]], 1)


def box_mask(boxes: np.ndarray, shape: tuple[int, int], grow: float = 0.15) -> np.ndarray:
    """Boolean pixel mask (h, w) of normalised boxes, each grown by `grow` of its size on every side."""
    h, w = shape
    m = np.zeros((h, w), bool)
    for x0, y0, x1, y1, *_ in boxes:
        bw, bh = x1 - x0, y1 - y0
        a, b = int(max(0, (x0 - grow * bw) * w)), int(max(0, (y0 - grow * bh) * h))
        c, d = int(min(w, np.ceil((x1 + grow * bw) * w))), int(min(h, np.ceil((y1 + grow * bh) * h)))
        m[b:d, a:c] = True
    return m
