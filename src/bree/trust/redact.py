"""Make a head-pixelated copy of a video or image: for evidence, sharing a clip with a reviewer, or labelling.

Persons come from the same detector + pose the pipeline uses (bree.detect.yolo), then bree.trust.privacy.Guard adds
the face-detector layer and the hold. Output is H.264/mp4v with the original frame size, no overlay, no audio.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from bree.events.observations import PersonObs
from bree.trust.privacy import Guard


def make_person_backend(imgsz: int | None = None, runtime: str = "pytorch"):
    """People + pose only (no products), picked for this machine like `bree run` does."""
    from bree.cli import _weights
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    hw = detect_hardware()
    return YoloBackend(_weights(hw.pose_model), _weights(hw.detect_model), {}, device=hw.device,
                       imgsz=imgsz or hw.imgsz, products=False, runtime=runtime, person_conf=0.25)


def persons_of(backend, img: np.ndarray) -> list[PersonObs]:
    people, _ = backend(img)
    kp = people.keypoints
    return [PersonObs(i, tuple(map(float, b)), float(c), None if kp is None else kp[i])
            for i, (b, c) in enumerate(zip(people.boxes, people.confs))]


def redact_image(img: np.ndarray, backend=None, guard: Guard | None = None) -> np.ndarray:
    """One still, pixelated: face detector on this very frame at full width, person detector if a backend is given."""
    from bree.trust.privacy import still_guard
    guard = guard or still_guard()
    return guard.protect(img, persons_of(backend, img) if backend else None, inplace=False)


def redact_video(src: str | Path, dst: str | Path, backend=None, guard: Guard | None = None, stride: int = 1,
                 max_frames: int | None = None, person_every: int = 1) -> dict:
    """Write dst with every frame pixelated. stride>1 drops frames (the output fps is divided to keep real time).
    person_every>1 runs the person detector on every Nth frame and reuses its boxes for the frames between."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    guard = guard or Guard(face_every=2, hold=8)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))
    n_in = n_out = 0
    persons: list[PersonObs] = []
    t0 = time.time()
    while max_frames is None or n_out < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        n_in += 1
        if (n_in - 1) % stride:
            continue
        if backend is not None and n_out % person_every == 0:
            persons = persons_of(backend, fr)
        vw.write(guard.protect(fr, persons))
        n_out += 1
    cap.release()
    vw.release()
    return {"src": str(src), "dst": str(dst), "frames_in": n_in, "frames_out": n_out, "seconds": round(time.time() - t0, 1),
            "fps_processed": round(n_out / max(time.time() - t0, 1e-6), 1), **guard.stats}
