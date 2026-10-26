"""Perception cache: run the detector, pose and tracker over a video file ONCE, keep only metadata.

Why: the event engine, ledger, gates and detectors in this package are cheap and get tuned a lot; the detector
is not. A cached segment is `frames.jsonl.gz` (person boxes with track ids, confidence and keypoints; product boxes;
no pixels, so no faces) plus `health.npz` (per-frame image health features and 32x18 grey thumbnails, see health.py).
`load_frames` turns the cache back into FrameObs for `replay.py`.

Frames are sampled with `stride` (every n-th decoded frame); times are source times, so every downstream rule that
works in seconds is unaffected. The tracker is told the effective fps.
"""
from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.theft.health import FEATURES, frame_features, grey_small

CACHE_VERSION = 1


@dataclass
class CacheInfo:
    video: str
    frames: int
    stride: int
    fps: float            # source fps
    duration_s: float
    wall_s: float
    width: int
    height: int
    version: int = CACHE_VERSION


def cache_paths(out_dir: str | Path) -> tuple[Path, Path, Path]:
    out = Path(out_dir)
    return out / "frames.jsonl.gz", out / "health.npz", out / "info.json"


def is_cached(out_dir: str | Path) -> bool:
    return all(p.exists() for p in cache_paths(out_dir)) and \
        json.loads(cache_paths(out_dir)[2].read_text()).get("version") == CACHE_VERSION


def cache_video(video: str | Path, out_dir: str | Path, backend, stride: int = 2, max_frames: int | None = None,
                health_only: bool = False, log=None) -> CacheInfo:
    """Decode `video`, run `backend` + tracker on every `stride`-th frame, write the cache into `out_dir`.
    `health_only` skips the detector (tamper experiments need only the health features)."""
    import cv2
    from bree.track.bytetrack import Tracker
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_p, health_p, info_p = cache_paths(out)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 0 < fps < 240 else 15.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = None if health_only else Tracker(fps / stride)
    rows, thumbs, ts, prev = [], [], [], None
    i = n = 0
    with gzip.open(frames_p, "wt") as fz:
        while max_frames is None or i < max_frames:
            if not cap.grab():
                break
            if i % stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                t = i / fps
                feats, thumb = frame_features(grey_small(img), prev)
                prev = thumb
                rows.append([feats[k] for k in FEATURES])
                thumbs.append(thumb)
                ts.append(t)
                persons, products = [], []
                if tracker is not None:
                    pd, qd = backend(img)
                    persons, products = tracker.update(pd, qd, img.shape[:2])
                fz.write(json.dumps({
                    "frame": i, "t": round(t, 3),
                    "persons": [{"id": p.track_id, "bbox": [round(v, 1) for v in p.bbox], "conf": round(p.conf, 3),
                                 "kpts": None if p.keypoints is None else np.round(p.keypoints, 1).tolist()}
                                for p in persons],
                    "products": [{"id": q.track_id, "bbox": [round(v, 1) for v in q.bbox], "cat": q.category,
                                  "conf": round(q.conf, 3)} for q in products]}) + "\n")
                n += 1
                if log and n % 500 == 0:
                    log(f"  {Path(video).name}: {n} frames, {n * stride / fps:.0f}s of video, {time.time() - t0:.0f}s wall")
            i += 1
    cap.release()
    np.savez_compressed(health_p, t=np.array(ts, np.float32), x=np.array(rows, np.float32), names=np.array(FEATURES),
                        thumbs=np.array(thumbs, np.uint8))
    info = CacheInfo(str(video), n, stride, float(fps), i / fps, round(time.time() - t0, 1), w, h)
    info_p.write_text(json.dumps(info.__dict__, indent=1))
    return info


def load_info(cache_dir: str | Path) -> CacheInfo:
    return CacheInfo(**json.loads(cache_paths(cache_dir)[2].read_text()))


def load_frames(cache_dir: str | Path) -> Iterator[FrameObs]:
    with gzip.open(cache_paths(cache_dir)[0], "rt") as f:
        for line in f:
            d = json.loads(line)
            persons = [PersonObs(p["id"], tuple(p["bbox"]), p["conf"],
                                 None if p["kpts"] is None else np.asarray(p["kpts"], np.float32))
                       for p in d["persons"]]
            products = [ProductObs(q["id"], tuple(q["bbox"]), q["cat"], q["conf"]) for q in d["products"]]
            yield FrameObs(d["frame"], d["t"], persons, products)


def load_health(cache_dir: str | Path) -> dict:
    z = np.load(cache_paths(cache_dir)[1], allow_pickle=False)
    return {"t": z["t"], "x": z["x"], "names": [str(n) for n in z["names"]], "thumbs": z["thumbs"]}
