"""Tamper SIMULATED on REAL frames: how fast does bree.theft.tamper notice, and does it stay quiet otherwise?

We have no real tamper clip (the one camera with an obscured lens is not in the recorded channels). So: decode a real
segment, from `t0` on replace each frame by a tampered version of itself (cover, black-out, blur, shift, rotate, white-out,
frozen picture, and a dimmed-lights negative that must NOT alarm), and run the monitor on both. The unmodified segment
is the false-alarm check. Anything measured here is SIMULATED tampering; real tampering will differ (a hand
is not a flat grey rectangle).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bree.theft.cache import load_frames
from bree.theft.health import frame_features, grey_small
from bree.theft.tamper import KINDS, TamperConfig, TamperEvent, TamperMonitor

TAMPERS = ("cover_flat", "blackout", "cover_half", "blur", "shift", "rotate", "whiteout", "freeze", "dim_lights", "none")
SHOULD_ALARM = {"cover_flat": "covered", "blackout": "dark", "cover_half": "moved", "blur": "blurred", "shift": "moved",
                "rotate": "moved", "whiteout": "covered", "freeze": "frozen", "dim_lights": None, "none": None}


def tamper_frame(grey: np.ndarray, kind: str, held: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    h, w = grey.shape
    noise = rng.normal(0, 2.0, grey.shape)
    if kind == "cover_flat":
        return np.clip(60 + noise, 0, 255).astype(np.uint8)
    if kind == "blackout":
        return np.clip(4 + noise, 0, 255).astype(np.uint8)
    if kind == "cover_half":
        out = grey.copy()
        out[:, : int(w * 0.55)] = np.clip(35 + noise[:, : int(w * 0.55)], 0, 255).astype(np.uint8)
        return out
    if kind == "blur":
        return cv2.GaussianBlur(grey, (0, 0), 7.0)
    if kind == "shift":
        return cv2.warpAffine(grey, np.float32([[1, 0, -0.12 * w], [0, 1, 0.04 * h]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "rotate":
        return cv2.warpAffine(grey, cv2.getRotationMatrix2D((w / 2, h / 2), 14, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if kind == "whiteout":
        return np.clip(228 + noise, 0, 255).astype(np.uint8)
    if kind == "freeze":
        return held if held is not None else grey
    if kind == "dim_lights":
        return np.clip(grey.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    return grey


def load_grey(video: str | Path, cache_dir: str | Path, stride: int = 3, max_t: float | None = None):
    """[(t, grey_small frame, normalised person boxes)] for the cached frames of one segment (decode once, reuse for every
    tamper kind). Frames are decoded at `stride` exactly like the cache, so the cached person boxes line up."""
    cap = cv2.VideoCapture(str(video))
    frames = load_frames(cache_dir)
    out, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, img = cap.retrieve()
            obs = next(frames, None)
            if not ok or obs is None or (max_t is not None and obs.t > max_t):
                break
            h, w = img.shape[:2]
            out.append((obs.t, grey_small(img), [(p.bbox[0] / w, p.bbox[1] / h, p.bbox[2] / w, p.bbox[3] / h) for p in obs.persons]))
        i += 1
    cap.release()
    return out


def run_tamper(frames, kind: str, t0: float, cfg: TamperConfig | None = None, seed: int = 0) -> list[TamperEvent]:
    """Events from one run over `load_grey` frames: `kind` applied from `t0` seconds on (kind "none" = untouched)."""
    rng = np.random.default_rng(seed)
    mon = TamperMonitor(cfg)
    held, prev, events, last_t = None, None, [], 0.0
    for t, g, boxes in frames:
        if t >= t0 and held is None:
            held = g.copy()
        g2 = tamper_frame(g, kind, held, rng) if t >= t0 else g
        f, th = frame_features(g2, prev)
        prev = th
        events += mon.update(t, f, th, boxes)
        last_t = t
    return events + mon.flush(last_t)


def latency(events: list[TamperEvent], kind_expected: str | None, t0: float) -> dict:
    starts = [e for e in events if e.phase == "start"]
    hit = [e for e in starts if e.kind == kind_expected and e.t_start >= t0 - 1] if kind_expected else []
    return {"alarms": [(e.kind, round(e.t_start, 1)) for e in starts],
            "detected": bool(hit), "delay_s": round(min(e.t for e in hit) - t0, 1) if hit else None,
            "other_kinds": sorted({e.kind for e in starts if e.kind != kind_expected})}
