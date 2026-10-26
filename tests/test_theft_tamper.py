"""Tamper monitor on synthetic scenes: covered, dark, blurred, moved, frozen, and no false alarm for people or lighting."""
from __future__ import annotations

import cv2
import numpy as np

from bree.theft.health import frame_features
from bree.theft.tamper import TamperConfig, TamperMonitor
from bree.theft.tamper_sim import tamper_frame


def scene(seed=0, h=180, w=320):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(60, 190, (h // 6, w // 6)).astype(np.uint8), (0, 0), 1.0)    # floor / wall texture
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):                      # a busy store: random rectangles = shelves, labels, signs
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 30))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(10, 40)), y + int(rng.integers(8, 30))), int(rng.integers(30, 230)), -1)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def run(frames_fn, seconds, fps=10, cfg=None, boxes_fn=None):
    mon, prev, ev = TamperMonitor(cfg or TamperConfig()), None, []
    rng = np.random.default_rng(1)
    for k in range(int(seconds * fps)):
        t = k / fps
        g = frames_fn(t)
        g = np.clip(g.astype(np.float32) + rng.normal(0, 1.5, g.shape), 0, 255).astype(np.uint8)   # sensor noise
        f, th = frame_features(g, prev)
        prev = th
        ev += mon.update(t, f, th, boxes_fn(t) if boxes_fn else None)
    return ev


def kinds(events):
    return [(e.kind, e.phase) for e in events]


def test_quiet_scene_has_no_alarms():
    base = scene()
    assert run(lambda t: base, 120) == []


def test_person_walking_through_is_not_tamper():
    base = scene()

    def frame(t):
        g = base.copy()
        x = int(40 + 5 * t)
        cv2.rectangle(g, (x, 40), (x + 50, 170), 20, -1)         # a dark person-sized blob
        return g
    boxes = lambda t: [((40 + 5 * t) / 320, 40 / 180, (90 + 5 * t) / 320, 1.0)]
    assert run(frame, 50, boxes_fn=boxes) == []


def test_lighting_change_is_not_tamper():
    base = scene()
    ev = run(lambda t: (base * (1.0 if t < 60 else 0.6)).astype(np.uint8), 120)
    assert ev == []


def test_lens_covered_flat_and_dark_and_cleared():
    base, rng = scene(), np.random.default_rng(0)
    ev = run(lambda t: tamper_frame(base, "cover_flat", None, rng) if 40 <= t < 80 else base, 120)
    assert [(k, p) for k, p in kinds(ev)] == [("covered", "start"), ("covered", "end")]
    start = next(e for e in ev if e.phase == "start")
    assert 40 <= start.t_start <= 42 and start.t - start.t_start >= 8 - 0.2
    ev = run(lambda t: tamper_frame(base, "blackout", None, rng) if t >= 40 else base, 80)
    assert ("dark", "start") in kinds(ev)


def test_blur_and_moved_and_rotated():
    base, rng = scene(), np.random.default_rng(0)
    assert ("blurred", "start") in kinds(run(lambda t: tamper_frame(base, "blur", None, rng) if t >= 40 else base, 100))
    for k in ("shift", "rotate", "cover_half"):
        ev = run(lambda t, k=k: tamper_frame(base, k, None, rng) if t >= 40 else base, 120)
        assert ("moved", "start") in kinds(ev), k


def test_frozen_stream():
    base = scene()
    held = base.copy()

    def frame(t):
        return base if t < 40 else held
    mon, prev, ev = TamperMonitor(), None, []
    for k in range(800):
        t = k / 10
        g = base + (np.random.default_rng(k).integers(0, 3, base.shape).astype(np.uint8) if t < 40 else 0)
        f, th = frame_features(g, prev)
        prev = th
        ev += mon.update(t, f, th)
    assert ("frozen", "start") in kinds(ev)


def test_dark_is_expected_at_night_when_configured():
    base, rng = scene(), np.random.default_rng(0)
    cfg = TamperConfig(expect_dark=((0.0, 86400.0),))
    mon = TamperMonitor(cfg, clock=lambda t: t)
    prev, ev = None, []
    for k in range(800):
        t = k / 10
        g = tamper_frame(base, "blackout", None, rng) if t >= 40 else base
        f, th = frame_features(g, prev)
        prev = th
        ev += mon.update(t, f, th)
    assert not [e for e in ev if e.kind == "dark"]
