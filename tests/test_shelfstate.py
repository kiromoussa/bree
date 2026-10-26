"""Shelf-state monitor (bree/shelfstate): SYNTHETIC pictures with known truth, no footage."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from bree.shelfstate.evidence import pixelate_people
from bree.shelfstate.model import ShelfStateConfig, StateMonitor
from bree.shelfstate.regions import CameraRegions, Region

W, H = 640, 360


def scene(seed=0, stocked=True):
    """A shelf wall: coloured product blocks with texture on the left half, plain floor on the right."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (150, 160, 170), np.uint8)
    if stocked:
        for y in range(20, 200, 24):
            for x in range(20, 300, 20):
                col = tuple(int(v) for v in rng.integers(30, 200, 3))
                cv2.rectangle(img, (x, y), (x + 16, y + 20), col, -1)
                cv2.line(img, (x, y + 6), (x + 16, y + 6), (255, 255, 255), 1)
    return img


REG = CameraRegions("t", [Region("floor", "floor", [[0.55, 0.3], [1, 0.3], [1, 1], [0.55, 1]]),
                          Region("shelf", "shelf", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.6], [0.0, 0.6]])])
CFG = ShelfStateConfig(settle_s=8, clear_s=4, still_s=2, cooldown_s=5)


def run(frames, persons=None, cfg=CFG, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    m = StateMonitor("t", REG, cfg)
    evs = []
    for i, f in enumerate(frames):
        if noise:
            f = np.clip(f.astype(np.float32) + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        p = persons(i) if persons else None
        evs += m.update(float(i), f, p)
    return evs


def test_static_scene_is_silent_even_with_noise_and_gain_steps():
    base = scene()
    frames = [base] * 30 + [np.clip(base * 1.25, 0, 255).astype(np.uint8)] * 30 + [np.clip(base * 0.8, 0, 255).astype(np.uint8)] * 30
    assert run(frames, noise=3.0) == []


def test_removed_stock_is_one_stock_down_event_at_the_right_place():
    full, part = scene(), scene()
    part[20:68, 20:120] = (150, 160, 170)       # a block of facings taken off the shelf
    evs = run([full] * 25 + [part] * 40)
    assert [e.kind for e in evs] == ["stock_down"]
    e = evs[0]
    assert e.region_id == "shelf" and 0.0 <= e.box[0] < 0.2 and e.box[2] < 0.3 and e.conf > 0.5
    assert e.t - e.t_start >= 8


def test_restock_is_reported_as_restock():
    empty, stocked = scene(), scene()
    empty[20:68, 20:120] = (150, 160, 170)
    evs = run([empty] * 25 + [stocked] * 40)
    assert evs and all(e.kind == "restock" for e in evs)


def test_change_under_a_person_waits_until_the_person_is_gone():
    full, part = scene(), scene()
    part[20:68, 20:120] = (150, 160, 170)
    box = np.array([[0.02, 0.02, 0.2, 0.25, 0.9]])
    frames = [full] * 25 + [part] * 60
    evs = run(frames, persons=lambda i: box if 20 <= i < 55 else None)
    assert len(evs) == 1
    assert evs[0].t >= 55 + CFG.clear_s - 1          # not while the person stands there


def test_short_change_is_not_an_event():
    full, part = scene(), scene()
    part[20:68, 20:120] = (0, 0, 0)
    assert run([full] * 25 + [part] * 4 + [full] * 30) == []


def test_floor_object_and_floor_spill_kinds():
    floor = scene()
    obj = floor.copy()
    cv2.rectangle(obj, (400, 250), (430, 280), (40, 40, 200), -1)       # a coloured item on the floor
    cv2.line(obj, (402, 252), (428, 278), (255, 255, 255), 2)
    evs = run([floor] * 25 + [obj] * 40)
    assert [e.kind for e in evs] == ["floor_object"]
    spill = floor.copy()
    cv2.ellipse(spill, (480, 250), (45, 25), 0, 0, 360, (90, 95, 100), -1)   # darker, featureless patch
    evs = run([floor] * 25 + [spill] * 40)
    assert [e.kind for e in evs] == ["floor_spill"]


def test_ignore_region_never_fires():
    reg = CameraRegions("t", [Region("win", "ignore", [[0, 0], [1, 0], [1, 1], [0, 1]])])
    full, part = scene(), scene()
    part[20:68, 20:120] = (0, 0, 0)
    m = StateMonitor("t", reg, CFG)
    assert [e for i, f in enumerate([full] * 25 + [part] * 40) for e in m.update(float(i), f)] == []


def test_whole_picture_light_change_is_not_events():
    base = scene()
    lit = np.clip(base.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    assert run([base] * 25 + [lit] * 40) == []


def test_pixelate_people_changes_only_the_box():
    img = scene()
    out = pixelate_people(img, np.array([[0.1, 0.1, 0.3, 0.5, 0.9]]), grow=0.0)
    assert not np.array_equal(out[60:150, 80:180], img[60:150, 80:180])
    assert np.array_equal(out[:, 300:], img[:, 300:])
