"""Bulk-grab / sweep detector on synthetic wrist tracks."""
from __future__ import annotations

import numpy as np

from bree.events.observations import FrameObs, PersonObs
from bree.events.zones import StoreConfig, Zone
from bree.events.types import Catalog
from bree.theft.sweep import SweepConfig, SweepDetector

SHELF = Zone("shelf_a", "shelf", np.array([[400, 200], [1200, 200], [1200, 600], [400, 600]], float))
STORE = StoreConfig("t", "c", (2560, 1440), 10, [SHELF], {}, Catalog({}))


def person(wrist_xy, pid=1, h=400):
    kp = np.zeros((17, 3), np.float32)
    x, y = wrist_xy
    kp[9] = (x, y, 0.9)                       # left wrist
    kp[7] = (x - 20, y + 30, 0.9)             # left elbow (hand = wrist + 0.5 forearm)
    return PersonObs(pid, (300, 100, 500, 100 + h), 0.9, kp)


def feed(det, path, fps=10.0):
    out = []
    for k, xy in enumerate(path):
        out += det.update(FrameObs(k, k / fps, [person(xy)]))
    return out


def test_fast_swipe_along_the_shelf_is_a_sweep():
    det = SweepDetector(STORE)
    path = [(450 + 70 * k, 400) for k in range(11)]       # 700 px in 1 s = 1.75 body heights, straight
    out = feed(det, path)
    assert out and out[0].kind in ("sweep", "both") and out[0].zone == "shelf_a" and out[0].score >= 0.7


def test_slow_single_reach_is_nothing():
    det = SweepDetector(STORE)
    path = [(800, 800)] * 5 + [(800, 500 - 3 * k) for k in range(30)] + [(800, 800)] * 5
    assert feed(det, path) == []


def test_repeated_reach_ins_fire_once_then_cool_down():
    det = SweepDetector(STORE)
    path = []
    for _ in range(7):
        path += [(800, 800)] * 4 + [(800, 650)] + [(800, 450)] * 2 + [(800, 650)]      # in and out of the shelf, 0.8 s each cycle (out = 200 px below it)
    out = feed(det, path)
    assert len(out) == 1 and out[0].kind in ("repeat", "both") and out[0].detail["reach_ins"] >= 5


def test_small_or_pose_less_people_are_ignored():
    det = SweepDetector(STORE)
    p = person((800, 400), h=30)
    assert det.update(FrameObs(0, 0.0, [p])) == []
    q = PersonObs(2, (300, 100, 500, 500), 0.9, None)
    assert det.update(FrameObs(1, 0.1, [q])) == [] and det.stats["with_hand"] == 0


def test_hand_jittering_on_the_zone_edge_is_one_reach_not_many():
    det = SweepDetector(STORE)
    path = [(800, 610 if k % 2 else 590) for k in range(120)]      # resting on the lower edge of the shelf, +-10 px
    assert feed(det, path) == []
