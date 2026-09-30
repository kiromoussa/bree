"""Multi-camera handoff: one shopper walks from camera A's view into camera B's view."""
from __future__ import annotations

import numpy as np
import pytest

from bree.events.observations import FrameObs, PersonObs
from bree.events.types import Event, EventType as E
from bree.track.multicam import MultiCamIdentity, homography, to_floor


def test_homography_recovers_mapping():
    img = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
    floor = [(0, 0), (8, 0), (8, 4.5), (0, 4.5)]
    H = homography(img, floor)
    assert to_floor(H, 640, 360) == pytest.approx((4.0, 2.25), abs=1e-6)


def _cams():
    # Camera A sees floor x in [0, 6] m, camera B sees x in [4, 10] m (2 m overlap), both 1280 px wide.
    img = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
    HA = homography(img, [(0, 0), (6, 0), (6, 4), (0, 4)])
    HB = homography(img, [(4, 0), (10, 0), (10, 4), (4, 4)])
    return MultiCamIdentity({"A": HA, "B": HB}, foot_point="bottom")


def _person(tid, floor_x, cam_x0, y_px=600):
    px = (floor_x - cam_x0) / 6 * 1280
    return PersonObs(tid, (px - 20, y_px - 150, px + 20, y_px), 0.9)


def test_handoff_keeps_one_global_id():
    mc = _cams()
    t, ids = 0.0, set()
    for x in np.arange(1.0, 9.0, 0.1):          # walk left to right at 1.5 m/s
        t += 0.1 / 1.5
        if x <= 6:
            mc.update("A", FrameObs(0, t, [_person(7, x, 0)]))
        if x >= 4:
            mc.update("B", FrameObs(0, t, [_person(3, x, 4)]))
    assert mc.local_to_global[("A", 7)] == mc.local_to_global[("B", 3)]
    assert len(mc.people) == 1 and len(mc.handoffs) == 1


def test_two_people_stay_separate():
    mc = _cams()
    for k, x in enumerate(np.arange(4.2, 5.8, 0.1)):
        t = k * 0.1
        mc.update("A", FrameObs(0, t, [_person(1, x, 0), _person(2, x, 0, y_px=200)]))
        mc.update("B", FrameObs(0, t, [_person(11, x, 4), _person(12, x, 4, y_px=200)]))
    assert mc.local_to_global[("A", 1)] == mc.local_to_global[("B", 11)]
    assert mc.local_to_global[("A", 2)] == mc.local_to_global[("B", 12)]
    assert mc.local_to_global[("A", 1)] != mc.local_to_global[("A", 2)]


def test_remap_drops_duplicate_enter_and_rewrites_ids():
    mc = _cams()
    mc.update("A", FrameObs(0, 0.0, [_person(7, 5.0, 0)]))
    mc.update("B", FrameObs(0, 0.1, [_person(3, 5.05, 4)]))
    gid = mc.local_to_global[("A", 7)]
    evs = mc.remap("B", [Event(E.ENTER, 0.1, 3), Event(E.PICK, 0.5, 3, item="soda_bottle", zone="cooler_bank")])
    assert [e.type for e in evs] == [E.PICK] and evs[0].person_id == gid and evs[0].meta["camera"] == "B"
