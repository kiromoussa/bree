"""Multi-camera handoff: one shopper walks from camera A's view into camera B's view;
and a whole store (door, cooler, register cameras) feeding one ledger."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bree.detect.toy import synth_keypoints
from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.events.types import Event, EventType as E, LineItem, Payment
from bree.events.zones import load_store_config, merge_stores
from bree.ledger import build_ledger
from bree.track.multicam import MultiCamIdentity, StoreEvents, homography, to_floor

ROOT = Path(__file__).resolve().parents[1]


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


def test_coming_back_into_a_camera_keeps_the_global_id():
    """Regression: a camera used to exclude every global id it had ever held, so a shopper who
    left the door camera's view and came back to it (to leave) became a new person."""
    mc = _cams()
    mc.update("A", FrameObs(0, 0.0, [_person(7, 5.0, 0)]))
    mc.update("B", FrameObs(0, 0.0, [_person(3, 5.0, 4)]))         # overlap: handoff A -> B
    for k, x in enumerate(np.arange(6.5, 9.5, 0.5)):                # only B sees them now
        mc.update("B", FrameObs(0, 1.0 + k, [_person(3, x, 4)]))
    for k, x in enumerate(np.arange(9.0, 4.5, -0.5)):               # walk back
        t = 8.0 + k * 0.3
        mc.update("B", FrameObs(0, t, [_person(3, x, 4)]))
        if x <= 6:
            mc.update("A", FrameObs(0, t, [_person(8, x, 0)]))      # new local track in A
    assert mc.local_to_global[("A", 8)] == mc.local_to_global[("A", 7)] == mc.local_to_global[("B", 3)]
    assert len(mc.people) == 1


# ------------------------------------------- one store, three cameras, one ledger

FPS = 15.0
# All three cameras look at parts of one floor (same pixel frame, 100 px = 1 m, for readability);
# each one has only the zones it can see and only reports people inside its view.
FLOOR = [[0, 0, 0, 0], [1280, 0, 12.8, 0], [1280, 720, 12.8, 7.2], [0, 720, 0, 7.2]]
VIEWS = {"door": lambda x, y: y > 430 and x < 800,
         "cooler": lambda x, y: y < 480,
         "register": lambda x, y: x > 700 and y > 400}
ZONES = {"door": {"door"}, "cooler": {"cooler_bank", "shelf_A", "shelf_B", "shelf_C"}, "register": {"register"}}


def _stores():
    base = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    return {cam: replace(base, camera_id=cam, floor_points=FLOOR, zones=[z for z in base.zones if z.name in names],
                         terminals={t: z for t, z in base.terminals.items() if z in names})
            for cam, names in ZONES.items()}


def _walk(a, b, step=10.0):
    n = max(1, int(math.dist(a, b) // step))
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]


def _rest(x, y):
    return [(x - 30, y - 62), (x + 30, y - 62)]


def _visit(via_register: bool) -> list:
    """Per tick: (foot x, foot y, hands, soda (x, y) or None). Door -> cooler pick -> (register) -> door."""
    ticks = [(85, 690, _rest(85, 690), None)] * 5
    ticks += [(x, y, _rest(x, y), None) for x, y in _walk((85, 690), (420, 575)) + _walk((420, 575), (420, 300))]
    fx, fy, (ix, iy) = 420, 300, (450, 95)                         # as tests/test_engine._pick_sequence
    hr = (fx + 30, fy - 62)
    ticks += [(fx, fy, _rest(fx, fy), (ix, iy))] * 5
    path = [(hr[0] + (ix - hr[0]) * u, hr[1] + (iy - hr[1]) * u) for u in np.linspace(0, 1, 8)]
    ticks += [(fx, fy, [(fx - 30, fy - 62), h], (ix, iy)) for h in path]
    ticks += [(fx, fy, [(fx - 30, fy - 62), h], (h[0], h[1] + 12)) for h in reversed(path)]
    ticks += [(fx, fy, _rest(fx, fy), (hr[0], hr[1] + 12))] * 10
    route = _walk((420, 300), (420, 575))
    if via_register:
        route += _walk((420, 575), (1100, 575)) + [(1100, 575)] * 60 + _walk((1100, 575), (1100, 640)) \
            + _walk((1100, 640), (60, 640))
    else:
        route += _walk((420, 575), (420, 640)) + _walk((420, 640), (60, 640))
    ticks += [(x, y, _rest(x, y), (x + 30, y - 50)) for x, y in route]
    return ticks + [None] * 60                                      # gone: door camera confirms the exit


def camera_frames(via_register: bool) -> dict[str, list[FrameObs]]:
    """What each camera reports: the person only inside its view (a new local track id every time
    they come back into view), the soda only on the cooler camera."""
    out = {cam: [] for cam in VIEWS}
    local = {cam: 0 for cam in VIEWS}
    was = {cam: False for cam in VIEWS}
    for f, tick in enumerate(_visit(via_register)):
        for cam, sees in VIEWS.items():
            persons, products = [], []
            if tick is not None and sees(tick[0], tick[1]):
                x, y, hands, soda = tick
                local[cam] += not was[cam]
                box = (x - 22, y - 158, x + 22, y)
                persons.append(PersonObs(10 * local[cam] + len(cam), box, 0.9, synth_keypoints(box, hands)))
                if cam == "cooler" and soda is not None:
                    products.append(ProductObs(7, (soda[0] - 7, soda[1] - 9, soda[0] + 7, soda[1] + 9),
                                               "soda_bottle", 0.85))
                was[cam] = True
            else:
                was[cam] = False
            out[cam].append(FrameObs(f, f / FPS, persons, products))
    return out


def _run_store_events(via_register: bool, payments: list[Payment]):
    stores = _stores()
    se = StoreEvents(stores)
    led = build_ledger(merge_stores(list(stores.values())))
    frames, events = camera_frames(via_register), []
    pending = sorted(payments, key=lambda p: p.t)
    for f in range(len(frames["door"])):
        t = f / FPS
        while pending and pending[0].t <= t:
            led.on_payment(pending.pop(0))
        for cam in VIEWS:
            for ev in se.update(cam, frames[cam][f]):
                events.append(ev)
                led.on_event(ev)
        led.tick(t)
    for ev in se.flush():
        events.append(ev)
        led.on_event(ev)
    led.finalize()
    return se, led, events


def test_three_cameras_one_visit_one_ledger():
    """Pick on the cooler camera, pay at the register camera, exit at the door camera: one person."""
    se, led, events = _run_store_events(True, [Payment(14.0, "pos_1", [LineItem(sku="COKE-20OZ")], txn_id="T1")])
    seen = {(e.type, e.meta.get("camera")) for e in events}
    assert {(E.PICK, "cooler"), (E.PAY, "register"), (E.EXIT, "door")} <= seen
    assert {e.person_id for e in events} == {1} and [e.type for e in events].count(E.ENTER) == 1
    assert list(led.people) == [1]
    p = led.people[1]
    assert p.reconciled and [it.category for it in p.basket] == ["soda_bottle"] and len(p.paid) == 1
    assert led.alerts == [] and led.decisions[0]["unpaid"] == []
    assert len(se.identity.handoffs) >= 4          # door -> cooler -> door -> register -> door


def test_three_cameras_walkout_is_one_flagged_visit():
    se, led, events = _run_store_events(False, [])
    assert list(led.people) == [1] and [e.type for e in events].count(E.EXIT) == 1
    assert len(led.alerts) == 1 and led.alerts[0].person_id == 1
    assert [i.category for i in led.alerts[0].unpaid_items] == ["soda_bottle"]


def test_multicam_needs_floor_points_and_consistent_zones():
    stores = _stores()
    stores["door"] = replace(stores["door"], floor_points=None)
    with pytest.raises(ValueError, match="floor_points"):
        StoreEvents(stores)
    a, b = _stores()["door"], _stores()["register"]
    b = replace(b, zones=[replace(z, name="door") for z in b.zones])   # "door" as a register zone
    with pytest.raises(ValueError, match="door"):
        merge_stores([a, b])
    single = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    assert single.floor_points is None and StoreEvents({"cam": single}).identity is None


def test_shadow_multicam_runs_one_store_ledger(tmp_path, monkeypatch):
    """`bree shadow` with `multicam:` through run_store: three camera 'streams' (no video decoding,
    no YOLO: the tracker replays this test's per-camera observations), one would-be alert."""
    import json
    import sys
    import types

    import bree.cli
    import bree.events.zones
    import bree.pipeline
    from bree.ingest.source import Frame
    from bree.shadow import ShadowLog, load_shadow_config, run_shadow

    frames, stores = camera_frames(via_register=False), _stores()
    cams = list(VIEWS)

    class FakeSource:                    # rtsp_url = camera name; the pixel (0, 0) says who/when
        def __init__(self, src, max_frames=None):
            self.cam, self.fps, self.live = src, FPS, False

        def __iter__(self):
            for f in range(len(frames[self.cam])):
                img = np.zeros((90, 160, 3), np.uint8)
                img[0, 0] = (cams.index(self.cam), f // 256, f % 256)
                yield Frame(f, f / FPS, img, 0.0)

    class FakeTracker:
        def __init__(self, fps):
            pass

        def update(self, det, _products, _shape):
            obs = frames[cams[det[0]]][det[1] * 256 + det[2]]
            return obs.persons, obs.products

    class FakeBackend:
        name = "fake"

        def __call__(self, img):
            return tuple(int(v) for v in img[0, 0]), None

    monkeypatch.setattr(bree.pipeline, "VideoSource", FakeSource)
    monkeypatch.setitem(sys.modules, "bree.track.bytetrack", types.SimpleNamespace(Tracker=FakeTracker))
    monkeypatch.setattr(bree.events.zones, "load_store_config", lambda p: stores[str(p)])
    monkeypatch.setattr(bree.cli, "make_backend", lambda *a, **k: FakeBackend())
    cfg_path = tmp_path / "shadow.yaml"
    cfg_path.write_text(json.dumps({"cameras": [{"name": f"{c}_cam", "rtsp_url": c, "store": c} for c in cams],
                                    "output_dir": str(tmp_path / "out"), "review_port": 0, "multicam": True}))
    cfg = load_shadow_config(cfg_path)
    assert cfg.multicam == {}
    log = run_shadow(cfg, once=True)
    [rec] = ShadowLog(tmp_path / "out").labelled()
    assert rec["camera"] == "store" and rec["unpaid"][0]["category"] == "soda_bottle"
    assert rec["clip"] and (tmp_path / "out" / rec["clip"]).stat().st_size > 0
    [session] = (tmp_path / "out" / "store").iterdir()
    summary = json.loads((session / "summary.json").read_text())
    assert summary["persons_tracked"] == 1 and summary["handoffs"] >= 2
    assert summary["events"]["exit"] == 1 and summary["events"]["enter"] == 1
    assert log.summary()["would_be_alerts"] == 1


def test_floor_points_from_yaml(tmp_path):
    src = (ROOT / "configs" / "store_gas_station_small.yaml").read_text()
    cfg = tmp_path / "cam.yaml"
    cfg.write_text(src.replace("  fps: 15\n", "  fps: 15\n  floor_points: [[0, 0, 0, 0], [1280, 0, 12.8, 0], "
                                              "[1280, 720, 12.8, 7.2], [0, 720, 0, 7.2]]\n", 1))
    H = load_store_config(cfg).floor_homography()
    assert to_floor(H, 640, 360) == pytest.approx((6.4, 3.6), abs=1e-6)
