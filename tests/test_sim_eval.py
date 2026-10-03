"""Simulator scoring harness: adapter geometry, POS feed, scorer matching, and the whole chain on the
synthetic fixture (TOY DATA standing in for Isaac Sim output, which has not been produced yet)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from bree.sim import isaac_adapter as ia
from bree.sim.isaac.convert import project
from bree.sim.sim_eval import final_alerts, score
from bree.sim.sim_fixture import LAYOUT


def test_camera_params_match_the_kit_worked_example():
    # isaac-sim/README.md: camera at three (2.0, 2.6, -1.0), yaw 90 deg, pitch -30 deg looks toward -x and down.
    cam = {"id": "c", "position": [2.0, 2.6, -1.0], "yaw": math.pi / 2, "pitch": -math.pi / 6, "hfov": 90,
           "resolution": [1280, 720], "kind": "shelf"}
    assert ia.camera_forward(cam) == pytest.approx((-0.866, -0.5, 0.0), abs=1e-3)
    p = ia.camera_params_from_layout(cam)
    view, proj, w, h, note = ia.load_camera(p, cam)
    assert "row-vector" in note and "WARNING" not in note
    # a point 1 m to the camera's right (three +z is its left when it looks down -x, so right is -z) lands right of centre
    ahead = [2.0 - 4 * 0.866, 2.6 - 4 * 0.5, -1.0]
    px, front = project([ia.to_usd(ahead), ia.to_usd([ahead[0], ahead[1], ahead[2] - 1.0])], view, proj, w, h)
    assert front.all() and px[0] == pytest.approx((640, 360), abs=0.5)
    assert px[1][0] == pytest.approx(640 + 640 / 4, abs=1) and px[1][1] == pytest.approx(360, abs=1)
    # column-vector files (transposed matrices) are detected and still project the same
    t = {**p, "cameraViewTransform": np.array(p["cameraViewTransform"]).reshape(4, 4).T.reshape(-1).tolist(),
         "cameraProjection": np.array(p["cameraProjection"]).reshape(4, 4).T.reshape(-1).tolist()}
    assert "transposed" in ia.load_camera(t, cam)[4]


def test_zones_and_floor_points_from_layout():
    zones = {z["name"]: z for z in ia.zones_3d(LAYOUT)}
    assert {n: z["kind"] for n, z in zones.items()} == {"cooler-1": "cooler", "gondola-1": "shelf",
                                                         "counter": "register", "door-back": "exit"}
    reg = np.array(zones["counter"]["points"])
    assert np.allclose(reg[:, 1], 0) and reg[:, 2].min() == pytest.approx(0.7) and reg[:, 2].max() == pytest.approx(1.9)
    door = np.array(zones["door-back"]["points"])            # strip runs from the back wall into the store
    assert door[:, 2].max() == pytest.approx(-1.5) and door[:, 2].min() == pytest.approx(-3.2)
    cam = LAYOUT["cameras"][0]
    view, proj, w, h, _ = ia.load_camera(ia.camera_params_from_layout(cam), cam)
    fp = ia.floor_points(LAYOUT, view, proj, w, h)
    assert len(fp) >= 4
    from bree.track.multicam import homography, to_floor
    H = homography([p[:2] for p in fp], [p[2:] for p in fp])
    px, _ = project([ia.to_usd((1.0, 0.0, -1.0))], view, proj, w, h)
    assert to_floor(H, *px[0]) == pytest.approx((1.0, -1.0), abs=0.02)


TRUTH = [
    {"t": 5.0, "shopper": "Character", "skuId": "candy_bar", "slotId": "G1-S1", "outcome": "paid", "tResolved": 14.0,
     "cameras": [{"id": "cam-shelf-01", "px": 30}]},
    {"t": 6.0, "shopper": "Character", "skuId": "chips_bag", "slotId": "G1-S2", "outcome": "paid", "tResolved": 14.0, "cameras": []},
    {"t": 26.0, "shopper": "Character_01", "skuId": "energy_drink", "slotId": "C1-S2", "outcome": "concealed", "tResolved": 28.0,
     "cameras": [{"id": "cam-cooler-01", "px": 25}]},
    {"t": 40.0, "shopper": "Character_02", "skuId": "soda_bottle", "slotId": "C1-S1", "outcome": "concealed", "tResolved": 42.0,
     "cameras": []},
]


def test_pos_feed_delay_dropout_noise():
    skus = [s["id"] for s in LAYOUT["skus"]]
    rec, info = ia.pos_feed(TRUTH, skus, delay_s=2.0, delay_sd_s=0.0)
    assert len(rec) == 1 and rec[0]["t"] == 16.0 and rec[0]["terminal"] == "pos_1"
    assert sorted(i["sku"] for i in rec[0]["items"]) == ["candy_bar", "chips_bag"]
    from bree.ledger.payments import parse_payment
    assert parse_payment(rec[0]).t == 16.0                       # the pipeline's own parser accepts it
    rec, info = ia.pos_feed(TRUTH, skus, dropout=1.0)
    assert rec == [] and info["dropped"][0]["shopper"] == "Character"
    rec, info = ia.pos_feed(TRUTH, skus, sku_noise=1.0, seed=3)
    assert len(info["sku_swapped"]) == 2 and all(s["true"] != s["rung_as"] for s in info["sku_swapped"])
    assert ia.pos_feed(TRUTH, skus, sku_noise=0.5, seed=7) == ia.pos_feed(TRUTH, skus, sku_noise=0.5, seed=7)


def _alert(aid, t_exit, items, tier="alert", person=1):
    return {"alert_id": aid, "person_id": person, "tier": tier, "confidence": 0.9, "t_exit": t_exit, "t_emitted": t_exit + 5,
            "latency_s": 5.0, "group": [], "retracts": None,
            "unpaid_items": [{"category": c, "sku": None, "t_pick": t, "zone": z} for c, t, z in items]}


def test_score_matches_alerts_to_ground_truth():
    alerts = [_alert("A1", 33.0, [("soda_bottle", 26.3, "cooler-1")], person=3),      # right thief, wrong SKU
              _alert("A2", 20.0, [("candy_bar", 5.2, "gondola-1")], person=1),        # honest shopper flagged
              _alert("A3", 50.0, [("chips_bag", 47.0, "gondola-1")], person=9),       # matches nobody
              _alert("A4", 60.0, [("jerky", 55.0, None)], tier="review", person=8)]
    events = [{"type": "pick", "t": 26.3, "person_id": 3, "meta": {"camera": "cam-cooler-01"}},
              {"type": "pick", "t": 5.2, "person_id": 1, "meta": {"camera": "cam-shelf-01"}}]
    card = score(TRUTH, alerts, LAYOUT, duration_s=1800.0, pipeline_events=events)
    s = card["summary"]
    assert (s["concealed"], s["caught"], s["theft_recall"]) == (2, 1, 0.5)
    assert (s["alerts"], s["false_alerts"], s["alert_precision"]) == (3, 2, pytest.approx(0.333))
    assert s["sku_correct_rate"] == 0.0 and s["false_alerts_per_hour"] == 4.0
    assert s["time_to_alert_s"] == {"median": 10.0, "max": 10.0}
    assert s["false_alerts_on_honest_shoppers"] == 1 and s["false_alerts_unmatched_to_any_shopper"] == 1
    assert s["review_tier_alerts"] == 1
    assert card["by_zone_kind"]["cooler"]["theft_recall"] == 0.5 and card["by_zone_kind"]["shelf"]["false_alerts"] == 2
    assert card["by_zone"]["cooler-1"]["caught"] == 1
    assert card["by_camera_kind"]["cooler"]["theft_recall"] == 1.0 and card["by_camera_kind"]["unseen"]["caught"] == 0
    assert card["by_camera_kind"]["shelf"]["false_alerts"] == 1
    # an alert before the concealment, or long after it, does not count
    assert score(TRUTH, [_alert("A1", 10.0, [("energy_drink", 26.0, "cooler-1")])], LAYOUT, 60.0)["summary"]["caught"] == 0
    assert score(TRUTH, [_alert("A1", 400.0, [("energy_drink", 26.0, "cooler-1")])], LAYOUT, 60.0)["summary"]["caught"] == 0


def test_retracted_alerts_are_dropped():
    recs = [_alert("A1", 10.0, []), _alert("A2", 12.0, []),
            {**_alert("A3", 10.0, [], tier="retracted"), "retracts": "A1"}, {**_alert("A4", 12.0, [], tier="review"), "retracts": "A2"}]
    assert [(a["alert_id"], a["tier"]) for a in final_alerts(recs)] == [("A2", "review")]


@pytest.mark.vision     # runs the real pipeline (tracker needs ultralytics); about 80 s
def test_fixture_end_to_end(tmp_path):
    from bree.sim.sim_eval import run
    from bree.sim.sim_fixture import make_fixture
    fx = make_fixture(tmp_path / "sim")
    assert [e["outcome"] for e in fx["events"]] == ["paid", "concealed"]
    card = run(tmp_path / "sim", fx["layout"], tmp_path / "eval", backend="toy")
    cams = {c["id"]: c for c in card["run"]["cameras"]}
    assert set(cams) == {"cam-overhead-01", "cam-cooler-01", "cam-shelf-01"}
    assert card["run"]["fps"] == 10.0 and not card["run"]["warnings"]
    assert cams["cam-cooler-01"]["zones"] == ["cooler-1"] and "gondola-1" in cams["cam-shelf-01"]["zones"]
    assert cams["cam-overhead-01"]["zones"] == ["counter"]
    s = card["summary"]
    assert (s["concealed"], s["caught"], s["alerts"], s["false_alerts"]) == (1, 1, 1, 0)
    assert s["sku_correct_rate"] == 1.0 and s["time_to_alert_s"]["max"] < 20
    assert card["alerts"][0]["shopper"] == "Character_01" and card["alerts"][0]["camera_kind"] == "cooler"
    assert card["by_zone_kind"]["cooler"]["theft_recall"] == 1.0
    assert json.loads((tmp_path / "eval" / "scorecard.json").read_text())["summary"] == s
    assert "Theft recall" in (tmp_path / "eval" / "scorecard.md").read_text()
