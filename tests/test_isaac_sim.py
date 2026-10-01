"""Isaac Sim pipeline pieces that run without Isaac Sim: plans, geometry, annotator parsing, conversion, overlays."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from bree.eval.toy_eval import match_alerts
from bree.events.zones import load_store_config
from bree.sim.isaac import convert as cv
from bree.sim.isaac import plan as pl

ROOT = Path(__file__).resolve().parents[1]
CATALOG = pl.load_catalog()


@pytest.fixture(scope="module")
def plans():
    return [pl.plan_episode(s, t, CATALOG) for s in range(1, 21) for t in range(2)]


def test_plans_are_deterministic_and_split_by_scene(plans):
    assert json.dumps(pl.plan_episode(3, 1, CATALOG)) == json.dumps(pl.plan_episode(3, 1, CATALOG))
    by_scene = {}
    for p in plans:
        by_scene.setdefault(p["scene_seed"], set()).add(p["split"])
        assert p["split"] == pl.split_for_scene(p["scene_seed"])
    assert all(len(s) == 1 for s in by_scene.values())               # every take of a scene in one split
    counts = [pl.split_for_scene(s) for s in range(1000)]
    assert counts.count("train") == 800 and counts.count("val") == 100 and counts.count("test") == 100


def test_facing_convention():
    assert pl.facing_angle(0, -1) == pytest.approx(0)       # angle 0 faces -y (asset bind pose)
    assert pl.facing_angle(1, 0) == pytest.approx(90)
    assert pl.facing_angle(0, 1) == pytest.approx(180)
    for ang in (0, 37, 90, 180, 270):
        fx, fy = pl.forward(ang)
        assert pl.facing_angle(fx, fy) == pytest.approx(ang % 360, abs=1e-6)
        rx, ry = pl.right_of(ang)
        assert fx * ry - fy * rx == pytest.approx(-1)        # right is clockwise of forward seen from above


def test_reach_stand_puts_the_right_hand_on_the_product():
    for face in ((0, 1), (1, 0), (-1, 0)):
        sx, sy, ang = pl.reach_stand((1.0, 2.0), face)
        fx, fy = pl.forward(ang)
        rx, ry = pl.right_of(ang)
        assert (sx + pl.REACH_FWD * fx + pl.REACH_RIGHT * rx, sy + pl.REACH_FWD * fy + pl.REACH_RIGHT * ry) == \
            pytest.approx((1.0, 2.0))
        assert (fx, fy) == pytest.approx(face)


def _inside_fixture(x, y, fixtures, margin=0.15):
    for f in fixtures:
        if f["name"] in ("floor", "ceiling", "door_mat") or f["center"][2] - f["size"][2] / 2 > 1.8:
            continue
        if abs(x - f["center"][0]) < f["size"][0] / 2 + margin and abs(y - f["center"][1]) < f["size"][1] / 2 + margin:
            return f["name"]
    return None


def test_every_planned_stop_is_walkable(plans):
    """Every GoTo target is inside the floor area and not inside a fixture (shelf, cooler, counter, wall)."""
    for p in plans:
        fixtures = p["scene"]["fixtures"]
        for person in p["people"]:
            for c in person["commands"]:
                w = c.split()
                if w[0] in ("GoTo", "Dequeue"):
                    x, y = float(w[-4]), float(w[-3])
                    hit = _inside_fixture(x, y, fixtures)
                    assert hit is None, (p["episode"], person["name"], c, hit)
        for x, y, *_ in p["queue"]["spots"]:
            assert _inside_fixture(x, y, fixtures) is None


def test_products_sit_on_shelves_and_reaches_target_pickable_ones(plans):
    p = plans[0]
    prods = {q["id"]: q for q in p["scene"]["products"]}
    for q in prods.values():
        if q["zone"].startswith("shelf_"):
            gx = pl.GONDOLA_X["ABC".index(q["zone"][-1])]
            assert abs(q["pos"][0] - gx) < pl.GONDOLA_W / 2          # on the gondola, not in the aisle
    for plan in plans:
        for person in plan["people"]:
            for a in person["actions"]:
                if a and a["verb"] in ("take", "put", "restock"):
                    assert prods_of(plan)[a["product"]]["pickable"]


def prods_of(plan):
    return {q["id"]: q for q in plan["scene"]["products"]}


def test_intent_matches_archetype(plans):
    seen = set()
    for plan in plans:
        for person in plan["people"]:
            i, a = person["intent"], person["archetype"]
            seen.add(a)
            assert i["thief"] == bool(i["stolen"])
            if a in ("normal_buy", "pocket_then_pay", "browse", "put_back", "cashier", "restock"):
                assert not i["thief"], (plan["episode"], person["name"], a)
            if a in ("conceal_walkout", "walkout_in_hand", "conceal_partial_pay"):
                assert i["thief"]
            if a.startswith("conceal") or a == "pocket_then_pay":
                assert i["concealed"] and any(x and x["verb"] == "conceal" for x in person["actions"])
            if a == "conceal_partial_pay":
                assert i["paid"] and set(i["concealed"]) == set(i["stolen"])
            assert len(person["commands"]) == len(person["actions"])
    assert {"normal_buy", "put_back", "browse", "conceal_walkout", "walkout_in_hand"} <= seen


def test_command_file_and_facing_offset(plans):
    text = pl.command_file(plans[0])
    lines = text.strip().split("\n")
    q = plans[0]["queue"]["name"]
    assert lines[0] == f"Queue {q}" and lines[1].startswith(f"Queue_Spot {q} 0 ")
    names = {p["name"] for p in plans[0]["people"]}
    assert all(line.split()[0] in names for line in lines[1 + len(plans[0]["queue"]["spots"]):])
    shifted = pl.with_facing_offset("A GoTo 1.00 2.00 0 270.0\nA GoTo 1.00 2.00 0 _\nQueue_Spot q 0 1 2 0 90.0", 90)
    assert shifted.split("\n") == ["A GoTo 1.00 2.00 0 0.0", "A GoTo 1.00 2.00 0 _", "Queue_Spot q 0 1 2 0 180.0"]
    assert pl.with_facing_offset(text, 0) == text


# ------------------------------------------------------------------ camera geometry
def _perspective_row(fov_x_deg, w, h, near=0.05, far=100.0):
    f = 1 / math.tan(math.radians(fov_x_deg) / 2)
    col = np.array([[f, 0, 0, 0], [0, f * w / h, 0, 0],
                    [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])
    return col.T                                              # row-vector form, as Replicator reports it


def _camera_params(eye, target, w=1280, h=720):
    cam_to_world = np.array(pl.look_at(eye, target))
    return {"cameraViewTransform": np.linalg.inv(cam_to_world).reshape(-1).tolist(),
            "cameraProjection": _perspective_row(80, w, h).reshape(-1).tolist(), "renderProductResolution": [w, h]}


def test_look_at_and_projection_agree():
    cam = _camera_params((5.6, -4.2, 3.0), (-1.5, 2.8, 0.9))
    view, proj, w, h = cv.camera_matrices(cam)
    px, front = cv.project([(-1.5, 2.8, 0.9)], view, proj, w, h)
    assert front.all() and px[0] == pytest.approx((640, 360), abs=1e-6)
    above, _ = cv.project([(-1.5, 2.8, 1.9)], view, proj, w, h)
    assert above[0][1] < 360                                    # higher in the world = higher in the image
    _, behind = cv.project([(8.0, -7.0, 3.5)], view, proj, w, h)
    assert not behind.any()
    for c in pl.camera_rig(5, 0):
        assert 2.5 <= c["eye"][2] <= 3.0


def test_reprojection_check_detects_transposed_matrices():
    cam = _camera_params((0, 1, 2.9), (0, 3.9, 1.1))
    view, proj, w, h = cv.camera_matrices(cam)
    pts = [(0.2, 3.5, 1.2), (-0.4, 3.0, 0.5), (0.5, 3.9, 1.6)]
    px, _ = cv.project(pts, view, proj, w, h)
    pairs = list(zip(pts, px.tolist()))
    assert cv.reprojection_error(cam, pairs) == (0.0, False)
    cam_t = {**cam, "cameraViewTransform": view.T.reshape(-1).tolist(), "cameraProjection": proj.T.reshape(-1).tolist()}
    assert cv.reprojection_error(cam_t, pairs) == (0.0, True)


def test_zone_polygons_from_3d_boxes():
    cam = _camera_params((0, -3, 3.0), (0, 0, 0))
    zones = [{"name": "register", "kind": "register", "lo": [-0.5, -0.5, 0], "hi": [0.5, 0.5, 0]},
             {"name": "shelf_A", "kind": "shelf", "lo": [-0.5, 1, 0], "hi": [0.5, 2, 1.4]},
             {"name": "behind", "kind": "exit", "lo": [-1, -9, 0], "hi": [1, -8, 0]}]
    polys = {z["name"]: z for z in cv.zone_polygons(zones, cam)}
    assert set(polys) == {"register", "shelf_A"}
    assert len(polys["register"]["polygon"]) == 4
    assert len(polys["shelf_A"]["polygon"]) >= 4
    poly = np.array(polys["register"]["polygon"])
    assert poly[:, 0].min() < 640 < poly[:, 0].max() and poly[:, 1].min() < 360 < poly[:, 1].max()
    assert cv.hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]) == [[0, 0], [1, 0], [1, 1], [0, 1]]


# ------------------------------------------------------------------ annotator parsing
def _bbox_data():
    dt = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
                   ("occlusionRatio", "<f4")])
    return {"data": np.array([(0, 10, 20, 110, 320, 0.1), (1, 50, 60, 70, 90, 0.0)], dtype=dt),
            "info": {"idToLabels": {"0": {"class": "person"}, "1": {"class": "soda_bottle"}},
                     "primPaths": ["/World/Characters/Shopper_01/x/skelroot", "/World/Store/Products/cooler_0_2_1"],
                     "bboxIds": np.array([0, 1])}}


def _skeleton_data(w=1280):
    j = ["RL_BoneRoot", "RL_BoneRoot/Hip/L_Thigh", "RL_BoneRoot/Hip/Head/L_Eye", "RL_BoneRoot/Hip/Head/R_Eye",
         "RL_BoneRoot/Hip/L_Upperarm/L_Forearm/L_Hand", "RL_BoneRoot/Hip/R_Upperarm/R_Forearm/R_Hand"]
    p2 = [[0, 0], [100, 300], [101, 100], [111, 100], [w + 50, 200], [140, 210]]
    p3 = [[0, 0, 0], [0, 0, 0.9], [0, 0, 1.5], [0.1, 0, 1.5], [0.6, 0, 1.3], [-0.6, 0, 1.3]]
    biped = ["Root", "Root/Pelvis"]
    return {"numSkeletons": 2,
            "skelPath": ["/World/Characters/Shopper_01/x/skel", "/World/Characters/Biped_Setup/skel"],
            "skeletonJoints": [str(j), str(biped)], "inView": np.array([True, False]),
            "translations2d": np.array(p2 + [[0, 0], [0, 0]], float).reshape(-1), "translations2dSizes": np.array([6, 2]),
            "globalTranslations": np.array(p3 + [[0, 0, 0]] * 2, float).reshape(-1),
            "globalTranslationsSizes": np.array([6, 2]),
            "jointOcclusions": np.array([False, False, False, False, False, True, False, False]),
            "jointOcclusionsSizes": np.array([6, 2])}


def test_bbox_rows_and_skeleton_parsing_and_coco17():
    rows = cv.bbox_rows(_bbox_data())
    assert rows[0] == {"label": "person", "prim": "/World/Characters/Shopper_01/x/skelroot", "box": [10, 20, 110, 320],
                       "occ": pytest.approx(0.1, abs=1e-3)}
    assert rows[1]["label"] == "soda_bottle"
    json.dumps(rows)
    sk = cv.split_skeletons(_skeleton_data())
    assert len(sk) == 1 and sk[0]["joints"] == ["L_Thigh", "L_Eye", "R_Eye", "L_Hand", "R_Hand"]   # Biped dropped
    json.dumps(sk)
    kps = cv.coco17(sk[0], 1280, 720)
    assert len(kps) == 17
    assert kps[cv.COCO17.index("left_hip")] == [100, 300, 2]
    assert kps[cv.COCO17.index("nose")] == [106, 100, 2]                 # midpoint of the eyes
    assert kps[cv.COCO17.index("left_wrist")][2] == 0                     # outside the image: unlabeled
    assert kps[cv.COCO17.index("right_wrist")] == [140, 210, 1]          # occluded
    assert kps[cv.COCO17.index("left_ear")][2] == 0 and kps[cv.COCO17.index("left_knee")][2] == 0
    assert cv.split_skeletons({"numSkeletons": 0}) == [] and cv.bbox_rows(None) == []


# ------------------------------------------------------------------ episode -> clips, overlay, check, scoring
def _fake_raw_episode(tmp: Path) -> Path:
    import cv2
    plan = pl.plan_episode(7, 0, CATALOG)
    plan["cameras"] = plan["cameras"][:1]
    plan["people"] = [p for p in plan["people"] if p["role"] == "shopper"][:1]
    person = plan["people"][0]
    prod = next(q for q in plan["scene"]["products"] if q["pickable"])
    person["intent"] = {"picked": [prod["id"]], "put_back": [], "concealed": [prod["id"]], "paid": [],
                        "stolen": [prod["id"]], "thief": True}
    raw = tmp / "raw" / plan["episode"]
    cam_dir = raw / plan["cameras"][0]["name"]
    (cam_dir / "rgb").mkdir(parents=True)
    c = plan["cameras"][0]
    (cam_dir / "camera.json").write_text(json.dumps(_camera_params(c["eye"], c["target"])))
    name = person["name"]
    events = [{"t": 1.0, "frame": 15, "person": name, "action": "enter"},
              {"t": 2.0, "frame": 30, "person": name, "action": "pick", "product": prod["id"], "category": prod["category"],
               "sku": prod["sku"], "hand": "right"},
              {"t": 3.0, "frame": 45, "person": name, "action": "conceal", "product": prod["id"],
               "category": prod["category"], "sku": prod["sku"], "anchor": "pocket"},
              {"t": 4.0, "frame": 60, "person": name, "action": "exit"}]
    sk = _skeleton_data()
    with open(cam_dir / "ann.jsonl", "w") as fa, open(raw / "world.jsonl", "w") as fw:
        for k in range(4):
            img = np.full((720, 1280, 3), 90, np.uint8)
            cv2.imwrite(str(cam_dir / "rgb" / f"{k:05d}.jpg"), img)
            boxes = cv.bbox_rows(_bbox_data())
            boxes[0]["prim"] = f"/World/Characters/{name}/x/skelroot"
            boxes[1]["prim"] = f"/World/Store/Products/{prod['id']}"
            sk["skelPath"][0] = f"/World/Characters/{name}/x/skel"
            fa.write(json.dumps({"f": k, "t": k / 15, "boxes": boxes, "skeletons": cv.split_skeletons(sk)}) + "\n")
            fw.write(json.dumps({"f": k, "t": k / 15, "people": {}, "products": {prod["id"]: {
                "state": "hand", "holder": name, "hand": "right", "pos": [0, 0, 1]}}}) + "\n")
    (raw / "plan.json").write_text(json.dumps(plan))
    (raw / "events.json").write_text(json.dumps(events))
    (raw / "meta.json").write_text(json.dumps({"warnings": []}))
    (raw / "done.json").write_text("{}")
    return raw


def test_convert_overlay_check_and_score(tmp_path):
    raw = _fake_raw_episode(tmp_path)
    out = tmp_path / "out"
    n = cv.convert_all(raw.parent, out, ROOT / "configs" / "store_gas_station_small.yaml", delete_raw=True)
    assert n == 1 and (raw / "converted.json").exists() and not (raw / "cam_main" / "rgb").exists()
    assert cv.convert_all(raw.parent, out, ROOT / "configs" / "store_gas_station_small.yaml") == 0   # idempotent
    split = json.loads((raw / "plan.json").read_text())["split"]
    stem = out / split / f"isaac_{raw.name}_cam_main"
    truth = json.loads(Path(f"{stem}.truth.json").read_text())
    p = truth["people"][0]
    assert p["thief"] and p["stolen"] == p["concealed"] and p["t_enter"] == 1.0 and p["t_exit"] == 4.0
    assert p["matches_plan"] and truth["frames"] == 4 and truth["reprojection_px"] is not None
    store = load_store_config(f"{stem}.store.yaml")                      # the pipeline can load it
    assert store.camera_id == "cam_main" and store.resolution == (1280, 720)
    frames = [json.loads(line) for line in Path(f"{stem}.frames.jsonl").read_text().splitlines()]
    assert frames[0]["persons"][0]["track_id"] == 1 and len(frames[0]["persons"][0]["keypoints"]) == 17
    assert frames[0]["products"][0]["holder"] == p["name"]
    assert Path(f"{stem}.mp4").stat().st_size > 0
    ov = cv.overlay_clip(stem, tmp_path / "ov")
    assert ov.exists() and ov.stat().st_size > 0
    assert cv.pick_clips_for_overlay(out, 5) == [stem]
    chk = cv.check_clip(stem)
    assert chk["frames"] == 4 and chk["events"]["conceal"] == 1 and chk["thieves_without_exit"] == []
    rows = match_alerts(truth, [{"alert_id": "a1", "t_exit": 4.5, "tier": "alert", "confidence": 0.8,
                                 "unpaid_items": [{"category": p["stolen"][0]}], "latency_s": 1.0}])
    assert rows[0]["thief"] and rows[0]["tier"] == "alert"


def test_people_truth_put_back_and_paid_are_not_stolen():
    plan = {"categories": {"a": "soda_bottle", "b": "candy_bar"},
            "people": [{"name": "S", "role": "shopper", "archetype": "put_back",
                        "intent": {"picked": ["a", "b"], "stolen": []}},
                       {"name": "E", "role": "employee", "archetype": "cashier", "intent": {"picked": [], "stolen": []}}]}
    ev = [{"t": 1, "person": "S", "action": "enter"}, {"t": 2, "person": "S", "action": "pick", "product": "a"},
          {"t": 3, "person": "S", "action": "put_back", "product": "a"}, {"t": 4, "person": "S", "action": "pick", "product": "b"},
          {"t": 5, "person": "S", "action": "pay", "products": ["b"]}, {"t": 6, "person": "S", "action": "exit"}]
    s, e = cv.people_truth(plan, ev)
    assert not s["thief"] and s["picked"] == ["soda_bottle", "candy_bar"] and s["put_back"] == ["soda_bottle"]
    assert s["matches_plan"]
    assert e["t_exit"] is None and e["t_enter"] == 0.0
    assert match_alerts({"clip": "c", "people": [s, e]}, []) == [
        {"clip": "c", "person": "S", "thief": False, "stolen": [], "tier": None, "confidence": None,
         "flagged_items": [], "decision_latency_s": None}]
