"""Camera calibration to the floor plan (bree/calib/camera.py, scripts/calibrate.py), triangulation and the
3D slot of a pick (bree/calib/slots.py, StoreEvents). Synthetic geometry with known ground truth."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from bree.calib.camera import Camera, calibrate, check, from_layout, marks_to_points
from bree.calib.slots import NoiseModel, SlotLocator, SlotMap, perturbed, triangulate
from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.events.types import Catalog, EventType as E
from bree.events.zones import StoreConfig, Zone
from bree.track.multicam import StoreEvents, homography, to_floor

cv2 = pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "tests" / "fixtures" / "calib_layout.json").read_text())
spec = importlib.util.spec_from_file_location("calibrate_script", ROOT / "scripts" / "calibrate.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def look_at(cid, pos, target, hfov=60, res=(1280, 720)) -> dict:
    d = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    return {"id": cid, "position": list(pos), "yaw": math.atan2(-d[0], -d[2]), "pitch": math.asin(d[1]),
            "hfov": hfov, "resolution": list(res), "kind": "shelf"}


def marks(cam: Camera, pts3d, rng=None, noise=0.0) -> np.ndarray:
    px, z = cam.project(pts3d)
    assert cam.in_frame(px, z).all()
    if noise:
        px = px + rng.normal(0, noise, px.shape)
    p = np.asarray(pts3d, float)
    return np.c_[px, p[:, 0], p[:, 2], p[:, 1]]                       # [x_px, y_px, x_m, y_m, h_m]


CAM = from_layout(look_at("c", (0.5, 2.6, 4.0), (0.0, 0.0, 0.0)))
FLOOR = [(x, 0.0, z) for x in (-1.5, 0.0, 1.5) for z in (-1.5, 0.0, 1.5)]
RAISED = [(-1.0, 1.2, -0.5), (1.0, 0.9, 0.5), (0.3, 0.7, -1.0)]


# ------------------------------------------------------------------ camera model


def test_layout_camera_projects_like_the_simulator_adapter():
    """Same pixels as the Replicator-style matrices bree.sim.isaac_adapter builds from the same layout camera."""
    from bree.sim.isaac.convert import project
    from bree.sim.isaac_adapter import camera_params_from_layout, to_usd
    rng = np.random.default_rng(0)
    X = rng.uniform([-9, 0, -6.5], [9, 3, 6.5], (300, 3))
    for c in LAYOUT["cameras"]:
        cam, p = from_layout(c), camera_params_from_layout(c)
        px, z = cam.project(X)
        ref, front = project([to_usd(x) for x in X], p["cameraViewTransform"], p["cameraProjection"], *c["resolution"])
        assert ((z > 1e-6) == front).all() and np.abs(px[front] - ref[front]).max() < 1e-4
        assert np.linalg.det(cam.R) == pytest.approx(1.0)


def test_floor_homography_of_a_pose_maps_pixels_to_the_floor_plan():
    px, _ = CAM.project(FLOOR)
    for (u, v), (x, _, z) in zip(px, FLOOR):
        assert to_floor(CAM.floor_homography(), u, v) == pytest.approx((x, z), abs=1e-6)
    assert Camera.from_dict(CAM.to_dict()).project(RAISED)[0] == pytest.approx(CAM.project(RAISED)[0], abs=0.05)


def test_homography_is_stable_with_large_pixel_coordinates_and_noise():
    """4MP pixels next to metres: the normalised fit keeps the leave-one-out floor error at the noise level."""
    big = from_layout(look_at("c", (0.5, 2.6, 4.0), (0.0, 0.0, 0.0), res=(2688, 1520)))
    m = marks(big, FLOOR, np.random.default_rng(1), noise=2.0)
    H = homography(m[:, :2], m[:, 2:4])
    err = [math.dist(to_floor(H, u, v), (x, y)) for u, v, x, y, _ in m]
    assert max(err) < 0.02


# ------------------------------------------------------------------ calibrate


def test_floor_marks_and_a_known_lens_give_the_pose():
    cam, rep = calibrate(marks(CAM, FLOOR), CAM.resolution, f=CAM.f)
    assert rep["method"] == "pnp" and rep["rms_px"] < 1e-3 and rep["floor_loo_rms_m"] < 1e-6
    assert np.linalg.norm(cam.C - CAM.C) < 1e-3 and np.abs(cam.R - CAM.R).max() < 1e-4


def test_six_marks_give_the_focal_length_too():
    cam, rep = calibrate(marks(CAM, FLOOR + RAISED), CAM.resolution)
    assert rep["method"] == "pnp+focal" and rep["hfov_deg"] == pytest.approx(60.0, abs=0.05)
    assert np.linalg.norm(cam.C - CAM.C) < 0.01


def test_noisy_marks_report_their_error_and_stay_close():
    m = marks(CAM, FLOOR + RAISED, np.random.default_rng(0), noise=2.0)
    cam, rep = calibrate(m, CAM.resolution, f=CAM.f)
    assert 0.5 < rep["rms_px"] < 4.0 and rep["max_px"] >= rep["rms_px"]
    assert np.linalg.norm(cam.C - CAM.C) < 0.05


def test_four_floor_marks_without_a_lens_is_homography_only_and_unverified():
    m = marks(CAM, FLOOR[:2] + FLOOR[6:8])
    cam, rep = calibrate(m, CAM.resolution)
    assert cam is None and rep["method"] == "homography" and rep["floor_loo_rms_m"] is None
    out = check(m, None, CAM.resolution)
    assert not out["ok"] and "exactly 4" in out["problems"][0]


def test_marks_on_one_line_are_not_a_floor_mapping():
    cam, rep = calibrate(marks(CAM, [(x, 0.0, 0.0) for x in (-1.5, -0.5, 0.5, 1.5)]), CAM.resolution)
    assert cam is None and rep["method"] is None
    assert not check(marks(CAM, FLOOR[:3]), None, CAM.resolution)["ok"]


def test_check_flags_a_camera_that_moved():
    """Calibration stored, then the camera is turned 2 degrees: the marks clicked on a new frame no longer fit."""
    stored = calibrate(marks(CAM, FLOOR + RAISED), CAM.resolution, f=CAM.f)[0].to_dict()
    assert check(marks(CAM, FLOOR + RAISED), stored, CAM.resolution)["ok"]
    a = math.radians(2.0)
    bumped = Camera(CAM.f, CAM.cx, CAM.cy,
                    np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]) @ CAM.R,
                    CAM.C, CAM.resolution)
    out = check(marks(bumped, FLOOR + RAISED), stored, CAM.resolution, max_px=5.0)
    assert not out["ok"] and out["rms_px"] > 20 and "reprojection" in out["problems"][0]


def _store_yaml(tmp_path, name, m, extra=None):
    raw = yaml.safe_load((ROOT / "configs" / "store_gas_station_small.yaml").read_text())
    raw["camera"] = {"id": name, "resolution": list(CAM.resolution), "fps": 15,
                     "floor_points": [[float(v) for v in r[:4]] for r in m if r[4] == 0],
                     "marks_3d": [[float(v) for v in r] for r in m if r[4] != 0], **(extra or {})}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_script_fit_writes_the_calibration_and_check_passes_then_fails(tmp_path, capsys):
    from bree.events.zones import load_store_config
    p = _store_yaml(tmp_path, "good", marks(CAM, FLOOR + RAISED, np.random.default_rng(0), noise=1.0))
    assert cli.main(["fit", str(p), "--hfov", "60", "--in-place"]) == 0
    cfg = yaml.safe_load(p.read_text())["camera"]
    got = Camera.from_dict(cfg["calibration"])
    assert cfg["calibration"]["method"] == "pnp" and cfg["calibration"]["rms_px"] < 3
    assert np.linalg.norm(got.C - CAM.C) < 0.05
    assert load_store_config(p).floor_points                           # the file still loads as a store config
    assert cli.main(["check", str(p)]) == 0
    # A second camera whose marks were clicked after it was knocked: check fails it and exits 1.
    cfg["floor_points"] = [[r[0] + 40.0, r[1], r[2], r[3]] for r in cfg["floor_points"]]
    cfg["marks_3d"] = [[r[0] + 40.0, *r[1:]] for r in cfg["marks_3d"]]
    cfg["id"] = "knocked"
    raw = yaml.safe_load(p.read_text())
    raw["camera"] = cfg
    q = tmp_path / "knocked.yaml"
    q.write_text(yaml.safe_dump(raw))
    capsys.readouterr()
    assert cli.main(["check", str(p), str(q), "--max-px", "5"]) == 1
    out = capsys.readouterr().out
    assert "FAIL  knocked" in out and "OK    good" in out and "1 of 2 camera(s) pass" in out


def test_script_fit_from_the_layout_alone_is_marked_unverified(tmp_path):
    lay = tmp_path / "layout.json"
    lay.write_text(json.dumps({"version": 1, "cameras": [look_at("c", (0.5, 2.6, 4.0), (0.0, 0.0, 0.0))]}))
    p = _store_yaml(tmp_path, "c", np.zeros((0, 5)))
    assert cli.main(["fit", str(p), "--layout", str(lay), "--in-place"]) == 0
    cfg = yaml.safe_load(p.read_text())["camera"]
    assert cfg["calibration"]["method"] == "layout" and len(cfg["floor_points"]) >= 4
    for u, v, x, z in cfg["floor_points"]:                             # generated from the pose, on the floor plan
        assert to_floor(CAM.floor_homography(), u, v) == pytest.approx((x, z), abs=0.01)
    assert cli.main(["check", str(p)]) == 1
    # With marks, the fit is compared with the design pose.
    p2 = _store_yaml(tmp_path, "c", marks(CAM, FLOOR))
    assert cli.main(["fit", str(p2), "--layout", str(lay), "--in-place"]) == 0
    c2 = yaml.safe_load(p2.read_text())["camera"]["calibration"]
    assert c2["method"] == "pnp" and c2["moved_m"] < 0.01 and c2["turned_deg"] < 0.05


# ------------------------------------------------------------------ triangulation


A = from_layout(look_at("A", (-2.0, 1.6, 2.5), (0.0, 1.0, 0.3)))
B = from_layout(look_at("B", (2.0, 1.6, 2.5), (0.0, 1.0, 0.3)))


def test_triangulation_recovers_a_point_and_predicts_its_own_error():
    X = np.array([0.1, 1.0, 0.4])
    exact = triangulate([(A, A.project(X)[0][0]), (B, B.project(X)[0][0])])
    assert np.linalg.norm(exact["xyz"] - X) < 1e-6 and exact["rms_px"] < 1e-6
    assert triangulate([(A, (640, 360))]) is None                      # one view is a line, not a point
    noise, rng = NoiseModel(pixel_px=4.0, cam_pos_m=0.02, cam_rot_deg=0.2), np.random.default_rng(0)
    err, sig = [], []
    for _ in range(400):                                               # Monte Carlo of the same model
        views = [(perturbed(c, rng, noise), c.project(X)[0][0] + rng.normal(0, noise.pixel_px, 2)) for c in (A, B)]
        t = triangulate(views, noise)
        err.append(t["xyz"] - X); sig.append(t["sigma_m"])
    worst_axis = math.sqrt(np.linalg.eigvalsh(np.cov(np.array(err).T))[-1])
    assert 0.7 < worst_axis / np.mean(sig) < 1.4                       # predicted sigma matches the spread
    assert NoiseModel(4.0, 0.0, 0.0).sigma_px(A, 3.0) == 4.0 < noise.sigma_px(A, 3.0)


def test_error_model_is_the_simulators():
    """bree/software/sim-prototype/js/cameras.js: per ray s = sqrt((pxSigma / px per metre)^2 + (range * calib)^2),
    and for two rays at angle theta  rms^2 = (s1^2 + s2^2) / sin^2(theta) + s1^2 s2^2 / (s1^2 + s2^2)."""
    assert NoiseModel.sim() == NoiseModel(1 / math.sqrt(12), 0.0, 0.1)
    for noise in (NoiseModel.sim(), NoiseModel.sim(2.0, 0.3)):
        for X in (np.array([0.1, 1.0, 0.4]), np.array([-0.6, 0.5, 0.3])):
            s, d = [], []
            for cam in (A, B):
                rng_ = float(np.linalg.norm(X - cam.C))
                px_per_m = cam.f / rng_                                # at the image centre, as the simulator's on-axis value
                s.append(math.hypot(noise.pixel_px / px_per_m, rng_ * math.radians(noise.cam_rot_deg)))
                d.append((X - cam.C) / rng_)
            sin2 = 1 - float(d[0] @ d[1]) ** 2
            q = s[0] ** 2 + s[1] ** 2
            sim_err = math.sqrt(q / sin2 + s[0] ** 2 * s[1] ** 2 / q)
            got = triangulate([(A, A.project(X)[0][0]), (B, B.project(X)[0][0])], noise)["rms_m"]
            # Off-axis pixels are slightly finer in a pinhole. This is the formula at the image centre, NOT the
            # simulator's output: with its lens edge loss and range in place of depth the simulator reads up to
            # about 11% higher on the 45 camera layout (bree/calib/slots.py docstring, audit 2026-10-05).
            assert got == pytest.approx(sim_err, rel=0.05)


def test_fixture_cameras_match_the_recommended_layout():
    """The fixture is a trimmed copy of ~/bree/software/shared/layouts/recommended-47.json (labels such as zone
    and mount dropped). Every field the fixture keeps must equal the layout's, so a layout change cannot drift
    past the calibration tests unseen. Skipped when the research repo is not next to this one."""
    src = Path.home() / "bree" / "software" / "shared" / "layouts" / "recommended-47.json"
    if not src.exists():
        pytest.skip(f"{src} not present")
    real = {c["id"]: c for c in json.loads(src.read_text())["cameras"]}
    mine = {c["id"]: c for c in LAYOUT["cameras"]}
    assert sorted(mine) == sorted(real) and len(mine) == 45
    assert all(mine[i][k] == real[i][k] for i in mine for k in mine[i])
    assert {k for c in mine.values() for k in c} >= {"position", "yaw", "pitch", "hfov", "resolution", "kind"}


def test_lines_of_sight_that_do_not_meet_are_rejected():
    """Two cameras looking at different hands: the residual is far over the gate, so no 3D point is used."""
    sm = SlotMap(SHELF)
    loc = SlotLocator({"A": A, "B": B}, sm, NoiseModel(3.0, 0.0, 0.0))
    here, there = np.array([0.15, 1.0, 0.35]), np.array([-0.3, 0.6, 0.9])
    loc.observe("A", 0.0, 1, {"right": (tuple(A.project(here)[0][0]), None)})
    loc.observe("B", 0.0, 1, {"right": (tuple(B.project(there)[0][0]), None)})
    assert loc.locate("A", 1, "right", 0.0, 0.0)["source"] == "hand_1view"


# ------------------------------------------------------------------ slots

SHELF = {"store": {"width": 6, "depth": 6, "height": 3},
         "fixtures": [{"id": "G", "type": "counter", "position": [0, 0.7, 0], "rotationY": 0.0, "size": [1.0, 1.4, 0.6]}],
         "slots": [{"id": f"S{r}-{k}", "skuId": f"sku{r}{k // 2}", "fixtureId": "G",
                    "position": [round(-0.35 + 0.1 * k, 3), y, 0.25], "size": [0.1, 0.2, 0.1]}
                   for r, y in enumerate((0.6, 1.0)) for k in range(8)]}
TARGET = "S1-5"                                                        # x = 0.15, y = 1.0, front at z = 0.30


def test_slot_map_fronts_and_normals():
    sm = SlotMap(SHELF)
    i = sm.ids.index(TARGET)
    assert sm.front[i] == pytest.approx([0.15, 1.0, 0.30]) and sm.normal[i] == pytest.approx([0, 0, 1])
    assert sm.nearest([0.16, 1.02, 0.36])[0] == i
    big = SlotMap(LAYOUT)                                              # gondola slots face away from its centre plane
    g = {f["id"]: f for f in LAYOUT["fixtures"]}
    for k, fid in enumerate(big.fixtures):
        if g[fid]["type"] == "gondola":
            assert (big.front[k] - np.asarray(g[fid]["position"])) @ big.normal[k] > 0
    assert len(big) == 2349 and big.visible(from_layout(LAYOUT["cameras"][0]), big.front).sum() > 0


def test_two_views_find_the_slot_where_one_view_is_pushed_sideways():
    """Fingertips 12 cm in front of the slot. One oblique camera sees the hand on the line to the next slot
    along; two cameras place it in 3D."""
    sm = SlotMap(SHELF)
    hand = sm.front[sm.ids.index(TARGET)] + [0, 0, 0.12]
    noise = NoiseModel(2.0, 0.0, 0.0)
    two, one = SlotLocator({"A": A, "B": B}, sm, noise), SlotLocator({"A": A}, sm, noise)
    for name, cam in (("A", A), ("B", B)):
        obs = {"right": (tuple(cam.project(hand)[0][0]), None)}
        two.observe(name, 0.0, 1, obs); one.observe(name, 0.0, 1, obs)
    got2, got1 = two.locate("A", 1, "right", 0.0, 0.0), one.locate("A", 1, "right", 0.0, 0.0)
    assert got2["id"] == TARGET and got2["views"] == 2 and got2["source"] == "hand_2view" and got2["sigma_m"] < 0.03
    assert got1["views"] == 1 and got1["id"] != TARGET and got1["id"].startswith("S1-")
    # The item's own pixel on the shelf has no such gap: one view is enough when the detector saw it there.
    item = one.locate("A", 1, "right", 0.0, 0.0, item_uv=tuple(A.project(sm.front[sm.ids.index(TARGET)])[0][0]))
    assert item["id"] == TARGET and item["source"] == "item_1view"
    assert two.locate("A", 2, "right", 0.0, 0.0) is None               # nobody else's hand


# ------------------------------------------------------------------ StoreEvents: a pick gets its slot

FPS = 15.0
KP = {"l_sh": 5, "r_sh": 6, "r_el": 8, "r_wr": 10, "l_hip": 11, "r_hip": 12, "l_an": 15, "r_an": 16}


def _pick_frames():
    """3D script: a shopper at (0, 0, 1.0) facing the shelf reaches the TARGET slot with the right hand, takes the
    item and stands holding it. Per tick: body points and the item's position (layout frame)."""
    sm = SlotMap(SHELF)
    F = sm.front[sm.ids.index(TARGET)]
    body = {"l_sh": (-0.2, 1.4, 1.0), "r_sh": (0.2, 1.4, 1.0), "l_hip": (-0.15, 0.9, 1.0), "r_hip": (0.15, 0.9, 1.0),
            "l_an": (-0.1, 0.0, 1.0), "r_an": (0.1, 0.0, 1.0), "head": (0.0, 1.7, 1.0)}
    rest_w, rest_e = np.array([0.25, 0.9, 1.1]), np.array([0.25, 1.15, 1.0])
    a = (F - body["r_sh"]) / np.linalg.norm(F - body["r_sh"])
    tip = F + [0, 0, 0.03]
    far_w, far_e = tip - 0.12 * a, tip - 0.38 * a
    ticks = [(rest_w, rest_e, F)] * 8
    ticks += [(rest_w + (far_w - rest_w) * u, rest_e + (far_e - rest_e) * u, F) for u in np.linspace(0, 1, 9)]
    back = [(far_w + (rest_w - far_w) * u, far_e + (rest_e - far_e) * u) for u in np.linspace(0, 1, 9)]
    ticks += [(w, e, w + [0, -0.04, 0]) for w, e in back]
    ticks += [(rest_w, rest_e, rest_w + [0, -0.04, 0])] * 14
    return body, ticks


def _obs(cam: Camera, tid: int, f: int, body, w, e, item, with_item: bool) -> FrameObs:
    pts = {**body, "r_wr": w, "r_el": e}
    px = {k: cam.project(v)[0][0] for k, v in pts.items()}
    xs, ys = [p[0] for p in px.values()], [p[1] for p in px.values()]
    kp = np.zeros((17, 3), np.float32)
    for k, i in KP.items():
        kp[i] = (*px[k], 0.9)
    person = PersonObs(tid, (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys)), 0.9, kp)
    prods = []
    if with_item:
        u, v = cam.project(item)[0][0]
        prods.append(ProductObs(7, (u - 8, v - 10, u + 8, v + 10), "soda_bottle", 0.85))
    return FrameObs(f, f / FPS, [person], prods)


def _run_pick(calibration: dict, slots=SHELF, slot_layout=None):
    from bree.sim.isaac.convert import hull
    sm = SlotMap(SHELF)
    corners = [(x, y, z) for x in (-0.4, 0.4) for y in (0.5, 1.1) for z in (0.2, 0.3)]
    zone = Zone("shelf_G", "shelf", np.array(hull(A.project(corners)[0]), float))
    floor = lambda cam: [[*cam.project([(x, 0, z)])[0][0], x, z] for x, z in ((-1, 0.5), (1, 0.5), (1, 2), (-1, 2))]   # noqa: E731
    rules = {"closed_world": False, **({"slot_layout": str(slot_layout)} if slot_layout else {})}
    stores = {"A": StoreConfig("s", "A", A.resolution, FPS, [zone], {}, Catalog({}), rules=rules, floor_points=floor(A)),
              "B": StoreConfig("s", "B", B.resolution, FPS, [], {}, Catalog({}), rules=rules, floor_points=floor(B))}
    se = StoreEvents(stores, {"calibration": calibration, "slots": slots})
    body, ticks = _pick_frames()
    events = []
    for f, (w, e, item) in enumerate(ticks):
        events += se.update("A", _obs(A, 1, f, body, w, e, item, True))
        events += se.update("B", _obs(B, 5, f, body, w, e, item, False))
    return se, [ev for ev in events + se.flush() if ev.type == E.PICK], sm


def test_pick_gets_a_3d_slot_from_two_cameras():
    se, picks, sm = _run_pick({"A": A, "B": B.to_dict()})
    assert len(picks) == 1 and picks[0].zone == "shelf_G" and picks[0].meta["camera"] == "A"
    slot = picks[0].meta["slot"]
    assert slot["id"] == TARGET and slot["sku"] == "sku12" and slot["views"] == 2 and slot["source"] == "hand_2view"
    assert set(slot["cameras"]) == {"A", "B"} and slot["sigma_m"] is not None
    assert np.linalg.norm(np.asarray(slot["xyz"]) - sm.front[sm.ids.index(TARGET)]) < 0.08
    assert len(se.identity.people) == 1                                # the two cameras' tracks are one person


def test_pick_falls_back_to_one_view_and_is_off_without_a_slot_layout():
    _, picks, _ = _run_pick({"A": A})                                  # camera B not calibrated
    slot = picks[0].meta["slot"]
    assert slot["views"] == 1 and slot["source"] == "item_1view" and slot["id"] == TARGET
    se, picks, _ = _run_pick({"A": A, "B": B}, slots=None)             # default: no slot layout, nothing added
    assert se.locator is None and len(picks) == 1 and "slot" not in picks[0].meta
    se, picks, _ = _run_pick({})                                       # no calibrated camera: same
    assert se.locator is None and "slot" not in picks[0].meta


def test_slot_layout_can_come_from_the_store_rules(tmp_path):
    (tmp_path / "layout.json").write_text(json.dumps(SHELF))
    _, picks, _ = _run_pick({"A": A, "B": B}, slots=None, slot_layout=tmp_path / "layout.json")
    assert picks[0].meta["slot"]["id"] == TARGET and picks[0].meta["slot"]["views"] == 2


def test_marks_to_points():
    m = marks_to_points([[10, 20, 1, 2]], [[30, 40, 3, 4, 1.5]])
    assert m.tolist() == [[10, 20, 1, 2, 0], [30, 40, 3, 4, 1.5]] and marks_to_points(None, None).shape == (0, 5)


def test_store_yaml_calibration_reaches_the_pipeline(tmp_path):
    """The `camera.calibration` block scripts/calibrate.py writes is loaded with the store YAML, so StoreEvents
    finds the calibrated camera (and the slot layout from `rules.slot_layout`) without anything passed in code."""
    import yaml as _yaml
    from bree.events.zones import load_store_config
    from bree.track.multicam import StoreEvents
    cfg = _yaml.safe_load((ROOT / "configs" / "store_gas_station_small.yaml").read_text())
    cfg["camera"]["calibration"] = {**A.to_dict(), "method": "pnp"}
    cfg["rules"] = {**cfg.get("rules", {}), "slot_layout": str(tmp_path / "layout.json")}
    (tmp_path / "layout.json").write_text(json.dumps(SHELF))
    (tmp_path / "s.yaml").write_text(_yaml.safe_dump(cfg))
    store = load_store_config(tmp_path / "s.yaml")
    assert store.calibration["method"] == "pnp"
    se = StoreEvents({"a": store})
    assert se.locator is not None and se.cameras["a"].f == pytest.approx(A.f)
