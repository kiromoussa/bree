"""Calibration, 3D slot and store-wide identity bench. SYNTHETIC geometry only: no footage, no rendering.

  python -m bree.calib.bench [--cameras <layout with cameras> --store <layout with fixtures and slots>]

Default layout: tests/fixtures/calib_layout.json, the recommended 45 camera layout in its store.

Three tables, all from the camera layout JSON and seeded random draws:
  calibration  every layout camera is calibrated from marks it could see (floor grid points and fixture
               corners, with click noise) and compared with its true pose.
  slots        a hand reaches every slot; two-view triangulation vs the one-view fallbacks, under the
               pixel-noise + calibration-noise model of bree/calib/slots.py.
  identity     shoppers walk the store under the layout's overhead cameras (2 and 4 of them); each camera
               hands out its own track ids. Open-world handoff vs the store-wide closed world, with and
               without (synthetic) appearance features.
Limits: fixtures block lines of sight, shoppers do not; bodies are points (foot, torso, wrist, elbow); the
lens has no distortion; appearance features are drawn, not extracted from pixels.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from bree.calib.camera import Camera, calibrate, from_layout
from bree.calib.slots import NoiseModel, SlotLocator, SlotMap, perturbed
from bree.events.observations import FrameObs, PersonObs
from bree.events.types import EventType
from bree.track import reid as R

# One file with the recommended 45 camera layout's cameras and the store's fixtures and slots (copied from
# bree/software/shared/layouts: recommended-47.json and baseline-47.json).
FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "calib_layout.json"
DOOR = (0.0, 6.2)                                   # just inside the entrance (layout x, z)
DOOR_ZONE = (-1.2, 1.2, 4.9, 6.5)                   # x0, x1, z0, z1: the floor strip inside the door
AISLES = (-7.8, -4.7, -2.1, 0.5, 3.5)               # x of the walkways between and beside the gondolas
FRONT_Z, BACK_Z = 2.4, -5.0                         # cross aisles


def blocked(boxes, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Does a solid fixture block the line from C to each point of P (stopping 3 cm short of it)?"""
    seg = P - C
    rng_ = np.linalg.norm(seg, axis=1)
    seg = seg * (1 - 0.03 / np.maximum(rng_, 0.03))[:, None]
    out = np.zeros(len(P), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in boxes:
            t1, t2 = (lo - C) / seg, (hi - C) / seg
            tn, tf = np.fmin(t1, t2).max(1), np.fmax(t1, t2).min(1)
            out |= (tn <= tf) & (tf >= 0) & (tn <= 1)
    return out


def sees(cam: Camera, boxes, P: np.ndarray, margin: float = 0.0) -> np.ndarray:
    px, z = cam.project(P)
    return cam.in_frame(px, z, margin) & ~blocked(boxes, cam.C, P)


def stat(v) -> dict:
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": len(v), "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4),
            "max": round(float(v.max()), 4)}


# ------------------------------------------------------------------ calibration


def candidate_marks(cam: Camera, sm: SlotMap, store: dict) -> np.ndarray:
    """(N, 5) [x_px, y_px, x_m, y_m, h_m] of every mark this camera could be given: a 0.5 m floor grid, the
    corners of the solid fixtures and the shelf edge in front of every 7th slot (all positions the layout
    file knows), in frame (50 px margin) and not hidden by a fixture."""
    w, d = store["width"], store["depth"]
    P = [(x, 0.0, z) for x in np.arange(-w / 2 + 0.25, w / 2, 0.5) for z in np.arange(-d / 2 + 0.25, d / 2, 0.5)
         if not any(lo[0] <= x <= hi[0] and lo[2] <= z <= hi[2] for lo, hi in sm.boxes)]
    for lo, hi in sm.boxes:
        P += [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    n = len(P)
    P = np.asarray(P + [tuple(f) for f in sm.front], float)
    ok = sees(cam, sm.boxes, P, margin=50.0)
    edge = sm.visible(cam, sm.front, margin_px=50.0)                # the shelf edge at every 7th slot
    edge[np.arange(len(edge)) % 7 != 0] = False
    ok[n:] = edge
    px, _ = cam.project(P[ok])
    return np.c_[px, P[ok][:, 0], P[ok][:, 2], P[ok][:, 1]]


def calibrated(cam: Camera, sm: SlotMap, store: dict, rng, n_marks: int = 12, click_px: float = 2.0,
               known_f: bool = True) -> tuple[Camera | None, dict, int]:
    marks = candidate_marks(cam, sm, store)
    if len(marks) > n_marks:
        marks = marks[rng.choice(len(marks), n_marks, replace=False)]
    marks = marks.copy()
    marks[:, :2] += rng.normal(0, click_px, (len(marks), 2))
    est, rep = calibrate(marks, cam.resolution, f=cam.f if known_f else None)
    return est, rep, len(marks)


def calibration_rows(layout_cams: list[dict], sm: SlotMap, store: dict, seed: int = 0, n_marks: int = 12) -> dict:
    out = {}
    for click_px in (1.0, 2.0, 5.0):
        for known_f in (True, False):
            rng = np.random.default_rng(seed)
            rms, pos, rot, fov, loo, n_ok, few = [], [], [], [], [], 0, []
            for c in layout_cams:
                true = from_layout(c)
                est, rep, n = calibrated(true, sm, store, rng, n_marks, click_px, known_f)
                if est is None:
                    few.append(f"{c['id']} ({n} marks)")
                    continue
                n_ok += 1
                rms.append(rep["rms_px"]); loo.append(rep["floor_loo_rms_m"])
                pos.append(float(np.linalg.norm(est.C - true.C)))
                rot.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(est.R @ true.R.T) - 1) / 2)))))
                fov.append(abs(est.hfov_deg - c["hfov"]) / c["hfov"] * 100)
            out[f"click_{click_px:g}px_{'lens_known' if known_f else 'focal_estimated'}"] = {
                "cameras": len(layout_cams), "calibrated": n_ok, "not_calibrated": few,
                "reprojection_rms_px": stat(rms), "position_error_m": stat(pos), "rotation_error_deg": stat(rot),
                "hfov_error_pct": stat(fov), "floor_leave_one_out_m": stat(loo)}
            print(f"[calib-bench] calibration click={click_px:g}px known_f={known_f}: {n_ok}/{len(layout_cams)} "
                  f"pos {stat(pos)} rot {stat(rot)}", flush=True)
    return {"what": "SYNTHETIC marks: each layout camera calibrated from up to %d marks it sees (floor grid, "
                    "fixture corners, shelf edges), Gaussian click noise" % n_marks, "seed": seed, "rows": out}


# ------------------------------------------------------------------ slots


def slot_rows(layout_cams: list[dict], sm: SlotMap, store: dict, seed: int = 0, tool_click_px: float = 2.0,
              only: str | None = None) -> dict:
    """One reach per slot. The arm comes from a shoulder 0.45 m out from the shelf face; at the moment that
    is measured the fingertips are 0 to 10 cm short of the slot's front centre (uniform), the wrist 0.12 m
    behind them and the elbow 0.26 m further back (forearm direction jittered by 10 degrees). Each camera that sees the slot reports wrist and elbow with pixel noise; the cameras the
    estimator uses are off by the calibration noise. Truth = the slot reached."""
    true = {c["id"]: from_layout(c) for c in layout_cams if c["kind"] != "overhead"}
    vis = np.stack([sm.visible(cam, sm.front) for cam in true.values()])           # (cameras, slots)
    names = list(true)
    n_views = vis.sum(0)
    coverage = {"slots": len(sm), "item_cameras": len(true), "seen_by_0": int((n_views == 0).sum()),
                "seen_by_1": int((n_views == 1).sum()), "seen_by_2_or_more": int((n_views >= 2).sum())}
    print(f"[calib-bench] slots: {coverage}", flush=True)
    by_fixture = defaultdict(list)
    for i, f in enumerate(sm.fixtures):
        by_fixture[f].append(i)
    settings = [("pixel 0.29 px, pointing 0.1 deg (the simulator's defaults)", NoiseModel.sim()),
                ("pixel 2 px, perfect calibration", NoiseModel(2.0, 0.0, 0.0)),
                ("pixel 5 px, perfect calibration", NoiseModel(5.0, 0.0, 0.0)),
                ("pixel 5 px, calibration 1 cm / 0.1 deg", NoiseModel(5.0, 0.01, 0.1)),
                ("pixel 5 px, calibration 2 cm / 0.2 deg (default model)", NoiseModel()),
                ("pixel 5 px, calibration 5 cm / 0.5 deg", NoiseModel(5.0, 0.05, 0.5)),
                ("pixel 10 px, calibration 2 cm / 0.2 deg", NoiseModel(10.0, 0.02, 0.2)),
                ("pixel 20 px, calibration 2 cm / 0.2 deg", NoiseModel(20.0, 0.02, 0.2)),
                (f"pixel 5 px, cameras calibrated by the tool ({tool_click_px:g} px clicks, 12 marks)", None)]
    out = {}
    for label, noise in settings:
        if only and only not in label:
            continue
        rng = np.random.default_rng(seed)
        if noise is None:
            used, skipped = {}, []
            for n, cam in true.items():
                est, _, _ = calibrated(cam, sm, store, rng, 12, tool_click_px, True)
                if est is None:
                    skipped.append(n)
                else:
                    used[n] = est
            noise = NoiseModel(5.0, 0.02, 0.2)
        else:
            used, skipped = {n: perturbed(cam, rng, noise) for n, cam in true.items()}, []
        loc = SlotLocator(used, sm, noise)
        tot, err3d, sig = Counter(), [], []
        for i in range(len(sm)):
            cams = [names[k] for k in np.flatnonzero(vis[:, i]) if names[k] in used]
            if not cams:
                continue
            n, F = sm.normal[i], sm.front[i]
            up = np.array([0.0, 1.0, 0.0])
            side = np.cross(up, n)
            shoulder = F + 0.45 * n + rng.uniform(-0.2, 0.2) * side
            shoulder[1] = min(max(F[1] + rng.uniform(0.0, 0.5), 0.9), 1.45)
            a = (F - shoulder) / np.linalg.norm(F - shoulder)
            wrist = F - (0.12 + rng.uniform(0.0, 0.10)) * a          # fingertips 0 to 10 cm short of the slot
            fa = a + math.radians(10) * rng.normal(size=3)
            elbow = wrist - 0.26 * fa / np.linalg.norm(fa)
            loc.buf.clear()
            first = cams[int(rng.integers(len(cams)))]                 # the camera whose engine raised the PICK
            one = SlotLocator({first: used[first]}, sm, noise)
            for c in cams:
                (wp, ep), _ = true[c].project(np.stack([wrist, elbow]))
                hands = {"right": (tuple(wp + rng.normal(0, noise.pixel_px, 2)), tuple(ep + rng.normal(0, noise.pixel_px, 2)))}
                loc.observe(c, 0.0, 1, hands)
                one.observe(c, 0.0, 1, hands)
            (ip,), _ = true[first].project(F[None])
            item_uv = tuple(ip + rng.normal(0, noise.pixel_px, 2))
            got = {"deployed": loc.locate(first, 1, "right", 0.0, 0.0),
                   "hand_1view": one.locate(first, 1, "right", 0.0, 0.0),
                   "item_1view": one.locate(first, 1, "right", 0.0, 0.0, item_uv=item_uv)}
            two = got["deployed"] is not None and got["deployed"]["views"] >= 2
            tot["slots_with_a_view"] += 1
            tot["slots_two_view"] += two
            for k, g in got.items():
                hit = g is not None and g["id"] == sm.ids[i]
                sku = g is not None and g["sku"] == sm.skus[i]
                tot[f"{k}_exact"] += hit
                tot[f"{k}_sku"] += sku
                if two:                                                 # same slots for the head-to-head
                    tot[f"{k}_exact_on_two_view_slots"] += hit
                    tot[f"{k}_sku_on_two_view_slots"] += sku
            if two:
                tot["plain_nearest_exact_on_two_view_slots"] += sm.nearest(got["deployed"]["xyz"])[0] == i
                err3d.append(float(np.linalg.norm(np.asarray(got["deployed"]["xyz"]) - F)))
                sig.append(got["deployed"]["sigma_m"])
        n1, n2 = max(tot["slots_with_a_view"], 1), max(tot["slots_two_view"], 1)
        row = {"slots_with_a_view": tot["slots_with_a_view"], "slots_two_view": tot["slots_two_view"],
               "cameras_not_calibrated": skipped,
               "two_view_exact": round(tot["deployed_exact_on_two_view_slots"] / n2, 4),
               "two_view_right_sku": round(tot["deployed_sku_on_two_view_slots"] / n2, 4),
               "two_view_plain_nearest_exact": round(tot["plain_nearest_exact_on_two_view_slots"] / n2, 4),
               "one_view_hand_exact_same_slots": round(tot["hand_1view_exact_on_two_view_slots"] / n2, 4),
               "one_view_hand_right_sku_same_slots": round(tot["hand_1view_sku_on_two_view_slots"] / n2, 4),
               "one_view_item_exact_same_slots": round(tot["item_1view_exact_on_two_view_slots"] / n2, 4),
               "one_view_item_right_sku_same_slots": round(tot["item_1view_sku_on_two_view_slots"] / n2, 4),
               "deployed_exact_all_seen_slots": round(tot["deployed_exact"] / n1, 4),
               "deployed_right_sku_all_seen_slots": round(tot["deployed_sku"] / n1, 4),
               "one_view_hand_exact_all_seen_slots": round(tot["hand_1view_exact"] / n1, 4),
               "one_view_item_exact_all_seen_slots": round(tot["item_1view_exact"] / n1, 4),
               "two_view_error_m": stat(err3d), "two_view_predicted_sigma_m": stat(sig)}
        out[label] = row
        print(f"[calib-bench] slots {label}: {row}", flush=True)
    return {"what": "SYNTHETIC reaches, one per slot; item cameras of the layout; fixtures block, bodies do not",
            "seed": seed, "coverage": coverage, "rows": out}


# ------------------------------------------------------------------ identity


def _route(p, q):
    """Walkable polyline between two spots (layout x, z): along an aisle to a cross aisle, across, and in."""
    if abs(p[0] - q[0]) < 1e-6:
        return [q]
    front = p[1] >= FRONT_Z or q[1] >= FRONT_Z
    zc = FRONT_Z if front or abs(p[1] - FRONT_Z) + abs(q[1] - FRONT_Z) <= abs(p[1] - BACK_Z) + abs(q[1] - BACK_Z) else BACK_Z
    return [(p[0], zc), (q[0], zc), q]


def _spot(rng):
    k = rng.random()
    if k < 0.6:
        return (AISLES[int(rng.integers(len(AISLES)))], float(rng.uniform(-3.8, 1.3)))
    if k < 0.75:
        return (AISLES[int(rng.integers(len(AISLES)))], -5.2)
    if k < 0.9:
        return (float(rng.uniform(-8.0, 3.8)), float(rng.uniform(2.6, 5.4)))
    return (3.9, 3.8)                                                   # in front of the register


def _look(rng, like=None):
    """Same synthetic appearance model as bree.eval.reid_bench.synthetic_store_rows (side-view quality)."""
    common = np.ones(32) / np.sqrt(32)
    e = rng.normal(size=32)
    e = common + 0.8 * e / np.linalg.norm(e) if like is None else like[0] + 0.5 * e / np.linalg.norm(e)
    return e / np.linalg.norm(e), (rng.integers(0, 52, 3) if like is None else like[1])


def _feat(rng, lk, t):
    e = lk[0] + 0.06 * rng.normal(size=32)
    c = np.full((3, 52), 0.002) + 0.004 * rng.random((3, 52))
    c[np.arange(3), lk[1]] = 1.0
    return R.ReidFeatures(t, (e / np.linalg.norm(e)).astype(np.float32), (c / c.sum(1, keepdims=True)).astype(np.float32))


def episode(rng, cams: dict[str, Camera], boxes, fps: float = 10.0, foot_px: float = 4.0, speed: float = 1.2,
            p_lookalike: float = 0.25, keep_id_s: float = 0.5) -> tuple[list, dict]:
    """One scripted visit of 2 to 5 shoppers. Returns (frames, exits): frames = [(t, {camera: [(track id,
    shopper, bbox, look)]})], exits = {shopper: time they walked out}. A camera detects a shopper when feet
    and torso are in frame and no fixture hides them. A camera's track id survives `keep_id_s` unseen, then
    the shopper gets a new one. Every 4 to 10 s one or two shoppers are missed by every camera for 0.5 to
    10 s (half the time two at once, and half of those trade places meanwhile)."""
    dt = 1.0 / fps
    ag = []
    for i in range(int(rng.integers(2, 6))):
        lk = _look(rng, ag[int(rng.integers(i))]["look"] if i and rng.random() < p_lookalike else None)
        ag.append({"id": i, "t_in": float(rng.uniform(6, 25)), "pos": np.array(DOOR), "path": [], "wait": 0.0,
                   "legs": int(rng.integers(2, 5)), "hidden": 0.0, "gone": False, "leaving": False, "look": lk})
    frames, exits, t, next_occ = [], {}, 0.0, float(rng.uniform(8, 14))
    last = {}                                                          # (camera, shopper) -> (track id, t last seen)
    next_tid = {c: 1 for c in cams}
    while not all(a["gone"] for a in ag) and t < 240:
        inside = [a for a in ag if t >= a["t_in"] and not a["gone"]]
        free = [a for a in inside if a["hidden"] <= 0 and not a["leaving"] and a["pos"][1] < 4.5]
        if t >= next_occ and free:
            next_occ = t + float(rng.uniform(4, 10))
            pick = list(rng.choice(len(free), size=min(len(free), 1 + int(rng.random() < 0.5)), replace=False))
            gap = float(rng.uniform(0.5, 10))
            for k in pick:
                free[k]["hidden"] = gap
            if len(pick) == 2 and rng.random() < 0.5:                  # trade places while unseen
                a, b = free[pick[0]], free[pick[1]]
                pa, pb = tuple(a["pos"]), tuple(b["pos"])
                a["path"], b["path"], a["wait"], b["wait"] = _route(pa, pb), _route(pb, pa), 0.0, 0.0
        for a in inside:
            if a["wait"] > 0:
                a["wait"] -= dt
            else:
                if not a["path"]:
                    if a["leaving"]:
                        a["gone"], exits[a["id"]] = True, t
                        continue
                    a["legs"] -= 1
                    a["leaving"] = a["legs"] < 0
                    a["path"] = _route(tuple(a["pos"]), DOOR if a["leaving"] else _spot(rng))
                step = np.asarray(a["path"][0]) - a["pos"]
                dist = float(np.linalg.norm(step))
                if dist > speed * dt:
                    a["pos"] = a["pos"] + step / dist * speed * dt
                else:
                    a["pos"] = np.asarray(a["path"].pop(0), float)
                    if not a["path"] and not a["leaving"]:
                        a["wait"] = float(rng.uniform(2, 6))
            if a["hidden"] > 0:
                a["hidden"] -= dt
        vis = [a for a in inside if not a["gone"] and a["hidden"] <= 0]
        row = {c: [] for c in cams}
        if vis:
            feet = np.array([(a["pos"][0], 0.0, a["pos"][1]) for a in vis])
            for c, cam in cams.items():
                ok = sees(cam, boxes, feet) & sees(cam, boxes, feet + [0, 1.2, 0])
                fp, _ = cam.project(feet)
                hp, _ = cam.project(feet + [0, 1.7, 0])
                for k, a in enumerate(vis):
                    if not ok[k]:
                        continue
                    tid, seen = last.get((c, a["id"]), (None, -1e9))
                    if t - seen > keep_id_s:
                        tid, next_tid[c] = next_tid[c], next_tid[c] + 1
                    last[(c, a["id"])] = (tid, t)
                    u, v = fp[k] + rng.normal(0, foot_px, 2)
                    h = max(float(np.linalg.norm(fp[k] - hp[k])), 20.0)
                    row[c].append((tid, a["id"], (u - 0.15 * h, v - h, u + 0.15 * h, v), a["look"]))
        frames.append((t, row))
        t += dt
    return frames, exits


def identity_rows(layout_cams: list[dict], sm: SlotMap, store: dict, names: tuple[str, ...], episodes: int = 200,
                  seed: int = 0, click_px: float = 2.0, foot_px: float = 4.0) -> dict:
    """The whole identity chain as the pipeline runs it (StoreEvents: one EventEngine per camera + the
    store-wide identity), on scripted tracks. Each camera is calibrated by the tool from 12 noisy marks; a
    camera that sees the door strip gets it as its `exit` zone. An identity is what the ledger would see:
    the global id on the events of a track.
      handoffs_live  a new track of a shopper some other camera is tracking at that moment
      handoffs_gap   a new track of a shopper nobody is tracking at that moment (occlusion or blind spot)
      ..._ok         it joined the identity that shopper last had
      false_merges   a track joined an identity whose latest track was another shopper (silent = that
                     identity was not marked uncertain)
      splits         identities per shopper beyond the first
      clean_visits   the shopper has one identity and nobody else ever shared it
      exits_ok       EXIT events for an identity whose shopper had really left; exits_wrong = the rest"""
    from bree.events.types import Catalog
    from bree.events.zones import StoreConfig, Zone
    from bree.track.multicam import StoreEvents
    true = {c["id"]: from_layout(c) for c in layout_cams if c["id"] in names}
    rng0 = np.random.default_rng(seed)
    est = {n: calibrated(cam, sm, store, rng0, 12, click_px, True)[0] for n, cam in true.items()}
    x0, x1, z0, z1 = DOOR_ZONE
    door_pts = np.array([(x, 0.0, z) for x in np.linspace(x0, x1, 7) for z in np.linspace(z0, z1 - 0.2, 5)])
    stores, door_cams = {}, []
    for n, cam in true.items():
        zones = []
        if (sees(cam, sm.boxes, door_pts) & sees(cam, sm.boxes, door_pts + [0, 1.2, 0])).mean() >= 0.5:
            px, _ = est[n].project([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)])
            zones.append(Zone("door", "exit", px))
            door_cams.append(n)
        stores[n] = StoreConfig(n, n, cam.resolution, 10.0, zones, {}, Catalog({}))
    rng = np.random.default_rng(seed + 1)
    eps = [(*episode(rng, true, sm.boxes, foot_px=foot_px), int(rng.integers(1 << 30))) for _ in range(episodes)]
    variants = {"off": {}, "reid": {"reid": True}, "closed_world": {"closed_world": True},
                "closed_world_reid": {"closed_world": True, "reid": True}}
    out = {}
    for name, rules in variants.items():
        tot = Counter()
        for frames, exits, fseed in eps:                               # same episodes for every variant
            frng = np.random.default_rng(fseed)
            se = StoreEvents({n: replace(s, rules=dict(rules)) for n, s in stores.items()}, {"calibration": est})
            mc = se.identity
            gt, owner_of, known = {}, {}, set()                        # (cam, tid) -> shopper; gid -> shopper of its latest track
            events = []
            for fi, (t, row) in enumerate(frames):
                for c, obs in row.items():
                    persons = [PersonObs(tid, box, 0.9, reid=_feat(frng, lk, t) if rules.get("reid") else None)
                               for tid, _, box, lk in obs]
                    events += se.update(c, FrameObs(fi, t, persons, []))
                    eng = se.engines[c]
                    for tid, who, _, _ in obs:
                        key = (c, tid)
                        gid = mc.local_to_global.get((c, eng.alias.get(tid, tid)))
                        if key in gt or gid is None:
                            continue
                        gt[key] = who
                        tot["tracks"] += 1
                        kind = "live" if any(w == who for c2, o2 in row.items() if c2 != c for _, w, _, _ in o2) else "gap"
                        if who in known:
                            tot[f"handoffs_{kind}"] += 1
                        if gid in owner_of:
                            ok = owner_of[gid] == who
                            tot[f"handoffs_{kind}_ok"] += ok and who in known
                            tot["false_merges"] += not ok
                            tot["false_merges_silent"] += (not ok) and mc.people[gid].uncertain_until < t
                        known.add(who)
                        owner_of[gid] = who
            events += se.flush()
            for ev in events:
                if ev.type == EventType.EXIT:
                    who = owner_of.get(ev.person_id)
                    tot["exits_ok" if who in exits and exits[who] <= ev.t + 1.0 else "exits_wrong"] += 1
            ident = {key: mc.local_to_global[(key[0], se.engines[key[0]].alias.get(key[1], key[1]))] for key in gt}
            owners = defaultdict(set)
            for key, gid in ident.items():
                owners[gid].add(gt[key])
            for who in set(gt.values()):
                mine = {gid for key, gid in ident.items() if gt[key] == who}
                tot["shoppers"] += 1
                tot["splits"] += len(mine) - 1
                tot["clean_visits"] += len(mine) == 1 and owners[next(iter(mine))] == {who}
                tot["visits_uncertain"] += any(mc.people[g].uncertain_until >= mc.people[g].t_last for g in mine)
            tot["shoppers_never_seen"] += len(exits) - len(set(gt.values()))
            tot.update({f"cw_{k}": v for k, v in mc.cw_stats.items()})
            tot["left_inside_at_end"] += mc.occupancy() if rules.get("closed_world") else 0
        ho = tot["handoffs_live"] + tot["handoffs_gap"]
        row = {k: tot[k] for k in ("shoppers", "tracks", "handoffs_live", "handoffs_live_ok", "handoffs_gap",
                                   "handoffs_gap_ok", "false_merges", "false_merges_silent", "splits", "clean_visits",
                                   "visits_uncertain", "shoppers_never_seen", "exits_ok", "exits_wrong", "left_inside_at_end")}
        row["handoff_success"] = round((tot["handoffs_live_ok"] + tot["handoffs_gap_ok"]) / max(ho, 1), 4)
        row["handoff_live_success"] = round(tot["handoffs_live_ok"] / max(tot["handoffs_live"], 1), 4)
        row["handoff_gap_success"] = round(tot["handoffs_gap_ok"] / max(tot["handoffs_gap"], 1), 4)
        row["clean_visit_rate"] = round(tot["clean_visits"] / max(tot["shoppers"], 1), 4)
        row["closed_world"] = {k[3:]: v for k, v in tot.items() if k.startswith("cw_")}
        out[name] = row
        print(f"[calib-bench] identity {len(names)} cameras {name}: {row}", flush=True)
    return {"what": "SYNTHETIC shoppers under the layout's overhead cameras; identity logic only",
            "cameras": list(names), "door_cameras": door_cams, "episodes": episodes, "seed": seed,
            "click_px": click_px, "foot_px": foot_px, "rows": out}


def run(cameras_path: Path = FIXTURE, store_path: Path = FIXTURE, episodes: int = 200, seed: int = 0, parts=("calibration", "slots", "identity")) -> dict:
    cams = json.loads(Path(cameras_path).read_text())["cameras"]
    layout = json.loads(Path(store_path).read_text())
    sm, store = SlotMap(layout), layout["store"]
    t0 = time.perf_counter()
    out = {"cameras_layout": str(cameras_path), "store_layout": str(store_path), "synthetic": True}
    if "calibration" in parts:
        out["calibration"] = calibration_rows(cams, sm, store, seed)
    if "slots" in parts:
        out["slots"] = slot_rows(cams, sm, store, seed)
    if "identity" in parts:
        overhead = [c["id"] for c in cams if c["kind"] == "overhead"]
        out["identity_2_cameras"] = identity_rows(cams, sm, store, tuple(overhead[:2]), episodes, seed)
        out["identity_4_cameras"] = identity_rows(cams, sm, store, tuple(overhead[:4]), episodes, seed)
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cameras", default=str(FIXTURE), help="layout JSON with the cameras")
    ap.add_argument("--store", default=str(FIXTURE), help="layout JSON with fixtures and slots")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="calibration,slots,identity")
    ap.add_argument("--out", default="results/calib_bench.json")
    a = ap.parse_args(argv)
    res = run(Path(a.cameras), Path(a.store), a.episodes, a.seed, tuple(a.parts.split(",")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[calib-bench] wrote {a.out} ({res['wall_s']} s)")


if __name__ == "__main__":
    main()
