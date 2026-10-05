"""Isaac Sim kit output (bree/software/isaac-sim) -> what the BREE pipeline consumes.

Per camera: one MP4 at the sim's written frame rate and one store YAML whose zones are the
layout's fixtures projected through that camera's Replicator `camera_params` into pixel polygons.
Plus a POS feed built from the ground-truth "paid" events.

Layout coordinates are three.js (metres, y up, +z toward the front door); camera_params are in the
USD stage (z up). (x, y, z)_three -> (x, -z, y)_usd, as in the kit's bree_layout.py.

NOT RUN ON REAL ISAAC OUTPUT YET. The folder walk and the matrix convention are tolerant on
purpose (see `find_cameras`, `load_camera`), and tested on the synthetic fixture only.
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import yaml

from bree.sim.isaac.convert import _video_writer, hull, project

ROOT = Path(__file__).resolve().parents[3]
BASE_STORE = ROOT / "configs" / "store_gas_station_small.yaml"
# layout fixture type -> pipeline zone kind. backbar / lottery / wall get no zone (customers do not pick there).
ZONE_KIND = {"gondola": "shelf", "wall_shelf": "shelf", "cooler_door": "cooler", "walk_in": "cooler",
             "counter": "register", "door": "exit"}
REGISTER_DEPTH_M = 1.2    # floor strip on the customer side of the counter
EXIT_DEPTH_M = 1.5        # floor strip inside the door
TERMINAL = "pos_1"


# ------------------------------------------------------------------ geometry
def to_usd(p):
    return (p[0], -p[2], p[1])


def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def local_to_world(f, local):
    o = rot_y(local, f["rotationY"])
    return tuple(f["position"][i] + o[i] for i in range(3))


def prim_name(s: str) -> str:
    n = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return n if n and not n[0].isdigit() else "_" + n


def camera_forward(cam) -> tuple:
    """three.js forward of a layout camera: Ry(yaw) Rx(pitch) (0, 0, -1)."""
    cp, sp = math.cos(cam["pitch"]), math.sin(cam["pitch"])
    return rot_y((0.0, sp, -cp), cam["yaw"])


def camera_params_from_layout(cam, near: float = 0.03, far: float = 60.0) -> dict:
    """Replicator-style camera_params (row-vector matrices, flattened) for a layout camera. Used by the
    synthetic fixture, and as the fallback when a camera folder has no camera_params file."""
    w, h = cam["resolution"]
    fx = (w / 2) / math.tan(math.radians(cam["hfov"]) / 2)
    cy, sy, cp, sp = math.cos(cam["yaw"]), math.sin(cam["yaw"]), math.cos(cam["pitch"]), math.sin(cam["pitch"])
    r_three = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    r_usd = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ r_three          # camera axes in the USD world
    view = np.eye(4)
    view[:3, :3] = r_usd.T
    view[:3, 3] = -r_usd.T @ np.array(to_usd(cam["position"]))
    proj = np.array([[2 * fx / w, 0, 0, 0], [0, 2 * fx / h, 0, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    return {"renderProductResolution": [w, h], "cameraViewTransform": view.T.reshape(-1).tolist(),
            "cameraProjection": proj.T.reshape(-1).tolist(), "cameraNearFar": [near, far]}


def load_camera(params: dict, cam: dict) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    """(view, proj, w, h, note). Self-check: a point 3 m down the layout camera's optical axis must land
    on the image centre. If the transposed matrices do that better they are used (column-vector files)."""
    w, h = (int(v) for v in params["renderProductResolution"])
    view = np.asarray(params["cameraViewTransform"], float).reshape(4, 4)
    proj = np.asarray(params["cameraProjection"], float).reshape(4, 4)
    f = camera_forward(cam)
    probe = to_usd([cam["position"][i] + 3.0 * f[i] for i in range(3)])
    best = None
    for name, (v, p) in (("row-vector", (view, proj)), ("transposed", (view.T, proj.T))):
        px, front = project([probe], v, p, w, h)
        err = float(np.hypot(px[0][0] - w / 2, px[0][1] - h / 2)) if front[0] else float("inf")
        if best is None or err < best[0]:
            best = (err, v, p, name)
    err, view, proj, name = best
    note = f"{name}, optical-axis check {err:.1f} px"
    if err > 5:
        note += " WARNING: camera_params disagree with the layout camera; zones for this camera are suspect"
    return view, proj, w, h, note


def zones_3d(layout) -> list[dict]:
    """One zone per usable fixture, as three.js world points. Shelves / coolers (only those holding slots):
    the hull of the fixture's slot boxes, i.e. where product sits, not the whole carcass (the event engine
    only counts an item as in a hand once it is outside the merch zone, so the zone must be the stock).
    Counter: the floor strip on its customer side (+z local). Door: the floor strip inside it."""
    slots: dict[str, list] = {}
    for s in layout["slots"]:
        slots.setdefault(s["fixtureId"], []).append(s)
    out = []
    for f in layout["fixtures"]:
        kind = ZONE_KIND.get(f["type"])
        if kind is None:
            continue
        w, hh, d = f["size"]
        if kind in ("shelf", "cooler"):
            if f["id"] not in slots:
                continue
            pts = []
            for s in slots[f["id"]]:       # slot size is in the fixture's local axes, its position is a world centre
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            o = rot_y((sx * s["size"][0] / 2, sy * s["size"][1] / 2, sz * s["size"][2] / 2), f["rotationY"])
                            pts.append(tuple(s["position"][i] + o[i] for i in range(3)))
        else:
            if kind == "register":
                z0, z1 = d / 2, d / 2 + REGISTER_DEPTH_M
            else:       # inward = the side of the door plane that faces the store centre
                n = rot_y((0, 0, 1), f["rotationY"])
                sign = -1.0 if n[0] * f["position"][0] + n[2] * f["position"][2] > 0 else 1.0
                z0, z1 = sorted((-0.2 * sign, EXIT_DEPTH_M * sign))
            pts = [local_to_world(f, (x, -f["position"][1], z)) for x in (-w / 2, w / 2) for z in (z0, z1)]
        out.append({"name": f["id"], "kind": kind, "fixture_type": f["type"], "points": pts})
    return out


def project_zone(points, view, proj, w, h):
    """Image polygon of a zone, clipped to the frame, and its area in px^2. None if it is not in view.
    Points behind the camera are left out and the rest still make the zone: a camera mounted beside the
    fixture it watches (the recommended layout's rail cameras look along the aisle) always has part of that
    fixture behind it. ponytail: not a true clip at the near plane, so the polygon stops at the last slot in
    front of the camera; with a slot every few centimetres the missing strip is outside the frame anyway."""
    import cv2
    px, front = project([to_usd(p) for p in points], view, proj, w, h)
    px = [p for p, ok in zip(px, front) if ok]
    if len(px) < 3:
        return None
    poly = np.array(hull(px), np.float32)
    if len(poly) < 3:
        return None
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    area, inter = cv2.intersectConvexConvex(poly, frame)
    if area < 100 or inter is None:
        return None
    return [[int(round(x)), int(round(y))] for x, y in inter.reshape(-1, 2)], float(area)


def floor_points(layout, view, proj, w, h, step: float = 0.5, keep: int = 12) -> list[list[float]]:
    """[x_px, y_px, x_m, y_m] marks for the multi-camera floor plan: a floor grid, the points this camera
    sees, spread out. Floor plan metres = layout (x, z)."""
    sw, sd = layout["store"]["width"], layout["store"]["depth"]
    grid = [(x, z) for x in np.arange(-sw / 2 + 0.25, sw / 2, step) for z in np.arange(-sd / 2 + 0.25, sd / 2, step)]
    px, front = project([to_usd((x, 0.0, z)) for x, z in grid], view, proj, w, h)
    seen = [(float(u), float(v), float(x), float(z)) for (u, v), ok, (x, z) in zip(px, front, grid)
            if ok and 0 <= u < w and 0 <= v < h]
    if len(seen) > keep:
        seen = [seen[i] for i in np.linspace(0, len(seen) - 1, keep).round().astype(int)]
    return [[round(u, 1), round(v, 1), round(x, 3), round(z, 3)] for u, v, x, z in seen]


# ------------------------------------------------------------------ Isaac folders
def find_cameras(sim_out: Path, layout) -> dict[str, dict]:
    """{camera id: {"frames": [png paths in order], "params": dict | None}}. Any nesting works
    (BasicWriter writes <cam>/rgb_0001.png, IRA's writer may add batch_NN/ira_output/ and an rgb/ level):
    a frame belongs to the nearest parent folder named like a layout camera id or its USD prim name."""
    names = {}
    for c in layout["cameras"]:
        names[c["id"]] = names[prim_name(c["id"])] = c["id"]
    cams: dict[str, dict] = {}
    for png in sim_out.rglob("rgb_*.png"):
        if "static" in png.relative_to(sim_out).parts:      # still frames without people
            continue
        cam_dir = next((p for p in png.parents if p.name in names), None)
        if cam_dir is None:
            continue
        c = cams.setdefault(names[cam_dir.name], {"frames": [], "dir": cam_dir})
        c["frames"].append(png)
    for c in cams.values():
        c["frames"].sort(key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        pfile = next(iter(sorted(c["dir"].rglob("camera_params_*.json"))), None)
        c["params"] = json.loads(pfile.read_text()) if pfile else None
    return cams


def load_events(sim_out: Path) -> list[dict]:
    f = sim_out / "events.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def infer_fps(sim_out: Path, events: list[dict], n_frames: int, default: float = 10.0) -> tuple[float, str]:
    """Written frames / simulated seconds. Sim fps comes from the events (frame / t), the length from
    IRA's config.yaml (simulation_length frames, or simulation_duration seconds in 6.1)."""
    conf = next(iter(sorted(sim_out.rglob("config.yaml"))), None)
    if conf is not None and n_frames:
        ira = (yaml.safe_load(conf.read_text()) or {}).get("isaacsim.replicator.agent", {})
        dur = ira.get("simulation_duration")
        length = (ira.get("global") or {}).get("simulation_length")
        ev = next((e for e in events if e.get("t") and e.get("frame")), None)
        if dur is None and length and ev:
            dur = length / round(ev["frame"] / ev["t"])      # event frame = round(t * sim fps)
        if dur:
            return round(n_frames / dur, 3), f"inferred: {n_frames} frames over {dur:.1f} s"
    return default, f"assumed {default} (no config.yaml + events to infer it from; pass --fps)"


# ------------------------------------------------------------------ adapter
def adapt(sim_out, layout_path, work, fps: float | None = None, zone_owner: str = "best") -> dict:
    """Write <work>/<cam>.mp4 and <work>/<cam>.store.yaml for every camera with frames.
    zone_owner: "best" gives each fixture zone to the one camera that sees it largest (the pipeline
    does not de-duplicate a pick seen by two cameras), with the door and register zones going to a
    people camera when one sees them; "all" gives every zone to every camera that sees it."""
    import cv2
    sim_out, work = Path(sim_out), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    layout = json.loads(Path(layout_path).read_text())
    base = yaml.safe_load(BASE_STORE.read_text())
    events = load_events(sim_out)
    found = find_cameras(sim_out, layout)
    if not found:
        raise SystemExit(f"no rgb_*.png under a folder named like a layout camera in {sim_out}")
    by_id = {c["id"]: c for c in layout["cameras"]}
    fps_note = "given"
    if fps is None:
        fps, fps_note = infer_fps(sim_out, events, max(len(c["frames"]) for c in found.values()))
    zones3d = zones_3d(layout)
    cams, seen = {}, {}                       # seen: zone name -> [(area, camera id, polygon)]
    for cid, c in sorted(found.items()):
        params = c["params"] or camera_params_from_layout(by_id[cid])
        view, proj, w, h, note = load_camera(params, by_id[cid])
        if c["params"] is None:
            note += "; no camera_params file, matrices rebuilt from the layout"
        cams[cid] = {"view": view, "proj": proj, "w": w, "h": h, "note": note, "frames": c["frames"], "zones": []}
        for z in zones3d:
            hit = project_zone(z["points"], view, proj, w, h)
            if hit:
                seen.setdefault(z["name"], []).append((hit[1], cid, hit[0]))
    kinds = {z["name"]: z["kind"] for z in zones3d}
    item_cam = lambda cid: by_id[cid]["kind"] in ("shelf", "cooler")     # noqa: E731
    for name, views in seen.items():
        if zone_owner == "best" and kinds[name] not in ("shelf", "cooler"):
            # the door and the register belong to a camera that watches people (overhead, entrance, checkout)
            # when one sees them: an item camera is aimed at a shelf and, as a node, only sends frames on a trigger
            views = [v for v in views if not item_cam(v[1])] or views
        for area, cid, poly in (sorted(views, reverse=True)[:1] if zone_owner == "best" else views):
            cams[cid]["zones"].append({"name": name, "kind": kinds[name], "polygon": poly})
    catalog = {s["id"]: {"category": s["id"], "price": s["price"]} for s in layout["skus"]}
    out = {"fps": fps, "fps_note": fps_note, "cameras": [], "zone_owner": zone_owner,
           "zones_unseen": sorted(set(kinds) - set(seen))}
    for cid, c in cams.items():
        video = work / f"{cid}.mp4"
        vw = _video_writer(video, fps, c["w"], c["h"])
        for png in c["frames"]:
            img = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (c["h"], c["w"]):
                img = np.zeros((c["h"], c["w"], 3), np.uint8) if img is None else cv2.resize(img, (c["w"], c["h"]))
            vw.write(img)
        vw.release()
        names = {z["name"] for z in c["zones"]}
        reg = next((z["name"] for z in c["zones"] if z["kind"] == "register"), None)
        store = {"store": {"name": f"Isaac Sim layout ({cid})"},
                 "camera": {"id": cid, "resolution": [c["w"], c["h"]], "fps": fps,
                            "floor_points": floor_points(layout, c["view"], c["proj"], c["w"], c["h"])},
                 "zones": c["zones"], "terminals": {TERMINAL: reg} if reg else {},
                 # a detector trained on these renders outputs the SKU id; one ledger category per SKU
                 "product_classes": {s: s for s in catalog}, "catalog": catalog,
                 "rules": base.get("rules", {}), "ledger": base.get("ledger", {})}
        ypath = work / f"{cid}.store.yaml"
        ypath.write_text(yaml.safe_dump(store, sort_keys=False, default_flow_style=None))
        out["cameras"].append({"id": cid, "kind": by_id[cid]["kind"], "video": str(video), "store": str(ypath),
                               "frames": len(c["frames"]), "zones": sorted(names), "camera_check": c["note"],
                               "floor_points": len(store["camera"]["floor_points"])})
    out["duration_s"] = round(max(c["frames"] for c in out["cameras"]) / fps, 2)
    (work / "adapter.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ POS feed
def pos_feed(events: list[dict], skus: list[str], delay_s: float = 1.0, delay_sd_s: float = 0.5,
             dropout: float = 0.0, sku_noise: float = 0.0, seed: int = 0) -> tuple[list[dict], dict]:
    """Ground-truth "paid" events -> receipts in the pipeline's payment format (`t` = stream seconds).
    One receipt per shopper, stamped when the last item was paid plus a delay (normal, floored at 0).
    dropout: chance a whole receipt never arrives. sku_noise: chance a line is rung up as another SKU."""
    rng = random.Random(seed)
    by_shopper: dict[str, list[dict]] = {}
    for e in events:
        if e.get("outcome") == "paid":
            by_shopper.setdefault(e["shopper"], []).append(e)
    receipts, dropped, swapped = [], [], []
    for n, (shopper, evs) in enumerate(sorted(by_shopper.items(), key=lambda kv: max(e["tResolved"] for e in kv[1])), 1):
        txn = f"SIM{n:04d}"
        t = max(e["tResolved"] for e in evs) + max(0.0, rng.gauss(delay_s, delay_sd_s))
        lost = rng.random() < dropout
        items: dict[str, int] = {}
        for e in evs:
            sku = e["skuId"]
            if len(skus) > 1 and rng.random() < sku_noise:
                sku = rng.choice([s for s in skus if s != e["skuId"]])
                swapped.append({"txn_id": txn, "shopper": shopper, "true": e["skuId"], "rung_as": sku})
            items[sku] = items.get(sku, 0) + 1
        if lost:
            dropped.append({"txn_id": txn, "shopper": shopper})
            continue
        receipts.append({"t": round(t, 2), "terminal": TERMINAL, "txn_id": txn,
                         "items": [{"sku": s, "qty": q} for s, q in items.items()]})
    receipts.sort(key=lambda r: r["t"])
    return receipts, {"receipts": len(receipts), "dropped": dropped, "sku_swapped": swapped,
                      "delay_s": delay_s, "delay_sd_s": delay_sd_s, "dropout": dropout, "sku_noise": sku_noise, "seed": seed}
