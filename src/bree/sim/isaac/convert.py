"""Isaac Sim recordings -> clips that `bree.eval.toy_eval` can score, plus label overlays and checks.

Needs numpy, and OpenCV for video. No Isaac Sim. `bbox_rows` and `split_skeletons` are also
imported by gen_cstore.py (inside Isaac Sim) to turn annotator output into JSON lines.

    python -m bree.sim.isaac.convert convert RAW_DIR OUT_DIR [--delete-raw] [--watch]
    python -m bree.sim.isaac.convert overlay OUT_DIR --n 5 --dest OUT_DIR/overlays
    python -m bree.sim.isaac.convert check OUT_DIR

One clip = one camera of one episode. Per clip, in OUT_DIR/<split>/:
    isaac_<episode>_<cam>.mp4             RGB video (fps_out)
    isaac_<episode>_<cam>.truth.json      bree.sim.toy_render truth schema (+ events, camera, split, ...)
    isaac_<episode>_<cam>.payments.jsonl  POS receipts, same format as the toy clips
    isaac_<episode>_<cam>.store.yaml      store config with zones projected into this camera
    isaac_<episode>_<cam>.frames.jsonl    per frame: person boxes + track ids + COCO-17 keypoints,
                                          product boxes + ids + categories + holder, events
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
          "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
          "left_ankle", "right_ankle"]
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
              (14, 16), (0, 1), (0, 2)]
# COCO keypoint -> skeleton joint leaf names, first found wins. Isaac/People characters (Reallusion naming,
# checked on F_Business_02.usd from the 4.5 asset pack) first, then the Biped_Setup naming
# (checked on push_button.skelanim.usd). No nose or ear joints exist: nose = midpoint of the eyes
# (or the Head joint), ears are left unlabeled (v = 0).
JOINTS = {
    "left_eye": ["L_Eye"], "right_eye": ["R_Eye"],
    "left_shoulder": ["L_Upperarm", "L_UpArm"], "right_shoulder": ["R_Upperarm", "R_UpArm"],
    "left_elbow": ["L_Forearm", "L_LoArm"], "right_elbow": ["R_Forearm", "R_LoArm"],
    "left_wrist": ["L_Hand", "L_Wrist"], "right_wrist": ["R_Hand", "R_Wrist"],
    "left_hip": ["L_Thigh", "L_UpLeg"], "right_hip": ["R_Thigh", "R_UpLeg"],
    "left_knee": ["L_Calf", "L_LoLeg"], "right_knee": ["R_Calf", "R_LoLeg"],
    "left_ankle": ["L_Foot", "L_Ankle"], "right_ankle": ["R_Foot", "R_Ankle"],
}
KEEP_JOINTS = {j for c in JOINTS.values() for j in c} | {"Head"}
PRODUCT_ROOT = "/World/Store/Products/"
CHAR_ROOT = "/World/Characters/"
EVENT_TRUTH = {"pick": "picked", "put_back": "put_back", "conceal": "concealed"}


# ------------------------------------------------------------------ annotator output -> JSON rows
def bbox_rows(data) -> list[dict]:
    """Replicator `bounding_box_2d_tight` output -> [{label, prim, box[x0,y0,x1,y1], occ}]."""
    if data is None or len(data.get("data", [])) == 0:
        return []
    info = data["info"]
    labels = {str(k): v for k, v in info.get("idToLabels", {}).items()}
    paths = list(info.get("primPaths", []))
    rows = []
    for i, r in enumerate(data["data"]):
        lab = labels.get(str(int(r["semanticId"])), "")
        if isinstance(lab, dict):
            lab = lab.get("class", next(iter(lab.values()), ""))
        rows.append({"label": str(lab).split(",")[0], "prim": str(paths[i]) if i < len(paths) else "",
                     "box": [int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"])],
                     "occ": round(float(r["occlusionRatio"]), 3) if "occlusionRatio" in r.dtype.names else None})
    return rows


def _joint_names(entry) -> list[str]:
    if isinstance(entry, (list, tuple, np.ndarray)):
        return [str(j).split("/")[-1] for j in entry]
    s = str(entry).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return [j.split("/")[-1] for j in s.split(",") if j]


def _per_skeleton(arr, sizes, width):
    arr = np.asarray(arr, dtype=float).reshape(-1, width) if np.size(arr) else np.zeros((0, width))
    offs = np.concatenate([[0], np.cumsum(np.asarray(sizes, dtype=int))])
    return [arr[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]


def split_skeletons(data, prefix: str = "/World/Characters") -> list[dict]:
    """Replicator `skeleton_data` output (flat arrays + per-skeleton sizes) -> per-skeleton rows,
    keeping only the joints COCO-17 needs. Skeletons outside `prefix` (e.g. the hidden Biped_Setup) are dropped."""
    if data is None:
        return []
    if "numSkeletons" not in data and "skeleton_data" in data:      # backwards-compatible string form
        data = eval(data["skeleton_data"], {"array": np.array, "float32": np.float32, "int32": np.int32,  # noqa: S307
                                            "uint32": np.uint32, "bool_": np.bool_})
    n = int(data.get("numSkeletons", 0) or 0)
    if n == 0:
        return []
    p2d = _per_skeleton(data["translations2d"], data["translations2dSizes"], 2)
    p3d = _per_skeleton(data.get("globalTranslations", []), data.get("globalTranslationsSizes", [0] * n), 3)
    occ = _per_skeleton(data.get("jointOcclusions", []), data.get("jointOcclusionsSizes", [0] * n), 1)
    names = [_joint_names(e) for e in data["skeletonJoints"]]
    in_view = list(np.asarray(data.get("inView", [True] * n)).reshape(-1))
    out = []
    for i in range(n):
        path = str(data["skelPath"][i])
        if prefix and not path.startswith(prefix.rstrip("/") + "/") or "Biped_Setup" in path:
            continue
        keep = [k for k, j in enumerate(names[i]) if j in KEEP_JOINTS and k < len(p2d[i])]
        out.append({
            "skel": path, "in_view": bool(in_view[i]) if i < len(in_view) else True,
            "joints": [names[i][k] for k in keep],
            "p2d": [[round(float(v), 1) for v in p2d[i][k]] for k in keep],
            "p3d": [[round(float(v), 3) for v in p3d[i][k]] for k in keep] if len(p3d[i]) else [],
            "occ": [bool(occ[i][k][0]) for k in keep] if len(occ[i]) else [],
        })
    return out


def coco17(skel: dict, w: int, h: int) -> list[list[float]]:
    """One skeleton row -> 17 x [x, y, v] (COCO: v=2 visible, 1 labeled but occluded, 0 not labeled)."""
    idx = {j: k for k, j in enumerate(skel["joints"])}

    def kp(names):
        for n in names:
            if n in idx:
                k = idx[n]
                x, y = skel["p2d"][k]
                if not (0 <= x < w and 0 <= y < h):
                    return [0.0, 0.0, 0]
                occluded = bool(skel["occ"][k]) if skel.get("occ") else False
                return [round(x, 1), round(y, 1), 1 if occluded else 2]
        return [0.0, 0.0, 0]

    out = {name: kp(JOINTS[name]) for name in JOINTS}
    le, re = out["left_eye"], out["right_eye"]
    if le[2] and re[2]:
        out["nose"] = [round((le[0] + re[0]) / 2, 1), round((le[1] + re[1]) / 2, 1), min(le[2], re[2])]
    else:
        out["nose"] = kp(["Head"])
    out["left_ear"] = out["right_ear"] = [0.0, 0.0, 0]
    return [out[n] for n in COCO17]


# ------------------------------------------------------------------ geometry
def project(points, view, proj, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """World points -> pixels with Replicator camera_params matrices (row-vector convention: p @ view @ proj,
    as IRA 0.5.14 uses them). Returns (N x 2 pixels, N bool in front of the camera)."""
    p = np.c_[np.asarray(points, float).reshape(-1, 3), np.ones(len(np.asarray(points).reshape(-1, 3)))]
    clip = p @ np.asarray(view, float).reshape(4, 4) @ np.asarray(proj, float).reshape(4, 4)
    wc = clip[:, 3]
    front = wc > 1e-6
    ndc = clip[:, :2] / np.where(front, wc, 1.0)[:, None]
    px = np.c_[(ndc[:, 0] + 1) * 0.5 * w, (1 - ndc[:, 1]) * 0.5 * h]
    return px, front


def camera_matrices(cam: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    w, h = (int(v) for v in cam["renderProductResolution"])
    return (np.asarray(cam["cameraViewTransform"], float).reshape(4, 4),
            np.asarray(cam["cameraProjection"], float).reshape(4, 4), w, h)


def reprojection_error(cam: dict, pairs: list[tuple[list, list]]) -> tuple[float | None, bool]:
    """Median pixel error of projecting skeleton 3D joints vs the annotator's own 2D joints, and whether the
    matrices had to be transposed (column-vector convention) to agree. A self-check on the projection."""
    if not pairs:
        return None, False
    view, proj, w, h = camera_matrices(cam)
    p3, p2 = np.array([a for a, _ in pairs], float), np.array([b for _, b in pairs], float)
    best = None
    for transposed in (False, True):
        v, pr = (view.T, proj.T) if transposed else (view, proj)
        px, front = project(p3, v, pr, w, h)
        if front.any():
            err = float(np.median(np.linalg.norm(px[front] - p2[front], axis=1)))
            if best is None or err < best[0]:
                best = (err, transposed)
    return (round(best[0], 2), best[1]) if best else (None, False)


def hull(points) -> list[list[float]]:
    """Convex hull (Andrew's monotone chain), counter-clockwise."""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def zone_polygons(zones3d: list[dict], cam: dict, transposed: bool = False) -> list[dict]:
    """3D zone boxes (flat boxes = floor areas) -> image polygons for this camera. Zones that do not reach the
    image (or sit behind the camera) are left out, so a camera only gets the zones it can see."""
    view, proj, w, h = camera_matrices(cam)
    if transposed:
        view, proj = view.T, proj.T
    out = []
    for z in zones3d:
        lo, hi = z["lo"], z["hi"]
        zs = [lo[2]] if lo[2] == hi[2] else [lo[2], hi[2]]
        corners = [(x, y, zz) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in zs]
        px, front = project(corners, view, proj, w, h)
        if front.sum() < 3 or not front.all():
            continue                               # partly behind the camera: not a reliable image polygon
        poly = hull(px)
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        if max(xs) < 0 or min(xs) >= w or max(ys) < 0 or min(ys) >= h:
            continue
        out.append({"name": z["name"], "kind": z["kind"], "polygon": [[int(round(x)), int(round(y))] for x, y in poly]})
    return out


# ------------------------------------------------------------------ episode -> clips
def people_truth(plan: dict, events: list[dict]) -> list[dict]:
    """Toy-schema people rows from what actually happened (events), not just what was planned."""
    cats = plan["categories"]
    rows = []
    for track_id, p in enumerate(plan["people"], start=1):
        ev = [e for e in events if e["person"] == p["name"]]
        did = {k: [e["product"] for e in ev if e["action"] == k] for k in ("pick", "put_back", "conceal")}
        paid = {pid for e in ev if e["action"] == "pay" for pid in e.get("products", [])}
        stolen = [pid for pid in did["pick"] if pid not in did["put_back"] and pid not in paid]
        enters = [e["t"] for e in ev if e["action"] == "enter"]
        exits = [e["t"] for e in ev if e["action"] == "exit"]
        intent = p["intent"]
        rows.append({
            "name": p["name"], "track_id": track_id, "role": p["role"], "archetype": p["archetype"],
            "t_enter": round(enters[0], 2) if enters else (0.0 if p["role"] == "employee" else None),
            "t_exit": round(exits[-1], 2) if exits else None,
            "picked": [cats[i] for i in did["pick"]], "put_back": [cats[i] for i in did["put_back"]],
            "concealed": [cats[i] for i in did["conceal"]],
            "thief": bool(stolen), "stolen": [cats[i] for i in stolen],
            "matches_plan": sorted(stolen) == sorted(intent["stolen"]) and sorted(did["pick"]) == sorted(intent["picked"]),
        })
    return rows


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _video_writer(path: Path, fps: float, w: int, h: int):
    import cv2
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened():
            return vw
    raise RuntimeError(f"cannot open a video writer for {path}")


def store_config(base: dict, cam_name: str, w: int, h: int, fps: int, zones: list[dict], categories) -> dict:
    names = {z["name"] for z in zones}
    pc = dict(base.get("product_classes", {}))
    pc.update({c: c for c in sorted(set(categories))})      # a detector trained on these renders outputs categories
    return {"store": {"name": f"Isaac Sim gas-station c-store ({cam_name})"},
            "camera": {"id": cam_name, "resolution": [w, h], "fps": fps}, "zones": zones,
            "terminals": {t: z for t, z in base.get("terminals", {}).items() if z in names},
            "product_classes": pc, "catalog": base.get("catalog", {}), "rules": base.get("rules", {}),
            "ledger": base.get("ledger", {})}


def convert_episode(raw: Path, out_root: Path, base_store: dict, write_video: bool = True) -> list[dict]:
    plan = json.loads((raw / "plan.json").read_text())
    meta = json.loads((raw / "meta.json").read_text())
    events = json.loads((raw / "events.json").read_text())
    world = {r["f"]: r for r in _load_jsonl(raw / "world.jsonl")}
    fps = plan["fps_out"]
    split_dir = out_root / plan["split"]
    split_dir.mkdir(parents=True, exist_ok=True)
    truth_people = people_truth(plan, events)
    tid = {p["name"]: p["track_id"] for p in truth_people}
    roles = {p["name"]: p["role"] for p in truth_people}
    pays = sorted((e for e in events if e["action"] == "pay"), key=lambda e: e["t"])
    payments = "".join(json.dumps({"t": e["t"], "terminal": e["terminal"], "txn_id": e["txn"],
                                   "items": [{"sku": s, "qty": 1} for s in e["skus"]]}) + "\n" for e in pays)
    ev_by_frame: dict[int, list] = {}
    for e in events:
        ev_by_frame.setdefault(e["frame"], []).append(e)
    rows = []
    for c in plan["cameras"]:
        cam_dir = raw / c["name"]
        cam = json.loads((cam_dir / "camera.json").read_text())
        ann = _load_jsonl(cam_dir / "ann.jsonl")
        w, h = (int(v) for v in cam["renderProductResolution"])
        pairs = [(p3, p2) for r in ann[:: max(1, len(ann) // 20)] for s in r["skeletons"]
                 for p3, p2 in zip(s["p3d"], s["p2d"]) if s["p3d"]]
        err, transposed = reprojection_error(cam, pairs)
        clip = f"{plan['episode']}_{c['name']}"
        stem = split_dir / f"isaac_{clip}"
        zones = zone_polygons(plan["scene"]["zones3d"], cam, transposed)
        (stem.parent / f"{stem.name}.store.yaml").write_text(        # JSON is valid YAML
            json.dumps(store_config(base_store, c["name"], w, h, fps, zones, plan["categories"].values()), indent=1))
        (stem.parent / f"{stem.name}.payments.jsonl").write_text(payments)
        with open(stem.parent / f"{stem.name}.frames.jsonl", "w") as fh:
            for r in ann:
                held = world.get(r["f"], {}).get("products", {})
                skel_by_name = {s["skel"].split("/")[3]: s for s in r["skeletons"]}
                persons, products = [], []
                for b in r["boxes"]:
                    if b["prim"].startswith(CHAR_ROOT):
                        name = b["prim"].split("/")[3]
                        sk = skel_by_name.get(name)
                        persons.append({"track_id": tid.get(name), "name": name, "role": roles.get(name),
                                        "box": b["box"], "occ": b["occ"],
                                        "keypoints": coco17(sk, w, h) if sk else None})
                    elif b["prim"].startswith(PRODUCT_ROOT):
                        pid = b["prim"][len(PRODUCT_ROOT):].split("/")[0]
                        st = held.get(pid, {})
                        products.append({"id": pid, "category": plan["categories"].get(pid, b["label"]),
                                         "box": b["box"], "occ": b["occ"], "state": st.get("state", "shelf"),
                                         "holder": st.get("holder")})
                fh.write(json.dumps({"frame": r["f"], "t": r["t"], "persons": persons, "products": products,
                                     "events": ev_by_frame.get(r["f"], [])}) + "\n")
        if write_video:
            import cv2
            vw = _video_writer(stem.parent / f"{stem.name}.mp4", fps, w, h)
            for r in ann:
                img = cv2.imread(str(cam_dir / "rgb" / f"{r['f']:05d}.jpg"))
                vw.write(img if img is not None else np.zeros((h, w, 3), np.uint8))
            vw.release()
        truth = {
            "clip": clip, "description": ", ".join(f"{p['name']}: {p['archetype']}" for p in truth_people),
            "toy_data": False, "synthetic": "isaac-sim-4.5", "fps": fps, "frames": len(ann), "camera": c["name"],
            "episode": plan["episode"], "scene_seed": plan["scene_seed"], "take": plan["take"], "split": plan["split"],
            "flags": plan["flags"], "lighting": plan["lighting"]["mode"], "people": truth_people,
            "events": events, "reprojection_px": err,
            "label_notes": "persons/keypoints/products are rendered ground truth; product actions are scripted "
                           "(attach to the hand joint, slide to a body anchor and hide for conceal). Concealment has no "
                           "arm animation. nose = midpoint of the eyes, ears unlabeled.",
            "warnings": meta.get("warnings", []),
        }
        (stem.parent / f"{stem.name}.truth.json").write_text(json.dumps(truth, indent=1))
        rows.append({"clip": clip, "episode": plan["episode"], "camera": c["name"], "split": plan["split"],
                     "frames": len(ann), "people": len(truth_people),
                     "thieves": sum(p["thief"] for p in truth_people), "reprojection_px": err,
                     "zones": sorted(z["name"] for z in zones),
                     "plan_mismatch": [p["name"] for p in truth_people if not p["matches_plan"]],
                     "warnings": len(meta.get("warnings", []))})
    return rows


def convert_all(raw_root: Path, out_root: Path, base_store_path: Path, delete_raw: bool = False,
                write_video: bool = True) -> int:
    import yaml
    base = yaml.safe_load(base_store_path.read_text())
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for raw in sorted(p for p in raw_root.iterdir() if (p / "done.json").exists() and not (p / "converted.json").exists()):
        rows = convert_episode(raw, out_root, base, write_video)
        with open(out_root / "manifest.jsonl", "a") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        if delete_raw:
            for c in rows:
                shutil.rmtree(raw / c["camera"] / "rgb", ignore_errors=True)
        (raw / "converted.json").write_text(json.dumps({"clips": len(rows)}))
        print(f"{raw.name}: {len(rows)} clips")
        n += len(rows)
    return n


# ------------------------------------------------------------------ overlays + checks
def _color(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(i * 7919 + 3)
    return tuple(int(v) for v in rng.integers(60, 255, 3))


def draw_overlay(img: np.ndarray, frame: dict, recent_events: list[dict], zones: list[dict] | None = None) -> np.ndarray:
    import cv2
    for z in zones or []:
        cv2.polylines(img, [np.asarray(z["polygon"], np.int32)], True, (200, 200, 200), 1)
        x, y = z["polygon"][0]
        cv2.putText(img, z["name"], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for p in frame["persons"]:
        col = _color(p["track_id"] or 0)
        x0, y0, x1, y1 = p["box"]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        cv2.putText(img, f"#{p['track_id']} {p['name']} {p['role'] or ''}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        kps = p.get("keypoints")
        if kps:
            for a, b in COCO_EDGES:
                if kps[a][2] and kps[b][2]:
                    cv2.line(img, (int(kps[a][0]), int(kps[a][1])), (int(kps[b][0]), int(kps[b][1])), col, 1)
            for x, y, v in kps:
                if v:
                    cv2.circle(img, (int(x), int(y)), 3, col, -1 if v == 2 else 1)
    for pr in frame["products"]:
        x0, y0, x1, y1 = pr["box"]
        held = pr["state"] != "shelf"
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255) if held else (0, 200, 0), 2 if held else 1)
        if held:
            cv2.putText(img, f"{pr['category']} [{pr['state']} {pr['holder']}]", (x0, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for i, e in enumerate(recent_events):
        extra = e.get("category") or e.get("txn") or ""
        txt = f"t={e['t']:.1f} {e['person']} {e['action'].upper()} {extra} {e.get('anchor', '')}"
        cv2.putText(img, txt, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"t={frame['t']:.2f} f={frame['frame']}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return img


def overlay_clip(stem: Path, dest: Path, hold_s: float = 1.5) -> Path:
    import cv2
    frames = _load_jsonl(stem.parent / f"{stem.name}.frames.jsonl")
    truth = json.loads((stem.parent / f"{stem.name}.truth.json").read_text())
    zones = json.loads((stem.parent / f"{stem.name}.store.yaml").read_text())["zones"]
    cap = cv2.VideoCapture(str(stem.parent / f"{stem.name}.mp4"))
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{stem.name}.overlay.mp4"
    vw = None
    for fr in frames:
        ok, img = cap.read()
        if not ok:
            break
        if vw is None:
            vw = _video_writer(out, truth["fps"], img.shape[1], img.shape[0])
        recent = [e for e in truth["events"] if 0 <= fr["t"] - e["t"] <= hold_s]
        vw.write(draw_overlay(img, fr, recent, zones))
    cap.release()
    if vw is not None:
        vw.release()
    return out


def pick_clips_for_overlay(out_root: Path, n: int) -> list[Path]:
    """Prefer clips with the most kinds of labelled events, then thieves, spread over episodes."""
    scored = []
    for tp in out_root.rglob("isaac_*.truth.json"):
        t = json.loads(tp.read_text())
        kinds = {e["action"] for e in t["events"]}
        scored.append((len(kinds) + sum(p["thief"] for p in t["people"]), t["episode"], tp))
    scored.sort(key=lambda r: (-r[0], r[1]))
    picked, seen = [], set()
    for _, ep, tp in scored:
        if ep not in seen:
            picked.append(tp.parent / tp.name[: -len(".truth.json")])
            seen.add(ep)
        if len(picked) == n:
            break
    return picked


def check_clip(stem: Path) -> dict:
    """Label sanity numbers for one clip: boxes/keypoints present, held products near a wrist, event coverage."""
    frames = _load_jsonl(stem.parent / f"{stem.name}.frames.jsonl")
    truth = json.loads((stem.parent / f"{stem.name}.truth.json").read_text())
    n_person = n_kp = 0
    near = total_held = 0
    for fr in frames:
        persons = {p["name"]: p for p in fr["persons"]}
        for p in fr["persons"]:
            n_person += 1
            n_kp += sum(1 for k in (p["keypoints"] or []) if k[2] == 2)
        for pr in fr["products"]:
            holder = persons.get(pr["holder"]) if pr["state"] == "hand" else None
            if not holder or not holder["keypoints"]:
                continue
            wrists = [holder["keypoints"][i] for i in (9, 10) if holder["keypoints"][i][2]]
            if not wrists:
                continue
            cx, cy = (pr["box"][0] + pr["box"][2]) / 2, (pr["box"][1] + pr["box"][3]) / 2
            scale = max(1, holder["box"][3] - holder["box"][1])
            total_held += 1
            near += min(np.hypot(cx - x, cy - y) for x, y, _ in wrists) < 0.25 * scale
    actions: dict[str, int] = {}
    for e in truth["events"]:
        actions[e["action"]] = actions.get(e["action"], 0) + 1
    return {"clip": truth["clip"], "frames": len(frames), "person_boxes_per_frame": round(n_person / max(1, len(frames)), 2),
            "visible_kps_per_person": round(n_kp / max(1, n_person), 1),
            "held_product_near_wrist": round(near / total_held, 3) if total_held else None,
            "reprojection_px": truth.get("reprojection_px"), "events": actions,
            "thieves_without_exit": [p["name"] for p in truth["people"] if p["thief"] and p["t_exit"] is None],
            "warnings": len(truth.get("warnings", []))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("raw")
    c.add_argument("out")
    c.add_argument("--store", default=str(Path(__file__).resolve().parents[4] / "configs" / "store_gas_station_small.yaml"))
    c.add_argument("--delete-raw", action="store_true", help="delete JPEG frames once an episode is converted")
    c.add_argument("--watch", action="store_true", help="keep converting episodes as the generator finishes them")
    o = sub.add_parser("overlay")
    o.add_argument("out")
    o.add_argument("--n", type=int, default=5)
    o.add_argument("--dest", default=None)
    k = sub.add_parser("check")
    k.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        while True:
            convert_all(Path(a.raw), Path(a.out), Path(a.store), a.delete_raw)
            if not a.watch:
                break
            time.sleep(30)
    elif a.cmd == "overlay":
        out = Path(a.out)
        for stem in pick_clips_for_overlay(out, a.n):
            print(overlay_clip(stem, Path(a.dest) if a.dest else out / "overlays"))
    else:
        rows = [check_clip(tp.parent / tp.name[: -len(".truth.json")]) for tp in sorted(Path(a.out).rglob("isaac_*.truth.json"))]
        for r in rows:
            print(json.dumps(r))
        (Path(a.out) / "check.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
