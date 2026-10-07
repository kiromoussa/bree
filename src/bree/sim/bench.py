"""The fixed benchmark: run the pipeline on a split of SIMULATED store clips and score it against ground truth.

    python -m bree.sim.bench dev            (make bench-dev)    -> results/bench_dev.json and .md
    python -m bree.sim.bench test           (make bench-test)   -> results/bench_test.json and .md  (final numbers only)
    python -m bree.sim.bench dev --smoke    one clip, 60 frames per camera, nothing written to results/
    python -m bree.sim.bench dev --score-only                   rescore finished runs (out/bench/dev/clip_*)
    python -m bree.sim.bench dev2 --stress                      (make bench-dev2)   the tuning set of generator 2, with the stress column
    python -m bree.sim.bench checkpoint [--batch K] --stress    the latest (or the K-th) rendered batch of the checkpoint pool

Stress: --stress runs the pipeline a second time on inputs made worse without ground truth and reports it as a second
column: --drop-item-cameras 0.2 (that share of the item cameras is missing, drawn with --stress-seed), --calib-noise-deg 0.5
(each camera's pose is off by that much per axis), --register-delay-s 5 (receipts arrive later), --wrong-planogram 0.05
(that share of the slots names another product). Give the flags by themselves for one stress at a time.

Clips: scripts/bench/manifest.json and scripts/bench/README.md. A clip folder holds what the pipeline may read
(<camera>.mp4, calibration.json, layout.json with the planogram, register.jsonl, clip.json) and a truth/ folder.
The pipeline is run on a folder of links that leaves truth/ out (`public_view`), so the runner cannot read it
by accident. Only `score_clip` opens truth/.

The runner is one function(clip, out): by default `bree.shelf.store:run` (shelf events from the item cameras, floor
tracks from the people cameras). `--runner bree.sim.bench:run_pipeline` is the per-camera engine it replaced here. What the
scorer reads from <out>/pipeline, and nothing else:
  events.jsonl         bree.events.types.Event rows. PICK: `sku` (or `item`) is the SKU claim, `meta.slot.id` the
                       slot claim, `zone` the fixture, `person_id` the store-wide identity.
  alerts.jsonl         bree.alerts.types.Alert rows (retractions applied by the scorer).
  frames_<cam>.jsonl   per processed frame: t, persons [{id, bbox, kpts}], products [{bbox, cat}]. A frame that
                       is not in this log did not reach the pipeline. Optional per row: "hands": [[x, y], ...];
                       optional per person: "gid" (the store-wide identity of that track).
  person_ids.jsonl     {camera, t, ids: {track id: store-wide id}} when the frame log has no "gid".
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "scripts" / "bench" / "manifest.json"
NOTE = "SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage."
ITEM_KINDS = ("shelf", "cooler", "checkout")
STAGES = ("in_view", "frame_reached_pipeline", "hand_or_item_detected", "shelf_event_emitted", "right_slot", "right_sku",
          "associated_to_a_shopper", "right_shopper", "conceal_or_pay_classified", "ledger_basket", "alert")
STAGE_HELP = {
    "in_view": "a rendered shelf, cooler or counter camera sees the item in the hand, or the reaching hand, around the pick",
    "frame_reached_pipeline": "one of those frames is in the pipeline's frame log (the camera node sent it)",
    "hand_or_item_detected": "in such a frame: a product box on the item (IoU 0.3), or a person box on the picker with a wrist or hand point near the true hand",
    "shelf_event_emitted": "a PICK event within 3 s of the true pick (each PICK is paired with one true pick at most)",
    "right_slot": "that PICK's meta.slot.id is the true slot",
    "right_sku": "that PICK's sku (or item) is the true SKU",
    "associated_to_a_shopper": "that PICK has a person_id that appears in the frame logs",
    "right_shopper": "the boxes of that person_id around the pick sit on the true shopper",
    "conceal_or_pay_classified": "a CONCEAL (stolen), PAY (paid) or PUT_BACK (put back) event on the true shopper near the true time",
    "ledger_basket": "stolen: an alert or review lists this pick as unpaid. Paid or put back: none does",
    "alert": "stolen: that record is alert tier. Paid or put back: no alert or review on this shopper",
}


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def clip_dirs(split: str, only: list[int] | None = None, batch: int | None = None) -> list[Path]:
    m = manifest()
    seeds = m["splits"][split]["seeds"]
    if "batches" in m["splits"][split]:      # the checkpoint pool: one rendered batch, the latest unless asked
        b = m["splits"][split]["batches"]
        if not b or (batch and batch > len(b)):
            raise SystemExit(f"{split}: {len(b)} batches rendered. Render the next one: .venv/bin/python scripts/bench/render_split.py {split} --next 6")
        seeds = b[(batch or len(b)) - 1]
    elif not isinstance(seeds, list):      # train: a seed range; the clips rendered so far (make shelf-clips) are a second tuning set
        lo, hi = (int(v) for v in seeds.split(":"))
        seeds = sorted(s for d in (ROOT / m["root"] / split).glob("clip_*") if d.name[5:].isdigit() and lo <= (s := int(d.name[5:])) < hi and (d / "truth").is_dir())
    return [ROOT / m["root"] / split / f"clip_{s}" for s in seeds if not only or s in only]


def load_calibration(clip: Path) -> dict:
    """{camera id: bree.calib.camera.Camera} from a clip's calibration.json (pose as installed, roll and lens term included)."""
    import numpy as np
    from bree.calib.camera import Camera
    return {c["id"]: Camera(c["fx"], c["cx"], c["cy"], np.asarray(c["R"], float), np.asarray(c["position"], float), tuple(c["resolution"]), float(c.get("k_div") or 0.0))
            for c in json.loads((Path(clip) / "calibration.json").read_text())["cameras"]}


STRESS_DEFAULT = {"drop_item_cameras": 0.2, "calib_noise_deg": 0.5, "register_delay_s": 5.0, "wrong_planogram": 0.05}


def public_view(clip: Path, work: Path, stress: dict | None = None) -> Path:
    """Links to everything in the clip except the ground truth. This is the folder the pipeline is given.
    With `stress`, some of those inputs are made worse first (apply_stress). No ground truth is read for that."""
    pub = work / "clip"
    shutil.rmtree(pub, ignore_errors=True)
    pub.mkdir(parents=True)
    for f in Path(clip).iterdir():
        if f.is_file() and f.name != "render.log":
            (pub / f.name).symlink_to(f.resolve())
    if stress and any(stress.get(k) for k in STRESS_DEFAULT):
        (work / "stress.json").write_text(json.dumps(apply_stress(Path(clip), pub, stress), indent=1))
    return pub


def apply_stress(clip: Path, pub: Path, st: dict) -> dict:
    """Worse inputs, drawn from (stress seed, clip seed): item cameras missing, a calibration that is off, late
    receipts, a planogram that names the wrong product for some slots. Reads only what the pipeline may read."""
    import numpy as np
    meta, calib, layout = (json.loads((clip / f).read_text()) for f in ("clip.json", "calibration.json", "layout.json"))
    R, done = random.Random(f"{st.get('seed', 1)}:{meta['seed']}"), {k: st.get(k) or 0 for k in STRESS_DEFAULT} | {"seed": st.get("seed", 1)}

    def write(name: str, text: str) -> None:
        (pub / name).unlink()
        (pub / name).write_text(text)
    if st.get("drop_item_cameras"):       # the register camera stays: without it nobody pays
        item = sorted(c["id"] for c in calib["cameras"] if c["kind"] in ITEM_KINDS and c["id"] in meta["cameras"] and not c["id"].startswith("REGISTER"))
        gone = set(R.sample(item, round(st["drop_item_cameras"] * len(item))))
        meta["cameras"] = [c for c in meta["cameras"] if c not in gone]
        calib["cameras"] = [c for c in calib["cameras"] if c["id"] not in gone]
        for c in gone:
            (pub / f"{c}.mp4").unlink(missing_ok=True)
        done["cameras_dropped"] = sorted(gone)
    if st.get("calib_noise_deg"):         # a small rotation about each camera axis; the layout's own poses get noise of the same size
        sd = math.radians(st["calib_noise_deg"])
        for c in calib["cameras"]:
            rx, ry, rz = (R.gauss(0, sd) for _ in range(3))
            Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
            Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
            Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
            c["R"] = (Rz @ Ry @ Rx @ np.asarray(c["R"], float)).round(8).tolist()
        for c in layout["cameras"]:
            for k in ("yaw", "pitch", "roll"):
                if c.get(k) is not None:
                    c[k] += R.gauss(0, sd)
    if st.get("register_delay_s"):
        rows = _jsonl(clip / "register.jsonl")
        for r in rows:
            r["t"] = round(r["t"] + st["register_delay_s"], 2)
        write("register.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    if st.get("wrong_planogram"):
        skus, ids = sorted({x["skuId"] for x in layout["slots"]}), sorted(x["id"] for x in layout["slots"])
        wrong = set(R.sample(ids, round(st["wrong_planogram"] * len(ids))))
        for x in layout["slots"]:
            if x["id"] in wrong:
                x["skuId"] = R.choice([k for k in skus if k != x["skuId"]])
        done["slots_renamed"] = len(wrong)
    write("clip.json", json.dumps(meta, indent=1))
    write("calibration.json", json.dumps(calib, indent=1))
    write("layout.json", json.dumps(layout))
    return done


# ------------------------------------------------------------------ the pipeline under test
def edge_stage(sim_out: Path, layout: Path, out: Path, max_frames: int | None) -> tuple[dict, dict]:
    """Play each merch camera through a camera node into a hub. Returns ({camera: BurstSource}, stats)."""
    import yaml
    from bree.edge.hub import BurstSource, Hub
    from bree.edge.node import build, load_node_config
    from bree.sim.isaac_adapter import adapt
    ad = adapt(sim_out, layout, out / "node_inputs", zone_owner="all")     # a node triggers on every shelf it sees
    hub = Hub(out / "hub", token="e2e")
    httpd = hub.serve("127.0.0.1", 0)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    sources, rows = {}, {}
    for c in ad["cameras"]:
        raw = yaml.safe_load(Path(c["store"]).read_text())
        merch = [z for z in raw["zones"] if z["kind"] in ("shelf", "cooler")]
        if c["kind"] not in ("shelf", "cooler") or not merch:      # the hardware plan: item cameras are the Pi nodes
            rows[c["id"]] = {"kind": c["kind"], "frames": c["frames"],
                             "path": "continuous stream" + ("" if c["kind"] not in ("shelf", "cooler") else " (no shelf or cooler zone in view)")}
            continue
        raw["node"] = {"source": c["video"], "fps": ad["fps"], "hub_url": url, "token": "e2e", "spool_dir": str(out / "spool" / c["id"])}
        cfg_path = out / "node_inputs" / f"{c['id']}.node.yaml"
        cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
        node = build(load_node_config(cfg_path))
        if max_frames:
            node.source._src.max_frames = max_frames
        stop = threading.Event()
        th = threading.Thread(target=node.uplink.run, args=(stop,), daemon=True)
        th.start()
        node.run()
        drained = node.uplink.drain(300)
        stop.set()
        th.join(5)
        src = BurstSource(out / "hub" / c["id"])
        sources[c["id"]] = src
        sent = len(src)
        rows[c["id"]] = {"kind": c["kind"], "path": "node -> hub", "frames": node.frames, "trigger_zones": len(merch),
                         "triggers": node.triggers, "bursts": node.bursts, "frames_at_hub": sent,
                         "share_of_frames_sent": round(sent / node.frames, 3) if node.frames else None,
                         "burst_bytes": node.burst_bytes, "spool_drained": drained, "node_errors": len(node.errors),
                         "trigger_ms_per_frame": round(1000 * node.trigger_s / max(node.frames, 1), 3),
                         # every frame at the mean size of the frames that were sent (an estimate, not a second encode)
                         "full_stream_bytes_estimate": round(node.burst_bytes / sent * node.frames) if sent else None}
    hub.q.join()
    httpd.shutdown()
    nodes = [r for r in rows.values() if r["path"] == "node -> hub"]
    total, sent = sum(r["frames"] for r in nodes), sum(r["frames_at_hub"] for r in nodes)
    return sources, {"cameras": rows, "node_cameras": len(nodes), "node_frames": total, "node_frames_sent": sent,
                     "share_of_node_frames_sent": round(sent / total, 3) if total else None,
                     "hub_errors": hub.error_count}


def run_pipeline(clip: Path, out: Path, backend: str = "sim_sku", edge: bool = True, closed_world: bool = True,
                 slots: bool = True, max_frames: int | None = None, verbose: bool = False) -> dict:
    """The pipeline as it stands, on one clip (the public view: no truth in `clip`). Camera nodes and hub for the
    item cameras, then one run_store over all cameras: products from the sim-trained SKU detector, store-wide
    closed-world identity, 3D slot per pick from the clip's calibration, the register feed as payments."""
    import bree.pipeline as P
    from bree.cli import make_backend
    from bree.events.zones import load_store_config, merge_stores
    from bree.ledger.payments import JsonlPayments
    from bree.sim.isaac_adapter import adapt
    from bree.sim.sim_eval import sim_handoff
    clip, out = Path(clip), Path(out)
    t0 = time.time()
    layout_path = clip / "layout.json"
    layout = json.loads(layout_path.read_text())
    sources, edge_stats = edge_stage(clip, layout_path, out, max_frames) if edge else ({}, None)
    t1 = time.time()
    ad = adapt(clip, layout_path, out / "inputs")
    stores = [load_store_config(c["store"]) for c in ad["cameras"]]
    cams = [P.CameraInput(c["id"], sources.get(c["id"], c["video"]), s) for c, s in zip(ad["cameras"], stores)]
    (out / "pipeline").mkdir(parents=True, exist_ok=True)
    made, ids = [], (out / "pipeline" / "person_ids.jsonl").open("w")

    class Fusion(P.StoreEvents):          # keep a handle on the identity layer to log track id -> store-wide id
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    def on_frame(fr, obs, events, ledger, vis):
        if obs.persons:
            ids.write(json.dumps({"camera": fr.camera, "t": round(obs.t, 3),
                                  "ids": {str(p.track_id): made[0].person_id(fr.camera, p.track_id) for p in obs.persons}}) + "\n")
    keep, P.StoreEvents = P.StoreEvents, Fusion
    try:
        summary = P.run_store(cams, make_backend(backend, merge_stores(stores)), out / "pipeline",
                              payments=JsonlPayments(clip / "register.jsonl"), save_video=False, max_frames=max_frames,
                              verbose=verbose, on_frame=on_frame,
                              handoff=sim_handoff(layout, closed_world, slots, load_calibration(clip)))
    finally:
        P.StoreEvents = keep
        ids.close()
    info = {"backend": backend, "edge": edge_stats, "closed_world": closed_world, "slots_3d": slots, "max_frames": max_frames,
            "fps": ad["fps"], "cameras": [{k: c[k] for k in ("id", "kind", "frames", "zones")} for c in ad["cameras"]],
            "zones_unseen": ad["zones_unseen"], "frames_processed": summary.frames, "pipeline_fps": summary.pipeline_fps,
            "identity": summary.identity, "pipeline_events": summary.events,
            "wall_s": {"edge": round(t1 - t0, 1), "pipeline": round(time.time() - t1, 1)}}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------ scoring
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix, 1e-9)


def _rate(n, d):
    return round(n / d, 3) if d else None


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def score_clip(clip: Path, run: Path, pick_tol_s: float = 3.0) -> dict:
    """Ground truth of `clip` against the pipeline outputs in `run`/pipeline. See STAGE_HELP for the funnel."""
    from bree.sim.sim_eval import final_alerts
    clip, pipe = Path(clip), Path(run) / "pipeline"
    meta = json.loads((clip / "clip.json").read_text())
    fps, kind = meta["fps"], {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    res = {c["id"]: c["resolution"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    truth = _jsonl(clip / "truth" / "events.jsonl")
    shoppers = json.loads((clip / "truth" / "shoppers.json").read_text())
    # generator 2 only: staff takes and puts, touches, the options of the clip, what is really in each slot
    acts = _jsonl(clip / "truth" / "acts.jsonl")
    faults = json.loads((clip / "truth" / "faults.json").read_text()) if (clip / "truth" / "faults.json").exists() else {}
    misplaced = set(json.loads((clip / "truth" / "planogram.json").read_text())["slots_where_the_item_differs"]) if (clip / "truth" / "planogram.json").exists() else None
    staff = {s["shopper"] for s in shoppers if s.get("role") == "staff"}
    fixture_of = {x["id"]: x["fixtureId"] for x in json.loads((clip / "layout.json").read_text())["slots"]}
    staff_takes = [a for a in acts if a["kind"] == "staff_take"]
    targets = truth + staff_takes                 # a PICK on a staff take is a true shelf event, not a false pick
    tf: dict[tuple[str, int], dict] = {(r["camera"], r["frame"]): r for r in _jsonl(clip / "truth" / "frames.jsonl")}
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for (cam, f), r in tf.items():
        by_frame[f].append(r)
    # pipeline outputs
    events = _jsonl(pipe / "events.jsonl")
    alerts = final_alerts(_jsonl(pipe / "alerts.jsonl"))
    gids = {(r["camera"], round(r["t"] * fps)): r["ids"] for r in _jsonl(pipe / "person_ids.jsonl")}
    logs = {cam: {round(r["t"] * fps): r for r in _jsonl(pipe / f"frames_{cam}.jsonl")} for cam in meta["cameras"]}
    # every pipeline person box that sits on a true person: (t, store-wide id, shopper)
    seen: dict[int, list[tuple[float, str]]] = defaultdict(list)
    all_ids: set[int] = set()
    for cam, log in logs.items():
        for f, row in log.items():
            gt = (tf.get((cam, f)) or {}).get("persons", [])
            for p in row["persons"]:
                gid = p.get("gid", (gids.get((cam, f)) or {}).get(str(p["id"])))
                if gid is None:
                    continue
                all_ids.add(gid)
                best = max(gt, key=lambda g: _iou(g["bbox"], p["bbox"]), default=None)
                if best and _iou(best["bbox"], p["bbox"]) >= 0.3:
                    seen[gid].append((row["t"], best["shopper"]))

    def shopper_of(gid, t=None, win=2.5):
        """Who a store-wide id is, near time t (ids can swap), else over the whole clip. None: never on a person."""
        obs = seen.get(gid) or []
        near = [s for (tt, s) in obs if t is not None and abs(tt - t) <= win]
        votes = Counter(near or [s for _, s in obs])
        return votes.most_common(1)[0][0] if votes else None

    # ---- picks: pair pipeline PICK events with true picks, the ones at the right fixture first
    picks = [e for e in events if e["type"] == "pick"]
    slot_of = lambda e: (e.get("meta") or {}).get("slot") or {}      # noqa: E731
    at = lambda p, fixture: p.get("zone") == fixture or slot_of(p).get("fixture") == fixture      # noqa: E731
    pairs = sorted(((not at(p, g["fixtureId"]), abs(p["t"] - g["t"]), i, j)
                    for i, p in enumerate(picks) for j, g in enumerate(targets) if abs(p["t"] - g["t"]) <= pick_tol_s))
    of_truth, used = {}, set()
    for _, _, i, j in pairs:
        if i not in used and j not in of_truth:
            of_truth[j] = picks[i]
            used.add(i)
    unpaid = [(a, it) for a in alerts for it in a["unpaid_items"]]

    def alert_shopper(a):
        votes = Counter(s for g in [a["person_id"], *a.get("group", [])] for (t, s) in seen.get(g, []) if t <= a["t_emitted"])
        if votes:
            return votes.most_common(1)[0][0]
        near = Counter(g["shopper"] for it in a["unpaid_items"] for g in truth if abs(it["t_pick"] - g["t"]) <= pick_tol_s)
        return near.most_common(1)[0][0] if near else None
    for a in alerts:
        a["_shopper"] = alert_shopper(a)
    listed = lambda ev: [(a, it) for a, it in unpaid if ev and abs(it["t_pick"] - ev["t"]) < 0.06 and ev.get("person_id") in [a["person_id"], *a.get("group", [])]]      # noqa: E731
    mount = lambda cam: re.sub(r"^[A-Z0-9]+-|-\d+$", "", cam) if cam else "none"      # noqa: E731
    blocks = [b for b in (faults.get("camera_faults") or {}).get("block", [])]
    bumps = [b for b in (faults.get("camera_faults") or {}).get("bump", [])]
    drop = faults.get("dropped_scan") or {}
    night = faults.get("light") == "night"

    rows = []
    for j, g in enumerate(truth):
        f0 = round(g["t"] * fps)
        item_fr, hand_fr = [], []             # (camera, frame, true item box) / (camera, frame, true person row)
        for f in range(f0 - fps, f0 + round(2.5 * fps) + 1):
            for r in by_frame.get(f, []):
                if kind.get(r["camera"]) not in ITEM_KINDS:
                    continue
                if f >= f0:
                    item_fr += [(r["camera"], f, it) for it in r["items"] if it["kind"] == "hand" and it["shopper"] == g["shopper"] and it["slot"] == g["slotId"] and it["vis_px"] >= 20]
                if f <= f0 + fps:
                    w, h = res[r["camera"]]
                    hand_fr += [(r["camera"], f, p) for p in r["persons"] if p["shopper"] == g["shopper"] and p.get("hand") and 0 <= p["hand"][0] < w and 0 <= p["hand"][1] < h]
        st = dict.fromkeys(STAGES, False)
        st["in_view"] = bool(item_fr or hand_fr)
        reached_i = [(c, f, x) for c, f, x in item_fr if f in logs.get(c, {})]
        reached_h = [(c, f, x) for c, f, x in hand_fr if f in logs.get(c, {})]
        st["frame_reached_pipeline"] = bool(reached_i or reached_h)
        item_det = sku_det = person_det = hand_det = False
        for c, f, it in reached_i:
            hit = [q for q in logs[c][f]["products"] if _iou(q["bbox"], it["bbox"]) >= 0.3]
            item_det |= bool(hit)
            sku_det |= any(q["cat"] == g["skuId"] for q in hit)
        for c, f, p in reached_h:
            row = logs[c][f]
            near = lambda pt, box: (pt[0] - p["hand"][0]) ** 2 + (pt[1] - p["hand"][1]) ** 2 <= max(40.0, 0.15 * max(box[2] - box[0], box[3] - box[1])) ** 2  # noqa: E731
            hand_det |= any(near(pt, p["bbox"]) for pt in row.get("hands") or [])
            for q in row["persons"]:
                if _iou(q["bbox"], p["bbox"]) >= 0.3:
                    person_det = True
                    hand_det |= any(k[2] >= 0.3 and near(k, q["bbox"]) for k in (q.get("kpts") or [])[9:11])
        st["hand_or_item_detected"] = item_det or hand_det
        ev = of_truth.get(j)
        st["shelf_event_emitted"] = ev is not None
        claim = pid = who = None
        if ev:
            claim, pid = ev.get("sku") or ev.get("item"), ev.get("person_id")
            st["right_slot"] = slot_of(ev).get("id") == g["slotId"]
            st["right_sku"] = claim == g["skuId"]
            st["associated_to_a_shopper"] = pid is not None and pid in all_ids
            who = shopper_of(pid, ev["t"]) if st["associated_to_a_shopper"] else None
            st["right_shopper"] = who == g["shopper"]
        want, t_res = {"concealed": ("conceal", g["tConceal"]), "paid": ("pay", g["tPay"]), "put_back": ("put_back", g["tPutBack"])}.get(g["outcome"], (None, None))
        if want and t_res is not None:        # PAY marks a register visit: wide window; the others happen at one moment
            lo, hi = (t_res - 8, t_res + 6) if want == "pay" else (t_res - pick_tol_s, t_res + pick_tol_s)
            who_ok = {g["shopper"], g.get("paidBy") or g["shopper"]} if want == "pay" else {g["shopper"]}       # a group: the one who pays is not the one who took it
            st["conceal_or_pay_classified"] = any(e["type"] == want and lo <= e["t"] <= hi and shopper_of(e.get("person_id"), e["t"]) in who_ok for e in events)
        mine = listed(ev)
        if g["outcome"] in ("concealed", "theft"):
            st["ledger_basket"] = bool(mine)
            st["alert"] = any(a["tier"] == "alert" for a, _ in mine)
        else:
            st["ledger_basket"] = not mine
            st["alert"] = not any(a["_shopper"] == g["shopper"] for a in alerts)
        lost = next((s for s in STAGES if not st[s]), None)
        best = next((c["id"] for c in g["cameras"] if c["id"] in meta["cameras"] and kind.get(c["id"]) in ITEM_KINDS), None)
        tags = [t for t, on in (("night", night), ("group", bool(g.get("group"))), ("scan_dropped", bool(g.get("scanDropped"))),
                                ("slot_holds_another_product", misplaced is not None and g["slotId"] in misplaced),
                                ("best_camera_covered", any(b["camera"] == best and b["t0"] <= g["t"] < b["t1"] for b in blocks)),
                                ("best_camera_knocked", any(b["camera"] == best and b["t"] <= g["t"] for b in bumps))) if on]
        rows.append({"clip": meta["seed"], "shopper": g["shopper"], "t": g["t"], "sku": g["skuId"], "slot": g["slotId"], "fixture": g["fixtureId"],
                     "zone": g["zone"], "outcome": g["outcome"], "stages": st, "lost_at": lost, "tags": tags, "camera_mount": mount(best),
                     "detail": {"item_frames": len(item_fr), "hand_frames": len(hand_fr), "item_frames_reached": len(reached_i), "hand_frames_reached": len(reached_h),
                                "item_detected": item_det, "right_sku_detected": sku_det, "person_detected": person_det, "hand_detected": hand_det,
                                "event_t": ev["t"] if ev else None, "event_zone": ev.get("zone") if ev else None, "event_sku": claim,
                                "event_slot": slot_of(ev).get("id") if ev else None, "right_fixture": bool(ev) and (ev.get("zone") == g["fixtureId"] or slot_of(ev).get("fixture") == g["fixtureId"]),
                                "event_person": pid, "event_shopper": who,
                                "best_item_camera": next((c["id"] for c in g["cameras"] if c["id"] in meta["cameras"] and kind.get(c["id"]) in ITEM_KINDS), None)}})

    # ---- thefts and alerts
    role = lambda n: "staff" if n in staff else "clerk" if n == "clerk" else "nobody" if n is None else "shopper"      # noqa: E731
    stolen = [(j, g) for j, g in enumerate(truth) if g["outcome"] == "concealed"]
    thieves = {g["shopper"] for _, g in stolen}
    first_conceal = {s: min(g["tConceal"] for _, g in stolen if g["shopper"] == s) for s in thieves}
    alert_rows = []
    for a in alerts:
        s = a["_shopper"]
        true = s in thieves and first_conceal[s] <= a["t_emitted"]
        alert_rows.append({"clip": meta["seed"], "alert_id": a["alert_id"], "tier": a["tier"], "person_id": a["person_id"], "shopper": s, "role": role(s), "true_theft": true,
                           "t_emitted": a["t_emitted"], "skus": [it.get("sku") or it["category"] for it in a["unpaid_items"]], "confidence": a["confidence"]})
    theft_rows = []
    for j, g in stolen:
        hits = [r for r in alert_rows if r["shopper"] == g["shopper"] and r["t_emitted"] >= g["tConceal"]]
        first = min((r for r in hits if r["tier"] == "alert"), key=lambda r: r["t_emitted"], default=None)
        theft_rows.append({"clip": meta["seed"], "shopper": g["shopper"], "sku": g["skuId"], "slot": g["slotId"], "t_pick": g["t"], "t_conceal": g["tConceal"],
                           "alert": first is not None, "alert_or_review": bool(hits), "sku_in_alert": bool(first) and g["skuId"] in first["skus"],
                           "listed_unpaid": bool(listed(of_truth.get(j))), "tags": rows[j]["tags"],
                           "time_to_alert_s": round(first["t_emitted"] - g["tConceal"], 2) if first else None})
    # ---- staff takes, touches, false PICK events
    worst = lambda ev: "alert" if any(a["tier"] == "alert" for a, _ in listed(ev)) else "review" if listed(ev) else None      # noqa: E731
    staff_rows = [{"clip": meta["seed"], "t": a["t"], "shopper": a["shopper"], "slot": a["slotId"], "outcome": a["outcome"],
                   "pick_event": of_truth.get(len(truth) + k) is not None, "listed_unpaid": worst(of_truth.get(len(truth) + k))} for k, a in enumerate(staff_takes)]
    at_touch, touch_rows = set(), []
    for a in (a for a in acts if a["kind"] == "touch"):
        near = sorted((abs(p["t"] - a["t"]), i) for i, p in enumerate(picks) if i not in used and i not in at_touch and abs(p["t"] - a["t"]) <= pick_tol_s and at(p, a["fixtureId"]))
        ev = picks[near[0][1]] if near else None
        if near:
            at_touch.add(near[0][1])
        touch_rows.append({"clip": meta["seed"], "t": a["t"], "shopper": a["shopper"], "slot": a["slotId"], "counted_as_pick": ev is not None, "listed_unpaid": worst(ev)})
    cam_of = lambda p: ((p.get("meta") or {}).get("shelf") or {}).get("camera_id")      # noqa: E731
    false_picks = [{"clip": meta["seed"], "t": p["t"], "camera": cam_of(p), "mount": mount(cam_of(p)), "at_a_touch": i in at_touch, "listed_unpaid": worst(p)}
                   for i, p in enumerate(picks) if i not in used]
    # ---- put-backs: a shopper's (into the same slot or another one) and staff puts; a PUT_BACK event is paired with one of them at most
    true_puts = [{"t": g["tPutBack"], "slot": g.get("putBackSlot") or g["slotId"], "shopper": g["shopper"], "other_slot": bool(g.get("putBackSlot")), "staff": False}
                 for g in truth if g["outcome"] == "put_back" and g.get("tPutBack") is not None]
    true_puts += [{"t": a["t"], "slot": a["slotId"], "shopper": a["shopper"], "other_slot": False, "staff": True} for a in acts if a["kind"] == "staff_put"]
    puts = [e for e in events if e["type"] == "put_back"]
    put_of, put_used = {}, set()
    for _, _, i, j in sorted((not at(p, fixture_of.get(g["slot"])), abs(p["t"] - g["t"]), i, j) for i, p in enumerate(puts) for j, g in enumerate(true_puts) if abs(p["t"] - g["t"]) <= pick_tol_s):
        if i not in put_used and j not in put_of:
            put_of[j] = puts[i]
            put_used.add(i)
    put_rows = [{"clip": meta["seed"], **g, "found": j in put_of, "right_slot": j in put_of and slot_of(put_of[j]).get("id") == g["slot"],
                 "right_shopper": j in put_of and shopper_of(put_of[j].get("person_id"), put_of[j]["t"]) == g["shopper"]} for j, g in enumerate(true_puts)]
    # ---- every person of the clip: who was flagged, and what was special about them
    people = []
    for srow in shoppers:
        n, mine = srow["shopper"], [g for g in truth if g["shopper"] == srow["shopper"]]
        recs = [r for r in alert_rows if r["shopper"] == n]
        tags = [t for t, on in (("night", night), ("group_carries_not_pays", any(g.get("paidBy") for g in mine)), ("group_pays_for_another", any(g.get("paidBy") == n for g in truth)),
                                ("scan_dropped", n in (drop.get("paidBy"), drop.get("shopper"))), ("put_back", any(g["outcome"] == "put_back" for g in mine)),
                                ("put_back_other_slot", any(g.get("putBackSlot") for g in mine)), ("touched_a_shelf", any(a["kind"] == "touch" and a["shopper"] == n for a in acts))) if on]
        people.append({"clip": meta["seed"], "shopper": n, "role": srow.get("role") or "shopper", "thief": n in thieves, "group": srow.get("group"),
                       "alerted": any(r["tier"] == "alert" for r in recs), "reviewed": any(r["tier"] == "review" for r in recs), "tags": tags})
    # ---- identity
    names = [s["shopper"] for s in shoppers]
    per = {n: sorted(g for g, obs in seen.items() if Counter(s for _, s in obs).most_common(1)[0][0] == n and len(obs) >= 5) for n in names}
    def two_shoppers(obs) -> bool:        # under 90 percent of an identity's boxes sit on its main shopper
        c = Counter(s for _, s in obs if s != "clerk")
        return bool(c) and c.most_common(1)[0][1] < 0.9 * sum(c.values())
    solid = [obs for obs in seen.values() if len(obs) >= 5]
    mixed = sum(1 for obs in solid if two_shoppers(obs))
    run_info = json.loads((Path(run) / "run.json").read_text()) if (Path(run) / "run.json").exists() else {}
    return {"clip": meta["seed"], "cameras": meta["cameras"], "sim_seconds": meta["sim_seconds"], "shoppers": len(names), "staff": len(staff), "thieves": len(thieves),
            "honest": [n for n in names if n not in thieves and n not in staff], "picks": rows, "thefts": theft_rows, "alerts": alert_rows,
            "people": people, "staff_takes": staff_rows, "touches": touch_rows, "false_picks": false_picks, "put_backs": put_rows,
            "pipeline_put_backs": len(puts), "pipeline_put_backs_paired": len(put_used),
            "features": [k for k, v in (faults.get("features") or {}).items() if v], "planogram_given": "nominal" if meta.get("generator") == 2 else "exact",
            "pipeline_picks": len(picks), "pipeline_picks_paired": len(used),
            "identity": {"store_wide_ids": len(all_ids), "ids_per_shopper": {n: len(v) for n, v in per.items()}, "shoppers_never_tracked": [n for n, v in per.items() if not v],
                         "ids_with_5_boxes": len(solid), "ids_covering_two_shoppers": mixed, "enter_events": sum(1 for e in events if e["type"] == "enter")},
            "pipeline_events": dict(Counter(e["type"] for e in events)), "run": run_info}


def _boot(units, n: int = 2000, seed: int = 0):
    """95 percent interval of sum(num) / sum(den) when the units (shoppers, or clips) are drawn again with replacement."""
    import numpy as np
    a = np.asarray(units, float).reshape(-1, 2)
    if not len(a) or a[:, 1].sum() == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    num, den = a[idx, 0].sum(1), a[idx, 1].sum(1)
    r = num[den > 0] / den[den > 0]
    return [round(float(np.percentile(r, 2.5)), 3), round(float(np.percentile(r, 97.5)), 3)]


def intervals(clips: list[dict]) -> dict:
    """Bootstrap intervals of the headline rates. Rates about people and their picks draw shoppers again; rates about
    the pipeline's own events (precision, identities) have no shopper to hang on and draw clips again."""
    per: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    for c in clips:
        for t in c["thefts"]:
            u = per[c["clip"], t["shopper"]]
            u["thefts"] += 1
            u["alert"] += t["alert"]
            u["flag"] += t["alert_or_review"]
        for p in c["picks"]:
            u, st = per[c["clip"], p["shopper"]], p["stages"]
            u["picks"] += 1
            u["found"] += st["shelf_event_emitted"]
            for k in ("right_slot", "right_sku", "right_shopper"):
                u[k] += st["shelf_event_emitted"] and st[k]
        for g in c.get("put_backs", []):
            if not g["staff"]:
                u = per[c["clip"], g["shopper"]]
                u["puts"] += 1
                u["put_found"] += g["found"]
        for q in c.get("people", []):
            if q["role"] == "shopper" and not q["thief"]:
                u = per[c["clip"], q["shopper"]]
                u["honest"] += 1
                u["h_flag"] += q["alerted"] or q["reviewed"]
                u["h_alert"] += q["alerted"]
    U = list(per.values())
    by = lambda num, den: _boot([(u[num], u[den]) for u in U if u[den]])      # noqa: E731
    ids = lambda c: (c["identity"]["store_wide_ids"], c["shoppers"])      # noqa: E731
    return {"over": "shoppers, 2000 draws; the last four over clips",
            "theft_recall_alert": by("alert", "thefts"), "theft_recall_alert_or_review": by("flag", "thefts"),
            "honest_shoppers_flagged_rate": by("h_flag", "honest"), "honest_shoppers_alerted_rate": by("h_alert", "honest"),
            "pick_recall": by("found", "picks"), "right_slot_of_paired_picks": by("right_slot", "found"), "right_sku_of_paired_picks": by("right_sku", "found"),
            "right_shopper_of_paired_picks": by("right_shopper", "found"), "put_back_recall": by("put_found", "puts"),
            "pick_precision": _boot([(c["pipeline_picks_paired"], c["pipeline_picks"]) for c in clips]),
            "put_back_precision": _boot([(c.get("pipeline_put_backs_paired", 0), c.get("pipeline_put_backs", 0)) for c in clips]),
            "store_wide_ids_per_shopper": _boot([ids(c) for c in clips]),
            "merged_identity_rate": _boot([(c["identity"]["ids_covering_two_shoppers"], c["identity"].get("ids_with_5_boxes", 0)) for c in clips])}


def aggregate(clips: list[dict]) -> dict:
    picks = [p for c in clips for p in c["picks"]]
    thefts = [t for c in clips for t in c["thefts"]]
    alerts = [a for c in clips for a in c["alerts"]]
    tier = lambda t: [a for a in alerts if a["tier"] == t]      # noqa: E731
    al, rv = tier("alert"), tier("review")
    honest = {(c["clip"], n) for c in clips for n in c["honest"]}
    on_honest = lambda rows: sum(1 for a in rows if (a["clip"], a["shopper"]) in honest)      # noqa: E731
    paired = [p for p in picks if p["stages"]["shelf_event_emitted"]]
    tta = [t["time_to_alert_s"] for t in thefts if t["alert"]]
    ids = [n for c in clips for n in c["identity"]["ids_per_shopper"].values()]
    n_shoppers, n_ids = sum(c["shoppers"] for c in clips), sum(c["identity"]["store_wide_ids"] for c in clips)
    pp = sum(c["pipeline_picks"] for c in clips)
    people = [q for c in clips for q in c.get("people", [])]
    hp = [q for q in people if q["role"] == "shopper" and not q["thief"]]
    sp = [q for q in people if q["role"] == "staff"]
    takes, touches, fp = ([x for c in clips for x in c.get(k, [])] for k in ("staff_takes", "touches", "false_picks"))
    puts = [g for c in clips for g in c.get("put_backs", []) if not g["staff"]]
    staff_puts = [g for c in clips for g in c.get("put_backs", []) if g["staff"]]
    ppb, ppb_ok = sum(c.get("pipeline_put_backs", 0) for c in clips), sum(c.get("pipeline_put_backs_paired", 0) for c in clips)
    solid = sum(c["identity"].get("ids_with_5_boxes", 0) for c in clips)
    flagged = lambda q: q["alerted"] or q["reviewed"]      # noqa: E731

    def funnel(rows):
        reached, out = len(rows), []
        for s in STAGES:
            lost = sum(1 for p in rows if p["lost_at"] == s)
            out.append({"stage": s, "passed_on_its_own": sum(1 for p in rows if p["stages"][s]), "reached": reached, "lost_here": lost})
            reached -= lost
        return {"picks": len(rows), "through_every_stage": reached, "stages": out}
    groups = lambda key: {k: funnel([p for p in picks if p[key] == k]) for k in sorted({p[key] for p in picks})}      # noqa: E731

    def pick_group(rows):
        ok = [p for p in rows if p["stages"]["shelf_event_emitted"]]
        return {"picks": len(rows), "found": len(ok), "right_slot": sum(p["stages"]["right_slot"] for p in ok), "right_sku": sum(p["stages"]["right_sku"] for p in ok),
                "right_shopper": sum(p["stages"]["right_shopper"] for p in ok), "through_every_stage": sum(p["lost_at"] is None for p in rows)}
    def tagged(rows, t):          # "none": nothing special about it (night aside)
        return [r for r in rows if not [x for x in r.get("tags", []) if x != "night"]] if t == "none" else [r for r in rows if t in r.get("tags", [])]
    ptags = sorted({t for p in picks for t in p.get("tags", [])}) + ["none"]
    htags = sorted({t for q in hp for t in q["tags"]}) + ["none"]
    sim_s = sum(c["sim_seconds"] for c in clips)
    return {
        "scorecard": {
            "clips": len(clips), "sim_seconds": round(sim_s, 1), "shoppers": n_shoppers - len(sp), "staff_members": len(sp), "thieves": sum(c["thieves"] for c in clips), "honest_shoppers": len(honest),
            "true_picks": len(picks), "stolen_items": len(thefts),
            "planogram_given": "/".join(sorted({c.get("planogram_given", "exact") for c in clips})),
            "theft_recall_alert": _rate(sum(t["alert"] for t in thefts), len(thefts)), "thefts_alerted": sum(t["alert"] for t in thefts),
            "theft_recall_review_only": _rate(sum(t["alert_or_review"] and not t["alert"] for t in thefts), len(thefts)), "thefts_reviewed_only": sum(t["alert_or_review"] and not t["alert"] for t in thefts),
            "theft_recall_alert_or_review": _rate(sum(t["alert_or_review"] for t in thefts), len(thefts)), "thefts_alerted_or_reviewed": sum(t["alert_or_review"] for t in thefts),
            "thefts_alerted_with_the_right_sku": sum(t["sku_in_alert"] for t in thefts), "stolen_items_listed_unpaid": sum(t.get("listed_unpaid", False) for t in thefts),
            "alerts": len(al), "true_alerts": sum(a["true_theft"] for a in al), "alert_precision": _rate(sum(a["true_theft"] for a in al), len(al)),
            "false_alerts_on_honest_shoppers": on_honest(al), "false_alerts_on_nobody": sum(1 for a in al if a["shopper"] in (None, "clerk")),
            "reviews": len(rv), "reviews_on_honest_shoppers": on_honest(rv),
            "honest_shoppers_flagged": sum(map(flagged, hp)), "honest_shoppers_flagged_rate": _rate(sum(map(flagged, hp)), len(hp)),
            "honest_shoppers_alerted": sum(q["alerted"] for q in hp), "honest_shoppers_alerted_rate": _rate(sum(q["alerted"] for q in hp), len(hp)),
            "honest_shoppers_reviewed_only": sum(q["reviewed"] and not q["alerted"] for q in hp),
            "staff_members_flagged": sum(map(flagged, sp)), "staff_members_alerted": sum(q["alerted"] for q in sp),
            "staff_takes": len(takes), "staff_takes_with_a_pick_event": sum(x["pick_event"] for x in takes), "staff_takes_listed_unpaid": sum(bool(x["listed_unpaid"]) for x in takes),
            "staff_puts": len(staff_puts), "staff_puts_with_a_put_event": sum(g["found"] for g in staff_puts),
            "touches": len(touches), "touches_counted_as_picks": sum(x["counted_as_pick"] for x in touches), "touches_listed_unpaid": sum(bool(x["listed_unpaid"]) for x in touches),
            "false_alerts_per_hour": round((len(al) - sum(a["true_theft"] for a in al)) / sim_s * 3600, 1) if sim_s else None,
            "pipeline_picks": pp, "pipeline_picks_paired": sum(c["pipeline_picks_paired"] for c in clips), "false_picks_listed_unpaid": sum(bool(x["listed_unpaid"]) for x in fp),
            "pick_recall": _rate(len(paired), len(picks)), "picks_found": len(paired), "pick_precision": _rate(sum(c["pipeline_picks_paired"] for c in clips), pp),
            "right_sku_of_paired_picks": _rate(sum(p["stages"]["right_sku"] for p in paired), len(paired)),
            "right_sku_of_all_true_picks": _rate(sum(p["stages"]["right_sku"] for p in paired), len(picks)),
            "right_slot_of_paired_picks": _rate(sum(p["stages"]["right_slot"] for p in paired), len(paired)),
            "right_shopper_of_paired_picks": _rate(sum(p["stages"]["right_shopper"] for p in paired), len(paired)),
            "put_backs": len(puts), "put_backs_found": sum(g["found"] for g in puts), "put_back_recall": _rate(sum(g["found"] for g in puts), len(puts)),
            "put_backs_into_another_slot": sum(g["other_slot"] for g in puts), "put_backs_into_another_slot_found": sum(g["found"] for g in puts if g["other_slot"]),
            "put_backs_found_with_the_right_slot": sum(g["right_slot"] for g in puts),
            "pipeline_put_backs": ppb, "pipeline_put_backs_paired": ppb_ok, "put_back_precision": _rate(ppb_ok, ppb),
            "time_to_alert_s": {"median": round(statistics.median(tta), 2), "max": max(tta)} if tta else None,
            "store_wide_ids_per_shopper": _rate(n_ids, n_shoppers), "store_wide_ids": n_ids, "people": n_shoppers,
            "ids_on_one_shopper_mean": round(statistics.mean(ids), 2) if ids else None, "ids_on_one_shopper_max": max(ids, default=None),
            "shoppers_never_tracked": sum(len(c["identity"]["shoppers_never_tracked"]) for c in clips),
            "ids_covering_two_shoppers": sum(c["identity"]["ids_covering_two_shoppers"] for c in clips), "ids_with_5_boxes": solid,
            "merged_identity_rate": _rate(sum(c["identity"]["ids_covering_two_shoppers"] for c in clips), solid),
        },
        "intervals": intervals(clips),
        "funnel": funnel(picks), "funnel_by_outcome": groups("outcome"), "funnel_by_zone": groups("zone"),
        "by_camera_mount": {m: pick_group([p for p in picks if p.get("camera_mount") == m]) | {"false_pick_events": sum(1 for x in fp if x["mount"] == m)}
                            for m in sorted({p.get("camera_mount", "none") for p in picks} | {x["mount"] for x in fp})},
        "picks_by_tag": {t: pick_group(tagged(picks, t)) for t in ptags},
        "thefts_by_tag": {t: {"stolen_items": len(r), "alert": sum(x["alert"] for x in r), "alert_or_review": sum(x["alert_or_review"] for x in r)} for t in ptags for r in [tagged(thefts, t)] if r},
        "honest_shoppers_by_tag": {t: {"honest_shoppers": len(r), "alerted": sum(q["alerted"] for q in r), "flagged": sum(map(flagged, r))} for t in htags for r in [tagged(hp, t)]},
        "clips_by_option": {f: _option(clips, f) for f in sorted({f for c in clips for f in c.get("features", [])})},
        "detail_counts": {k: sum(1 for p in picks if p["detail"][k]) for k in ("item_detected", "right_sku_detected", "person_detected", "hand_detected", "right_fixture")}
        | {"picks_with_item_frames": sum(1 for p in picks if p["detail"]["item_frames"]), "picks_with_item_frames_reached": sum(1 for p in picks if p["detail"]["item_frames_reached"])},
    }


def _option(clips: list[dict], f: str) -> dict:
    """The same four numbers on the clips that have a scenario option and on the ones that do not (few clips: read as a hint)."""
    def four(cs):
        th, pk = [t for c in cs for t in c["thefts"]], [p for c in cs for p in c["picks"]]
        hp = [q for c in cs for q in c.get("people", []) if q["role"] == "shopper" and not q["thief"]]
        return {"clips": len(cs), "thefts_flagged": f"{sum(t['alert_or_review'] for t in th)}/{len(th)}", "honest_flagged": f"{sum(q['alerted'] or q['reviewed'] for q in hp)}/{len(hp)}",
                "picks_found": f"{sum(p['stages']['shelf_event_emitted'] for p in pk)}/{len(pk)}",
                "ids_per_person": _rate(sum(c["identity"]["store_wide_ids"] for c in cs), sum(c["shoppers"] for c in cs))}
    return {"with": four([c for c in clips if f in c.get("features", [])]), "without": four([c for c in clips if f not in c.get("features", [])])}


def _score_rows(s: dict, ci: dict) -> list[tuple[str, str]]:
    pct = lambda v: "n/a" if v is None else f"{v:.1%}"      # noqa: E731
    iv = lambda k: "" if not ci or not ci.get(k) else f" [{ci[k][0]:.2f} to {ci[k][1]:.2f}]"      # noqa: E731
    frac = lambda n, d: f"{n} of {d}" + (f" ({n / d:.1%})" if d else "")      # noqa: E731
    tta, g = s["time_to_alert_s"], s.get
    return [
        ("Thefts flagged, alert tier", frac(s["thefts_alerted"], s["stolen_items"]) + iv("theft_recall_alert")),
        ("Thefts flagged, review tier only", frac(g("thefts_reviewed_only", s["thefts_alerted_or_reviewed"] - s["thefts_alerted"]), s["stolen_items"])),
        ("Thefts flagged, alert or review", frac(s["thefts_alerted_or_reviewed"], s["stolen_items"]) + iv("theft_recall_alert_or_review")),
        ("Stolen items listed as unpaid on a record", frac(g("stolen_items_listed_unpaid", 0), s["stolen_items"])),
        ("Alert precision", f"{pct(s['alert_precision'])} ({s['true_alerts']}/{s['alerts']})"),
        ("Honest shoppers flagged, alert or review", frac(g("honest_shoppers_flagged", 0), s["honest_shoppers"]) + iv("honest_shoppers_flagged_rate")),
        ("Honest shoppers flagged, alert tier", frac(g("honest_shoppers_alerted", 0), s["honest_shoppers"]) + iv("honest_shoppers_alerted_rate")),
        ("Honest shoppers flagged, review tier only", frac(g("honest_shoppers_reviewed_only", 0), s["honest_shoppers"])),
        ("Staff members flagged, alert or review (of them alert tier)", f"{g('staff_members_flagged', 0)} of {g('staff_members', 0)} ({g('staff_members_alerted', 0)})"),
        ("Staff takes listed as unpaid on a record", f"{g('staff_takes_listed_unpaid', 0)} of {g('staff_takes', 0)} (a PICK event on {g('staff_takes_with_a_pick_event', 0)})"),
        ("Shifted items counted as a pick (of them listed as unpaid)", f"{g('touches_counted_as_picks', 0)} of {g('touches', 0)} ({g('touches_listed_unpaid', 0)})"),
        ("Picks found", frac(g("picks_found", round((s["pick_recall"] or 0) * s["true_picks"])), s["true_picks"]) + iv("pick_recall")),
        ("Pick precision (staff takes count as true)", f"{pct(s['pick_precision'])} of {s['pipeline_picks']} PICK events" + iv("pick_precision")),
        ("Right slot, of paired picks", pct(s["right_slot_of_paired_picks"]) + iv("right_slot_of_paired_picks")),
        (f"Right SKU, of paired picks ({s.get('planogram_given', 'exact')} planogram given)", pct(s["right_sku_of_paired_picks"]) + iv("right_sku_of_paired_picks") + f"; of all true picks {pct(s['right_sku_of_all_true_picks'])}"),
        ("Right shopper, of paired picks", pct(s["right_shopper_of_paired_picks"]) + iv("right_shopper_of_paired_picks")),
        ("Put-backs found (recall)", frac(g("put_backs_found", 0), g("put_backs", 0)) + iv("put_back_recall")),
        ("Put-back precision (staff puts count as true)", f"{pct(g('put_back_precision'))} of {g('pipeline_put_backs', 0)} PUT_BACK events" + iv("put_back_precision")),
        ("Put-backs into another slot found", f"{g('put_backs_into_another_slot_found', 0)} of {g('put_backs_into_another_slot', 0)}"),
        ("Identities per person", f"{s['store_wide_ids_per_shopper']} ({s['store_wide_ids']} for {g('people', s['shoppers'])})" + iv("store_wide_ids_per_shopper")),
        ("Identities that cover two people", f"{s['ids_covering_two_shoppers']} of {g('ids_with_5_boxes', '?')} ({pct(g('merged_identity_rate'))})" + iv("merged_identity_rate")),
        ("People never tracked", str(s["shoppers_never_tracked"])),
        ("False alerts per hour", str(s["false_alerts_per_hour"])),
        ("Time to alert after concealment", "n/a" if not tta else f"{tta['median']} s median, {tta['max']} s max"),
    ]


def markdown(res: dict) -> str:
    s, st = res["scorecard"], res.get("stress")
    L = [f"# Benchmark, {res['split']} split", "", NOTE, "",
         f"{s['clips']} clips ({', '.join(str(c['clip']) for c in res['clips'])}), {s['sim_seconds']:.0f} s of sim time, {s['shoppers']} shoppers "
         f"({s['thieves']} thieves, {s['honest_shoppers']} honest), {s.get('staff_members', 0)} staff, {s['true_picks']} picks, {s['stolen_items']} stolen items. "
         f"Runner `{res['runner']}`, options {json.dumps(res['options'])}. Scored {res['scored_at']}, commit {res['commit']}.", "",
         "## Scorecard", "",
         "Square brackets: 95 percent bootstrap interval (shoppers drawn again 2000 times; clips for precision and identities). "
         "The sample is small, so read the interval before the point value."]
    if st:
        L += ["", f"Stressed: the same clips and pipeline with worse inputs, {json.dumps(st['settings'])} (`python -m bree.sim.bench --help`). No ground truth is used to make them worse."]
    plain, hard = _score_rows(s, res.get("intervals")), _score_rows(st["scorecard"], st.get("intervals")) if st else None
    same = st and st.get("plain_on_the_same_clips")
    mid = _score_rows(same["scorecard"], same.get("intervals")) if same else None
    n = st["scorecard"]["clips"] if st else 0
    L += ["", "| Metric | Value |" + (f" Plain, the {n} clips that were also stressed |" if same else "") + (f" Stressed ({n} clips) |" if st else ""), "|---|---|" + ("---|" if same else "") + ("---|" if st else "")]
    L += [f"| {k} | {v} |" + (f" {mid[i][1]} |" if same else "") + (f" {hard[i][1]} |" if st else "") for i, (k, v) in enumerate(plain)] + [""]

    def table(f, title):
        out = [f"### {title}: {f['picks']} picks, {f['through_every_stage']} through every stage", "",
               "| stage | reached this stage | lost here | passed on its own |", "|---|---|---|---|"]
        return out + [f"| {r['stage'].replace('_', ' ')} | {r['reached']} | {r['lost_here']} | {r['passed_on_its_own']} |" for r in f["stages"]] + [""]
    L += ["## Funnel: where each true pick is lost", "",
          "Each true pick walks the stages in order and is counted at the first one it fails. \"Passed on its own\" counts the stage "
          "for every pick, whatever happened before it.", ""]
    L += table(res["funnel"], "All picks")
    if st:
        L += table(st["funnel"], "All picks, stressed")
    for key, title in (("funnel_by_outcome", "Outcome"), ("funnel_by_zone", "Zone")):
        for k, f in res[key].items():
            L += table(f, f"{title} {k}")
    L += ["What a stage means:", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in STAGE_HELP.items()] + [""]
    name = lambda t: t.replace("_", " ")      # noqa: E731
    if res.get("picks_by_tag"):
        L += ["## What makes it hard", "", "A pick, a theft or a shopper can carry several tags; \"none\" has none of them (night aside).", "",
              "| picks | n | found | right slot | right SKU | right shopper | through every stage |", "|---|---|---|---|---|---|---|"]
        L += [f"| {name(t)} | {r['picks']} | {r['found']} | {r['right_slot']} | {r['right_sku']} | {r['right_shopper']} | {r['through_every_stage']} |" for t, r in res["picks_by_tag"].items() if r["picks"]]
        L += ["", "| stolen items | n | alert tier | alert or review |", "|---|---|---|---|"]
        L += [f"| {name(t)} | {r['stolen_items']} | {r['alert']} | {r['alert_or_review']} |" for t, r in res["thefts_by_tag"].items()]
        L += ["", "| honest shoppers | n | flagged, alert tier | flagged, alert or review |", "|---|---|---|---|"]
        L += [f"| {name(t)} | {r['honest_shoppers']} | {r['alerted']} | {r['flagged']} |" for t, r in res["honest_shoppers_by_tag"].items() if r["honest_shoppers"]]
        if res.get("clips_by_option"):
            L += ["", "Clips with a scenario option against clips without it (whole clips, so other options are mixed in: a hint, not a measurement):", "",
                  "| option | clips with, without | thefts flagged | honest shoppers flagged | picks found | identities per person |", "|---|---|---|---|---|---|"]
            L += [f"| {f} | {r['with']['clips']}, {r['without']['clips']} | {r['with']['thefts_flagged']}, {r['without']['thefts_flagged']} | {r['with']['honest_flagged']}, {r['without']['honest_flagged']} | "
                  f"{r['with']['picks_found']}, {r['without']['picks_found']} | {r['with']['ids_per_person']}, {r['without']['ids_per_person']} |" for f, r in res["clips_by_option"].items()]
        L += ["", "## By camera kind (the mount of the item camera with the best view of the pick)", "",
              "| mount | picks | found | right slot | right SKU | right shopper | false PICK events from this mount |", "|---|---|---|---|---|---|---|"]
        L += [f"| {m} | {r['picks']} | {r['found']} | {r['right_slot']} | {r['right_sku']} | {r['right_shopper']} | {r['false_pick_events']} |" for m, r in res["by_camera_mount"].items()] + [""]
    d = res["detail_counts"]
    L += ["## Detection detail (independent counts over all true picks)", "", "| | picks |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in d.items()] + [""]
    L += ["## Per clip", "", "| clip | options | cameras | shoppers | thieves | picks | PICK events | stolen | flagged | alerts | false | honest flagged | ids | wall s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["clips"]:
        al = [a for a in c["alerts"] if a["tier"] == "alert"]
        w = (c.get("run") or {}).get("wall_s") or {}
        hp = [q for q in c.get("people", []) if q["role"] == "shopper" and not q["thief"]]
        L.append(f"| {c['clip']} | {' '.join(c.get('features', []))} | {len(c['cameras'])} | {c['shoppers'] - c.get('staff', 0)} | {c['thieves']} | {len(c['picks'])} | {c['pipeline_picks']} | {len(c['thefts'])} | "
                 f"{sum(t['alert_or_review'] for t in c['thefts'])} | {len(al)} | {sum(not a['true_theft'] for a in al)} | {sum(q['alerted'] or q['reviewed'] for q in hp)} of {len(hp)} | {c['identity']['store_wide_ids']} | {w.get('pipeline', '')} |")
    L += ["", "## Every true pick", "", "| clip | t | shopper | zone | slot | sku | outcome | tags | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["clips"]:
        for p in c["picks"]:
            d = p["detail"]
            L.append(f"| {p['clip']} | {p['t']} | {p['shopper']} | {p['zone']} | {p['slot']} | {p['sku']} | {p['outcome']} | {' '.join(p.get('tags', []))} | {(p['lost_at'] or 'none').replace('_', ' ')} | "
                     f"{d['item_frames']} ({d['item_frames_reached']}) | {'y' if d['item_detected'] else ''} | {'y' if d['right_sku_detected'] else ''} | {'y' if d['person_detected'] else ''} | {d['event_sku'] or ''} | {d['event_shopper'] or ''} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ command
def _one(args) -> None:
    """Child process: run the pipeline on one clip (public view only)."""
    clip, out = Path(args.run_one[0]), Path(args.run_one[1])
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    mod, fn = args.runner.split(":")
    kw = {"max_frames": args.max_frames} if args.max_frames else {}
    if args.runner == "bree.sim.bench:run_pipeline":
        kw |= {"backend": args.backend, "edge": not args.no_edge, "verbose": args.verbose}
    elif args.runner == "bree.shelf.store:run":
        kw |= {"verbose": args.verbose}
    pub = public_view(clip, out, json.loads(args.stress_json) if args.stress_json else None)
    if args.reuse and (Path(args.reuse) / "pipeline" / "shelf_events.jsonl").exists():
        reuse_plain(Path(args.reuse) / "pipeline", out / "pipeline", set(json.loads((pub / "clip.json").read_text())["cameras"]))
    getattr(importlib.import_module(mod), fn)(pub, out, **kw)


def reuse_plain(plain: Path, pipe: Path, cameras: set) -> None:
    """A stress that leaves the pictures, the poses and the planogram alone (cameras missing, late receipts) changes
    nothing a remaining camera reports: each camera is read by itself. So the stored person boxes and the shelf events
    and looks of the remaining cameras of the plain run are that stressed run's own, and only the join runs again."""
    (pipe / "conceal").mkdir(parents=True, exist_ok=True)
    for f in [*plain.glob("people_*.jsonl"), *(plain / "conceal").glob("looks_*.jsonl")]:
        if f.stem.split("_", 1)[1] in cameras and not (pipe / f.relative_to(plain)).exists():
            (pipe / f.relative_to(plain)).symlink_to(f.resolve())
    for name in ("shelf_events.jsonl", "shelf_status.jsonl"):
        if (plain / name).exists():
            (pipe / name).write_text("".join(json.dumps(r) + "\n" for r in _jsonl(plain / name) if r["camera_id"] in cameras))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", nargs="?", choices=sorted(manifest()["splits"]))
    ap.add_argument("--clips", help="comma separated seeds (default: the whole split)")
    ap.add_argument("--batch", type=int, help="checkpoint pool: which rendered batch, 1 is the first (default: the latest)")
    ap.add_argument("--jobs", type=int, default=2, help="clips run at the same time")
    ap.add_argument("--out", help="default: out/bench/<split>")
    ap.add_argument("--runner", default="bree.shelf.store:run", help="module:function(clip, out, **options) that runs the pipeline on one clip. "
                    "Default: shelf events and floor tracks (bree.shelf.store). bree.sim.bench:run_pipeline is the per-camera engine it replaced here")
    ap.add_argument("--keep", action="store_true", help="do not empty the run folders first: bree.shelf.store reuses its stored shelf events and person boxes "
                    "and repeats only tracking, association and the ledger")
    ap.add_argument("--backend", default="sim_sku")
    ap.add_argument("--no-edge", action="store_true", help="skip the camera nodes and hub: every camera streams every frame")
    ap.add_argument("--max-frames", type=int, default=None, help="per camera (smoke test)")
    ap.add_argument("--smoke", action="store_true", help="first clip of the split, 60 frames per camera, results/ untouched")
    ap.add_argument("--score-only", action="store_true", help="score finished runs again, no pipeline run")
    ap.add_argument("--name", default="", help="write results/bench_<split>_<name>.* instead of results/bench_<split>.*")
    ap.add_argument("--stress", action="store_true", help="also run and score the stressed inputs, at " + json.dumps(STRESS_DEFAULT) + " unless the flags below say otherwise")
    ap.add_argument("--stress-only", action="store_true", help="with --stress or its flags: do not run the plain inputs again (their finished runs are scored as they are)")
    ap.add_argument("--drop-item-cameras", type=float, help="stress: share of the item cameras that is missing (the register camera stays)")
    ap.add_argument("--calib-noise-deg", type=float, help="stress: error of every camera pose, degrees per axis (one standard deviation)")
    ap.add_argument("--register-delay-s", type=float, help="stress: every receipt arrives this much later")
    ap.add_argument("--wrong-planogram", type=float, help="stress: share of the slots whose planogram entry names another product")
    ap.add_argument("--stress-seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run-one", nargs=2, metavar=("CLIP", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("--stress-json", help=argparse.SUPPRESS)
    ap.add_argument("--reuse", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.run_one:
        return _one(a)
    if not a.split:
        ap.error("give a split: " + ", ".join(sorted(manifest()["splits"])))
    given = {k: getattr(a, k) for k in STRESS_DEFAULT if getattr(a, k) is not None}
    stress = ({**(STRESS_DEFAULT if a.stress else {}), **given, "seed": a.stress_seed}) if a.stress or given else None
    only = [int(s) for s in a.clips.split(",")] if a.clips else None
    clips = clip_dirs(a.split, only, a.batch)
    if a.smoke:
        clips, a.max_frames = clips[:1], a.max_frames or 60
    if a.score_only and any(not (c / "clip.json").exists() for c in clips):      # the split is still rendering: score what is there, leave results/ alone
        clips, only = [c for c in clips if (c / "clip.json").exists()], only or [-1]
    missing = [c for c in clips if not (c / "clip.json").exists()]
    if missing:
        raise SystemExit(f"not rendered: {', '.join(str(c) for c in missing)}\nrender with: .venv/bin/python scripts/bench/render_split.py {a.split}")
    batch = f"_{a.batch or len(manifest()['splits'][a.split]['batches'])}" if "batches" in manifest()["splits"][a.split] else ""
    tag = a.split + batch + (f"_{a.name}" if a.name else "")
    out = Path(a.out or ROOT / "out" / "bench" / tag)
    hard = Path(str(out) + "_stress")
    t0 = time.time()

    def run_all(folder: Path, st: dict | None) -> None:
        def go(clip: Path) -> int:
            run = folder / clip.name
            cmd = [sys.executable, "-m", "bree.sim.bench", "--run-one", str(clip), str(run), "--runner", a.runner, "--backend", a.backend]
            cmd += (["--no-edge"] if a.no_edge else []) + (["--max-frames", str(a.max_frames)] if a.max_frames else []) + (["--verbose"] if a.verbose else []) + (["--keep"] if a.keep else [])
            cmd += ["--stress-json", json.dumps(st)] if st else []
            if st and not st.get("calib_noise_deg") and not st.get("wrong_planogram") and a.runner == "bree.shelf.store:run" and not a.max_frames:
                cmd += ["--reuse", str(out / clip.name)]
            run.parent.mkdir(parents=True, exist_ok=True)
            with (folder / f"{clip.name}.log").open("w") as log:
                code = subprocess.run(cmd, stdout=log, stderr=log).returncode
            print(f"[bench {time.time() - t0:5.0f} s] {folder.name}/{clip.name}: {'done' if code == 0 else 'FAILED, see ' + str(folder / (clip.name + '.log'))}", flush=True)
            return code
        with ThreadPoolExecutor(a.jobs) as ex:
            if any(list(ex.map(go, clips))):
                raise SystemExit("a clip failed; nothing scored")
    if not a.score_only:
        if not (stress and a.stress_only):
            run_all(out, None)
        if stress:
            run_all(hard, stress)
    ran = lambda folder, c: (folder / c.name / "run.json").exists() and (folder / c.name / "pipeline" / "events.jsonl").exists()      # noqa: E731
    if a.score_only and not all(ran(out, c) for c in clips):      # runs still going: score the finished ones, leave results/ alone
        print("no finished run yet for: " + ", ".join(c.name for c in clips if not ran(out, c)))
        clips, only = [c for c in clips if ran(out, c)], only or [-1]
    scored = [score_clip(c, out / c.name) for c in clips]
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip())
    res = {"split": a.split + batch.replace("_", " batch "), "data": NOTE, "runner": a.runner, "options": {"backend": a.backend, "edge": not a.no_edge, "max_frames": a.max_frames},
           "scored_at": time.strftime("%Y-%m-%d %H:%M"), "commit": commit + (" plus uncommitted changes in src" if dirty else ""),
           "wall_s": round(time.time() - t0, 1), **aggregate(scored), "stage_help": STAGE_HELP, "clips": scored}
    if stress and any(ran(hard, c) for c in clips):
        worse = [score_clip(c, hard / c.name) for c in clips if ran(hard, c)]
        res["stress"] = {"settings": stress, "runs": str(hard.relative_to(ROOT)) if hard.is_relative_to(ROOT) else str(hard), **aggregate(worse), "clips": worse}
        if len(worse) < len(scored):      # the stressed runs cover fewer clips: the plain score of those same clips goes beside them
            res["stress"]["plain_on_the_same_clips"] = aggregate([c for c in scored if c["clip"] in {w["clip"] for w in worse}])
    full = not a.max_frames and not only
    targets = [out / "bench"] + ([ROOT / "results" / f"bench_{tag}"] if full else [])
    for t in targets:
        t.with_suffix(".json").write_text(json.dumps(res, indent=1))
        t.with_suffix(".md").write_text(markdown(res))
    print(markdown(res).split("## Per clip")[0])
    print("written: " + ", ".join(str(t) + ".json and .md" for t in targets) + ("" if full else "  (partial run: results/ left alone)"))


if __name__ == "__main__":
    main()
