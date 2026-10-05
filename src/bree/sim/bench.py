"""The fixed benchmark: run the pipeline on a split of SIMULATED store clips and score it against ground truth.

    python -m bree.sim.bench dev            (make bench-dev)    -> results/bench_dev.json and .md
    python -m bree.sim.bench test           (make bench-test)   -> results/bench_test.json and .md  (final numbers only)
    python -m bree.sim.bench dev --smoke    one clip, 60 frames per camera, nothing written to results/
    python -m bree.sim.bench dev --score-only                   rescore finished runs (out/bench/dev/clip_*)

Clips: scripts/bench/manifest.json and scripts/bench/README.md. A clip folder holds what the pipeline may read
(<camera>.mp4, calibration.json, layout.json with the planogram, register.jsonl, clip.json) and a truth/ folder.
The pipeline is run on a folder of links that leaves truth/ out (`public_view`), so the runner cannot read it
by accident. Only `score_clip` opens truth/.

The runner is one function, `run_pipeline(clip, out)` (replace it with --runner module:function). What the
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


def clip_dirs(split: str, only: list[int] | None = None) -> list[Path]:
    m = manifest()
    seeds = m["splits"][split]["seeds"]
    if not isinstance(seeds, list):
        raise SystemExit(f"{split} is a reserved seed range, not a rendered split")
    return [ROOT / m["root"] / split / f"clip_{s}" for s in seeds if not only or s in only]


def load_calibration(clip: Path) -> dict:
    """{camera id: bree.calib.camera.Camera} from a clip's calibration.json (true pose, roll included)."""
    import numpy as np
    from bree.calib.camera import Camera
    return {c["id"]: Camera(c["fx"], c["cx"], c["cy"], np.asarray(c["R"], float), np.asarray(c["position"], float), tuple(c["resolution"]))
            for c in json.loads((Path(clip) / "calibration.json").read_text())["cameras"]}


def public_view(clip: Path, work: Path) -> Path:
    """Links to everything in the clip except the ground truth. This is the folder the pipeline is given."""
    pub = work / "clip"
    shutil.rmtree(pub, ignore_errors=True)
    pub.mkdir(parents=True)
    for f in Path(clip).iterdir():
        if f.is_file() and f.name != "render.log":
            (pub / f.name).symlink_to(f.resolve())
    return pub


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
    pairs = sorted(((not (p.get("zone") == g["fixtureId"] or slot_of(p).get("fixture") == g["fixtureId"]), abs(p["t"] - g["t"]), i, j)
                    for i, p in enumerate(picks) for j, g in enumerate(truth) if abs(p["t"] - g["t"]) <= pick_tol_s))
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
            st["conceal_or_pay_classified"] = any(e["type"] == want and lo <= e["t"] <= hi and shopper_of(e.get("person_id"), e["t"]) == g["shopper"] for e in events)
        mine = [(a, it) for a, it in unpaid if ev and abs(it["t_pick"] - ev["t"]) < 0.06 and ev.get("person_id") in [a["person_id"], *a.get("group", [])]]
        if g["outcome"] in ("concealed", "theft"):
            st["ledger_basket"] = bool(mine)
            st["alert"] = any(a["tier"] == "alert" for a, _ in mine)
        else:
            st["ledger_basket"] = not mine
            st["alert"] = not any(a["_shopper"] == g["shopper"] for a in alerts)
        lost = next((s for s in STAGES if not st[s]), None)
        rows.append({"clip": meta["seed"], "shopper": g["shopper"], "t": g["t"], "sku": g["skuId"], "slot": g["slotId"], "fixture": g["fixtureId"],
                     "zone": g["zone"], "outcome": g["outcome"], "stages": st, "lost_at": lost,
                     "detail": {"item_frames": len(item_fr), "hand_frames": len(hand_fr), "item_frames_reached": len(reached_i), "hand_frames_reached": len(reached_h),
                                "item_detected": item_det, "right_sku_detected": sku_det, "person_detected": person_det, "hand_detected": hand_det,
                                "event_t": ev["t"] if ev else None, "event_zone": ev.get("zone") if ev else None, "event_sku": claim,
                                "event_slot": slot_of(ev).get("id") if ev else None, "right_fixture": bool(ev) and (ev.get("zone") == g["fixtureId"] or slot_of(ev).get("fixture") == g["fixtureId"]),
                                "event_person": pid, "event_shopper": who,
                                "best_item_camera": next((c["id"] for c in g["cameras"] if c["id"] in meta["cameras"] and kind.get(c["id"]) in ITEM_KINDS), None)}})

    # ---- thefts and alerts
    stolen = [g for g in truth if g["outcome"] == "concealed"]
    thieves = {g["shopper"] for g in stolen}
    first_conceal = {s: min(g["tConceal"] for g in stolen if g["shopper"] == s) for s in thieves}
    alert_rows = []
    for a in alerts:
        s = a["_shopper"]
        true = s in thieves and first_conceal[s] <= a["t_emitted"]
        alert_rows.append({"clip": meta["seed"], "alert_id": a["alert_id"], "tier": a["tier"], "person_id": a["person_id"], "shopper": s, "true_theft": true,
                           "t_emitted": a["t_emitted"], "skus": [it.get("sku") or it["category"] for it in a["unpaid_items"]], "confidence": a["confidence"]})
    theft_rows = []
    for g in stolen:
        hits = [r for r in alert_rows if r["shopper"] == g["shopper"] and r["t_emitted"] >= g["tConceal"]]
        first = min((r for r in hits if r["tier"] == "alert"), key=lambda r: r["t_emitted"], default=None)
        theft_rows.append({"clip": meta["seed"], "shopper": g["shopper"], "sku": g["skuId"], "slot": g["slotId"], "t_pick": g["t"], "t_conceal": g["tConceal"],
                           "alert": first is not None, "alert_or_review": bool(hits), "sku_in_alert": bool(first) and g["skuId"] in first["skus"],
                           "time_to_alert_s": round(first["t_emitted"] - g["tConceal"], 2) if first else None})
    # ---- identity
    names = [s["shopper"] for s in shoppers]
    per = {n: sorted(g for g, obs in seen.items() if Counter(s for _, s in obs).most_common(1)[0][0] == n and len(obs) >= 5) for n in names}
    def two_shoppers(obs) -> bool:        # under 90 percent of an identity's boxes sit on its main shopper
        c = Counter(s for _, s in obs if s != "clerk")
        return bool(c) and c.most_common(1)[0][1] < 0.9 * sum(c.values())
    mixed = sum(1 for obs in seen.values() if len(obs) >= 5 and two_shoppers(obs))
    run_info = json.loads((Path(run) / "run.json").read_text()) if (Path(run) / "run.json").exists() else {}
    return {"clip": meta["seed"], "cameras": meta["cameras"], "sim_seconds": meta["sim_seconds"], "shoppers": len(names), "thieves": len(thieves),
            "honest": [n for n in names if n not in thieves], "picks": rows, "thefts": theft_rows, "alerts": alert_rows,
            "pipeline_picks": len(picks), "pipeline_picks_paired": len(used),
            "identity": {"store_wide_ids": len(all_ids), "ids_per_shopper": {n: len(v) for n, v in per.items()}, "shoppers_never_tracked": [n for n, v in per.items() if not v],
                         "ids_covering_two_shoppers": mixed, "enter_events": sum(1 for e in events if e["type"] == "enter")},
            "pipeline_events": dict(Counter(e["type"] for e in events)), "run": run_info}


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

    def funnel(rows):
        reached, out = len(rows), []
        for s in STAGES:
            lost = sum(1 for p in rows if p["lost_at"] == s)
            out.append({"stage": s, "passed_on_its_own": sum(1 for p in rows if p["stages"][s]), "reached": reached, "lost_here": lost})
            reached -= lost
        return {"picks": len(rows), "through_every_stage": reached, "stages": out}
    groups = lambda key: {k: funnel([p for p in picks if p[key] == k]) for k in sorted({p[key] for p in picks})}      # noqa: E731
    sim_s = sum(c["sim_seconds"] for c in clips)
    return {
        "scorecard": {
            "clips": len(clips), "sim_seconds": round(sim_s, 1), "shoppers": n_shoppers, "thieves": sum(c["thieves"] for c in clips), "honest_shoppers": len(honest),
            "true_picks": len(picks), "stolen_items": len(thefts),
            "theft_recall_alert": _rate(sum(t["alert"] for t in thefts), len(thefts)), "thefts_alerted": sum(t["alert"] for t in thefts),
            "theft_recall_alert_or_review": _rate(sum(t["alert_or_review"] for t in thefts), len(thefts)), "thefts_alerted_or_reviewed": sum(t["alert_or_review"] for t in thefts),
            "thefts_alerted_with_the_right_sku": sum(t["sku_in_alert"] for t in thefts),
            "alerts": len(al), "true_alerts": sum(a["true_theft"] for a in al), "alert_precision": _rate(sum(a["true_theft"] for a in al), len(al)),
            "false_alerts_on_honest_shoppers": on_honest(al), "false_alerts_on_nobody": sum(1 for a in al if a["shopper"] in (None, "clerk")),
            "reviews": len(rv), "reviews_on_honest_shoppers": on_honest(rv),
            "false_alerts_per_hour": round((len(al) - sum(a["true_theft"] for a in al)) / sim_s * 3600, 1) if sim_s else None,
            "pipeline_picks": pp, "pick_recall": _rate(len(paired), len(picks)), "pick_precision": _rate(len(paired), pp),
            "right_sku_of_paired_picks": _rate(sum(p["stages"]["right_sku"] for p in paired), len(paired)),
            "right_sku_of_all_true_picks": _rate(sum(p["stages"]["right_sku"] for p in paired), len(picks)),
            "right_slot_of_paired_picks": _rate(sum(p["stages"]["right_slot"] for p in paired), len(paired)),
            "right_shopper_of_paired_picks": _rate(sum(p["stages"]["right_shopper"] for p in paired), len(paired)),
            "time_to_alert_s": {"median": round(statistics.median(tta), 2), "max": max(tta)} if tta else None,
            "store_wide_ids_per_shopper": _rate(n_ids, n_shoppers), "store_wide_ids": n_ids,
            "ids_on_one_shopper_mean": round(statistics.mean(ids), 2) if ids else None, "ids_on_one_shopper_max": max(ids, default=None),
            "shoppers_never_tracked": sum(len(c["identity"]["shoppers_never_tracked"]) for c in clips),
            "ids_covering_two_shoppers": sum(c["identity"]["ids_covering_two_shoppers"] for c in clips),
        },
        "funnel": funnel(picks), "funnel_by_outcome": groups("outcome"), "funnel_by_zone": groups("zone"),
        "detail_counts": {k: sum(1 for p in picks if p["detail"][k]) for k in ("item_detected", "right_sku_detected", "person_detected", "hand_detected", "right_fixture")}
        | {"picks_with_item_frames": sum(1 for p in picks if p["detail"]["item_frames"]), "picks_with_item_frames_reached": sum(1 for p in picks if p["detail"]["item_frames_reached"])},
    }


def markdown(res: dict) -> str:
    s, pct = res["scorecard"], lambda v: "n/a" if v is None else f"{v:.1%}"      # noqa: E731
    tta = s["time_to_alert_s"]
    L = [f"# Benchmark, {res['split']} split", "", NOTE, "",
         f"{s['clips']} clips ({', '.join(str(c['clip']) for c in res['clips'])}), {s['sim_seconds']:.0f} s of sim time, {s['shoppers']} shoppers "
         f"({s['thieves']} thieves, {s['honest_shoppers']} honest), {s['true_picks']} picks, {s['stolen_items']} stolen items. "
         f"Runner `{res['runner']}`, options {json.dumps(res['options'])}. Scored {res['scored_at']}, commit {res['commit']}.", "",
         "## Scorecard", "", "| Metric | Value |", "|---|---|",
         f"| Theft recall, alert tier | {pct(s['theft_recall_alert'])} ({s['thefts_alerted']}/{s['stolen_items']}) |",
         f"| Theft recall, alert or review | {pct(s['theft_recall_alert_or_review'])} ({s['thefts_alerted_or_reviewed']}/{s['stolen_items']}) |",
         f"| Stolen items alerted with the right SKU | {s['thefts_alerted_with_the_right_sku']}/{s['stolen_items']} |",
         f"| Alert precision | {pct(s['alert_precision'])} ({s['true_alerts']}/{s['alerts']}) |",
         f"| False alerts on honest shoppers | {s['false_alerts_on_honest_shoppers']} (of {s['honest_shoppers']} honest shoppers); on nobody: {s['false_alerts_on_nobody']} |",
         f"| Reviews on honest shoppers | {s['reviews_on_honest_shoppers']} (of {s['reviews']} reviews) |",
         f"| False alerts per hour | {s['false_alerts_per_hour']} |",
         f"| Pick recall | {pct(s['pick_recall'])} ({sum(1 for c in res['clips'] for p in c['picks'] if p['stages']['shelf_event_emitted'])}/{s['true_picks']}) |",
         f"| Pick precision | {pct(s['pick_precision'])} (of {s['pipeline_picks']} PICK events) |",
         f"| Right SKU, of paired picks | {pct(s['right_sku_of_paired_picks'])}; of all true picks {pct(s['right_sku_of_all_true_picks'])} |",
         f"| Right slot, of paired picks | {pct(s['right_slot_of_paired_picks'])} |",
         f"| Right shopper, of paired picks | {pct(s['right_shopper_of_paired_picks'])} |",
         f"| Time to alert after concealment | {'n/a' if not tta else str(tta['median']) + ' s median, ' + str(tta['max']) + ' s max'} |",
         f"| Store-wide identities per real shopper | {s['store_wide_ids_per_shopper']} ({s['store_wide_ids']} ids for {s['shoppers']} shoppers) |",
         f"| Identities sitting on one shopper | mean {s['ids_on_one_shopper_mean']}, max {s['ids_on_one_shopper_max']}; shoppers never tracked {s['shoppers_never_tracked']}; ids covering two shoppers {s['ids_covering_two_shoppers']} |", ""]

    def table(f, title):
        out = [f"### {title}: {f['picks']} picks, {f['through_every_stage']} through every stage", "",
               "| stage | reached this stage | lost here | passed on its own |", "|---|---|---|---|"]
        return out + [f"| {r['stage'].replace('_', ' ')} | {r['reached']} | {r['lost_here']} | {r['passed_on_its_own']} |" for r in f["stages"]] + [""]
    L += ["## Funnel: where each true pick is lost", "",
          "Each true pick walks the stages in order and is counted at the first one it fails. \"Passed on its own\" counts the stage "
          "for every pick, whatever happened before it.", ""]
    L += table(res["funnel"], "All picks")
    for key, title in (("funnel_by_outcome", "Outcome"), ("funnel_by_zone", "Zone")):
        for k, f in res[key].items():
            L += table(f, f"{title} {k}")
    L += ["What a stage means:", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in STAGE_HELP.items()] + [""]
    d = res["detail_counts"]
    L += ["## Detection detail (independent counts over all true picks)", "", "| | picks |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in d.items()] + [""]
    L += ["## Per clip", "", "| clip | cameras | shoppers | thieves | picks | PICK events | stolen | alerted | alerts | false | ids | wall s (edge + pipeline) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["clips"]:
        al = [a for a in c["alerts"] if a["tier"] == "alert"]
        w = (c.get("run") or {}).get("wall_s") or {}
        L.append(f"| {c['clip']} | {len(c['cameras'])} | {c['shoppers']} | {c['thieves']} | {len(c['picks'])} | {c['pipeline_picks']} | {len(c['thefts'])} | "
                 f"{sum(t['alert'] for t in c['thefts'])} | {len(al)} | {sum(not a['true_theft'] for a in al)} | {c['identity']['store_wide_ids']} | {w.get('edge', '')} + {w.get('pipeline', '')} |")
    L += ["", "## Every true pick", "", "| clip | t | shopper | zone | slot | sku | outcome | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res["clips"]:
        for p in c["picks"]:
            d = p["detail"]
            L.append(f"| {p['clip']} | {p['t']} | {p['shopper']} | {p['zone']} | {p['slot']} | {p['sku']} | {p['outcome']} | {(p['lost_at'] or 'none').replace('_', ' ')} | "
                     f"{d['item_frames']} ({d['item_frames_reached']}) | {'y' if d['item_detected'] else ''} | {'y' if d['right_sku_detected'] else ''} | {'y' if d['person_detected'] else ''} | {d['event_sku'] or ''} | {d['event_shopper'] or ''} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ command
def _one(args) -> None:
    """Child process: run the pipeline on one clip (public view only)."""
    clip, out = Path(args.run_one[0]), Path(args.run_one[1])
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    mod, fn = args.runner.split(":")
    kw = {"max_frames": args.max_frames} if args.max_frames else {}
    if args.runner == "bree.sim.bench:run_pipeline":
        kw |= {"backend": args.backend, "edge": not args.no_edge, "verbose": args.verbose}
    getattr(importlib.import_module(mod), fn)(public_view(clip, out), out, **kw)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", nargs="?", choices=["dev", "test"])
    ap.add_argument("--clips", help="comma separated seeds (default: the whole split)")
    ap.add_argument("--jobs", type=int, default=2, help="clips run at the same time")
    ap.add_argument("--out", help="default: out/bench/<split>")
    ap.add_argument("--runner", default="bree.sim.bench:run_pipeline", help="module:function(clip, out, **options) that runs the pipeline on one clip")
    ap.add_argument("--backend", default="sim_sku")
    ap.add_argument("--no-edge", action="store_true", help="skip the camera nodes and hub: every camera streams every frame")
    ap.add_argument("--max-frames", type=int, default=None, help="per camera (smoke test)")
    ap.add_argument("--smoke", action="store_true", help="first clip of the split, 60 frames per camera, results/ untouched")
    ap.add_argument("--score-only", action="store_true", help="score finished runs again, no pipeline run")
    ap.add_argument("--name", default="", help="write results/bench_<split>_<name>.* instead of results/bench_<split>.*")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--run-one", nargs=2, metavar=("CLIP", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.run_one:
        return _one(a)
    if not a.split:
        ap.error("give a split: dev or test")
    only = [int(s) for s in a.clips.split(",")] if a.clips else None
    clips = clip_dirs(a.split, only)
    if a.smoke:
        clips, a.max_frames = clips[:1], a.max_frames or 60
    missing = [c for c in clips if not (c / "clip.json").exists()]
    if missing:
        raise SystemExit(f"not rendered: {', '.join(str(c) for c in missing)}\nrender with: .venv/bin/python scripts/bench/render_split.py {a.split}")
    out = Path(a.out or ROOT / "out" / "bench" / (a.split + (f"_{a.name}" if a.name else "")))
    t0 = time.time()

    def go(clip: Path) -> int:
        run = out / clip.name
        cmd = [sys.executable, "-m", "bree.sim.bench", "--run-one", str(clip), str(run), "--runner", a.runner, "--backend", a.backend]
        cmd += (["--no-edge"] if a.no_edge else []) + (["--max-frames", str(a.max_frames)] if a.max_frames else []) + (["--verbose"] if a.verbose else [])
        run.parent.mkdir(parents=True, exist_ok=True)
        with (out / f"{clip.name}.log").open("w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=log).returncode
        print(f"[bench {time.time() - t0:5.0f} s] {clip.name}: {'done' if code == 0 else 'FAILED, see ' + str(out / (clip.name + '.log'))}", flush=True)
        return code
    if not a.score_only:
        with ThreadPoolExecutor(a.jobs) as ex:
            if any(list(ex.map(go, clips))):
                raise SystemExit("a clip failed; nothing scored")
    scored = [score_clip(c, out / c.name) for c in clips]
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout.strip())
    res = {"split": a.split, "data": NOTE, "runner": a.runner, "options": {"backend": a.backend, "edge": not a.no_edge, "max_frames": a.max_frames},
           "scored_at": time.strftime("%Y-%m-%d %H:%M"), "commit": commit + (" plus uncommitted changes in src" if dirty else ""),
           "wall_s": round(time.time() - t0, 1), **aggregate(scored), "stage_help": STAGE_HELP, "clips": scored}
    full = not a.max_frames and not only
    targets = [out / "bench"] + ([ROOT / "results" / f"bench_{a.split}{'_' + a.name if a.name else ''}"] if full else [])
    for t in targets:
        t.with_suffix(".json").write_text(json.dumps(res, indent=1))
        t.with_suffix(".md").write_text(markdown(res))
    print(markdown(res).split("## Per clip")[0])
    print("written: " + ", ".join(str(t) + ".json and .md" for t in targets) + ("" if full else "  (partial run: results/ left alone)"))


if __name__ == "__main__":
    main()
