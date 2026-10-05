#!/usr/bin/env python
"""One end-to-end chain on SIMULATED data. Every stage is the repo's real code path; only the data is simulated.

    .venv/bin/python scripts/e2e_sim_chain.py --sim-out data/synth/clip_5001_door     (make e2e-sim)

  simulator views      a clip from the browser store simulator copy (make sku-clip): <camera>/rgb_NNNN.png,
                       events.jsonl (ground truth), layout.json
  -> edge trigger      every camera that sees a shelf or cooler runs as a camera node (bree.edge.node): the
                       low-res zone trigger decides which full-res frames leave the node. That is every item
                       camera (layout kind shelf or cooler: the Pi nodes of the hardware plan). Checkout and
                       overhead cameras stream every frame, as a wired camera would.
  -> hub               a real hub on 127.0.0.1 (bree.edge.hub) stores the bursts sent over HTTP
  -> pipeline          one run over all cameras (bree.sim.sim_eval.run): node cameras are read back from the
                       hub's bursts (BurstSource, frames with gaps), products from the SIM-TRAINED SKU
                       detector, store-wide closed-world identity, 3D slot on each pick
  -> alerts            the pipeline's alerts.jsonl
  -> review store      alerts ingested into a review store (bree.review). The "reviewer" here is the
                       simulator's ground truth, not a person: it confirms an alert on a shopper who concealed
                       something, corrects the item when the SKU is wrong, rejects the rest
  -> owner report      the weekly owner report for this week
  -> scorecard         the make sim-eval scorer against the simulator's ground truth

Writes <out>/e2e.json and <out>/e2e.md (default <sim-out>/e2e). With --results-name NAME both are also copied
to results/NAME.* (the tracked, recorded result is results/e2e_sim_chain.*; the default is no copy, so a run
never replaces it by accident). --refunnel recounts "where a pick is lost" on a finished run without running
the chain again.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bree.edge.hub import BurstSource, Hub  # noqa: E402
from bree.edge.node import build, load_node_config  # noqa: E402
from bree.review.__main__ import ingest  # noqa: E402
from bree.review.metrics import metrics, owner_markdown, owner_report  # noqa: E402
from bree.review.store import ReviewStore  # noqa: E402
from bree.sim import sim_eval  # noqa: E402
from bree.sim.isaac_adapter import adapt  # noqa: E402

NOTE = "SIMULATED (browser store simulator copy, scripts/train/render_synth.mjs --clip). Not real footage."


def edge_stage(sim_out: Path, layout: Path, out: Path, max_frames: int | None) -> tuple[dict, dict]:
    """Play each merch camera through a node into a hub. Returns ({camera: BurstSource}, stats)."""
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


def funnel(sim_out: Path, run_dir: Path) -> dict | None:
    """Where the chain loses a pick, counted on the simulator's ground-truth boxes of items in a hand
    (truth_frames.jsonl, one row per camera frame). Per camera kind, for every such item box:
    did that frame reach the pipeline (the node sent it), was a TRACKED person in the frame (the pipeline's
    frame log lists tracks, not raw detections; the SKU detector looks around raw person detections, so it
    can find a product in a frame with no track), was a product box found on the item (IoU 0.5), with the
    right SKU, and does the item sit inside a shelf zone of that camera in the image (the event engine counts
    an item as taken only once it is outside the shelf polygon).
    The columns are independent counts over the same item boxes, NOT nested steps. The two
    `..._and_tracked_person` columns are the intersections."""
    import cv2
    import numpy as np
    tf = sim_out / "truth_frames.jsonl"
    if not tf.exists():
        return None
    cams = {c["id"]: c for c in json.loads((run_dir / "inputs" / "adapter.json").read_text())["cameras"]}
    fps = json.loads((run_dir / "inputs" / "adapter.json").read_text())["fps"]
    seen, zones = {}, {}
    for cid, c in cams.items():
        log = run_dir / "pipeline" / f"frames_{cid}.jsonl"
        seen[cid] = {round(r["t"] * fps): r for r in map(json.loads, log.read_text().splitlines())} if log.exists() else {}
        zones[cid] = [np.array(z["polygon"], np.float32) for z in yaml.safe_load(Path(c["store"]).read_text())["zones"] if z["kind"] in ("shelf", "cooler")]

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix, 1e-9)
    keys = ("item_boxes", "frame_reached_pipeline", "tracked_person_in_frame", "product_box_on_item", "right_sku",
            "product_box_on_item_and_tracked_person", "right_sku_and_tracked_person", "item_inside_a_shelf_zone_of_this_camera")
    out: dict[str, dict] = {}
    for row in map(json.loads, tf.read_text().splitlines()):
        cid = row["camera"]
        if cid not in cams:
            continue
        fr = seen[cid].get(row["frame"])
        for it in row["items"]:
            if it["kind"] != "hand":
                continue
            n = out.setdefault(cams[cid]["kind"], dict.fromkeys(keys, 0))
            n["item_boxes"] += 1
            cx, cy = (it["bbox"][0] + it["bbox"][2]) / 2, (it["bbox"][1] + it["bbox"][3]) / 2
            n["item_inside_a_shelf_zone_of_this_camera"] += any(cv2.pointPolygonTest(z, (cx, cy), False) >= 0 for z in zones[cid])
            if fr is None:
                continue
            n["frame_reached_pipeline"] += 1
            n["tracked_person_in_frame"] += bool(fr["persons"])
            hit = [q for q in fr["products"] if iou(q["bbox"], it["bbox"]) >= 0.5]
            right = any(q["cat"] == it["sku"] for q in hit)
            n["product_box_on_item"] += bool(hit)
            n["right_sku"] += right
            n["product_box_on_item_and_tracked_person"] += bool(hit) and bool(fr["persons"])
            n["right_sku_and_tracked_person"] += right and bool(fr["persons"])
    return out


def review_stage(card: dict, pipe: Path, out: Path) -> dict:
    """Alerts -> review store -> decisions from the simulator's ground truth -> metrics and owner report."""
    shutil.rmtree(out / "review", ignore_errors=True)
    store = ReviewStore(out / "review")
    got = ingest(store, str(pipe / "alerts.jsonl"), camera="sim_store")
    stolen: dict[str, set[str]] = {}
    for e in card["concealed_events"]:
        stolen.setdefault(e["shopper"], set()).add(e["skuId"])
    decisions = {"confirmed_theft": 0, "wrong_item": 0, "not_theft": 0}
    stored = {a["alert_id"]: a for a in store.alerts()}
    for row in card["alerts"]:
        aid = next((k for k in stored if k.startswith(f"{pipe.name}-{row['alert_id']}-")), None)
        if aid is None:
            continue
        true = stolen.get(row["shopper"], set()) if row["true_theft"] else set()
        if not true:
            d, fix = "not_theft", None
        elif true & set(row["skus"]):
            d, fix = "confirmed_theft", None
        else:
            d, fix = "wrong_item", sorted(true)[0]
        store.decide(aid, "sim_ground_truth", d, corrected_item=fix, note="decision from simulator ground truth, not a person")
        decisions[d] += 1
    today = dt.datetime.now(dt.timezone.utc).date()
    week = today - dt.timedelta(days=today.weekday())
    rep = owner_report(store, week, out / "review" / "reports")
    m = metrics(store)["overall"]
    store.close()
    return {"ingest": got, "decisions_by_sim_ground_truth": decisions, "reviewer": "simulator ground truth (not a person)",
            "metrics_overall": m, "owner_report": {k: rep[k] for k in ("week_start", "week_end", "alerts_sent", "customer_alerts", "staged_tests")},
            "owner_report_md": owner_markdown(rep)}


def markdown(res: dict) -> str:
    e, r = res["edge"], res["review"]
    L = ["# End-to-end chain on SIMULATED data", "", res["data"], "",
         f"Clip: {res['clip'].get('frames')} frames at {res['clip'].get('fps')} fps, cameras {', '.join(res['clip'].get('cameras', []))}. "
         f"Product weights: {res['weights']}. Person confidence threshold {res['person_conf']}. Wall time {res['wall_s']} s.", "",
         "## Edge: camera nodes and hub", "",
         "| camera | kind | path | frames | triggers | bursts | frames at hub | share sent | burst MB | est. full stream MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for cid, c in e["cameras"].items():
        mb = lambda v: "" if v is None else f"{v / 1e6:.1f}"     # noqa: E731
        L.append(f"| {cid} | {c['kind']} | {c['path']} | {c['frames']} | {c.get('triggers', '')} | {c.get('bursts', '')} | {c.get('frames_at_hub', '')} | "
                 f"{c.get('share_of_frames_sent', '')} | {mb(c.get('burst_bytes'))} | {mb(c.get('full_stream_bytes_estimate'))} |")
    L += ["", f"Node cameras: {e['node_cameras']}. Frames sent to the hub: {e['node_frames_sent']} of {e['node_frames']} "
              f"({e['share_of_node_frames_sent']}). Hub errors: {e['hub_errors']}.", "",
          "## Scorecard (make sim-eval scorer, through the whole chain)", "", sim_eval.markdown(res["scorecard"]).split("\n", 2)[2]]
    if res["scorecard"].get("product_recognition"):
        L += ["## Product recognition", "", "| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in res["scorecard"]["product_recognition"].items()] + [""]
    if res.get("funnel_items_in_hand"):
        f = res["funnel_items_in_hand"]
        keys = list(next(iter(f.values())))
        L += ["## Where a pick is lost (ground-truth boxes of items in a hand, per camera kind)", "",
              "Each column is its own count over the same item boxes, not a step that follows the one before. "
              "\"Tracked person\" is a track in the pipeline's frame log; the SKU detector looks around raw person "
              "detections, so a product box can be found in a frame with no track.", "",
              "| camera kind | " + " | ".join(k.replace("_", " ") for k in keys) + " |", "|---|" + "---|" * len(keys)]
        L += [f"| {kind} | " + " | ".join(str(n[k]) for k in keys) + " |" for kind, n in f.items()] + [""]
    L += ["## Review store and owner report", "",
          f"Reviewer: {r['reviewer']}. Ingested {r['ingest']['new']} alert(s) from {r['ingest']['records']} record(s); "
          f"decisions {r['decisions_by_sim_ground_truth']}.", "", r["owner_report_md"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-out", required=True, help="simulator clip folder (make sku-clip)")
    ap.add_argument("--out", help="default: <sim-out>/e2e")
    ap.add_argument("--backend", choices=["sim_sku", "yolo", "toy"], default="sim_sku")
    ap.add_argument("--no-edge", action="store_true", help="skip the node and hub: every camera streams every frame (for comparison)")
    ap.add_argument("--no-closed-world", action="store_true")
    ap.add_argument("--no-slots", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="per camera (smoke test)")
    ap.add_argument("--results-name", default="", help="also copy the result to results/<name>.json and .md "
                    "(the tracked recorded result is e2e_sim_chain; default: no copy)")
    ap.add_argument("--refunnel", action="store_true", help="do not run the chain: recount the funnel of the finished run in --out and rewrite its e2e.json and e2e.md")
    a = ap.parse_args(argv)
    sim_out = Path(a.sim_out)
    out = Path(a.out or sim_out / "e2e")
    layout = sim_out / "layout.json"

    def write(res: dict) -> None:
        (out / "e2e.json").write_text(json.dumps(res, indent=1))
        (out / "e2e.md").write_text(markdown(res))
        if a.results_name:
            for ext in ("json", "md"):
                shutil.copy(out / f"e2e.{ext}", ROOT / "results" / f"{a.results_name}.{ext}")
        print(markdown(res))
    if a.refunnel:
        res = json.loads((out / "e2e.json").read_text())
        res["funnel_items_in_hand"] = funnel(sim_out, out / "sim_eval")
        if Path(res["weights"]).is_absolute() and ROOT in Path(res["weights"]).parents:
            res["weights"] = str(Path(res["weights"]).relative_to(ROOT))
        return write(res)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.time()
    sources, edge = ({}, {"cameras": {}, "node_cameras": 0, "node_frames": 0, "node_frames_sent": 0,
                          "share_of_node_frames_sent": None, "hub_errors": 0}) if a.no_edge else edge_stage(sim_out, layout, out, a.max_frames)
    card = sim_eval.run(sim_out, layout, out / "sim_eval", a.backend, data_note=NOTE, max_frames=a.max_frames,
                        closed_world=not a.no_closed_world, slots=not a.no_slots, sources=sources)
    if a.backend == "sim_sku":
        from bree.sim.isaac_adapter import load_events
        from bree.train.backend import sku_weights
        from bree.train.sim_eval_sku import product_recognition
        card["product_recognition"] = product_recognition(load_events(sim_out), out / "sim_eval", json.loads(layout.read_text()))
        w = Path(sku_weights()).resolve()
        weights = str(w.relative_to(ROOT) if ROOT in w.parents else w)      # no home path in a tracked result file
    else:
        weights = "none (COCO classes)" if a.backend == "yolo" else "toy colours"
    (out / "sim_eval" / "scorecard.json").write_text(json.dumps(card, indent=1))
    review = review_stage(card, out / "sim_eval" / "pipeline", out)
    fun = funnel(sim_out, out / "sim_eval")
    clip = json.loads((sim_out / "clip.json").read_text()) if (sim_out / "clip.json").exists() else {}
    res = {"data": NOTE, "clip": clip, "backend": a.backend, "weights": weights, "edge_in_chain": not a.no_edge,
           "closed_world": not a.no_closed_world, "slots_3d": not a.no_slots, "max_frames": a.max_frames,
           "person_conf": float(os.environ.get("BREE_PERSON_CONF", 0.3)),
           "edge": edge, "scorecard": card, "funnel_items_in_hand": fun, "review": review, "wall_s": round(time.time() - t0, 1)}
    write(res)


if __name__ == "__main__":
    main()
