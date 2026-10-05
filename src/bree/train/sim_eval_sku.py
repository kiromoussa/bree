"""`make sim-eval` with the sim-trained SKU detector as the product backend.

    python -m bree.train.sim_eval_sku --sim-out data/synth/clip_5001 [--backend sim_sku|yolo] [--roi people|full]

Runs bree.sim.sim_eval.run (adapter -> pipeline -> scorer) unchanged, with the product half of the
perception backend swapped for bree.train.backend.SimSkuBackend, and adds a `product_recognition`
block to the scorecard: how many product boxes the pipeline saw, how many ground-truth picks it
turned into a PICK event (same fixture, within the scorer's time tolerance), and how many of those
carry the right SKU. `make sim-eval SIM_BACKEND=sim_sku` runs the same backend with its defaults; this
command adds `--roi`, `--weights` and the product_recognition block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def product_recognition(truth: list[dict], out: Path, layout: dict, pick_tol_s: float = 3.0) -> dict:
    fixture_of = {s["id"]: s["fixtureId"] for s in layout["slots"]}
    pipe = out / "pipeline"
    events = [json.loads(line) for line in (pipe / "events.jsonl").read_text().splitlines() if line.strip()]
    picks = [e for e in events if e.get("type") == "pick"]
    boxes = frames = 0
    for f in pipe.glob("frames*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                frames += 1
                boxes += len(json.loads(line)["products"])
    matched = right = 0
    used: set[int] = set()
    for e in truth:
        cand = [(abs(p["t"] - e["t"]), i) for i, p in enumerate(picks)
                if i not in used and abs(p["t"] - e["t"]) <= pick_tol_s and p.get("zone") in (None, fixture_of.get(e["slotId"]))]
        if cand:
            _, i = min(cand)
            used.add(i)
            matched += 1
            right += picks[i].get("item") == e["skuId"]
    return {"frames": frames, "product_boxes": boxes, "product_boxes_per_frame": round(boxes / frames, 2) if frames else None,
            "pipeline_pick_events": len(picks), "ground_truth_picks": len(truth), "picks_matched": matched, "picks_sku_right": right,
            "pick_recall": round(matched / len(truth), 3) if truth else None,
            "sku_right_of_matched": round(right / matched, 3) if matched else None}


def main(argv=None) -> None:
    import bree.cli as cli
    from bree.sim import sim_eval
    from bree.sim.isaac_adapter import load_events
    from bree.train.backend import make_sim_backend
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-out", required=True, help="simulator run folder (render_synth.mjs --clip, or an Isaac run)")
    ap.add_argument("--layout", help="default: <sim-out>/layout.json")
    ap.add_argument("--out", help="default: <sim-out>/sim_eval_<backend>")
    ap.add_argument("--backend", choices=["sim_sku", "yolo"], default="sim_sku", help="yolo = the COCO baseline, for comparison")
    ap.add_argument("--roi", choices=["people", "full"], default="people", help="run the SKU detector around people only, or on the whole frame")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--zone-owner", choices=["best", "all"], default="best")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--closed-world", action="store_true", help="store-wide closed-world identity across the cameras")
    ap.add_argument("--slots", action="store_true", help="3D slot of each pick from the layout's camera poses")
    a = ap.parse_args(argv)
    layout_path = a.layout or str(Path(a.sim_out) / "layout.json")
    out = Path(a.out or Path(a.sim_out) / f"sim_eval_{a.backend}")
    make0 = cli.make_backend
    if a.backend == "sim_sku":
        cli.make_backend = lambda kind, store, imgsz=None, products=True, runtime="pytorch": (
            make_sim_backend(store, imgsz, runtime, a.weights, a.roi) if kind == "sim_sku" else make0(kind, store, imgsz, products, runtime))
    try:
        card = sim_eval.run(a.sim_out, layout_path, out, a.backend, a.fps, a.zone_owner, None, save_video=a.save_video, max_frames=a.max_frames,
                            data_note="SIMULATED (browser store simulator copy, scripts/train/render_synth.mjs --clip).",
                            closed_world=a.closed_world, slots=a.slots)
    finally:
        cli.make_backend = make0
    card["product_recognition"] = product_recognition(load_events(Path(a.sim_out)), out, json.loads(Path(layout_path).read_text()))
    (out / "scorecard.json").write_text(json.dumps(card, indent=1))
    pr = card["product_recognition"]
    md = sim_eval.markdown(card) + "\n## Product recognition\n\n| | |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in pr.items())
    (out / "scorecard.md").write_text(md)
    print(md)
    print(f"scorecard: {out}/scorecard.json and scorecard.md")


if __name__ == "__main__":
    main()
