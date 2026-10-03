"""Score the theft pipeline against simulator ground truth.

    python -m bree.sim.sim_eval --sim-out ISAAC_RUN_DIR --layout layout.json [--backend yolo|toy]
    python -m bree.sim.sim_eval --fixture out/sim_fixture        # synthetic TOY fixture, no GPU

Steps: adapter (bree.sim.isaac_adapter: per-camera MP4 + store YAML, POS feed from "paid" events)
-> the normal multi-camera pipeline (bree.pipeline.run_store) -> match alerts to ground truth ->
<out>/scorecard.json + <out>/scorecard.md.

How an alert is matched (pipeline person ids are tracker ids, not shopper names):
  shopper  an unpaid item of the alert was picked within --pick-tol s of one of that shopper's
           ground-truth picks, at the same fixture (alert zone name = fixture id)
  time     the alert was emitted after the concealment and within --alert-window s of it
  SKU      some unpaid item of the alert has the concealed event's SKU
A concealed event counts as caught when an alert-tier alert on its shopper is inside the time window.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from bree.sim.isaac_adapter import ZONE_KIND, adapt, load_events, pos_feed


def final_alerts(records: list[dict]) -> list[dict]:
    """Apply retraction records (late receipts): each alert keeps its last tier."""
    by_id = {a["alert_id"]: dict(a) for a in records if not a.get("retracts")}
    for a in records:
        if a.get("retracts") in by_id:
            by_id[a["retracts"]]["tier"] = a["tier"]
    return [a for a in by_id.values() if a["tier"] in ("alert", "review")]


def _rate(n, d):
    return round(n / d, 3) if d else None


def score(truth: list[dict], alerts: list[dict], layout: dict, duration_s: float, pipeline_events: list[dict] = (),
          cameras: list[dict] = (), pick_tol_s: float = 3.0, alert_window_s: float = 300.0) -> dict:
    """truth: the sim's events (one per picked item). alerts: final alert dicts. cameras: adapter rows."""
    fixture_of = {s["id"]: s["fixtureId"] for s in layout["slots"]}
    ftype = {f["id"]: f["type"] for f in layout["fixtures"]}
    cam_kind = {c["id"]: c["kind"] for c in layout["cameras"]}
    ran = {c["id"] for c in cameras} or set(cam_kind)
    picks = [e for e in pipeline_events if e.get("type") == "pick"]
    only_cam = next(iter(ran)) if len(ran) == 1 else None

    def pick_camera(alert, item):        # which camera's engine produced this pick
        ev = next((e for e in picks if abs(e["t"] - item["t_pick"]) < 1e-3
                   and e["person_id"] in ([alert["person_id"]] + alert.get("group", []))), None)
        return (ev or {}).get("meta", {}).get("camera") or only_cam

    rows = []
    for a in alerts:
        votes: dict[str, list[float]] = {}
        for it in a["unpaid_items"]:
            for e in truth:
                dt = abs(it["t_pick"] - e["t"])
                if dt <= pick_tol_s and it.get("zone") in (None, fixture_of.get(e["slotId"])):
                    votes.setdefault(e["shopper"], []).append(dt)
        shopper = min(votes, key=lambda s: (-len(votes[s]), min(votes[s]))) if votes else None
        first = a["unpaid_items"][0] if a["unpaid_items"] else {}
        cam = pick_camera(a, first) if first else None
        rows.append({"alert_id": a["alert_id"], "tier": a["tier"], "confidence": a["confidence"], "shopper": shopper,
                     "t_emitted": a["t_emitted"], "t_exit": a["t_exit"], "latency_after_exit_s": a.get("latency_s"),
                     "skus": [it.get("sku") or it["category"] for it in a["unpaid_items"]],
                     "zone": first.get("zone"), "camera": cam, "camera_kind": cam_kind.get(cam),
                     "true_theft": False})

    concealed = [e for e in truth if e["outcome"] == "concealed"]
    thieves = {e["shopper"] for e in concealed}
    caught = []
    for e in concealed:
        hits = [r for r in rows if r["shopper"] == e["shopper"] and e["tResolved"] <= r["t_emitted"] <= e["tResolved"] + alert_window_s]
        for r in hits:
            r["true_theft"] = True
        best = min((r for r in hits if r["tier"] == "alert"), key=lambda r: r["t_emitted"], default=None)
        fx = fixture_of.get(e["slotId"])
        seen_by = sorted({cam_kind[c["id"]] for c in e.get("cameras") or [] if c["id"] in ran and c["id"] in cam_kind})
        caught.append({"shopper": e["shopper"], "skuId": e["skuId"], "slotId": e["slotId"], "fixture": fx,
                       "zone_kind": ZONE_KIND.get(ftype.get(fx)), "t_pick": e["t"], "t_concealed": e["tResolved"],
                       "seen_by_camera_kinds": seen_by or ["unseen"],
                       "caught": best is not None, "reviewed": any(r["tier"] == "review" for r in hits),
                       "alert_id": best["alert_id"] if best else None,
                       "sku_correct": e["skuId"] in best["skus"] if best else None,
                       "time_to_alert_s": round(best["t_emitted"] - e["tResolved"], 2) if best else None})

    alert_rows = [r for r in rows if r["tier"] == "alert"]
    false_alerts = [r for r in alert_rows if not r["true_theft"]]
    got = [c for c in caught if c["caught"]]
    tta = [c["time_to_alert_s"] for c in got]
    hours = duration_s / 3600

    def block(events, al):
        g = [c for c in events if c["caught"]]
        fa = [r for r in al if not r["true_theft"]]
        return {"concealed": len(events), "caught": len(g), "theft_recall": _rate(len(g), len(events)),
                "alerts": len(al), "false_alerts": len(fa), "alert_precision": _rate(len(al) - len(fa), len(al)),
                "sku_correct_rate": _rate(sum(1 for c in g if c["sku_correct"]), len(g)),
                "false_alerts_per_hour": round(len(fa) / hours, 2) if hours else None}

    def breakdown(event_keys, alert_key):
        keys = sorted({k for c in caught for k in event_keys(c)} | {r[alert_key] or "unknown" for r in alert_rows})
        return {k: block([c for c in caught if k in event_keys(c)], [r for r in alert_rows if (r[alert_key] or "unknown") == k])
                for k in keys}

    zone_kind_of = lambda z: ZONE_KIND.get(ftype.get(z))     # noqa: E731
    for r in rows:
        r["zone_kind"] = zone_kind_of(r["zone"])
    return {
        "summary": {**block(caught, alert_rows),
                    "theft_recall_alert_or_review": _rate(sum(1 for c in caught if c["caught"] or c["reviewed"]), len(caught)),
                    "thieves": len(thieves), "thieves_caught": len({c["shopper"] for c in got}),
                    "time_to_alert_s": {"median": round(statistics.median(tta), 2), "max": max(tta)} if tta else None,
                    "review_tier_alerts": sum(1 for r in rows if r["tier"] == "review"),
                    "false_alerts_unmatched_to_any_shopper": sum(1 for r in false_alerts if r["shopper"] is None),
                    "false_alerts_on_honest_shoppers": sum(1 for r in false_alerts if r["shopper"] is not None),
                    "ground_truth_picks": len(truth), "paid_items": sum(1 for e in truth if e["outcome"] == "paid"),
                    "duration_s": duration_s},
        "by_zone_kind": breakdown(lambda c: [c["zone_kind"] or "unknown"], "zone_kind"),
        "by_zone": breakdown(lambda c: [c["fixture"] or "unknown"], "zone"),
        # events: by the kinds of camera that had the item in view (an event can sit in several rows);
        # alerts: by the kind of camera whose pick produced the alert
        "by_camera_kind": breakdown(lambda c: c["seen_by_camera_kinds"], "camera_kind"),
        "concealed_events": caught, "alerts": rows,
        "matching": {"pick_tol_s": pick_tol_s, "alert_window_s": alert_window_s,
                     "time_to_alert": "alert emitted minus the moment of concealment (the sim does not log the exit time)"},
    }


def markdown(card: dict) -> str:
    s = card["summary"]
    fmt = lambda v: "n/a" if v is None else (f"{v:.1%}" if isinstance(v, float) and v <= 1 else str(v))  # noqa: E731
    tta = s["time_to_alert_s"]
    lines = [f"# Sim scorecard: {card['run']['source']}", "",
             f"{card['run']['data']} Backend `{card['run']['backend']}`, {len(card['run']['cameras'])} camera(s), "
             f"{s['duration_s']:.0f} s of sim time, {s['ground_truth_picks']} picks "
             f"({s['concealed']} concealed, {s['paid_items']} paid), {card['run']['pos']['receipts']} receipt(s) "
             f"({len(card['run']['pos']['dropped'])} dropped, {len(card['run']['pos']['sku_swapped'])} mis-rung).", "",
             "| Metric | Value |", "|---|---|",
             f"| Theft recall (alert tier) | {fmt(s['theft_recall'])} ({s['caught']}/{s['concealed']}) |",
             f"| Theft recall (alert or review) | {fmt(s['theft_recall_alert_or_review'])} |",
             f"| Alert precision | {fmt(s['alert_precision'])} ({s['alerts'] - s['false_alerts']}/{s['alerts']}) |",
             f"| SKU-correct rate (caught thefts) | {fmt(s['sku_correct_rate'])} |",
             f"| Time to alert after concealment | {'n/a' if not tta else str(tta['median']) + ' s median, ' + str(tta['max']) + ' s max'} |",
             f"| False alerts per hour | {s['false_alerts_per_hour']} ({s['false_alerts']} in {s['duration_s']:.0f} s) |",
             f"| Review-tier alerts | {s['review_tier_alerts']} |", ""]
    for title, key in (("By zone kind", "by_zone_kind"), ("By zone (fixture)", "by_zone"), ("By camera kind", "by_camera_kind")):
        lines += [f"## {title}", "", "| | concealed | caught | recall | alerts | false | precision | SKU correct |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {k} | {b['concealed']} | {b['caught']} | {fmt(b['theft_recall'])} | {b['alerts']} | {b['false_alerts']} | "
                  f"{fmt(b['alert_precision'])} | {fmt(b['sku_correct_rate'])} |" for k, b in card[key].items()]
        lines.append("")
    missed = [c for c in card["concealed_events"] if not c["caught"]]
    if missed:
        lines += ["## Missed thefts", ""] + [f"- {c['shopper']} {c['skuId']} at {c['fixture']} (t={c['t_pick']} s, seen by {', '.join(c['seen_by_camera_kinds'])})" for c in missed] + [""]
    if card["run"]["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in card["run"]["warnings"]] + [""]
    return "\n".join(lines)


def run(sim_out, layout_path, out_dir, backend: str = "yolo", fps: float | None = None, zone_owner: str = "best",
        pos: dict | None = None, pick_tol_s: float = 3.0, alert_window_s: float = 300.0, save_video: bool = False,
        max_frames: int | None = None, data_note: str = "Simulator data.") -> dict:
    from bree.cli import make_backend
    from bree.events.zones import load_store_config, merge_stores
    from bree.ledger.payments import JsonlPayments
    from bree.pipeline import CameraInput, run_store
    sim_out, out = Path(sim_out), Path(out_dir)
    layout = json.loads(Path(layout_path).read_text())
    truth = load_events(sim_out)
    ad = adapt(sim_out, layout_path, out / "inputs", fps=fps, zone_owner=zone_owner)
    receipts, pos_info = pos_feed(truth, [s["id"] for s in layout["skus"]], **(pos or {}))
    pay_path = out / "inputs" / "payments.jsonl"
    pay_path.write_text("".join(json.dumps(r) + "\n" for r in receipts))
    stores = [load_store_config(c["store"]) for c in ad["cameras"]]
    cams = [CameraInput(c["id"], c["video"], s) for c, s in zip(ad["cameras"], stores)]
    summary = run_store(cams, make_backend(backend, merge_stores(stores)), out / "pipeline", payments=JsonlPayments(pay_path),
                        save_video=save_video, max_frames=max_frames, verbose=False)
    ev_path = out / "pipeline" / "events.jsonl"
    pipeline_events = [json.loads(line) for line in ev_path.read_text().splitlines() if line.strip()]
    card = score(truth, final_alerts(summary.alerts), layout, ad["duration_s"], pipeline_events, ad["cameras"],
                 pick_tol_s, alert_window_s)
    warnings = [f"{c['id']}: {c['camera_check']}" for c in ad["cameras"] if "WARNING" in c["camera_check"]]
    warnings += [f"{c['id']}: only {c['floor_points']} floor points in view (multi-camera handoff needs 4+)"
                 for c in ad["cameras"] if c["floor_points"] < 4]
    if ad["zones_unseen"]:
        warnings.append(f"fixtures no camera sees (no zone): {', '.join(ad['zones_unseen'])}")
    if "assumed" in ad["fps_note"]:
        warnings.append(f"frame rate {ad['fps_note']}")
    if not truth:
        warnings.append("no events.jsonl in the sim output: nothing to score against")
    if any(e["outcome"] not in ("paid", "concealed") for e in truth):
        warnings.append("some ground-truth picks are neither paid nor concealed (in_hand at the end of the run); they are ignored")
    card["run"] = {"source": str(sim_out), "layout": str(layout_path), "data": data_note, "backend": backend,
                   "fps": ad["fps"], "fps_note": ad["fps_note"], "zone_owner": zone_owner, "cameras": ad["cameras"],
                   "pos": pos_info, "pipeline_events": summary.events, "pipeline_fps": summary.pipeline_fps,
                   "handoffs": summary.handoffs, "warnings": warnings}
    (out / "scorecard.json").write_text(json.dumps(card, indent=1))
    (out / "scorecard.md").write_text(markdown(card))
    return card


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-out", help="Isaac run folder: events.jsonl + camera folders of rgb_*.png somewhere below")
    ap.add_argument("--layout", help="shared layout JSON (default: <sim-out>/layout.json)")
    ap.add_argument("--fixture", metavar="DIR", help="generate the synthetic TOY fixture into DIR and score it (toy backend)")
    ap.add_argument("--out", help="default: <sim-out>/sim_eval")
    ap.add_argument("--backend", choices=["yolo", "toy"], default="yolo")
    ap.add_argument("--fps", type=float, default=None, help="written frame rate; default: inferred, else 10")
    ap.add_argument("--zone-owner", choices=["best", "all"], default="best",
                    help="best: each fixture zone goes to the camera that sees it largest; all: every camera that sees it")
    ap.add_argument("--pos-delay", type=float, default=1.0, help="mean seconds from paid to the receipt")
    ap.add_argument("--pos-delay-sd", type=float, default=0.5)
    ap.add_argument("--pos-dropout", type=float, default=0.0, help="chance a receipt never arrives")
    ap.add_argument("--pos-sku-noise", type=float, default=0.0, help="chance a line is rung up as another SKU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick-tol", type=float, default=3.0)
    ap.add_argument("--alert-window", type=float, default=300.0)
    ap.add_argument("--save-video", action="store_true", help="also write annotated_<camera>.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    note = "Simulator data."
    if a.fixture:
        from bree.sim.sim_fixture import make_fixture
        make_fixture(a.fixture)
        a.sim_out, a.backend, note = a.fixture, "toy", "TOY DATA (synthetic fixture, toy colour detector): proves the chain runs, not accuracy."
    if not a.sim_out:
        ap.error("give --sim-out (make sim-eval SIM_OUT=...) or --fixture DIR")
    layout = a.layout or str(Path(a.sim_out) / "layout.json")
    if not Path(layout).exists():
        ap.error(f"layout not found: {layout} (pass --layout / LAYOUT=)")
    out = a.out or str(Path(a.sim_out) / "sim_eval")
    card = run(a.sim_out, layout, out, a.backend, a.fps, a.zone_owner,
               {"delay_s": a.pos_delay, "delay_sd_s": a.pos_delay_sd, "dropout": a.pos_dropout,
                "sku_noise": a.pos_sku_noise, "seed": a.seed},
               a.pick_tol, a.alert_window, a.save_video, a.max_frames, note)
    print(markdown(card))
    print(f"scorecard: {out}/scorecard.json and scorecard.md; pipeline outputs in {out}/pipeline")


if __name__ == "__main__":
    main()
