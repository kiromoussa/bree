"""Run the full video pipeline on the toy clips and score it against their ground truth."""
from __future__ import annotations

import json
from pathlib import Path

from bree.events.zones import load_store_config
from bree.ledger.payments import JsonlPayments


def match_alerts(truth: dict, alerts: list[dict], tol_s: float = 3.0) -> list[dict]:
    """Pair each ground-truth person with the alert (if any) whose exit time matches."""
    rows = []
    used = set()
    for p in truth["people"]:
        hit = None
        for a in alerts:
            if a["alert_id"] in used:
                continue
            if abs(a["t_exit"] - p["t_exit"]) <= tol_s:
                hit = a
                used.add(a["alert_id"])
                break
        rows.append({"clip": truth["clip"], "person": p["name"], "thief": p["thief"], "stolen": p["stolen"],
                     "tier": hit["tier"] if hit else None, "confidence": hit["confidence"] if hit else None,
                     "flagged_items": [i["category"] for i in hit["unpaid_items"]] if hit else [],
                     "decision_latency_s": hit["latency_s"] if hit else None})
    for a in alerts:
        if a["alert_id"] not in used:
            rows.append({"clip": truth["clip"], "person": "?", "thief": False, "stolen": [], "tier": a["tier"],
                         "confidence": a["confidence"], "flagged_items": [i["category"] for i in a["unpaid_items"]],
                         "decision_latency_s": a["latency_s"], "unmatched": True})
    return rows


def run_toy_suite(toy_dir: str | Path, out_dir: str | Path, store_path: str | Path, verbose: bool = False):
    from bree.detect.toy import ToyBackend
    from bree.pipeline import run_pipeline
    toy_dir, out_dir = Path(toy_dir), Path(out_dir)
    store = load_store_config(store_path)
    rows, summaries = [], []
    for truth_path in sorted(toy_dir.glob("toy_*.truth.json")):
        truth = json.loads(truth_path.read_text())
        name = truth["clip"]
        video = toy_dir / f"toy_{name}.mp4"
        pays = JsonlPayments(toy_dir / f"toy_{name}.payments.jsonl")
        s = run_pipeline(str(video), store, ToyBackend(), out_dir / name, payments=pays,
                         save_video=True, verbose=verbose)
        summaries.append(s)
        rows += match_alerts(truth, s.alerts)
    return rows, summaries


def score_rows(rows: list[dict]) -> dict:
    thieves = [r for r in rows if r["thief"]]
    honest = [r for r in rows if not r["thief"]]
    tp = sum(1 for r in thieves if r["tier"] == "alert")
    fn = len(thieves) - tp
    fp = sum(1 for r in honest if r["tier"] == "alert")
    review_honest = sum(1 for r in honest if r["tier"] == "review")
    return {"people": len(rows), "thieves": len(thieves), "alerts_on_thieves": tp, "missed_thieves": fn,
            "alerts_on_honest": fp, "reviews_on_honest": review_honest,
            "reviews_on_thieves": sum(1 for r in thieves if r["tier"] == "review")}
