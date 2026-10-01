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
        if p.get("t_exit") is None:          # never left during the clip (staff, or cut off): no decision to score
            continue
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


def run_toy_suite(toy_dir: str | Path, out_dir: str | Path, store_path: str | Path, verbose: bool = False,
                  prefix: str = "toy_", make_backend=None):
    """Score every `<prefix><clip>.truth.json` in toy_dir. A clip's own `<prefix><clip>.store.yaml` (Isaac Sim
    clips carry zones projected into their camera) overrides `store_path`. `make_backend(store)` defaults to
    the toy colour detector."""
    from bree.pipeline import run_pipeline
    if make_backend is None:
        from bree.detect.toy import ToyBackend
        make_backend = lambda store: ToyBackend()  # noqa: E731
    toy_dir, out_dir = Path(toy_dir), Path(out_dir)
    default_store = load_store_config(store_path)
    rows, summaries = [], []
    for truth_path in sorted(toy_dir.glob(f"{prefix}*.truth.json")):
        truth = json.loads(truth_path.read_text())
        name = truth["clip"]
        video = toy_dir / f"{prefix}{name}.mp4"
        own = toy_dir / f"{prefix}{name}.store.yaml"
        store = load_store_config(own) if own.exists() else default_store
        pays = JsonlPayments(toy_dir / f"{prefix}{name}.payments.jsonl")
        s = run_pipeline(str(video), store, make_backend(store), out_dir / name, payments=pays,
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
