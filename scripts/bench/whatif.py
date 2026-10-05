"""What-if on the stored shelf events and person boxes of a finished bench run (SIMULATED): repeat tracking, association
and the ledger with other settings and score again. Nothing before tracking is run, results/ is not written.

    .venv/bin/python scripts/bench/whatif.py base                                  # dev and train as the code stands
    .venv/bin/python scripts/bench/whatif.py reach6 --reach '{"within_s": 6}'      # bree.shelf.store.one_act_per_reach
    --assoc '{"max_unseen_s": 3}'  --ledger '{"misread_factor": 1.0}'  --join '{"register_dwell_s": 2}'  --splits dev
    --floor '{"other_m": 99}'  (bree.track.floor.FloorConfig)
    --join '{"put_returns": "none"}' (or "both")  --reach '{"from_last": true}'       # the round 3 choices

Never give it the test split: it is for tuning. Output: out/bench/whatif/<name>/<split>/ and one line per split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bree.shelf.store import rejoin  # noqa: E402
from bree.sim.bench import aggregate, clip_dirs, public_view, score_clip  # noqa: E402
from bree.track.associate import AssocConfig  # noqa: E402
from bree.track.floor import FloorConfig  # noqa: E402


def run(name: str, split: str, join=None, reach=None, assoc=None, ledger=None, src: str | None = None, floor=None) -> dict:
    scored = []
    for clip in clip_dirs(split):
        stored = Path(src or ROOT / "out" / "bench" / split) / clip.name / "pipeline"
        out = ROOT / "out" / "bench" / "whatif" / name / split / clip.name
        (out / "pipeline").mkdir(parents=True, exist_ok=True)
        for f in [*stored.glob("people_*.jsonl"), stored / "shelf_events.jsonl"]:
            link = out / "pipeline" / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        rejoin(public_view(clip, out), out, review=False, floor=FloorConfig(**floor) if floor else None, join=join, reach=reach, assoc=AssocConfig(**assoc) if assoc else None, ledger=ledger)
        scored.append(score_clip(clip, out))
    res = {"split": split, **aggregate(scored), "clips": scored}
    (out.parent / "bench.json").write_text(json.dumps(res, indent=1))
    return res


def line(res: dict) -> str:
    s = res["scorecard"]
    missed = [f"{c['clip']} {t['shopper']} {t['sku']}" for c in res["clips"] for t in c["thefts"] if not t["alert_or_review"]]
    honest = [f"{c['clip']} {a['shopper']} {a['skus']}" for c in res["clips"] for a in c["alerts"] if not a["true_theft"]]
    return (f"{res['split']:5s} thefts {s['thefts_alerted_or_reviewed']}/{s['stolen_items']} (alert tier {s['thefts_alerted']})  honest flagged {s['reviews_on_honest_shoppers'] + s['false_alerts_on_honest_shoppers']}"
            f"/{s['honest_shoppers']}  picks {s['pick_recall']} prec {s['pick_precision']} ({s['pipeline_picks']})  sku {s['right_sku_of_paired_picks']}  slot {s['right_slot_of_paired_picks']}"
            f"  shopper {s['right_shopper_of_paired_picks']}  ids {s['store_wide_ids_per_shopper']}\n      missed: {missed}\n      honest: {honest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--splits", default="dev,train")
    for k in ("join", "reach", "assoc", "ledger", "floor"):
        ap.add_argument(f"--{k}", type=json.loads)
    a = ap.parse_args()
    if "test" in a.splits:
        raise SystemExit("the test split is for final numbers only")
    for split in a.splits.split(","):
        print(a.name, line(run(a.name, split, a.join, a.reach, a.assoc, a.ledger, floor=a.floor)), flush=True)
