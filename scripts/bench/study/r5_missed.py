"""Each stolen item on dev2 that no record flags (truth for scoring only): what the pick became and what the ledger
did with the person it was put on. usage: r5_missed.py [whatif name] [-v]"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
name = next((a for a in sys.argv[1:] if not a.startswith("-")), "r5_base")
run = ROOT / "out/bench/whatif" / name / "dev2"
for c in json.loads((run / "bench.json").read_text())["clips"]:
    logs = dict(re.findall(r"person (\d+):\n(.*?)(?=\n\nperson |\Z)", (run / f"clip_{c['clip']}/pipeline/ledger_log.txt").read_text(), re.S))
    for t in c["thefts"]:
        if t["alert_or_review"]:
            continue
        p = next(p for p in c["picks"] if p["shopper"] == t["shopper"] and p["sku"] == t["sku"] and abs(p["t"] - t["t_pick"]) < 0.5)
        d = p["detail"]
        print(f"\n== {c['clip']} {t['shopper']} {t['sku']} {t['slot']} pick {t['t_pick']} hide {t['t_conceal']} lost_at {p['lost_at']}: event t={d['event_t']} sku={d['event_sku']} slot={d['event_slot']} person={d['event_person']} on {d['event_shopper']}")
        if "-v" in sys.argv and d["event_person"] is not None:
            print(logs.get(str(d["event_person"]), "(no ledger record)"))
