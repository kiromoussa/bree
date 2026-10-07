"""Every ledger decision on dev2 with the review bar at 0 (truth for scoring only): which discounts sit on thieves and
which on honest shoppers. usage: r4_ledger.py [whatif name, run with --ledger '{"review_threshold": 0.01}']"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
name = sys.argv[1] if len(sys.argv) > 1 else "r4_all"
run = ROOT / "out/bench/whatif" / name / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
tab = Counter()
for c in json.loads((run / "bench.json").read_text())["clips"]:
    clip = ROOT / f"data/synth/bench/dev2/clip_{c['clip']}"
    pays = [r["t"] for r in J(clip / "register.jsonl")]
    full = {a["alert_id"]: a for a in J(run / f"clip_{c['clip']}/pipeline/alerts.jsonl")}
    for a in c["alerts"]:
        f = full[a["alert_id"]]
        r = " ".join(f["reasons"])
        visits = [tuple(float(x) for x in l.split("s register visit ")[1].rstrip("s").split("-")) for l in f["audit_log"] if "s register visit " in l]
        near = sum(any(v0 - 1.5 <= t <= v1 + 7.5 for v0, v1 in visits) for t in pays)
        flags = ("never at register" if "never went" in r else f"at register, no receipt ({'some' if near else 'no'} receipt stamped then)" if "no receipt matched" in r else "paid, not these",
                 "crowded" if "crowded pick" in r and "all have left" not in r else "", "misread" if "under another name" in r else "", "id uncertain" if any("identity uncertain" in l for l in f["audit_log"]) else "")
        who = "thief" if a["true_theft"] else a["role"] if a["role"] != "shopper" else "honest"
        tab[(flags, who, "kept" if f["confidence"] >= 0.4 else "dropped")] += 1
        if "-v" in sys.argv:
            print(c["clip"], a["shopper"], who, f["confidence"], [u["score"] for u in f["unpaid_items"]], flags, visits, near)
for k in sorted(tab, key=str):
    print(tab[k], *k)
