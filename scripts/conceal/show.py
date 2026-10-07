"""Summary of rows.jsonl: python scripts/conceal/show.py rows.jsonl [--all] [--p 0.5]"""
import json, sys
from collections import Counter
rows = [json.loads(x) for x in open(sys.argv[1])]
thr = float(sys.argv[sys.argv.index("--p") + 1]) if "--p" in sys.argv else 0.5
tp = sum(r["label"] and r["p"] >= thr for r in rows); fp = sum((not r["label"]) and r["p"] >= thr for r in rows); pos = sum(r["label"] for r in rows)
print(len(rows), "takes;", pos, "followed by a concealment;", Counter(r["where"] or "scored" for r in rows))
print(f"at p >= {thr}: true {tp}, false {fp}, missed {pos - tp}")
for r in rows:
    if r["label"] or r["p"] >= thr or "--all" in sys.argv:
        print(r["clip"][-4:], r["person_id"], r["shopper"], "thief" if r["thief"] else "honest", f"take {r['t']} {str(r['sku'])[:10]:10s} stop {r['stop']} {r['stop_why'][:5]}", (r["where"] or "-").ljust(5), "L", r["label"], "p", r["p"],
              "| cusum", r["cusum"], "before", r["seen_before"], "last", r["t_last"], "empty", r["empty_bins"], "seen", r["seen_after"], "cams", r["empty_cams"], "tail", r["tail_s"], "drop", r["y_drop"], "hand_back", r["hand_back_m"], "shelf_after", r["shelf_after"], "item_after", r["item_after"], "| hide", r["t_hide"], r["truth_pick"])
