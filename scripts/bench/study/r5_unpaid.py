"""Every unpaid item on a dev2 ledger record, by what it truly was (truth for scoring only).
usage: r5_unpaid.py [whatif name, run with --ledger '{"review_threshold": 0.01}'] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
name = next((a for a in sys.argv[1:] if not a.startswith("-")), "r5_all")
run = ROOT / "out/bench/whatif" / name / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
tab, per = Counter(), Counter()
for c in json.loads((run / "bench.json").read_text())["clips"]:
    full = {a["alert_id"]: a for a in J(run / f"clip_{c['clip']}/pipeline/alerts.jsonl")}
    role = {p["shopper"]: ("thief" if p["thief"] else "honest" if p["role"] == "shopper" else p["role"]) for p in c["people"]}
    for a in c["alerts"]:
        f = full[a["alert_id"]]
        who = role.get(a["shopper"], "nobody")
        kept = "kept" if f["confidence"] >= 0.4 else "dropped"
        kinds = []
        for u in f["unpaid_items"]:
            p = next((p for p in c["picks"] if p["detail"]["event_t"] is not None and abs(p["detail"]["event_t"] - u["t_pick"]) < 0.06 and p["detail"]["event_sku"] == u["category"]), None)
            if p is None:
                k = "no true take (false or repeated pick, staff put, shifted item)"
            elif p["shopper"] != a["shopper"]:
                k = f"take of another person ({p['outcome']})"
            else:
                k = {"concealed": "stolen", "paid": "paid", "put_back": "put back"}.get(p["outcome"], p["outcome"]) + ("" if p["sku"] == u["category"] else ", under another name")
            kinds.append(k)
            tab[(who, kept, k)] += 1
        per[(who, kept)] += 1
        if "-v" in sys.argv:
            print(c["clip"], a["shopper"], who, f["confidence"], [(u["category"], u["score"], k) for u, k in zip(f["unpaid_items"], kinds)], [r for r in f["reasons"] if "capped" in r or "register" in r or "paid for" in r])
for k in sorted(per):
    print(per[k], "records", *k)
for k in sorted(tab):
    print(tab[k], *k)
