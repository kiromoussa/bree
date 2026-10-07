"""Where each stolen item of dev2 goes in the concealment cue (truth for scoring only), and what the cue says about
the takes of honest shoppers and staff. usage: r4_conceal.py [whatif name, run with --conceal] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
run = ROOT / "out/bench/whatif" / (args[0] if args else "r4_ctier") / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
V = "-v" in sys.argv
tab, other = Counter(), Counter()
for c in json.loads((run / "bench.json").read_text())["clips"]:
    pipe = run / f"clip_{c['clip']}/pipeline"
    takes = J(pipe / "conceal_takes.jsonl") if (pipe / "conceal_takes.jsonl").exists() else []
    scores = json.loads((pipe / "conceal_scores.json").read_text()) if (pipe / "conceal_scores.json").exists() else {}
    al = {a["shopper"]: a for a in c["alerts"]}
    used = set()
    for p in c["picks"]:
        d, st = p["detail"], p["stages"]
        tk = next((i for i, t in enumerate(takes) if st["shelf_event_emitted"] and t["person_id"] == d["event_person"] and abs(t["t"] - d["event_t"]) < 0.3 and i not in used), None)
        if tk is not None:
            used.add(tk)
        t = takes[tk] if tk is not None else None
        if p["outcome"] != "concealed":
            if t is not None:
                other[(p["outcome"], "cue" if t["p"] >= 0.6 else t["where"] or ("p under 0.3" if t["p"] < 0.3 else "p 0.3 to 0.6"))] += 1
                if V and t["p"] >= 0.3:
                    print("   other", c["clip"], p["shopper"], p["outcome"], p["sku"], round(p["t"], 1), "p", t["p"], "right shopper", st["right_shopper"], {k: t[k] for k in ("cusum", "seen_before", "empty_bins", "stop_why")})
            continue
        a = al.get(p["shopper"])
        why = ("no PICK event" if not st["shelf_event_emitted"] else "PICK on another person" if not st["right_shopper"] else "take not in the cue (staff or no track)" if t is None
               else "cue" if t["p"] >= 0.6 else "read as put back" if t["where"] == "shelf" else "item not seen in the hand twice" if t["where"] == "not_seen" else "scored under the bar")
        tier = a["tier"] if a else "no record"
        tab[(why, tier)] += 1
        if V:
            print(c["clip"], p["shopper"], p["sku"], round(p["t"], 1), "|", why, "|", tier, "|", t and {k: t[k] for k in ("p", "cusum", "seen_before", "seen_after", "empty_bins", "tail_s", "stop_why", "back_m")}, "score", scores.get(str(d["event_person"]), {}).get("score"))
print("stolen items:")
for k, n in sorted(tab.items(), key=lambda x: -x[1]):
    print("  ", n, *k)
print("takes that were paid or put back:", dict(other))
