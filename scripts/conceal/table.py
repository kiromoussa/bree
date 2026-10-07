"""One table from results of scripts/conceal/eval.py: python scripts/conceal/table.py results/a.json [out/b.json ...] (several files are added up)"""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from collections import Counter
from eval import wilson
tot: dict = {}
for f in sys.argv[1:sys.argv.index("--rows") if "--rows" in sys.argv else None]:
    for v, s in json.load(open(f))["summary"].items():
        t = tot.setdefault(v, Counter()); why = tot.setdefault(v + "/why", Counter()); names = tot.setdefault(v + "/names", [])
        for k in ("thieves", "thieves_at_alert", "thieves_with_a_cue", "thieves_at_alert_or_review", "honest_shoppers", "honest_at_alert", "honest_with_a_cue", "honest_at_alert_or_review", "alerts_on_nobody", "cues_on_no_shopper"):
            t[k] += s.get(k, 0)
        why.update(s.get("thieves_not_at_alert_because", {})); names += s.get("who_alerted", []) + s.get("thieves_cued", [])
for v in [k for k in tot if "/" not in k]:
    t = tot[v]
    if v == "cue_before_ledger":
        print(f"{v:26s} thieves with a cue {t['thieves_with_a_cue']} of {t['thieves']} {wilson(t['thieves_with_a_cue'], t['thieves'])}, honest shoppers with a cue {t['honest_with_a_cue']} of {t['honest_shoppers']} {wilson(t['honest_with_a_cue'], t['honest_shoppers'])}, cues on nobody {t['cues_on_no_shopper']}: {sorted(tot[v + '/names'])}")
    else:
        print(f"{v:26s} thieves at alert {t['thieves_at_alert']} of {t['thieves']} {wilson(t['thieves_at_alert'], t['thieves'])}, at alert or review {t['thieves_at_alert_or_review']}; honest at alert {t['honest_at_alert']} of {t['honest_shoppers']} {wilson(t['honest_at_alert'], t['honest_shoppers'])}, honest at alert or review {t['honest_at_alert_or_review']}; alerts on nobody {t['alerts_on_nobody']}")
        print(f"{'':26s} at alert: {sorted(tot[v + '/names'])}; not at alert because: {dict(tot[v + '/why'])}")
# per take: python scripts/conceal/table.py --rows rows_a.jsonl [rows_b.jsonl ...] [--p 0.6]
if "--rows" in sys.argv:
    args = sys.argv[sys.argv.index("--rows") + 1:]
    thr = float(args[args.index("--p") + 1]) if "--p" in args else 0.6
    rows = [json.loads(x) for f in args if f.endswith(".jsonl") for x in open(f)]
    pos = sum(r["label"] for r in rows); tp = sum(r["label"] and r["p"] >= thr for r in rows); fp = sum((not r["label"]) and r["p"] >= thr for r in rows)
    print(f"{len(rows)} takes on {len({r['clip'] for r in rows})} clips, {pos} followed by a concealment; settled by a rule {dict(Counter(r['where'] for r in rows if r['where']))}, scored {sum(not r['where'] for r in rows)}")
    print(f"cue at p >= {thr}: true {tp}, false {fp}, missed {pos - tp}; precision {tp}/{tp + fp} {wilson(tp, tp + fp)}, recall {tp}/{pos} {wilson(tp, pos)}; false cues per take not followed by a concealment {fp}/{len(rows) - pos} {wilson(fp, len(rows) - pos)}")
    print(f"highest p of a take not followed by a concealment: {max((r['p'] for r in rows if not r['label']), default=None)}; lowest p of a cued true take: {min((r['p'] for r in rows if r['label'] and r['p'] >= thr), default=None)}")
