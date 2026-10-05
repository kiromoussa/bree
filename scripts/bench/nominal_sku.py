"""Re-score "right SKU" of a finished bench run against a NOMINAL planogram. Scoring only, reads truth.

The clip's layout.json holds the exact SKU of every slot, the simulator's single misplaced items included
(synth.js: 12 percent of slots). A real planogram would not know those. Here the nominal SKU of a slot is the
most common SKU of its planogram block (consecutive slots with the same base SKU on one fixture, base taken
from the simulator's unshuffled layout). A block whose most common SKU is tied is counted as "unclear" and
keeps the exact SKU, so the nominal figure is an upper bound for those.

  python scripts/bench/nominal_sku.py results/bench_test.json out/bench/test [base-layout.json]
"""
import json, sys
from collections import Counter
from pathlib import Path

BASE = "/Users/kiromoussa/bree/software/shared/example-layout.json"


def nominal(slots, base):
    out, unclear, i = {}, set(), 0
    while i < len(slots):
        j = i
        while j < len(slots) and base[j]["skuId"] == base[i]["skuId"] and base[j]["fixtureId"] == base[i]["fixtureId"]:
            j += 1
        top = Counter(s["skuId"] for s in slots[i:j]).most_common(2)
        tie = len(top) > 1 and top[0][1] == top[1][1]
        for s in slots[i:j]:
            out[s["id"]] = s["skuId"] if tie else top[0][0]
            if tie:
                unclear.add(s["id"])
        i = j
    return out, unclear


def main(bench, runs, base=BASE):
    base = json.load(open(base))["slots"]
    n = Counter()
    for c in json.load(open(bench))["clips"]:
        slots = json.load(open(Path(runs) / f"clip_{c['clip']}" / "clip" / "layout.json"))["slots"]
        assert [s["id"] for s in slots] == [s["id"] for s in base]
        exact = {s["id"]: s["skuId"] for s in slots}
        nom, unclear = nominal(slots, base)
        n["slots"] += len(slots)
        n["slots_misplaced"] += sum(exact[k] != nom[k] for k in exact)
        n["slots_unclear"] += len(unclear)
        for p in c["picks"]:
            d, mis = p["detail"], exact[p["slot"]] != nom[p["slot"]]
            n["picks"] += 1
            n["picks_on_misplaced_slot"] += mis
            n["picks_on_unclear_block"] += p["slot"] in unclear
            if d.get("event_slot") is None:
                continue
            n["paired"] += 1
            n["paired_on_misplaced_slot"] += mis
            n["right_sku_exact"] += d["event_sku"] == p["sku"]
            n["right_sku_nominal"] += nom.get(d["event_slot"]) == p["sku"]
            n["misplaced_scored_right_exact"] += mis and d["event_sku"] == p["sku"]
            if p["outcome"] == "concealed":
                n["stolen_paired"] += 1
                n["stolen_right_sku_exact"] += d["event_sku"] == p["sku"]
                n["stolen_right_sku_nominal"] += nom.get(d["event_slot"]) == p["sku"]
    n = dict(n)
    n["rate_exact"] = round(n["right_sku_exact"] / n["paired"], 3)
    n["rate_nominal"] = round(n["right_sku_nominal"] / n["paired"], 3)
    print(json.dumps(n, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
