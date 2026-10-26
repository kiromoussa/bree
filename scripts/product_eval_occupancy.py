"""Occupancy filter vs REAL-MODEL-LABELLED counts (agent-made labels, NOT ground truth).

Compares, at each labelled time, (a) the raw tracker count (every person box in that frame) and
(b) the count after bree.product.adapt's filters, with the filter settings frozen in AdaptConfig
(they were set before this was run and are not tuned on any split). Reports exact-match share and
mean absolute error per camera and split. Local data only; needs ~/Projects/bree-data.
usage: python scripts/product_eval_occupancy.py [out.json]
"""
import json
import sys
from pathlib import Path

from bree.product.adapt import AdaptConfig, _tracks, qualify

B = Path.home() / "Projects/bree-data"
L = B / "real/labels"
RUNS = {  # (channel, segment stem) -> baseline run dir
    ("ch07", "20261025_182851"): "ch07/full_a", ("ch07", "20261025_184620"): "ch07/full_b",
    ("ch29", "20261025_182853"): "ch29/out_A", ("ch29", "20261025_184621"): "ch29/out_B",
    ("ch12", "20261025_182855"): "ch12/run_a",
    ("ch14", "20261025_182857"): "ch14/full_20261025_182857", ("ch14", "20261025_183403"): "ch14/full_20261025_183403",
    ("ch15", "20261025_182859"): "ch15/full1", ("ch15", "20261025_185128"): "ch15/full2",
    ("ch10", "20261025_183408"): "ch10/run_act", ("ch10", "20261025_185127"): "ch10/run_heldout",
}


def labels():
    """-> [(channel, segment stem, split, t_s, count)]"""
    out = []
    d = json.load(open(L / "trumanrd_ch07_people.json"))
    for sp in ("dev", "heldout"):
        s = d[sp]["segment"][:15]
        out += [("ch07", s, sp, f["t_s"], f["count_inside"]) for f in d[sp]["frames"]]
    d = json.load(open(L / "trumanrd_ch29_people.json"))
    out += [("ch29", f["segment"][:15], "heldout" if "held" in f["split"] else "dev", f["t_s"], f["visible_persons"]) for f in d["frames"]]
    d = json.load(open(L / "trumanrd_ch12_people.json"))
    out += [("ch12", f["segment"][:15], "heldout" if "held" in f["split"] else "dev", f["t_s"], f["visible_people_inside"]) for f in d["frames"]]
    d = json.load(open(L / "trumanrd_ch14_people.json"))
    out += [("ch14", f["segment"].split("/")[-1][:15], "heldout" if "held" in f["split"] else "dev", f["t_s"], f["person_count"]) for f in d["frames"]]
    d = json.load(open(L / "trumanrd_ch15_people.json"))
    out += [("ch15", f["segment"][:15], "heldout" if f["segment"][:15] == "20261025_185128" else "dev", f["t_s"], f["visible_people_in_store"]) for f in d["frames"]]
    d = json.load(open(L / "trumanrd_ch10_people.json"))
    for name, s in d["sets"].items():
        sp = "heldout" if "held" in name else "dev"
        out += [("ch10", s["segment"][:15], sp, f["t_s"], f["n_people"]) for f in s["frames"]]
    return out


def main():
    cfg = AdaptConfig()
    cache, rows = {}, {}
    for ch, seg, sp, t, n in labels():
        run = RUNS.get((ch, seg))
        if run is None:
            continue
        if run not in cache:
            tr, _ = _tracks(B / "baseline" / run / "frames.jsonl")
            keep, _ = qualify(tr, cfg)
            cache[run] = (tr, keep)
        tr, keep = cache[run]
        raw = sum(1 for k in tr.values() if any(abs(p[0] - t) <= 0.04 for p in k.pts))
        filt = sum(1 for k in keep.values() if k.t0 <= t <= k.t1)
        r = rows.setdefault((ch, sp), {"n": 0, "raw_exact": 0, "filt_exact": 0, "raw_ae": 0, "filt_ae": 0, "raw_err": 0, "filt_err": 0})
        r["n"] += 1
        r["raw_exact"] += raw == n
        r["filt_exact"] += filt == n
        r["raw_ae"] += abs(raw - n)
        r["filt_ae"] += abs(filt - n)
        r["raw_err"] += raw - n
        r["filt_err"] += filt - n
    out = []
    print(f"{'cam':5} {'split':8} {'n':>3} | raw exact  MAE  bias | filtered exact  MAE  bias")
    for (ch, sp), r in sorted(rows.items()):
        n = r["n"]
        o = {"camera": ch, "split": sp, "n": n, "raw_exact": r["raw_exact"] / n, "raw_mae": r["raw_ae"] / n, "raw_bias": r["raw_err"] / n,
             "filtered_exact": r["filt_exact"] / n, "filtered_mae": r["filt_ae"] / n, "filtered_bias": r["filt_err"] / n,
             "label_kind": "REAL-MODEL-LABELLED"}
        out.append(o)
        print(f"{ch:5} {sp:8} {n:3} | {o['raw_exact']:8.0%} {o['raw_mae']:5.2f} {o['raw_bias']:+5.2f} | {o['filtered_exact']:12.0%} {o['filtered_mae']:5.2f} {o['filtered_bias']:+5.2f}")
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
