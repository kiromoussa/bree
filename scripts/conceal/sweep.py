"""Per-take results on TRAIN-seed tuning clips for a few settings of ConcealConfig (SIMULATED; tuning clips only).
  python scripts/conceal/sweep.py data/synth/bench/train out/bench/train out/conceal/train 4900 4901 ... -- put_before_s=0 seen_cost=4"""
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from bree.concealment.cue import ConcealConfig, load_model  # noqa: E402
from rows import rows_of  # noqa: E402
cut = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
clips, runs, looks = map(Path, sys.argv[1:4])
cfg = ConcealConfig(**{k: type(getattr(ConcealConfig(), k))(float(v)) for k, v in (x.split("=") for x in sys.argv[cut + 1:])})
model = None if "--no-model" in sys.argv[:cut] else load_model()
rows = [r for s in sys.argv[4:cut] if not s.startswith("--") for r in rows_of(clips / f"clip_{s}", runs / f"clip_{s}", looks / f"clip_{s}", cfg, model)]
pos, tp = sum(r["label"] for r in rows), sum(r["label"] and r["p"] >= cfg.p_cue for r in rows)
fp = [r for r in rows if not r["label"] and r["p"] >= cfg.p_cue]
thieves = {(r["clip"], r["shopper"]) for r in rows if r["thief"] and r["label"]}
print(f"{len(rows)} takes, {pos} followed by a concealment, where: {dict(Counter(r['where'] or 'scored' for r in rows))}")
print(f"cue at p >= {cfg.p_cue}: true {tp}, false {len(fp)}, missed {pos - tp}; thieves cued {len({(r['clip'], r['shopper']) for r in rows if r['label'] and r['p'] >= cfg.p_cue})} of {len(thieves)} with a labelled take; honest shoppers cued {len({(r['clip'], r['shopper']) for r in fp if not r['thief']})}")
for r in rows:
    if r["label"] or r["p"] >= cfg.p_cue or (not r["where"] and r["cusum"] >= 5):
        print(" ", r["clip"][-4:], r["shopper"], "thief " if r["thief"] else "honest", f"take {r['t']:5.1f}", (r["where"] or "scored").ljust(8), "label", r["label"], "p", f"{r['p']:.2f}", "cusum", r["cusum"], "seen_before", r["seen_before"], "seen_after", r["seen_after"], "empty", r["empty_bins"], r["truth_pick"])
