"""Held-item and hand sightings of the people near a moment, with the truth shopper under each track (bench TRAIN-seed clips):
  python scripts/conceal/peek.py train 4901 16.0 18.5"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/conceal"))
from bree.concealment.cue import ConcealConfig, load_looks, sightings
from bree.concealment.run import context
from rows import truth_of, shopper_at
split, seed, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
clip = ROOT / f"data/synth/bench/{split}/clip_{seed}"
ctx = context(clip, ROOT / f"out/bench/{split}/clip_{seed}")
cfg = ConcealConfig()
people = [p for p in ctx["people"] if not p.staff and p.path]
held, hands = sightings(load_looks(ROOT / f"out/conceal/{split}/clip_{seed}"), ctx["calib"], people, ctx["fps"], cfg, ctx["layout"])
tracks, events, _ = truth_of(clip)
for p in people:
    got = p.at((t0 + t1) / 2)
    if got is None: continue
    print("person", p.id, shopper_at(tracks, got[0], round((t0 + t1) / 2, 1)), "takes", [(a["t"], a["sku_id"]) for a, w in zip(ctx["acts"], ctx["who"]) if w == p.id and a["kind"] == "take"])
    for name, d in (("item", held), ("hand", hands)):
        by = {}
        for s in d.get(p.id, []):
            if t0 <= s.t <= t1: by.setdefault(s.cam, []).append(s)
        for cam, ss in by.items():
            print(" ", name, cam, " ".join(f"{s.t:.1f}:{int(s.u)},{int(s.v)}" + (f":{s.sku[:6]}" if s.sku else "") for s in ss))
