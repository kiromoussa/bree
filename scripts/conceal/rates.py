"""Calibration with truth (TRAIN-seed clips): in a moment when an item camera sees a hand of a tracked person or an item on
them, how often is an item sighting there, given that the shopper truly carries an item, or truly carries none?
  python scripts/conceal/rates.py data/synth/bench/train out/bench/train out/conceal/train 4900 4901 ..."""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from bree.concealment.cue import ConcealConfig, load_looks, sightings
from bree.concealment.run import context
from rows import shopper_at
clips, runs, looksd = map(Path, sys.argv[1:4]); cfg = ConcealConfig(); n = Counter()
for seed in sys.argv[4:]:
    clip = clips / f"clip_{seed}"; ctx = context(clip, runs / f"clip_{seed}")
    people = [p for p in ctx["people"] if not p.staff and p.path]
    held, hands = sightings(load_looks(looksd / f"clip_{seed}"), ctx["calib"], people, ctx["fps"], cfg, ctx["layout"])
    world, carry = {}, {}
    for line in (clip / "truth/tracks.jsonl").read_text().splitlines():
        r = json.loads(line)
        world[round(r["t"], 1)] = {s["shopper"]: (s["x"], s["z"], s["state"]) for s in r["shoppers"]}
        carry[round(r["t"], 1)] = {s["shopper"]: (s["carrying"], s["state"]) for s in r["shoppers"]}
    for p in people:
        seen = {s.t for s in held.get(p.id, [])}
        for t in sorted(seen | {s.t for s in hands.get(p.id, [])}):
            got = p.at(t)
            name, _ = shopper_at(world, got[0], t) if got is not None else (None, None)
            if name is None: continue
            c, st = carry[round(t, 1)][name]
            if st in ("reach", "retract", "putback", "pay"): continue
            n[("carrying" if c else "empty-handed", "walk" if st == "walk" else "stand", t in seen)] += 1
for a in ("carrying", "empty-handed"):
    for b in ("walk", "stand"):
        k, m = n[a, b, True], n[a, b, True] + n[a, b, False]
        print(f"{a:13s} {b:5s}: item sighting in {k} of {m} moments ({k / max(m, 1):.2f})")
