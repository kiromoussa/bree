"""Diagnosis with truth (TRAIN-seed clips): when an item camera truly shows an item in a shopper's hand, does the cue
have a held sighting for that person then, and if not, at which step was it lost?
  python scripts/conceal/why_missed.py data/synth/bench/train out/bench/train out/conceal/train 4900 4901"""
import json, sys, math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from bree.concealment.cue import ConcealConfig, load_looks, _slot_lines, _at_slot, _owner
from bree.concealment.run import context
from bree.shelf.events import ITEM_KINDS
from rows import truth_of, shopper_at
clips, runs, looksd = map(Path, sys.argv[1:4]); cfg = ConcealConfig(); tot = Counter(); MINPX = 400
for seed in sys.argv[4:]:
    clip = clips / f"clip_{seed}"; ctx = context(clip, runs / f"clip_{seed}")
    people = [p for p in ctx["people"] if not p.staff and p.path]
    looks = {c: {r["f"]: r for r in rows} for c, rows in load_looks(looksd / f"clip_{seed}").items()}
    tracks, events, _ = truth_of(clip)
    lines = {c: _slot_lines(ctx["calib"][c], ctx["layout"]) for c in looks}
    for line in (clip / "truth/frames.jsonl").read_text().splitlines():
        r = json.loads(line); cam = r["camera"]
        if cam not in looks or r["frame"] % 2: continue
        for it in r["items"]:
            if it["kind"] != "hand" or it["vis_px"] < MINPX or not it["shopper"]: continue
            st = next((p["state"] for p in r["persons"] if p["shopper"] == it["shopper"]), "?")
            grp = "walk" if st == "walk" else "stand"
            x0, y0, x1, y1 = it["bbox"]; u, v = (x0 + x1) / 2, (y0 + y1) / 2; t = r["t"]
            look = looks[cam].get(r["frame"])
            if look is None: tot[grp, "1 detector did not look"] += 1; continue
            near = [d for d in look["items"] if abs((d[0] + d[2]) / 2 - u) < 0.6 * (x1 - x0) + 15 and abs((d[1] + d[3]) / 2 - v) < 0.6 * (y1 - y0) + 15]
            if not near: tot[grp, "2 no detection there"] += 1; continue
            d = max(near, key=lambda d: d[4])
            if d[4] < cfg.conf: tot[grp, "3 low confidence"] += 1; continue
            if d[8]: tot[grp, "4 called stock"] += 1; continue
            if d[6] < cfg.fg: tot[grp, "5 not different from shelf picture"] += 1; continue
            if _at_slot(lines[cam].get(d[5]), (d[0] + d[2]) / 2, (d[1] + d[3]) / 2, math.hypot(d[2] - d[0], d[3] - d[1]), cfg.slot_k): tot[grp, "6 on a slot of its product"] += 1; continue
            own = _owner(ctx["calib"][cam], (d[0] + d[2]) / 2, (d[1] + d[3]) / 2, t, people, cfg)
            if own is None: tot[grp, "7 no person fits"] += 1; continue
            by = {p.id: p for p in people}
            name, _ = shopper_at(tracks, by[own[0]].at(t)[0], t)
            tot[grp, "8 seen, right person" if name == it["shopper"] else "9 seen, other person"] += 1
for grp in ("stand", "walk"):
    n = sum(v for (g, _), v in tot.items() if g == grp)
    print(grp, n, {k: f"{v} ({v / n:.0%})" for (g, k), v in sorted(tot.items()) if g == grp})
