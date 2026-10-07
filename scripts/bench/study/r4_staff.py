"""Staff on the floor on dev2 (truth for scoring only): what the tracker and the shelf acts know about each identity.
usage: r4_staff.py [seeds...]"""
import json, sys, math
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from bree.concealment.run import context  # noqa: E402
from bree.track.floor import app_dist  # noqa: E402
seeds = sys.argv[1:] or [str(s) for s in range(11001, 11021)]
for seed in seeds:
    clip = ROOT / f"data/synth/bench/dev2/clip_{seed}"
    roles = {s["shopper"]: s["role"] for s in json.loads((clip / "truth/shoppers.json").read_text())}
    if "staff" not in roles.values() and len(sys.argv) < 2:
        continue
    tracks = {}
    for line in (clip / "truth/tracks.jsonl").read_text().splitlines():
        r = json.loads(line)
        tracks[round(r["t"], 1)] = {s["shopper"]: (s["x"], s["z"]) for s in r["shoppers"]}
    ctx = context(clip, ROOT / f"out/bench/dev2/clip_{seed}")
    clerk = [p for p in ctx["people"] if getattr(p, "staff", False)]
    for p in ctx["people"]:
        votes = Counter()
        for t, x, z in p.path[::5]:
            d = sorted((math.hypot(a - x, b - z), n) for n, (a, b) in (tracks.get(round(t, 1)) or {}).items())
            if d and d[0][0] <= 0.6:
                votes[d[0][1]] += 1
        name = votes.most_common(1)[0][0] if votes else None
        mine = [a for a, w in zip(ctx["acts"], ctx["who"]) if w == p.id]
        dc = min((app_dist(p.app, c.app) for c in clerk if app_dist(p.app, c.app) is not None), default=None)
        print(seed, p.id, name, roles.get(name, "clerk?"), "staff" if p.staff else "", p.born, f"{p.path[0][0]:.0f}-{p.path[-1][0]:.0f}s" if p.path else "", p.state,
              "takes", sum(a["kind"] == "take" for a in mine), "puts", sum(a["kind"] == "put" for a in mine), "arrived", sum(bool(a.get("arrived")) for a in mine),
              "clerk colour dist", None if dc is None else round(dc, 2), "looks", p.app_n, dict(votes.most_common(3)))
