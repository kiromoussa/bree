"""Two what-if runs on the clips both have (while a shelf pass is still running): the one-line scorecard of each.
usage: r6_sub.py <whatif name a> <whatif name b> [split]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/bench"))
from bree.sim.bench import aggregate  # noqa: E402
from whatif import line  # noqa: E402
split = sys.argv[3] if len(sys.argv) > 3 else "dev2"
runs = {n: {c["clip"]: c for c in json.loads((ROOT / "out/bench/whatif" / n / split / "bench.json").read_text())["clips"]} for n in sys.argv[1:3]}
both = sorted(set.intersection(*(set(r) for r in runs.values())))
for n, r in runs.items():
    print(n, line({"split": split, **aggregate([r[c] for c in both]), "clips": [r[c] for c in both]}).split("\n")[0])
