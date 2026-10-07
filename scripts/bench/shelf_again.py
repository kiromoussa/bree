"""Run the shelf pass again on a split whose people boxes are already stored (SIMULATED clips): the person detector
does not change when only the item cameras' code does. Writes out/bench/<name>/<split>/clip_<seed>, which
`whatif.py --src out/bench/<name>/<split>` then joins and scores.

    nohup .venv/bin/python scripts/bench/shelf_again.py r2 dev2 > out/bench/r2/shelf.log 2>&1 &
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bree.shelf.store import run  # noqa: E402
from bree.sim.bench import clip_dirs, public_view  # noqa: E402

if __name__ == "__main__":
    name, split = sys.argv[1], sys.argv[2]
    for clip in clip_dirs(split):
        out, old = ROOT / "out" / "bench" / name / split / clip.name, ROOT / "out" / "bench" / split / clip.name / "pipeline"
        if (out / "run.json").exists() or not old.exists():
            continue
        (out / "pipeline").mkdir(parents=True, exist_ok=True)
        for f in old.glob("people_*.jsonl"):
            if not (out / "pipeline" / f.name).exists():
                (out / "pipeline" / f.name).symlink_to(f.resolve())
        t = time.time()
        run(public_view(clip, out), out)
        print(clip.name, round(time.time() - t), "s", flush=True)
