#!/usr/bin/env python
"""Render every clip of a benchmark split (scripts/bench/manifest.json) with render_clip.mjs, a few at a time.

    .venv/bin/python scripts/bench/render_split.py dev [--jobs 3]        # skips clips that are already rendered
    .venv/bin/python scripts/bench/render_split.py train --seeds 1240:1250

Then rewrites scripts/bench/clips.json: per clip the cameras, frame count, what happens in it and the render time.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAN = json.loads((HERE / "manifest.json").read_text())


def seeds_of(split: str, override: str | None = None) -> list[int]:
    s = override or MAN["splits"][split]["seeds"]
    if isinstance(s, str):
        a, b = map(int, s.split(":"))
        if override:
            lo, hi = map(int, MAN["splits"][split]["seeds"].split(":")) if isinstance(MAN["splits"][split]["seeds"], str) else (a, b)
            assert lo <= a and b <= hi, f"seeds {s} are outside the {split} range {lo}:{hi}"
        return list(range(a, b))
    return list(s)


def clip_dir(split: str, seed: int) -> Path:
    return ROOT / MAN["root"] / split / f"clip_{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=list(MAN["splits"]))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seeds", help="A:B inside the split's range (train: the range is not rendered by default)")
    a = ap.parse_args()
    if a.split in ("train", "scratch") and not a.seeds:
        sys.exit(f"{a.split} is a reserved seed range: pass --seeds A:B")

    def one(seed: int) -> int:
        out = clip_dir(a.split, seed)
        if (out / "clip.json").exists():
            return 0
        out.mkdir(parents=True, exist_ok=True)
        with (out / "render.log").open("w") as log:
            return subprocess.run(["node", str(HERE / "render_clip.mjs"), "--seed", str(seed), "--out", str(out), "--layout",
                                   str(Path.home() / "bree/software/shared/layouts" / MAN["layout"])], stdout=log, stderr=log).returncode
    todo = seeds_of(a.split, a.seeds)
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(one, todo))
    clips = json.loads((HERE / "clips.json").read_text()) if (HERE / "clips.json").exists() else {}
    for seed in todo:
        d = clip_dir(a.split, seed)
        if (d / "clip.json").exists():
            c, r = json.loads((d / "clip.json").read_text()), json.loads((d / "truth" / "render.json").read_text())
            clips.setdefault(a.split, {})[str(seed)] = {k: r[k] for k in ("sim_seconds", "frames", "shoppers", "thieves", "picks", "outcomes", "zones",
                                                                         "picks_without_an_item_camera", "picks_with_two_rendered_views", "render_seconds")} | {
                "cameras": c["cameras"], "same_shelf_pair": c["params"]["same_shelf_pair"], "simulator": c["simulator"]["commit"]}
    (HERE / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
    print(f"{a.split}: {sum(1 for c in codes if c == 0)} of {len(todo)} clips ok; see scripts/bench/clips.json")
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
