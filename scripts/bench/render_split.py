#!/usr/bin/env python
"""Render every clip of a benchmark split (scripts/bench/manifest.json) with render_clip.mjs, a few at a time.

    .venv/bin/python scripts/bench/render_split.py dev [--jobs 3]        # skips clips that are already rendered
    .venv/bin/python scripts/bench/render_split.py train --seeds 1240:1250
    .venv/bin/python scripts/bench/render_split.py dev2 --jobs 3         # generator 2 (the split's "gen" in the manifest)
    .venv/bin/python scripts/bench/render_split.py checkpoint --next 6   # the next 6 unused seeds of the held-out pool, marked used first
    .venv/bin/python scripts/bench/render_split.py checkpoint            # finish the batches already marked used (after an interrupted render)

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
    ap.add_argument("--next", type=int, help="checkpoint pool: take this many unused seeds, mark them used in the manifest, render them")
    a = ap.parse_args()
    if a.split in ("train", "scratch") and not a.seeds:
        sys.exit(f"{a.split} is a reserved seed range: pass --seeds A:B")
    sp, pool = MAN["splits"][a.split], None
    if "batches" in sp:      # a seed is used from the moment it is handed out, whether or not its render finishes
        pool = [s for s in seeds_of(a.split) if s not in sp["used"]]
        if a.next:
            if len(pool) < a.next:
                sys.exit(f"{a.split}: only {len(pool)} unused seeds left")
            sp["used"] += pool[:a.next]
            sp["batches"].append(pool[:a.next])
            (HERE / "manifest.json").write_text(json.dumps(MAN, indent=1) + "\n")
            print(f"{a.split}: batch {len(sp['batches'])} is seeds {pool[:a.next]}; {len(pool) - a.next} unused seeds left", flush=True)
        if not sp["batches"]:
            sys.exit(f"{a.split}: nothing handed out yet: pass --next 6")
    elif a.next:
        sys.exit("--next is for the checkpoint pool")
    gen = ["--gen", str(sp["gen"])] if sp.get("gen", 1) != 1 else []

    def one(seed: int) -> int:
        out = clip_dir(a.split, seed)
        if (out / "clip.json").exists():
            return 0
        out.mkdir(parents=True, exist_ok=True)
        for _ in range(3):      # headless Chrome sometimes closes mid clip ("browser has been closed"): start the clip again
            with (out / "render.log").open("w") as log:
                code = subprocess.run(["node", str(HERE / "render_clip.mjs"), "--seed", str(seed), "--out", str(out), "--layout",
                                       str(Path.home() / "bree/software/shared/layouts" / MAN["layout"]), *gen], stdout=log, stderr=log).returncode
            if code == 0:
                break
        return code
    todo = (sp["batches"][-1] if a.next else [s for b in sp["batches"] for s in b]) if pool is not None else seeds_of(a.split, a.seeds)
    with ThreadPoolExecutor(a.jobs) as ex:
        codes = list(ex.map(one, todo))
    clips = json.loads((HERE / "clips.json").read_text()) if (HERE / "clips.json").exists() else {}
    for seed in todo:
        d = clip_dir(a.split, seed)
        if (d / "clip.json").exists():
            c, r = json.loads((d / "clip.json").read_text()), json.loads((d / "truth" / "render.json").read_text())
            keys = ("sim_seconds", "frames", "shoppers", "staff", "thieves", "picks", "outcomes", "zones", "features", "aisles", "group_picks", "put_backs_into_another_slot", "acts",
                    "picks_without_an_item_camera", "picks_with_two_rendered_views", "item_cameras", "frames_drawn", "frames_repeated_because_nothing_moved", "render_seconds")
            clips.setdefault(a.split, {})[str(seed)] = {k: r[k] for k in keys if k in r} | {
                "cameras": c["cameras"], "same_shelf_pair": (c.get("params") or r["params"])["same_shelf_pair"], "simulator": c["simulator"]["commit"]}
    (HERE / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
    done = [clips[a.split][str(s)] for s in todo if str(s) in clips.get(a.split, {})]
    if done:
        sec, cams = [d["render_seconds"] for d in done], [len(d["cameras"]) for d in done]
        print(f"render cost: {sum(sec) / len(sec) / 60:.1f} min per clip (min {min(sec) / 60:.1f}, max {max(sec) / 60:.1f}), {sum(cams) / len(cams):.1f} cameras per clip, {a.jobs} clips at a time")
    print(f"{a.split}: {sum(1 for c in codes if c == 0)} of {len(todo)} clips ok; see scripts/bench/clips.json")
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
