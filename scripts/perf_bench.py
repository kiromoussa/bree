"""A/B benchmark of `bree run` profiles on one clip: pipeline_fps, per-stage ms, process CPU%.

  python scripts/perf_bench.py --source clip.mp4 --frames 1800 --reps 3 \
      --profile baseline=BREE_PERF=off --profile fast=BREE_PERF=fast --out ~/Projects/bree-data/perf

Profiles are run round-robin (A B A B ...) so background load hits them alike. Each run is a child
process; CPU% is (user+sys CPU seconds of the child) / wall seconds, so 100 = one full core.
Prints a Markdown table of medians. Outputs go under --out; never point it at a shared folder.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics as st
import subprocess
import sys
import time
from pathlib import Path


def run_once(source: str, frames: int, env_extra: dict, out: Path, extra: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "bree.cli", "run", "--source", source, "--out", str(out),
           "--max-frames", str(frames), "--runtime", "pytorch", *extra]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env={**os.environ, **env_extra}, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if p.returncode:
        raise SystemExit(f"run failed:\n{p.stderr[-2000:]}")
    s = json.loads((out / "summary.json").read_text())
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return {"pipeline_fps": s["pipeline_fps"], "frames": s["frames"], "wall_total_s": round(wall, 1),
            "cpu_pct": round(100 * cpu / wall, 1), "cpu_s": round(cpu, 1), **{f"{k}_ms": v for k, v in s["stage_ms"].items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=1800)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile", action="append", required=True, help="name=K=V[,K=V...][|extra cli args]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    profiles = {}
    for spec in a.profile:
        name, _, rest = spec.partition("=")
        envs, _, extra = rest.partition("|")
        profiles[name] = ({k: v for k, v in (kv.split("=", 1) for kv in envs.split(",") if kv)}, extra.split())
    out = Path(a.out).expanduser()
    runs: dict[str, list[dict]] = {n: [] for n in profiles}
    for rep in range(a.reps):
        for name, (env, extra) in profiles.items():
            r = run_once(a.source, a.frames, env, out / name, extra)
            runs[name].append(r)
            print(name, rep, json.dumps(r), flush=True)
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps({"source": Path(a.source).name, "frames": a.frames, "profiles": a.profile, "runs": runs}) + "\n")
    keys = ["pipeline_fps", "cpu_pct", "detect_ms", "track_ms", "render_ms"]
    print("\n| profile | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
    for n, rs in runs.items():
        print(f"| {n} | " + " | ".join(f"{st.median(r[k] for r in rs):.2f}" for k in keys) + " |")


if __name__ == "__main__":
    main()
