"""python -m bree.shelfstate run VIDEO [VIDEO ...] --camera ch14 --regions configs/shelfstate/trumanrd_ch14.yaml --out out/shelf14

Real-footage state monitor: shelf stock changes, restocks, floor hazards, cooler doors. Heuristic, uncalibrated, and
measured only on a few minutes of one store (docs/ultra/shelf.md). Confidences are scores, not probabilities."""
from __future__ import annotations

import argparse
import json

from bree.shelfstate.regions import load_regions
from bree.shelfstate.runner import run_files


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bree.shelfstate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run on recorded segments, played as one continuous timeline")
    r.add_argument("videos", nargs="+")
    r.add_argument("--camera", required=True)
    r.add_argument("--regions", help="regions YAML (default: one shelf region over the whole picture)")
    r.add_argument("--out", required=True)
    r.add_argument("--fps", type=float, default=1.0)
    r.add_argument("--no-person-gate", action="store_true", help="skip the person detector (many false events near people)")
    r.add_argument("--cache", help="folder of sampled frames + person boxes made by an earlier run (skips decoding)")
    r.add_argument("--clock-offset-h", type=float, default=0.0, help="hours added to the file-name clock for event wall times")
    a = ap.parse_args(argv)
    det = None
    if not a.no_person_gate:
        from bree.shelfstate.frames import PersonDetector
        det = PersonDetector()
    s = run_files(a.videos, a.camera, load_regions(a.regions) if a.regions else None, a.out, fps=a.fps, detector=det,
                  cache_dir=a.cache, clock_offset_h=a.clock_offset_h, verbose=True)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
