#!/usr/bin/env python3
"""Platform measurements on recorded footage (local only): role stability and overlay-clock reads, per segment.

  PYTHONPATH=src python scripts/platform_eval.py --footage ~/Projects/bree-data/real/trumanrd --out ~/Projects/bree-data/platform

Role truth below is the camera description from the pilot brief (REAL-MODEL-LABELLED: an agent looked at the
stills; the operator has not confirmed), and the role thresholds were adjusted after seeing the same 7 channels,
so the match rate is a tuning-set number, not an accuracy. Stability = does one 5-minute segment alone give the
same role as the other segments. Writes JSON only (no images).
"""
import argparse
import json
import time
from pathlib import Path

from bree.ingest.overlay import check_file_clock
from bree.onboard.discover import ChannelSample, sample_file
from bree.onboard.roles import suggest_role
from bree.onboard.stats import YoloScene, compute_stats

BRIEF = {7: "door", 29: "register", 12: "aisle", 14: "aisle", 15: "cooler", 10: "cooler", 20: "outdoor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--footage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-clock", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.footage).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    scene = YoloScene()
    res = {"roles": [], "clock": []}
    for chdir in sorted(root.glob("ch*")):
        ch = int(chdir.name[2:])
        for seg in sorted(chdir.glob("*.mp4")):
            fr, ts, fps, size = sample_file(seg, 4)
            s = ChannelSample(ch, fr, ts, size, fps)
            sug = suggest_role(ch, compute_stats(s, scene).flat())
            res["roles"].append({"channel": ch, "segment": seg.name, "brief": BRIEF.get(ch), "suggested": sug.role,
                                 "confidence": round(sug.confidence, 2), "needs_review": sug.needs_review})
            print(res["roles"][-1], flush=True)
            if not a.skip_clock:
                c = check_file_clock(seg, n=6)
                res["clock"].append({"channel": ch, "segment": seg.name, **c["estimate"], "slope": c["overlay_seconds_per_file_second"]})
                print(res["clock"][-1], flush=True)
    (out / "eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
