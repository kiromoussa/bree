"""Tamper monitor on real footage: false alarms on untouched segments (REAL-UNLABELLED) and detection of tampering
applied to the real frames (SIMULATED tampering). python scripts/theft_tamper_eval.py [--channels 7,29] [--t0 120]
Writes ~/Projects/bree-data/theft/tamper_eval.json. The segment that is ordered last per channel is the held-out one:
thresholds were set on the others (bree.theft.tamper defaults)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = Path.home() / "Projects" / "bree-data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="trumanrd")
    ap.add_argument("--channels", default=None)
    ap.add_argument("--t0", type=float, default=150.0, help="second of the segment where tampering starts")
    ap.add_argument("--segments", default=None, help="comma list of segment indexes per channel (default all)")
    a = ap.parse_args()
    from bree.theft.tamper_sim import SHOULD_ALARM, TAMPERS, latency, load_grey, run_tamper
    chans = [f"ch{int(c):02d}" for c in a.channels.split(",")] if a.channels else None
    res = {"clean": [], "tampered": {k: [] for k in TAMPERS if k != "none"}}
    cam_s = 0.0
    for ch in sorted((DATA / "real" / a.station).glob("ch*")):
        if chans and ch.name not in chans:
            continue
        for mp4 in sorted(ch.glob("*.mp4")):
            cdir = DATA / "theft" / "cache" / a.station / ch.name / mp4.stem
            if mp4.stat().st_size < 1_000_000 or not (cdir / "info.json").exists():
                continue
            info = json.loads((cdir / "info.json").read_text())
            frames = load_grey(mp4, cdir, stride=info["stride"])
            dur = frames[-1][0]
            cam_s += dur
            ev = run_tamper(frames, "none", 1e9)
            res["clean"].append({"cam": ch.name, "seg": mp4.stem, "seconds": round(dur), "alarms": [(e.kind, round(e.t_start, 1)) for e in ev if e.phase == "start"]})
            for k in res["tampered"]:
                lat = latency(run_tamper(frames, k, a.t0), SHOULD_ALARM[k], a.t0)
                res["tampered"][k].append({"cam": ch.name, "seg": mp4.stem, **lat})
            print(ch.name, mp4.stem, "clean alarms:", res["clean"][-1]["alarms"], flush=True)
    res["camera_seconds"] = round(cam_s)
    Path(DATA / "theft" / "tamper_eval.json").write_text(json.dumps(res, indent=1))
    n_clean = sum(len(c["alarms"]) for c in res["clean"])
    print(f"REAL-UNLABELLED: {n_clean} tamper alarm(s) on {cam_s / 3600:.2f} camera-hours of untouched footage")
    for k, rows in res["tampered"].items():
        det = [r for r in rows if r["detected"]]
        want = SHOULD_ALARM[k]
        bad = [r for r in rows if r["alarms"] and (want is None or r["other_kinds"])]
        d = sorted(r["delay_s"] for r in det)
        print(f"SIMULATED {k:11s}: expected {want}: detected {len(det)}/{len(rows)}{" (none expected: none is correct)" if want is None else ""}, median delay {d[len(d) // 2] if d else None}s, "
              f"runs with an unexpected kind {len(bad)}")


if __name__ == "__main__":
    main()
