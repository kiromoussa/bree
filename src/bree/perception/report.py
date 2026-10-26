"""Before / after person-count scorecard on the agent-labelled frames (REAL-MODEL-LABELLED, never ground truth).

    python -m bree.perception.report [--out DIR]

For every labelled channel: the baseline (bree default: yolo26s, imgsz 640, conf 0.3, no filters) is scored on DEV and
HELDOUT. The tuned setting is chosen on DEV frames only (detector configuration x confidence, with ghost regions learned
from the DEV detections without labels), frozen, then scored once on HELDOUT. Reads the cached detections written by
bree.perception.sweep, so it needs no model run. The labels are tiny (about 40 frames per segment, one afternoon).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bree.perception.ghost import find_ghosts, ghost_polygon
from bree.perception.labels import DATA_DIR, load_labels
from bree.perception.profile import IgnoreRegion, PerceptionProfile, keep_mask
from bree.perception.scoring import f1, load_dets, score

CONFIGS = ["s640", "s960", "s1280", "s640_t2x2", "s640_t3x2", "s960_t2x2"]
CONFS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
BASELINE = ("s640", 0.30)


def learn_ghosts(config: str, channel: int, data_dir: Path = DATA_DIR) -> list[IgnoreRegion]:
    """Static low-confidence recurring boxes in the DEV frames' detections (labels are not used)."""
    dets = load_dets(config, channel, data_dir)
    dev = [lf.key for lf in load_labels(channel, data_dir) if lf.split == "dev" and lf.key in dets]
    if not dev:
        return []
    shape = tuple(dets[dev[0]]["shape"])
    seq = []
    for k in dev:
        c = np.array(dets[k]["confs"], float)
        b = np.array(dets[k]["boxes"], float).reshape(-1, 4)
        m = c >= 0.2
        seq.append((b[m], c[m]))
    return [IgnoreRegion(f"ghost_{i}", ghost_polygon(g), f"static, mean conf {g.mean_conf}, in {g.frac_frames:.0%} of dev frames")
            for i, g in enumerate(find_ghosts(seq, shape, min_frames=8))]


def calibrated_ghosts(channel: int, data_dir: Path = DATA_DIR) -> list[IgnoreRegion] | None:
    """Ghost regions bree.perception.calibrate wrote from a segment that is not the held-out one (60 sampled frames)."""
    from bree.perception.profile import load_profile
    p = data_dir / "perception" / "calib" / f"ch{channel:02d}.yaml"
    return load_profile(p).ignore if p.exists() else None


def _filt(ignore: list[IgnoreRegion]):
    prof = PerceptionProfile(person_conf=0.0, ignore=ignore)
    return lambda b, c, shape: keep_mask(prof, b, c, shape)


def _rank(s: dict) -> tuple:
    """DEV selection key: F1 when the DEV frames contain people, otherwise fewest false-alarm frames."""
    return (f1(s), -s.get("false_alarm_frames", 0), s.get("exact_count", 0))


def run(channels=(7, 10, 12, 14, 15, 29), data_dir: Path = DATA_DIR) -> dict:
    timing = json.loads((data_dir / "perception" / "sweep_timing.json").read_text())
    out = {}
    for ch in channels:
        ghosts = calibrated_ghosts(ch, data_dir)
        if ghosts is None:
            ghosts = learn_ghosts("s640", ch, data_dir)
        filt = _filt(ghosts)
        base = {sp: score(BASELINE[0], ch, sp, BASELINE[1], data_dir=data_dir) for sp in ("dev", "heldout")}
        grid = [(cfg, conf, score(cfg, ch, "dev", conf, filt, data_dir)) for cfg in CONFIGS for conf in CONFS]
        top = max(_rank(s)[0] for _, _, s in grid)
        # cheapest configuration within 0.01 F1 of the best on DEV (a tie is not worth a slower detector); ties in
        # confidence go to the one nearest the default, so a DEV set without people (ch15) cannot pick a threshold
        near = [(timing.get(cfg, {}).get(str(ch)) or 0, -_rank(s)[1], abs(conf - BASELINE[1]), cfg, conf, s) for cfg, conf, s in grid
                if _rank(s)[0] >= top - 0.01]
        _, _, _, cfg, conf, best_s = min(near, key=lambda x: x[:3])
        pick = (cfg, conf)
        held = score(pick[0], ch, "heldout", pick[1], filt, data_dir)
        out[ch] = {"baseline": base, "tuned": {"config": pick[0], "conf": pick[1], "ghosts": [g.__dict__ for g in ghosts],
                                               "dev": best_s, "heldout": held,
                                               "ms_per_frame": timing.get(pick[0], {}).get(str(ch))},
                   "label_kind": "REAL-MODEL-LABELLED"}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DATA_DIR / "perception" / "scorecard.json"))
    a = ap.parse_args(argv)
    res = run()
    Path(a.out).write_text(json.dumps(res, indent=1))
    keys = ("recall", "precision", "exact_count", "mae", "false_alarm_frames")
    for ch, r in res.items():
        t = r["tuned"]
        print(f"ch{ch:02d} tuned {t['config']} conf {t['conf']} ghosts {len(t['ghosts'])} ({t['ms_per_frame']} ms)")
        for sp in ("dev", "heldout"):
            print(f"   {sp:8s} base  ", {k: r["baseline"][sp].get(k) for k in keys})
            print(f"   {sp:8s} tuned ", {k: t[sp].get(k) for k in keys})


if __name__ == "__main__":
    main()
