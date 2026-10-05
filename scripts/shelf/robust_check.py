"""How the pixel comparison holds up when the whole picture shifts or the light changes (SIMULATED, one TRAIN-seed clip,
no detector). The disturbance is applied to the decoded frames of every item camera; scoring is eval_shelf.score_clip.

    .venv/bin/python scripts/shelf/robust_check.py 4903          -> out/shelf/robust_4903.md
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
CASES = {
    "none": lambda im, f, fps: im,
    "shift 3 px from frame 50": lambda im, f, fps: im if f < 50 else cv2.warpAffine(im, np.float32([[1, 0, 3], [0, 1, 0]]), im.shape[1::-1], borderMode=cv2.BORDER_REPLICATE),
    "shift 8 px from frame 50": lambda im, f, fps: im if f < 50 else cv2.warpAffine(im, np.float32([[1, 0, 8], [0, 1, 0]]), im.shape[1::-1], borderMode=cv2.BORDER_REPLICATE),
    "dim to 30 percent over 30 s": lambda im, f, fps: cv2.convertScaleAbs(im, alpha=1 - 0.7 * min(f / fps / 30, 1)),
    "10 percent brightness step at 5 s": lambda im, f, fps: im if f < 5 * fps else cv2.convertScaleAbs(im, alpha=0.9),
    "brighten by 60 percent at 5 s": lambda im, f, fps: im if f < 5 * fps else cv2.convertScaleAbs(im, alpha=1.6),
}


def one(args):
    clip, cam_id, case = args
    from bree.shelf.events import ShelfCamera, frames
    from bree.sim.bench import load_calibration
    cv2.setNumThreads(1)
    clip = Path(clip)
    layout, fps = json.loads((clip / "layout.json").read_text()), float(json.loads((clip / "clip.json").read_text())["fps"])
    sc, out = ShelfCamera(cam_id, load_calibration(clip)[cam_id], layout, fps), []
    for f, im in enumerate(frames(clip / f"{cam_id}.mp4")):
        out += sc.update(CASES[case](im, f, fps))
    return out + sc.finish(), sc.status


def main() -> None:
    from eval_shelf import ITEM_KINDS, score_clip, table
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4903
    assert 1000 <= seed < 5000, "TRAIN seeds only"
    clip = ROOT / f"data/synth/bench/train/clip_{seed}"
    meta = json.loads((clip / "clip.json").read_text())
    kinds = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    cams = [c for c in meta["cameras"] if kinds[c] in ITEM_KINDS]
    slots = {s["id"]: s for s in json.loads((clip / "layout.json").read_text())["slots"]}
    lines = [f"# Pixel comparison under a shift or a light change (SIMULATED, TRAIN seed {seed}, no detector, {len(cams)} item cameras)", "",
             "| disturbance | pairs | recall | right slot | take events | precision | strict precision | status records |", "|---|---|---|---|---|---|---|---|"]
    only = sys.argv[2:]
    with ProcessPoolExecutor(5) as ex:
        for case in [c for c in CASES if not only or any(o in c for o in only)]:
            runs = list(ex.map(one, [(str(clip), c, case) for c in cams]))
            evs = sorted((e for r in runs for e in r[0]), key=lambda e: e["t"])
            status: dict[str, int] = {}
            for s in (s for r in runs for s in r[1]):
                status[s["status"]] = status.get(s["status"], 0) + 1
            pairs, rows = score_clip(clip, evs, kinds, meta["cameras"], 20.0, 3.0, 0.6, slots)
            r = table(pairs, rows, "take")[-1]
            lines.append(f"| {case} | {r['pairs']} | {r['recall']} | {r['right_slot']} | {r['events']} | {r['precision']} | {r['strict_precision']} | {status or 'none'} |")
            print(lines[-1], flush=True)
    (ROOT / f"out/shelf/robust_{seed}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
