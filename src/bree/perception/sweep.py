"""Run detector configurations over the cached labelled frames and store the raw (low-confidence) detections, so
confidence / mask / NMS choices can be swept offline without running the model again.

    python -m bree.perception.sweep --config s640 --config t2x2 --channels 7,10,12,14,15,29

Detections go to <data>/perception/dets/<config>/chNN.json (box pixels + conf per labelled frame key; no images).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bree.perception import frames as fr
from bree.perception.labels import DATA_DIR, load_labels
from bree.perception.sliced import SliceSpec, sliced_predict

# name -> (weights, imgsz, SliceSpec)
CONFIGS: dict[str, tuple[str, int, SliceSpec]] = {
    "s640": ("yolo26s.pt", 640, SliceSpec()),
    "s960": ("yolo26s.pt", 960, SliceSpec()),
    "s1280": ("yolo26s.pt", 1280, SliceSpec()),
    "n640": ("yolo26n.pt", 640, SliceSpec()),
    "n1280": ("yolo26n.pt", 1280, SliceSpec()),
    "s640_t2x2": ("yolo26s.pt", 640, SliceSpec(2, 2)),
    "s640_t3x2": ("yolo26s.pt", 640, SliceSpec(3, 2)),
    "s640_t4x3": ("yolo26s.pt", 640, SliceSpec(4, 3)),
    "s960_t2x2": ("yolo26s.pt", 960, SliceSpec(2, 2, window_imgsz=960)),
}


def run_config(name: str, channels: list[int], data_dir: Path = DATA_DIR, conf: float = 0.15, device: str = "cuda:0",
               spec_override: tuple | None = None) -> dict:
    from ultralytics import YOLO
    wts, imgsz, spec = spec_override or CONFIGS[name]
    model = YOLO(str(Path(__file__).resolve().parents[3] / "models" / wts))
    out_dir = data_dir / "perception" / "dets" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    for ch in channels:
        lfs = load_labels(ch, data_dir)
        fr.extract(lfs, data_dir)
        res, t_tot, n = {}, 0.0, 0
        for lf in lfs:
            im = fr.load(lf, data_dir)
            if im is None:
                continue
            if n == 0:                                    # warm-up, not timed
                sliced_predict(model, im, spec, imgsz, conf, [0], device)
            t0 = time.perf_counter()
            b, c, _ = sliced_predict(model, im, spec, imgsz, conf, [0], device)
            t_tot += time.perf_counter() - t0
            n += 1
            res[lf.key] = {"shape": list(im.shape[:2]), "boxes": np.round(b, 1).tolist(), "confs": np.round(c, 3).tolist()}
        (out_dir / f"ch{ch:02d}.json").write_text(json.dumps(res))
        timing[ch] = round(1000 * t_tot / max(n, 1), 1)
        print(f"{name} ch{ch:02d}: {n} frames, {timing[ch]} ms/frame (shared GPU)", flush=True)
    return timing


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True, choices=list(CONFIGS))
    ap.add_argument("--channels", default="7,10,12,14,15,29")
    args = ap.parse_args(argv)
    chans = [int(c) for c in args.channels.split(",")]
    t = {c: run_config(c, chans) for c in args.config}
    (DATA_DIR / "perception" / "sweep_timing.json").write_text(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
