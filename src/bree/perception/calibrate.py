"""Produce a perception profile for a camera from a few minutes of footage.

    python -m bree.perception.calibrate --video clip.mp4 --camera ch29 --role counter_overhead \
        --out configs/perception/mystation_ch29.yaml [--samples 60] [--max-ms 250] [--labels-channel 29]

What it does (all on your own clip, nothing leaves the machine):
 1. samples frames evenly across the clip;
 2. runs a ladder of detector configurations (plain 640, 960, tiled 2x2, 3x2 ...), timing each;
 3. learns static ghost regions (bree.perception.ghost) from the richest configuration;
 4. picks the cheapest configuration that recovers >= `--recover` of the confident people the richest one finds
    (REAL-UNLABELLED proxy: agreement with the highest-recall candidate, NOT accuracy; false positives that survive
    ghost removal count towards the richest configuration too). With `--labels-channel` and agent labels for that
    channel, DEV frames score the candidates by F1 instead (REAL-MODEL-LABELLED) and held-out frames are never touched;
 5. writes the profile YAML (with provenance) and a JSON report next to it.

The thresholds this script picks are starting points. Look at `--preview DIR` images (written locally, they contain
people) before trusting a ghost region.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from bree.perception.ghost import find_ghosts, ghost_polygon
from bree.perception.profile import IgnoreRegion, PerceptionProfile, keep_mask, save_profile
from bree.perception.sliced import SliceSpec, sliced_predict

LADDER: list[tuple[str, str, int, SliceSpec]] = [     # name, weights, imgsz, slices ; cheapest first
    ("s640", "yolo26s.pt", 640, SliceSpec()),
    ("s960", "yolo26s.pt", 960, SliceSpec()),
    ("s640_t2x2", "yolo26s.pt", 640, SliceSpec(2, 2)),
    ("s640_t3x2", "yolo26s.pt", 640, SliceSpec(3, 2)),
]
ROLE_DEFAULTS = {     # tracker / gating starting points per camera role
    "door": dict(min_track_age_s=0.5, count_enter_exit=True, new_track_thresh=0.5),
    "counter_overhead": dict(min_track_age_s=1.0, count_enter_exit=False, new_track_thresh=0.5),
    "aisle": dict(min_track_age_s=1.0, count_enter_exit=False, new_track_thresh=0.5),
    "cooler": dict(min_track_age_s=1.0, count_enter_exit=False, new_track_thresh=0.5),
    "wide_floor": dict(min_track_age_s=1.0, count_enter_exit=False, new_track_thresh=0.5),
    "outdoor": dict(min_track_age_s=1.0, count_enter_exit=False, new_track_thresh=0.5),
}


def sample_frames(video: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    out, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, im = cap.retrieve()
            if ok:
                out.append(im)
        i += 1
    cap.release()
    return out


def _detect_all(model, frames, wts_imgsz_spec, device, conf=0.15):
    _, imgsz, spec = wts_imgsz_spec
    sliced_predict(model, frames[0], spec, imgsz, conf, [0], device)       # warm-up
    t0 = time.perf_counter()
    dets = []
    for im in frames:
        b, c, _ = sliced_predict(model, im, spec, imgsz, conf, [0], device)
        dets.append((b, c))
    return dets, 1000 * (time.perf_counter() - t0) / len(frames)


def calibrate(video: str, camera: str, role: str, samples: int = 60, max_ms: float | None = None, recover: float = 0.9,
              conf: float = 0.3, device: str | None = None, labels_channel: int | None = None,
              preview: Path | None = None) -> tuple[PerceptionProfile, dict]:
    from ultralytics import YOLO
    from bree.cli import _weights
    from bree.hw import detect_hardware
    device = device or detect_hardware().device
    frames = sample_frames(video, samples)
    if len(frames) < 10:
        raise SystemExit(f"only {len(frames)} frames could be sampled from {video}")
    shape = frames[0].shape[:2]
    models, results = {}, {}
    for name, wts, imgsz, spec in LADDER:
        m = models.setdefault(wts, YOLO(_weights(wts)))
        dets, ms = _detect_all(m, frames, (wts, imgsz, spec), device)
        results[name] = {"dets": dets, "ms": ms, "wts": wts, "imgsz": imgsz, "spec": spec}
    # Ghosts come from the richest configuration (most low-confidence boxes show recurring ones best).
    rich_name = max(results, key=lambda k: sum(int((c >= conf).sum()) for _, c in results[k]["dets"]))
    ghosts = find_ghosts(results[rich_name]["dets"], shape)
    ignore = [IgnoreRegion(f"ghost_{i}", ghost_polygon(g), f"static, mean conf {g.mean_conf}, in {g.frac_frames:.0%} of samples")
              for i, g in enumerate(ghosts)]
    prof = PerceptionProfile(camera=camera, role=role, ignore=ignore, person_conf=conf)
    # Candidate scoring.
    def confident(name):
        out = 0
        for b, c in results[name]["dets"]:
            out += int(keep_mask(prof, b, c, shape).sum())
        return out
    counts = {k: confident(k) for k in results}
    report = {"camera": camera, "role": role, "video": Path(video).name, "samples": len(frames),
              "frame_shape": list(shape), "ghosts": [g.__dict__ for g in ghosts],
              "candidates": {k: {"ms_per_frame": round(v["ms"], 1), "confident_people": counts[k]} for k, v in results.items()}}
    top = max(counts.values())
    ok = [k for k in results if counts[k] >= recover * top and (max_ms is None or results[k]["ms"] <= max_ms)]
    pick = ok[0] if ok else min(results, key=lambda k: results[k]["ms"])
    report["selection"] = {"rule": f"cheapest candidate with >= {recover:.0%} of the richest candidate's confident people"
                                   + (f" and <= {max_ms} ms/frame" if max_ms else ""),
                           "chosen": pick, "label_kind": "REAL-UNLABELLED (proxy, not accuracy)"}
    if labels_channel is not None:
        report["dev_labels"] = _score_on_dev_labels(labels_channel, results, models, device, prof)
    r = results[pick]
    prof.weights, prof.imgsz, prof.slices = r["wts"], r["imgsz"], r["spec"]
    for k, v in ROLE_DEFAULTS[role].items():
        setattr(prof, k, v)
    prof.provenance = {"calibrated_from": Path(video).name, "samples": len(frames), "label_kind": "REAL-UNLABELLED",
                       "selection": report["selection"]["rule"], "ms_per_frame_measured": round(r["ms"], 1),
                       "note": "starting point; verify ghost regions by eye before relying on them"}
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        im = frames[len(frames) // 2].copy()
        h, w = shape
        for reg in prof.ignore:
            pts = (np.array(reg.polygon) * [w, h]).astype(np.int32)
            cv2.polylines(im, [pts], True, (0, 0, 255), 4)
        cv2.imwrite(str(preview / f"{camera}_ghosts.jpg"), cv2.resize(im, (1280, int(1280 * h / w))))
    return prof, report


def _score_on_dev_labels(channel: int, results: dict, models: dict, device: str, prof: PerceptionProfile) -> dict:
    """Candidate F1 on the DEV label frames only (REAL-MODEL-LABELLED)."""
    from bree.perception import frames as fr
    from bree.perception.evalcounts import match_frame, summarize
    from bree.perception.labels import load_labels
    from bree.perception.scoring import f1
    lfs = [l for l in load_labels(channel) if l.split == "dev"]
    fr.extract(lfs)
    out = {"label_kind": "REAL-MODEL-LABELLED", "split": "dev", "frames": len(lfs), "candidates": {}}
    for name, r in results.items():
        scores = []
        for lf in lfs:
            im = fr.load(lf)
            if im is None:
                continue
            b, c, _ = sliced_predict(models[r["wts"]], im, r["spec"], r["imgsz"], 0.15, [0], device)
            k = keep_mask(prof, b, c, im.shape[:2])
            scores.append(match_frame(lf.centres, b[k], im.shape[:2]))
        s = summarize(scores)
        out["candidates"][name] = {**s, "f1": round(f1(s), 3)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--role", default="wide_floor", choices=list(ROLE_DEFAULTS))
    ap.add_argument("--out", required=True, help="profile YAML to write")
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--max-ms", type=float, default=None, help="per-frame detector time budget")
    ap.add_argument("--recover", type=float, default=0.9)
    ap.add_argument("--labels-channel", type=int, default=None, help="also score candidates on DEV agent labels")
    ap.add_argument("--preview", default=None, help="directory for a local ghost-region preview image (contains people)")
    args = ap.parse_args(argv)
    prof, report = calibrate(args.video, args.camera, args.role, args.samples, args.max_ms, args.recover,
                             labels_channel=args.labels_channel, preview=Path(args.preview) if args.preview else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_profile(prof, out)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"profile": str(out), "chosen": report["selection"]["chosen"], "ghosts": len(prof.ignore),
                      "candidates": report["candidates"]}, indent=1))


if __name__ == "__main__":
    main()
