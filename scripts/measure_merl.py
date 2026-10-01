"""Measure the pose/track layer on MERL Shopping (real video, overhead camera, one shopper per video).

MERL labels hand actions, not products, and its products (boxed/bagged snacks) are not COCO classes,
so this measures the half of a PICK that the pose model owns (the reach), not product picks:
  reach recall      GT "Reach To Shelf" / "Hand In Shelf" instances with a wrist seen inside the shelf
                    zone during the instance (+-0.5 s)
  false reaches/min wrist-in-shelf runs (>= 2 frames) that overlap no labelled shelf action (1, 2, 3)
  tracks per video  one shopper per video, so every extra person track id is a fragmentation / ID switch
Uses MERL's test subjects (27-41) only. The shelf zone was drawn on a training-split frame (10_1).
Runs at 15 fps (every 2nd frame) like the pipeline. Writes results/merl_measure.json.
"""
import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from bree.detect.yolo import YoloBackend
from bree.events.observations import L_WRIST, R_WRIST
from bree.events.zones import Zone
from bree.hw import detect_hardware
from bree.track.bytetrack import Tracker

ROOT = Path("data/merl")
SHELF = Zone("shelf", "shelf", [(190, 30), (880, 30), (880, 300), (190, 300)])   # 920x680 frames
STRIDE, PAD_S, KPT_CONF = 2, 0.5, 0.3
ap = argparse.ArgumentParser()
ap.add_argument("limit", nargs="?", type=int, default=None)
ap.add_argument("--split", choices=("train", "test"), default="test")   # tune on train (subjects 1-20), report test
ap.add_argument("--det"), ap.add_argument("--pose")
ap.add_argument("--imgsz", type=int), ap.add_argument("--crop", type=int, default=160)
ap.add_argument("--person-conf", type=float, default=0.3)
ap.add_argument("--buffer", type=float, default=2.0)
ap.add_argument("--new-track-thresh", type=float)
ap.add_argument("--every", type=int, default=1)   # take every k-th video (quick tuning subsets)
ap.add_argument("--out", default="results/merl_measure.json")
args = ap.parse_args()
limit = args.limit

hw = detect_hardware()
det_w, pose_w, imgsz = args.det or hw.detect_model, args.pose or hw.pose_model, args.imgsz or hw.imgsz
backend = YoloBackend(f"models/{pose_w}", f"models/{det_w}", {}, device=hw.device, imgsz=imgsz,
                      pose_crop=args.crop, person_conf=args.person_conf, products=False)
bt = {"new_track_thresh": args.new_track_thresh} if args.new_track_thresh is not None else None
vids = sorted(glob.glob(str(ROOT / "Videos_MERL_Shopping_Dataset/*_crop.mp4")),
              key=lambda f: tuple(map(int, re.findall(r"(\d+)_(\d+)_crop", f)[0])))
subj = lambda v: int(Path(v).name.split("_")[0])
vids = [v for v in vids if (subj(v) >= 27 if args.split == "test" else subj(v) <= 20)][::args.every][:limit]

per_video, t_det = [], []
for vf in vids:
    stem = Path(vf).name.replace("_crop.mp4", "")
    tl = loadmat(ROOT / f"Labels_MERL_Shopping_Dataset/{stem}_label.mat")["tlabs"]
    gt = {k + 1: np.asarray(tl[k][0]).reshape(-1, 2) for k in range(5)}
    cap = cv2.VideoCapture(vf)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = Tracker(fps / STRIDE, person_buffer_s=args.buffer, bytetrack=bt)
    in_zone, ids, fi = [], set(), 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if fi % STRIDE == 0:
            t0 = time.perf_counter()
            persons, _ = tracker.update(*backend(im), im.shape[:2])
            t_det.append(time.perf_counter() - t0)
            ids.update(p.track_id for p in persons)
            hit = any(q and SHELF.contains(*q) for p in persons for q in (p.kpt(L_WRIST, KPT_CONF), p.kpt(R_WRIST, KPT_CONF)))
            in_zone.append((fi, hit))
        fi += 1
    frames = np.array([f for f, _ in in_zone])
    hits = np.array([h for _, h in in_zone])
    pad = PAD_S * fps

    reach_gt = np.concatenate([gt[1], gt[3]])
    found = [bool(hits[(frames >= s - pad) & (frames <= e + pad)].any()) for s, e in reach_gt]
    # detected runs of >= 2 sampled frames with a wrist in the shelf zone
    runs, start = [], None
    for f, h in zip(list(frames) + [None], list(hits) + [False]):
        if h and start is None:
            start = f
        elif not h and start is not None:
            runs.append((start, prev))
            start = None
        prev = f
    runs = [(s, e) for s, e in runs if e > s]
    shelf_gt = np.concatenate([gt[1], gt[2], gt[3]])
    false_runs = [r for r in runs if not any(r[0] <= e + pad and r[1] >= s - pad for s, e in shelf_gt)]
    minutes = fi / fps / 60
    per_video.append({"video": stem, "minutes": round(minutes, 2), "reach_instances": len(reach_gt),
                      "reach_found": int(sum(found)), "detected_reach_runs": len(runs),
                      "false_reach_runs": len(false_runs), "person_track_ids": len(ids)})
    print(per_video[-1], flush=True)

tot = lambda k: sum(v[k] for v in per_video)
mins = tot("minutes")
res = {
    "source": f"MERL Shopping Dataset, {'test subjects 27-41' if args.split == 'test' else 'train subjects 1-20'} (real video, overhead lab store, 1 shopper/video)",
    "license": "MERL research dataset (see data/LICENSES.md): evaluation only",
    "hardware": {"device": hw.device, "device_name": hw.device_name}, "stride": STRIDE, "split": args.split,
    "config": {"det": det_w, "pose": pose_w, "imgsz": imgsz, "crop": args.crop, "person_conf": args.person_conf,
               "buffer_s": args.buffer, "new_track_thresh": args.new_track_thresh}, "n_videos": len(per_video), "minutes": round(mins, 1),
    "reach_recall": tot("reach_found") / max(tot("reach_instances"), 1),
    "missed_reach_rate": 1 - tot("reach_found") / max(tot("reach_instances"), 1),
    "reach_instances": tot("reach_instances"),
    "false_reach_runs_per_min": tot("false_reach_runs") / mins,
    "extra_person_tracks_per_video": float(np.mean([v["person_track_ids"] - 1 for v in per_video])),
    "extra_person_tracks_per_min": sum(max(v["person_track_ids"] - 1, 0) for v in per_video) / mins,
    "detect_pose_track_ms_per_frame": 1000 * float(np.mean(t_det)),
    "per_video": per_video,
}
Path("results").mkdir(exist_ok=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print({k: v for k, v in res.items() if k != "per_video"})
