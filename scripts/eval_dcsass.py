"""DCSASS Shoplifting clips through our own perception (EVALUATION ONLY: cut from UCF-Crime, research use).
895 short clips from 28 UCF-Crime shoplifting videos, each labelled 0/1 (Labels/Shoplifting.csv).
Clip score = max frame score (max over people); reports clip-level AUC-ROC / AUC-PR / EER and the share of
clips that trigger at the fixed thresholds. Writes results/conceal_dcsass.json.
"""
import csv
import json
from pathlib import Path

import numpy as np

from bree.conceal import THRESH, frame_scores, load_bundle, metrics, model_track_scorer, rule_scores, video_triggers
from eval_ucf import extract_file, to_video

D = Path("data/dcsass/DCSASS Dataset")
CACHE = Path("out/dcsass_tracks")

if __name__ == "__main__":
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    from bree.track.bytetrack import Tracker
    CACHE.mkdir(parents=True, exist_ok=True)
    hw = detect_hardware()
    backend = YoloBackend(f"models/{hw.pose_model}", f"models/{hw.detect_model}", {}, device=hw.device,
                          imgsz=hw.imgsz, products=False)
    rows = [r for r in csv.reader(open(D / "Labels/Shoplifting.csv")) if len(r) == 3 and r[2] in "01"]
    clips = []
    for name, _, lab in rows:
        f = D / "Shoplifting" / f"{name.rsplit('_', 1)[0]}.mp4" / f"{name}.mp4"
        if f.exists():
            clips.append((to_video(extract_file(f, CACHE / f"{name}.npz", backend, Tracker)), int(lab)))
    print(f"{len(clips)} clips, {sum(l for _, l in clips)} shoplifting", flush=True)
    bundle = load_bundle(Path("models/conceal_poselift.pt"))
    y = np.array([l for _, l in clips])
    res = {"dataset": "DCSASS Shoplifting (Kaggle, cut from UCF-Crime), evaluation only", "perception": hw.to_dict(),
           "n_clips": len(clips), "n_shoplifting": int(y.sum()),
           "note": "clip-level labels; clips come from 28 source videos, so clips are not independent"}
    for k, sc in {"model": model_track_scorer(bundle), "rule": rule_scores}.items():
        s = np.array([frame_scores(v, sc).max(initial=0.0) for v, _ in clips])
        trig = {str(th): np.array([bool(video_triggers(v, sc, th)) for v, _ in clips]) for th in THRESH[k]}
        res[k] = {**metrics(y, s), "clip_trigger_rate": {th: {"shoplifting": float(t[y == 1].mean()),
                                                              "normal": float(t[y == 0].mean())} for th, t in trig.items()}}
        print(k, res[k], flush=True)
    Path("results/conceal_dcsass.json").write_text(json.dumps(res, indent=1))
