"""UCF-Crime through our own perception (EVALUATION ONLY: research dataset).

1. extract  run the pipeline's detector + top-down pose + ByteTrack on each video at 15 fps and cache the
            person tracks to out/ucf_tracks/<video>.npz (boxes, keypoints, track ids; no pixels).
2. score    concealment classifier (trained on PoseLift, whose poses came from a different detector/pose
            model: HRNet) and the pose-only rule, on
              - the Shoplifting videos of the official test split, with the official temporal annotations
                -> frame AUC-ROC / AUC-PR / EER
              - a seeded sample of Testing_Normal_Videos -> would-be triggers per hour (same trigger
                definition as scripts/eval_retails.py). These are general surveillance scenes, not stores.
Writes results/conceal_ucf.json.
Usage: python scripts/eval_ucf.py [n_normal_videos=40]
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from bree.conceal import MERGE_S, MIN_RUN, THRESH, video_triggers, Track, Video, frame_scores, load_bundle, metrics, model_track_scorer, rule_scores

U = Path("data/ucf_crime")
CACHE = Path("out/ucf_tracks")
FPS = 15.0
N_NORMAL = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def members(zpath, pattern):
    with zipfile.ZipFile(zpath) as z:
        return sorted(n for n in z.namelist() if re.search(pattern, n))


def extract(zpath: Path, member: str, backend, Tracker) -> Path:
    name = Path(member).stem
    out = CACHE / f"{name}.npz"
    if out.exists():
        return out
    tmp = Path("out/ucf_tmp.mp4")
    with zipfile.ZipFile(zpath) as z, open(tmp, "wb") as f:
        f.write(z.read(member))
    cap = cv2.VideoCapture(str(tmp))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / FPS)), 1)
    tracker = Tracker(src_fps / step)
    rows, fi = [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if fi % step == 0:
            persons, _ = tracker.update(*backend(im), im.shape[:2])
            for p in persons:
                kp = p.keypoints if p.keypoints is not None else np.zeros((17, 3), np.float32)
                rows.append((fi, p.track_id, *p.bbox, *kp.reshape(-1)))
        fi += 1
    a = np.asarray(rows, np.float32).reshape(-1, 6 + 51)
    np.savez_compressed(out, rows=a, n_frames=fi, src_fps=src_fps, step=step)
    return out


def to_video(npz: Path, labels_src=None) -> Video:
    d = np.load(npz)
    a, step, n = d["rows"], int(d["step"]), int(d["n_frames"])
    tracks = {}
    for tid in np.unique(a[:, 1]) if len(a) else []:
        r = a[a[:, 1] == tid]
        tracks[int(tid)] = Track((r[:, 0] // step).astype(int), r[:, 6:].reshape(-1, 17, 3), r[:, 2:6])
    n15 = (n + step - 1) // step
    labels = np.zeros(n15, np.int8)
    if labels_src:   # official temporal annotation, in source-frame numbers
        for s, e in labels_src:
            labels[s // step:e // step + 1] = 1
    return Video(npz.stem, "ucf", "test", n15, tracks, labels)




if __name__ == "__main__":
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    from bree.track.bytetrack import Tracker
    CACHE.mkdir(parents=True, exist_ok=True)
    hw = detect_hardware()
    backend = YoloBackend(f"models/{hw.pose_model}", f"models/{hw.detect_model}", {}, device=hw.device,
                          imgsz=hw.imgsz, products=False)

    ann = {}
    for line in (U / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").read_text().split("\n"):
        p = line.split()
        if len(p) >= 6 and p[1] == "Shoplifting":
            iv = [(int(p[2]), int(p[3]))] + ([(int(p[4]), int(p[5]))] if int(p[4]) >= 0 else [])
            ann[Path(p[0]).stem] = iv
    shop = [m for m in members(U / "Anomaly-Videos-Part-4.zip", r"Shoplifting/.*\.mp4$") if Path(m).stem in ann]
    normal_all = members(U / "Testing_Normal_Videos.zip", r"\.mp4$")
    rng = np.random.default_rng(0)
    normal = [normal_all[i] for i in sorted(rng.choice(len(normal_all), min(N_NORMAL, len(normal_all)), replace=False))]
    print(f"{len(shop)} shoplifting test videos, {len(normal)} of {len(normal_all)} normal test videos", flush=True)

    shop_v = [to_video(extract(U / "Anomaly-Videos-Part-4.zip", m, backend, Tracker), ann[Path(m).stem]) for m in shop]
    print("shoplifting extracted", flush=True)
    norm_v = [to_video(extract(U / "Testing_Normal_Videos.zip", m, backend, Tracker)) for m in normal]
    print("normal extracted", flush=True)

    bundle = load_bundle(Path("models/conceal_poselift.pt"))
    scorers = {"model": model_track_scorer(bundle), "rule": rule_scores}
    hours = sum(v.n_frames for v in norm_v) / FPS / 3600
    res = {"dataset": "UCF-Crime official test split (Dropbox of the authors), evaluation only",
           "perception": {**hw.to_dict(), "fps": FPS}, "model": "models/conceal_poselift.pt",
           "shoplifting": {"videos": [v.name for v in shop_v]},
           "normal": {"videos": [v.name for v in norm_v], "hours": hours}}
    y = np.concatenate([v.labels for v in shop_v])
    for k, sc in scorers.items():
        s = np.concatenate([frame_scores(v, sc) for v in shop_v])
        res["shoplifting"][k] = metrics(y, s)
        res["normal"][k] = {"triggers_per_hour_at": {str(th): sum(len(video_triggers(v, sc, th)) for v in norm_v) / hours
                                                      for th in THRESH[k]}}
        print(k, res["shoplifting"][k], res["normal"][k], flush=True)
    Path("results/conceal_ucf.json").write_text(json.dumps(res, indent=1))
