"""Detection cache + offline tracker bench on real segments.

    python -m bree.perception.trackbench cache --channel 12 --segment 20261025_182855 --config s640_t2x2 --stride 3
    python -m bree.perception.trackbench run   --channel 12 --segment 20261025_182855 --config s640_t2x2 --profile p.yaml

`cache` runs the detector on every `stride`-th frame (low confidence floor 0.15, no pose) and stores boxes in
<data>/perception/trackdets/. `run` replays them through the profile's filters, tracker and track gate and prints
track statistics: ids, how many lived at least 1 s, median life, and (if the segment has agent labels) the count of
confirmed tracks alive at each labelled time against the label (REAL-MODEL-LABELLED, not ground truth).
Boxes carry no images, but are stored outside git with the footage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bree.detect.base import Detections
from bree.perception.evalcounts import match_frame, summarize
from bree.perception.labels import DATA_DIR, load_labels
from bree.perception.profile import PerceptionProfile, keep_mask, load_profile
from bree.perception.sweep import CONFIGS, resolve_weights


def det_path(channel: int, segment: str, config: str, stride: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "perception" / "trackdets" / f"ch{channel:02d}_{segment}_{config}_s{stride}.json"


def cache(channel: int, segment: str, config: str, stride: int = 3, max_s: float | None = None, device: str = "cuda:0",
          data_dir: Path = DATA_DIR) -> Path:
    from ultralytics import YOLO
    from bree.perception.sliced import sliced_predict
    wts, imgsz, spec = CONFIGS[config]
    model = YOLO(resolve_weights(wts))
    video = data_dir / "real" / "trumanrd" / f"ch{channel:02d}" / f"{segment}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, im = cap.retrieve()
            if not ok:
                break
            if max_s and i / fps > max_s:
                break
            b, c, _ = sliced_predict(model, im, spec, imgsz, 0.15, [0], device)
            out.append({"f": i, "shape": list(im.shape[:2]), "boxes": np.round(b, 1).tolist(), "confs": np.round(c, 3).tolist()})
        i += 1
    p = det_path(channel, segment, config, stride, data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"fps": fps, "stride": stride, "frames": out}))
    return p


def replay(det_file: Path, profile: PerceptionProfile, channel: int, segment: str, label_split: str = "dev",
           data_dir: Path = DATA_DIR) -> dict:
    from bree.perception.gate import GatedTracker
    d = json.loads(det_file.read_text())
    eff_fps = d["fps"] / d["stride"]
    trk = GatedTracker(eff_fps, profile)
    adapt = None
    if profile.adaptive_ghost_s > 0:
        from bree.perception.adaptive import AdaptiveGhostMask
        adapt = AdaptiveGhostMask(eff_fps, min_s=profile.adaptive_ghost_s)
    raw = {}                                      # id -> frame indices (before the gate): needs the inner tracker
    life: dict[int, list[int]] = {}
    alive: list[tuple[float, list[tuple]]] = []   # (t, confirmed person boxes)
    for fr in d["frames"]:
        b = np.array(fr["boxes"], float).reshape(-1, 4)
        c = np.array(fr["confs"], float)
        shape = tuple(fr["shape"])
        k = keep_mask(profile, b, c, shape)
        if adapt is not None:
            k[np.where(k)[0][~adapt.update(b[k], c[k], shape)]] = False
        det = Detections(b[k].astype(np.float32), c[k].astype(np.float32), ["person"] * int(k.sum()), None)
        from bree.track.bytetrack import Tracker
        inner = Tracker.update(trk, det, Detections.empty(), shape)[0]       # tracker output before the gate
        for p in inner:
            raw.setdefault(p.track_id, []).append(fr["f"])
        persons = trk.gate.filter(inner)
        for p in persons:
            life.setdefault(p.track_id, []).append(fr["f"])
        alive.append((fr["f"] / d["fps"], [p.bbox for p in persons], shape))
    fps = d["fps"]
    secs = lambda fs: (fs[-1] - fs[0]) / fps + d["stride"] / fps
    raw_len = sorted(secs(v) for v in raw.values())
    conf_len = sorted(secs(v) for v in life.values())
    stats = {"frames": len(d["frames"]), "effective_fps": round(eff_fps, 2),
             "ids_ungated": len(raw), "ids_confirmed": len(life),
             "ids_ge_1s_ungated": sum(x >= 1 for x in raw_len), "median_life_s_ungated": round(float(np.median(raw_len)), 2) if raw_len else 0,
             "median_life_s_confirmed": round(float(np.median(conf_len)), 2) if conf_len else 0}
    scores = []
    for lf in load_labels(channel, data_dir):
        if lf.segment != segment or lf.split != label_split:
            continue
        t, boxes, shape = min(alive, key=lambda a: abs(a[0] - lf.t_s))
        if abs(t - lf.t_s) > 0.5:
            continue
        scores.append(match_frame(lf.centres, np.array(boxes, float).reshape(-1, 4), shape))
    stats["tracked_count_vs_labels"] = {**summarize(scores), "label_kind": "REAL-MODEL-LABELLED", "split": label_split}
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["cache", "run"])
    ap.add_argument("--channel", type=int, required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--config", default="s640")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--profile")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args(argv)
    if a.mode == "cache":
        print(cache(a.channel, a.segment, a.config, a.stride))
    else:
        prof = load_profile(a.profile) if a.profile else PerceptionProfile()
        print(json.dumps(replay(det_path(a.channel, a.segment, a.config, a.stride), prof, a.channel, a.segment, a.split), indent=1))


if __name__ == "__main__":
    main()
