"""Person detection + tracking on MOT16 train sequences (real footage with identity ground truth; research licence,
EVALUATION ONLY). Simplified MOT protocol: ground truth = pedestrians (class 1) marked for evaluation; IoU 0.5
matching via motmetrics. Every frame, native fps. Writes results/mot16.json.
Settings (fixed before running, not tuned here):
  A  default before overnight: YOLO26s detector, person conf 0.3, ByteTrack buffer 2 s
  B  A + drop person boxes >= 85% inside a larger one (current default)
  C  overnight MERL-tuned: YOLO26n, conf 0.15, buffer 5 s, new-track threshold 0.15, + B's duplicate removal
"""
import json
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

from bree.detect.yolo import YoloBackend
from bree.hw import detect_hardware
from bree.events.engine import EngineRules, EventEngine
from bree.events.observations import FrameObs
from bree.events.zones import load_store_config
from bree.track.bytetrack import Tracker

ROOT = Path("data/mot16/train")


def iou_dist(a: np.ndarray, b: np.ndarray, max_iou: float = 0.5) -> np.ndarray:
    """1 - IoU for xywh boxes, NaN where IoU < max_iou (motmetrics' iou_matrix uses np.asfarray, gone in NumPy 2)."""
    if len(a) == 0 or len(b) == 0:
        return np.full((len(a), len(b)), np.nan)
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None])).clip(0)
    ih = (np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None])).clip(0)
    inter = iw * ih
    iou = inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)
    d = 1 - iou
    d[iou < max_iou] = np.nan
    return d
SETTINGS = {"A": dict(det="yolo26s.pt", pose="yolo26s-pose.pt", conf=0.3, buffer=2.0, ntt=None, dedupe=0.0),
            "B": dict(det="yolo26s.pt", pose="yolo26s-pose.pt", conf=0.3, buffer=2.0, ntt=None, dedupe=0.85),
            "C": dict(det="yolo26n.pt", pose="yolo26n-pose.pt", conf=0.15, buffer=5.0, ntt=0.15, dedupe=0.85),
            # B scored on the event engine's person ids (track stitching, default 10 s / 1 body height; no door
            # zone here, so any new track may be stitched): do position-only merges confuse different people?
            "B_stitch": dict(det="yolo26s.pt", pose="yolo26s-pose.pt", conf=0.3, buffer=2.0, ntt=None, dedupe=0.85,
                             stitch=True, guard=False),
            # EXPLORATORY (added after seeing B_stitch): stricter stitching distance.
            "B_stitch_d05": dict(det="yolo26s.pt", pose="yolo26s-pose.pt", conf=0.3, buffer=2.0, ntt=None, dedupe=0.85,
                                 stitch=True, stitch_dist=0.5, guard=False),
            # Stitching with the ambiguity guard (engine default from 2026-10-01 ~03:50 EDT); B_stitch above ran
            # before the guard existed.
            "B_stitch_guard": dict(det="yolo26s.pt", pose="yolo26s-pose.pt", conf=0.3, buffer=2.0, ntt=None,
                                   dedupe=0.85, stitch=True, guard=True)}
import sys
ONLY = sys.argv[1:] or list(SETTINGS)
hw = detect_hardware()
res = {"dataset": "MOT16 train (motchallenge.net), evaluation only", "device": hw.device, "settings": SETTINGS, "results": {}}
seqs = sorted(p for p in ROOT.iterdir() if (p / "gt/gt.txt").exists())
prev = json.loads(Path("results/mot16.json").read_text()) if Path("results/mot16.json").exists() else None
if prev:
    res["results"].update(prev["results"])
for name, cfg in [(n, SETTINGS[n]) for n in ONLY]:
    backend = YoloBackend(f"models/{cfg['pose']}", f"models/{cfg['det']}", {}, device=hw.device, imgsz=hw.imgsz,
                          person_conf=cfg["conf"], products=False, dedupe_inside=cfg["dedupe"])
    accs = []
    for seq in seqs:
        info = dict(l.strip().split("=") for l in (seq / "seqinfo.ini").read_text().splitlines() if "=" in l)
        fps = float(info["frameRate"])
        gt = np.loadtxt(seq / "gt/gt.txt", delimiter=",")
        gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
        tracker = Tracker(fps, person_buffer_s=cfg["buffer"],
                          bytetrack={"new_track_thresh": cfg["ntt"]} if cfg["ntt"] is not None else None)
        acc = mm.MOTAccumulator(auto_id=True)
        st = load_store_config("configs/no_zones.yaml")
        eng = EventEngine(st, EngineRules.from_dict({**st.rules, "stitch_dist": cfg.get("stitch_dist", 1.0),
                                                   "stitch_ambiguous_skip": cfg.get("guard", True)})) \
            if cfg.get("stitch") else None
        for fi, img in enumerate(sorted((seq / "img1").glob("*.jpg")), start=1):
            im = cv2.imread(str(img))
            persons, _ = tracker.update(*backend(im), im.shape[:2])
            if eng is not None:
                eng.update(FrameObs(fi, fi / fps, persons, []))
            g = gt[gt[:, 0] == fi]
            gb = g[:, 2:6]
            hb = np.array([[p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]] for p in persons]).reshape(-1, 4)
            ids = [eng.alias.get(p.track_id, p.track_id) for p in persons] if eng is not None else [p.track_id for p in persons]
            acc.update(g[:, 1].astype(int).tolist(), ids,
                       iou_dist(gb, hb))
        accs.append(acc)
        print(name, seq.name, flush=True)
    mh = mm.metrics.create()
    summ = mh.compute_many(accs, names=[s.name for s in seqs], generate_overall=True,
                           metrics=["mota", "idf1", "num_switches", "num_fragmentations", "recall", "precision",
                                    "mostly_tracked", "num_unique_objects"])
    res["results"][name] = {seq: {k: float(v) for k, v in row.items()} for seq, row in summ.to_dict("index").items()}
    o = res["results"][name]["OVERALL"]
    print(name, {k: round(v, 3) for k, v in o.items()}, flush=True)
Path("results/mot16.json").write_text(json.dumps(res, indent=1))
