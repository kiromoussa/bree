"""ONNX vs PyTorch parity on real data -> results/onnx_parity.json.

YOLO: detector boxes on a few real frames (people matched by IoU), and pose keypoints on the SAME person boxes
(so pose differences are not detector differences), for every ONNX Runtime provider available here.
Concealment classifier: exports models/conceal_poselift.onnx and compares scores on PoseLift test tracks.
Not a speed test (scripts/speed.py is).
Usage: PYTHONPATH=src python scripts/onnx_parity.py [--frames 4] [--size s] [--coreml-units ALL] [--out PATH]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bree.detect.yolo import YoloBackend
from bree.edge.ort import active_provider, ort_providers
from bree.events.zones import load_store_config

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=4, help="frames per video")
ap.add_argument("--size", default="s", help="model size: n | s | m")
ap.add_argument("--coreml-units", default="ALL", help="CoreML MLComputeUnits: ALL | CPUAndNeuralEngine | CPUOnly ...")
ap.add_argument("--videos", type=int, default=6, help="PoseLift test clips for the classifier check")
ap.add_argument("--out", default="results/onnx_parity.json")
args = ap.parse_args()

M = Path("models")
VIDEOS = ["data/real/vtest.avi", "data/merl/Videos_MERL_Shopping_Dataset/27_1_crop.mp4"]
store = load_store_config("configs/store_gas_station_small.yaml")


def read_frames(path, n):
    cap, out = cv2.VideoCapture(path), []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    for idx in np.linspace(30, max(total - 30, 31), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if ok:
            out.append(im)
    return out


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def compare(ref: YoloBackend, other: YoloBackend, frames) -> dict:
    box_d, kp_d, conf_d, unmatched, n_ref, kp_n = [], [], [], 0, 0, 0
    for im in frames:
        (pa, _), (pb, _) = ref(im), other(im)
        n_ref += len(pa)
        if len(pa) and len(pb):
            m = iou(pa.boxes, pb.boxes)
            j = m.argmax(1)
            ok = m[np.arange(len(pa)), j] > 0.5
            unmatched += int((~ok).sum()) + max(len(pb) - int(ok.sum()), 0)
            box_d += np.abs(pa.boxes[ok] - pb.boxes[j[ok]]).max(1).tolist()
            conf_d += np.abs(pa.confs[ok] - pb.confs[j[ok]]).tolist()
        else:
            unmatched += len(pa) + len(pb)
        if len(pa):   # pose on the reference boxes for both runtimes
            ka, kb = ref.pose(im, pa.boxes), other.pose(im, pa.boxes)
            vis = (ka[..., 2] > 0.3) & (kb[..., 2] > 0.3)
            kp_n += int(vis.sum())
            if vis.any():
                kp_d += np.linalg.norm(ka[..., :2] - kb[..., :2], axis=-1)[vis].tolist()
                conf_d += np.abs(ka[..., 2] - kb[..., 2])[vis].tolist()
    pct = lambda x, q: float(f"{np.percentile(x, q):.3g}") if x else None
    return {"people_pytorch": n_ref, "unmatched_people": unmatched,
            "box_max_px": pct(box_d, 100), "box_median_px": pct(box_d, 50),
            "keypoints_compared": kp_n, "kpt_max_px": pct(kp_d, 100), "kpt_p99_px": pct(kp_d, 99),
            "kpt_median_px": pct(kp_d, 50), "conf_max_abs": pct(conf_d, 100),
            "det_same_letterbox": same_letterbox(ref, other, frames)}


def same_letterbox(ref: YoloBackend, other: YoloBackend, frames) -> dict:
    """Detector alone with PyTorch fed the same square 640x640 letterbox as the static ONNX graph (rect=False):
    separates runtime numerics from the padding difference (PyTorch pads to the frame's aspect, e.g. 640x480)."""
    box, conf, n_diff = [0.0], [0.0], 0
    for im in frames:
        a = ref.det.predict(im, imgsz=ref.imgsz, device="cpu", conf=0.25, rect=False, verbose=False)[0].boxes
        b = other.det.predict(im, imgsz=other.imgsz, device="cpu", conf=0.25, verbose=False)[0].boxes
        if len(a) != len(b) or (a.cls != b.cls).any():
            n_diff += 1
            continue
        if len(a):
            box.append(float((a.xyxy - b.xyxy).abs().max()))
            conf.append(float((a.conf - b.conf).abs().max()))
    return {"frames_with_different_detections": n_diff, "box_max_px": float(f"{max(box):.3g}"),
            "conf_max_abs": float(f"{max(conf):.3g}")}


frames = {v: read_frames(v, args.frames) for v in VIDEOS if Path(v).exists()}
det, pose = str(M / f"yolo26{args.size}.pt"), str(M / f"yolo26{args.size}-pose.pt")
ref = YoloBackend(pose, det, store.product_classes, device="cpu")
variants = {"onnxruntime cpu": ["CPUExecutionProvider"]}
for p in ort_providers():
    name = p[0] if isinstance(p, tuple) else p
    if name == "CoreMLExecutionProvider":
        p = (name, {**p[1], "MLComputeUnits": args.coreml_units})
    if name != "CPUExecutionProvider":
        variants[f"onnxruntime {name.replace('ExecutionProvider', '').lower()}"] = [p, "CPUExecutionProvider"]

yolo = []
for label, providers in variants.items():
    b = YoloBackend(pose, det, store.product_classes, runtime="onnx", providers=providers)
    got = active_provider(b.det)
    for v, fr in frames.items():
        r = {"runtime": label, "provider_active": got, "video": v, "frames": len(fr), **compare(ref, b, fr)}
        print(r, flush=True)
        yolo.append(r)

# Concealment classifier
from bree.conceal import load_bundle, load_poselift, model_track_scorer, onnx_track_scorer   # noqa: E402
from bree.edge.export import export_conceal   # noqa: E402
onnx_path = export_conceal(M / "conceal_poselift.pt")
pt_score, ox_score = model_track_scorer(load_bundle(M / "conceal_poselift.pt")), onnx_track_scorer(onnx_path)
vids = [v for v in load_poselift(Path("data/poselift/PoseLift/Pickle_files")) if v.split == "test"][:args.videos]
d, n = [], 0
for v in vids:
    for t in v.tracks.values():
        a, b = pt_score(t), ox_score(t)
        d.append(float(np.abs(a - b).max()))
        n += len(a)
conceal = {"onnx": str(onnx_path), "poselift_test_clips": len(vids), "tracks": len(d), "windows": n,
           "score_max_abs_diff": max(d), "note": "dynamic batch; mu/sd normalisation + sigmoid baked into the graph"}
print(conceal)
Path(args.out).write_text(json.dumps({"models": f"yolo26{args.size}", "reference": "pytorch cpu",
                                      "coreml_units": args.coreml_units, "yolo": yolo, "conceal": conceal}, indent=1))
