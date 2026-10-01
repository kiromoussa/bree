"""Full-pipeline speed (detector + top-down pose + ByteTrack + events) per runtime on this machine.
Run with nothing else busy. Writes results/speed.json.
Usage: python scripts/speed.py [video] [frames=300] [out=results/speed.json]
ONNX Runtime rows: CPU, plus one per accelerator provider present (TensorRT, CUDA, CoreML), each falling back down
the chain TensorRT > CUDA > CoreML > CPU for any node it can't run (bree.edge.ort).
"""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch

from bree.detect.yolo import YoloBackend
from bree.edge.ort import active_provider, ort_providers
from bree.events.engine import EventEngine
from bree.events.observations import FrameObs
from bree.events.zones import load_store_config
from bree.track.bytetrack import Tracker

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/merl/Videos_MERL_Shopping_Dataset/27_1_crop.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/speed.json")
store = load_store_config("configs/store_gas_station_small.yaml")
M = Path("models")


def frames():
    cap = cv2.VideoCapture(VIDEO)
    out = []
    while len(out) < N:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    return out


def run(label, det, pose, device, providers=None):
    runtime = "onnx" if providers else "pytorch"
    b = YoloBackend(pose, det, store.product_classes, device=device, runtime=runtime, providers=providers)
    imgs = frames()
    tracker, engine = Tracker(15), EventEngine(store)
    for im in imgs[:10]:   # warm-up (also builds TensorRT engines / CoreML models on first use)
        b(im)
    t = time.perf_counter()
    for i, im in enumerate(imgs):
        persons, products = tracker.update(*b(im), im.shape[:2])
        engine.update(FrameObs(i, i / 15, persons, products))
    dt = (time.perf_counter() - t) / len(imgs)
    r = {"runtime": label, "detect_model": Path(det).name, "pose_model": Path(pose).name, "device": device,
         "fps": 1 / dt, "ms_per_frame": 1000 * dt, "frames": len(imgs)}
    if providers:   # ONNX Runtime silently drops providers that fail to load: record what actually ran
        r["device"] = active_provider(b.det)
    print(r, flush=True)
    return r


chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
    if platform.system() == "Darwin" else platform.processor()
chain = ort_providers(cache_dir=M / "ort_cache")
runs = []
for size in ("n", "s"):
    det, pose = str(M / f"yolo26{size}.pt"), str(M / f"yolo26{size}-pose.pt")
    runs.append(run("pytorch cpu", det, pose, "cpu"))
    if torch.backends.mps.is_available():
        runs.append(run("pytorch mps", det, pose, "mps"))
    if torch.cuda.is_available():
        runs.append(run("pytorch cuda", det, pose, "cuda:0"))
    runs.append(run("onnxruntime cpu", det, pose, "cpu", ["CPUExecutionProvider"]))
    for i, p in enumerate(chain[:-1]):   # each accelerator, with the rest of the chain as fallback
        name = (p[0] if isinstance(p, tuple) else p).replace("ExecutionProvider", "").lower()
        runs.append(run(f"onnxruntime {name}", det, pose, "cpu", chain[i:]))
res = {"machine": f"{chip} ({platform.platform()})", "video": VIDEO, "frames": N, "runs": runs,
       "note": "ONNX rows run static exports (detector 640x640, pose 160x160 batch 1, one call per person crop) via "
               "bree.edge.ort; TensorRT/CUDA rows appear only with onnxruntime-gpu on an NVIDIA machine."}
OUT.write_text(json.dumps(res, indent=1))
