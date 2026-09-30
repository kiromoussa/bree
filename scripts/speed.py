"""Full-pipeline speed (detector + top-down pose + ByteTrack + events) per runtime on this machine.
Run with nothing else busy. Writes results/speed.json.
Usage: python scripts/speed.py [video] [frames=300]
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
from bree.events.engine import EventEngine
from bree.events.observations import FrameObs
from bree.events.zones import load_store_config
from bree.track.bytetrack import Tracker

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "data/merl/Videos_MERL_Shopping_Dataset/27_1_crop.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
store = load_store_config("configs/store_gas_station_small.yaml")
M = Path("models")


def onnx(name: str, size: int) -> str:
    dst = M / f"{Path(name).stem}_{size}.onnx"
    if not dst.exists():
        from ultralytics import YOLO
        Path(YOLO(str(M / name)).export(format="onnx", imgsz=size, simplify=True, verbose=False)).rename(dst)
    return str(dst)


def frames():
    cap = cv2.VideoCapture(VIDEO)
    out = []
    while len(out) < N:
        ok, im = cap.read()
        if not ok:
            break
        out.append(im)
    return out


def run(label, det, pose, device):
    b = YoloBackend(pose, det, store.product_classes, device=device)
    imgs = frames()
    tracker, engine = Tracker(15), EventEngine(store)
    for im in imgs[:10]:   # warm-up
        b(im)
    t = time.perf_counter()
    for i, im in enumerate(imgs):
        persons, products = tracker.update(*b(im), im.shape[:2])
        engine.update(FrameObs(i, i / 15, persons, products))
    dt = (time.perf_counter() - t) / len(imgs)
    r = {"runtime": label, "detect_model": Path(det).name, "pose_model": Path(pose).name, "device": device,
         "fps": 1 / dt, "ms_per_frame": 1000 * dt, "frames": len(imgs)}
    print(r, flush=True)
    return r


chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
    if platform.system() == "Darwin" else platform.processor()
runs = []
for size in ("n", "s"):
    det, pose = str(M / f"yolo26{size}.pt"), str(M / f"yolo26{size}-pose.pt")
    runs.append(run("pytorch cpu", det, pose, "cpu"))
    if torch.backends.mps.is_available():
        runs.append(run("pytorch mps", det, pose, "mps"))
    if torch.cuda.is_available():
        runs.append(run("pytorch cuda", det, pose, "cuda:0"))
    runs.append(run("onnxruntime cpu", onnx(f"yolo26{size}.pt", 640), onnx(f"yolo26{size}-pose.pt", 160), "cpu"))
res = {"machine": f"{chip} ({platform.platform()})", "video": VIDEO, "frames": N, "runs": runs,
       "note": "TensorRT needs an NVIDIA GPU (A100 / Jetson); not measurable on this machine."}
Path("results/speed.json").write_text(json.dumps(res, indent=1))
