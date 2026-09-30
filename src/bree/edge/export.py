"""Export the YOLO models to ONNX (portable to ONNX Runtime / TensorRT / OpenVINO / RKNN)
and compare CPU latency against PyTorch on a real frame.

TensorRT engines must be built on the target device itself (`yolo export format=engine`
on the Jetson), so here we only produce ONNX and note the command.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def _time(fn, n: int = 20) -> float:
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def export_and_benchmark(root: Path, imgsz: int = 640) -> dict:
    from ultralytics import YOLO
    models = root / "models"
    frame_path = root / "data" / "real" / "vtest.avi"
    cap = cv2.VideoCapture(str(frame_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ok, frame = cap.read()
    if not ok:
        frame = np.full((576, 768, 3), 128, np.uint8)
    crop = frame[100:300, 380:470] if ok else frame[:200, :90]

    out = {"frame": str(frame_path) if ok else "synthetic grey frame", "results": []}
    for name, size, img in (("yolo26n.pt", imgsz, frame), ("yolo26n-pose.pt", 160, crop)):
        pt = models / name
        onnx_path = pt.with_suffix(".onnx")
        exported = YOLO(str(pt)).export(format="onnx", imgsz=size, dynamic=False, simplify=True, verbose=False)
        if Path(exported) != onnx_path:
            shutil.move(exported, onnx_path)
        m_pt, m_onnx = YOLO(str(pt)), YOLO(str(onnx_path), task="pose" if "pose" in name else "detect")
        ms_pt = _time(lambda: m_pt.predict(img, imgsz=size, verbose=False, device="cpu"))
        ms_onnx = _time(lambda: m_onnx.predict(img, imgsz=size, verbose=False, device="cpu"))
        n_pt = len(m_pt.predict(img, imgsz=size, verbose=False, device="cpu")[0].boxes)
        n_onnx = len(m_onnx.predict(img, imgsz=size, verbose=False, device="cpu")[0].boxes)
        out["results"].append({"model": name, "imgsz": size, "onnx": str(onnx_path.relative_to(root)),
                               "onnx_mb": round(onnx_path.stat().st_size / 1e6, 2),
                               "cpu_ms_pytorch": round(ms_pt, 1), "cpu_ms_onnxruntime": round(ms_onnx, 1),
                               "detections_pytorch": n_pt, "detections_onnx": n_onnx})
    out["tensorrt_note"] = ("Build TensorRT engines on the target: `yolo export model=yolo26n.pt format=engine "
                            "half=True imgsz=640` on the Jetson itself.")
    return out
