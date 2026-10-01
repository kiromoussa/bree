"""Run the YOLO models from ONNX with ONNX Runtime on the best execution provider of this machine.

Ultralytics' ONNX backend only picks CUDA (device=cuda) or CoreML (device=mps), never TensorRT, and with
an accelerator device it also moves pre/post-processing there. We keep Ultralytics for letterboxing and
decoding (device "cpu", numpy in and out) and swap its ONNX Runtime session for one with our providers.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

_export_lock = threading.Lock()   # shadow mode loads one backend per camera thread

PREFERRED = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")


def ort_providers(available: list[str] | None = None, cache_dir: str | Path | None = None) -> list:
    """Best-first provider list for onnxruntime.InferenceSession: TensorRT > CUDA > CoreML > CPU, of those present.
    TensorRT engines are cached in cache_dir (a build takes minutes on a Jetson); delete it after changing weights."""
    if available is None:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    out = []
    for p in PREFERRED:
        if p not in available:
            continue
        if p == "TensorrtExecutionProvider" and cache_dir:
            p = (p, {"trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_dir)})
        elif p == "CoreMLExecutionProvider":
            # MLProgram runs the whole YOLO graph in CoreML; the older NeuralNetwork format splits it 7 ways.
            p = (p, {"ModelFormat": "MLProgram"})
        out.append(p)
    return out or ["CPUExecutionProvider"]


def onnx_path(weights: str | Path, imgsz: int) -> Path:
    w = Path(weights)
    return w.with_name(f"{w.stem}_{imgsz}.onnx")


def ensure_onnx(weights: str | Path, imgsz: int) -> str:
    """models/yolo26n.pt + 640 -> models/yolo26n_640.onnx (static shape, batch 1), exported once if missing."""
    dst = onnx_path(weights, imgsz)
    with _export_lock:
        if not dst.exists():
            from ultralytics import YOLO
            Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, simplify=True, verbose=False)).rename(dst)
    return str(dst)


def load_yolo(weights: str | Path, task: str, imgsz: int, runtime: str = "pytorch", providers: list | None = None):
    """An Ultralytics YOLO model. runtime="onnx": the static ONNX export at imgsz, run by ONNX Runtime with
    `providers` (default ort_providers()). Always call .predict(..., device="cpu") on an ONNX model: another
    device makes Ultralytics rebuild the predictor and drop the swapped session."""
    from ultralytics import YOLO
    if runtime == "pytorch":
        return YOLO(str(weights), task=task)
    if runtime != "onnx":
        raise ValueError(f"unknown runtime {runtime!r} (pytorch | onnx)")
    import onnxruntime
    path = ensure_onnx(weights, imgsz)
    m = YOLO(path, task=task)
    m.predict(np.zeros((imgsz, imgsz, 3), np.uint8), device="cpu", verbose=False)   # builds the predictor
    backend = m.predictor.model.backend
    backend.session = onnxruntime.InferenceSession(
        path, providers=providers or ort_providers(cache_dir=Path(path).parent / "ort_cache"))
    from ultralytics.utils import LOGGER
    LOGGER.info(f"bree: {Path(path).name} switched to {backend.session.get_providers()}")   # Ultralytics logs CPU above
    return m


def active_provider(model) -> str:
    """First provider the session actually got (ONNX Runtime silently drops ones that fail to load)."""
    return model.predictor.model.backend.session.get_providers()[0]
