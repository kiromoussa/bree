"""ONNX runtime path: provider choice (no deps), plus PyTorch parity checks that need weights (marked `vision`).
All of it runs on CPU (CPUExecutionProvider); accelerator parity is scripts/onnx_parity.py."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from bree.edge.ort import onnx_path, ort_providers

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CPU = ["CPUExecutionProvider"]


def _names(ps):
    return [p[0] if isinstance(p, tuple) else p for p in ps]


def test_providers_best_first_and_cpu_last():
    every = ["CPUExecutionProvider", "CoreMLExecutionProvider", "AzureExecutionProvider",
             "CUDAExecutionProvider", "TensorrtExecutionProvider"]
    ps = ort_providers(every, cache_dir="/tmp/trt")
    assert _names(ps) == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider",
                          "CPUExecutionProvider"]
    assert ps[0][1]["trt_engine_cache_path"] == "/tmp/trt"
    assert ps[2][1]["ModelFormat"] == "MLProgram"
    assert _names(ort_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])) == \
        ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert ort_providers(["AzureExecutionProvider"]) == CPU   # nothing usable: still CPU
    assert _names(ort_providers(["TensorrtExecutionProvider", "CPUExecutionProvider"]))[0] == "TensorrtExecutionProvider"


def test_onnx_path_names_size():
    assert onnx_path("models/yolo26n-pose.pt", 160) == Path("models/yolo26n-pose_160.onnx")


def _frame():
    import cv2
    video = ROOT / "data" / "real" / "vtest.avi"
    if not video.exists():
        pytest.skip("data/real/vtest.avi missing (make data)")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ok, img = cap.read()
    assert ok
    return img


def _need(*names):
    for n in names:
        if not (MODELS / n).exists():
            pytest.skip(f"models/{n} missing (make setup)")


@pytest.mark.vision
def test_onnx_backend_matches_pytorch_on_real_frame():
    _need("yolo26n.pt", "yolo26n-pose.pt")
    from bree.detect.yolo import YoloBackend
    from bree.edge.ort import active_provider
    img = _frame()
    args = (str(MODELS / "yolo26n-pose.pt"), str(MODELS / "yolo26n.pt"), {"bottle": "drink"})
    pt, ox = YoloBackend(*args), YoloBackend(*args, runtime="onnx", providers=CPU)
    assert active_provider(ox.det) == "CPUExecutionProvider"
    # Detector on the same square letterbox as the static graph: runtime numerics only.
    a = pt.det.predict(img, imgsz=640, device="cpu", rect=False, verbose=False)[0].boxes
    b = ox.det.predict(img, imgsz=640, device="cpu", verbose=False)[0].boxes
    assert len(a) == len(b) >= 4 and (a.cls == b.cls).all()
    assert float((a.xyxy - b.xyxy).abs().max()) < 0.01
    # Full backend: same people; pose on the same boxes (several crops, so per-crop ONNX calls) agrees.
    persons_pt, _ = pt(img)
    persons_ox, _ = ox(img)
    assert len(persons_ox) == len(persons_pt)
    ka, kb = pt.pose(img, persons_pt.boxes), ox.pose(img, persons_pt.boxes)
    assert np.abs(ka - kb).max() < 0.01


@pytest.mark.vision
def test_ensure_onnx_exports_static_model_once(tmp_path):
    _need("yolo26n-pose.pt")
    from bree.edge.ort import ensure_onnx
    w = tmp_path / "yolo26n-pose.pt"
    shutil.copy(MODELS / "yolo26n-pose.pt", w)
    p = Path(ensure_onnx(w, 96))
    assert p == tmp_path / "yolo26n-pose_96.onnx" and p.exists()
    mtime = p.stat().st_mtime
    assert ensure_onnx(w, 96) == str(p) and p.stat().st_mtime == mtime   # reused, not re-exported
    import onnxruntime
    assert onnxruntime.InferenceSession(str(p), providers=CPU).get_inputs()[0].shape == [1, 3, 96, 96]


@pytest.mark.vision
def test_conceal_onnx_matches_pytorch_with_dynamic_batch(tmp_path):
    _need("conceal_poselift.pt")
    from bree.conceal import Track, load_bundle, model_track_scorer, onnx_track_scorer
    from bree.edge.export import export_conceal
    out = export_conceal(MODELS / "conceal_poselift.pt", tmp_path / "c.onnx")
    pt, ox = model_track_scorer(load_bundle(MODELS / "conceal_poselift.pt")), onnx_track_scorer(out)
    rng = np.random.default_rng(0)
    for T in (1, 7, 60):   # batch = number of windows = track length
        kps = np.concatenate([rng.uniform(100, 300, (T, 17, 2)), rng.uniform(0, 1, (T, 17, 1))], -1).astype(np.float32)
        t = Track(np.arange(T), kps, np.zeros((T, 4), np.float32))
        a, b = pt(t), ox(t)
        assert a.shape == b.shape == (T,)
        assert np.abs(a - b).max() < 1e-5
