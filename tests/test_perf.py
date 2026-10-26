"""Speed / heat profiles (bree.perf), GPU decode colour conversion, capped + pixelated annotated video,
detector overlap. Tiny synthetic frames; GPU-only checks skip without CUDA / ffmpeg."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from bree import perf
from bree.alerts.annotate import draw_frame, head_box
from bree.alerts.videofile import FfmpegWriter, open_video_writer
from bree.detect.toy import ToyBackend, synth_keypoints
from bree.events.observations import PersonObs
from bree.ingest.source import NvdecSource, VideoSource, nv12_to_bgr, open_source, prefetch

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


def test_profile_resolution_env_and_overrides(monkeypatch):
    monkeypatch.delenv(perf.ENV, raising=False)
    assert perf.resolve().name == "balanced"
    off, fast = perf.resolve("off"), perf.resolve("fast")
    assert not (off.threads or off.cuda_blocking_sync or off.half or off.overlap or off.nvdec or off.nvenc or off.video_width)
    assert fast.half and fast.overlap and fast.nvdec and fast.nvenc and fast.video_width == 1280
    bal = perf.resolve("balanced")      # the default must not change what is computed or written
    assert bal.threads and bal.cuda_blocking_sync and not (bal.half or bal.overlap or bal.nvdec or bal.nvenc or bal.video_width)
    monkeypatch.setenv(perf.ENV, "fast,video_width=0,half=0")
    p = perf.resolve()
    assert p.video_width == 0 and not p.half and p.overlap
    assert perf.resolve("off").name == "off"            # an explicit argument beats the env var
    for bad in ("turbo", "fast,bogus=1", "fast,video_width="):
        with pytest.raises(ValueError):
            perf.resolve(bad)


def test_half_kwargs_only_for_pytorch_cuda():
    assert perf.half_kwargs(False, "cuda:0") == {}
    assert perf.half_kwargs(True, "cpu") == {}
    assert perf.half_kwargs(True, "cuda:0", "onnx") == {}
    pytest.importorskip("ultralytics")
    assert list(perf.half_kwargs(True, "cuda:0")) in (["half"], ["quantize"])


def test_apply_runtime_caps_threads(monkeypatch):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    before_torch, before_cv = torch.get_num_threads(), cv2.getNumThreads()
    try:
        done = perf.apply_runtime(perf.PerfProfile(threads=2))
        assert done["threads"] == 2 and torch.get_num_threads() == 2 and cv2.getNumThreads() <= 2
    finally:
        torch.set_num_threads(before_torch)
        cv2.setNumThreads(before_cv)
    assert perf.apply_runtime(perf.PerfProfile(), torch_needed=False) == {}      # "off": touches nothing


def test_nv12_conversion_matches_opencv():
    rng = np.random.default_rng(1)
    h, w = 48, 64
    bgr = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    nv12 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)                # I420 -> NV12 by interleaving U and V
    y, u, v = nv12[:h], nv12[h:h + h // 4].reshape(-1), nv12[h + h // 4:].reshape(-1)
    flat = np.concatenate([y.reshape(-1), np.stack([u, v], 1).reshape(-1)])
    got = nv12_to_bgr(torch.from_numpy(flat), w, h).numpy()
    ref = cv2.cvtColor(flat.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    assert got.shape == (h, w, 3) and got.dtype == np.uint8
    assert np.abs(got.astype(int) - ref.astype(int)).max() <= 4


def test_draw_frame_scaled_keeps_heads_pixelated():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (400, 640, 3), dtype=np.uint8)           # noise: any unblurred patch is obvious
    box = (200, 40, 300, 360)
    p = PersonObs(1, box, 0.9, synth_keypoints(box, []))
    full = draw_frame(img, None, [p], [], [], blur=True)
    half = draw_frame(img, None, [p], [], [], blur=True, scale=0.5)
    assert full.shape == img.shape and half.shape == (200, 320, 3)
    for out, s in ((full, 1.0), (half, 0.5)):
        x1, y1, x2, y2 = (int(v * s) for v in head_box(p))
        roi = out[y1 + 4:y2, x1 + 4:x2 - 4].reshape(-1, 3)              # below the box edge line and its label
        assert len(np.unique(roi, axis=0)) <= 25                        # 5x5 pixelation: at most 25 colours
    assert draw_frame(img, None, [p], [], [], scale=1.0).shape == img.shape


def test_prefetch_keeps_order_and_reraises():
    assert list(prefetch(iter(range(50)), depth=2)) == list(range(50))

    def boom():
        yield 1
        raise RuntimeError("decode failed")
    it = prefetch(boom())
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="decode failed"):
        next(it)


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@needs_ffmpeg
def test_ffmpeg_writer_round_trip(tmp_path):
    path = str(tmp_path / "a.mp4")
    wr = FfmpegWriter(path, 10.0, (64, 48), codec="mpeg4")             # software codec: no GPU needed
    for i in range(12):
        wr.write(np.full((48, 64, 3), 20 * i, np.uint8))
    wr.release()
    cap = cv2.VideoCapture(path)
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12


def test_open_video_writer_falls_back_to_opencv(tmp_path, monkeypatch):
    monkeypatch.setattr("bree.alerts.videofile.ffmpeg_encoder_works", lambda name: False)
    wr = open_video_writer(str(tmp_path / "b.mp4"), 10.0, (64, 48), nvenc=True)
    assert isinstance(wr, cv2.VideoWriter)
    wr.release()


def _h264_clip(path: Path, n: int = 12) -> bool:
    w = cv2.VideoWriter(str(path.with_suffix(".src.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8), (0, 0), 6)
    for i in range(n):
        w.write(np.roll(base, i * 3, axis=1))
    w.release()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path.with_suffix(".src.mp4")), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(path)], capture_output=True)
    return r.returncode == 0


@needs_ffmpeg
def test_open_source_default_and_fallbacks(tmp_path):
    clip = tmp_path / "c.mp4"
    if not _h264_clip(clip):
        pytest.skip("ffmpeg has no libx264")
    assert isinstance(open_source(str(clip)), VideoSource)               # nvdec off: the old path
    src = open_source(str(clip), max_frames=8, nvdec=True)
    assert isinstance(src, NvdecSource if torch.cuda.is_available() else VideoSource)


@needs_ffmpeg
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvdec_frames_match_opencv(tmp_path):
    clip = tmp_path / "c.mp4"
    if not _h264_clip(clip):
        pytest.skip("ffmpeg has no libx264")
    ref = list(VideoSource(str(clip), max_frames=10))
    src = open_source(str(clip), max_frames=10, nvdec=True)
    got = list(src)
    if getattr(src, "fallback", False):
        pytest.skip("NVDEC refused this stream on this machine (fell back to OpenCV, as designed)")
    assert len(got) == len(ref) == 10 and src.fps == pytest.approx(10.0, abs=0.1)
    for a, b in zip(got, ref):
        assert a.image.shape == b.image.shape and a.index == b.index and a.t == pytest.approx(b.t)
        assert np.abs(a.image.astype(int) - b.image.astype(int)).mean() < 3.0


class _OverlapToy(ToyBackend):
    overlap_ok = True


def _toy_run(tmp_path, name, perf_spec, backend_cls):
    from bree.events.zones import load_store_config
    from bree.pipeline import run_pipeline
    from bree.sim.toy_render import render, scenarios
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    clip = tmp_path / "toy_walkout.mp4"
    if not clip.exists():
        render(next(s for s in scenarios() if s.name == "walkout"), store, tmp_path, seed=0)
    out = tmp_path / name
    s = run_pipeline(str(clip), store, backend_cls(), out, verbose=False, perf=perf_spec)
    return s, out


def test_overlap_and_balanced_leave_results_unchanged(tmp_path):
    base, bout = _toy_run(tmp_path, "off", "off", ToyBackend)
    for name, spec, cls in (("bal", "balanced", ToyBackend),
                            ("ovl", "off,overlap=1", _OverlapToy)):
        s, out = _toy_run(tmp_path, name, spec, cls)
        assert (out / "frames.jsonl").read_text() == (bout / "frames.jsonl").read_text()
        assert (out / "events.jsonl").read_text() == (bout / "events.jsonl").read_text()
        assert s.frames == base.frames and s.events == base.events and len(s.alerts) == len(base.alerts)


def test_capped_video_width_and_alert_clip(tmp_path):
    s, out = _toy_run(tmp_path, "cap", "off,video_width=320", ToyBackend)
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == s.frames
    assert len(s.alerts) == 1 and Path(s.alerts[0]["clip_path"]).exists()
