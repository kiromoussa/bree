"""Speed / heat profiles: how hard one `bree run` leans on the CPU, and what moves to the GPU.

Choose with `bree run --perf NAME` or the BREE_PERF env var (the flag wins). A profile name can be followed by
`key=value` overrides: BREE_PERF=fast,video_width=0,half=0

  off       exactly the old behaviour: no thread caps, spinning CUDA sync, fp32, full-res mp4v, sequential
  balanced  (default) same outputs as `off`, far less CPU: thread caps and blocking CUDA sync
  fast      balanced + fp16 models, detector overlapped with the rest of the frame, NVDEC decode and
            h264_nvenc encode when this machine has them (clean fallback to the CPU paths), annotated.mp4
            capped to 1280 px wide

Measured numbers: docs/ultra/perf.md. Nothing here is a hard dependency: no CUDA, no ffmpeg, no NVENC -> the
plain OpenCV path runs.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace, fields

ENV = "BREE_PERF"
DEFAULT_PROFILE = "balanced"


@dataclass(frozen=True)
class PerfProfile:
    name: str = DEFAULT_PROFILE
    threads: int = 0              # 0 = leave library defaults; else cap torch / OpenCV / OpenMP / BLAS threads
    cuda_blocking_sync: bool = False   # sleep (not spin) while waiting on the GPU: same speed, ~1/3 of the CPU
    half: bool = False            # fp16 inference for the CUDA detector and pose (detections within tolerance)
    overlap: bool = False         # run detector for frame N+1 while the CPU finishes frame N
    nvdec: bool = False           # decode files on the GPU through an ffmpeg pipe (falls back to OpenCV)
    nvenc: bool = False           # encode annotated.mp4 with h264_nvenc via ffmpeg (falls back to OpenCV mp4v)
    video_width: int = 0          # cap annotated.mp4 width (0 = native); heads are pixelated at the final size


PROFILES = {
    "off": PerfProfile("off"),
    "balanced": PerfProfile("balanced", threads=4, cuda_blocking_sync=True),
    "fast": PerfProfile("fast", threads=4, cuda_blocking_sync=True, half=True, overlap=True, nvdec=True,
                        nvenc=True, video_width=1280),
}


def _coerce(field_type: str, v: str):
    if field_type == "bool":
        return v.strip().lower() in ("1", "true", "yes", "on")
    return int(v)


def resolve(spec: str | None = None) -> PerfProfile:
    """`spec` (CLI flag) or $BREE_PERF -> profile. "name" or "name,key=value,..."; unknown names raise ValueError."""
    spec = spec if spec is not None else os.environ.get(ENV) or DEFAULT_PROFILE
    name, *over = [s.strip() for s in spec.split(",") if s.strip()] or [DEFAULT_PROFILE]
    if name not in PROFILES:
        raise ValueError(f"unknown perf profile {name!r} (choose from {', '.join(PROFILES)})")
    types = {f.name: str(f.type) for f in fields(PerfProfile)}
    kw = {}
    for o in over:
        k, _, v = o.partition("=")
        if k not in types or k == "name" or not v:
            raise ValueError(f"bad perf override {o!r}: use one of {[t for t in types if t != 'name']} as key=value")
        kw[k] = _coerce(types[k], v)
    return replace(PROFILES[name], **kw)


def _cudart():
    """The libcudart that PyTorch already loaded (so the setting reaches the context PyTorch creates)."""
    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "libcudart.so" in line}
    except OSError:
        return None
    for p in sorted(paths):
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    return None


def set_cuda_blocking_sync() -> bool:
    """cudaDeviceScheduleBlockingSync: the host thread sleeps while the GPU works instead of spinning a core at
    100%. Must run before the first CUDA call of the process (a no-op, returns False, once a context exists)."""
    if "torch" not in sys.modules or not sys.platform.startswith("linux"):
        return False
    import torch
    if not torch.cuda.is_available() or torch.cuda.is_initialized():
        return False
    lib = _cudart()
    return bool(lib is not None and lib.cudaSetDeviceFlags(0x04) == 0)


def apply_runtime(p: PerfProfile, torch_needed: bool = True) -> dict:
    """Thread caps + CUDA sync mode for this process. Call once, early (before the models load). Returns what it did."""
    done: dict = {}
    if p.threads > 0:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(k, str(p.threads))     # honoured by libraries imported after this point
        import cv2
        cv2.setNumThreads(p.threads)
        done["threads"] = p.threads
    if torch_needed:
        import torch
        if p.threads > 0:
            torch.set_num_threads(p.threads)
        if p.cuda_blocking_sync:
            done["cuda_blocking_sync"] = set_cuda_blocking_sync()
    return done


_encoder_ok: dict[str, bool] = {}


def ffmpeg_encoder_works(name: str = "h264_nvenc") -> bool:
    """True when ffmpeg is installed and really encodes with `name` here (listing is not enough: no GPU -> fails)."""
    if name not in _encoder_ok:
        ok = False
        if shutil.which("ffmpeg"):
            try:
                r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:d=0.2",
                                    "-c:v", name, "-f", "null", "-"], capture_output=True, timeout=20)
                ok = r.returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        _encoder_ok[name] = ok
    return _encoder_ok[name]


def half_kwargs(half: bool, device: str, runtime: str = "pytorch") -> dict:
    """Ultralytics predict() kwargs for fp16: only PyTorch weights on a CUDA device (ONNX / CPU ignore or break it).
    Newer Ultralytics renamed `half` to `quantize=16`."""
    if not half or runtime != "pytorch" or not str(device).startswith("cuda"):
        return {}
    from ultralytics.cfg import DEFAULT_CFG_DICT
    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
