"""Video input: files, webcams, and RTSP/HTTP streams, as one iterator of frames."""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    t: float              # seconds since stream start (file: frame index / fps; live: wall clock)
    image: np.ndarray     # BGR
    wall: float           # time.time() when the frame was read (latency measurement)
    camera: str | None = None   # set by the pipeline when several cameras feed one store


class VideoSource:
    """`src` is a file path, a webcam index ("0"), or an rtsp:// / http(s):// URL."""

    def __init__(self, src: str | int, max_frames: int | None = None, reconnect: bool = True):
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        is_url = isinstance(self.src, str) and self.src.startswith(("rtsp://", "http://", "https://"))
        self.live = isinstance(self.src, int) or is_url
        self.max_frames = max_frames
        self.reconnect = reconnect and self.live
        self.cap = self._open()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 and fps < 240 else 15.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise IOError(f"cannot open video source {self.src!r}")
        return cap

    def __iter__(self) -> Iterator[Frame]:
        i, t0, failures = 0, time.time(), 0
        while self.max_frames is None or i < self.max_frames:
            ok, img = self.cap.read()
            if not ok:
                if not self.reconnect or failures >= 5:
                    break
                failures += 1          # RTSP hiccup: back off and reopen
                time.sleep(min(2 ** failures, 10))
                self.cap.release()
                self.cap = self._open()
                continue
            failures = 0
            now = time.time()
            t = (now - t0) if self.live else i / self.fps
            yield Frame(i, t, img, now)
            i += 1
        self.cap.release()


def nv12_to_bgr(nv12, width: int, height: int, full_range: bool = False):
    """NV12 bytes (uint8 torch tensor, 1.5 bytes per pixel, any device) -> BGR uint8 (height, width, 3) tensor.
    BT.601, the matrix OpenCV / swscale use for untagged video (what VideoSource gives today); within +-3 levels of
    cv2.cvtColor(COLOR_YUV2BGR_NV12) (chroma upsampling differs by a hair)."""
    import torch
    y = nv12[:width * height].view(height, width).float()
    uv = nv12[width * height:width * height * 3 // 2].view(height // 2, width // 2, 2).float()
    u = uv[..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128
    v = uv[..., 1].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128
    luma, chroma = (1.0, 1.0) if full_range else (255 / 219, 255 / 224)
    yy = (y - (0 if full_range else 16)) * luma
    kr, kb = 0.299, 0.114
    rv, bu = chroma * 2 * (1 - kr) * v, chroma * 2 * (1 - kb) * u
    g = yy - (kr * rv + kb * bu) / (1 - kr - kb)
    return torch.stack([yy + bu, g, yy + rv], -1).round_().clamp_(0, 255).to(torch.uint8)


def _probe(path: str) -> dict | None:
    """ffprobe the first video stream: codec, size, fps, pixel format, range. None if ffprobe is missing or fails."""
    if not shutil.which("ffprobe"):
        return None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height,avg_frame_rate,pix_fmt,color_range", "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        st = json.loads(r.stdout)["streams"][0]
        num, _, den = st["avg_frame_rate"].partition("/")
        st["fps"] = float(num) / float(den or 1)
        return st
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None


class NvdecSource:
    """A video FILE decoded on the GPU (NVDEC, through an ffmpeg pipe), colour-converted on the GPU, handed over as
    the same BGR frames as VideoSource. Same attributes (`fps`, `width`, `height`, `live`) and Frame timing as
    VideoSource. Use `open_source(..., nvdec=True)`: it falls back to VideoSource when this cannot run."""

    live = False

    def __init__(self, path: str, info: dict, max_frames: int | None = None):
        self.src, self.max_frames = path, max_frames
        self.width, self.height = int(info["width"]), int(info["height"])
        fps = info["fps"]
        self.fps = fps if fps and 0 < fps < 240 else 15.0
        self.full_range = info.get("pix_fmt") == "yuvj420p" or info.get("color_range") == "pc"
        self.fallback = False      # set when the ffmpeg pipe yields nothing: __iter__ then reads via VideoSource

    def _cmd(self) -> list[str]:
        cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(self.src)]
        if self.max_frames is not None:
            cmd += ["-frames:v", str(self.max_frames)]
        return cmd + ["-vf", "hwdownload,format=nv12", "-f", "rawvideo", "-"]

    def __iter__(self) -> Iterator[Frame]:
        import torch
        w, h = self.width, self.height
        nbytes = w * h * 3 // 2
        pin = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
        view = memoryview(pin.numpy())
        proc = subprocess.Popen(self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 2)
        i = 0
        try:
            while self.max_frames is None or i < self.max_frames:
                got = 0
                while got < nbytes:        # a pipe read may return less than a frame
                    n = proc.stdout.readinto(view[got:])
                    if not n:
                        break
                    got += n
                if got < nbytes:
                    break
                bgr = nv12_to_bgr(pin.cuda(non_blocking=True), w, h, self.full_range).cpu().numpy()
                yield Frame(i, i / self.fps, bgr, time.time())
                i += 1
        finally:
            proc.kill()
            proc.wait()
        if i == 0:      # NVDEC refused this stream (or no GPU): hand the whole file to OpenCV
            self.fallback = True
            yield from VideoSource(self.src, max_frames=self.max_frames)


def open_source(src, max_frames: int | None = None, reconnect: bool = True, nvdec: bool = False):
    """VideoSource, or NvdecSource when `nvdec` is set and this is a local H.264/HEVC file that the GPU can decode
    (CUDA torch, ffmpeg + ffprobe present, 8-bit 4:2:0, even size). Anything else: the plain VideoSource."""
    if nvdec and isinstance(src, (str, Path)) and Path(str(src)).is_file():
        info = _probe(str(src))
        if (info and shutil.which("ffmpeg") and info.get("codec_name") in ("h264", "hevc")
                and info.get("pix_fmt") in ("yuv420p", "yuvj420p")
                and info["width"] % 2 == 0 and info["height"] % 2 == 0 and _cuda_ok()):
            return NvdecSource(str(src), info, max_frames)
    return VideoSource(src, max_frames=max_frames, reconnect=reconnect)


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def prefetch(frames: Iterator[Frame], depth: int = 2) -> Iterator[Frame]:
    """Read `frames` on a background thread, `depth` ahead, so decoding overlaps the work on the previous frame.
    Order is unchanged and errors re-raise in the consumer. For files only: a live stream would just queue stale frames."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    end = object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def worker():
        try:
            for f in frames:
                if not put(f):
                    return
            put(end)
        except BaseException as e:     # re-raised in the consumer
            put(e)

    threading.Thread(target=worker, daemon=True, name="bree-prefetch").start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
