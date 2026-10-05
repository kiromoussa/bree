"""Camera node main loop: low-res trigger -> full-res burst with pre-roll -> spool -> hub.

    python3 -m bree.edge.node --config /etc/bree/node.yaml

The config is a store YAML (camera.id, camera.resolution, zones: the same file scripts/draw_zones.py
writes) plus a `node:` block; see deploy/pi/node.example.yaml. Only shelf and cooler zones trigger.
Needs numpy, OpenCV and PyYAML (and picamera2 on the Pi); nothing else from the pipeline.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from bree.edge.capture import Burst, BurstRecorder, FileSource, FrameSource, _jpeg
from bree.edge.trigger import TriggerConfig, ZoneTrigger
from bree.edge.uplink import Spool, Uplink


@dataclass
class NodeConfig:
    camera_id: str
    resolution: tuple[int, int]                  # pixels the zone polygons are drawn in
    zones: dict[str, np.ndarray]                 # merch zones only
    source: str = "picamera2"                    # picamera2 | a video file | a webcam index ("0")
    main_size: tuple[int, int] | None = None     # picamera2 full-res stream; default = resolution
    fps: float = 10.0                            # capture rate of both streams
    hub_url: str = "http://127.0.0.1:8787"
    token: str = ""
    spool_dir: str = "/var/lib/bree-node/spool"
    spool_max_mb: float = 2000.0
    pre_roll_s: float = 2.0
    post_roll_s: float = 1.5
    max_burst_s: float = 6.0
    jpeg_quality: int = 85
    heartbeat_s: float = 10.0
    lowres_uplink_fps: float = 0.0               # 0 = the low-res stream stays on the node
    lowres_chunk_s: float = 5.0
    lowres_quality: int = 70
    stuck_window_s: float = 300.0                # the heartbeat reports the share of this window the trigger was active
    stuck_share: float = 0.9                     # ... and stuck_on: true above this (shake, flicker, a bad zone)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


def load_node_config(path: str | Path) -> NodeConfig:
    raw = yaml.safe_load(Path(path).read_text())
    node = dict(raw.get("node") or {})
    unknown = set(node) - {f.name for f in fields(NodeConfig)}
    if unknown:
        raise ValueError(f"unknown node setting(s): {sorted(unknown)}")
    zones = {z["name"]: np.asarray(z["polygon"], dtype=float) for z in raw.get("zones", [])
             if z.get("kind") in ("shelf", "cooler")}
    if not zones:
        raise ValueError("no shelf or cooler zone in the config: nothing would ever trigger")
    cam = raw.get("camera", {})
    node["trigger"] = TriggerConfig.from_dict(node.get("trigger"))
    for k in ("main_size",):
        if node.get(k):
            node[k] = tuple(node[k])
    return NodeConfig(camera_id=cam.get("id", "cam0"), resolution=tuple(cam.get("resolution", [1280, 720])),
                      zones=zones, **node)


def make_source(cfg: NodeConfig) -> FrameSource:
    if cfg.source == "picamera2":
        from bree.edge.capture import Picamera2Source
        return Picamera2Source(cfg.main_size or cfg.resolution, cfg.fps, cfg.trigger.width)
    return FileSource(cfg.source, cfg.fps, cfg.trigger.width)


def pack_burst(b: Burst, source: FrameSource, quality: int, pressure: float = 0.0) -> tuple[dict, list[bytes]]:
    """Header + JPEGs for one burst. Back-pressure: over half a spool every second pre/post frame is
    left out; over 90 % only the frames with a zone active are sent."""
    frames = b.frames
    thin = "none"
    if pressure > 0.9:
        frames, thin = [f for f in frames if f.phase == "during"] or frames[:1], "during_only"
    elif pressure > 0.5:
        frames, thin = [f for i, f in enumerate(frames) if f.phase == "during" or i % 2 == 0], "half_roll"
    header = {"kind": "burst", "trigger_id": b.trigger_id, "part": b.part, "t_trigger": round(b.t_trigger, 4),
              "zones": b.zones, "fps": source.fps, "size": list(source.size), "thinned": thin,
              "frames": [{"t": round(f.t, 4), "zones": f.zones, "phase": f.phase} for f in frames]}
    return header, [f.main.result() if isinstance(f.main, Future) else source.encode(f.main, quality) for f in frames]


class Node:
    """The loop without the network thread, so tests and the measurement script can drive it."""

    def __init__(self, cfg: NodeConfig, source: FrameSource, uplink: Uplink):
        self.cfg, self.source, self.uplink = cfg, source, uplink
        self.trigger = ZoneTrigger(cfg.zones, cfg.resolution, cfg.trigger)
        self.recorder = BurstRecorder(cfg.camera_id, source.fps, cfg.pre_roll_s, cfg.post_roll_s, cfg.max_burst_s)
        self.frames = self.triggers = self.bursts = self.burst_bytes = self.lowres_bytes = 0
        self.trigger_s = 0.0                      # wall seconds spent in the trigger
        self.active_share = 0.0                   # moving average over about cfg.stuck_window_s
        self.active_log: list[tuple[float, tuple[str, ...]]] = []   # (t, zones) per frame, for measurement
        self.keep_log = False
        self._low: list[tuple[float, bytes]] = []
        self._low_next = 0.0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._slots = threading.BoundedSemaphore(max(4, self.recorder.ring.maxlen))   # raw frames waiting to be encoded
        self._stuck_n = cfg.stuck_window_s * source.fps
        self.errors: list[str] = []
        uplink.status = self.status

    def status(self) -> dict:
        return {"frames": self.frames, "triggers": self.triggers, "bursts": self.bursts,
                "trigger_ms": round(1000 * self.trigger_s / max(self.frames, 1), 3), "node_errors": len(self.errors),
                "active_share": round(self.active_share, 3),
                "stuck_on": self.frames >= self._stuck_n and self.active_share > self.cfg.stuck_share}

    def _encode_new(self, b: Burst) -> None:
        """Hand the burst's raw frames to the encode thread and keep only the pending JPEG. A burst
        therefore never sits in RAM as raw frames: at 2304x1296 that would be 4.5 MB a frame."""
        for f in b.frames:
            if not isinstance(f.main, Future):
                self._slots.acquire()              # capture waits here if the encoder is a ring's worth behind
                f.main = self._pool.submit(self._encode, f.main)

    def _encode(self, main) -> bytes:
        try:
            return self.source.encode(main, self.cfg.jpeg_quality)
        finally:
            self._slots.release()

    def _send(self, b: Burst) -> None:
        self._encode_new(b)
        self._pool.submit(self._finish, b)         # one worker, in order: its frames are encoded by then

    def _finish(self, b: Burst) -> None:
        try:
            header, blobs = pack_burst(b, self.source, self.cfg.jpeg_quality, self.uplink.pressure())
            self.burst_bytes += self.uplink.submit(header, blobs)
            self.bursts += 1
            self.triggers += b.part == 0
        except Exception as e:                     # a failed encode or a full disk must not stop capture
            self.errors.append(repr(e))

    def _lowres(self, t: float, lores: np.ndarray, flush: bool = False) -> None:
        c = self.cfg
        if not flush and t >= self._low_next:
            self._low_next = t + 1.0 / c.lowres_uplink_fps
            self._low.append((t, _jpeg(lores, c.lowres_quality)))
        if self._low and (flush or t - self._low[0][0] >= c.lowres_chunk_s):
            header = {"kind": "lowres", "fps": c.lowres_uplink_fps, "size": list(lores.shape[1::-1]),
                      "frames": [{"t": round(ft, 4), "zones": [], "phase": "lowres"} for ft, _ in self._low]}
            self.lowres_bytes += self.uplink.submit(header, [b for _, b in self._low])
            self._low = []

    def run(self, stop: threading.Event | None = None) -> None:
        lores, t = None, 0.0
        for fr in self.source.frames():
            t, lores = fr.t, fr.lores
            t0 = time.perf_counter()
            active = self.trigger.update(fr.lores, fr.t)
            self.trigger_s += time.perf_counter() - t0
            self.frames += 1
            self.active_share += (bool(active) - self.active_share) / max(1.0, min(self.frames, self._stuck_n))
            if self.keep_log:
                self.active_log.append((fr.t, tuple(sorted(active))))
            for b in self.recorder.push(fr.t, fr.main, active):
                self._send(b)
            if self.recorder.cur is not None:      # a burst is open: encode its frames as they arrive
                self._encode_new(self.recorder.cur)
            if self.cfg.lowres_uplink_fps > 0:
                self._lowres(fr.t, fr.lores)
            if stop is not None and stop.is_set():
                break
        for b in self.recorder.flush():
            self._send(b)
        self._pool.shutdown(wait=True)             # every burst is in the spool before run() returns
        if self.cfg.lowres_uplink_fps > 0 and lores is not None:
            self._lowres(t, lores, flush=True)


def build(cfg: NodeConfig, source: FrameSource | None = None) -> Node:
    source = source or make_source(cfg)
    uplink = Uplink(cfg.hub_url, cfg.camera_id, Spool(cfg.spool_dir, int(cfg.spool_max_mb * 1e6)), cfg.token,
                    cfg.heartbeat_s, clock="monotonic" if getattr(source, "live", True) else "media")
    return Node(cfg, source, uplink)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import signal
    ap = argparse.ArgumentParser(description="BREE camera node")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", help="override node.source (a video file or webcam index, for testing)")
    a = ap.parse_args(argv)
    cfg = load_node_config(a.config)
    if a.source:
        cfg.source = a.source
    node = build(cfg)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sender = threading.Thread(target=node.uplink.run, args=(stop,), daemon=True)
    sender.start()
    print(f"node {cfg.camera_id}: {cfg.source} {node.source.size} at {node.source.fps:.1f} fps, "
          f"{len(cfg.zones)} zone(s), hub {cfg.hub_url}", flush=True)
    try:
        node.run(stop)
    except KeyboardInterrupt:
        pass
    stop.set()
    sender.join(timeout=2)                    # a POST may still be in flight: step() holds a lock, so drain waits for it
    node.uplink.drain(10)                     # a file source ends: send what is left; a Pi being stopped: best effort
    print(f"node {cfg.camera_id}: stopped, {node.status()}, {len(node.uplink.spool)} message(s) left in the spool")


if __name__ == "__main__":
    main()
