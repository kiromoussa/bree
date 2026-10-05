"""Hub receiver for camera nodes (runs on the edge box). Wire format and reasons: see uplink.py.

    POST /v1/burst      one message -> <out>/<camera>/<kind>/<boot_id>_<seq>/ frame_0000.jpg ... + burst.json
    POST /v1/heartbeat  node health  -> <out>/<camera>/heartbeats.jsonl, answers {"hub_time": ...}
    GET  /v1/health     every node's last heartbeat, age, clock offset, counters

A burst is on disk before the hub answers 200, and a retry of the same (camera, boot_id, seq) is
answered 200 without storing it twice. Stored bursts wait in a bounded queue for `on_burst` (the vision
pipeline, slow). When that queue is full the hub answers 429 + Retry-After and the nodes keep their
bursts in their spools: that is the back-pressure.

A burst folder gets a `.done` file once `on_burst` has run on it. Bursts on disk without one (the hub
stopped with a full queue, or two requests raced for the last queue slot) are found by a scan at start
and whenever the queue runs empty after such a race, so a stored burst always reaches the pipeline.

Time: for each node boot the hub keeps offset = (hub wall clock at receive) - (node monotonic at send),
the smallest over the last `OFFSET_WINDOW` requests (the smallest is the one that waited least on the
wire). A frame's hub time is its monotonic time + that offset; burst.json carries both.
"""
from __future__ import annotations

import hmac
import json
import queue
import re
import shutil
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from bree.edge.uplink import unpack

MAX_BODY = 256 * 1024 * 1024
OFFSET_WINDOW = 30          # requests; short enough to follow crystal drift, long enough to skip a slow one
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")   # never starts with a dot: no "." or ".." path parts
DONE = ".done"              # marker in a burst folder: on_burst has run on it (or failed; see the file)


class Hub:
    def __init__(self, out_dir: str | Path, token: str = "", queue_max: int = 32,
                 on_burst: Callable[[Path, dict], None] | None = None, stale_s: float = 30.0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.token, self.on_burst, self.stale_s = token, on_burst, stale_s
        self.q: queue.Queue = queue.Queue(maxsize=queue_max)
        self.nodes: dict[str, dict] = {}
        self._offsets: dict[tuple[str, str], deque] = {}
        self._lock = threading.Lock()
        self.errors: deque[str] = deque(maxlen=100)     # the last 100; error_count has the total
        self.error_count = 0
        self._rescan = threading.Event()                # set = bursts may be on disk that never got queued
        self._rescan.set()                              # at start: whatever the last run left behind
        threading.Thread(target=self._work, daemon=True).start()

    # ------------------------------------------------------------------ requests
    def handle(self, method: str, path: str, headers, body: bytes) -> tuple[int, dict, dict]:
        """(status, JSON reply, extra response headers). No sockets in here, so tests can call it."""
        recv = time.time()
        if self.token and not hmac.compare_digest(headers.get("Authorization", "").encode(), f"Bearer {self.token}".encode()):
            return 401, {"error": "bad token"}, {}
        if method == "GET" and path == "/v1/health":
            return 200, self.health(), {}
        if method != "POST":
            return 404, {"error": "not found"}, {}
        try:
            if path == "/v1/heartbeat":
                hb = json.loads(body)
                cam = self._cam(hb.get("camera_id"))
                self._clock(cam, str(hb.get("boot_id")), headers, recv)
                with self._lock:
                    self.nodes.setdefault(cam, {}).update(last_heartbeat=recv, heartbeat=hb)
                    with open(self._dir(cam) / "heartbeats.jsonl", "a") as f:
                        f.write(json.dumps({"hub_time": recv, **hb}) + "\n")
                return 200, {"hub_time": time.time()}, {}
            if path == "/v1/burst":
                header, blobs = unpack(body)
                cam, kind, boot = self._cam(header.get("camera_id")), header.get("kind", "burst"), str(header.get("boot_id"))
                if not isinstance(kind, str) or not _NAME.match(kind) or not _NAME.match(boot) or len(blobs) != len(header.get("frames", [])):
                    raise ValueError("bad header")
                dest = self._dir(cam) / kind / f"{boot}_{int(header['seq']):010d}"
                if (dest / "burst.json").exists():
                    return 200, {"stored": True, "duplicate": True}, {}
                if self.q.full():
                    return 429, {"error": "busy"}, {"Retry-After": "2"}
                offset = None
                if header.get("clock") == "monotonic":
                    if headers.get("X-Boot", boot) == boot:
                        offset = self._clock(cam, boot, headers, recv)
                    else:   # spooled before the node rebooted: only the old boot's own offset applies
                        with self._lock:
                            offset = min(self._offsets.get((cam, boot), [None]), default=None)
                for fr in header["frames"]:
                    fr["hub_time"] = None if offset is None else round(float(fr["t"]) + offset, 4)
                meta = {**header, "received": recv, "clock_offset": offset}
                tmp = dest.with_name(dest.name + ".tmp")
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                for i, b in enumerate(blobs):
                    (tmp / f"frame_{i:04d}.jpg").write_bytes(b)
                (tmp / "burst.json").write_text(json.dumps(meta))
                tmp.rename(dest)                         # all or nothing: a crash leaves only a .tmp folder
                with self._lock:
                    n = self.nodes.setdefault(cam, {})
                    n[kind + "s"] = n.get(kind + "s", 0) + 1
                    n["bytes"] = n.get("bytes", 0) + len(body)
                self.q.put_nowait((dest, meta))
                return 200, {"stored": True}, {}
        except queue.Full:                               # on disk; a second handler won the last slot.
            self._rescan.set()                           # The worker picks it up from disk when the queue empties.
            return 200, {"stored": True, "queued": False}, {}
        except (ValueError, KeyError, TypeError) as e:
            return 400, {"error": str(e)}, {}
        return 404, {"error": "not found"}, {}

    def _cam(self, cam) -> str:
        if not isinstance(cam, str) or not _NAME.match(cam):
            raise ValueError("bad camera_id")
        return cam

    def _dir(self, cam: str) -> Path:
        d = self.out / cam
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _clock(self, cam: str, boot: str, headers, recv: float) -> float | None:
        try:
            sent = float(headers.get("X-Sent-Mono"))
        except (TypeError, ValueError):
            return None
        with self._lock:
            d = self._offsets.setdefault((cam, boot), deque(maxlen=OFFSET_WINDOW))
            d.append(recv - sent)
            off = min(d)
            self.nodes.setdefault(cam, {})["mono_offset"] = off
        return off

    def health(self) -> dict:
        now = time.time()
        with self._lock:
            return {"queue": self.q.qsize(), "errors": self.error_count,
                    "last_error": self.errors[-1] if self.errors else None, "nodes": {
                cam: {**n, "age_s": (age := round(now - n["last_heartbeat"], 1) if "last_heartbeat" in n else None),
                      "stale": age is None or age > self.stale_s,
                      "stuck_on": bool(n.get("heartbeat", {}).get("stuck_on"))} for cam, n in self.nodes.items()}}

    # ------------------------------------------------------------------ pipeline side
    def _process(self, dest: Path, meta: dict) -> None:
        if not self.on_burst or meta.get("kind", "burst") != "burst" or (dest / DONE).exists():
            return
        err = ""
        try:
            self.on_burst(dest, meta)
        except Exception as e:                           # one bad burst must not stop the hub
            err = repr(e)
            self.error_count += 1
            self.errors.append(f"{dest.name}: {err}")
        (dest / DONE).write_text(err)                    # failed ones are marked too: no retry loop on a bad burst

    def _unprocessed(self) -> list[Path]:
        # ponytail: walks every burst folder; fine for a start-up or after-a-race scan, not for every request.
        return sorted(p.parent for p in self.out.glob("*/burst/*/burst.json") if not (p.parent / DONE).exists())

    def _work(self) -> None:
        while True:
            try:
                dest, meta = self.q.get(timeout=0.5)
            except queue.Empty:
                if self._rescan.is_set() and self.on_burst:
                    self._rescan.clear()
                    for d in self._unprocessed():
                        try:
                            self._process(d, json.loads((d / "burst.json").read_text()))
                        except (OSError, ValueError) as e:
                            self.error_count += 1
                            self.errors.append(f"{d.name}: {e!r}")
                continue
            try:
                self._process(dest, meta)
            finally:
                self.q.task_done()

    def serve(self, host: str = "0.0.0.0", port: int = 8787) -> ThreadingHTTPServer:
        hub = self

        class H(BaseHTTPRequestHandler):
            def _do(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    code, reply, extra = 413, {"error": "too large"}, {"Connection": "close"}
                else:
                    code, reply, extra = hub.handle(method, self.path, self.headers, self.rfile.read(n))
                data = json.dumps(reply).encode()
                self.send_response(code)
                for k, v in {"Content-Type": "application/json", "Content-Length": str(len(data)), **extra}.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                self._do("POST")

            def do_GET(self):
                self._do("GET")

            def log_message(self, *a):
                pass

        httpd = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd


def burst_video(burst_dir: Path, meta: dict) -> Path:
    """The burst's JPEGs as burst.mp4 at the node's frame rate: the form the pipeline's VideoSource reads."""
    import cv2
    out = burst_dir / "burst.mp4"
    vw = None
    for p in sorted(burst_dir.glob("frame_*.jpg")):
        img = cv2.imread(str(p))
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(meta["fps"]), img.shape[1::-1])
        vw.write(img)
    if vw is not None:
        vw.release()
    return out


class BurstSource:
    """Every stored burst of one camera as ONE frame stream with gaps, in the shape the pipeline reads
    (bree.ingest.source.VideoSource: `fps`, `live`, iterating Frame). Give it to bree.pipeline.CameraInput
    as `source`, so several node cameras (and ordinary streams) run into one ledger with one identity pool.

    Frame time: the hub's wall time (`hub_time`) minus `t0` when the node sent a monotonic clock; else the
    node's own media time (a file played as the camera) minus `t0`. Use the same `t0` for every camera of
    a store (default: this camera's first frame for hub time, 0 for media time).

    ponytail: reads what is on disk when iteration starts (recorded bursts), it does not follow a live
    hub; and the tracker is not told about the gaps, so a track can carry over a short gap between two
    bursts. Upgrade: a tailing source plus a tracker reset when the gap is longer than its lost buffer."""
    live = False

    def __init__(self, camera_dir: str | Path, t0: float | None = None, max_frames: int | None = None):
        rows = {}
        for bj in sorted(Path(camera_dir).glob("burst/*/burst.json")):
            meta = json.loads(bj.read_text())
            for i, fr in enumerate(meta["frames"]):
                wall = fr.get("hub_time")
                rows.setdefault(round(wall if wall is not None else fr["t"], 4), (wall is not None, bj.parent / f"frame_{i:04d}.jpg"))
            self.fps = float(meta["fps"])
            self.width, self.height = meta["size"]
        if not rows:
            self.fps, self.width, self.height = 10.0, 0, 0
        times = sorted(rows)
        self.t0 = t0 if t0 is not None else (times[0] if rows and rows[times[0]][0] else 0.0)
        self.rows = [(t, rows[t][1]) for t in times][:max_frames]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        import cv2

        from bree.ingest.source import Frame
        for i, (t, path) in enumerate(self.rows):
            yield Frame(i, t - self.t0, cv2.imread(str(path)), time.time())


def pipeline_feed(stores: dict[str, str], backend: str = "yolo") -> Callable[[Path, dict], None]:
    """on_burst that runs the existing vision pipeline on each burst. stores: camera id -> store YAML
    (zones in that camera's full-res pixels). Output in <burst>/pipeline/ (events.jsonl, alerts.jsonl ...).
    Event times there are seconds from the burst's first frame; add burst.json frames[0].hub_time.

    ponytail: one pipeline run per burst, so a shopper has no identity or basket across bursts.
    Joining bursts into one ledger needs a pipeline source that accepts gaps (pipeline.py, not here)."""
    from bree.cli import make_backend
    from bree.events.zones import load_store_config
    from bree.pipeline import run_pipeline
    loaded = {cam: load_store_config(p) for cam, p in stores.items()}
    backends = {cam: make_backend(backend, s) for cam, s in loaded.items()}

    def on_burst(burst_dir: Path, meta: dict) -> None:
        cam = meta["camera_id"]
        if cam in loaded:
            run_pipeline(str(burst_video(burst_dir, meta)), loaded[cam], backends[cam], burst_dir / "pipeline",
                         save_video=False, verbose=False)
    return on_burst


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="BREE hub: receive bursts and heartbeats from camera nodes")
    ap.add_argument("--out", default="out/hub")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--token", default="", help="shared secret; nodes send it as a Bearer token")
    ap.add_argument("--store", action="append", default=[], metavar="CAMERA=STORE.yaml",
                    help="run the vision pipeline on that camera's bursts (repeatable)")
    ap.add_argument("--backend", default="yolo", choices=["yolo", "toy", "sim_sku"])
    a = ap.parse_args(argv)
    stores = dict(s.split("=", 1) for s in a.store)
    hub = Hub(a.out, a.token, on_burst=pipeline_feed(stores, a.backend) if stores else None)
    hub.serve(a.host, a.port)
    print(f"hub listening on {a.host}:{a.port}, storing in {a.out}" + ("" if a.token else "  (NO TOKEN SET)"), flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
