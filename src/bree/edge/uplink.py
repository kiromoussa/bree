"""Node -> hub transport: plain HTTP POST, a disk spool for when the hub is away, back-pressure,
heartbeats. Standard library only.

Why HTTP and not MQTT: a burst is a few megabytes of JPEG, which is what a POST body is for and
what MQTT brokers are tuned against; both ends need nothing beyond Python's standard library (no broker
to install, run and secure on the hub); the repo already takes POS payments the same way; and the hub
can say "slow down" with a status code (429 + Retry-After). The nodes sit on wired PoE, so MQTT's
strength on flaky links buys little. What MQTT would give for free (queued delivery, last-will) is the
spool and the heartbeat below.

Wire format of one message (POST /v1/burst, application/octet-stream):
    b"BREE1" | 4-byte big-endian header length | header JSON (utf-8) | frame 0 JPEG | frame 1 JPEG ...
The header lists `sizes` (bytes of each JPEG) and `frames` (one {t, zones, phase} per JPEG).
Request headers: Authorization: Bearer <token>, X-Sent-Mono: <node monotonic seconds at send>.
Replies: 200 stored (or already stored: retries are safe, the hub dedupes on camera/boot_id/seq),
429 or 503 + Retry-After = hub busy (keep the message, wait), 401 = wrong token (keep, wait),
any other 4xx = the hub will never take this message (it is dropped and counted).

Heartbeat (POST /v1/heartbeat, JSON) every `heartbeat_s`: spool depth, drops, counters, clocks.
The hub answers {"hub_time": unix seconds}.

Clocks. Frame times are the node's monotonic clock (it never jumps, and a Pi Zero has no battery clock
so its wall time is wrong until NTP answers). `boot_id` changes on every start because monotonic time
restarts. The hub turns monotonic into its own wall time from X-Sent-Mono on every request (see
hub.py), so frames spooled while offline still get the right time once the node is back, as long as
it did not reboot in between. Run chrony or systemd-timesyncd against the hub anyway: the heartbeat
reports the node's wall-clock offset so a drifting node shows up on the hub.
"""
from __future__ import annotations

import http.client
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

MAGIC = b"BREE1"


def pack(header: dict, blobs: list[bytes]) -> bytes:
    h = json.dumps({**header, "sizes": [len(b) for b in blobs]}, separators=(",", ":")).encode()
    return MAGIC + struct.pack(">I", len(h)) + h + b"".join(blobs)


def unpack(data: bytes) -> tuple[dict, list[bytes]]:
    n = len(MAGIC)
    if data[:n] != MAGIC or len(data) < n + 4:
        raise ValueError("not a BREE1 message")
    (hl,) = struct.unpack(">I", data[n:n + 4])
    body = n + 4 + hl
    if body > len(data):
        raise ValueError("truncated header")
    header = json.loads(data[n + 4:body])
    sizes = header.get("sizes")
    if not isinstance(sizes, list) or any(not isinstance(s, int) or s < 0 for s in sizes) \
            or body + sum(sizes) != len(data):
        raise ValueError("sizes do not match the body")
    blobs, o = [], body
    for s in sizes:
        blobs.append(data[o:o + s])
        o += s
    return header, blobs


class Spool:
    """Messages waiting for the hub, one file each, oldest first. Over `max_bytes` the oldest are
    deleted (and counted): on a full card the newest evidence is the one still worth sending."""

    def __init__(self, folder: str | Path, max_bytes: int = 2_000_000_000):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.dropped = 0
        self._lock = threading.Lock()
        for p in self.dir.glob("*.tmp"):
            p.unlink()                                   # half-written when the power went
        for p in self.dir.glob("*.msg"):
            if not p.stem.isdigit():                     # not ours: set it aside so the node still starts
                p.rename(p.with_suffix(".bad"))
        self._files = sorted(self.dir.glob("*.msg"), key=lambda p: int(p.stem))
        self.bytes = sum(p.stat().st_size for p in self._files)
        self._next = int(self._files[-1].stem) + 1 if self._files else 0

    def put(self, data: bytes) -> None:
        with self._lock:
            p = self.dir / f"{self._next:012d}.msg"
            self._next += 1
            tmp = p.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp.rename(p)
            self._files.append(p)
            self.bytes += len(data)
            while self.bytes > self.max_bytes and len(self._files) > 1:
                self._remove(self._files[0])
                self.dropped += 1

    def oldest(self) -> Path | None:
        with self._lock:
            return self._files[0] if self._files else None

    def remove(self, p: Path) -> None:
        with self._lock:
            self._remove(p)

    def _remove(self, p: Path) -> None:
        if p in self._files:
            self._files.remove(p)
            self.bytes -= p.stat().st_size
            p.unlink()

    def __len__(self) -> int:
        return len(self._files)


class Uplink:
    def __init__(self, hub_url: str, camera_id: str, spool: Spool, token: str = "", heartbeat_s: float = 10.0,
                 timeout_s: float = 10.0, status: Callable[[], dict] | None = None, clock: str = "monotonic"):
        self.hub_url, self.camera_id, self.spool, self.token = hub_url.rstrip("/"), camera_id, spool, token
        self.heartbeat_s, self.timeout_s, self.status, self.clock = heartbeat_s, timeout_s, status, clock
        self.boot_id = uuid.uuid4().hex[:12]
        self.seq = 0
        self.sent = self.sent_bytes = self.rejected = self.busy = self.errors = 0
        self.clock_offset_s: float | None = None        # hub wall clock minus node wall clock, last heartbeat
        self._backoff = 0.0
        self._not_before = 0.0
        self._next_heartbeat = 0.0
        self._lock = threading.Lock()
        self._step_lock = threading.Lock()              # one sender at a time (run() thread vs drain())

    def submit(self, header: dict, blobs: list[bytes]) -> int:
        """Spool one message. Returns its size in bytes. Never blocks on the network."""
        with self._lock:
            seq, self.seq = self.seq, self.seq + 1
        data = pack({**header, "camera_id": self.camera_id, "boot_id": self.boot_id, "seq": seq,
                     "clock": self.clock}, blobs)
        self.spool.put(data)
        return len(data)

    def pressure(self) -> float:
        """0 = spool empty, 1 = spool full. The node thins bursts as this rises."""
        return min(1.0, self.spool.bytes / max(self.spool.max_bytes, 1))

    def _post(self, path: str, body: bytes, ctype: str) -> tuple[int, dict, bytes]:
        req = urllib.request.Request(self.hub_url + path, data=body, method="POST", headers={
            "Content-Type": ctype, "Authorization": f"Bearer {self.token}",
            "X-Camera": self.camera_id, "X-Boot": self.boot_id, "X-Sent-Mono": f"{time.monotonic():.6f}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def _wait(self, headers: dict, default: float) -> None:
        try:
            s = float(headers.get("Retry-After", default))
        except ValueError:
            s = default
        self._not_before = time.monotonic() + min(max(s, 0.1), 60.0)

    def heartbeat(self) -> dict:
        t0, wall = time.monotonic(), time.time()
        body = {"camera_id": self.camera_id, "boot_id": self.boot_id, "t_mono": t0, "t_wall": wall,
                "spool_files": len(self.spool), "spool_bytes": self.spool.bytes, "spool_dropped": self.spool.dropped,
                "sent": self.sent, "sent_bytes": self.sent_bytes, "rejected": self.rejected, "busy": self.busy,
                "net_errors": self.errors, "clock_offset_s": self.clock_offset_s,
                **(self.status() if self.status else {})}
        code, _, reply = self._post("/v1/heartbeat", json.dumps(body).encode(), "application/json")
        if code == 200:
            rtt = time.monotonic() - t0
            self.clock_offset_s = round(json.loads(reply)["hub_time"] - (wall + rtt / 2), 4)
        return body

    def step(self) -> bool:
        """One round: a heartbeat if due, then the oldest spooled message. True if a message left."""
        with self._step_lock:
            return self._step()

    def _step(self) -> bool:
        now = time.monotonic()
        if now < self._not_before:
            return False
        try:
            if now >= self._next_heartbeat:
                self._next_heartbeat = now + self.heartbeat_s
                self.heartbeat()
            p = self.spool.oldest()
            if p is None:
                return False
            data = p.read_bytes()
            code, headers, _ = self._post("/v1/burst", data, "application/octet-stream")
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
            # hub down, cable out, DNS, timeout, or a reply that is not ours (garbage status line,
            # body cut short by a hub killed mid-reply, JSON without hub_time)
            self._fail()
            return False
        self._backoff = 0.0
        if code == 200:
            self.spool.remove(p)
            self.sent += 1
            self.sent_bytes += len(data)
            return True
        if code in (429, 503):                           # hub busy: back-pressure
            self.busy += 1
            self._wait(headers, 2.0)
        elif code == 401:                                # wrong token: a config error, keep the evidence
            self.errors += 1
            self._wait({}, 30.0)
        else:                                            # the hub will never take it: do not block the queue
            self.spool.remove(p)
            self.rejected += 1
        return False

    def _fail(self) -> None:
        self.errors += 1
        self._backoff = min(60.0, max(1.0, self._backoff * 2))
        self._not_before = time.monotonic() + self._backoff

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                sent = self.step()
            except Exception:                            # the sender thread must never die: the process
                self._fail()                             # would live on, spooling and sending nothing
                sent = False
            if not sent:
                stop.wait(0.2)

    def drain(self, timeout_s: float = 30.0) -> bool:
        """Send until the spool is empty (tests, shutdown). False if it did not empty in time."""
        end = time.monotonic() + timeout_s
        while len(self.spool) and time.monotonic() < end:
            if not self.step():
                time.sleep(0.05)
        return not len(self.spool)
