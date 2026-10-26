"""Rolling segment recorder for a station's DVR (replaces scripts/imou_record.py's own copy of the client).

Per channel: fresh live address (ImouClient, read-only methods) -> ffmpeg -c copy of the stream for `seg_seconds`
into <data>/real/<station>/chNN/<YYYYmmdd_HHMMSS>.mp4.part, renamed only when finished (readers never see a partial
file). Video only, no audio. Footage dir and status file stay outside every git repo and are mode 700 / 600.

The lesson of 2026-10-25: when the platform's flow allowance ran out it served a 4 s stub video, and a recorder that
retried every minute on 7 channels fetched stubs for an hour. This one has a circuit breaker: `max_stubs` stub
segments in a row (any channel) trip it, all channels pause and then one probe channel tries again after a cool-down
that doubles from 30 min to 6 h. The state is written to the status file, so `bree fleet status` shows it.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from bree.ingest.imou import ImouClient, ImouError, StreamStaleError, load_credentials, scrub

MAX_STREAMS = 8
MIN_SEGMENT_BYTES = 100_000


def probe_duration(path: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


class CircuitBreaker:
    """Counts consecutive stub segments across channels. Open = nobody records; after the cool-down one probe is allowed."""

    def __init__(self, max_stubs: int = 3, cooldown_s: float = 1800.0, max_cooldown_s: float = 21600.0, clock=time.time):
        self.max_stubs, self.base, self.max = max_stubs, cooldown_s, max_cooldown_s
        self.stubs, self.trips, self.open_until, self._clock = 0, 0, 0.0, clock
        self._lock = threading.Lock()

    def record(self, good: bool) -> None:
        with self._lock:
            if good:
                self.stubs, self.trips, self.open_until = 0, 0, 0.0
                return
            self.stubs += 1
            if self.stubs >= self.max_stubs:
                self.open_until = self._clock() + min(self.base * 2 ** self.trips, self.max)
                self.trips += 1
                self.stubs = self.max_stubs - 1       # after the cool-down a single stub trips it again

    def wait_s(self) -> float:
        with self._lock:
            return max(0.0, self.open_until - self._clock())

    @property
    def open(self) -> bool:
        return self.wait_s() > 0


class Recorder:
    def __init__(self, client: ImouClient, serial: str, station: str, channels: list[int], footage_root: Path, status_path: Path,
                 seg_seconds: int = 300, stream: str = "main", cap_gb: float = 80.0, hours: float = 12.0,
                 breaker: CircuitBreaker | None = None, health_sinks: dict | None = None, log=print, stop_file: Path | None = None,
                 min_bytes: int = MIN_SEGMENT_BYTES):
        assert len(channels) <= MAX_STREAMS, f"at most {MAX_STREAMS} concurrent streams"
        self.client, self.serial, self.station, self.channels = client, serial, station, channels
        self.root, self.status_path = Path(footage_root), Path(status_path)
        self.seg, self.stream, self.cap = seg_seconds, stream, cap_gb * 1e9
        self.deadline = time.time() + hours * 3600
        self.breaker = breaker or CircuitBreaker()
        self.sinks = health_sinks or {}
        self.log, self.stop_file, self.min_bytes = log, stop_file, min_bytes
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.state = {"station": station, "serial": serial, "started": time.time(), "pid": os.getpid(), "state": "starting",
                      "total_bytes": 0, "breaker_trips": 0, "channels": {str(c): {"segments": 0, "stubs": 0, "last_segment": None,
                      "last_segment_time": None, "consecutive_failures": 0, "errors": []} for c in channels}}

    # -- status
    def _write(self) -> None:
        with self._lock:
            self.state["updated"] = time.time()
            self.state["breaker_trips"] = self.breaker.trips
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.status_path)

    def _note(self, ch: int, ok: bool | None = None, seg: str | None = None, err: str | None = None, stub: bool = False) -> None:
        with self._lock:
            c = self.state["channels"][str(ch)]
            if ok:
                c.update(last_segment=seg, last_segment_time=time.time(), consecutive_failures=0)
                c["segments"] += 1
            if stub:
                c["stubs"] += 1
            if err:
                c["consecutive_failures"] += 1
                c["errors"] = (c["errors"] + [{"t": time.strftime("%F %T"), "e": err}])[-5:]
        sink = self.sinks.get(ch)
        if sink:
            sink.update(True, state=("ok" if ok else "stub" if stub else "reconnecting"), last_error=err,
                        last_frame_wall=time.time() if ok else sink.d.get("last_frame_wall"),
                        consecutive_failures=c["consecutive_failures"], stub_opens=c["stubs"])
        self._write()

    # -- one segment
    def _ffmpeg(self, url: str, part: Path, dur: int, ch: int) -> int:
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-rw_timeout", "20000000", "-i", url, "-map", "0:v:0",
               "-an", "-c", "copy", "-t", str(dur), "-f", "mp4", "-movflags", "+faststart", "-y", str(part)]
        pr = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        t0, last_size = time.time(), 0
        while pr.poll() is None:
            if self.stop.is_set() or time.time() - t0 > dur + 120:
                pr.send_signal(signal.SIGINT)       # finalise the mp4: the partial segment is kept
                try:
                    pr.wait(20)
                except subprocess.TimeoutExpired:
                    pr.kill()
                break
            size = part.stat().st_size if part.exists() else 0
            if size > last_size and ch in self.sinks:          # growing file = frames are arriving: heartbeat
                self.sinks[ch].update(last_frame_wall=time.time(), state="ok")
                last_size = size
            self.stop.wait(2)
        self._stderr = scrub(pr.stderr.read() if pr.stderr else "", self.client._secrets())[-200:]
        return pr.returncode or 0

    def record_one(self, ch: int) -> str:
        """Record one segment. Returns the finished file name; raises StreamStaleError for a stub, ImouError otherwise."""
        out = self.root / f"ch{ch:02d}"
        out.mkdir(parents=True, exist_ok=True)
        os.chmod(out, 0o700)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        part, final = out / f"{stamp}.mp4.part", out / f"{stamp}.mp4"
        dur = int(min(self.seg, max(5, self.deadline - time.time())))
        try:
            url = self.client.live_url(self.serial, ch, self.stream)
            rc = self._ffmpeg(url, part, dur, ch)
            size = part.stat().st_size if part.exists() else 0
            if size < self.min_bytes:
                if rc != 0:
                    raise ImouError(f"ffmpeg rc={rc} {self._stderr}")
                raise StreamStaleError(f"segment too small ({size} bytes): stub stream")
            got = probe_duration(part)
            if got < 0.5 * dur and not self.stop.is_set():
                raise StreamStaleError(f"segment is {got:.1f} s of the {dur} s asked for: stub or cut stream")
            os.replace(part, final)
            return final.name
        finally:
            if part.exists():
                part.unlink()

    def _enforce_cap(self) -> int:
        files = sorted(self.root.glob("ch*/*.mp4"), key=lambda f: f.name[:15] + f.parent.name)
        total = sum(f.stat().st_size for f in files)
        while total > self.cap and files:
            f = files.pop(0)
            total -= f.stat().st_size
            f.unlink()
            self.log(f"cap: deleted {f.parent.name}/{f.name}")
        return total

    def _loop(self, ch: int, probe_only_first: bool) -> None:
        backoff = 5.0
        while not self.stop.is_set() and time.time() < self.deadline:
            wait = self.breaker.wait_s()
            if wait > 0:                                       # breaker open: everybody pauses; channel 0 of the list probes after
                self.state["state"] = "stale_stream_paused"
                self._write()
                self.stop.wait(min(wait, 60))
                continue
            if self.breaker.stubs >= self.breaker.max_stubs - 1 and ch != self.channels[0] and self.breaker.trips:
                self.stop.wait(30)                              # half-open: only the first channel probes
                continue
            try:
                name = self.record_one(ch)
                self.breaker.record(True)
                self.state["state"] = "recording"
                self._note(ch, ok=True, seg=name)
                backoff = 5.0
                with self._lock:
                    self.state["total_bytes"] = self._enforce_cap()
            except StreamStaleError as e:
                self.breaker.record(False)
                msg = scrub(e, self.client._secrets())
                self.log(f"ch{ch:02d} stub: {msg} (breaker stubs={self.breaker.stubs}, open={self.breaker.open})")
                self._note(ch, err=msg, stub=True)
                self.stop.wait(15)
            except ImouError as e:
                msg = scrub(f"{type(e).__name__}: {e}", self.client._secrets())
                self.log(f"ch{ch:02d} error: {msg} (retry in {backoff:.0f}s)")
                self._note(ch, err=msg)
                self.stop.wait(backoff)
                backoff = min(backoff * 2, 300)

    def run(self) -> None:
        self._write()
        ths = []
        for ch in self.channels:
            t = threading.Thread(target=self._loop, args=(ch, False), daemon=True)
            t.start()
            ths.append(t)
            self.stop.wait(1)
        while not self.stop.is_set() and time.time() < self.deadline and any(t.is_alive() for t in ths):
            if self.stop_file and self.stop_file.exists():
                self.stop.set()
            self.stop.wait(2)
        self.stop.set()
        for t in ths:
            t.join(timeout=60)
        for f in self.root.glob("ch*/*.part"):
            f.unlink()
        self.state["state"] = "stopped"
        self._write()


def main(args) -> None:
    from bree.fleet.registry import Fleet
    from bree.ingest.imou import HealthSink
    fleet = Fleet(args.fleet)
    serial, channels = args.serial, [int(c) for c in args.channels.split(",")] if args.channels else None
    try:
        st = fleet.load(args.station)
        serial = serial or st.serial
        channels = channels or [c.channel for c in st.cameras if c.enabled and c.role not in ("outdoor", "obscured", "backroom", "unknown")]
    except KeyError:
        st = None
    if not serial or not channels:
        raise SystemExit("need --serial and --channels, or an onboarded station in the fleet registry")
    data = Path(args.data).expanduser()
    root = data / "real" / args.station
    log_f = open(data / f"recorder_{args.station}.log", "a", buffering=1)
    os.chmod(data / f"recorder_{args.station}.log", 0o600)
    sinks = {c: HealthSink(fleet.health_path(args.station, c), args.station, serial, c, every_s=5) for c in channels} if st else {}
    rec = Recorder(ImouClient(load_credentials()), serial, args.station, channels, root, data / f"recorder_status_{args.station}.json",
                   args.seg_seconds, args.stream, args.cap_gb, args.hours, health_sinks=sinks,
                   log=lambda m: log_f.write(f"{time.strftime('%F %T')} {m}\n"), stop_file=data / f"recorder_{args.station}.stop")
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: rec.stop.set())
    rec.log(f"start pid={os.getpid()} channels={channels} seg={args.seg_seconds}s stream={args.stream}")
    rec.run()
    rec.log("exit")
