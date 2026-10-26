"""Imou Open Platform client and live source: `bree run --source imou://<serial>/<channel>`.

Dahua/Imou DVRs that are bound to an Imou developer app can be watched without any port forwarding: the
platform hands out a short-lived HLS address per channel. This module is the production form of the throwaway
recorder script: signed API calls, token renewal, live-address renewal, reconnect with back-off, sub/main stream
choice, frame timestamps (wall clock, optionally shifted by the DVR-clock offset measured from the video overlay)
and a circuit breaker for the failure that bit the first pilot day.

Read-only by construction: the client refuses every API method except READ_ONLY (accessToken, deviceOnline,
bindDeviceLive, getLiveStreamInfo). `bindDevice` (attaching a brand-new DVR to the app) exists for `bree onboard`
and needs an explicit `allow_bind=True`; it has only ever run against the mock server in this repo.

Failure story (what the caller sees):
  ImouConfigError      no credentials / bad imou:// URL: fix the configuration, retrying does not help
  ImouAuthError        the platform rejected the app id/secret or the token could not be renewed
  DeviceOfflineError   deviceOnline says the DVR or the channel is offline
  StreamStaleError     the platform returns a "stub": a stream that ends after a few seconds, over and over. On
                       2026-10-25 this was the flow allowance running out (platform error 11003, a 4 s stub video
                       instead of live video). The source stops after `max_stub_retries` opens and never hammers
                       a metered API; a person has to look at the account.
  StreamUnavailable    network/platform trouble outlasted `max_wait_s`
All derive from ImouError. Messages never contain the app secret, a token or a live address (those carry credentials).

Credentials come from the environment or ~/.config/bree/imou.env: IMOU_APP_ID, IMOU_APP_SECRET, IMOU_DC (data
center: sg | fk | or). Never from argv. The URL carries only the serial and channel.

Not covered by real-device tests: the live HLS behaviour of the real platform after the free flow allowance ran out
(no more live footage was available); everything else is tested against bree.ingest.imou_mock (no network in tests).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2

from bree.ingest.source import Frame

ENV_FILE = Path.home() / ".config" / "bree" / "imou.env"
READ_ONLY = frozenset({"accessToken", "deviceOnline", "bindDeviceLive", "getLiveStreamInfo"})
BIND_METHODS = frozenset({"bindDevice"})
STREAM_IDS = {"main": 0, "sub": 1}
MAX_STREAMS = 8                      # never open more than this many live streams for one account
URL_RE = re.compile(r"\S+://\S+")
# Platform result codes we act on (everything else surfaces as ImouError with the code and message).
CODE_OK = "0"
CODE_TOKEN_EXPIRED = {"TK1002", "TK1001"}
CODE_LIVE_EXISTS = "LV1001"          # a live session already exists for this channel: fine, ask for its address
CODE_QUOTA = {"11003", "FL1003"}      # insufficient flow resources


class ImouError(RuntimeError):
    """Base class. `code` is the platform result code when there is one."""

    def __init__(self, msg: str, code: str | None = None):
        super().__init__(msg)
        self.code = code


class ImouConfigError(ImouError):
    pass


class ImouAuthError(ImouError):
    pass


class DeviceOfflineError(ImouError):
    pass


class StreamStaleError(ImouError):
    pass


class StreamUnavailable(ImouError):
    pass


# ----------------------------------------------------------------------------- credentials, signing, URLs

@dataclass(frozen=True)
class ImouCredentials:
    app_id: str
    app_secret: str = field(repr=False)     # never shown by repr/logging
    dc: str = "or"
    base_url: str | None = None      # IMOU_BASE_URL: point at a mock platform (bree.ingest.imou_mock) instead of the real one

    def __repr__(self) -> str:
        return f"ImouCredentials(app_id={self.app_id!r}, dc={self.dc!r})"


def load_credentials(environ: dict | None = None, env_file: str | Path | None = ENV_FILE) -> ImouCredentials:
    """IMOU_APP_ID / IMOU_APP_SECRET / IMOU_DC from the environment, falling back to the env file."""
    env = dict(os.environ if environ is None else environ)
    if env_file and Path(env_file).exists():
        for ln in Path(env_file).read_text().splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    missing = [k for k in ("IMOU_APP_ID", "IMOU_APP_SECRET") if not env.get(k)]
    if missing:
        raise ImouConfigError(f"missing {', '.join(missing)} (set them in the environment or {env_file})")
    return ImouCredentials(env["IMOU_APP_ID"], env["IMOU_APP_SECRET"], env.get("IMOU_DC", "or"), env.get("IMOU_BASE_URL"))


def sign(secret: str, ts: int, nonce: str) -> str:
    key = hashlib.sha256(secret.encode()).hexdigest()
    msg = f"time:{ts},nonce:{nonce},appSecret:{secret}"
    return base64.b64encode(hmac.new(key.encode(), msg.encode(), hashlib.sha256).digest()).decode()


def scrub(text, secrets=()) -> str:
    """Remove URLs (live addresses carry tokens) and known secrets before anything is logged or shown."""
    text = URL_RE.sub("<url>", str(text))
    for s in secrets:
        if s:
            text = text.replace(s, "<secret>")
    return text


@dataclass(frozen=True)
class ImouTarget:
    serial: str
    channel: int
    stream: str = "sub"        # "sub" (704x480, cheap) or "main" (up to 2560x1440 HEVC)
    params: dict = field(default_factory=dict, hash=False, compare=False)


def is_imou_url(src) -> bool:
    return isinstance(src, str) and src.startswith("imou://")


def parse_imou_url(url: str) -> ImouTarget:
    """imou://SERIAL/CHANNEL[?stream=sub|main&clock_offset=-7200&station=trumanrd&calibrate_clock=1]"""
    u = urllib.parse.urlsplit(url)
    if u.scheme != "imou":
        raise ImouConfigError(f"not an imou:// URL: {scrub(url)!r}")
    serial = u.netloc
    chan = u.path.strip("/")
    if not serial or not re.fullmatch(r"[A-Za-z0-9]+", serial) or not chan.isdigit():
        raise ImouConfigError("expected imou://<serial>/<channel>, e.g. imou://AH040CBPAZ893C5/7")
    q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
    stream = q.pop("stream", "sub")
    if stream not in STREAM_IDS:
        raise ImouConfigError(f"stream must be sub or main, got {stream!r}")
    if "@" in u.netloc or u.password or u.username:
        raise ImouConfigError("do not put credentials in the URL")
    return ImouTarget(serial, int(chan), stream, q)


# ----------------------------------------------------------------------------- API client

@dataclass
class DeviceStatus:
    online: bool
    channels: dict[int, bool]          # channel id -> online


class ImouClient:
    """Signed JSON calls to https://openapi-<dc>.easy4ip.com. `base_url` points tests at the mock server."""

    def __init__(self, creds: ImouCredentials, base_url: str | None = None, timeout: float = 30.0,
                 allow_bind: bool = False, clock: Callable[[], float] = time.time):
        self.creds = creds
        self.base_url = (base_url or creds.base_url or f"https://openapi-{creds.dc}.easy4ip.com:443").rstrip("/")
        self.timeout = timeout
        self.allow_bind = allow_bind
        self._clock = clock
        self._tok: str | None = None
        self._tok_exp = 0.0
        self._lock = threading.Lock()
        self.calls = 0

    # -- plumbing
    def _secrets(self) -> tuple:
        return (self.creds.app_secret, self._tok)

    def call(self, method: str, params: dict) -> dict:
        allowed = READ_ONLY | (BIND_METHODS if self.allow_bind else frozenset())
        if method not in allowed:
            raise ImouConfigError(f"API method {method!r} is not allowed (read-only client)")
        ts, nonce = int(self._clock()), str(uuid.uuid4())
        body = {"system": {"ver": "1.0", "appId": self.creds.app_id, "sign": sign(self.creds.app_secret, ts, nonce),
                           "time": ts, "nonce": nonce}, "id": str(uuid.uuid4()), "params": params}
        req = urllib.request.Request(f"{self.base_url}/openapi/{method}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                res = json.load(r)["result"]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            raise StreamUnavailable(scrub(f"{method}: {type(e).__name__}: {e}", self._secrets())) from None
        return res

    def _checked(self, method: str, params_fn: Callable[[str], dict], ok_codes=(CODE_OK,)) -> dict:
        """Call with a token; on an expired-token answer renew once and retry."""
        for attempt in (0, 1):
            res = self.call(method, params_fn(self.token(force=attempt == 1)))
            code = str(res.get("code"))
            if code in CODE_TOKEN_EXPIRED and attempt == 0:
                continue
            if code in ok_codes:
                return res
            if code in CODE_QUOTA:
                raise StreamStaleError(f"{method}: platform flow allowance exhausted ({code})", code)
            raise ImouError(scrub(f"{method} {code} {res.get('msg')}", self._secrets()), code)
        raise ImouAuthError(f"{method}: token rejected after renewal")

    # -- token
    def token(self, force: bool = False) -> str:
        with self._lock:
            if not force and self._tok and self._clock() < self._tok_exp - 3600:   # renew an hour early
                return self._tok
            r = self.call("accessToken", {})
            if str(r.get("code")) != CODE_OK:
                raise ImouAuthError(scrub(f"accessToken {r.get('code')} {r.get('msg')}", self._secrets()), str(r.get("code")))
            self._tok = r["data"]["accessToken"]
            self._tok_exp = self._clock() + int(r["data"].get("expireTime", 3600))
            return self._tok

    # -- read-only methods
    def device_online(self, serial: str) -> DeviceStatus:
        res = self._checked("deviceOnline", lambda t: {"token": t, "deviceId": serial})
        d = res.get("data") or {}
        chans = {}
        for c in d.get("channels") or []:
            chans[int(c["channelId"])] = str(c.get("onLine")) in ("1", "True", "true")
        return DeviceStatus(str(d.get("onLine")) in ("1", "True", "true"), chans)

    def live_url(self, serial: str, channel: int, stream: str = "sub") -> str:
        """Fresh HLS address for one channel. Addresses expire, so ask again instead of keeping them."""
        sid = STREAM_IDS[stream]
        self._checked("bindDeviceLive", lambda t: {"token": t, "deviceId": serial, "channelId": str(channel), "streamId": sid},
                      ok_codes=(CODE_OK, CODE_LIVE_EXISTS))
        res = self._checked("getLiveStreamInfo", lambda t: {"token": t, "deviceId": serial, "channelId": str(channel)})
        streams = [s for s in (res.get("data") or {}).get("streams", []) if s.get("hls")]
        want = [s for s in streams if s.get("streamId") == sid] or streams     # fall back to whatever is offered
        want.sort(key=lambda s: not s["hls"].startswith("https"))
        if not want:
            raise ImouError(f"no stream address for channel {channel}")
        return want[0]["hls"]

    def bind_device(self, serial: str, code: str) -> None:
        """Attach a new DVR (serial + the device's verification code/password) to the app. State-changing."""
        if not self.allow_bind:
            raise ImouConfigError("bind_device needs ImouClient(allow_bind=True)")
        self._checked("bindDevice", lambda t: {"token": t, "deviceId": serial, "code": code})


# ----------------------------------------------------------------------------- clock + health

@dataclass
class StationClock:
    """Maps this machine's wall clock to station (DVR overlay) time: station = wall + offset_s."""
    offset_s: float = 0.0
    source: str = "none"       # none | config | overlay

    def station_ts(self, wall: float) -> float:
        return wall + self.offset_s


@dataclass
class StampedFrame(Frame):
    station_ts: float = 0.0    # epoch seconds on the station (DVR) clock: wall + clock offset


class HealthSink:
    """Heartbeat file for one camera, read by bree.fleet. Written atomically at most every `every_s` seconds."""

    def __init__(self, path: str | Path | None, station: str, serial: str, channel: int, every_s: float = 5.0):
        self.path = Path(path) if path else None
        self.d = {"station": station, "serial": serial, "channel": channel, "state": "starting", "frames": 0,
                  "last_frame_wall": None, "last_frame_station_ts": None, "reconnects": 0, "stub_opens": 0,
                  "consecutive_failures": 0, "last_error": None, "started": time.time(), "pid": os.getpid(),
                  "width": None, "height": None, "fps": None, "stream": None, "clock_offset_s": None}
        self.every_s, self._last = every_s, 0.0

    def update(self, force: bool = False, **kw) -> None:
        self.d.update(kw)
        now = time.time()
        if self.path is None or (not force and now - self._last < self.every_s):
            return
        self._last = now
        self.d["updated"] = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------------- live source

class ImouSource:
    """Frame source with the same interface as bree.ingest.source.VideoSource (`fps`, `width`, `height`, `live`,
    iteration over Frame), so the pipeline, `bree run` and shadow mode treat it like any camera.

    `t` is seconds since the source started (wall clock); each frame is a StampedFrame whose `station_ts` also
    carries the station clock. The first open happens in the constructor, so a wrong serial, an offline
    device or a stale stream fails at start-up, not on the first frame.
    """

    live = True

    def __init__(self, target: str | ImouTarget, client: ImouClient | None = None, creds: ImouCredentials | None = None,
                 clock: StationClock | None = None, health: HealthSink | None = None, max_frames: int | None = None,
                 min_run_s: float = 8.0, max_stub_retries: int = 3, max_wait_s: float = 600.0, backoff_cap_s: float = 30.0,
                 read_timeout_s: float = 20.0, check_online: bool = True, sleep: Callable[[float], None] = time.sleep):
        self.target = parse_imou_url(target) if isinstance(target, str) else target
        if client is None:
            client = ImouClient(creds or load_credentials())
        self.client = client
        p = self.target.params
        if clock is None:
            clock = StationClock(float(p["clock_offset"]), "config") if "clock_offset" in p else StationClock()
        self.clock = clock
        if health is None:
            path = None
            if p.get("station"):      # imou://...?station=<id>: heartbeat where `bree fleet status` looks for it
                from bree.fleet.registry import Fleet
                try:
                    path = Fleet().health_path(p["station"], self.target.channel)
                except ValueError as e:
                    raise ImouConfigError(str(e)) from None
            health = HealthSink(path, p.get("station", ""), self.target.serial, self.target.channel)
        self.health = health
        self.max_frames, self.min_run_s, self.max_stub_retries = max_frames, min_run_s, max_stub_retries
        self.max_wait_s, self.backoff_cap_s, self.read_timeout_s, self._sleep = max_wait_s, backoff_cap_s, read_timeout_s, sleep
        self.reconnect = True
        self.stub_opens = 0
        self.cap: cv2.VideoCapture | None = None
        self._opened_at = 0.0
        self._run_frames = 0
        if check_online:
            st = client.device_online(self.target.serial)
            if not st.online:
                self.health.update(True, state="offline", last_error="device offline")
                raise DeviceOfflineError(f"device {self.target.serial} is offline")
            if st.channels and not st.channels.get(self.target.channel, True):
                self.health.update(True, state="offline", last_error=f"channel {self.target.channel} offline")
                raise DeviceOfflineError(f"channel {self.target.channel} of {self.target.serial} is offline")
        self._open()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 0 < fps < 240 else 15.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if p.get("calibrate_clock") in ("1", "true"):
            self.calibrate_clock()
        self.health.update(True, state="ok", width=self.width, height=self.height, fps=self.fps, stream=self.target.stream,
                           clock_offset_s=self.clock.offset_s)

    # -- connection
    def _open(self) -> None:
        """Ask the platform for a fresh address and open it. Raises ImouError subclasses with scrubbed text."""
        url = self.client.live_url(self.target.serial, self.target.channel, self.target.stream)
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.read_timeout_s * 1000),
                                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_s * 1000)])
        if not self.cap.isOpened():
            self.cap = None
            raise StreamUnavailable("live address did not open")
        self._opened_at, self._run_frames = time.time(), 0

    def calibrate_clock(self, n: int = 5, step_s: float = 1.0):
        """Read the DVR clock burned into n frames (about step_s apart) and set the station clock offset to
        overlay time minus this machine's local time. The estimate includes the stream latency (a few seconds
        for HLS). Frames used here are not delivered. Leaves the clock alone if the overlay is unreadable."""
        from bree.ingest.overlay import DAY, estimate_offset, read_clock
        samples, skip = [], max(1, int(self.fps * step_s))
        for _ in range(n):
            img = None
            for _ in range(skip):
                ok, f = self.cap.read()
                if not ok:
                    break
                img = f
            if img is None:
                break
            lt = time.localtime()
            samples.append(((lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec) % DAY, read_clock(img)))
        est = estimate_offset(samples)
        if est.ok:
            self.clock = StationClock(est.offset_s, "overlay")
        return est

    def _stream_ended(self) -> None:
        """The stream stopped giving frames. A run shorter than min_run_s of media is a stub, not a hiccup."""
        media_s = self._run_frames / max(self.fps, 1.0)
        if media_s < self.min_run_s:
            self.stub_opens += 1
            self.health.update(True, stub_opens=self.stub_opens, state="stub")
            if self.stub_opens >= self.max_stub_retries:
                msg = (f"channel {self.target.channel}: {self.stub_opens} live addresses in a row ended after "
                       f"< {self.min_run_s:.0f} s of video: a stub stream (flow allowance used up or a stale live "
                       "session on the platform). Not retrying: check the Imou account.")
                self.health.update(True, state="stale_stream", last_error=msg)
                raise StreamStaleError(msg)
        else:
            self.stub_opens = 0

    def __iter__(self) -> Iterator[Frame]:
        i, t0 = 0, time.time()
        fail_since, backoff = None, 1.0
        while self.max_frames is None or i < self.max_frames:
            ok, img = (self.cap.read() if self.cap is not None else (False, None))
            if ok:
                if fail_since is not None:
                    fail_since, backoff = None, 1.0
                    self.health.update(True, state="ok", consecutive_failures=0)
                self._run_frames += 1
                now = time.time()
                self.health.update(frames=i + 1, last_frame_wall=now, last_frame_station_ts=self.clock.station_ts(now))
                yield StampedFrame(i, now - t0, img, now, None, self.clock.station_ts(now))
                i += 1
                continue
            if not self.reconnect:
                break
            if self.cap is not None:
                self._stream_ended()          # may raise StreamStaleError
            fail_since = fail_since or time.time()
            if time.time() - fail_since > self.max_wait_s:
                msg = f"no video for {self.max_wait_s:.0f} s (channel {self.target.channel})"
                self.health.update(True, state="down", last_error=msg)
                raise StreamUnavailable(msg)
            self._sleep(backoff)
            backoff = min(backoff * 2, self.backoff_cap_s)
            self.health.update(True, reconnects=self.health.d["reconnects"] + 1, state="reconnecting")
            try:
                self._open()
            except StreamStaleError:
                raise
            except (ImouError, OSError) as e:
                self.health.update(True, consecutive_failures=self.health.d["consecutive_failures"] + 1,
                                   last_error=scrub(e, self.client._secrets()))
        self.health.update(True, state="stopped")
        if self.cap is not None:
            self.cap.release()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
