#!/usr/bin/env python3
"""Record rolling segments from the store DVR through the Imou Open Platform (read-only APIs only).

Per channel, one loop: renew token if needed -> bindDeviceLive -> getLiveStreamInfo -> ffmpeg -c copy of the main
stream for SEG_SECONDS -> write to <name>.part, rename to YYYYmmdd_HHMMSS.mp4 only when finished.
Credentials come from ~/.config/bree/imou.env (or the environment); they are never logged or written out.
Footage goes under ~/Projects/bree-data/real/<station>/chNN/ (outside every git repo). Video only, no audio.
Stdlib only. Status: <data>/recorder_status.json. Stop with SIGTERM (or touch <data>/recorder.stop).
"""
import argparse, base64, hashlib, hmac, json, os, re, signal, subprocess, sys, threading, time, urllib.request, uuid
from datetime import datetime
from pathlib import Path

ENV_FILE = Path.home() / ".config/bree/imou.env"
DEFAULT_CHANNELS = [7, 29, 12, 14, 15, 10, 20]
URL_RE = re.compile(r"\S+://\S+")
STOP = threading.Event()


def load_env():
    env = dict(os.environ)
    if ENV_FILE.exists():
        for ln in ENV_FILE.read_text().splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip("'\""))
    return env


def scrub(text, secrets=()):
    """Remove URLs (live addresses carry tokens) and any known secret from text before it is logged."""
    text = URL_RE.sub("<url>", str(text))
    for s in secrets:
        if s:
            text = text.replace(s, "<secret>")
    return text


def sign(secret, ts, nonce):
    key = hashlib.sha256(secret.encode()).hexdigest()
    msg = f"time:{ts},nonce:{nonce},appSecret:{secret}"
    return base64.b64encode(hmac.new(key.encode(), msg.encode(), hashlib.sha256).digest()).decode()


class Imou:
    """Minimal client. Only these four read-only methods are ever called."""
    ALLOWED = {"accessToken", "deviceOnline", "bindDeviceLive", "getLiveStreamInfo"}

    def __init__(self, env):
        self.dc, self.app_id, self.secret, self.dev = env["IMOU_DC"], env["IMOU_APP_ID"], env["IMOU_APP_SECRET"], env["DEVICE_SERIAL"]
        self._tok, self._exp, self._lock = None, 0, threading.Lock()

    def call(self, method, params):
        assert method in self.ALLOWED, method
        ts, nonce = int(time.time()), str(uuid.uuid4())
        body = {"system": {"ver": "1.0", "appId": self.app_id, "sign": sign(self.secret, ts, nonce), "time": ts, "nonce": nonce},
                "id": str(uuid.uuid4()), "params": params}
        req = urllib.request.Request(f"https://openapi-{self.dc}.easy4ip.com:443/openapi/{method}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=30))["result"]

    def token(self):
        with self._lock:
            if self._tok and time.time() < self._exp - 3600:
                return self._tok
            r = self.call("accessToken", {})
            if r.get("code") != "0":
                raise RuntimeError(f"accessToken {r.get('code')} {r.get('msg')}")
            self._tok = r["data"]["accessToken"]
            self._exp = time.time() + int(r["data"].get("expireTime", 3600))
            return self._tok

    def live_url(self, ch):
        """Fresh main-stream (streamId 0) HLS address for one channel."""
        tok = self.token()
        self.call("bindDeviceLive", {"token": tok, "deviceId": self.dev, "channelId": str(ch), "streamId": 0})  # LV1001 = already live: fine
        r = self.call("getLiveStreamInfo", {"token": tok, "deviceId": self.dev, "channelId": str(ch)})
        if r.get("code") != "0":
            raise RuntimeError(f"getLiveStreamInfo {r.get('code')} {r.get('msg')}")
        streams = [s for s in (r.get("data") or {}).get("streams", []) if s.get("streamId") == 0 and s.get("hls")]
        streams.sort(key=lambda s: not s["hls"].startswith("https"))
        if not streams:
            raise RuntimeError("no main stream address")
        return streams[0]["hls"]


class Status:
    def __init__(self, path, channels):
        self.path, self.lock = path, threading.Lock()
        self.d = {"started": time.time(), "pid": os.getpid(), "updated": 0, "total_bytes": 0, "channels":
                  {str(c): {"last_segment": None, "last_segment_time": None, "segments": 0, "consecutive_failures": 0, "errors": []} for c in channels}}

    def update(self, ch, ok=None, seg=None, err=None):
        with self.lock:
            c = self.d["channels"][str(ch)]
            if ok:
                c.update(last_segment=seg, last_segment_time=time.time(), consecutive_failures=0)
                c["segments"] += 1
            if err:
                c["consecutive_failures"] += 1
                c["errors"] = (c["errors"] + [{"t": time.strftime("%F %T"), "e": err}])[-5:]
            self.write()

    def write(self):
        self.d["updated"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        os.replace(tmp, self.path)


def total_bytes(root):
    return sum(f.stat().st_size for f in root.glob("ch*/*.mp4"))


def enforce_cap(root, cap_bytes, log):
    files = sorted(root.glob("ch*/*.mp4"), key=lambda f: f.name[:15] + f.parent.name)  # oldest first (names are timestamps)
    total = sum(f.stat().st_size for f in files)
    while total > cap_bytes and files:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink()
        log(f"cap: deleted {f.parent.name}/{f.name}")
    return total


def record_loop(ch, api, root, status, args, log, deadline):
    out_dir = root / f"ch{ch:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    backoff = 5
    while not STOP.is_set() and time.time() < deadline:
        t0 = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        part, final = out_dir / f"{stamp}.mp4.part", out_dir / f"{stamp}.mp4"
        dur = int(min(args.seg_seconds, max(10, deadline - t0)))
        try:
            url = api.live_url(ch)
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-rw_timeout", "20000000", "-i", url,
                   "-map", "0:v:0", "-an", "-c", "copy", "-t", str(dur), "-f", "mp4", "-movflags", "+faststart", "-y", str(part)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=dur + 120)
            size = part.stat().st_size if part.exists() else 0
            if p.returncode != 0 and size < 100_000:
                raise RuntimeError(f"ffmpeg rc={p.returncode} {scrub(p.stderr.strip()[-200:])}")
            if size < 100_000:
                raise RuntimeError(f"segment too small ({size} bytes)")
            os.replace(part, final)  # readers only ever see finished files
            status.update(ch, ok=True, seg=final.name)
            backoff = 5
            if time.time() - t0 < dur * 0.5:  # ended much early: stream cut; short pause so we do not spin
                STOP.wait(5)
            with status.lock:
                status.d["total_bytes"] = enforce_cap(root, args.cap_gb * 1e9, log)
        except Exception as e:
            msg = scrub(f"{type(e).__name__}: {e}", [api.secret])
            log(f"ch{ch:02d} error: {msg} (retry in {backoff}s)")
            status.update(ch, err=msg)
            if part.exists():
                part.unlink()
            STOP.wait(backoff)
            backoff = min(backoff * 2, 300)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--station", default="trumanrd")
    ap.add_argument("--data", default=str(Path.home() / "Projects/bree-data"))
    ap.add_argument("--channels", default=",".join(map(str, DEFAULT_CHANNELS)))
    ap.add_argument("--seg-seconds", type=int, default=300)
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--cap-gb", type=float, default=80)
    args = ap.parse_args()
    channels = [int(c) for c in args.channels.split(",")]
    assert len(channels) <= 8, "at most 8 concurrent streams"
    data = Path(args.data)
    root = data / "real" / args.station
    log_f = open(data / "recorder.log", "a", buffering=1)
    os.chmod(data / "recorder.log", 0o600)

    def log(m):
        log_f.write(f"{time.strftime('%F %T')} {m}\n")

    env = load_env()
    api = Imou(env)
    status = Status(data / "recorder_status.json", channels)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: STOP.set())
    deadline = time.time() + args.hours * 3600
    log(f"start pid={os.getpid()} channels={channels} seg={args.seg_seconds}s hours={args.hours} cap={args.cap_gb}GB")
    status.write()
    ths = []
    for ch in channels:
        t = threading.Thread(target=record_loop, args=(ch, api, root, status, args, log, deadline), daemon=True)
        t.start()
        ths.append(t)
        STOP.wait(2)  # stagger starts
    while not STOP.is_set() and time.time() < deadline and any(t.is_alive() for t in ths):
        if (data / "recorder.stop").exists():
            STOP.set()
        STOP.wait(5)
    STOP.set()
    for t in ths:
        t.join(timeout=30)  # ffmpeg children finish their segment or time out; unfinished .part files are removed
    for f in root.glob("ch*/*.part"):
        f.unlink()
    log("exit")


if __name__ == "__main__":
    main()
