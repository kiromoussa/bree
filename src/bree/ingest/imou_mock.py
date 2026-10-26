"""A local fake of the Imou Open Platform, for tests and offline demos (no network, no credentials, no metered flow).

It speaks the same signed-JSON protocol as bree.ingest.imou.ImouClient for accessToken, deviceOnline, bindDevice,
bindDeviceLive and getLiveStreamInfo, verifies the signature, hands out HLS addresses that point back at itself
and can misbehave on request:

  mock.expire_token()               next call with the old token gets TK1002 (client must renew)
  mock.set_channel(5, "offline")    deviceOnline reports channel 5 offline
  mock.set_channel(5, "stub")       live address is a 3 s stub playlist (what the platform served on 2026-10-25
                                    once the free flow allowance ran out)
  mock.quota_exhausted = True       every channel serves the stub
  mock.device_online = False        whole DVR offline

The video is generated with ffmpeg (a moving test pattern per channel) unless `video` (an mp4 path) is given.
It is a protocol double: it says nothing about how the real platform behaves beyond what the repo has seen.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bree.ingest.imou import sign

APP_ID, APP_SECRET = "mock-app", "mock-secret-not-real"


def _hls(out: Path, seconds: int, size: str, rate: int, src_video: str | None, pattern: str = "testsrc") -> None:
    out.mkdir(parents=True, exist_ok=True)
    inp = ["-i", src_video] if src_video else ["-f", "lavfi", "-i", f"{pattern}=size={size}:rate={rate}"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inp, "-t", str(seconds), "-c:v", "libx264", "-g", str(rate),
                    "-pix_fmt", "yuv420p", "-f", "hls", "-hls_time", "2", "-hls_list_size", "0", "-hls_playlist_type", "vod",
                    str(out / "index.m3u8")], check=True)


class MockImou:
    def __init__(self, serial: str = "MOCKSERIAL001", channels: dict[int, str] | None = None, live_seconds: int = 12,
                 size: str = "320x240", rate: int = 10, video: str | None = None, device_code: str = "mock-code"):
        self.serial, self.device_code = serial, device_code
        self.channels = dict(channels or {0: "ok", 1: "ok"})     # channel id -> ok | offline | stub
        self.device_online = True
        self.bound = True
        self.quota_exhausted = False
        self.token = None
        self.token_ttl = 7200
        self.log: list[dict] = []                                # {"method", "code"}: never tokens or secrets
        self._lock = threading.Lock()
        self._tmp = Path(tempfile.mkdtemp(prefix="imou_mock_"))
        _hls(self._tmp / "live", live_seconds, size, rate, video)
        _hls(self._tmp / "stub", 3, size, rate, None, "smptebars")
        mock = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                method = self.path.rsplit("/", 1)[-1]
                res = mock._handle(method, body)
                with mock._lock:
                    mock.log.append({"method": method, "code": res["code"]})
                data = json.dumps({"result": res, "id": body.get("id")}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                p = (mock._tmp / self.path.lstrip("/").split("?")[0]).resolve()
                if mock._tmp.resolve() not in p.parents or not p.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.apple.mpegurl" if p.suffix == ".m3u8" else "video/mp2t")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    # -- controls
    def expire_token(self) -> None:
        self.token = None

    def set_channel(self, ch: int, mode: str) -> None:
        self.channels[ch] = mode

    def methods(self) -> list[str]:
        return [e["method"] for e in self.log]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- protocol
    def _handle(self, method: str, body: dict) -> dict:
        sysd, params = body.get("system", {}), body.get("params", {})
        if sysd.get("appId") != APP_ID or sysd.get("sign") != sign(APP_SECRET, sysd.get("time", 0), sysd.get("nonce", "")):
            return {"code": "SN1001", "msg": "sign error"}
        if method == "accessToken":
            self.token = "tok-" + uuid.uuid4().hex
            return {"code": "0", "msg": "success", "data": {"accessToken": self.token, "expireTime": self.token_ttl}}
        if self.token is None or params.get("token") != self.token:
            return {"code": "TK1002", "msg": "token expired"}
        if params.get("deviceId") != self.serial:
            return {"code": "DV1001", "msg": "device not found"}
        if method == "deviceOnline":
            return {"code": "0", "msg": "success", "data": {
                "onLine": "1" if self.device_online else "0",
                "channels": [{"channelId": str(c), "onLine": "0" if m == "offline" else "1"} for c, m in sorted(self.channels.items())]}}
        if method == "bindDevice":
            if params.get("code") != self.device_code:
                return {"code": "DV1007", "msg": "wrong device code"}
            self.bound = True
            return {"code": "0", "msg": "success"}
        if method in ("bindDeviceLive", "getLiveStreamInfo"):
            ch = int(params.get("channelId", -1))
            if ch not in self.channels or self.channels[ch] == "offline" or not self.device_online:
                return {"code": "LV1002", "msg": "channel offline"}
            if method == "bindDeviceLive":
                return {"code": "LV1001", "msg": "live already exists"} if getattr(self, "_live", None) == ch else \
                    self._set_live(ch)
            mode = "stub" if (self.quota_exhausted or self.channels[ch] == "stub") else "live"
            u = f"{self.base_url}/{mode}/index.m3u8"
            return {"code": "0", "msg": "success", "data": {"streams": [
                {"streamId": 0, "hls": u, "coverUrl": ""},
                {"streamId": 1, "hls": u, "coverUrl": ""}]}}
        return {"code": "OP1000", "msg": "unknown method"}

    def _set_live(self, ch: int) -> dict:
        self._live = ch
        return {"code": "0", "msg": "success"}
