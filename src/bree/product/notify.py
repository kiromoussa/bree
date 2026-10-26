"""Threshold rules and pluggable notification sinks.

Nothing here talks to the network. A `WebhookSink` only *records* what it would POST in an outbox file
(`dry_run` is the only mode implemented): wiring a real transport is a deliberate, separate step.

A `Rule` watches one signal kind and fires when the condition has held for `for_s` seconds. After
firing it stays quiet for `cooldown_s` (so a long queue is one notification, not one per sample).
"""
from __future__ import annotations

import json
import logging
import operator
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bree.product.signals import Signal

log = logging.getLogger("bree.product.notify")
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}
SEVERITIES = ("info", "warning", "urgent")


@dataclass
class Rule:
    name: str
    kind: str                       # signal kind watched
    op: str = ">="
    threshold: float = 1.0
    for_s: float = 0.0              # condition must hold this long
    cooldown_s: float = 600.0
    severity: str = "warning"
    message: str = ""               # {value} {zone} {camera} {for_s} are filled in
    zone: str | None = None         # None = any zone
    state: str | None = None        # for kind 'shelf': match meta.state (e.g. 'empty')
    scope: str = "camera"           # 'camera': one clock per camera+zone; 'store': max over cameras

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"op must be one of {sorted(OPS)}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")


# Plain-language defaults for a convenience store. Thresholds are starting points, not measured.
DEFAULT_RULES = [
    Rule("long_queue", "queue", ">=", 3, for_s=60, cooldown_s=900, severity="warning", scope="store",
         message="{value:.0f} people have been waiting at the register for {for_s:.0f}s. Open a second register?"),
    Rule("crowded", "occupancy", ">=", 15, for_s=120, cooldown_s=1800, severity="info", scope="store",
         message="The store has had {value:.0f} people inside for {for_s:.0f}s."),
    Rule("cooler_open", "shelf", ">=", 0, for_s=180, cooldown_s=1800, severity="warning", state="door_open",
         message="A cooler door ({zone}, {camera}) looks open for {for_s:.0f}s."),
    Rule("shelf_empty", "shelf", ">=", 0, for_s=600, cooldown_s=3600, severity="info", state="empty",
         message="{zone} ({camera}) looks empty."),
    Rule("camera_silent", "health", "<=", 0, for_s=120, cooldown_s=1800, severity="urgent",
         message="Camera {camera} has sent nothing for {for_s:.0f}s."),
]


@dataclass
class Notification:
    ts: float
    rule: str
    severity: str
    message: str
    camera: str | None = None
    zone: str | None = None
    value: float | None = None
    source: str = "live"

    def to_dict(self) -> dict:
        return asdict(self)


class Sink:
    def send(self, n: Notification) -> None:
        raise NotImplementedError


class LogSink(Sink):
    """Writes to the python logger (stderr by default)."""
    def send(self, n):
        log.log({"info": logging.INFO, "warning": logging.WARNING, "urgent": logging.ERROR}[n.severity],
                "[%s] %s", n.rule, n.message)


class FileSink(Sink):
    """Appends one JSON line per notification; the dashboard's bell reads this file."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def send(self, n):
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(n.to_dict()) + "\n")

    def read(self, limit: int = 50) -> list[dict]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-limit * 3:]:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        return rows[-limit:]


class WebhookSink(Sink):
    """STUB: records the request it would make (url + JSON body) to an outbox file. Never opens a socket."""
    def __init__(self, url: str, outbox: str | Path):
        self.url = url
        self.file = FileSink(outbox)

    def send(self, n):
        self.file.send(Notification(**{**n.to_dict(), "message": n.message}))
        with open(Path(self.file.path).with_suffix(".webhook.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"would_post_to": self.url, "body": n.to_dict(), "sent": False}) + "\n")


class Notifier:
    """Feeds signals through the rules; `feed()` returns the notifications it fired and hands each to every sink."""
    def __init__(self, rules=None, sinks=None, store_tz=None):
        self.rules = list(DEFAULT_RULES if rules is None else rules)
        self.sinks = list(sinks or [])
        self._since: dict[tuple, float] = {}          # (rule, key) -> ts condition started holding
        self._last_fire: dict[tuple, float] = {}
        self._seen: dict[tuple, float] = {}
        self._pending: dict[tuple, dict[str, float]] = {}   # store scope: camera -> latest value

    MAX_GAP_S = 900.0     # a condition does not "keep holding" across a silence longer than this

    def _applies(self, r: Rule, s: Signal) -> bool:
        return s.kind == r.kind and (r.zone is None or s.zone == r.zone)

    def feed(self, signals) -> list[Notification]:
        fired: list[Notification] = []
        for s in sorted(signals, key=lambda x: x.ts):
            for r in self.rules:
                if not self._applies(r, s):
                    continue
                if r.scope == "store":
                    cams = self._pending.setdefault((r.name,), {})
                    cams[s.camera] = s.value
                    value, key = max(cams.values()), (r.name, "store", None)
                else:
                    value, key = s.value, (r.name, s.camera, s.zone)
                if r.state is not None:      # the state match is the condition; `value` is just fullness
                    holds = s.meta.get("state") == r.state
                else:
                    holds = OPS[r.op](value, r.threshold)
                seen = self._seen.get(key)
                self._seen[key] = s.ts
                if not holds or (seen is not None and s.ts - seen > self.MAX_GAP_S):
                    self._since.pop(key, None)
                    if not holds:
                        continue
                t0 = self._since.setdefault(key, s.ts)
                if s.ts - t0 >= r.for_s and s.ts - self._last_fire.get(key, -1e18) >= r.cooldown_s:
                    self._last_fire[key] = s.ts
                    msg = (r.message or r.name).format(value=value, zone=s.zone or "", camera=s.camera, for_s=s.ts - t0)
                    fired.append(Notification(s.ts, r.name, r.severity, msg, s.camera, s.zone, value, s.source))
        for n in fired:
            for sink in self.sinks:
                try:
                    sink.send(n)
                except Exception:      # a broken sink must never stop the others or the pipeline
                    log.exception("sink %s failed", type(sink).__name__)
        return fired

    def check_silence(self, last_ts_by_camera: dict[str, float], now: float, max_silence_s: float = 120.0) -> list[Notification]:
        """Silence is the absence of a signal, so it needs the wall clock, not a feed. Fires per camera."""
        out = []
        for cam, last in last_ts_by_camera.items():
            key = ("camera_silent", cam, None)
            if now - last >= max_silence_s and now - self._last_fire.get(key, -1e18) >= 1800:
                self._last_fire[key] = now
                out.append(Notification(now, "camera_silent", "urgent",
                                        f"Camera {cam} has sent nothing for {now - last:.0f}s.", cam, None, 0.0))
        for n in out:
            for sink in self.sinks:
                try:
                    sink.send(n)
                except Exception:
                    log.exception("sink %s failed", type(sink).__name__)
        return out
