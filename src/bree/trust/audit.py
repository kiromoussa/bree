"""Append-only audit log: who or what touched footage, evidence, exports or configuration, and when.

One JSON line per record, each carrying the SHA-256 of the previous line (a hash chain), so editing or deleting a
line in the middle is detected by `verify()`. Truncating the tail is detected only if the last hash was kept
elsewhere (`bree trust audit --head` prints it; the daily digest should include it). The log never holds image
content, plate text, tokens or passwords: only the actor, the action, a target name and small counters.

Actions in use: serve_clip, serve_frame, export, label_export, purge, purge_dry_run, raw_record_start, config_change,
credential_use, login_fail, retention_hold. Free text is allowed in `detail`, but values that look like a secret or a
URL with a query string are replaced (`scrub`).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

GENESIS = "0" * 64
_SECRETISH = re.compile(r"(?i)(secret|token|password|passwd|authorization|api[_-]?key|sign)")
_URL = re.compile(r"(\w+://[^\s?\"']+)\?[^\s\"']*")


def scrub(v):
    """Drop anything secret-looking from a value that is about to be logged."""
    if isinstance(v, dict):
        return {k: ("<redacted>" if _SECRETISH.search(str(k)) else scrub(x)) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [scrub(x) for x in v]
    if isinstance(v, str):
        return _URL.sub(r"\1?<redacted>", v)
    return v


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last = self._tail_hash()
        if not self.path.exists():
            self.path.touch(mode=0o600)

    def _tail_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        with self.path.open("rb") as f:
            f.seek(max(0, self.path.stat().st_size - 8192))
            last = [ln for ln in f.read().splitlines() if ln.strip()][-1]
        return json.loads(last)["hash"]

    def record(self, actor: str, action: str, target: str = "", **detail) -> dict:
        rec = {"ts": round(time.time(), 3), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "actor": str(actor),
               "action": action, "target": scrub(str(target)), "detail": scrub(detail), "pid": os.getpid()}
        with self._lock:
            rec["prev"] = self._last
            body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
            rec["hash"] = hashlib.sha256(body.encode()).hexdigest()
            with self.path.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
            self._last = rec["hash"]
        return rec

    @property
    def head(self) -> str:
        return self._last

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def verify(self, expect_head: str | None = None) -> tuple[bool, str]:
        """(ok, message). Recomputes every hash; with expect_head also checks the log was not truncated."""
        prev = GENESIS
        for i, rec in enumerate(self.read()):
            h = rec.pop("hash", None)
            if rec.get("prev") != prev:
                return False, f"line {i + 1}: chain broken (a line was edited, removed or reordered)"
            if hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != h:
                return False, f"line {i + 1}: content does not match its hash"
            prev = h
        if expect_head is not None and expect_head != prev:
            return False, "head hash differs from the one recorded earlier (log truncated or appended elsewhere)"
        return True, f"ok, {len(self.read())} records, head {prev[:12]}"


class NullAudit:
    """Stand-in when auditing is off (tests, simulations)."""
    head = GENESIS

    def record(self, *a, **k) -> dict:
        return {}
