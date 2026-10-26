"""Pre-flight and runtime health: clock, disk, credential files, folder permissions, retention, audit chain.

`doctor()` returns a list of checks {name, status: ok|warn|fail, detail}; `bree trust doctor` prints them and exits
non-zero on any fail. A systemd timer runs it every few minutes (deploy/systemd/bree-doctor.timer) and the pipeline
unit's ExecStartPre runs it once, so a box with a dead clock or a full disk does not start recording.

Disk-full behaviour (`DiskGuard`): below `low_gb` free, purge evictable footage early (oldest first) and keep going;
below `critical_gb`, stop writing video and images (alerts.jsonl and the ledger still write: they are KBs) until space
returns, and say so in the status file. The pipeline asks `guard.allow_video()` before opening a writer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from bree.trust import secrets as _secrets


def _chk(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def clock_check(max_offset_s: float = 2.0) -> dict:
    """Is the system clock synchronised? Reads timedatectl (systemd) then chronyc; wall time matters for joining video
    to POS receipts, so a drifting clock silently breaks the ledger."""
    try:
        out = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "-p", "Timezone", "-p", "NTP"],
                             capture_output=True, text=True, timeout=5).stdout
        kv = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
        if kv.get("NTPSynchronized") == "yes":
            return _chk("clock", "ok", f"NTP synchronised, timezone {kv.get('Timezone', '?')}")
        if kv:
            return _chk("clock", "fail", f"NTP not synchronised (NTP service {kv.get('NTP', '?')}): enable chrony or "
                                         "systemd-timesyncd; receipts will not line up with video")
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        out = subprocess.run(["chronyc", "tracking"], capture_output=True, text=True, timeout=5).stdout
        for ln in out.splitlines():
            if ln.startswith("System time"):
                off = float(ln.split(":")[1].split()[0])
                return _chk("clock", "ok" if off <= max_offset_s else "fail", f"chrony offset {off:.3f} s")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return _chk("clock", "warn", "cannot tell whether the clock is synchronised (no timedatectl or chronyc)")


def device_clock_check(device_epoch: float, now: float | None = None, max_offset_s: float = 5.0) -> dict:
    """Compare a camera/DVR clock (e.g. parsed from its overlay) with this machine's. Offsets that are whole hours
    mean a timezone difference, not drift: the Truman Rd DVR overlay read about 2 h behind the recorder's clock."""
    now = time.time() if now is None else now
    off = device_epoch - now
    if abs(off) <= max_offset_s:
        return _chk("device_clock", "ok", f"offset {off:+.1f} s")
    hours = off / 3600
    zone = f" (about {round(hours)} h: a timezone setting, record it as the station's clock offset)" \
        if abs(hours - round(hours)) < 0.05 and round(hours) else ""
    return _chk("device_clock", "warn", f"offset {off:+.0f} s{zone}")


def disk_check(path: str | Path, low_gb: float = 20, critical_gb: float = 5) -> dict:
    free = shutil.disk_usage(path).free / 1e9
    st = "ok" if free >= low_gb else "warn" if free >= critical_gb else "fail"
    return _chk("disk", st, f"{free:.1f} GB free at {path} (low {low_gb}, critical {critical_gb})")


def credential_check(paths: list[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            out.append(_chk(f"credential {p.name}", "warn", "not present"))
            continue
        probs = _secrets.file_problems(p)
        out.append(_chk(f"credential {p.name}", "fail" if probs else "ok", "; ".join(probs) or "owner-only (0600)"))
    return out


def folder_check(paths: list[str | Path]) -> list[dict]:
    """Footage and evidence folders must not be readable by other users (mode 0700/0750, never world-readable)."""
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if not p.exists():
            continue
        mode = p.stat().st_mode & 0o777
        out.append(_chk(f"folder {p.name}", "fail" if mode & 0o007 else "ok", f"mode {mode:04o}"))
    return out


def doctor(data_dir: str | Path | None = None, credentials: list | None = None, audit_path: str | Path | None = None,
           retention_policy: str | Path | None = None) -> list[dict]:
    checks = [clock_check()]
    if data_dir:
        checks.append(disk_check(data_dir))
        checks += folder_check([data_dir])
    checks += credential_check(credentials or [Path.home() / ".config/bree/imou.env"])
    if audit_path and Path(audit_path).exists():
        from bree.trust.audit import AuditLog
        ok, msg = AuditLog(audit_path).verify()
        checks.append(_chk("audit chain", "ok" if ok else "fail", msg))
    if retention_policy and Path(retention_policy).exists():
        from bree.trust.retention import Policy, plan
        try:
            items = plan(Policy.load(retention_policy))
            checks.append(_chk("retention", "ok" if len(items) < 5000 else "warn",
                               f"{len(items)} files / {sum(i['bytes'] for i in items) / 1e9:.2f} GB are past their limit "
                               "(the purge timer should keep this near zero)"))
        except (OSError, ValueError, KeyError) as e:
            checks.append(_chk("retention", "fail", f"policy unusable: {e}"))
    return checks


def worst(checks: list[dict]) -> str:
    return "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"


@dataclass
class DiskGuard:
    """Disk-full policy for the process that writes footage. Cheap: one statvfs per `every_s`."""
    path: str
    low_gb: float = 20
    critical_gb: float = 5
    every_s: float = 30
    on_low: object = None            # callable run when free space drops under low_gb (e.g. an early purge)
    _t: float = 0.0
    _free: float = 1e9
    state: str = "ok"

    def _refresh(self) -> None:
        now = time.time()
        if now - self._t < self.every_s:
            return
        self._t = now
        self._free = shutil.disk_usage(self.path).free / 1e9
        prev = self.state
        self.state = "ok" if self._free >= self.low_gb else "low" if self._free >= self.critical_gb else "critical"
        if self.state != "ok" and prev == "ok" and callable(self.on_low):
            self.on_low()

    def allow_video(self) -> bool:
        """False at critical: do not open new video/image writers; alerts.jsonl and the ledger keep working."""
        self._refresh()
        return self.state != "critical"

    def status(self) -> dict:
        self._refresh()
        return {"disk_state": self.state, "disk_free_gb": round(self._free, 1)}


def write_status(path: str | Path, **fields) -> None:
    """Atomic small status file for the dashboard / doctor timer (no secrets)."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated": time.time(), **fields}, indent=1))
    os.replace(tmp, p)
