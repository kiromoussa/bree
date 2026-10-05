"""Plate data store with retention limits and automatic deletion.

What is kept and for how long:
- A plate read of a vehicle that paid is never written anywhere: it lives in memory in the
  DriveOffMonitor and is dropped a few minutes after the vehicle leaves.
- A plate read attached to a drive-off review alert is written here (text, confidence, a small
  plate crop). It is deleted `retention_hours` after the event (default 72) unless a reviewer
  confirms the drive-off, which extends it to `confirmed_retention_days` after the event
  (default 30, so there is time to hand it to the police). A reviewer saying "not theft", or a
  late payment retracting the alert, deletes it at once.
- Every lookup of a plate text is written to an access log (who, why, when). The log holds the
  record id, never the plate text, and is itself purged after `access_log_days`.

`purge()` runs when the store is opened and on every write. Call it from a timer too (the review
server already has an hourly one) so an idle store still deletes on time.

Not done here: encryption at rest. Put the folder on an encrypted volume.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

DEFAULT_RETENTION_HOURS = 72.0
DEFAULT_CONFIRMED_DAYS = 30.0
DEFAULT_ACCESS_LOG_DAYS = 365.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
  record_id TEXT PRIMARY KEY, alert_id TEXT, pump_id TEXT, camera_id TEXT, plate_text TEXT, conf REAL,
  crop_path TEXT, event_ts REAL, expires_ts REAL, confirmed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS access_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, record_id TEXT, who TEXT, purpose TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS plates_alert ON plates(alert_id);
"""


class PlateStore:
    def __init__(self, folder: str | Path, retention_hours: float | None = None,
                 confirmed_retention_days: float | None = None, access_log_days: float | None = None,
                 now: float | None = None):
        self.dir = Path(folder)
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "plates.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA secure_delete=ON")       # deleted plate text is zeroed in the file
        self.db.executescript(_SCHEMA)
        for key, val in (("retention_hours", retention_hours), ("confirmed_retention_days", confirmed_retention_days),
                         ("access_log_days", access_log_days)):
            if val is not None:
                if val <= 0:
                    raise ValueError(f"{key} must be positive")
                self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(val)))
        self.db.commit()
        self.purge(now)

    def _cfg(self, key: str, default: float) -> float:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return float(row[0]) if row else default

    @property
    def retention_hours(self) -> float:
        return self._cfg("retention_hours", DEFAULT_RETENTION_HOURS)

    @property
    def confirmed_retention_days(self) -> float:
        return self._cfg("confirmed_retention_days", DEFAULT_CONFIRMED_DAYS)

    def _log(self, record_id: str, who: str, purpose: str, action: str, now: float) -> None:
        self.db.execute("INSERT INTO access_log (ts, record_id, who, purpose, action) VALUES (?,?,?,?,?)",
                        (now, record_id, who, purpose, action))

    def add(self, alert_id: str, plate_text: str, conf: float, event_ts: float, pump_id: str | None = None,
            camera_id: str | None = None, crop_jpg: bytes | None = None, now: float | None = None) -> str:
        """Store one plate read for a drive-off alert. Returns the record id (what the alert carries)."""
        now = time.time() if now is None else now
        rid = uuid.uuid4().hex[:12]
        crop = None
        if crop_jpg:
            crop = str(self.dir / "crops" / f"{rid}.jpg")
            Path(crop).write_bytes(crop_jpg)
        with self.lock:
            self.db.execute("INSERT INTO plates VALUES (?,?,?,?,?,?,?,?,?,0)",
                            (rid, alert_id, pump_id, camera_id, plate_text, float(conf), crop, float(event_ts),
                             float(event_ts) + self.retention_hours * 3600))
            self._log(rid, "system", f"drive-off alert {alert_id}", "write", now)
            self.db.commit()
        self.purge(now)
        return rid

    def get(self, record_id: str, who: str, purpose: str, now: float | None = None) -> dict | None:
        """Read a plate record. `who` and `purpose` are required and logged."""
        if not (who or "").strip() or not (purpose or "").strip():
            raise ValueError("a plate lookup needs who and purpose (both are logged)")
        now = time.time() if now is None else now
        self.purge(now)
        with self.lock:
            row = self.db.execute("SELECT * FROM plates WHERE record_id=?", (record_id,)).fetchone()
            self._log(record_id, who.strip(), purpose.strip(), "read" if row else "read_missing", now)
            self.db.commit()
        return dict(row) if row else None

    def for_alert(self, alert_id: str) -> list[str]:
        """Record ids for an alert (ids only, so no access log entry)."""
        with self.lock:
            return [r[0] for r in self.db.execute("SELECT record_id FROM plates WHERE alert_id=?", (alert_id,))]

    def confirm(self, alert_id: str, now: float | None = None) -> int:
        """A reviewer confirmed the drive-off: keep the plate `confirmed_retention_days` from the event."""
        now = time.time() if now is None else now
        with self.lock:
            cur = self.db.execute("UPDATE plates SET confirmed=1, expires_ts=event_ts+? WHERE alert_id=?",
                                  (self.confirmed_retention_days * 86400, alert_id))
            for rid in self.for_alert(alert_id):
                self._log(rid, "system", "reviewer confirmed", "extend", now)
            self.db.commit()
        return cur.rowcount

    def delete_alert(self, alert_id: str, reason: str = "retracted", now: float | None = None) -> int:
        """Delete every plate record of an alert now (late payment, or reviewer said not theft)."""
        now = time.time() if now is None else now
        with self.lock:
            rows = self.db.execute("SELECT record_id, crop_path FROM plates WHERE alert_id=?", (alert_id,)).fetchall()
            self._delete(rows, reason, now)
        return len(rows)

    def _delete(self, rows, reason: str, now: float) -> None:
        for r in rows:
            if r["crop_path"]:
                Path(r["crop_path"]).unlink(missing_ok=True)
            self.db.execute("DELETE FROM plates WHERE record_id=?", (r["record_id"],))
            self._log(r["record_id"], "system", reason, "delete", now)
        self.db.commit()

    def sync_reviews(self, review_store, now: float | None = None) -> dict:
        """Apply reviewer decisions from a bree.review ReviewStore: confirmed_theft extends, not_theft
        deletes, a fully retracted alert deletes. Call after decisions come in (or on the hourly timer)."""
        out = {"confirmed": 0, "deleted": 0}
        for a in review_store.alerts():
            if not self.for_alert(a["alert_id"]):
                continue
            if a["decision"] == "not_theft" or a.get("retracted_tier") == "retracted":
                out["deleted"] += self.delete_alert(a["alert_id"], "reviewer: not theft" if a["decision"] else "retracted", now)
            elif a["decision"] == "confirmed_theft":
                out["confirmed"] += self.confirm(a["alert_id"], now)
        return out

    def purge(self, now: float | None = None) -> dict:
        """Delete expired plate records, their crops, orphan crop files and old access log rows."""
        now = time.time() if now is None else now
        with self.lock:
            rows = self.db.execute("SELECT record_id, crop_path FROM plates WHERE expires_ts <= ?", (now,)).fetchall()
            self._delete(rows, "retention expired", now)
            keep = {Path(r[0]).name for r in self.db.execute("SELECT crop_path FROM plates WHERE crop_path IS NOT NULL")}
            orphans = [f for f in (self.dir / "crops").glob("*.jpg") if f.name not in keep]
            for f in orphans:
                f.unlink(missing_ok=True)
            cur = self.db.execute("DELETE FROM access_log WHERE ts < ?",
                                  (now - self._cfg("access_log_days", DEFAULT_ACCESS_LOG_DAYS) * 86400,))
            self.db.commit()
        return {"plates_deleted": len(rows), "orphan_crops_deleted": len(orphans), "log_rows_deleted": cur.rowcount}

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM plates").fetchone()[0]

    def access_log(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM access_log ORDER BY id")]

    def close(self) -> None:
        self.db.close()
