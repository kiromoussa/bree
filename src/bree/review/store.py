"""Review store: one SQLite file with every alert sent to review and every reviewer decision.

Folder layout (`ReviewStore(folder)`):
  review.sqlite        tables: alerts, reviews (append-only), staged_tests, meta
  datasets/<name>/     label exports (bree.review.labels), covered by the same retention rule.
                       An export written anywhere else is recorded in `meta` and covered too.
                       A copy somebody makes of a dataset folder is not.

Privacy rules enforced here:
- Only head-pixelated media is referenced. The check is by folder name, so it is best effort: a clip
  must sit in a folder named `alerts` (where `bree.alerts.writer` puts them), a frame image in a
  folder named `frames` (where `save_evidence_frame`, which pixelates heads, should write), and any
  path with a `raw...` folder in it is refused. Nothing looks at the pixels.
- No identity features: items, receipt lines and frames are stored through a fixed list of fields
  (ITEM_KEYS, PAID_KEYS, FRAME_KEYS, PRODUCT_KEYS). Any other key is dropped on ingest, whatever it
  is called. A person is only a per-session track number.
- Retention: `retention_days` (default 30, stored in the DB). `purge()` deletes the clip, the frame
  images and the stored keypoints of every alert older than that, and expired examples in
  every exported dataset. It runs every time the store is opened and every hour while the review
  page is up. The decision row stays (counts and precision over time still work); it holds no images.

Event time: `event_ts` is the record's wall-clock `time` when it has one (shadow records: the moment
the alert was emitted, a few seconds after the exit). Plain pipeline alerts carry only video seconds,
so without an explicit `event_ts` the time of ingest is used. Time to decision, week buckets and
retention all run from `event_ts`.

A `frames` entry (all optional except `t`), pixels are in the camera's full resolution:
  {"t": 4.9, "event": "pick" | "conceal" | "exit", "item": "energy_drink",
   "image": "<head-pixelated jpg, no overlays>", "width": 1280, "height": 720,
   "item_box": [x1, y1, x2, y2],                      # the alerted item in this frame
   "products": [{"id", "bbox", "cat", "conf"}],       # every product box the detector saw
   "kpts": [[x, y, conf] * 17]}                       # pose of the alerted person (COCO-17)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

DECISIONS = ("confirmed_theft", "not_theft", "wrong_item", "unclear")
DISPUTED = "disputed"            # two reviewers gave different decisions
DEFAULT_RETENTION_DAYS = 30.0
RETRACTED = "retracted"          # retracted_tier of an alert a late receipt withdrew completely
# The only fields kept from an alert's items, receipt lines, frames and product boxes.
ITEM_KEYS = ("category", "sku", "t_pick", "zone", "pick_confidence", "concealed", "held_at_exit",
             "ambiguous_with", "score")
PAID_KEYS = ("category", "sku", "name", "qty", "price")
FRAME_KEYS = ("t", "event", "item", "image", "width", "height", "item_box", "kpts")
PRODUCT_KEYS = ("id", "bbox", "cat", "conf")
_RAW_PART = re.compile(r"raw", re.I)                  # raw/, RAW/, raw_segments/, rawclips/ ...
_UNBLURRED = re.compile(r"unblur|unpixel", re.I)
_MEDIA = {".mp4", ".jpg", ".jpeg", ".png"}
_FOLDER = {".mp4": "alerts", ".jpg": "frames", ".jpeg": "frames", ".png": "frames"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
  alert_id TEXT PRIMARY KEY, camera TEXT, zone TEXT, predicted_item TEXT, items TEXT,
  register_match TEXT, register TEXT, identity_uncertain INTEGER, tier TEXT, confidence REAL,
  person_track INTEGER, clip_path TEXT, frames TEXT, reasons TEXT,
  event_ts REAL, created_ts REAL, retracted_tier TEXT, staged_test_id TEXT, purged_ts REAL);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT NOT NULL REFERENCES alerts(alert_id),
  reviewer_id TEXT NOT NULL, decision TEXT NOT NULL, corrected_item TEXT,
  pick_seen INTEGER, conceal_seen INTEGER, note TEXT,
  opened_ts REAL, decided_ts REAL, time_to_decision_s REAL);
CREATE TABLE IF NOT EXISTS staged_tests (
  test_id TEXT PRIMARY KEY, ts REAL, camera TEXT, zone TEXT, item TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS reviews_alert ON reviews(alert_id);
"""


def _only(rows, keys) -> list[dict]:
    """The dicts in `rows` cut down to `keys`. Everything else (re-ID vectors, crops, ...) is dropped."""
    return [{k: r[k] for k in keys if k in r} for r in rows or [] if isinstance(r, dict)]


def normalise_item(text) -> str | None:
    """One spelling per item name: trimmed, lower case, spaces to underscores."""
    return re.sub(r"\s+", "_", str(text or "").strip().lower()) or None


def write_json(path: Path, obj) -> None:
    """Write through a temp file and rename, so an interrupted write never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _blurred_media(path, base_dir=None) -> str | None:
    """Absolute path of a clip / frame, or None. Raw recordings (not pixelated) are refused."""
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = Path(base_dir or ".") / p
    p = p.resolve()
    if p.suffix.lower() not in _MEDIA:
        raise ValueError(f"refusing {p}: not a clip or frame image")
    if _is_raw(p):
        raise ValueError(f"refusing {p}: raw recordings are not head-pixelated")
    if p.parent.name != _FOLDER[p.suffix.lower()]:
        raise ValueError(f"refusing {p}: clips must be in a folder named 'alerts' and frame images in a "
                         "folder named 'frames' (where the head-pixelating writers put them)")
    return str(p)


def _is_raw(p: Path) -> bool:
    return any(_RAW_PART.match(part) or _UNBLURRED.search(part) for part in p.parts[:-1])


def _event_time(text: str) -> float:
    try:                                        # shadow records: local time with a UTC offset
        return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:                          # no offset: read as local time
        return time.mktime(time.strptime(text[:19], "%Y-%m-%dT%H:%M:%S"))


def register_match(visited: bool, paid: list, unpaid: list) -> str:
    """How the register record lines up with what the cameras saw."""
    if not visited and not paid:
        return "no_register_visit"
    if not paid:
        return "visited_no_receipt"
    return "receipt_missing_item" if unpaid else "receipt_matches"


def save_evidence_frame(img, persons, path) -> str:
    """Write one frame for label export: heads pixelated (same routine as the alert clips), no
    overlays, full resolution. This is the only way a frame should reach the review store, and
    the store only takes frame images from a folder named `frames`, so write there."""
    import cv2
    from bree.alerts.annotate import blur_heads
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blur_heads(img.copy(), persons), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


def frames_from_log(frames_jsonl, alert: dict, pre_s: float = 1.6, post_s: float = 1.0) -> list[dict]:
    """Pose windows for an alert from the pipeline's existing `frames.jsonl` (needs log_frames on):
    the alerted person's keypoints around each pick and conceal event. No images and no item box
    (the log does not say which product box was the item), so these feed the pick / conceal
    classifiers only."""
    pids = set(alert.get("group") or []) | {alert.get("person_id")}
    rows = [json.loads(line) for line in Path(frames_jsonl).read_text().splitlines() if line.strip()]
    marks = [(e["t"], e["type"], e.get("item")) for r in rows for e in r.get("events", [])
             if e["person_id"] in pids and e["type"] in ("pick", "conceal")]
    out = []
    for t0, kind, item in marks:
        for r in rows:
            if t0 - pre_s <= r["t"] <= t0 + post_s:
                k = next((p["kpts"] for p in r["persons"] if p["id"] in pids and p.get("kpts")), None)
                if k:
                    out.append({"t": r["t"], "event": kind, "item": item, "kpts": k})
    return out


class ReviewStore:
    def __init__(self, folder: str | Path, retention_days: float | None = None, now: float | None = None):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "review.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if retention_days is not None:
            if retention_days <= 0:
                raise ValueError("retention_days must be positive")
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('retention_days', ?)", (str(retention_days),))
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('store_id', ?)", (uuid.uuid4().hex,))
        self.db.commit()
        self.store_id = self.db.execute("SELECT value FROM meta WHERE key='store_id'").fetchone()[0]
        self.purge(now)

    # ------------------------------------------------------------------ config
    @property
    def retention_days(self) -> float:
        row = self.db.execute("SELECT value FROM meta WHERE key='retention_days'").fetchone()
        return float(row[0]) if row else DEFAULT_RETENTION_DAYS

    def _all(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ alerts in
    def add_alert(self, rec: dict, camera: str | None = None, event_ts: float | None = None,
                  frames: list[dict] | None = None, base_dir: str | Path | None = None,
                  staged_test_id: str | None = None, now: float | None = None) -> str | None:
        """Record one alert sent to review. `rec` is `Alert.to_dict()` or a shadow-mode
        `would_be_record`. A retraction record updates the alert it retracts and is never stored
        as an alert of its own. Returns the alert id, or None when this id is already stored
        (dedup) or the retracted alert is not in the store."""
        now = time.time() if now is None else now
        aid = str(rec.get("id") or rec["alert_id"])
        if rec.get("retracts"):
            with self.lock:
                cur = self.db.execute("UPDATE alerts SET retracted_tier=? WHERE alert_id=?",
                                      (rec.get("tier"), str(rec["retracts"])))
                self.db.commit()
            return str(rec["retracts"]) if cur.rowcount else None
        unpaid = _only(rec.get("unpaid_items") or rec.get("unpaid"), ITEM_KEYS)
        paid = _only(rec.get("paid_items") or rec.get("paid"), PAID_KEYS)
        top = max(unpaid, key=lambda i: i.get("score", 0.0), default={})
        if event_ts is None:
            event_ts = _event_time(rec["time"]) if rec.get("time") else now
        src = frames if frames is not None else rec.get("frames") or []
        frames = _only(src, FRAME_KEYS)
        for f, raw in zip(frames, (r for r in src if isinstance(r, dict))):
            f["image"] = _blurred_media(f.get("image"), base_dir)
            if "products" in raw:
                f["products"] = _only(raw["products"], PRODUCT_KEYS)
        reasons = [str(r) for r in rec.get("reasons") or []]
        row = (aid, camera or rec.get("camera"), top.get("zone"), top.get("sku") or top.get("category"),
               json.dumps(unpaid), register_match(bool(rec.get("visited_register")), paid, unpaid),
               json.dumps({"visited_register": bool(rec.get("visited_register")), "paid_items": paid}),
               int(any("identity uncertain" in r for r in reasons)), rec.get("tier"), rec.get("confidence"),
               rec.get("person_id"), _blurred_media(rec.get("clip_path") or rec.get("clip"), base_dir),
               json.dumps(frames), json.dumps(reasons), float(event_ts), now, None, staged_test_id, None)
        with self.lock:
            cur = self.db.execute("INSERT OR IGNORE INTO alerts VALUES (" + ",".join("?" * len(row)) + ")", row)
            self.db.commit()
        return aid if cur.rowcount else None

    def add_staged_test(self, test_id: str, ts: float, camera: str | None = None, zone: str | None = None,
                        item: str | None = None, note: str = "") -> None:
        """A test theft staged by the owner: when, where, what. Matched to alerts in the owner report."""
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO staged_tests VALUES (?,?,?,?,?,?)",
                            (test_id, float(ts), camera, zone, item, note))
            self.db.commit()

    # ------------------------------------------------------------------ decisions
    def decide(self, alert_id: str, reviewer_id: str, decision: str, corrected_item: str | None = None,
               pick_seen: bool | None = None, conceal_seen: bool | None = None, note: str = "",
               time_to_decision_s: float | None = None, now: float | None = None) -> dict:
        """One reviewer's decision. Append-only: a changed mind is a new row, the latest row per
        reviewer counts. `time_to_decision_s` is how long the reviewer looked at the alert."""
        now = time.time() if now is None else now
        reviewer_id = (reviewer_id or "").strip()
        # A correction only means something for wrong_item: anything else left in the field is dropped.
        # One spelling per item ("Chips " and "CHIPS" are "chips"), so typing variants add no classes.
        corrected_item = normalise_item(corrected_item) if decision == "wrong_item" else None
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if not reviewer_id:
            raise ValueError("reviewer id is required")
        if decision == "wrong_item" and not corrected_item:
            raise ValueError("wrong_item needs the corrected item")
        if not self._all("SELECT 1 FROM alerts WHERE alert_id=?", (alert_id,)):
            raise ValueError(f"unknown alert {alert_id!r}")
        if time_to_decision_s is not None:
            if not math.isfinite(float(time_to_decision_s)):
                raise ValueError("time to decision must be a finite number of seconds")
            time_to_decision_s = max(0.0, float(time_to_decision_s))
        b = lambda v: None if v is None else int(bool(v))  # noqa: E731
        row = (alert_id, reviewer_id, decision, corrected_item, b(pick_seen), b(conceal_seen), note.strip(),
               None if time_to_decision_s is None else now - time_to_decision_s, now, time_to_decision_s)
        with self.lock:
            self.db.execute("INSERT INTO reviews (alert_id, reviewer_id, decision, corrected_item, pick_seen,"
                            " conceal_seen, note, opened_ts, decided_ts, time_to_decision_s)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?)", row)
            self.db.commit()
        return {"alert_id": alert_id, "reviewer_id": reviewer_id, "decision": decision,
                "corrected_item": corrected_item, "decided_ts": now}

    # ------------------------------------------------------------------ reads
    def alerts(self, since: float | None = None, until: float | None = None) -> list[dict]:
        """Every alert with its reviews (latest per reviewer, oldest reviewer first) and the
        final decision: the shared decision, `disputed` if reviewers differ, None if unreviewed."""
        rows = self._all("SELECT * FROM alerts WHERE event_ts >= ? AND event_ts < ? ORDER BY event_ts, alert_id",
                         (since if since is not None else -1e18, until if until is not None else 1e18))
        latest: dict[tuple[str, str], dict] = {}
        first: dict[tuple[str, str], float] = {}
        for r in self._all("SELECT * FROM reviews ORDER BY id"):
            key = (r["alert_id"], r["reviewer_id"])
            first.setdefault(key, r["decided_ts"])
            latest[key] = r
        for a in rows:
            for k in ("items", "register", "frames", "reasons"):
                a[k] = json.loads(a[k]) if a[k] else []
            a["identity_uncertain"] = bool(a["identity_uncertain"])
            revs = sorted((r for (aid, _), r in latest.items() if aid == a["alert_id"]),
                          key=lambda r: first[(r["alert_id"], r["reviewer_id"])])
            a["reviews"] = revs
            kinds = {r["decision"] for r in revs}
            a["decision"] = None if not revs else kinds.pop() if len(kinds) == 1 else DISPUTED
            a["first_decided_ts"] = min((first[(a["alert_id"], r["reviewer_id"])] for r in revs), default=None)
        return rows

    def next_for(self, reviewer_id: str, alert_id: str | None = None) -> tuple[dict | None, int]:
        """Oldest alert this reviewer has not decided yet, and how many are left. Purged alerts and
        alerts a late receipt withdrew completely are not queued. With `alert_id`: that alert
        instead (the page's "back" button), reviewed or not, as long as it is not purged."""
        live = [a for a in self.alerts() if not a["purged_ts"]]
        todo = [a for a in live if a["retracted_tier"] != RETRACTED
                and reviewer_id not in {r["reviewer_id"] for r in a["reviews"]}]
        if alert_id is not None:
            return next((a for a in live if a["alert_id"] == alert_id), None), len(todo)
        return (todo[0] if todo else None), len(todo)

    def clip_file(self, alert_id: str) -> Path | None:
        row = self._all("SELECT clip_path FROM alerts WHERE alert_id=?", (alert_id,))
        if not row or not row[0]["clip_path"]:
            return None
        p = Path(row[0]["clip_path"])
        return p if p.is_file() and p.suffix == ".mp4" and not _is_raw(p) else None

    def staged_tests(self) -> list[dict]:
        return self._all("SELECT * FROM staged_tests ORDER BY ts")

    def known_items(self) -> list[str]:
        """Item names the page suggests: everything the pipeline named plus earlier corrections."""
        rows = self._all("SELECT corrected_item AS i FROM reviews")
        return sorted(self.pipeline_items() | {r["i"] for r in rows if r["i"]})

    def pipeline_items(self) -> set[str]:
        """Item categories (and predicted items) the pipeline itself has named in an alert."""
        out = set()
        for r in self._all("SELECT predicted_item, items FROM alerts"):
            out |= {r["predicted_item"]} | {i.get("category") for i in json.loads(r["items"] or "[]")}
        return out - {None, ""}

    # ------------------------------------------------------------------ retention
    def dataset_dirs(self) -> list[Path]:
        """Every folder a label export was written to: <store>/datasets/* plus recorded ones."""
        row = self._all("SELECT value FROM meta WHERE key='dataset_dirs'")
        dirs = {mf.parent.resolve() for mf in self.dir.glob("datasets/*/manifest.json")}
        return sorted(dirs | {Path(p) for p in (json.loads(row[0]["value"]) if row else [])})

    def register_dataset(self, folder: str | Path) -> None:
        """Called by the label export, so retention also reaches datasets outside the store."""
        with self.lock:
            dirs = sorted({str(p) for p in self.dataset_dirs()} | {str(Path(folder).resolve())})
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('dataset_dirs', ?)", (json.dumps(dirs),))
            self.db.commit()

    def purge(self, now: float | None = None) -> dict:
        """Delete media and keypoints of alerts older than the retention period, and expired
        examples in every exported dataset. Safe to call any time."""
        now = time.time() if now is None else now
        cutoff = now - self.retention_days * 86400
        files = n = 0
        with self.lock:
            for a in self.db.execute("SELECT alert_id, clip_path, frames FROM alerts"
                                     " WHERE event_ts < ? AND purged_ts IS NULL", (cutoff,)).fetchall():
                paths = [a["clip_path"]] + [f.get("image") for f in json.loads(a["frames"] or "[]")]
                for p in filter(None, paths):
                    if Path(p).suffix.lower() in _MEDIA and Path(p).is_file():
                        Path(p).unlink()
                        files += 1
                self.db.execute("UPDATE alerts SET clip_path=NULL, frames=NULL, purged_ts=? WHERE alert_id=?",
                                (now, a["alert_id"]))
                n += 1
            self.db.commit()
        examples, errors = 0, []
        for mf in (d / "manifest.json" for d in self.dataset_dirs()):
            if not mf.is_file():                # the folder was moved or deleted by hand
                continue
            try:                                # one unreadable manifest must not stop the store opening
                m = json.loads(mf.read_text())
                keep = [e for e in m["examples"] if e["event_ts"] >= cutoff]    # today's retention, not the one at export
                for e in m["examples"]:
                    if e["event_ts"] < cutoff:
                        for k in ("image", "labels", "window"):
                            if e.get(k):
                                (mf.parent / e[k]).unlink(missing_ok=True)
                        examples += 1
                if len(keep) != len(m["examples"]):
                    m["examples"] = keep
                    m["counts"] = {**m.get("counts", {}),
                                   **{t: sum(1 for e in keep if e["task"] == t) for t in ("detector", "pick", "conceal")},
                                   "unknown_classes": sum(1 for e in keep if e.get("class_known") is False)}
                    write_json(mf, m)
            except (ValueError, KeyError, TypeError, OSError) as e:    # JSONDecodeError is a ValueError
                errors.append({"folder": str(mf.parent), "error": f"{type(e).__name__}: {e}"})
        return {"alerts_purged": n, "files_deleted": files, "dataset_examples_deleted": examples,
                "dataset_errors": errors}

    def close(self) -> None:
        self.db.close()
