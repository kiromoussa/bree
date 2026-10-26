"""Read the burned-in DVR clock ("10-25-2026 16:29:12", top right) so frame times can be tied to the store clock.

Why: the DVR clock is not the edge box's clock (on the first Truman Rd. files it was 2 h behind the box).
POS receipts, shift schedules and the operator's "it happened at 4:30" are all in store time, so every
station needs a measured offset instead of a guess. Method: crop the corner, binarise bright text at a few
thresholds, run the `tesseract` CLI, keep reads that parse, and take the median offset over many frames.

Limits (see docs/ultra/platform.md): needs the `tesseract` binary on PATH (returns None without it); the
OSD position/format is a default for Dahua-style overlays and is configurable; the date part is often misread, so
only the time of day is trusted and the day is resolved so the offset lies within +-12 h of the reference clock.
It measures DVR clock minus reference clock; it cannot say which time zone the DVR is set to.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import cv2
import numpy as np

# (x0, y0, x1, y1) as fractions of the frame. Same place on every channel of the Truman Rd. DVR (7 and 29 checked).
DEFAULT_REGION = (0.74, 0.02, 0.99, 0.10)
THRESHOLDS = (240, 230, 250)
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})\s*$")
_FULL_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})")
DAY = 86400.0


def have_tesseract() -> bool:
    return shutil.which("tesseract") is not None


def _ocr(binary: np.ndarray, tmp: Path) -> str:
    cv2.imwrite(str(tmp), binary)
    try:
        r = subprocess.run(["tesseract", str(tmp), "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789-: "],
                           capture_output=True, text=True, timeout=60, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    except subprocess.TimeoutExpired:       # a loaded machine: count it as an unreadable frame, not a crash
        return ""
    return r.stdout.strip()


def _masks(gray: np.ndarray) -> list[np.ndarray]:
    """Text masks (text black on white). Plain bright-pixel thresholds work on dark backgrounds; on bright
    backgrounds (windows, ceiling lights) only 'bright pixel next to the dark OSD outline' isolates the text."""
    out = [cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)[1] for t in THRESHOLDS]
    for dark_t, white_t in ((60, 220), (60, 240), (90, 220)):
        near_dark = cv2.dilate((gray < dark_t).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        m = ((gray >= white_t) & near_dark).astype(np.uint8) * 255
        out.append(255 - cv2.dilate(m, np.ones((2, 2), np.uint8)))
    return out


def iter_candidate_texts(image: np.ndarray, region=DEFAULT_REGION):
    """Yield OCR text of the clock corner under several binarisations, on the whole overlay and on its time half
    (nothing if tesseract is missing). Lazy: callers stop as soon as enough reads agree."""
    if not have_tesseract():
        return
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region
    crop = image[int(y0 * h):int(y1 * h), int(max(0, x0 - 0.02) * w):int(min(1.0, x1) * w)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "c.png"
        for m in _masks(gray):
            for part in (m[:, int(m.shape[1] * 0.55):], m):    # the HH:MM:SS half, then the whole overlay
                b = part
                if b.shape[1] < 600:   # tesseract wants characters about 30+ px high
                    b = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                b = cv2.copyMakeBorder(b, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
                yield _ocr(b, tmp)


def candidate_texts(image: np.ndarray, region=DEFAULT_REGION) -> list[str]:
    return list(iter_candidate_texts(image, region))


def parse_time_of_day(texts: list[str]) -> int | None:
    """Seconds since midnight from the OCR candidates: the most common valid HH:MM:SS at the end of a line."""
    votes: dict[int, int] = {}
    for t in texts:
        m = _TIME_RE.search(t)
        if m:
            hh, mm, ss = map(int, m.groups())
            if hh < 24 and mm < 60 and ss < 60:
                s = hh * 3600 + mm * 60 + ss
                votes[s] = votes.get(s, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def parse_datetime(texts: list[str]) -> datetime | None:
    """Full MM-DD-YYYY HH:MM:SS when some candidate parses completely with a plausible year, else None."""
    for t in texts:
        m = _FULL_RE.search(t)
        if m:
            mo, dd, yy, hh, mi, ss = map(int, m.groups())
            try:
                if 2015 <= yy <= 2100:
                    return datetime(yy, mo, dd, hh, mi, ss)
            except ValueError:
                pass
    return None


def read_clock(image: np.ndarray, region=DEFAULT_REGION, agree: int = 2) -> int | None:
    """Time of day (seconds since midnight, DVR clock) shown in the frame, or None. Stops OCR-ing as soon as
    `agree` binarisations give the same time (most frames need 2-4 tesseract calls, hard ones all 14)."""
    seen, votes = [], {}
    for text in iter_candidate_texts(image, region):
        seen.append(text)
        t = parse_time_of_day([text])
        if t is not None:
            votes[t] = votes.get(t, 0) + 1
            if votes[t] >= agree:
                return t
    return parse_time_of_day(seen)


@dataclass
class ClockEstimate:
    offset_s: float | None      # DVR clock minus reference clock, seconds, within +-12 h; None = nothing readable
    spread_s: float | None      # median absolute deviation of the per-frame offsets (OCR noise / DVR jitter)
    n_read: int
    n_tried: int

    @property
    def ok(self) -> bool:
        return self.offset_s is not None and self.n_read >= 3 and (self.spread_s or 0) <= 2.0

    def to_dict(self) -> dict:
        return {"offset_s": self.offset_s, "spread_s": self.spread_s, "n_read": self.n_read, "n_tried": self.n_tried,
                "ok": self.ok}


def estimate_offset(samples: list[tuple[float, int | None]]) -> ClockEstimate:
    """`samples` = (reference time-of-day seconds of the frame, DVR time-of-day read from it or None).
    Per-sample offset is wrapped into (-12 h, 12 h]; an outlier-robust median is returned.
    Reads more than 5 s from the median are dropped (misreads) before the spread is computed."""
    offs = []
    for ref, got in samples:
        if got is not None:
            offs.append(((got - ref + DAY / 2) % DAY) - DAY / 2)
    if not offs:
        return ClockEstimate(None, None, 0, len(samples))
    m = median(offs)
    keep = [o for o in offs if abs(o - m) <= 5.0]
    m = median(keep)
    mad = median([abs(o - m) for o in keep])
    return ClockEstimate(float(m), float(mad), len(keep), len(samples))


def tod(dt: datetime) -> float:
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def file_start_from_name(path: str | Path) -> datetime | None:
    """Recorder file names are YYYYmmdd_HHMMSS.mp4 in the recorder machine's local time."""
    m = re.match(r"(\d{8})_(\d{6})", Path(path).name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def check_file_clock(path: str | Path, n: int = 8, region=DEFAULT_REGION) -> dict:
    """Offset of the overlay clock against the file-name clock, from n frames spread over the file, plus a
    linearity check (overlay deltas should equal file deltas; a jump means the DVR clock was changed or frames
    were dropped). Reference = name time + frame time. REAL-UNLABELLED: it compares two clocks, it is not truth."""
    start = file_start_from_name(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / fps if total > 0 else 0
    samples, rows = [], []
    for i in range(n):
        t = dur * (i + 0.5) / n
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        got = read_clock(img, region)
        ref = (tod(start) + t) % DAY if start else None
        rows.append({"t": round(t, 1), "overlay_tod": got})
        if ref is not None:
            samples.append((ref, got))
    cap.release()
    est = estimate_offset(samples) if samples else ClockEstimate(None, None, 0, 0)
    reads = []   # misreads (more than 5 s from the median offset) are left out of the drift fit
    for r in rows:
        if r["overlay_tod"] is not None and est.offset_s is not None and start is not None:
            o = ((r["overlay_tod"] - ((tod(start) + r["t"]) % DAY) + DAY / 2) % DAY) - DAY / 2
            if abs(o - est.offset_s) <= 5.0:
                reads.append((r["t"], r["overlay_tod"]))
    drift = None
    if len(reads) >= 3:   # least-squares slope of overlay seconds vs file seconds, wrapped at midnight
        ts = np.array([a for a, _ in reads])
        vs = np.unwrap(np.array([b for _, b in reads]) / DAY * 2 * np.pi) / (2 * np.pi) * DAY
        drift = float(np.polyfit(ts, vs, 1)[0])
    return {"file": str(path), "name_start": start.isoformat() if start else None, "estimate": est.to_dict(),
            "overlay_seconds_per_file_second": drift, "reads": rows}
