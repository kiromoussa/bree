"""The store signal contract (v1) and its on-disk log.

One signal is one JSON object on one line:
  {"v": 1, "ts": 1792970000.5, "store": "trumanrd", "camera": "ch07", "kind": "door_in",
   "value": 1, "zone": null, "source": "real_unlabelled", "meta": {}}

`ts` is UTC epoch seconds of the store's real wall clock (adapters apply the DVR clock offset).
`value` is a number whose meaning depends on `kind` (table below). `source` says how far to trust it;
it travels all the way to the dashboard and the digest so a number never loses its label.

The log is a folder of `signals-YYYYMMDD.jsonl` files (store-local date), append only. Writers may be
several processes: each append is one short write in append mode.
"""
from __future__ import annotations

import datetime as dt
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = 1

# kind -> (unit, what `value` means)
KINDS = {
    "occupancy": ("people", "people inside / in view at ts (a camera, or camera='store' for the whole store)"),
    "door_in": ("people", "people who came in through the door (usually 1 per signal)"),
    "door_out": ("people", "people who left through the door"),
    "queue": ("people", "people waiting at the register at ts"),
    "dwell": ("seconds", "one finished visit to a zone (zone is required)"),
    "shelf": ("0..1", "how full a shelf/cooler section looks, 1 = full; meta.state names a problem"),
    "health": ("0|1", "camera heartbeat: 1 = frames arriving; meta may hold fps"),
    "pos_sale": ("usd", "one POS receipt total, when a POS feed exists"),
}

# How much a reader should trust a signal. Order = increasing trust.
SOURCES = ("simulated", "real_unlabelled", "real_model_labelled", "real_ground_truth", "live")

SHELF_STATES = ("ok", "low", "empty", "door_open", "misplaced", "dirty", "unknown")


@dataclass
class Signal:
    ts: float
    kind: str
    value: float = 1.0
    store: str = "store"
    camera: str = "store"
    zone: str | None = None
    source: str = "live"
    meta: dict = field(default_factory=dict)
    v: int = VERSION

    def validate(self) -> "Signal":
        if self.kind not in KINDS:
            raise ValueError(f"unknown signal kind {self.kind!r}; known: {sorted(KINDS)}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; known: {SOURCES}")
        if not (self.ts == self.ts) or abs(self.ts) == float("inf"):
            raise ValueError("ts must be finite")
        if self.kind == "dwell" and not self.zone:
            raise ValueError("dwell needs a zone")
        if self.kind == "shelf" and self.meta.get("state", "ok") not in SHELF_STATES:
            raise ValueError(f"shelf meta.state must be one of {SHELF_STATES}")
        self.value = float(self.value)
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Signal":
        if int(d.get("v", VERSION)) != VERSION:
            raise ValueError(f"unsupported signal version {d.get('v')}")
        keys = ("ts", "kind", "value", "store", "camera", "zone", "source", "meta")
        return Signal(**{k: d[k] for k in keys if k in d}).validate()


def tzinfo(name: str | None):
    """Store time zone; None = this machine's local zone."""
    if name:
        return ZoneInfo(name)
    return dt.datetime.now().astimezone().tzinfo


def local_date(ts: float, tz) -> dt.date:
    return dt.datetime.fromtimestamp(ts, tz).date()


def day_bounds(day: dt.date, tz) -> tuple[float, float]:
    """[start, end) epoch seconds of a store-local day (DST days are 23 or 25 hours long)."""
    a = dt.datetime.combine(day, dt.time(), tzinfo=tz)
    b = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), tzinfo=tz)
    return a.timestamp(), b.timestamp()


class SignalLog:
    def __init__(self, folder: str | Path, tz: str | None = None):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tz = tzinfo(tz)
        self._lock = threading.Lock()

    def _file(self, day: dt.date) -> Path:
        return self.dir / f"signals-{day:%Y%m%d}.jsonl"

    def append(self, signals) -> int:
        """Validate and append. Invalid signals raise before anything is written."""
        sigs = [s.validate() for s in ([signals] if isinstance(signals, Signal) else list(signals))]
        by_day: dict[dt.date, list[str]] = {}
        for s in sigs:
            by_day.setdefault(local_date(s.ts, self.tz), []).append(json.dumps(s.to_dict(), separators=(",", ":")))
        with self._lock:
            for day, lines in by_day.items():
                with open(self._file(day), "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
        return len(sigs)

    def days(self) -> list[dt.date]:
        out = []
        for p in sorted(self.dir.glob("signals-*.jsonl")):
            try:
                out.append(dt.datetime.strptime(p.stem[8:], "%Y%m%d").date())
            except ValueError:
                pass
        return out

    def read(self, since: float | None = None, until: float | None = None, kinds=None) -> list[Signal]:
        """Signals with since <= ts < until, sorted by ts. Bad lines are skipped, never fatal
        (a half-written last line while a writer is running is normal)."""
        lo = since if since is not None else float("-inf")
        hi = until if until is not None else float("inf")
        out: list[Signal] = []
        for day in self.days():
            d0, d1 = day_bounds(day, self.tz)
            if d1 <= lo or d0 >= hi:
                # a signal can sit in a neighbouring day file only if a writer used another zone; skip fast
                continue
            try:
                text = self._file(day).read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    s = Signal.from_dict(json.loads(line))
                except (ValueError, KeyError, TypeError):
                    continue
                if lo <= s.ts < hi and (kinds is None or s.kind in kinds):
                    out.append(s)
        out.sort(key=lambda s: s.ts)
        return out

    def latest_ts(self) -> float | None:
        for day in reversed(self.days()):
            sigs = self.read(*day_bounds(day, self.tz))
            if sigs:
                return sigs[-1].ts
        return None
