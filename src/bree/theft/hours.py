"""After-hours intrusion and loitering, from gated person tracks and a clock.

Clock. Streams count seconds from the start of a segment; the store's rules are in local time of day. `StoreClock` maps
one to the other from the segment's start time and a per-station offset. The offset is a settings value, not a guess:
the DVR overlay on the Truman Rd. footage reads about 2 h before the recorder's file names and nobody has confirmed the
store's own time zone (docs/ultra/theft.md), so a station that has not set `clock_offset_s` gets no after-hours alarms
(`StoreClock.known` is False) rather than wrong ones.

After hours (`closed` windows from `hours: {open, close}` in the station config): an established person track inside the
store while it is closed is an `after_hours` incident. Indoor cameras: severity high (staff opening up are the usual
innocent cause, so `staff_grace` windows exist: 30 min before open by default). Outdoor cameras: severity low and
only reported as loitering, because cars and people pass a parking lot all night.

Loitering (`LoiterMonitor`): an established person who stays within `radius_px` of one spot for `dwell_s` while not in a
register / staff zone. Defaults by camera role: door 3 min, outdoor 5 min, aisle/cooler 10 min, counter never (queues and
staff). It is a nudge ("someone has been standing at the door for 6 minutes"), severity low, never part of a theft score.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import FrameObs
from bree.events.zones import Zone
from bree.theft.incidents import Incident

DAY = 86400.0
DEFAULT_DWELL_S = {"door": 180.0, "outdoor": 300.0, "aisle": 600.0, "cooler": 600.0, "counter": None}


@dataclass
class StoreClock:
    """seconds-of-local-day for a stream time: (epoch of stream start + clock_offset_s + t) shifted into local time.
    `start_epoch` is the stream start on the recorder's clock; `clock_offset_s` converts that to the store's local time of
    day. known=False when the offset has not been set for this station."""
    start_epoch: float = 0.0
    clock_offset_s: float | None = None

    @property
    def known(self) -> bool:
        return self.clock_offset_s is not None

    def local(self, t: float) -> float:
        return (self.start_epoch + (self.clock_offset_s or 0.0) + t) % DAY

    def weekday(self, t: float) -> int:
        return int(((self.start_epoch + (self.clock_offset_s or 0.0) + t) // DAY + 3) % 7)   # 0 = Monday (epoch day 0 was a Thursday)

    @staticmethod
    def from_segment_name(name: str, clock_offset_s: float | None = None, tz_of_name: dt.tzinfo | None = None) -> "StoreClock":
        """File names are YYYYmmdd_HHMMSS on the recorder's clock; `tz_of_name` is that clock's zone (None = a
        naive timestamp treated as UTC: only the time of day matters, so the offset then carries the whole correction)."""
        stem = name.rsplit("/", 1)[-1].split(".")[0]
        d = dt.datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").replace(tzinfo=tz_of_name or dt.timezone.utc)
        return StoreClock(d.timestamp(), clock_offset_s)


def _hm(s: str) -> float:
    h, m = s.split(":")[:2]
    return int(h) * 3600 + int(m) * 60


@dataclass
class HoursConfig:
    open: str = "06:00"
    close: str = "23:00"
    staff_grace_min: float = 30.0    # people inside this long before open are the people opening up: reported at low severity
    close_grace_min: float = 15.0    # and this long after close, the last customers and the closing routine
    min_age_s: float = 3.0           # a track must have lived this long (the gate's own filter comes first)
    cooldown_s: float = 300.0        # one incident per camera per this long (a person in the shop all night is one incident)

    @staticmethod
    def from_dict(d: dict) -> "HoursConfig":
        k = HoursConfig.__dataclass_fields__
        return HoursConfig(**{a: b for a, b in d.items() if a in k and a != "clock_offset_s"})


class AfterHoursMonitor:
    def __init__(self, clock: StoreClock, cfg: HoursConfig | None = None, outdoor: bool = False):
        self.clock, self.cfg, self.outdoor = clock, cfg or HoursConfig(), outdoor
        self.first_seen: dict[int, float] = {}
        self._last = -1e9
        self.stats = {"closed_person_frames": 0}

    def state(self, t: float) -> str:
        """"open", "grace" (just before open / just after close) or "closed"."""
        s = self.clock.local(t)
        o, c, c2 = _hm(self.cfg.open), _hm(self.cfg.close), None
        g_in, g_out = self.cfg.staff_grace_min * 60, self.cfg.close_grace_min * 60
        if o <= c:
            if o <= s < c:
                return "open"
            if o - g_in <= s < o or c <= s < c + g_out:
                return "grace"
            return "closed"
        # overnight opening (e.g. 05:00-02:00 next day): closed window is c..o
        if s >= o or s < c:
            return "open"
        if o - g_in <= s < o or c <= s < c + g_out:
            return "grace"
        return "closed"

    def update(self, obs: FrameObs) -> list[Incident]:
        if not self.clock.known or not obs.persons:
            return []
        st = self.state(obs.t)
        if st == "open":
            self.first_seen.clear()
            return []
        out = []
        for p in obs.persons:
            t0 = self.first_seen.setdefault(p.track_id, obs.t)
            if obs.t - t0 < self.cfg.min_age_s:
                continue
            self.stats["closed_person_frames"] += 1
            if obs.t - self._last < self.cfg.cooldown_s:
                continue
            self._last = obs.t
            sev = "low" if (st == "grace" or self.outdoor) else "high"
            where = "outside" if self.outdoor else "inside the store"
            hh = int(self.clock.local(obs.t) // 3600)
            mm = int(self.clock.local(obs.t) % 3600 // 60)
            out.append(Incident("after_hours", obs.t, t0, sev,
                                f"person {where} at {hh:02d}:{mm:02d}, store {'closing/opening grace' if st == 'grace' else 'closed'}",
                                {"track": p.track_id, "state": st, "local": f"{hh:02d}:{mm:02d}"}))
        return out


@dataclass
class LoiterConfig:
    radius_px: float = 120.0         # stays within this of where it settled (centre of the box foot point)
    dwell_s: float | None = None     # None = by camera role (DEFAULT_DWELL_S)
    cooldown_s: float = 600.0
    min_conf: float = 0.45


class LoiterMonitor:
    def __init__(self, role: str, cfg: LoiterConfig | None = None, exempt: list[Zone] | None = None):
        self.cfg = cfg or LoiterConfig()
        self.dwell_s = self.cfg.dwell_s if self.cfg.dwell_s is not None else DEFAULT_DWELL_S.get(role)
        self.exempt = exempt or []
        self.anchor: dict[int, tuple[float, float, float]] = {}     # id -> (t settled, x, y)
        self.fired: dict[int, float] = {}
        self.max_dwell = 0.0                                          # longest settled time seen (for rate tables)

    def update(self, obs: FrameObs) -> list[Incident]:
        if self.dwell_s is None:
            return []
        out = []
        for p in obs.persons:
            if p.conf < self.cfg.min_conf:
                continue
            x, y = p.foot_point()
            if any(z.contains(x, y) for z in self.exempt):
                self.anchor.pop(p.track_id, None)
                continue
            a = self.anchor.get(p.track_id)
            if a is None or np.hypot(x - a[1], y - a[2]) > self.cfg.radius_px:
                self.anchor[p.track_id] = (obs.t, x, y)
                continue
            dwell = obs.t - a[0]
            self.max_dwell = max(self.max_dwell, dwell)
            if dwell >= self.dwell_s and obs.t - self.fired.get(p.track_id, -1e9) >= self.cfg.cooldown_s:
                self.fired[p.track_id] = obs.t
                out.append(Incident("loiter", obs.t, a[0], "low", f"person has stayed in one spot for {dwell / 60:.0f} min",
                                    {"track": p.track_id, "dwell_s": round(dwell, 1), "at": [round(x), round(y)]}))
        return out
