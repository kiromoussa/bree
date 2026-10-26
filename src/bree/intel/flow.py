"""Streaming people-flow engine: door crossings, occupancy, zone dwell, register queue, staff vs customer, outcome proxies.

Input is what the detector sees (per sampled frame a list of person boxes, see bree.intel.detections); output is a
stream of JSON-able records (schema in docs/ultra/intel.md):

  door    {"type":"door","dir":"in"|"out","t","ts","track","via":"line"|"born"|"lost"}
  zone    {"type":"zone","track","zone","kind","t_in","t_out","dwell_s","still_s"}      when a visit ends
  sample  {"type":"sample","t","ts","in_view","customers","staff","queue","occupancy"}  every `sample_s`
  track   {"type":"track","track","role","t0","t1",...,"outcome"}                       when a track closes

Everything is camera-local and anonymous: a track id is a number for one walk through the view. No face, no identity, no
appearance model. A person who leaves the view and returns gets a new id, so per-person claims ("visited twice") are not made.

Door crossing: a track's smoothed foot point is on the outside (`s < -margin`) or inside (`s > +margin`) of the door line while it
is within the door's span. Outside -> inside is an entry, inside -> outside an exit (via "line"). The detector often loses
the person in the door's glare, so two weaker kinds of evidence are also used and flagged: a track that is born on the entry
mats and goes on inside is an entry (via "born"), one that is last seen on the mats after being inside is an exit (via "lost").

Occupancy is `initial + entries - exits` over customers (never below 0), where `initial` is the number of customers in view in the
first seconds. The camera does not see the whole store, so this drifts when a door crossing is missed; `customers` (in view now)
is the hard lower bound. Staff are the tracks that spend most of their time behind the counter; they never count as occupancy,
queue or visits.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from bree.intel.flowtrack import FlowConfig, FlowTracker, Track, foot_of
from bree.intel.layout import Layout


@dataclass
class IntelConfig:
    sample_s: float = 5.0
    smooth_n: int = 5             # median over this many samples
    zone_gap_s: float = 2.0       # a zone visit survives this long outside the zone
    zone_min_s: float = 1.0       # visits shorter than this are not reported
    still_px_per_s: float = 0.25  # speed below this many box heights per second counts as standing still
    staff_frac: float = 0.5       # share of a track's observations behind the counter that makes it staff
    staff_min_obs: int = 3
    initial_window_s: float = 6.0
    weak_min_obs: int = 8         # 'born' / 'lost' door evidence needs a track seen this many samples (ghosts flicker 3-4)
    side_persist: int = 3         # samples on a side of the door line before it counts as the track's side
    browse_min_s: float = 4.0     # time in browse zones that counts as "visited shelf"
    register_min_s: float = 3.0   # time in the register zone that counts as "went to register"
    track: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class _TS:
    """Per-track state."""
    first_t: float
    first_foot: np.ndarray
    recent: deque = field(default_factory=lambda: deque(maxlen=5))
    sides: list = field(default_factory=list)       # [(t, +1/-1)] changes only
    side: int = 0
    cand: int = 0                                    # side the track is moving to, and for how many samples
    cand_n: int = 0
    obs: int = 0
    staff_obs: int = 0
    zone_open: dict = field(default_factory=dict)   # zone name -> [t_in, t_last, still_s, kind]
    zone_done: list = field(default_factory=list)
    door_events: list = field(default_factory=list)  # (t, dir, via)
    born_on_mats: bool = False
    ever_inside: bool = False
    last_on_mats_t: float = -1.0
    last_foot: np.ndarray | None = None
    last_t: float = 0.0
    path_px: float = 0.0
    entered: bool = False
    exited: bool = False
    zone_flags: set = field(default_factory=set)


class FlowEngine:
    def __init__(self, layout: Layout, cfg: IntelConfig | None = None, start: datetime | None = None):
        self.L, self.cfg = layout, cfg or IntelConfig()
        self.tracker = FlowTracker(self.cfg.track)
        self.tracker.cfg.ignore = list(self.tracker.cfg.ignore) + [p.tolist() for p in layout.ignore]
        self.start = start
        self.ts: dict[int, _TS] = {}
        self._closed_n = 0
        self._next_sample = 0.0
        self.entries = self.exits = 0
        self.initial: int | None = None
        self.occupancy = 0
        self._staff_ids: set[int] = set()
        self.heat = {"customer": np.zeros((36, 64)), "staff": np.zeros((36, 64))}
        self._last_t: float | None = None
        self.mats = layout.zone_by_kind("door")
        self.registers = layout.zone_by_kind("register")
        self.staff_zones = layout.zone_by_kind("staff")

    # ---- time -----------------------------------------------------------------------------------------------
    def wall(self, t: float) -> str | None:
        if self.start is None:
            return None
        return (self.start + timedelta(seconds=t)).isoformat(timespec="seconds")

    # ---- per-step -------------------------------------------------------------------------------------------
    def step(self, t: float, persons: list[dict]) -> list[dict]:
        cfg, L = self.cfg, self.L
        out: list[dict] = []
        dt = (t - self._last_t) if self._last_t is not None else 0.0
        self._last_t = t
        seen = self.tracker.step(t, persons)
        for tr in seen:
            st = self.ts.get(tr.id)
            if st is None:
                st = self.ts[tr.id] = _TS(tr.t0, np.array(tr.path[0][1:3]))
                self._on_birth(tr, st)
            st.recent.append(tr.foot.copy())
            fx, fy = np.median(np.stack(st.recent), 0)
            st.obs += 1
            in_staff = any(z.contains(fx, fy) for z in self.staff_zones)
            st.staff_obs += in_staff
            if st.last_foot is not None:
                st.path_px += float(np.linalg.norm([fx, fy] - st.last_foot))
                speed = np.linalg.norm([fx, fy] - st.last_foot) / max(t - st.last_t, 1e-3) / tr.h
            else:
                speed = 9.0
            st.last_foot, st.last_t = np.array([fx, fy]), t
            if tr.id in self._staff_ids or self._is_staff(st):
                self._staff_ids.add(tr.id)
            role = "staff" if tr.id in self._staff_ids else "customer"
            gx, gy = min(int(fx / L.size[0] * 64), 63), min(int(fy / L.size[1] * 36), 36 - 1)
            if 0 <= gx < 64 and 0 <= gy < 36:
                self.heat[role][gy, gx] += dt if dt < 3 else 0.0
            self._door(tr, st, fx, fy, t, out)
            self._zones(tr, st, fx, fy, speed, t, out, role)
        # new closures
        closed = self.tracker.closed
        while self._closed_n < len(closed):
            out += self._close(closed[self._closed_n], t)
            self._closed_n += 1
        if self.initial is None and t >= self.cfg.initial_window_s:
            self.initial = len([1 for tr in seen if tr.id not in self._staff_ids and not self.ts[tr.id].entered])
            self.occupancy = max(0, self.initial + self.entries - self.exits)
        while t >= self._next_sample:
            out.append(self._sample(self._next_sample, seen))
            self._next_sample += cfg.sample_s
        return out

    def _is_staff(self, st: _TS) -> bool:
        return st.obs >= self.cfg.staff_min_obs and st.staff_obs / st.obs >= self.cfg.staff_frac and not st.entered

    def _on_birth(self, tr: Track, st: _TS) -> None:
        x, y = st.first_foot
        if any(z.contains(x, y) for z in self.mats) and self.L.door.signed(x, y) > -self.L.door.margin:
            st.born_on_mats = True

    # ---- door -----------------------------------------------------------------------------------------------
    def _door(self, tr: Track, st: _TS, x: float, y: float, t: float, out: list) -> None:
        d = self.L.door
        if d.in_span(x, y):
            s = d.signed(x, y)
            side = 1 if s > d.margin else -1 if s < -d.margin else 0
            if side and side != st.side:
                st.cand_n = st.cand_n + 1 if st.cand == side else 1
                st.cand = side
                if st.cand_n >= self.cfg.side_persist or not st.side:
                    if st.side:
                        kind = "in" if side > 0 else "out"
                        self._door_event(tr.id, st, t, kind, "line", out)
                    st.side = side
            elif side:
                st.cand_n = 0
        if st.side > 0 or d.signed(x, y) > d.margin:
            st.ever_inside = st.ever_inside or d.signed(x, y) > d.margin
        if any(z.contains(x, y) for z in self.mats):
            st.last_on_mats_t = t
        # born on the mats and now clearly inside, no line crossing seen: the glare hid the approach
        if st.born_on_mats and not st.entered and not st.door_events and d.signed(x, y) > 3 * d.margin \
                and st.obs >= self.cfg.weak_min_obs and not (tr.id in self._staff_ids):
            self._door_event(tr.id, st, t, "in", "born", out)
            st.side = 1

    def _door_event(self, tid: int, st: _TS, t: float, kind: str, via: str, out: list) -> None:
        if tid in self._staff_ids and via != "line":
            return
        st.door_events.append((t, kind, via))
        if kind == "in":
            st.entered = True
            self.entries += 1
        else:
            st.exited = True
            self.exits += 1
        if self.initial is not None:
            self.occupancy = max(0, self.initial + self.entries - self.exits)
        out.append({"type": "door", "dir": kind, "t": round(t, 2), "ts": self.wall(t), "track": tid, "via": via})

    # ---- zones ----------------------------------------------------------------------------------------------
    def _zones(self, tr, st, x, y, speed, t, out, role) -> None:
        cfg = self.cfg
        here = {z.name: z for z in self.L.zones_at(x, y)}
        for name, z in here.items():
            o = st.zone_open.get(name)
            if o is None:
                st.zone_open[name] = [t, t, 0.0, z.kind]
            else:
                if speed < cfg.still_px_per_s:
                    o[2] += t - o[1]
                o[1] = t
        for name in list(st.zone_open):
            o = st.zone_open[name]
            if name not in here and t - o[1] > cfg.zone_gap_s:
                self._end_zone(tr.id, st, name, role, out)

    def _end_zone(self, tid, st, name, role, out) -> None:
        t_in, t_last, still, kind = st.zone_open.pop(name)
        dwell = t_last - t_in
        if dwell >= self.cfg.zone_min_s and role == "customer" and kind != "staff":
            rec = {"type": "zone", "track": tid, "zone": name, "kind": kind, "t_in": round(t_in, 2), "t_out": round(t_last, 2),
                   "ts": self.wall(t_in), "dwell_s": round(dwell, 2), "still_s": round(min(still, dwell), 2)}
            st.zone_done.append(rec)
            out.append(rec)

    # ---- track close, sample --------------------------------------------------------------------------------
    def _close(self, tr: Track, now: float) -> list[dict]:
        st = self.ts.get(tr.id)
        out: list[dict] = []
        if st is None:
            return out
        t1 = tr.path[-1][0]
        role = "staff" if (tr.id in self._staff_ids or self._is_staff(st)) else "customer"
        for name in list(st.zone_open):
            self._end_zone(tr.id, st, name, role, out)
        # exit seen only as "last on the mats after being inside"
        if role == "customer" and st.ever_inside and not st.exited and st.last_on_mats_t >= 0 and st.obs >= self.cfg.weak_min_obs \
                and t1 - st.last_on_mats_t <= 1.0 and self._ends_towards_door(tr):
            self._door_event(tr.id, st, t1, "out", "lost", out)
        zk = {}
        for z in st.zone_done:
            zk[z["kind"]] = zk.get(z["kind"], 0.0) + z["dwell_s"]
        out.append({"type": "track", "track": tr.id, "role": role, "t0": round(tr.t0, 2), "t1": round(t1, 2),
                    "ts": self.wall(tr.t0), "obs": st.obs, "path_px": round(st.path_px), "entered": st.entered, "exited": st.exited,
                    "browse_s": round(zk.get("browse", 0.0), 1), "register_s": round(zk.get("register", 0.0), 1),
                    "outcome": self._outcome(st, zk) if role == "customer" else None})
        return out

    def _ends_towards_door(self, tr: Track) -> bool:
        pts = [p for p in tr.path if p[0] >= tr.path[-1][0] - 3.0]
        if len(pts) < 2:
            return False
        d = self.L.door
        return d.signed(pts[-1][1], pts[-1][2]) < d.signed(pts[0][1], pts[0][2]) - 1

    def _outcome(self, st: _TS, zk: dict) -> str:
        """Conversion proxy for one customer walk. Proxies: the camera sees a part of the store and a person can be split into
        two tracks, so 'left_without_register' is an upper bound on walk-outs without a till visit, not a measurement."""
        shelf = zk.get("browse", 0.0) >= self.cfg.browse_min_s
        reg = zk.get("register", 0.0) >= self.cfg.register_min_s
        if reg:
            return "registered" if shelf or st.entered else "registered"
        if st.exited and shelf:
            return "left_without_register"
        if st.exited:
            return "passed_through" if not shelf else "left_without_register"
        if shelf:
            return "browsed_not_seen_leaving"
        return "seen_briefly"

    def _sample(self, t: float, seen: list[Track]) -> dict:
        cust = [tr for tr in seen if tr.id not in self._staff_ids]
        queue = 0
        for tr in cust:
            st = self.ts[tr.id]
            if st.last_foot is not None and any(z.contains(*st.last_foot) for z in self.registers):
                queue += 1
        return {"type": "sample", "t": round(t, 2), "ts": self.wall(t), "in_view": len(seen), "customers": len(cust),
                "staff": len(seen) - len(cust), "queue": queue, "occupancy": self.occupancy if self.initial is not None else None}

    def finish(self) -> list[dict]:
        out: list[dict] = []
        t = self._last_t or 0.0
        n0 = len(self.tracker.closed)
        self.tracker.finish()
        for tr in self.tracker.closed[self._closed_n:]:
            out += self._close(tr, t)
        self._closed_n = len(self.tracker.closed)
        return out


def run_detections(frames: list[dict], layout: Layout, cfg: IntelConfig | None = None, start: datetime | None = None):
    """Run the engine over cached detections. Returns (records, engine)."""
    eng = FlowEngine(layout, cfg, start)
    recs: list[dict] = []
    for f in frames:
        recs += eng.step(f["t"], f["persons"])
    recs += eng.finish()
    return recs, eng
