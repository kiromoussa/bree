"""Turn a pipeline run (frames.jsonl from `bree run`) into store signals.

This is a *consumer* of the perception output, with deliberately conservative filters, because the
real-footage baselines showed the raw tracker churns: dozens of ids for a handful of people, ghost
boxes on displays, staff behind glass counted as customers. So a track only counts when it:
  - lives at least `min_track_s` and `min_frames`,
  - has median confidence >= `min_conf`,
  - is not a stationary low-confidence ghost (moves < `ghost_move_px` over >= `ghost_min_s` at
    median confidence < `ghost_conf`).
Counts after these filters are still REAL-UNLABELLED unless compared against labels (see
docs/ultra/product.md for the one comparison we have, REAL-MODEL-LABELLED, tiny sample).

Roles decide which signals a camera may produce (a camera that cannot see the door must not
produce door counts; the engine's per-track-birth `enter` event is NOT used for that reason):
  door     occupancy, door_in/out (track born/ended near a door zone), dwell, queue
  counter  occupancy, queue, dwell
  aisle    occupancy, dwell
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from bree.product.signals import Signal

ROLES = {"door": ("occupancy", "door", "dwell", "queue"),
         "counter": ("occupancy", "dwell", "queue"),
         "aisle": ("occupancy", "dwell")}


@dataclass
class AdaptConfig:
    store: str = "store"
    camera: str = "cam0"
    role: str = "aisle"
    source: str = "real_unlabelled"
    min_track_s: float = 2.0
    min_frames: int = 8
    min_conf: float = 0.45
    ghost_move_px: float = 40.0
    ghost_min_s: float = 30.0
    ghost_conf: float = 0.6
    sample_s: float = 5.0
    dwell_min_s: float = 3.0
    dwell_gap_s: float = 2.0
    door_margin_px: float = 150.0
    queue_zone: str | None = None        # zone name whose occupants are "waiting"
    staff_zone: str | None = None        # zone name behind the counter: tracks mostly in it are staff
    dwell_kinds: tuple = ("shelf", "cooler", "register")
    health_every_s: float = 30.0
    clock_offset_s: float = 0.0          # added to ts: fixes a known DVR/recorder clock error
    extra_meta: dict = field(default_factory=dict)


def segment_start(path: str | Path, tz=None) -> float:
    """Epoch of a recorder file name YYYYmmdd_HHMMSS(.mp4) read in `tz` (default: this machine's zone;
    the recorder names files in machine local time)."""
    m = re.search(r"(\d{8})_(\d{6})", Path(path).name)
    if not m:
        raise ValueError(f"no YYYYmmdd_HHMMSS in {path}")
    d = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return d.replace(tzinfo=tz or dt.datetime.now().astimezone().tzinfo).timestamp()


class _Track:
    __slots__ = ("id", "pts", "confs")

    def __init__(self, tid):
        self.id = tid
        self.pts: list[tuple[float, float, float]] = []   # t, foot x, foot y
        self.confs: list[float] = []

    @property
    def t0(self):
        return self.pts[0][0]

    @property
    def t1(self):
        return self.pts[-1][0]


def _tracks(frames_jsonl: str | Path):
    tr: dict[int, _Track] = {}
    last_t = 0.0
    with open(frames_jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                fr = json.loads(line)
            except ValueError:
                continue
            t = float(fr["t"])
            last_t = max(last_t, t)
            for p in fr.get("persons", []):
                x1, y1, x2, y2 = p["bbox"]
                k = tr.setdefault(p["id"], _Track(p["id"]))
                k.pts.append((t, (x1 + x2) / 2, y2))
                k.confs.append(float(p.get("conf", 1.0)))
    return tr, last_t


def _is_ghost(k: _Track, cfg: AdaptConfig) -> bool:
    if k.t1 - k.t0 < cfg.ghost_min_s or statistics.median(k.confs) >= cfg.ghost_conf:
        return False
    xs = [p[1] for p in k.pts]
    ys = [p[2] for p in k.pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < cfg.ghost_move_px


def qualify(tracks: dict[int, _Track], cfg: AdaptConfig) -> tuple[dict[int, _Track], dict]:
    keep, why = {}, {"tracks": len(tracks), "short": 0, "low_conf": 0, "ghost": 0}
    for tid, k in tracks.items():
        if k.t1 - k.t0 < cfg.min_track_s or len(k.pts) < cfg.min_frames:
            why["short"] += 1
        elif statistics.median(k.confs) < cfg.min_conf:
            why["low_conf"] += 1
        elif _is_ghost(k, cfg):
            why["ghost"] += 1
        else:
            keep[tid] = k
    why["kept"] = len(keep)
    return keep, why


def _at(k: _Track, t: float):
    """Foot point of the track sample nearest to t (None if t is outside its life)."""
    if t < k.t0 or t > k.t1:
        return None
    lo, hi = 0, len(k.pts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if k.pts[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    return k.pts[lo]


def _share_in(k: _Track, zone) -> float:
    return sum(1 for _, x, y in k.pts if zone.contains(x, y)) / len(k.pts)


def adapt_run(frames_jsonl: str | Path, start_ts: float, cfg: AdaptConfig, store_cfg=None) -> tuple[list[Signal], dict]:
    """Signals for one run + a small report of what was filtered. `store_cfg` is a bree StoreConfig
    (zones in the camera's pixel space) or None (then no door, dwell or queue signals)."""
    if cfg.role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    allow = ROLES[cfg.role]
    tracks, dur = _tracks(frames_jsonl)
    keep, rep = qualify(tracks, cfg)
    base = start_ts + cfg.clock_offset_s
    meta0 = {"role": cfg.role, **cfg.extra_meta}

    def sig(t, kind, value=1.0, zone=None, **meta):
        return Signal(ts=base + t, kind=kind, value=value, store=cfg.store, camera=cfg.camera, zone=zone,
                      source=cfg.source, meta={**meta0, **meta})

    out: list[Signal] = []
    t = 0.0
    while t <= dur:
        out.append(sig(t, "health", 1.0))
        t += cfg.health_every_s

    staff: set[int] = set()
    qz = sz = None
    if store_cfg is not None:
        names = {z.name: z for z in store_cfg.zones}
        sz = names.get(cfg.staff_zone) if cfg.staff_zone else None
        qz = names.get(cfg.queue_zone) if cfg.queue_zone else None
        if sz is not None:
            staff = {tid for tid, k in keep.items() if _share_in(k, sz) > 0.5}
    rep["staff_tracks"] = len(staff)
    customers = {tid: k for tid, k in keep.items() if tid not in staff}

    if "occupancy" in allow:
        t = 0.0
        while t <= dur:
            n = sum(1 for k in customers.values() if k.t0 <= t <= k.t1)
            out.append(sig(t, "occupancy", n))
            if "queue" in allow and qz is not None:
                q = sum(1 for k in customers.values() if (p := _at(k, t)) and qz.contains(p[1], p[2]))
                out.append(sig(t, "queue", q))
            t += cfg.sample_s

    if store_cfg is not None and "door" in allow:
        doors = store_cfg.zones_of("exit", "entrance")
        for tid, k in customers.items():
            if doors:
                d_in = min(z.distance(k.pts[0][1], k.pts[0][2]) for z in doors)
                d_out = min(z.distance(k.pts[-1][1], k.pts[-1][2]) for z in doors)
            else:
                d_in = d_out = float("inf")
            if d_in <= cfg.door_margin_px:
                out.append(sig(k.t0, "door_in", 1, track=tid))
            # a track still alive when the clip ends has not left
            if d_out <= cfg.door_margin_px and k.t1 < dur - 1.0:
                out.append(sig(k.t1, "door_out", 1, track=tid))

    if store_cfg is not None and "dwell" in allow:
        zones = [z for z in store_cfg.zones if z.kind in cfg.dwell_kinds]
        for tid, k in customers.items():
            for z in zones:
                run_start = run_end = None
                for tt, x, y in k.pts + [(float("inf"), 0, 0)]:
                    inside = tt != float("inf") and z.contains(x, y)
                    if inside:
                        if run_start is None:
                            run_start = tt
                        run_end = tt
                    elif run_start is not None and (tt - run_end > cfg.dwell_gap_s):
                        if run_end - run_start >= cfg.dwell_min_s:
                            out.append(sig(run_end, "dwell", round(run_end - run_start, 1), zone=z.name, track=tid))
                        run_start = run_end = None
    out.sort(key=lambda s: s.ts)
    rep["signals"] = len(out)
    return out, rep
