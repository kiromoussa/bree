"""Shadow mode over recorded real footage: every detector, one review queue, per-camera rates.

For each cached segment (bree.theft.cache): gate -> event engine -> POS-less ledger, plus the after-hours, loitering and
sweep observers on the same gated observations, plus the tamper monitor on the image-health features. Every finding becomes
an `Incident`, the queue is ranked (bree.theft.incidents), head-pixelated clips are cut for the top items, and the records
go into a normal shadow folder (`would_be_alerts.jsonl`) that `bree shadow-labels --serve` / the review page already show.
`rates()` turns the per-segment results into incidents per camera-hour with the honest bound: with zero incidents in `h`
camera-hours the 95% upper bound on the rate is about 3/h (rule of three), so "0 in 20 minutes" says nothing about a
target of 0.1 per hour.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bree.shadow import ShadowLog
from bree.theft.cache import load_health, load_info
from bree.theft.clips import write_clip
from bree.theft.gate import GateConfig
from bree.theft.hours import AfterHoursMonitor, HoursConfig, LoiterConfig, LoiterMonitor, StoreClock
from bree.theft.incidents import Incident, PriorityModel, from_alert, from_sweep, from_tamper, rank
from bree.theft.replay import ReplayResult, replay_cache
from bree.theft.station import Station
from bree.theft.sweep import SweepConfig, SweepDetector
from bree.theft.tamper import TamperConfig, TamperMonitor
from bree.theft.tiers import band_of


@dataclass
class SegmentRun:
    camera: str
    segment: str
    duration_s: float
    replay: ReplayResult
    incidents: list[Incident] = field(default_factory=list)
    tamper_windows: list[tuple[float, float]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def tamper_events(cache_dir: str | Path, station: Station, clock: StoreClock | None = None, cfg: TamperConfig | None = None):
    """TamperMonitor over a segment's cached health features and person boxes."""
    info, h = load_info(cache_dir), load_health(cache_dir)
    from bree.theft.cache import load_frames
    frames = load_frames(cache_dir)
    names = h["names"]
    c = cfg or TamperConfig(**{k: v for k, v in station.tamper.items() if k in TamperConfig.__dataclass_fields__})
    mon = TamperMonitor(c, clock=(clock.local if clock and clock.known else None))
    ev, t_last = [], 0.0
    for k, t in enumerate(h["t"]):
        obs = next(frames, None)
        boxes = [(p.bbox[0] / info.width, p.bbox[1] / info.height, p.bbox[2] / info.width, p.bbox[3] / info.height)
                 for p in (obs.persons if obs else [])]
        f = {n: float(h["x"][k][j]) for j, n in enumerate(names)}
        ev += mon.update(float(t), f, h["thumbs"][k], boxes)
        t_last = float(t)
    return ev + mon.flush(t_last)


def run_segment(cache_dir: str | Path, station: Station, clock: StoreClock | None = None, gate: GateConfig | None = None,
                camera: str | None = None, with_tamper: bool = True) -> SegmentRun:
    cam = camera or station.name
    clock = clock or StoreClock()
    obs = {"sweep": SweepDetector(station.store, SweepConfig(**station.sweep)),
           "loiter": LoiterMonitor(station.role, LoiterConfig(**station.loiter), exempt=station.staff)}
    if station.hours:
        obs["after_hours"] = AfterHoursMonitor(clock, HoursConfig.from_dict(station.hours), outdoor=station.role == "outdoor")
    rep = replay_cache(cache_dir, station, gate=gate, camera=cam, observers=obs)
    inc: list[Incident] = [from_alert(a, cam) for a in rep.alerts]
    inc += [from_sweep(e, cam) for e in rep.observed["sweep"]]
    inc += [Incident(i.kind, i.t, i.t_start, i.severity, i.headline, i.details, camera=cam) for i in rep.observed["loiter"]]
    inc += [Incident(i.kind, i.t, i.t_start, i.severity, i.headline, i.details, camera=cam) for i in rep.observed.get("after_hours", [])]
    windows = []
    if with_tamper:
        tev = tamper_events(cache_dir, station, clock)
        starts = [e for e in tev if e.phase == "start"]
        inc += [from_tamper(e, cam) for e in starts]
        windows = [(e.t_start, e.t) for e in tev if e.phase == "end"] + [(e.t_start, rep.duration_s) for e in starts
                                                                           if not any(x.phase == "end" and x.kind == e.kind for x in tev)]
    if clock.known:
        for i in inc:
            i.wall = clock.start_epoch + i.t
    return SegmentRun(cam, rep.segment, rep.duration_s, rep, inc, windows,
                      {"gate": rep.gate_stats, "events": rep.event_counts, "persons_raw": rep.persons_raw,
                       "persons_engine": rep.persons_engine, "sweep": obs["sweep"].stats})


def rates(runs: list[SegmentRun]) -> dict:
    """Incidents per camera-hour by kind, with the rule-of-three bound when none were seen. REAL-UNLABELLED: nobody has
    said which of these were worth a look."""
    hours: dict[str, float] = defaultdict(float)
    n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in runs:
        hours[r.camera] += r.duration_s / 3600
        for i in r.incidents:
            n[r.camera][i.kind] += 1
    kinds = sorted({k for c in n.values() for k in c} | {"theft_review", "tamper", "after_hours", "loiter", "bulk_grab"})
    out = {"camera_hours": round(sum(hours.values()), 3), "cameras": {}}
    for cam, h in sorted(hours.items()):
        out["cameras"][cam] = {"hours": round(h, 3), "counts": {k: n[cam].get(k, 0) for k in kinds},
                               "per_hour": {k: round(n[cam].get(k, 0) / h, 2) for k in kinds}}
    tot = {k: sum(n[c].get(k, 0) for c in n) for k in kinds}
    H = sum(hours.values())
    out["all"] = {"counts": tot, "per_hour": {k: round(v / H, 2) if H else None for k, v in tot.items()},
                  "upper95_per_hour_if_zero": round(3 / H, 2) if H else None}
    anyk = sum(tot.values())
    out["all"]["incidents"] = anyk
    out["all"]["upper95_per_hour_all_kinds"] = round(_poisson_upper(anyk) / H, 2) if H else None
    out["note"] = ("REAL-UNLABELLED: these are things the detectors flagged, not confirmed problems. Hours are camera-hours "
                   "of recorded footage, one afternoon, one store. An upper bound of 3/H follows from seeing none in H hours.")
    return out


def _poisson_upper(k: int, conf: float = 0.95) -> float:
    """Exact upper limit for a Poisson mean given k events (chi-square quantile via a bisection on the cdf)."""
    lo, hi = float(k), float(k) + 10 + 4 * math.sqrt(k + 1)
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = sum(math.exp(-mid) * mid ** j / math.factorial(j) for j in range(k + 1))
        lo, hi = (mid, hi) if cdf > 1 - conf else (lo, mid)
    return round((lo + hi) / 2, 3)


def write_review(runs: list[SegmentRun], out_dir: str | Path, videos: dict[tuple[str, str], Path],
                 caches: dict[tuple[str, str], Path], stations: dict[str, Station], model: PriorityModel | None = None,
                 top_clips: int = 40, session: str = "replay") -> ShadowLog:
    """Rank everything, cut clips for the top `top_clips` items, append records to a ShadowLog in out_dir."""
    log = ShadowLog(out_dir)
    out = Path(out_dir)
    queue = []
    for r in runs:
        tam = {r.camera: r.tamper_windows}
        for i in rank(r.incidents, model, tampered=tam):
            queue.append((r, i))
    queue.sort(key=lambda x: -x[1].priority)
    for n, (r, i) in enumerate(queue):
        rid = f"{r.camera}-{r.segment}-{n:03d}"
        clip = None
        if n < top_clips and (r.camera, r.segment) in videos:
            c = write_clip(videos[(r.camera, r.segment)], caches[(r.camera, r.segment)],
                           out / r.camera / r.segment / "alerts" / f"{rid}.mp4", i.t_start - 4, i.t + 4,
                           stations.get(r.camera), label=f"{i.kind} {i.band or ''}".strip())
            clip = str(c.relative_to(out)) if c else None
        wall = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(i.wall)) if i.wall else f"{r.segment} +{i.t:.0f}s"
        log.add_alert({
            "id": rid, "retracts": None, "time": wall, "camera": r.camera, "person_id": int(i.details.get("track", 0)),
            "group": [], "tier": "review", "confidence": round(i.priority / 100, 3),
            "concealment_seen": bool(i.details.get("concealment_seen")), "basket": i.details.get("items", []),
            "unpaid": [{"category": c} for c in i.details.get("items", [])], "paid": [],
            "visited_register": bool(i.details.get("visited_register")), "t_exit": round(i.t, 2), "clip": clip,
            "reasons": [i.headline] + i.priority_why, "audit_log": [str(i.details)],
            "kind": i.kind, "headline": i.headline, "priority": i.priority, "priority_why": i.priority_why, "band": i.band,
        })
    return log
