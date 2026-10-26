"""Replay cached perception (bree.theft.cache) through gate -> event engine -> ledger, POS-less.

This is what `bree theft-shadow` runs per segment, and what the false-alert measurements and the gate tuning use:
the detector never runs again, so a pass over 20 minutes of footage takes seconds.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bree.alerts.types import Alert
from bree.events.types import Event, EventType
from bree.ledger import build_ledger
from bree.theft.cache import load_frames, load_info
from bree.theft.gate import GateConfig, ObsGate
from bree.theft.station import Station
from bree.theft.tiers import POSLESS_LEDGER


@dataclass
class ReplayResult:
    camera: str
    segment: str
    duration_s: float
    alerts: list[Alert] = field(default_factory=list)       # final state (a retraction replaces its alert)
    events: list[Event] = field(default_factory=list)
    event_counts: dict = field(default_factory=dict)
    gate_stats: dict = field(default_factory=dict)
    persons_raw: int = 0
    persons_engine: int = 0
    engine_log: list[str] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)


def replay_cache(cache_dir: str | Path, station: Station, gate: GateConfig | None = None,
                 ledger_overrides: dict | None = None, rules: dict | None = None, camera: str | None = None,
                 gated: bool = True) -> ReplayResult:
    from bree.track.multicam import StoreEvents
    info = load_info(cache_dir)
    cam = camera or station.name
    store = station.store
    if rules:
        from dataclasses import replace
        store = replace(store, rules={**store.rules, **rules})
    fusion = StoreEvents({cam: store})
    ledger = build_ledger(store, **{**POSLESS_LEDGER, **(ledger_overrides or {})})
    g = ObsGate(gate or GateConfig(**station.gate), station.staff, station.ignore) if gated else None
    raw_ids, eng_ids, events, alerts = set(), set(), [], []
    for obs in load_frames(cache_dir):
        raw_ids.update(p.track_id for p in obs.persons)
        if g is not None:
            obs = g.update(obs)
        eng_ids.update(p.track_id for p in obs.persons)
        evs = fusion.update(cam, obs)
        for ev in evs:
            alerts += ledger.on_event(ev)
        events += evs
        alerts += ledger.tick(obs.t)
    for ev in fusion.flush():
        alerts += ledger.on_event(ev)
        events.append(ev)
    alerts += ledger.finalize()
    final = {a.retracts: a.tier for a in alerts if a.retracts}
    from dataclasses import replace as rep
    alerts = [rep(a, tier=final.get(a.alert_id, a.tier)) for a in alerts if not a.retracts]
    alerts = [a for a in alerts if a.tier in ("alert", "review")]
    return ReplayResult(
        camera=cam, segment=Path(cache_dir).name, duration_s=info.duration_s, alerts=alerts, events=events,
        event_counts=dict(Counter(e.type.value for e in events)), gate_stats=dict(g.stats) if g else {},
        persons_raw=len(raw_ids), persons_engine=len(eng_ids), engine_log=list(fusion.engines[cam].log),
        decisions=list(ledger.decisions))
