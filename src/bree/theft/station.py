"""Per-camera theft settings: the `theft:` block of a store YAML (ignored by bree.events.zones.load_store_config).

  theft:
    role: door | counter | aisle | cooler | outdoor      what the camera is for (which detectors make sense on it)
    staff_zones:  [{name, polygon}]    people standing here are staff (hidden from the engine, bree.theft.gate)
    ignore_zones: [{name, polygon}]    detections here are never people (reflections, posters, screens)
    gate:   {min_age_s: 1.0, ...}      bree.theft.gate.GateConfig overrides
    hours:  {open: "06:00", close: "23:00", clock_offset_s: 0}   after-hours rules (bree.theft.hours)
    tamper: {...}                      bree.theft.tamper.TamperConfig overrides
    sweep:  {...}                      bree.theft.sweep.SweepConfig overrides

Polygons are in the camera's own pixels, like `zones:`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from bree.events.zones import StoreConfig, Zone, load_store_config

ROLES = ("door", "counter", "aisle", "cooler", "outdoor")


@dataclass
class Station:
    store: StoreConfig
    name: str
    role: str = "aisle"
    staff: list[Zone] = field(default_factory=list)
    ignore: list[Zone] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    hours: dict = field(default_factory=dict)
    tamper: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)
    loiter: dict = field(default_factory=dict)


def _zones(items, kind: str) -> list[Zone]:
    return [Zone(z["name"], kind, np.asarray(z["polygon"], dtype=float)) for z in items or []]


def load_station(path: str | Path, name: str | None = None) -> Station:
    raw = yaml.safe_load(Path(path).read_text())
    t = raw.get("theft") or {}
    role = t.get("role", "aisle")
    if role not in ROLES:
        raise ValueError(f"theft.role must be one of {ROLES}, got {role!r}")
    return Station(store=load_store_config(path), name=name or Path(path).stem, role=role,
                   staff=_zones(t.get("staff_zones"), "shelf"), ignore=_zones(t.get("ignore_zones"), "shelf"),
                   gate=t.get("gate") or {}, hours=t.get("hours") or {}, tamper=t.get("tamper") or {},
                   sweep=t.get("sweep") or {}, loiter=t.get("loiter") or {})
