"""Station config: the few facts the product needs about one store (name, time zone, cameras, thresholds)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from bree.product.notify import DEFAULT_RULES, Rule


@dataclass
class Camera:
    id: str
    label: str = ""
    role: str = "aisle"           # door | counter | aisle   (see bree.product.adapt)
    store_yaml: str | None = None  # BREE zone file for this camera (needed for door/dwell/queue signals)
    queue_zone: str | None = None
    staff_zone: str | None = None


@dataclass
class Station:
    name: str = "My store"
    store_id: str = "store"
    tz: str | None = None
    tz_assumed: bool = False       # True = nobody has confirmed the zone / clock yet
    clock_offset_s: float = 0.0
    cameras: dict[str, Camera] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=lambda: list(DEFAULT_RULES))
    notes: list[str] = field(default_factory=list)   # shown on every digest (data caveats)

    def label(self, cam: str) -> str:
        c = self.cameras.get(cam)
        return (c.label if c and c.label else cam)


def load_station(path: str | Path | None) -> Station:
    if path is None:
        return Station()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cams = {}
    for cid, c in (raw.get("cameras") or {}).items():
        c = c or {}
        cams[str(cid)] = Camera(str(cid), c.get("label", ""), c.get("role", "aisle"), c.get("store_yaml"),
                                c.get("queue_zone"), c.get("staff_zone"))
    rules = list(DEFAULT_RULES)
    if "rules" in raw:
        by = {r.name: r for r in rules}
        for o in raw["rules"] or []:
            base = by.get(o.get("name"))
            by[o["name"]] = Rule(**{**(base.__dict__ if base else {}), **o})
        rules = list(by.values())
    return Station(name=raw.get("name", "My store"), store_id=raw.get("store_id", "store"), tz=raw.get("tz"),
                   tz_assumed=bool(raw.get("tz_assumed", False)), clock_offset_s=float(raw.get("clock_offset_s", 0.0)),
                   cameras=cams, rules=rules, notes=list(raw.get("notes") or []))
