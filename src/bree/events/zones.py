"""Store layout: named zones as polygons in camera pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from bree.events.types import Catalog

ZONE_KINDS = ("shelf", "cooler", "register", "exit")


@dataclass
class Zone:
    name: str
    kind: str                         # one of ZONE_KINDS
    polygon: np.ndarray               # (N, 2) float, pixel coords

    def contains(self, x: float, y: float) -> bool:
        """Even-odd ray casting point-in-polygon test."""
        poly = self.polygon
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def distance(self, x: float, y: float) -> float:
        """0 if inside, else distance to the nearest edge (pixels)."""
        if self.contains(x, y):
            return 0.0
        p = np.array([x, y], dtype=float)
        best = np.inf
        poly = self.polygon
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ab = b - a
            t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    @property
    def is_merch(self) -> bool:
        return self.kind in ("shelf", "cooler")


@dataclass
class StoreConfig:
    name: str
    camera_id: str
    resolution: tuple[int, int]
    fps: float
    zones: list[Zone]
    terminals: dict[str, str]                       # terminal -> zone name
    catalog: Catalog
    rules: dict = field(default_factory=dict)       # event-engine thresholds
    ledger: dict = field(default_factory=dict)      # LedgerConfig overrides
    product_classes: dict[str, str] = field(default_factory=dict)  # detector class -> category

    def zone(self, name: str) -> Zone:
        return next(z for z in self.zones if z.name == name)

    def zones_of(self, *kinds: str) -> list[Zone]:
        return [z for z in self.zones if z.kind in kinds]

    def zone_at(self, x: float, y: float, *kinds: str) -> Zone | None:
        for z in self.zones:
            if (not kinds or z.kind in kinds) and z.contains(x, y):
                return z
        return None

    @property
    def zone_kinds(self) -> dict[str, str]:
        return {z.name: z.kind for z in self.zones}


def load_store_config(path: str | Path) -> StoreConfig:
    raw = yaml.safe_load(Path(path).read_text())
    zones = []
    for z in raw["zones"]:
        if z["kind"] not in ZONE_KINDS:
            raise ValueError(f"zone {z['name']}: kind must be one of {ZONE_KINDS}")
        zones.append(Zone(z["name"], z["kind"], np.asarray(z["polygon"], dtype=float)))
    cam = raw.get("camera", {})
    catalog = {sku: info["category"] for sku, info in raw.get("catalog", {}).items()}
    return StoreConfig(
        name=raw.get("store", {}).get("name", Path(path).stem),
        camera_id=cam.get("id", "cam0"),
        resolution=tuple(cam.get("resolution", [1280, 720])),
        fps=float(cam.get("fps", 15)),
        zones=zones,
        terminals=raw.get("terminals", {}),
        catalog=Catalog(catalog),
        rules=raw.get("rules", {}),
        ledger=raw.get("ledger", {}),
        product_classes=raw.get("product_classes", {}),
    )
