"""Store layout: named zones as polygons in camera pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from bree.events.types import Catalog

# "exit" is the door (people come in and leave through it). "entrance" is a one-way way in: closed-world
# identity creates new people there too, but nobody is counted as leaving through it.
ZONE_KINDS = ("shelf", "cooler", "register", "exit", "entrance")


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
    # Multi-camera stores only: 4+ [x_px, y_px, x_m, y_m] marks (a pixel in this camera and the
    # same spot on the shared store floor plan, metres). None = single camera.
    floor_points: list | None = None
    # `camera.calibration`, the block scripts/calibrate.py writes (full pose). Used by the multi-camera
    # handoff (floor mapping from the pose) and the 3D slot of a pick. None = floor_points only.
    calibration: dict | None = None

    def floor_homography(self) -> np.ndarray:
        from bree.track.multicam import homography
        pts = np.asarray(self.floor_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 4 or len(pts) < 4:
            raise ValueError(f"camera {self.camera_id}: floor_points needs 4+ [x_px, y_px, x_m, y_m] rows")
        return homography(pts[:, :2], pts[:, 2:])

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
        floor_points=cam.get("floor_points"),
        calibration=cam.get("calibration"),
    )


def merge_stores(stores: list[StoreConfig]) -> StoreConfig:
    """The store as the ledger sees it when several cameras feed one ledger: every camera's zones
    (by name; the polygons stay per camera), terminals, catalog and product classes. Ledger
    settings come from the first camera's file. A zone name must mean the same kind everywhere."""
    if len(stores) == 1:
        return stores[0]
    kinds: dict[str, str] = {}
    for s in stores:
        for z in s.zones:
            if kinds.setdefault(z.name, z.kind) != z.kind:
                raise ValueError(f"zone {z.name!r} is {kinds[z.name]} in one camera and {z.kind} in another")
    first = stores[0]
    return StoreConfig(
        name=first.name, camera_id="+".join(s.camera_id for s in stores), resolution=first.resolution,
        fps=first.fps, zones=[z for s in stores for z in s.zones],
        terminals={k: v for s in stores for k, v in s.terminals.items()},
        catalog=Catalog({k: v for s in stores for k, v in s.catalog.sku_to_category.items()}),
        rules=first.rules, ledger=first.ledger,
        product_classes={k: v for s in stores for k, v in s.product_classes.items()})
