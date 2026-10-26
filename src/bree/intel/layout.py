"""Flow layout of one camera: door line, zones (browse / register / staff / door), ignore regions, clock offset and a
schematic floor plan. Coordinates in the YAML are in a reference frame (`ref_size`) and are scaled to the video size."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

KINDS = ("door", "browse", "register", "staff", "aisle")


def point_in_poly(poly, x: float, y: float) -> bool:
    ins, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            ins = not ins
        j = i
    return ins


@dataclass
class FlowZone:
    name: str
    kind: str
    polygon: np.ndarray

    def contains(self, x: float, y: float) -> bool:
        return point_in_poly(self.polygon, x, y)


@dataclass
class DoorLine:
    a: np.ndarray
    b: np.ndarray
    inside: np.ndarray
    margin: float

    def signed(self, x: float, y: float) -> float:
        """Distance in pixels from the door line along its normal, positive on the store side. Zero-ish outside the
        door's span is meaningless, so use `in_span` first."""
        d = self.b - self.a
        n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
        if np.dot(n, self.inside - self.a) < 0:
            n = -n
        return float(np.dot(np.array([x, y]) - self.a, n))

    def in_span(self, x: float, y: float, slack: float = 0.15) -> bool:
        d = self.b - self.a
        s = float(np.dot(np.array([x, y]) - self.a, d) / max(np.dot(d, d), 1e-9))
        return -slack <= s <= 1 + slack


@dataclass
class Layout:
    station: str
    camera: str
    size: tuple[int, int]
    door: DoorLine | None
    zones: list[FlowZone]
    ignore: list[np.ndarray] = field(default_factory=list)
    clock_offset_s: float = 0.0
    plan_size: tuple[float, float] = (100.0, 100.0)
    homography: np.ndarray | None = None

    def zones_at(self, x: float, y: float) -> list[FlowZone]:
        return [z for z in self.zones if z.contains(x, y)]

    def zone_by_kind(self, kind: str) -> list[FlowZone]:
        return [z for z in self.zones if z.kind == kind]

    def to_plan(self, x: float, y: float) -> tuple[float, float] | None:
        if self.homography is None:
            return None
        p = self.homography @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])


def load_layout(path: str | Path, size: tuple[int, int] | None = None) -> Layout:
    """`size` is the (width, height) of the video; defaults to the reference size."""
    d = yaml.safe_load(Path(path).read_text())
    rw, rh = d.get("ref_size", [1280, 720])
    w, h = size or (rw, rh)
    s = np.array([w / rw, h / rh], float)

    def pts(v):
        return np.asarray(v, float) * s

    dl = d.get("door_line")        # cameras that do not see a door (aisles, counter overhead) have none: no entries or exits
    door = DoorLine(pts(dl["a"]), pts(dl["b"]), pts(dl["inside"]), float(dl.get("margin_px", 12)) * float(s.mean())) if dl else None
    zones = []
    for z in d.get("zones", []):
        if z["kind"] not in KINDS:
            raise ValueError(f"zone {z['name']}: kind must be one of {KINDS}")
        zones.append(FlowZone(z["name"], z["kind"], pts(z["polygon"])))
    hom = None
    plan = d.get("plan")
    if plan:
        import cv2
        hom = cv2.getPerspectiveTransform(pts(plan["image_pts"]).astype(np.float32), np.asarray(plan["plan_pts"], np.float32))
    return Layout(d.get("station", ""), d.get("camera", ""), (w, h), door, zones, [pts(p) for p in d.get("ignore", [])],
                  float(d.get("clock", {}).get("file_name_offset_s", 0)), tuple(plan["size"]) if plan else (100.0, 100.0), hom)
