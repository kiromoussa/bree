"""The shared store floor plan: metres, regions, and one calibrated camera per channel.

JSON file (a person can read and correct it; bree.fusion.calibrate and bree.fusion.registry write the numbers):

  {"station": ..., "units": "m", "provenance": ..., "regions": {name: {"kind": ..., "poly": [[x, y], ...]}},
   "cameras": {"07": {"role": ..., "camera": <bree.calib.camera.Camera.to_dict>, "floor_px": [[u, v], ...] (polygon:
   where a person's feet can be placed), "max_range_m": ..., "heading": "-y"}}, "links": [{"a": ..., "b": ...}]}

Frame: the repo's layout frame (bree.calib.camera): plan (x_m, y_m) is the layout point (x, 0, z); y is up. A camera
carries its building-axes rotation from the Manhattan fit (bree.fusion.manhattan), its lens (f, k) and its
position; `heading` says which plan direction the picture looks along (up in the picture on the floor), because the
Manhattan fit knows the axes only up to sign.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bree.calib.camera import Camera

HEADINGS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def camera_from_axes(axes: np.ndarray, heading: str, f: float, k: float, size: tuple[int, int],
                     position_xy=(0.0, 0.0), height_m: float = 1.0) -> Camera:
    """Camera of the layout frame from a Manhattan frame (rows `axes` in the camera frame, x right, y down, z forward).
    The vertical axis (manhattan.vertical_axis) is turned to point up (negative camera y). The horizontal axis most along the optical axis is
    the heading, `heading` names its plan direction; right handedness of the layout frame (x, up, z) fixes the other."""
    from bree.fusion.manhattan import vertical_axis
    axes = np.asarray(axes, float)
    vertical = vertical_axis(axes)
    up = axes[vertical] * (1.0 if axes[vertical][1] < 0 else -1.0)
    hs = [axes[i] for i in range(3) if i != vertical]
    fwd = max(hs, key=lambda a: abs(a[2]))
    fwd = fwd * (1.0 if fwd[2] > 0 else -1.0)
    sign = {"+x": 1, "-x": -1, "+y": 1, "-y": -1}[heading]
    if heading[1] == "x":
        ex = fwd * sign
        ez = np.cross(ex, up)
    else:
        ez = fwd * sign
        ex = np.cross(up, ez)
    R = np.stack([ex, up, ez], 1)                  # columns: layout x, up, z in camera coordinates
    w, h = size
    C = np.array([position_xy[0], height_m, position_xy[1]], float)
    return Camera(f, w / 2, h / 2, R, C, (w, h), k)


def floor_xy(cam: Camera, u: float, v: float) -> tuple[float, float] | None:
    """Plan point of a pixel on the floor (y = 0), or None when the ray does not reach the floor."""
    d = cam.ray(u, v)
    if d[1] > -1e-3:
        return None
    t = -cam.C[1] / d[1]
    p = cam.C + d * t
    return float(p[0]), float(p[2])


def floor_xy_many(cam: Camera, pts) -> np.ndarray:
    return np.array([floor_xy(cam, float(u), float(v)) or (np.nan, np.nan) for u, v in pts], float)


def to_pixel(cam: Camera, x: float, y: float, height: float = 0.0) -> tuple[float, float] | None:
    px, depth = cam.project(np.array([[x, height, y]]))
    return (float(px[0, 0]), float(px[0, 1])) if depth[0] > 1e-6 else None


def footprint(cam: Camera, floor_px=None, max_range_m: float = 14.0, step_px: int = 32) -> np.ndarray:
    """Plan points (N, 2) a person standing there would be seen at: pixels of a grid over the picture (inside the
    `floor_px` polygon when given) whose ray reaches the floor within the range. Sparse; for overlap and coverage maps."""
    import cv2
    w, h = cam.resolution
    mask = None
    if floor_px is not None:
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.asarray(floor_px, np.int32)], 255)
    pts = []
    for v in range(step_px // 2, h, step_px):
        for u in range(step_px // 2, w, step_px):
            if mask is not None and not mask[v, u]:
                continue
            p = floor_xy(cam, u, v)
            if p is not None and math.hypot(p[0] - cam.C[0], p[1] - cam.C[2]) <= max_range_m:
                pts.append(p)
    return np.asarray(pts, float).reshape(-1, 2)


@dataclass
class PlanCamera:
    id: str
    camera: Camera
    role: str = ""
    heading: str = ""
    floor_px: list | None = None
    max_range_m: float = 14.0
    notes: str = ""
    place_mode: str = "feet"          # "feet": bottom of the box on the floor; "centre": box centre at hip height (looks straight down)
    clock_offset_s: float = 0.0       # add to this camera's recorder time to get store time (measured by bree.fusion.registry)


@dataclass
class Plan:
    station: str
    cameras: dict[str, PlanCamera] = field(default_factory=dict)
    regions: dict[str, dict] = field(default_factory=dict)
    links: list[dict] = field(default_factory=list)
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"station": self.station, "units": "m", "provenance": self.provenance, "regions": self.regions,
                "cameras": {c.id: {"role": c.role, "heading": c.heading, "camera": c.camera.to_dict(), "floor_px": c.floor_px,
                                   "max_range_m": c.max_range_m, "notes": c.notes, "place_mode": c.place_mode,
                                   "clock_offset_s": c.clock_offset_s} for c in self.cameras.values()},
                "links": self.links}

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        cams = {k: PlanCamera(k, Camera.from_dict(v["camera"]), v.get("role", ""), v.get("heading", ""), v.get("floor_px"),
                              v.get("max_range_m", 14.0), v.get("notes", ""), v.get("place_mode", "feet"),
                              v.get("clock_offset_s", 0.0)) for k, v in d.get("cameras", {}).items()}
        return Plan(d["station"], cams, d.get("regions", {}), d.get("links", []), d.get("provenance", ""))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @staticmethod
    def load(path) -> "Plan":
        return Plan.from_dict(json.loads(Path(path).read_text()))

    def region_at(self, x: float, y: float) -> str | None:
        """Smallest region containing the plan point (regions overlap: a cooler strip lies inside the sales floor)."""
        best, area = None, 1e18
        for name, r in self.regions.items():
            if _in_poly(r["poly"], x, y):
                a = _poly_area(r["poly"])
                if a < area:
                    best, area = name, a
        return best

    def place(self, cam: str, u: float, v: float) -> tuple[float, float] | None:
        """Plan point of a pixel of camera `cam` (a person's feet), or None when outside its valid range or floor polygon."""
        pc = self.cameras[cam]
        if pc.floor_px is not None and not _in_poly(pc.floor_px, u, v):
            return None
        p = floor_xy(pc.camera, u, v)
        if p is None or math.hypot(p[0] - pc.camera.C[0], p[1] - pc.camera.C[2]) > pc.max_range_m:
            return None
        return p

    def sees(self, cam: str, x: float, y: float) -> bool:
        """Whether the plan point projects into the camera's valid floor area (range, picture, floor polygon)."""
        pc = self.cameras[cam]
        pix = to_pixel(pc.camera, x, y)
        if pix is None:
            return False
        w, h = pc.camera.resolution
        if not (0 <= pix[0] < w and 0 <= pix[1] < h):
            return False
        if math.hypot(x - pc.camera.C[0], y - pc.camera.C[2]) > pc.max_range_m:
            return False
        return pc.floor_px is None or _in_poly(pc.floor_px, *pix)


def _in_poly(poly, x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0:
            inside = not inside
    return inside


def _poly_area(poly) -> float:
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2
