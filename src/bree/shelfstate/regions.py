"""Regions of one camera: which part of the picture is a shelf, a cooler door, floor, and so on.

A region is a polygon in normalised picture coordinates with a kind. The kind decides which events can come from it
(see `bree.shelfstate.model`). Cameras without a file get one region over the whole picture, kind "shelf".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("shelf", "cooler_door", "floor", "display", "ignore")


@dataclass
class Region:
    id: str
    kind: str
    polygon: list[list[float]]                 # normalised (x, y)
    params: dict = field(default_factory=dict)  # per-region overrides: settle_s, min_cells, ...

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean mask (h, w) of the polygon."""
        import cv2
        h, w = shape
        m = np.zeros((h, w), np.uint8)
        pts = np.round(np.asarray(self.polygon) * [w, h]).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)


@dataclass
class CameraRegions:
    camera_id: str
    regions: list[Region]
    role: str = "aisle"            # aisle | cooler | wide: documentation, not logic
    station: str = ""

    def label_map(self, shape: tuple[int, int]) -> tuple[np.ndarray, list[Region]]:
        """int map (h, w): index into `regions` of the region drawn last at each pixel, -1 outside every region."""
        out = np.full(shape, -1, np.int32)
        for i, r in enumerate(self.regions):
            out[r.mask(shape)] = i
        return out, self.regions


def whole_picture(camera_id: str) -> CameraRegions:
    return CameraRegions(camera_id, [Region("scene", "shelf", [[0, 0], [1, 0], [1, 1], [0, 1]])])


def load_regions(path: str | Path) -> CameraRegions:
    import yaml
    d = yaml.safe_load(Path(path).read_text())
    regs = [Region(r["id"], r.get("kind", "shelf"), r["polygon"], r.get("params", {})) for r in d["regions"]]
    for r in regs:
        if r.kind not in KINDS:
            raise ValueError(f"region {r.id}: kind {r.kind!r} not in {KINDS}")
    return CameraRegions(d["camera_id"], regs, d.get("role", "aisle"), d.get("station", ""))
