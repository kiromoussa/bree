"""Per-camera perception profile: how to run detection, pose and tracking on one real camera.

A profile is a YAML file (configs/perception/<station>_chNN.yaml, written by bree.perception.calibrate or by hand):

    camera: ch29
    role: counter_overhead         # door | counter_overhead | aisle | cooler | wide_floor | outdoor
    detector: {weights: yolo26s.pt, imgsz: 960, person_conf: 0.3}
    slices: {cols: 2, rows: 2, overlap: 0.2, full_frame: true}
    ignore: [{name: lottery_display, polygon: [[0.84, 0.1], [0.92, 0.1], [0.92, 0.25], [0.84, 0.25]]}]
    filters: {min_box_h_frac: 0.03, edge_margin_frac: 0.0, edge_conf: 0.0, adaptive_ghost_s: 0.0}
    pose: {enabled: true}
    tracker: {buffer_s: 2.0, new_track_thresh: 0.5}
    events: {min_track_age_s: 1.0, count_enter_exit: false}
    lens: {k1: 0.0}                # radial distortion (division model) for distortion-aware zones
    provenance: {...}              # where the numbers came from and with what label kind

Everything is optional; missing keys take the defaults below. `provenance` carries the label kind of any measured
number (REAL-MODEL-LABELLED, REAL-UNLABELLED, SIMULATED) so a profile never claims accuracy without labels behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from bree.perception.sliced import SliceSpec

ROLES = ("door", "counter_overhead", "aisle", "cooler", "wide_floor", "outdoor")


@dataclass
class IgnoreRegion:
    name: str
    polygon: list[list[float]]          # normalised 0..1 (x, y)
    reason: str = "static ghost"


@dataclass
class PerceptionProfile:
    camera: str = "default"
    role: str = "wide_floor"
    weights: str = "yolo26s.pt"
    pose_weights: str = "yolo26s-pose.pt"
    imgsz: int = 640
    person_conf: float = 0.3
    product_conf: float = 0.25
    dedupe_inside: float = 0.85
    slices: SliceSpec = field(default_factory=SliceSpec)
    ignore: list[IgnoreRegion] = field(default_factory=list)          # person ghosts
    fixtures: list[IgnoreRegion] = field(default_factory=list)        # static product-class objects (card terminal ...)
    min_box_h_frac: float = 0.0          # drop person boxes shorter than this fraction of the frame height
    edge_margin_frac: float = 0.0        # boxes whose centre is this close to a frame edge need edge_conf
    edge_conf: float = 0.0
    adaptive_ghost_s: float = 0.0        # >0: learn static low-confidence ghosts online after this many seconds (bree.perception.adaptive)
    pose: bool = True
    pose_crop: int = 160
    track_buffer_s: float = 2.0
    new_track_thresh: float | None = None    # ByteTrack: confidence needed to START a track
    track_high_thresh: float | None = None
    min_track_age_s: float = 0.0         # an 'enter' needs a track this old (event engine setting, see docs)
    count_enter_exit: bool = True        # advisory: False on cameras with no door in view (births are not entrances);
    #                                      read by traffic analytics, the event engine does not use it
    k1: float = 0.0                      # division-model radial distortion, see bree.perception.lens
    provenance: dict = field(default_factory=dict)

    def tracker_overrides(self) -> dict:
        o = {}
        if self.new_track_thresh is not None:
            o["new_track_thresh"] = self.new_track_thresh
        if self.track_high_thresh is not None:
            o["track_high_thresh"] = self.track_high_thresh
        return o


def _poly_contains(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd rule point-in-polygon for (N, 2) points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x0, y0, x1, y1 = *poly[i], *poly[(i + 1) % n]
        cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / ((y1 - y0) or 1e-12) + x0)
        inside ^= cross
    return inside


def fixture_mask(profile: PerceptionProfile, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """True for product boxes that are NOT a learned static fixture (centre outside every fixture region)."""
    keep = np.ones(len(boxes), bool)
    if not len(boxes) or not profile.fixtures:
        return keep
    h, w = shape[:2]
    pts = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h], 1)
    for reg in profile.fixtures:
        keep &= ~_poly_contains(np.asarray(reg.polygon, float), pts)
    return keep


def keep_mask(profile: PerceptionProfile, boxes: np.ndarray, confs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Which person boxes survive the profile's filters (ignore regions, minimum height, edge confidence)."""
    h, w = shape[:2]
    keep = np.ones(len(boxes), bool)
    if not len(boxes):
        return keep
    keep &= confs >= profile.person_conf
    if profile.min_box_h_frac:
        keep &= (boxes[:, 3] - boxes[:, 1]) >= profile.min_box_h_frac * h
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h
    if profile.edge_margin_frac and profile.edge_conf:
        m = profile.edge_margin_frac
        near = (cx < m) | (cx > 1 - m) | (cy < m) | (cy > 1 - m)
        keep &= ~(near & (confs < profile.edge_conf))
    pts = np.stack([cx, cy], 1)
    for reg in profile.ignore:
        keep &= ~_poly_contains(np.asarray(reg.polygon, float), pts)
    return keep


def _from_dict(d: dict) -> PerceptionProfile:
    det, sl, flt, pose, trk, ev = (d.get(k, {}) or {} for k in ("detector", "slices", "filters", "pose", "tracker", "events"))
    p = PerceptionProfile(
        camera=d.get("camera", "default"), role=d.get("role", "wide_floor"),
        weights=det.get("weights", "yolo26s.pt"), pose_weights=pose.get("weights", "yolo26s-pose.pt"),
        imgsz=det.get("imgsz", 640), person_conf=det.get("person_conf", 0.3), product_conf=det.get("product_conf", 0.25),
        dedupe_inside=det.get("dedupe_inside", 0.85),
        slices=SliceSpec(**{k: v for k, v in sl.items() if k in SliceSpec.__dataclass_fields__}),
        ignore=[IgnoreRegion(**r) for r in d.get("ignore", []) or []],
        fixtures=[IgnoreRegion(**r) for r in d.get("fixtures", []) or []],
        min_box_h_frac=flt.get("min_box_h_frac", 0.0), edge_margin_frac=flt.get("edge_margin_frac", 0.0),
        edge_conf=flt.get("edge_conf", 0.0), adaptive_ghost_s=flt.get("adaptive_ghost_s", 0.0), pose=pose.get("enabled", True), pose_crop=pose.get("crop", 160),
        track_buffer_s=trk.get("buffer_s", 2.0), new_track_thresh=trk.get("new_track_thresh"),
        track_high_thresh=trk.get("track_high_thresh"), min_track_age_s=ev.get("min_track_age_s", 0.0),
        count_enter_exit=ev.get("count_enter_exit", True),
        k1=(d.get("lens", {}) or {}).get("k1", 0.0), provenance=d.get("provenance", {}) or {})
    if p.role not in ROLES:
        raise ValueError(f"unknown role {p.role!r}; one of {ROLES}")
    return p


def to_dict(p: PerceptionProfile) -> dict:
    return {
        "camera": p.camera, "role": p.role,
        "detector": {"weights": p.weights, "imgsz": p.imgsz, "person_conf": p.person_conf,
                     "product_conf": p.product_conf, "dedupe_inside": p.dedupe_inside},
        "slices": asdict(p.slices),
        "ignore": [asdict(r) for r in p.ignore],
        "fixtures": [asdict(r) for r in p.fixtures],
        "filters": {"min_box_h_frac": p.min_box_h_frac, "edge_margin_frac": p.edge_margin_frac, "edge_conf": p.edge_conf,
                    "adaptive_ghost_s": p.adaptive_ghost_s},
        "pose": {"enabled": p.pose, "weights": p.pose_weights, "crop": p.pose_crop},
        "tracker": {"buffer_s": p.track_buffer_s, "new_track_thresh": p.new_track_thresh,
                    "track_high_thresh": p.track_high_thresh},
        "events": {"min_track_age_s": p.min_track_age_s, "count_enter_exit": p.count_enter_exit},
        "lens": {"k1": p.k1}, "provenance": p.provenance,
    }


def load_profile(path: str | Path) -> PerceptionProfile:
    import yaml
    return _from_dict(yaml.safe_load(Path(path).read_text()) or {})


def save_profile(p: PerceptionProfile, path: str | Path) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(to_dict(p), sort_keys=False, default_flow_style=None))
