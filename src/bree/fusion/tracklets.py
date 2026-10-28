"""Single-camera tracklets from cached person detections (bree.fusion.detect_cache).

One camera, 5 detections per second, boxes plus clothing colour. A small ByteTrack-style tracker (high confidence
boxes first, then low confidence boxes for the tracks still unmatched; new tracks only from confident boxes) whose
association cost is box overlap, centre distance and clothing colour. It is deliberately short-memory: it never
tries to bridge a long gap or another camera, that is bree.fusion.handoff's job on the finished tracklets.

A Tracklet keeps what the handoff needs: times on one clock (seconds since the epoch of the recorder), boxes, the
foot point in the picture, and a clothing-colour summary taken only from frames where the person is whole and not
at the picture border. `static` marks a track that does not move (a mannequin, a poster, a reflection) so it can be
left out of counts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

SEGMENT_FMT = "%Y%m%d_%H%M%S"


@dataclass
class TrackerConfig:
    high_conf: float = 0.45           # boxes at least this confident can start a track and are matched first
    low_conf: float = 0.25            # boxes down to this are matched to tracks that are still unmatched
    max_gap_s: float = 2.0            # a track survives this long without a box
    gate_iou: float = 0.1             # a match needs this overlap with the track's predicted box, or a close centre
    gate_centre: float = 0.6          # ... centre closer than this many box heights
    app_weight: float = 0.5           # weight of the clothing colour distance in the match cost
    min_hits: int = 3                 # a track needs this many boxes to be kept
    min_mean_conf: float = 0.4        # ... and this mean confidence
    static_move_m: float = 0.15       # a track whose feet move less than this share of a box height is static


@dataclass
class Tracklet:
    cam: str
    id: int
    t: list[float] = field(default_factory=list)          # seconds, recorder clock (epoch)
    box: list[list[float]] = field(default_factory=list)  # x0 y0 x1 y1, source pixels
    conf: list[float] = field(default_factory=list)
    app: list[np.ndarray | None] = field(default_factory=list)
    static: bool = False

    @property
    def t0(self) -> float:
        return self.t[0]

    @property
    def t1(self) -> float:
        return self.t[-1]

    def foot(self, i: int) -> tuple[float, float]:
        b = self.box[i]
        return (b[0] + b[2]) / 2, b[3]

    def at(self, t: float, tol: float = 0.35) -> int | None:
        """Index of the sample nearest to time t, or None if none within tol seconds."""
        i = int(np.searchsorted(self.t, t))
        c = [j for j in (i - 1, i) if 0 <= j < len(self.t)]
        if not c:
            return None
        j = min(c, key=lambda j: abs(self.t[j] - t))
        return j if abs(self.t[j] - t) <= tol else None


def load_segment(path: Path) -> list[tuple[float, list[dict]]]:
    """[(epoch seconds of the recorder clock, [{"bbox", "conf", "app"}])] of one cached segment."""
    t0 = datetime.strptime(path.stem, SEGMENT_FMT).timestamp()
    out = []
    with path.open() as fh:
        head = json.loads(fh.readline())
        for line in fh:
            r = json.loads(line)
            out.append((t0 + r["t"], r["persons"]))
    return out


def load_camera(cache: Path, cam: str, t_from: float | None = None, t_to: float | None = None,
                offsets: dict[str, float] | None = None) -> list[tuple[float, list[dict]]]:
    """All cached frames of a camera. `offsets` maps segment name to seconds added to the recorder clock (the DVR clock
    offset of that segment, bree.fusion.clock), so every camera ends up on the DVR's one clock; t_from/t_to are on that clock."""
    out = []
    for p in sorted((cache / f"ch{cam}").glob("*.jsonl")):
        o = (offsets or {}).get(p.stem, 0.0)
        out += [(t + o, d) for t, d in load_segment(p)]
    if t_from is not None:
        out = [x for x in out if x[0] >= t_from]
    if t_to is not None:
        out = [x for x in out if x[0] <= t_to]
    return out


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def app_distance(a, b) -> float | None:
    """Bhattacharyya-style distance (0 same .. 1 different) of two 104-number clothing colour vectors (upper then
    lower body histograms, each 52, each sums to about 1 when the body part was seen); None if either has no colour."""
    if a is None or b is None:
        return None
    d = []
    for k in (0, 52):
        x, y = a[k:k + 52], b[k:k + 52]
        sx, sy = x.sum(), y.sum()
        if sx < 0.5 or sy < 0.5:
            continue
        d.append(1.0 - float(np.sqrt(x / sx * y / sy).sum()))
    return float(np.mean(d)) if d else None


class _Live:
    def __init__(self, tid: int, t: float, box, conf: float, app):
        self.tl = Tracklet("", tid)
        self.vel = np.zeros(4)
        self.app = app
        self.miss = 0
        self.add(t, box, conf, app, first=True)

    def add(self, t, box, conf, app, first=False):
        if not first and len(self.tl.box):
            dt = max(t - self.tl.t[-1], 1e-3)
            self.vel = 0.5 * self.vel + 0.5 * (np.asarray(box) - np.asarray(self.tl.box[-1])) / dt * 0.2
        self.tl.t.append(t)
        self.tl.box.append([float(v) for v in box])
        self.tl.conf.append(float(conf))
        self.tl.app.append(app)
        if app is not None:
            self.app = app if self.app is None else 0.7 * self.app + 0.3 * app
        self.miss = 0

    def predicted(self, t: float):
        dt = (t - self.tl.t[-1]) / 0.2
        return np.asarray(self.tl.box[-1]) + self.vel * dt


def track_camera(frames: list[tuple[float, list[dict]]], cam: str, cfg: TrackerConfig | None = None) -> list[Tracklet]:
    """Tracklets of one camera from its detection frames [(t, [{"bbox", "conf", "app"}])]."""
    cfg = cfg or TrackerConfig()
    live: list[_Live] = []
    done: list[Tracklet] = []
    nid = 0
    for t, persons in frames:
        dets = [(np.asarray(p["bbox"], float), float(p["conf"]), np.asarray(p["app"], float) if p.get("app") else None)
                for p in persons if p["conf"] >= cfg.low_conf]
        high = [i for i, d in enumerate(dets) if d[1] >= cfg.high_conf]
        low = [i for i, d in enumerate(dets) if d[1] < cfg.high_conf]
        used_det: set[int] = set()
        unmatched = list(range(len(live)))
        for pool in (high, low):
            if not unmatched or not pool:
                continue
            C = np.full((len(unmatched), len(pool)), 1e6)
            for a, li in enumerate(unmatched):
                pb = live[li].predicted(t)
                hgt = max(pb[3] - pb[1], 1.0)
                for b, di in enumerate(pool):
                    if di in used_det:
                        continue
                    box = dets[di][0]
                    iou = _iou(pb, box)
                    dc = np.hypot((pb[0] + pb[2]) / 2 - (box[0] + box[2]) / 2, (pb[1] + pb[3]) / 2 - (box[1] + box[3]) / 2) / hgt
                    if iou < cfg.gate_iou and dc > cfg.gate_centre:
                        continue
                    ad = app_distance(live[li].app, dets[di][2])
                    C[a, b] = (1.0 - iou) + 0.5 * dc + cfg.app_weight * (ad if ad is not None else 0.2)
            r, c = linear_sum_assignment(C)
            matched = []
            for a, b in zip(r, c):
                if C[a, b] < 1e5:
                    li, di = unmatched[a], pool[b]
                    live[li].add(t, *dets[di][:2], dets[di][2])
                    used_det.add(di)
                    matched.append(li)
            unmatched = [x for x in unmatched if x not in matched]
        for li in unmatched:
            live[li].miss += 1
        for di in high:
            if di not in used_det:
                nid += 1
                lv = _Live(nid, t, *dets[di][:2], dets[di][2])
                live.append(lv)
        keep = []
        for lv in live:
            if lv.miss and t - lv.tl.t[-1] > cfg.max_gap_s:
                done.append(lv.tl)
            else:
                keep.append(lv)
        live = keep
    done += [lv.tl for lv in live]
    out = []
    for tl in done:
        tl.cam = cam
        if len(tl.t) >= cfg.min_hits and float(np.mean(tl.conf)) >= cfg.min_mean_conf:
            out.append(tl)
    for tl in out:
        tl.static = is_static(tl, cfg)
    return sorted(out, key=lambda x: x.t0)


def is_static(tl: Tracklet, cfg: TrackerConfig) -> bool:
    """A tracklet that lasts at least 20 s without its feet moving more than static_move_m box heights."""
    if tl.t1 - tl.t0 < 20.0:
        return False
    f = np.array([tl.foot(i) for i in range(len(tl.t))])
    h = float(np.median([b[3] - b[1] for b in tl.box]))
    return float(np.hypot(*(f.max(0) - f.min(0)))) < cfg.static_move_m * h


def summary_colour(tl: Tracklet, img_size: tuple[int, int], border: float = 0.02) -> np.ndarray | None:
    """Clothing colour summary of a tracklet: the mean of the colour vectors of the frames where the box is inside the
    picture (not within `border` of an edge) and confident. None when there is no such frame."""
    w, h = img_size
    rows = []
    for b, c, a in zip(tl.box, tl.conf, tl.app):
        if a is None or c < 0.5:
            continue
        if b[0] < w * border or b[2] > w * (1 - border) or b[3] > h * (1 - border) or b[1] < h * border:
            continue
        rows.append(a)
    return np.mean(rows, 0) if rows else None
