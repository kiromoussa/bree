"""Person tracker tuned for flow counting: few identities, long tracks, no births from flicker.

The pipeline's ByteTrack gave 53-62 ids for 5-8 real people on the door camera (REAL-MODEL-LABELLED, see
docs/ultra/intel.md): low-confidence boxes on staff behind the counter glass and at the frame edge flickered and each
reappearance became a new id. Flow numbers (entries, dwell, queue) need identities that survive, so this tracker:

  * associates in two passes (confident boxes first, weak boxes only onto tracks that exist), so a weak box can keep a
    track alive but never create one;
  * confirms a track only after `confirm_hits` detections with mean confidence at least `confirm_conf`;
  * does not let a track be born from a box that mostly lies inside a live track (`dup_inside`) or in an ignore region;
  * keeps a lost track for `lost_s` and lets a new track take it over when it appears where the lost one was heading
    (`reclaim`), which is the common fragmentation: a person hidden for a moment by a rack, glass or the frame edge.

Time is the video clock. Positions are the foot point (bottom centre of the box). Pure numpy/scipy: no model here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FlowConfig:
    hi_conf: float = 0.5          # a box this confident can start a track
    lo_conf: float = 0.25         # a box this confident can continue one
    confirm_hits: int = 3         # detections before a track is real
    confirm_conf: float = 0.45    # ... with at least this mean confidence
    lost_s: float = 3.0           # keep an unmatched track this long
    gate: float = 1.8             # match gate, in half box heights (min `gate_min_px`)
    gate_min_px: float = 60.0
    dup_inside: float = 0.7       # a new box this much inside a live track's box is not a new person
    reclaim_s: float = 6.0        # a new track may take over a lost track up to this long after it vanished
    reclaim_px: float = 1.5       # ... when it starts within this many box heights of the lost track's projected foot
    size_ratio: float = 1.7       # a box whose height differs from the track's by more than this factor is another person
    vel_decay: float = 0.9
    ignore: list = field(default_factory=list)   # polygons (list of [x, y]) where boxes are dropped (foot point inside)


@dataclass
class Track:
    id: int
    t0: float
    foot: np.ndarray
    box: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t_last: float = 0.0
    hits: int = 1
    conf_sum: float = 0.0
    confirmed: bool = False
    lost: bool = False
    path: list = field(default_factory=list)    # (t, fx, fy, conf, x0, y0, x1, y1)
    parent: int | None = None                   # id of the track this one took over

    @property
    def h(self) -> float:
        return max(float(self.box[3] - self.box[1]), 20.0)

    def project(self, t: float) -> np.ndarray:
        return self.foot + self.vel * min(t - self.t_last, 1.5)


def foot_of(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, b[3]], float)


def _inside(poly, x, y) -> bool:
    ins, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            ins = not ins
        j = i
    return ins


def _inter_frac(a, b) -> float:
    """Share of box a that lies inside box b."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return max(iw, 0) * max(ih, 0) / area


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class FlowTracker:
    def __init__(self, cfg: FlowConfig | None = None):
        self.cfg = cfg or FlowConfig()
        self.tracks: list[Track] = []     # live (incl. lost, not yet expired)
        self._next = 1
        self.closed: list[Track] = []     # finished, confirmed tracks

    def _cost(self, tr: Track, box, t: float) -> float:
        c = self.cfg
        d = np.linalg.norm(tr.project(t) - foot_of(box))
        scale = max(c.gate * tr.h / 2, c.gate_min_px)
        iou = _iou(tr.box, box)
        r = max(float(box[3] - box[1]), 20.0) / tr.h
        if max(r, 1 / r) > c.size_ratio:
            return 1e6
        if d > scale and iou < 0.1:
            return 1e6
        return d / scale + (1 - iou)

    def _match(self, trs, dets, t):
        if not trs or not dets:
            return [], list(range(len(trs))), list(range(len(dets)))
        m = np.array([[self._cost(tr, d["bbox"], t) for d in dets] for tr in trs])
        r, c = linear_sum_assignment(m)
        pairs = [(i, j) for i, j in zip(r, c) if m[i, j] < 1e5]
        mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
        return pairs, [i for i in range(len(trs)) if i not in mi], [j for j in range(len(dets)) if j not in mj]

    def _update(self, tr: Track, d: dict, t: float) -> None:
        f = foot_of(d["bbox"])
        dt = max(t - tr.t_last, 1e-3)
        tr.vel = 0.6 * tr.vel + 0.4 * (f - tr.foot) / dt if tr.hits > 1 else np.zeros(2)
        tr.foot, tr.box, tr.t_last = f, np.asarray(d["bbox"], float), t
        tr.hits += 1
        tr.conf_sum += d["conf"]
        tr.lost = False
        tr.path.append((t, float(f[0]), float(f[1]), d["conf"], *[float(v) for v in d["bbox"]]))
        if not tr.confirmed and tr.hits >= self.cfg.confirm_hits and tr.conf_sum / tr.hits >= self.cfg.confirm_conf:
            tr.confirmed = True

    def step(self, t: float, persons: list[dict]) -> list[Track]:
        """Feed one sampled frame. Returns the confirmed tracks seen in it (in id order)."""
        c = self.cfg
        dets = [p for p in persons if p["conf"] >= c.lo_conf
                and not any(_inside(poly, *foot_of(p["bbox"])) for poly in c.ignore)]
        hi = [i for i, p in enumerate(dets) if p["conf"] >= c.hi_conf]
        lo = [i for i, p in enumerate(dets) if p["conf"] < c.hi_conf]
        live = self.tracks
        seen: set[int] = set()
        # pass 1: confident boxes onto all tracks
        pairs, un_t, un_h = self._match(live, [dets[i] for i in hi], t)
        for ti, dj in pairs:
            self._update(live[ti], dets[hi[dj]], t)
            seen.add(live[ti].id)
        # pass 2: weak boxes onto tracks still unmatched (they can only continue, never start)
        rest = [live[i] for i in un_t]
        pairs2, _, _ = self._match(rest, [dets[i] for i in lo], t)
        for ti, dj in pairs2:
            self._update(rest[ti], dets[lo[dj]], t)
            seen.add(rest[ti].id)
        # births from confident boxes that matched nothing
        for dj in un_h:
            d = dets[hi[dj]]
            if any(_inter_frac(d["bbox"], tr.box) > c.dup_inside for tr in live if tr.id in seen or not tr.lost):
                continue
            tr = Track(self._next, t, foot_of(d["bbox"]), np.asarray(d["bbox"], float), t_last=t,
                       conf_sum=d["conf"], path=[(t, *foot_of(d["bbox"]).tolist(), d["conf"], *map(float, d["bbox"]))])
            self._next += 1
            # take over a lost track that was heading here
            cands = [o for o in live if o.lost and o.confirmed and t - o.t_last <= c.reclaim_s and o.id not in seen
                     and np.linalg.norm(o.project(t) - tr.foot) <= c.reclaim_px * o.h]
            if cands:
                o = min(cands, key=lambda o: np.linalg.norm(o.project(t) - tr.foot))
                tr.parent, tr.t0, tr.path, tr.hits, tr.conf_sum, tr.confirmed = o.id, o.t0, o.path + tr.path, o.hits + 1, o.conf_sum + d["conf"], True
                tr.id = o.id
                self._next -= 1
                live.remove(o)
            live.append(tr)
            seen.add(tr.id)
        # age
        keep = []
        for tr in live:
            if tr.id not in seen:
                tr.lost = True
                if t - tr.t_last > c.lost_s:
                    if tr.confirmed:
                        self.closed.append(tr)
                    continue
            keep.append(tr)
        self.tracks = keep
        return sorted((tr for tr in keep if tr.confirmed and tr.id in seen), key=lambda tr: tr.id)

    def finish(self) -> list[Track]:
        """End of stream: close everything. All confirmed tracks, by first appearance."""
        for tr in self.tracks:
            if tr.confirmed:
                self.closed.append(tr)
        self.tracks = []
        return sorted(self.closed, key=lambda tr: tr.t0)
