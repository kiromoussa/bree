"""One identity per person across all cameras of the store, from finished tracklets.

Input: per-camera tracklets (bree.fusion.tracklets), the calibrated plan (bree.fusion.plan: each camera's floor
points in store metres, its clock offset) and nothing else. No face is used. Output: visits (one per person), each a
list of tracklets from any camera with a fused floor trajectory.

Two tracklets of DIFFERENT people are never the same visit if they are seen together by one camera. Two tracklets of the
same person are linked by evidence from three places:

  seen together   the two cameras overlap and both place the person at the same moment: the two floor points must agree
                  (median distance over the shared samples). This is the strongest evidence and needs no appearance.
  walked there    one tracklet ends, the other starts later, possibly in another camera whose field does not overlap:
                  the floor distance between where the first was lost and where the second appears must be walkable in
                  the gap (walking speed, slack for the calibration), and the gap itself is penalised mildly.
  looks the same  body-colour distance (upper and lower body histograms, never the head). Likelihood ratios are
                  learnt WITHOUT labels from the data itself: positives are tightly co-located simultaneous tracklets of
                  two cameras (same person for sure), negatives are tracklets that overlap in time in ONE camera (two
                  people for sure) (AppModel).

Clustering is greedy agglomerative on the summed log likelihood ratio, never merging two clusters that hold
two tracklets of one camera at the same time, never merging clusters that disagree on the floor when seen together.
`HandoffConfig` documents every threshold; the values were set by reading the first tracklets and checked on a
labelled split (bree.fusion.eval), never fitted on the held-out part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bree.fusion.plan import Plan
from bree.fusion.registry import foot_point
from bree.fusion.tracklets import Tracklet, app_distance, summary_colour


@dataclass
class HandoffConfig:
    walk_mps: float = 1.8              # how fast a person can cover the floor between two sightings
    slack_m: float = 1.0               # calibration and placement error allowed on every distance
    sigma_m: float = 0.8               # scale of a distance disagreement in the log likelihood ratio
    same_place_m: float = 1.2          # seen together: median disagreement at most this to be the same person
    never_same_m: float = 3.0          # seen together, further apart than this at every shared moment: different people
    min_overlap_samples: int = 4       # shared samples needed for the "seen together" evidence
    max_gap_s: float = 90.0            # no link between tracklets further apart in time than this
    gap_cost_per_s: float = 0.04       # log odds lost per second of gap
    same_cam_gap_s: float = 4.0        # a same-camera link needs a gap at most this (the tracker's own memory ends at 2 s)
    link_threshold: float = 1.0        # clusters are merged only above this summed log likelihood ratio
    use_appearance: bool = True
    app_weight: float = 1.0
    use_walk: bool = True
    use_together: bool = True
    seen_together_bonus: float = 3.0   # log odds of a tight, long enough co-sighting
    min_len_s: float = 1.5             # shorter tracklets are not clustered (they stay singleton visits)


@dataclass
class Node:
    tl: Tracklet
    cam: str
    t: np.ndarray                      # store-clock seconds of every sample
    xy: np.ndarray                     # (N, 2) plan metres, NaN where the sample could not be placed
    colour: np.ndarray | None
    first: tuple[float, float, float] | None = None   # (t, x, y) of the first placed sample
    last: tuple[float, float, float] | None = None
    placed: int = 0

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    @property
    def key(self) -> str:
        return f"{self.cam}:{self.tl.id}"


def prepare(plan: Plan, tracklets: dict[str, list[Tracklet]], size=(2560, 1440), min_conf: float = 0.45) -> list[Node]:
    """Nodes: tracklet times moved to the store clock, boxes placed on the floor with each camera's calibration."""
    nodes = []
    for cam, tls in tracklets.items():
        pc = plan.cameras[cam]
        for tl in tls:
            if tl.static:
                continue
            t = np.asarray(tl.t) + pc.clock_offset_s
            xy = np.full((len(t), 2), np.nan)
            for i, b in enumerate(tl.box):
                if tl.conf[i] < min_conf:
                    continue
                p = foot_point(pc.camera, b, pc.place_mode)
                if p is not None and math.hypot(p[0] - pc.camera.C[0], p[1] - pc.camera.C[2]) <= pc.max_range_m:
                    xy[i] = p
            ok = ~np.isnan(xy[:, 0])
            n = Node(tl, cam, t, xy, summary_colour(tl, pc.camera.resolution), placed=int(ok.sum()))
            if ok.any():
                i0, i1 = int(np.argmax(ok)), int(len(ok) - 1 - np.argmax(ok[::-1]))
                n.first = (float(t[i0]), *map(float, xy[i0]))
                n.last = (float(t[i1]), *map(float, xy[i1]))
            nodes.append(n)
    return sorted(nodes, key=lambda n: n.t0)


def common_samples(a: Node, b: Node, tol: float = 0.21) -> tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of samples of a and b within `tol` seconds of each other where both are placed."""
    lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
    if hi < lo:
        return np.array([], int), np.array([], int)
    ia = np.flatnonzero((a.t >= lo - tol) & (a.t <= hi + tol) & ~np.isnan(a.xy[:, 0]))
    if not len(ia):
        return ia, ia
    j = np.searchsorted(b.t, a.t[ia])
    j = np.clip(j, 1, len(b.t) - 1)
    left, right = j - 1, j
    jb = np.where(np.abs(b.t[left] - a.t[ia]) <= np.abs(b.t[right] - a.t[ia]), left, right)
    ok = (np.abs(b.t[jb] - a.t[ia]) <= tol) & ~np.isnan(b.xy[jb, 0])
    return ia[ok], jb[ok]


def overlap_s(a: Node, b: Node) -> float:
    return max(0.0, min(a.t1, b.t1) - max(a.t0, b.t0))


class AppModel:
    """Log likelihood ratio of a body-colour distance, learnt from the data without labels (see module text)."""

    def __init__(self, pos: list[float], neg: list[float], bins: int = 12):
        self.edges = np.linspace(0.0, 0.8, bins + 1)
        h_pos = np.histogram(np.clip(pos, 0, 0.8 - 1e-9), self.edges)[0] + 1.0
        h_neg = np.histogram(np.clip(neg, 0, 0.8 - 1e-9), self.edges)[0] + 1.0
        self.llr = np.log((h_pos / h_pos.sum()) / (h_neg / h_neg.sum()))
        k = np.array([0.25, 0.5, 0.25])
        self.llr = np.convolve(np.pad(self.llr, 1, mode="edge"), k, mode="valid")
        self.n_pos, self.n_neg = len(pos), len(neg)

    def __call__(self, d: float | None) -> float:
        if d is None:
            return 0.0
        i = int(np.clip(np.searchsorted(self.edges, d, side="right") - 1, 0, len(self.llr) - 1))
        return float(self.llr[i])


def fit_app_model(nodes: list[Node], cfg: HandoffConfig) -> AppModel | None:
    """Positives: simultaneous tracklets of two cameras that agree within 0.5 m on at least 8 shared samples.
    Negatives: simultaneous tracklets of one camera (two boxes at once are two people)."""
    pos, neg = [], []
    for i, a in enumerate(nodes):
        if a.colour is None:
            continue
        for b in nodes[i + 1:]:
            if b.t0 > a.t1:
                break
            if b.colour is None or overlap_s(a, b) < 2.0:
                continue
            d = app_distance(a.colour, b.colour)
            if d is None:
                continue
            if a.cam == b.cam:
                neg.append(d)
            else:
                ia, ib = common_samples(a, b)
                if len(ia) >= 8 and np.median(np.linalg.norm(a.xy[ia] - b.xy[ib], axis=1)) < 0.5:
                    pos.append(d)
    if len(pos) < 5 or len(neg) < 5:
        return None
    return AppModel(pos, neg)


def pair_evidence(a: Node, b: Node, cfg: HandoffConfig, app: AppModel | None = None) -> tuple[float, str] | None:
    """(log likelihood ratio that a and b are one person, kind of evidence) or None when they cannot be."""
    if a.t0 > b.t0:
        a, b = b, a
    ov = overlap_s(a, b)
    score, kind = 0.0, ""
    if ov > 0.5:                                    # simultaneous
        if a.cam == b.cam:
            return None
        ia, ib = common_samples(a, b)
        if len(ia) < cfg.min_overlap_samples:
            return None                             # no joint floor evidence: cannot say (neither link nor veto)
        d = np.linalg.norm(a.xy[ia] - b.xy[ib], axis=1)
        dm = float(np.median(d))
        if not cfg.use_together:
            return None
        if dm > cfg.never_same_m and float(np.percentile(d, 20)) > cfg.never_same_m:
            return None
        score = cfg.seen_together_bonus - (dm / cfg.sigma_m) ** 2 / 2 * (1.0 if dm > cfg.same_place_m else 0.3)
        score += 0.15 * min(len(ia), 20)
        kind = "together"
    else:                                           # b starts after a ended
        g = b.t0 - a.t1
        if g > cfg.max_gap_s or g < -0.5:
            return None
        if a.cam == b.cam and g > cfg.same_cam_gap_s:
            return None
        if cfg.use_walk and a.last is not None and b.first is not None:
            d = math.hypot(a.last[1] - b.first[1], a.last[2] - b.first[2])
            reach = cfg.walk_mps * max(b.first[0] - a.last[0], 0.0) + cfg.slack_m
            over = max(0.0, d - reach)
            score += 1.0 - (over / cfg.sigma_m) ** 2 / 2 - 0.4 * min(d / max(reach, 0.1), 3.0)
            if over > 4 * cfg.sigma_m:
                return None
        score -= cfg.gap_cost_per_s * max(g, 0.0)
        kind = "walk"
    if cfg.use_appearance and app is not None:
        score += cfg.app_weight * app(app_distance(a.colour, b.colour))
    return score, kind


@dataclass
class Visit:
    id: int
    nodes: list[Node] = field(default_factory=list)

    @property
    def t0(self) -> float:
        return min(n.t0 for n in self.nodes)

    @property
    def t1(self) -> float:
        return max(n.t1 for n in self.nodes)

    @property
    def cams(self) -> list[str]:
        return sorted({n.cam for n in self.nodes})


def cluster(nodes: list[Node], cfg: HandoffConfig | None = None, app: AppModel | None = None) -> list[Visit]:
    """Greedy agglomerative clustering of tracklets into visits (see module text)."""
    cfg = cfg or HandoffConfig()
    use = [n for n in nodes if n.t1 - n.t0 >= cfg.min_len_s]
    skip = [n for n in nodes if n.t1 - n.t0 < cfg.min_len_s]
    edges = []
    for i, a in enumerate(use):
        for j in range(i + 1, len(use)):
            b = use[j]
            if b.t0 - a.t1 > cfg.max_gap_s:
                # tracklets are sorted by start; later ones may still overlap a long a: only break when b starts after all of a
                if b.t0 > a.t1 + cfg.max_gap_s:
                    break
                continue
            ev = pair_evidence(a, b, cfg, app)
            if ev is not None:
                edges.append((ev[0], i, j, ev[1]))
    edges.sort(key=lambda e: -e[0])
    parent = list(range(len(use)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    members: dict[int, list[int]] = {i: [i] for i in range(len(use))}
    vetoed: set[tuple[int, int]] = set()
    for s, i, j, kind in edges:
        if s < cfg.link_threshold:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if _conflict(use, members[ri], members[rj], cfg):
            continue
        parent[rj] = ri
        members[ri] += members.pop(rj)
    visits = [Visit(0, [use[k] for k in sorted(m, key=lambda k: use[k].t0)]) for m in members.values()]
    visits += [Visit(0, [n]) for n in skip]
    visits.sort(key=lambda v: v.t0)
    for k, v in enumerate(visits, 1):
        v.id = k
    return visits


def _conflict(use: list[Node], A: list[int], B: list[int], cfg: HandoffConfig) -> bool:
    """Two clusters cannot merge when they hold simultaneous tracklets of one camera, or tracklets seen together by two
    cameras that disagree on the floor."""
    for i in A:
        a = use[i]
        for j in B:
            b = use[j]
            if overlap_s(a, b) <= 0.5:
                continue
            if a.cam == b.cam:
                return True
            ia, ib = common_samples(a, b)
            if len(ia) >= cfg.min_overlap_samples:
                d = np.linalg.norm(a.xy[ia] - b.xy[ib], axis=1)
                if float(np.median(d)) > cfg.never_same_m:
                    return True
    return False


def fuse_track(v: Visit, hz: float = 5.0) -> np.ndarray:
    """Store-wide floor trajectory of a visit, rows [t, x, y, n_cameras]: on a regular time grid the median of the
    placed samples of all its tracklets at that moment (overlapping cameras vote, the median ignores one bad placement)."""
    ts = np.arange(v.t0, v.t1, 1.0 / hz)
    out = []
    for t in ts:
        pts = []
        for n in v.nodes:
            if n.t0 - 0.3 <= t <= n.t1 + 0.3:
                i = int(np.argmin(np.abs(n.t - t)))
                if abs(n.t[i] - t) <= 0.3 and not np.isnan(n.xy[i, 0]):
                    pts.append(n.xy[i])
        if pts:
            P = np.asarray(pts)
            out.append([t, *np.median(P, 0), len(pts)])
    return np.asarray(out).reshape(-1, 4)
