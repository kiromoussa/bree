"""Store-wide people tracking on the floor plan, from the cameras that see whole people (overhead, entrance).

Each camera's person detections are put on the floor with its calibration, all cameras' points of one moment are
merged, and ONE tracker follows people in store metres. A person keeps one identity while any camera sees them.
There are no per-camera track ids to merge: that step is what broke identity before (see REPORT.md, association).

The store is a closed world:
  births     at the door (or in the first second, for whoever is already inside);
  hand-back  a track that starts anywhere else is someone who was lost; it takes that identity back when the walk
             between the two places is possible, and is marked uncertain when two lost people both fit;
  second box a new track on top of a person who is being tracked is a second detection of them, not a person;
  inside     a track that starts inside, stays, and fits nobody lost is an entry nobody saw: a new person, marked;
  deaths     lost at the door for `exit_after_s`, or `timeout_s` unseen.
Positions and time first. Clothing colour (a detection's optional "app": hue and saturation histograms of the upper
and lower body from bree.track.people, never the head) is used at the two moments position cannot decide: who
reappeared after being lost, and who is who after two people passed within `app_near_m` of each other (their
identities are swapped back when the colours say so, and the doubt mark is dropped when the colours tell them apart).
The histograms live in memory for the visit only; detections without "app" give the position-only tracker.

A person is placed by the hips (pose keypoints, 0.93 m above the floor), else the shoulders, else the box (feet, or
the head at an assumed height). Measured on DEV clip 7001 (SIMULATED): hips 0.09 to 0.17 m median error per camera,
box 0.14 to 0.63 m.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from bree.calib.camera import Camera

L_SHO, R_SHO, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


@dataclass
class FloorConfig:
    hip_m: float = 0.93               # height of the hip keypoints above the floor
    shoulder_m: float = 1.42
    kpt_conf: float = 0.5
    person_height_m: tuple[float, float] = (1.35, 2.05)   # a box whose height on the floor is outside this is cut off
    assumed_height_m: float = 1.7     # for a cut box: the head is this high
    max_range_m: float = 13.0         # a detection further than this from its camera is not placed
    sigma_kpt_m: float = 0.12         # floor error of a keypoint placement, across the line of sight
    sigma_h_m: tuple[float, float] = (0.08, 0.12)         # how wrong the assumed height can be: hips, shoulders
    sigma_across_m: tuple[float, float] = (0.15, 0.02)    # box placement: metres, plus metres per metre of range
    sigma_along: tuple[float, float] = (0.08, 0.3)        # box placement, share of the range: feet seen, head only
    gate: float = 9.2                 # squared Mahalanobis distance (2 degrees of freedom, 99 percent)
    gate_m: float = 1.2               # and never further than this
    box_weight: float = 4.0           # cost of a box that does not overlap the track's last box in that camera
    accel_mps: float = 4.0            # how hard a person can speed up or turn (process noise of the filter)
    confirm_hits: int = 4             # frames with a detection before a new track counts
    duplicate_m: float = 0.4          # a new track this close to a tracked person is a second box of them
    duplicate_far_m: float = 0.8      # ... or this close when no single camera sees both
    second_track_m: float = 1.0       # a track that starts inside this close to a tracked person, with nobody lost, is a second track of them
    door_m: float = 2.0               # births and exits happen this close to the door
    doorway_m: float = 0.3            # lost this close to the door line (or past it): left, at once
    walk_mps: float = 2.2             # a lost person can have walked at most this fast
    drift_mps: float = 0.7            # how far a lost person usually is from where they were lost, per second
    rival_ratio: float = 0.3          # a second lost person this likely (relative to the best) makes the call uncertain
    coast_s: float = 1.0              # keep predicting an unseen track this long
    lost_after_s: float = 0.3
    exit_after_s: float = 1.5         # unseen this long near the door: gone
    inside_birth_s: float = 1.5       # a track inside that fits nobody lost becomes a new person after this long
    inside_birth_share: float = 0.6   # ... if it was detected in at least this share of the frames since it started
    timeout_s: float = 3600.0         # unseen this long anywhere: dropped
    start_s: float = 1.0              # whoever is seen in the first second was already inside
    encounter_m: float = 0.25         # mark two identities uncertain when they come this close (0: off)
    app_near_m: float = 0.8           # two people this close: their boxes overlap, clothing colour is not read
    app_frames: int = 5               # clear looks at each of the two after they part, before deciding who is who
    app_margin: float = 0.08          # straight and crossed colour match differ by less than this: cannot tell
    app_scale: float = 0.15           # hand-back: a lost person is e times less likely per this much colour distance
    # hand-back: a track that starts further than other_m from where a lost person was last seen, in clothes more
    # different than other_app, is not that person. Off (inf). At 2.0 m it refuses 12 of 25 wrong hand-backs and 0 of 45
    # right ones on the TRAIN-seed clips (5 of 6 and 0 of 26 on DEV), but with identity right more often the ledger's
    # doubt discounts drop one more theft and review one more honest shopper there (round 4, REPORT.md).
    other_m: float = float("inf")
    other_app: float = 0.25
    app_wait_s: float = 20.0          # no clear look at both for this long after they met: left undecided
    one_side: bool = False            # ... unless one of them was seen clear: then that one is compared with both colours, and a hand-back keeps the open meetings. Off: round 3 on DEV2, REPORT.md
    staff_share: float = 0.7          # an identity that spends this share of its time behind the counter is staff


def _on_plane(cam: Camera, u: float, v: float, height: float) -> np.ndarray | None:
    d = cam.ray(u, v)
    if d[1] > -1e-3:
        return None
    return cam.C + d * (height - cam.C[1]) / d[1]


def box_point(cam: Camera, box, cfg: FloorConfig) -> tuple[np.ndarray, float, bool] | None:
    """Floor (x, z) of a person box, how far it is from the camera, and whether the feet were used. Feet if the box
    shows the whole person, else the head at an assumed height (a shelf hides the legs, or the frame cuts them)."""
    x0, y0, x1, y1 = box[:4]
    u, (w, h) = (x0 + x1) / 2, cam.resolution
    feet = _on_plane(cam, u, y1, 0.0) if y1 < h - 3 else None
    if feet is not None:      # how tall is the box if the feet are really there?
        d = cam.ray(u, y0)
        flat = np.hypot(*(feet - cam.C)[[0, 2]])
        top = cam.C[1] + d[1] * flat / max(np.hypot(d[0], d[2]), 1e-6)
        if cfg.person_height_m[0] <= top <= cfg.person_height_m[1]:
            return feet[[0, 2]], float(flat), True
    head = _on_plane(cam, u, y0, cfg.assumed_height_m) if y0 > 3 else None
    if head is None:
        return None
    return head[[0, 2]], float(np.hypot(*(head - cam.C)[[0, 2]])), False


def floor_point(cam: Camera, det, cfg: FloorConfig) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Floor (x, z), its 2x2 covariance, and whether it is good enough to start a track (placed by keypoints, or a
    plain box when the detector gives no keypoints at all) for one detection: {"bbox", "kpts"?} or a plain box."""
    box = det["bbox"] if isinstance(det, dict) else det
    k = det.get("kpts") if isinstance(det, dict) else None
    c_xz = cam.C[[0, 2]]
    if k is not None:
        k = np.asarray(k, float)
        for (a, b), height, sh in (((L_HIP, R_HIP), cfg.hip_m, cfg.sigma_h_m[0]), ((L_SHO, R_SHO), cfg.shoulder_m, cfg.sigma_h_m[1])):
            if min(k[a, 2], k[b, 2]) < cfg.kpt_conf:
                continue
            p = _on_plane(cam, *(k[[a, b], :2].mean(0)), height)
            if p is None:
                continue
            xz = p[[0, 2]]
            rng = float(np.hypot(*(xz - c_xz)))
            if rng > cfg.max_range_m:
                return None
            along = (xz - c_xz) / max(rng, 1e-6)
            s_along = sh * rng / max(cam.C[1] - height, 0.3)      # a wrong height moves the point along the line of sight
            return xz, cfg.sigma_kpt_m ** 2 * np.eye(2) + s_along ** 2 * np.outer(along, along), True
    got = box_point(cam, box, cfg)
    if got is None or got[1] > cfg.max_range_m:
        return None
    xz, rng, feet = got
    along = (xz - c_xz) / max(rng, 1e-6)
    across = np.array([-along[1], along[0]])
    sa = cfg.sigma_along[0 if feet else 1] * rng + 0.1
    sc = cfg.sigma_across_m[0] + cfg.sigma_across_m[1] * rng
    return xz, sa ** 2 * np.outer(along, along) + sc ** 2 * np.outer(across, across), k is None


def _footprint(f: dict) -> tuple[float, float, float, float]:
    """(x0, x1, z0, z1) of a fixture on the floor (rotationY of a quarter turn swaps its sides)."""
    (x, _, z), (sx, _, sz) = f["position"], f["size"]
    if abs(np.sin(f.get("rotationY", 0.0))) > 0.7:
        sx, sz = sz, sx
    return x - sx / 2, x + sx / 2, z - sz / 2, z + sz / 2


def counter_zones(layout: dict | None, customer_m: float = 1.2) -> tuple[tuple | None, tuple | None]:
    """((x0, x1, z0, z1) behind the counter, the same for the strip in front of it where customers stand to pay).
    Behind is the side of the backbar (else of the nearer wall). A real store marks both once at install; here they
    are read off the layout."""
    c = next((f for f in (layout or {}).get("fixtures", []) if f["type"] == "counter"), None)
    if c is None:
        return None, None
    x0, x1, z0, z1 = _footprint(c)
    W, D = layout["store"]["width"] / 2, layout["store"]["depth"] / 2
    bb = next((f for f in layout["fixtures"] if f["type"] == "backbar"), None)
    if x1 - x0 <= z1 - z0:      # counter runs along z: staff on the +x or -x side
        plus = bb["position"][0] > (x0 + x1) / 2 if bb else W - x1 < x0 + W
        far = (_footprint(bb)[0] if plus else _footprint(bb)[1]) if bb else (W if plus else -W)
        return ((x1, far, z0, z1), (x0 - customer_m, x0, z0, z1)) if plus else ((far, x0, z0, z1), (x1, x1 + customer_m, z0, z1))
    plus = bb["position"][2] > (z0 + z1) / 2 if bb else D - z1 < z0 + D
    far = (_footprint(bb)[2] if plus else _footprint(bb)[3]) if bb else (D if plus else -D)
    return ((x0, x1, z1, far), (x0, x1, z0 - customer_m, z0)) if plus else ((x0, x1, far, z0), (x0, x1, z1, z1 + customer_m))


def in_zone(zone, x: float, z: float, pad: float = 0.0) -> bool:
    return zone is not None and zone[0] - pad <= x <= zone[1] + pad and zone[2] - pad <= z <= zone[3] + pad


def _bbox(det):
    return det["bbox"] if isinstance(det, dict) else det[:4]


def _box_break(k, cam: str, det, t: float) -> float:
    """0 when the detection overlaps the track's last box in this camera fully, 1 when not at all (or no recent box: 0.5)."""
    last = k.boxes.get(cam)
    if last is None or t - last[0] > 0.5:
        return 0.5
    a, b = last[1], _bbox(det)
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 1.0
    return 1.0 - w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h)


@dataclass(eq=False)        # a track is itself, not its numbers: `tracks.remove(q)` compared numpy fields with == and crashed (stressed dev2 11004, 11007)
class Track:
    id: int
    x: np.ndarray                      # x, z, vx, vz on the floor (constant velocity Kalman filter)
    t_first: float
    t_seen: float
    hits: int = 1
    frames: int = 1                    # frames since it started (hits / frames: how steadily it is detected)
    P: np.ndarray = field(default_factory=lambda: np.eye(4))      # covariance of x
    state: str = "new"                 # new, live, lost, exited, timed_out
    path: list = field(default_factory=list)      # (t, x, z) while seen
    kpts: list = field(default_factory=list)      # (t, camera, 17x3 keypoints) while seen
    uncertain: str | None = None
    t_uncertain: float | None = None   # from when: events before the mix-up are not in doubt
    t_pred: float = 0.0
    last: np.ndarray | None = None     # pos when last seen
    v_last: np.ndarray = field(default_factory=lambda: np.zeros(2))   # velocity when last seen
    start: np.ndarray | None = None    # pos when first seen (as a new track)
    staff: bool = False
    born: str = ""                     # door, start, inside
    t_exit: float | None = None
    boxes: dict = field(default_factory=dict)   # camera -> (t, last box there)
    cams: set = field(default_factory=set)   # cameras that saw it: this frame (a person), or ever (a new track)
    app: np.ndarray | None = None      # clothing colour: sum of the histograms of clear looks, 2 x 52
    app_n: int = 0

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]

    def at(self, t: float) -> tuple[np.ndarray, float] | None:
        """Floor (x, z) at time t and how long before or after t the person was actually seen (0 when seen then).
        None outside the identity's life."""
        if not self.path or t < self.path[0][0] - 0.5 or t > self.path[-1][0] + 0.5:
            return None
        i = bisect.bisect_left(self.path, (t,))
        if i == 0:
            a = b = self.path[0]
        elif i == len(self.path):
            a = b = self.path[-1]
        else:
            a, b = self.path[i - 1], self.path[i]
        w = 0.0 if b[0] == a[0] else min(max((t - a[0]) / (b[0] - a[0]), 0.0), 1.0)
        return np.array([a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])]), float(min(abs(t - a[0]), abs(t - b[0])))


def _app(det) -> np.ndarray | None:
    a = det.get("app") if isinstance(det, dict) else None
    return None if a is None else np.asarray(a, float).reshape(2, -1)


def app_dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Clothing colour distance of two histogram sums, 0 (same) to 1: one minus the Bhattacharyya coefficient, mean of
    the body parts both have."""
    if a is None or b is None:
        return None
    d = [1.0 - float(np.sqrt(x * y).sum() / np.sqrt(x.sum() * y.sum())) for x, y in zip(a, b) if x.sum() > 0 and y.sum() > 0]
    return float(np.mean(d)) if d else None


def _swap_identity(a: Track, b: Track, t0: float) -> None:
    """The two tracks have followed each other's person since t0: each takes the other's identity and its past."""
    ia, ib = bisect.bisect_left(a.path, (t0,)), bisect.bisect_left(b.path, (t0,))
    a.path, b.path = b.path[:ib] + a.path[ia:], a.path[:ia] + b.path[ib:]
    ka, kb = sum(k[0] < t0 for k in a.kpts), sum(k[0] < t0 for k in b.kpts)
    a.kpts, b.kpts = b.kpts[:kb] + a.kpts[ka:], a.kpts[:ka] + b.kpts[kb:]
    for f in ("id", "born", "staff", "uncertain", "t_uncertain", "t_first", "app", "app_n"):
        va, vb = getattr(a, f), getattr(b, f)
        setattr(a, f, vb), setattr(b, f, va)


class FloorTracker:
    def __init__(self, cams: dict[str, Camera], door_xz=None, cfg: FloorConfig | None = None, layout: dict | None = None):
        if door_xz is None:
            door_xz = np.asarray(next(f for f in layout["fixtures"] if f["type"] == "door")["position"], float)[[0, 2]]
        self.cams, self.door, self.cfg = cams, np.asarray(door_xz, float), cfg or FloorConfig()
        self.staff_zone, self.register_zone = counter_zones(layout)
        self.counts: dict[str, int] = {}
        self.tracks: list[Track] = []
        self.n = 0
        self.t0: float | None = None
        self.log: list[str] = []
        self.meetings: list[dict] = []      # two people who came close and have not been looked at apart yet
        self.swaps: list[tuple[float, float, int, int]] = []     # (from, to, id, id): ids given out in that span were crossed
        self.settled: list[dict] = []       # every meeting that ended: when, who, the two colour matches, what was decided

    def _count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1

    def update(self, t: float, boxes: dict[str, list]) -> dict[str, list[int | None]]:
        """All people cameras' detections of one moment ({"bbox", "kpts"?} or plain boxes) -> per camera, the
        store-wide id of each detection (None: not a person yet, or not placed)."""
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        out: dict[str, list[int | None]] = {cam: [None] * len(bs) for cam, bs in boxes.items()}
        for k in self.tracks:         # predict
            if k.state in ("new", "live", "lost") and t > k.t_pred:
                dt = t - k.t_pred
                k.t_pred = t
                k.frames += 1
                if t - k.t_seen > c.coast_s:
                    k.x[2:] = 0.0                 # stopped predicting: they are somewhere near where they were lost
                    continue
                F = np.eye(4)
                F[0, 2] = F[1, 3] = dt
                q = c.accel_mps ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
                Q = np.zeros((4, 4))
                Q[np.ix_([0, 2], [0, 2])] = Q[np.ix_([1, 3], [1, 3])] = q
                k.x, k.P = F @ k.x, F @ k.P @ F.T + Q
        hit: dict[int, list] = {}
        looks: dict[int, np.ndarray] = {}      # clothing colour seen this frame, per known person
        for cam, dets in boxes.items():      # each camera gives a track one detection at most
            pts = [(i, *p) for i, b in enumerate(dets) if (p := floor_point(self.cams[cam], b, c)) is not None]
            self.counts["not_placed"] = self.counts.get("not_placed", 0) + len(dets) - len(pts)
            if not pts:
                continue
            active = [k for k in self.tracks if k.state in ("new", "live") or (k.state == "lost" and t - k.t_seen <= c.coast_s)]
            cost = np.full((len(active), len(pts)), 1e6)
            for a, k in enumerate(active):
                for j, (_, xz, R, _s) in enumerate(pts):
                    d = xz - k.pos
                    if d @ d <= c.gate_m ** 2 and (m := d @ np.linalg.solve(k.P[:2, :2] + R, d)) <= c.gate:
                        # known people first; and the box should continue this track's last box in this camera, which
                        # keeps two people apart when they stand close and one of them is not detected for a moment
                        cost[a, j] = m + (0.0 if k.state != "new" else 0.5) + c.box_weight * _box_break(k, cam, dets[pts[j][0]], t)
            taken = set()
            if active:
                for a, j in zip(*linear_sum_assignment(cost)):
                    if cost[a, j] >= 1e6:
                        continue
                    k, (i, xz, R, _s) = active[a], pts[j]
                    taken.add(j)
                    K = k.P[:, :2] @ np.linalg.inv(k.P[:2, :2] + R)
                    k.x, k.P = k.x + K @ (xz - k.pos), k.P - K @ k.P[:2, :]
                    hit.setdefault(id(k), []).append((cam, i, dets[i]))
                    k.boxes[cam] = (t, _bbox(dets[i]))
            for j, (i, xz, R, strong) in enumerate(pts):
                if j not in taken and strong:      # a box placed without keypoints can keep a track going, not start one
                    P0 = np.eye(4) * c.walk_mps ** 2 / 2
                    P0[:2, :2] = R
                    k = Track(-1, np.array([xz[0], xz[1], 0.0, 0.0]), t, t, P=P0, t_pred=t, last=xz.copy(), start=xz.copy())
                    self.tracks.append(k)
                    hit.setdefault(id(k), []).append((cam, i, None))       # counted as seen below, hits stays 1
        for k in self.tracks:
            got = hit.get(id(k))
            if got is not None:
                if k.t_seen < t:
                    k.hits += 1
                sp = float(np.linalg.norm(k.vel))
                if sp > 1.5 * c.walk_mps:
                    k.x[2:] *= 1.5 * c.walk_mps / sp
                k.last, k.v_last, k.t_seen = k.pos.copy(), k.vel.copy(), t
                k.cams = {cam for cam, _, _ in got} if k.state != "new" else k.cams | {cam for cam, _, _ in got}
                if k.state == "lost":
                    k.state = "live"
                seen = [h for _, _, det in got if (h := _app(det)) is not None]
                if seen and k.state == "new":         # a new track's colour, to compare with whoever is lost
                    k.app, k.app_n = (0 if k.app is None else k.app) + np.sum(seen, axis=0), k.app_n + 1
                elif seen:
                    looks[id(k)] = np.sum(seen, axis=0)
                if k.state == "new" and k.hits >= c.confirm_hits:
                    self._confirm(k, t)
                if k.state == "live":
                    k.path.append((t, float(k.pos[0]), float(k.pos[1])))
                    for cam, i, det in got:
                        if det is None:
                            continue
                        out[cam][i] = k.id
                        if isinstance(det, dict) and det.get("kpts") is not None:
                            k.kpts.append((t, cam, np.asarray(det["kpts"], float)))
            elif k.state == "new" and t - k.t_seen > 0.5:
                k.state = "gone"
            elif k.state == "live" and t - k.t_seen > c.lost_after_s:
                k.state = "lost"
            if k.state == "lost":
                if self._through_door(k.last) or (t - k.t_seen > c.exit_after_s and np.linalg.norm(k.last - self.door) <= c.door_m):
                    k.state, k.t_exit = "exited", k.t_seen
                    self.log.append(f"{t:7.1f}s id {k.id} left (last seen {k.t_seen:.1f}s near the door)")
                elif t - k.t_seen > c.timeout_s:
                    k.state = "timed_out"
                    self._count("timeouts")
                    self.log.append(f"{t:7.1f}s id {k.id} not seen for {c.timeout_s:.0f}s without an exit: dropped")
        self.tracks = [k for k in self.tracks if k.state != "gone"]
        self._colours(t, looks)
        if c.encounter_m > 0:         # two people this close cannot be told apart by position: they may have been swapped
            live = [k for k in self.tracks if k.state == "live" and k.id > 0 and k.t_seen >= t - c.lost_after_s]
            for i, a in enumerate(live):
                for b in live[i + 1:]:
                    if np.linalg.norm(a.pos - b.pos) <= c.encounter_m and (a.uncertain is None or b.uncertain is None):
                        why = f"ids {a.id} and {b.id} were within {c.encounter_m:.1f} m of each other at {t:.1f}s and may have been swapped"
                        m = next((m for m in self.meetings if {id(m["a"]), id(m["b"])} == {id(a), id(b)}), None)
                        for k in (a, b):
                            if k.uncertain is None:
                                k.uncertain, k.t_uncertain = why, t
                                if m is not None:
                                    m["marks"].append(why)
                        self._count("encounter_marks")
        return out

    def _colours(self, t: float, looks: dict) -> None:
        """Clothing colour: remember it from clear looks, and after two people met, use it to say who is who."""
        c = self.cfg
        live = [k for k in self.tracks if k.state == "live" and k.id > 0 and k.t_seen >= t - c.lost_after_s]
        crowded = set()
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if np.linalg.norm(a.pos - b.pos) <= c.app_near_m:
                    crowded |= {id(a), id(b)}
                    if a.app_n >= 3 and b.app_n >= 3 and not any({id(m["a"]), id(m["b"])} == {id(a), id(b)} for m in self.meetings):
                        self.meetings.append({"t0": t, "a": a, "b": b, "fa": [], "fb": [], "marks": []})
        for k in live:
            if id(k) not in looks or id(k) in crowded:
                continue
            mine = [(m, "fa" if m["a"] is k else "fb") for m in self.meetings if m["a"] is k or m["b"] is k]
            for m, side in mine:
                m[side].append(looks[id(k)])
            if not mine:                              # while a meeting is open the long-term colour is left alone
                k.app, k.app_n = (0 if k.app is None else k.app) + looks[id(k)], k.app_n + 1
        for m in list(self.meetings):
            if not any(q is m for q in self.meetings):
                continue                              # dropped a moment ago: one of its two was just swapped by another meeting
            a, b = m["a"], m["b"]
            here = [k.state in ("live", "lost") for k in (a, b)]
            over = t - m["t0"] > c.app_wait_s or not any(here)
            na, nb = len(m["fa"]), len(m["fb"])
            # one_side: one of the two has left the store (or the wait is over) and only the other was seen clear. Their
            # clothes are still compared with both remembered colours: half the evidence, so half the margin.
            one = c.one_side and max(na, nb) >= c.app_frames and min(na, nb) < c.app_frames and (over or not all(here))
            if (na >= c.app_frames and nb >= c.app_frames) or one:
                fa, fb = (np.sum(f, axis=0) if len(f) >= c.app_frames else None for f in (m["fa"], m["fb"]))
                pairs = [(x, f) for x, f in ((a, fa), (b, fb)) if f is not None]
                other = {id(a): b, id(b): a}
                straight, crossed = sum(app_dist(x.app, f) for x, f in pairs), sum(app_dist(other[id(x)].app, f) for x, f in pairs)
                margin = c.app_margin * len(pairs) / 2
                self.settled.append({"t0": m["t0"], "t": t, "ids": (a.id, b.id), "straight": straight, "crossed": crossed, "sides": len(pairs),
                                     "said": "cannot tell" if abs(straight - crossed) < margin else "swapped" if crossed < straight else "not swapped"})
                if abs(straight - crossed) >= margin:
                    if crossed < straight:
                        self.log.append(f"{t:7.1f}s ids {a.id} and {b.id} had been swapped when they met at {m['t0']:.1f}s: put right by clothing colour "
                                        f"(crossed {crossed:.2f}, straight {straight:.2f}{', one of them seen' if one else ''})")
                        self.swaps.append((m["t0"], t, a.id, b.id))
                        _swap_identity(a, b, m["t0"])
                        self._count("swaps_put_right")
                        self.meetings = [q for q in self.meetings if q is m or not ({id(q["a"]), id(q["b"])} & {id(a), id(b)})]
                    else:
                        self._count("meetings_told_apart")
                    for k in (a, b):                  # the doubt this meeting raised is settled
                        if k.uncertain in m["marks"]:
                            k.uncertain = k.t_uncertain = None
                else:
                    self._count("meetings_not_told_apart")
                    why = f"ids {a.id} and {b.id} met at {m['t0']:.1f}s and their clothing colours are too alike to tell who is who"
                    for k in (a, b):
                        if k.uncertain is None:
                            k.uncertain, k.t_uncertain = why, m["t0"]
                for x, f in pairs:
                    x.app = x.app + f
                self.meetings.remove(m)
            elif over or (not c.one_side and not all(here)):
                self.settled.append({"t0": m["t0"], "t": t, "ids": (a.id, b.id), "said": "never seen apart", "looks": (na, nb)})
                self.meetings.remove(m)               # never seen apart: any doubt mark stays

    def _through_door(self, xz: np.ndarray) -> bool:
        """Last seen in the doorway or past it: gone, not lost."""
        out = self.door / max(np.linalg.norm(self.door), 1e-6)          # from the middle of the store out through the door
        d = xz - self.door
        return float(d @ out) >= -self.cfg.doorway_m and abs(float(d @ np.array([-out[1], out[0]]))) <= self.cfg.door_m

    def _confirm(self, k: Track, t: float) -> None:
        """A new track has lasted: whose is it? A second box, a lost person who could have walked here, or a new person."""
        c = self.cfg
        near = sorted(((float(np.linalg.norm(q.pos - k.pos)), q) for q in self.tracks if q is not k and q.state == "live" and q.t_seen >= t - c.lost_after_s), key=lambda x: x[0])
        # a second box of a tracked person: right on top of them, or close and no camera shows the two apart
        if any(d <= c.duplicate_m or (d <= c.duplicate_far_m and not (q.cams & k.cams)) for d, q in near):
            self._count("second_box_frames")
            return                                         # stays "new": no person, no events, until it steps away or ends
        at_door = np.linalg.norm(k.pos - self.door) <= c.door_m
        heading_in = np.linalg.norm(k.pos - self.door) - np.linalg.norm(k.start - self.door) >= 0.1
        cand = []
        for q in self.tracks:
            if q.state != "lost":
                continue
            gap, d = max(k.t_first - q.t_seen, 0.1), float(np.linalg.norm(q.last - k.start))
            if d > 1.0 + c.walk_mps * gap:
                continue
            if at_door and np.linalg.norm(q.last - self.door) <= c.door_m:
                to_door = (self.door - q.last) / max(np.linalg.norm(self.door - q.last), 1e-6)
                if q.v_last @ to_door >= 0.4 and heading_in:
                    continue                               # someone walked out and someone else walks in right behind
            s = min(0.5 + c.drift_mps * gap, 6.0)           # where a lost person is after `gap`: near where they were lost
            colour = app_dist(q.app, k.app)                 # and they still wear the same clothes
            if colour is not None and d > c.other_m and colour > c.other_app:
                self._count("hand_backs_refused")
                continue                                   # far from where they were lost and in other clothes: somebody else
            cand.append((float(np.exp(-d * d / (2 * s * s)) / (s * s)) * (1.0 if colour is None else float(np.exp(-colour / c.app_scale))), q))
        cand.sort(key=lambda x: -x[0])
        if cand:
            q = cand[0][1]
            gap, colour = k.t_first - q.t_seen, app_dist(q.app, k.app)
            self.tracks.remove(q)
            k.id, k.path, k.kpts, k.born, k.staff = q.id, q.path, q.kpts, q.born, q.staff
            k.uncertain, k.t_uncertain, k.t_first = q.uncertain, q.t_uncertain, q.t_first
            k.app, k.app_n = q.app, q.app_n
            if c.one_side:                                 # the meetings the lost track was in go on with the track that carries the person now
                for m in self.meetings:
                    for side in ("a", "b"):
                        if m[side] is q:
                            m[side] = k
            rivals = [r for s, r in cand[1:] if s >= c.rival_ratio * cand[0][0]]
            if rivals:
                why = f"ids {', '.join(str(r.id) for r in [q] + rivals)} were all lost and could be the person who reappeared at {t:.1f}s"
                for r in [k] + rivals:
                    if r.uncertain is None:
                        r.uncertain, r.t_uncertain = why, min(r.t_seen, k.t_first) if r is not k else q.t_seen
                self._count("uncertain_marks")
            self._count("hand_backs")
            self.log.append(f"{t:7.1f}s id {k.id} seen again after {gap:.1f}s" + (" (uncertain)" if rivals else "")
                            + f" [{np.linalg.norm(q.last - k.start):.1f} m from where lost, colour {'none' if colour is None else format(colour, '.2f')}, lost {np.linalg.norm(q.last - self.door):.1f} m"
                              f" and back {np.linalg.norm(k.start - self.door):.1f} m from the door{', walking in' if heading_in else ''}, {len(cand)} lost fit, new track seen {k.app_n} and lost one {q.app_n} times clear]")
        elif at_door or k.t_first - self.t0 <= c.start_s:
            self.n += 1
            k.id, k.born = self.n, "door" if at_door else "start"
            self._count("births_" + k.born)
        elif any(d <= c.second_track_m and in_zone(self.staff_zone, *q.pos) == in_zone(self.staff_zone, *k.pos) for d, q in near):
            self._count("second_track_frames")
            return                                         # nobody is lost and a tracked person stands right here: a second track of them (two
            #                                                cameras place one person apart), not an entry nobody saw. If theirs is lost, this one takes over
        elif t - k.t_first < c.inside_birth_s or k.hits < c.inside_birth_share * k.frames:
            return                                         # wait: a flicker, a second box now and then, or someone about to be seen properly
        else:
            self.n += 1
            k.id, k.born = self.n, "inside"
            k.uncertain, k.t_uncertain = "first seen inside the store with nobody unaccounted for (entry not seen)", k.t_first
            self._count("births_inside")
            self.log.append(f"{t:7.1f}s id {k.id} first seen inside the store, away from the door, nobody lost fits"
                            + (f" [nearest tracked person: id {near[0][1].id}, {near[0][0]:.1f} m away, {np.linalg.norm(near[0][1].pos - k.start):.1f} m from where this track started at {k.t_first:.1f}s]" if near else ""))
        k.state = "live"

    def people(self) -> list[Track]:
        out = [k for k in self.tracks if k.id > 0]
        for k in out:
            if k.path:
                k.staff = sum(in_zone(self.staff_zone, x, z) for _, x, z in k.path) >= self.cfg.staff_share * len(k.path)
        return out

    def finish(self, t: float) -> None:
        """End of the video: whoever was last seen at the door and is not in view has left."""
        for k in self.tracks:
            if k.state == "lost" and np.linalg.norm(k.last - self.door) <= self.cfg.door_m:
                k.state, k.t_exit = "exited", k.t_seen
