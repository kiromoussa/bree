"""Multi-camera handoff without faces.

Each camera maps its pixels to a shared store floor plan (metres) with a
homography computed from 4+ floor points marked once at install time. A person
keeps one GLOBAL id across cameras by position and timing only:

  - a new local track in camera B is linked to the global person whose last
    floor position (seen by any camera) is within `max_dist_m` and who was seen
    within `max_gap_s` (overlapping views: at the same moment; adjacent views:
    just after leaving the other camera);
  - otherwise it's a new person.

Events from every camera's EventEngine go through `remap()` before the ledger:
local ids become global ids, and duplicate ENTERs of an already known person are
dropped. Only cameras that can see the door should have an `exit` zone.
`StoreEvents` bundles the per-camera engines + this handoff; the pipeline
(`bree run` with several --source/--store pairs, `bree shadow` with `multicam:`)
feeds its events to ONE ledger per store.
With `reid=True` (bree/track/reid.py: body appearance, never the face) a position/time candidate that
looks different is rejected, appearance settles which of several nearby people it is, and someone seen up
to `reid_long_s` ago can be linked at a high match probability if they could have walked there at
`max_speed_mps` (cameras that don't overlap, e.g. cooler camera -> register camera).
Privacy: appearance features live only in memory and only for one visit (DECISIONS: re-ID); with reid off
none are used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import FrameObs
from bree.events.types import Event, EventType
from bree.track.reid import match_prob, remember


def homography(img_pts, floor_pts) -> np.ndarray:
    """3x3 H with floor ~ H @ img, from >= 4 point pairs (DLT, least squares)."""
    A = []
    for (x, y), (u, v) in zip(img_pts, floor_pts):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def to_floor(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass
class _Global:
    gid: int
    pos: tuple[float, float]
    t_last: float
    cams: set[str] = field(default_factory=set)
    gallery: list = field(default_factory=list)      # recent ReidFeatures (memory only)


class MultiCamIdentity:
    def __init__(self, homographies: dict[str, np.ndarray], max_dist_m: float = 1.2, max_gap_s: float = 3.0,
                 foot_point: str = "bottom", reid: bool = False, reid_reject: float = 0.2,
                 reid_long_s: float = 60.0, reid_long_p: float = 0.3, max_speed_mps: float = 2.0,
                 reid_max_age_s: float = 7200.0):
        self.H = homographies
        self.reid, self.reid_reject = reid, reid_reject
        self.reid_long_s, self.reid_long_p, self.max_speed_mps = reid_long_s, reid_long_p, max_speed_mps
        self.reid_max_age_s = reid_max_age_s
        self.max_dist_m, self.max_gap_s = max_dist_m, max_gap_s
        self.foot_point = foot_point
        self.local_to_global: dict[tuple[str, int], int] = {}
        self.people: dict[int, _Global] = {}
        self._next = 1
        self.handoffs: list[tuple[float, str, int, int]] = []   # (t, camera, local id, global id)
        self.local_seen: dict[tuple[str, int], float] = {}      # live local tracks: last seen time
        self.handed: set[tuple[str, int]] = set()                # local tracks linked to an earlier person
        # ponytail: local_to_global / people / handoffs grow for the process lifetime (like
        # Ledger.people); prune by t_last if a store runs for weeks without a restart.

    def update(self, cam: str, obs: FrameObs) -> None:
        """Assign/refresh global ids for this camera's tracks at this frame."""
        # A global id is taken in this camera only while one of its local tracks is still live
        # here. (Excluding every id the camera ever held meant a shopper who left the door
        # camera's view and came back to leave got a new id, and the visit never reconciled.)
        self.local_seen = {k: t for k, t in self.local_seen.items() if obs.t - t <= self.max_gap_s}
        claimed_now = {self.local_to_global[k] for k in self.local_seen if k[0] == cam}
        for p in obs.persons:
            self.local_seen[(cam, p.track_id)] = obs.t
            fx, fy = to_floor(self.H[cam], *p.foot_point(self.foot_point))
            key = (cam, p.track_id)
            gid = self.local_to_global.get(key)
            if gid is None:
                gid = self._match(cam, (fx, fy), obs.t, exclude=claimed_now, f=p.reid if self.reid else None)
                if gid is None:
                    gid = self._next
                    self._next += 1
                    self.people[gid] = _Global(gid, (fx, fy), obs.t)
                else:
                    self.handoffs.append((obs.t, cam, p.track_id, gid))
                    self.handed.add(key)
                self.local_to_global[key] = gid
                claimed_now.add(gid)
            g = self.people[gid]
            g.pos, g.t_last = (fx, fy), obs.t
            g.cams.add(cam)
            if self.reid:
                remember(g.gallery, p.reid, obs.t, self.reid_max_age_s)
        if self.reid:     # forget appearance once nobody can be linked to it any more
            for g in self.people.values():
                if g.gallery and obs.t - g.t_last > max(self.max_gap_s, self.reid_long_s):
                    g.gallery.clear()

    def _match(self, cam: str, pos, t: float, exclude: set[int], f=None) -> int | None:
        cands = []                                   # (distance m, gid, match probability or None)
        for g in self.people.values():
            gap = t - g.t_last
            if g.gid in exclude or gap > max(self.max_gap_s, self.reid_long_s if f is not None else 0.0):
                continue
            d = math.dist(pos, g.pos)
            p = match_prob(g.gallery, f) if f is not None else None
            ok = gap <= self.max_gap_s and d <= self.max_dist_m
            if not ok and p is not None and p >= self.reid_long_p and d <= self.max_dist_m + self.max_speed_mps * gap:
                ok = True
            if ok and (p is None or p >= self.reid_reject):
                cands.append((d, g.gid, p))
        if not cands:
            return None
        if f is not None and all(c[2] is not None for c in cands):
            return max(cands, key=lambda c: (c[2], -c[0]))[1]     # best-looking match; ties: nearest
        return min(cands)[1]

    def remap(self, cam: str, events: list[Event]) -> list[Event]:
        out = []
        for ev in events:
            gid = self.local_to_global.get((cam, ev.person_id))
            if gid is None:
                continue
            if ev.type == EventType.ENTER and (cam, ev.person_id) in self.handed:
                continue          # same person walking into another camera's view
            ev.person_id = gid
            ev.candidates = [self.local_to_global.get((cam, c), c) for c in ev.candidates]
            ev.meta = {**ev.meta, "camera": cam}
            out.append(ev)
        return out


class StoreEvents:
    """One event stream per store: an EventEngine per camera (its own zones, in its own pixels)
    and, with several cameras, floor-plane handoff so every event carries a global person id.
    One camera: the engine's events pass through unchanged (local id = person id)."""

    def __init__(self, stores: dict, handoff: dict | None = None):
        from bree.events.engine import EventEngine
        self.engines = {name: EventEngine(s) for name, s in stores.items()}
        self.identity: MultiCamIdentity | None = None
        if len(stores) > 1:
            missing = [n for n, s in stores.items() if s.floor_points is None]
            if missing:
                raise ValueError(f"multi-camera store: camera(s) {missing} need camera.floor_points "
                                 "(4+ [x_px, y_px, x_m, y_m] marks) in their store YAML")
            for e in self.engines.values():
                if e.r.closed_world:
                    # ponytail: closed-world identity is per camera for now. With several cameras a person walking in
                    # from another camera's area would be taken for someone lost here, so it is switched off and the
                    # floor-plane handoff below stays in charge. Upgrade: one lost pool per store in MultiCamIdentity.
                    e.r.closed_world = False
                    e.log.append("closed_world is single-camera only: off for this multi-camera store")
            foot = next(iter(stores.values())).rules.get("foot_point", "bottom")
            handoff = {"reid": any(e.r.reid for e in self.engines.values()), **(handoff or {})}
            self.identity = MultiCamIdentity({n: s.floor_homography() for n, s in stores.items()},
                                             foot_point=foot, **handoff)

    def update(self, cam: str, obs: FrameObs) -> list[Event]:
        if self.identity is None:
            return self.engines[cam].update(obs)
        self.identity.update(cam, obs)
        return self.identity.remap(cam, self.engines[cam].update(obs))

    def flush(self) -> list[Event]:
        """End of stream on every camera."""
        if self.identity is None:
            return next(iter(self.engines.values())).flush()
        out = []
        for cam, eng in self.engines.items():
            out += self.identity.remap(cam, eng.flush())
        return sorted(out, key=lambda e: e.t)

    def person_id(self, cam: str, local_id: int) -> int | None:
        return self.identity.local_to_global.get((cam, local_id)) if self.identity else local_id
