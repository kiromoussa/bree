"""Multi-camera handoff without faces or appearance embeddings.

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
Privacy: no appearance features are computed or stored.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import FrameObs
from bree.events.types import Event, EventType


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


class MultiCamIdentity:
    def __init__(self, homographies: dict[str, np.ndarray], max_dist_m: float = 1.2, max_gap_s: float = 3.0,
                 foot_point: str = "bottom"):
        self.H = homographies
        self.max_dist_m, self.max_gap_s = max_dist_m, max_gap_s
        self.foot_point = foot_point
        self.local_to_global: dict[tuple[str, int], int] = {}
        self.people: dict[int, _Global] = {}
        self._next = 1
        self.handoffs: list[tuple[float, str, int, int]] = []   # (t, camera, local id, global id)

    def update(self, cam: str, obs: FrameObs) -> None:
        """Assign/refresh global ids for this camera's tracks at this frame."""
        claimed_now = {g for (c, _), g in self.local_to_global.items() if c == cam}
        for p in obs.persons:
            fx, fy = to_floor(self.H[cam], *p.foot_point(self.foot_point))
            key = (cam, p.track_id)
            gid = self.local_to_global.get(key)
            if gid is None:
                gid = self._match(cam, (fx, fy), obs.t, exclude=claimed_now)
                if gid is None:
                    gid = self._next
                    self._next += 1
                    self.people[gid] = _Global(gid, (fx, fy), obs.t)
                else:
                    self.handoffs.append((obs.t, cam, p.track_id, gid))
                self.local_to_global[key] = gid
                claimed_now.add(gid)
            g = self.people[gid]
            g.pos, g.t_last = (fx, fy), obs.t
            g.cams.add(cam)

    def _match(self, cam: str, pos, t: float, exclude: set[int]) -> int | None:
        best, best_d = None, self.max_dist_m
        for g in self.people.values():
            if g.gid in exclude or t - g.t_last > self.max_gap_s:
                continue
            d = math.dist(pos, g.pos)
            if d <= best_d:
                best, best_d = g.gid, d
        return best

    def remap(self, cam: str, events: list[Event]) -> list[Event]:
        out = []
        for ev in events:
            gid = self.local_to_global.get((cam, ev.person_id))
            if gid is None:
                continue
            is_handoff = any(h[1] == cam and h[2] == ev.person_id for h in self.handoffs)
            if ev.type == EventType.ENTER and is_handoff:
                continue          # same person walking into another camera's view
            ev.person_id = gid
            ev.candidates = [self.local_to_global.get((cam, c), c) for c in ev.candidates]
            ev.meta = {**ev.meta, "camera": cam}
            out.append(ev)
        return out
