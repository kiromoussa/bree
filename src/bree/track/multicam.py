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

Closed world across cameras (`closed_world=True`; StoreEvents turns it on when a camera's rules say
`closed_world: true`): ONE identity pool for the store.
  births    only at a door: a track that starts in an `exit` / `entrance` zone of the camera that sees it,
            and is not someone another camera is watching at that spot right now.
  hand-off  a track that starts anywhere else is someone already inside. Candidates are the people no track
            of this camera holds: someone another camera sees now (only if they stand at that floor
            position) and everyone lost (wherever they were last seen). One assignment over all new tracks
            of the frame, scored like the single-camera rule (bree/events/engine.py CW_WEIGHTS: appearance
            when re-ID is on, walkability on the floor plan, time since lost).
  deaths    the EXIT event of a door camera (dropped if another camera still sees them afterwards), or
            `timeout_s` unseen.
  uncertain a call closer than `margin` marks both identities; the mark stays on the store-wide identity,
            so every later event from any camera carries `identity_uncertain` and the ledger caps the
            alert at review.
In this mode the per-camera engines neither stitch nor run their own closed world: every local track is
placed here.

3D slot of a pick: with calibrated cameras (bree/calib) and a slot layout, StoreEvents adds `meta["slot"]`
to each PICK (bree/calib/slots.py SlotLocator): two views when two cameras saw the hand, else one view.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST, FrameObs
from bree.events.types import Event, EventType
from bree.track.reid import match_prob, remember


def homography(img_pts, floor_pts) -> np.ndarray:
    """3x3 H with floor ~ H @ img, from >= 4 point pairs (DLT, least squares). Both point sets are shifted
    and scaled to unit size first (Hartley): pixels in the thousands next to metres make the raw system
    badly conditioned, which shows as soon as the marks carry click noise."""
    a, b = np.asarray(img_pts, float), np.asarray(floor_pts, float)

    def norm(p):
        c, s = p.mean(0), max(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())), 1e-12) / math.sqrt(2)
        return (p - c) / s, np.array([[1 / s, 0, -c[0] / s], [0, 1 / s, -c[1] / s], [0, 0, 1]])
    (a, Ta), (b, Tb) = norm(a), norm(b)
    A = []
    for (x, y), (u, v) in zip(a, b):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, float))
    H = np.linalg.inv(Tb) @ vt[-1].reshape(3, 3) @ Ta
    H = H / np.linalg.norm(H)
    # Sign: w > 0 at the marks, so to_floor can tell a pixel on the floor from one at or above the horizon.
    return H if (H @ np.append(np.asarray(img_pts, float).mean(0), 1.0))[2] > 0 else -H


def to_floor(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Floor-plan point of a pixel, or None when the line of sight through it never meets the floor
    (the pixel is at or above the horizon: w <= 0 with the sign convention of homography() and
    Camera.floor_homography())."""
    p = H @ np.array([x, y, 1.0])
    if p[2] <= 1e-12 * max(abs(p[0]), abs(p[1]), 1.0):
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


@dataclass
class _Global:
    gid: int
    pos: tuple[float, float]
    t_last: float
    cams: set[str] = field(default_factory=set)
    gallery: list = field(default_factory=list)      # recent ReidFeatures (memory only)
    # Closed world only:
    t_first: float = 0.0
    born: str = "open"                               # entrance / warmup / no_door / missed_entry ("open" = flag off)
    exited: bool = False
    timed_out: bool = False
    uncertain_until: float = -1.0
    uncertain_why: str = ""


class MultiCamIdentity:
    def __init__(self, homographies: dict[str, np.ndarray], max_dist_m: float = 1.2, max_gap_s: float = 3.0,
                 foot_point: str = "bottom", reid: bool = False, reid_reject: float = 0.2,
                 reid_long_s: float = 60.0, reid_long_p: float = 0.3, max_speed_mps: float = 2.0,
                 reid_max_age_s: float = 7200.0, closed_world: bool = False, entry: dict | None = None,
                 timeout_s: float = 3600.0, warmup_s: float = 5.0, margin: float = 0.1,
                 uncertain_s: float = 3600.0, min_store_time_s: float = 2.0, exit_veto_s: float = 1.0,
                 door_gap_s: float = 0.5, frame_sizes: dict | None = None, edge_px: float = 2.0):
        self.H = homographies
        # camera -> (width, height). With it, a person box whose bottom is within edge_px of the frame bottom
        # has its feet out of frame: the box bottom is then not where they stand (_foot).
        self.frame_sizes, self.edge_px = frame_sizes or {}, edge_px
        self._cam_t: dict[str, float] = {}                       # time of each camera's latest frame
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
        # Closed world (module docstring). entry: camera -> f(PersonObs) -> True when the person stands in
        # that camera's door zone. No entry at all = a store with no door in view: births are "no_door".
        self.closed_world, self.entry = closed_world, entry or {}
        self.timeout_s, self.warmup_s, self.margin, self.uncertain_s = timeout_s, warmup_s, margin, uncertain_s
        self.min_store_time_s, self.exit_veto_s, self.door_gap_s = min_store_time_s, exit_veto_s, door_gap_s
        self.t0: float | None = None
        self.cw_stats: Counter = Counter()
        self.log: list[str] = []
        self._unplaced: dict[tuple[str, int], float] = {}        # second box on someone visible: first seen

    def update(self, cam: str, obs: FrameObs) -> None:
        """Assign/refresh global ids for this camera's tracks at this frame."""
        # A global id is taken in this camera only while one of its local tracks is still live
        # here. (Excluding every id the camera ever held meant a shopper who left the door
        # camera's view and came back to leave got a new id, and the visit never reconciled.)
        self.local_seen = {k: t for k, t in self.local_seen.items() if obs.t - t <= self.max_gap_s
                           and k in self.local_to_global}
        claimed_now = {self.local_to_global[k] for k in self.local_seen if k[0] == cam}
        self._cam_t[cam] = obs.t
        if self.closed_world:
            self._place(cam, obs)
        for p in obs.persons:
            pos, sure = self._foot(cam, p)
            key = (cam, p.track_id)
            gid = self.local_to_global.get(key)
            if gid is None and (self.closed_world or pos is None):
                continue                                     # left unplaced this frame (second box on someone)
            self.local_seen[key] = obs.t
            if gid is None:
                gid = self._match(cam, pos, obs.t, exclude=claimed_now, f=p.reid if self.reid else None)
                if gid is None:
                    gid = self._next
                    self._next += 1
                    self.people[gid] = _Global(gid, pos, obs.t)
                else:
                    self.handoffs.append((obs.t, cam, p.track_id, gid))
                    self.handed.add(key)
                self.local_to_global[key] = gid
                claimed_now.add(gid)
            g = self.people[gid]
            if sure:                                         # feet out of frame: keep the last position that had them
                g.pos = pos
            g.t_last = obs.t
            g.cams.add(cam)
            if self.reid:
                remember(g.gallery, p.reid, obs.t, self.reid_max_age_s)
        if self.reid and not self.closed_world:     # forget appearance once nobody can be linked to it any more
            for g in self.people.values():
                if g.gallery and obs.t - g.t_last > max(self.max_gap_s, self.reid_long_s):
                    g.gallery.clear()

    def _foot(self, cam: str, p) -> tuple[tuple[float, float] | None, bool]:
        """(floor position of the person or None, whether it can be trusted). None: the foot pixel is at or
        above the horizon. Not trusted: the box bottom touches the frame bottom, so the feet are out of frame
        and the box bottom is some point up the body (a floor position metres too far from the camera)."""
        pos = to_floor(self.H[cam], *p.foot_point(self.foot_point))
        h = self.frame_sizes.get(cam, (0, 0))[1]
        return pos, pos is not None and not (h and self.foot_point == "bottom" and p.bbox[3] >= h - self.edge_px)

    def _watched_elsewhere(self, g: _Global, cam: str, t: float) -> bool:
        """Another camera has this person in its latest frame, and that frame is at most door_gap_s old."""
        return any(c != cam and gid == g.gid and (ts := self.local_seen.get((c, tid))) is not None
                   and ts >= self._cam_t.get(c, math.inf) and t - ts <= self.door_gap_s
                   for (c, tid), gid in self.local_to_global.items())

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

    # ---------------------------------------------------------- closed world

    def occupancy(self) -> int:
        """People inside the store right now, seen or not: identities created minus exits and timeouts."""
        return sum(1 for g in self.people.values() if not g.exited and not g.timed_out
                   and (g.born == "entrance" or g.t_last - g.t_first >= self.min_store_time_s))

    def _birth(self, key, pos, t: float, how: str) -> None:
        gid, self._next = self._next, self._next + 1
        self.people[gid] = _Global(gid, pos, t, t_first=t, born=how)
        self.local_to_global[key] = gid
        self.cw_stats[f"births_{how}"] += 1
        if how == "missed_entry":
            self.log.append(f"{t:.1f}s {key[0]} track {key[1]} appeared inside with nobody unaccounted for: "
                            f"missed entry, new person {gid}")

    def _place(self, cam: str, obs: FrameObs) -> None:
        """Every track of this camera not seen before: a hand-off, one of the lost people, or a birth."""
        from scipy.optimize import linear_sum_assignment

        from bree.events.engine import CW_WEIGHTS
        t = obs.t
        if self.t0 is None:
            self.t0 = t
        for g in self.people.values():                       # deaths by timeout
            if not g.exited and not g.timed_out and t - g.t_last > self.timeout_s:
                g.timed_out = True
                g.gallery.clear()
                self.cw_stats["timeouts"] += 1
                self.log.append(f"{t:.1f}s person {g.gid} not seen for {self.timeout_s:.0f}s without an exit: dropped")
        for p in obs.persons:                                # a track of someone who exited is seen again
            key = (cam, p.track_id)
            g = self.people.get(self.local_to_global.get(key))
            if g is not None and (g.exited or g.timed_out):
                del self.local_to_global[key]                # placed again below: a door birth or a hand-off
                self.handed.discard(key)
                self.cw_stats["tracks_seen_after_exit"] += 1
                self.log.append(f"{t:.1f}s {cam} track {p.track_id} is seen again after person {g.gid} "
                                f"{'exited' if g.exited else 'timed out'}: placed again")
        new, pos, cut = [], [], 0
        for p in obs.persons:
            if (cam, p.track_id) in self.local_to_global:
                continue
            xy, sure = self._foot(cam, p)
            if xy is None:                                   # no floor position at all: cannot be placed
                self.cw_stats["frames_no_floor_point"] += 1
                continue
            new.append(p)
            pos.append(xy)
            cut += not sure
        if not new:
            return
        self.cw_stats["placed_with_feet_out_of_frame"] += cut
        here = {self.local_to_global[(cam, p.track_id)] for p in obs.persons if (cam, p.track_id) in self.local_to_global}
        door = [bool(self.entry.get(cam, lambda p: False)(p)) for p in new]
        pool = [g for g in self.people.values() if not g.exited and not g.timed_out and g.gid not in here]
        S = np.full((len(new), len(pool)), -1.0)             # -1 = not a candidate
        P: dict[tuple[int, int], float | None] = {}
        wa, ww, wt = CW_WEIGHTS
        for i, p in enumerate(new):
            for j, g in enumerate(pool):
                gap = max(t - g.t_last, 0.0)
                d = math.dist(pos[i], g.pos)
                reach = self.max_dist_m + self.max_speed_mps * gap
                # Another camera sees them now (or this one just did). At the door the test is strict: only someone
                # in another camera's latest frame. A track that ended at the door a moment ago is someone who
                # walked out, and must not be handed to the next person walking in, however short the gap.
                live = self._watched_elsewhere(g, cam, t) if door[i] else gap <= self.max_gap_s
                prob = match_prob(g.gallery, p.reid) if self.reid and p.reid is not None else None
                if live:
                    if d > reach or (prob is not None and prob < self.reid_reject):
                        continue                             # seen somewhere else, or looks different: not them
                elif door[i] or not (g.born == "entrance" or g.t_last - g.t_first >= self.min_store_time_s):
                    continue                                 # a track at the door is not a lost person; flickers never are
                app = 0.5 if prob is None else min(1.0, prob / self.reid_long_p)
                S[i, j] = wa * app + ww / (1.0 + (d / reach) ** 2) + wt * math.exp(-gap / 60.0)
                P[i, j] = prob
        pairs = [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(-S)) if S[i, j] >= 0] if pool else []
        col_of = dict(pairs)
        for i, j in pairs:
            p, g, prob = new[i], pool[j], P[i, j]
            key = (cam, p.track_id)
            gap = t - g.t_last
            self.local_to_global[key] = g.gid
            self.handoffs.append((t, cam, p.track_id, g.gid))
            self.handed.add(key)
            self.cw_stats["handoffs_live" if gap <= self.max_gap_s else "assigned_lost"] += 1
            rivals = [(S[i, k], pool[k]) for k in range(len(pool)) if k != j and S[i, k] >= 0]
            rivals += [(S[k, j], pool[col_of[k]] if k in col_of else None) for k in range(len(new)) if k != i and S[k, j] >= 0]
            s2, other = max(rivals, key=lambda r: r[0]) if rivals else (-1.0, None)
            why = None
            if rivals and S[i, j] - s2 < self.margin:
                why = (f"{cam} track {p.track_id} could be person {g.gid} or "
                       f"{other.gid if other else 'a missed entry'} (scores {S[i, j]:.2f} vs {s2:.2f})")
            elif prob is not None and prob < self.reid_reject:
                why, other = f"{cam} track {p.track_id} taken as person {g.gid} but looks different (p={prob:.2f})", None
            if why:
                for q in ([g, other] if other else [g]):
                    q.uncertain_until, q.uncertain_why = t + self.uncertain_s, why
                    self.cw_stats["uncertain_marks"] += 1
                self.log.append(f"{t:.1f}s identity uncertain: {why}")
        waiting = set()
        for i, p in enumerate(new):
            if i in col_of:
                continue
            key = (cam, p.track_id)
            if door[i]:
                self._birth(key, pos[i], t, "entrance")
            elif t - self.t0 <= self.warmup_s:
                self._birth(key, pos[i], t, "warmup")
            else:
                # More new tracks than people unaccounted for. A box that starts on top of someone this camera
                # already tracks is far more often a second detection of them than an entry nobody saw: wait
                # (no person, no events) until it steps away or has lasted min_store_time_s.
                since = self._unplaced.setdefault(key, t)
                if t - since < self.min_store_time_s and any(
                        math.dist(pos[i], self.people[gid].pos) <= self.max_dist_m for gid in here):
                    self.cw_stats["second_box_frames_ignored"] += 1
                    waiting.add(key)
                    continue
                self._birth(key, pos[i], t, "missed_entry" if self.entry else "no_door")
        self._unplaced = {k: v for k, v in self._unplaced.items() if k in waiting or k[0] != cam}

    def remap(self, cam: str, events: list[Event]) -> list[Event]:
        out = []
        for ev in events:
            gid = self.local_to_global.get((cam, ev.person_id))
            if gid is None:
                continue
            if ev.type == EventType.ENTER and (cam, ev.person_id) in self.handed:
                continue          # same person walking into another camera's view
            g = self.people[gid]
            if self.closed_world:
                if ev.type == EventType.EXIT:
                    if g.exited:
                        continue                                 # a second door camera saw the same exit
                    if g.t_last > ev.t + self.exit_veto_s:
                        self.cw_stats["exits_vetoed"] += 1       # another camera has seen them since: still inside
                        self.log.append(f"{ev.t:.1f}s person {gid} left {cam}'s door zone but was seen after: no exit")
                        continue
                    g.exited = True
                    g.gallery.clear()
                    self.cw_stats["exits"] += 1
                if ev.type == EventType.ENTER:
                    ev.meta = {**ev.meta, "born": g.born, **({"missed_entry": True} if g.born == "missed_entry" else {})}
                if ev.t <= g.uncertain_until:                    # the ledger caps this person's alerts at review
                    ev.meta = {**ev.meta, "identity_uncertain": g.uncertain_why,
                               "identity_uncertain_until": g.uncertain_until}
            ev.person_id = gid
            ev.candidates = [self.local_to_global.get((cam, c), c) for c in ev.candidates]
            ev.meta = {**ev.meta, "camera": cam}
            out.append(ev)
        return out


class StoreEvents:
    """One event stream per store: an EventEngine per camera (its own zones, in its own pixels)
    and, with several cameras, floor-plane handoff so every event carries a global person id.
    One camera: the engine's events pass through unchanged (local id = person id).

    `handoff`: MultiCamIdentity settings, plus three keys used here:
      calibration  {camera: bree.calib Camera or its dict}. Default: each store's `calibration` (the
                   `camera.calibration` block scripts/calibrate.py writes). A calibrated camera's floor
                   mapping comes from its pose instead of the raw floor marks.
      slots        a layout JSON path, a layout dict or a SlotMap. Default: `rules.slot_layout` of any camera.
                   With it and at least one calibrated camera, every PICK gets `meta["slot"]`.
      noise        bree.calib.slots.NoiseModel (or its dict) for the triangulation error."""

    def __init__(self, stores: dict, handoff: dict | None = None):
        from bree.events.engine import ENTRY_KINDS, EventEngine
        handoff = dict(handoff or {})
        self.engines = {name: EventEngine(s) for name, s in stores.items()}
        self.identity: MultiCamIdentity | None = None
        self.cameras, self.locator = self._calibration(stores, handoff)
        if len(stores) > 1:
            missing = [n for n, s in stores.items() if s.floor_points is None and n not in self.cameras]
            if missing:
                raise ValueError(f"multi-camera store: camera(s) {missing} need camera.floor_points "
                                 "(4+ [x_px, y_px, x_m, y_m] marks) in their store YAML")
            first = next(iter(self.engines.values())).r
            foot = next(iter(stores.values())).rules.get("foot_point", "bottom")
            cw = handoff.pop("closed_world", any(e.r.closed_world for e in self.engines.values()))
            for e in self.engines.values():
                if e.r.closed_world or cw:
                    # One pool for the store (MultiCamIdentity). A per-camera pool would take a person walking
                    # in from another camera's area for someone lost in its own view.
                    e.r.closed_world = False
                    if cw:
                        e.r.stitch_dist = 0.0
                    e.log.append("closed world runs store-wide (one identity pool for all cameras)" if cw
                                 else "closed_world is off for this multi-camera store (handoff closed_world: false)")
            if cw:
                entry = {n: (lambda p, s=s: s.zone_at(*p.foot_point(foot), *ENTRY_KINDS) is not None)
                         for n, s in stores.items() if s.zones_of(*ENTRY_KINDS)}
                handoff = {"closed_world": True, "entry": entry, "timeout_s": first.closed_world_timeout_s,
                           "warmup_s": first.closed_world_warmup_s, "margin": first.closed_world_margin,
                           "uncertain_s": first.closed_world_uncertain_s, "min_store_time_s": first.min_store_time_s,
                           **handoff}
            handoff = {"reid": any(e.r.reid for e in self.engines.values()),
                       "frame_sizes": {n: s.resolution for n, s in stores.items()}, **handoff}
            self.identity = MultiCamIdentity(
                {n: self.cameras[n].floor_homography() if n in self.cameras else s.floor_homography()
                 for n, s in stores.items()}, foot_point=foot, **handoff)

    def _calibration(self, stores: dict, handoff: dict):
        """(calibrated cameras, SlotLocator or None). Pops the keys MultiCamIdentity does not know."""
        given, slots, noise = handoff.pop("calibration", None) or {}, handoff.pop("slots", None), handoff.pop("noise", None)
        cams = {}
        for n, s in stores.items():
            c = given.get(n) or getattr(s, "calibration", None)
            if c:
                from bree.calib.camera import Camera
                cams[n] = Camera.from_dict(c) if isinstance(c, dict) else c
        slots = slots or next((s.rules["slot_layout"] for s in stores.values() if s.rules.get("slot_layout")), None)
        if not slots or not cams:
            return cams, None
        import json
        from pathlib import Path

        from bree.calib.slots import NoiseModel, SlotLocator, SlotMap
        if isinstance(slots, (str, Path)):
            slots = json.loads(Path(slots).read_text())
        r = next(iter(self.engines.values())).r
        return cams, SlotLocator(cams, slots if isinstance(slots, SlotMap) else SlotMap(slots),
                                 NoiseModel(**noise) if isinstance(noise, dict) else noise, extend=r.hand_extend)

    def _slots(self, cam: str, obs: FrameObs, events: list[Event]) -> None:
        """Feed the hands seen in this frame to the locator, and give each PICK its slot. Runs before the
        ids are made global: the engine's own person record says which hand and which reach."""
        eng = self.engines[cam]
        gid = (lambda tid: self.identity.local_to_global.get((cam, tid))) if self.identity else (lambda tid: eng.alias.get(tid, tid))
        for p in obs.persons:
            g = gid(p.track_id)
            hands = {name: (w, p.kpt(e, eng.r.kpt_conf)) for name, wi, e in (("left", L_WRIST, L_ELBOW), ("right", R_WRIST, R_ELBOW))
                     if (w := p.kpt(wi, eng.r.kpt_conf)) is not None}
            if g is not None and hands:
                self.locator.observe(cam, obs.t, g, hands)
        for ev in events:
            ps, g = eng.people.get(ev.person_id), gid(ev.person_id)
            if ev.type != EventType.PICK or ps is None or g is None:
                continue
            track = ev.meta.get("product_track")
            held = ps.held.get(track)
            reach = next((rc for rc in reversed(ps.reaches) if rc.zone == ev.zone and rc.t_start <= ev.t), None)
            t0, t1 = (reach.t_start, reach.t_end if reach.t_end is not None else ev.t) if reach else (ev.t - eng.r.reach_window_s, ev.t)
            # The item's own pixel counts only if the detector first saw it on that shelf (not already in the hand).
            first = eng.prod_first.get(track)
            on_shelf = first is not None and (z := eng.store.zone_at(*first, "shelf", "cooler")) is not None and z.name == ev.zone
            slot = self.locator.locate(cam, g, held.wrist if held else None, t0, t1, item_uv=first if on_shelf else None)
            if slot is not None:
                ev.meta = {**ev.meta, "slot": slot}

    def update(self, cam: str, obs: FrameObs) -> list[Event]:
        if self.identity is not None:
            self.identity.update(cam, obs)
        events = self.engines[cam].update(obs)
        if self.locator is not None:
            self._slots(cam, obs, events)
        return events if self.identity is None else self.identity.remap(cam, events)

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
