"""Rule-based store event engine: per-frame perception -> store events.

Readable heuristics, one rule per event. All thresholds live in `EngineRules`
and can be overridden from the store YAML (`rules:` section).

  ENTER    first frame a person track appears.
  PICK     a product track shows up in someone's hand (within hold_radius of a
           wrist, outside any shelf/cooler zone) for hold_min_frames, shortly
           after that person's wrist was inside a shelf/cooler zone. The pick is
           credited to that zone. If other people reached into the same zone at
           about the same time, they are listed as candidates (crowded pick).
  PUT_BACK an in-hand product leaves the hand while the hand or the product is
           inside a shelf/cooler zone.
  CONCEAL  an in-hand product vanishes (track lost) while it was inside the
           person's torso box (shoulders-to-hips: pockets, waistband, bag at the
           hip) and the person is NOT at the register.
  PAY      person's feet stay in the register zone for register_dwell_s
           (phase=start), then leave it (phase=end). The actual payment content
           comes from the POS via the ledger.
  EXIT     person track ends while their feet were in the exit zone, after
           having been inside the store. Carries the items still in hand.

What happens to an in-hand product that is released anywhere else (on the
counter, or just lost to occlusion) is logged but produces no event.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.events.types import Event, EventType
from bree.events.zones import StoreConfig


@dataclass
class EngineRules:
    kpt_conf: float = 0.3
    foot_point: str = "bottom"      # "bottom" for angled CCTV, "center" for top-down views
    hold_radius_px: float = 60.0    # product centre within this of a wrist = in that hand
    hold_min_frames: int = 3        # consecutive-ish frames in hand before we believe it
    reach_window_s: float = 1.5     # in-hand item appearing this soon after a reach = pick from that zone
    crowd_window_s: float = 1.0     # others reaching into the same zone within this = ambiguous pick
    release_s: float = 0.5          # in-hand item away from the hand this long = released
    relink_s: float = 1.0           # same-category item reappearing in hand this soon = same item (occlusion)
    torso_margin: float = 0.25
    register_dwell_s: float = 2.5
    register_leave_s: float = 0.7   # brief steps out of the register zone don't end the visit
    person_lost_s: float = 1.5      # person unseen this long inside the store = lost (revived if seen again)
    exit_confirm_s: float = 3.0     # person unseen this long after being at the door = EXIT (brief door occlusions aren't exits)
    conceal_confirm_s: float = 1.5  # in-hand item must stay unseen this long before CONCEAL (cancelled if it reappears)
    zone_pad_px: float = 25.0       # a picked product must have been seen within this of the zone polygon
    min_pick_move_px: float = 15.0  # ...and must have moved at least this far (a hand brushing past isn't a pick)
    handoff_frames: int = 5         # frames near someone else's hand (owner's hand visibly away) to transfer an item
    min_store_time_s: float = 2.0   # shorter tracks never count as a store visit
    stitch_s: float = 10.0          # a NEW track id appearing away from the door this soon after someone was lost
    stitch_dist: float = 1.0        # ...within this many of their body heights continues their visit (0 = off)
    hand_extend: float = 0.5        # reach point = wrist + this * (wrist - elbow): MERL test reach recall 64% -> 88% (DECISIONS)

    @staticmethod
    def from_dict(d: dict) -> "EngineRules":
        known = EngineRules.__dataclass_fields__
        return EngineRules(**{k: v for k, v in d.items() if k in known})


@dataclass
class Reach:
    zone: str
    wrist: str
    t_start: float
    t_end: float | None = None      # None while the wrist is still in the zone


@dataclass
class HeldItem:
    product_id: int
    category: str
    t_first: float
    wrist: str | None = None                      # hand that first held it ("left"/"right")
    frames: int = 1
    confirmed: bool = False
    confs: list[float] = field(default_factory=list)
    t_near: float = 0.0                           # last time it was within reach of the holder's hand
    near_zone: str | None = None                  # shelf/cooler zone hand or item was in, at t_near
    near_torso: bool = False                      # item inside holder's torso box at t_near
    at_register: bool = False                     # holder at register at t_near
    released: bool = False
    picked: bool = False                          # a PICK was emitted for it (vs. brought in / re-acquired)
    handoff: dict[int, int] = field(default_factory=dict)   # other person -> consecutive frames near them


@dataclass
class PersonState:
    pid: int
    t_first: float
    t_last: float
    last: PersonObs
    been_inside: bool = False                     # feet seen outside the exit zone
    in_exit_zone: bool = False
    wrist_zone: dict[str, str | None] = field(default_factory=dict)
    reaches: list[Reach] = field(default_factory=list)
    register_since: float | None = None
    register_last_in: float | None = None
    register_open: bool = False
    held: dict[int, HeldItem] = field(default_factory=dict)
    picked_tracks: set[int] = field(default_factory=set)      # product tracks this person already picked
    conceal_pending: list[HeldItem] = field(default_factory=list)
    done: bool = False
    exited: bool = False


class EventEngine:
    def __init__(self, store: StoreConfig, rules: EngineRules | None = None):
        self.store = store
        self.r = rules or EngineRules.from_dict(store.rules)
        self.people: dict[int, PersonState] = {}
        self.alias: dict[int, int] = {}          # tracker id -> person id it was stitched to
        self._present: set[int] = set()
        self.owner: dict[int, int] = {}          # product track id -> person id holding it
        self.log: list[str] = []                 # non-event observations (occlusion losses etc.)
        # Product track history (for pick/conceal evidence): last seen time, first seen centre,
        # and the last time it was seen in (or right next to) each shelf/cooler zone.
        self.prod_seen: dict[int, float] = {}
        self.prod_first: dict[int, tuple[float, float]] = {}
        self.prod_center: dict[int, tuple[float, float]] = {}
        self.prod_in_zone: dict[int, dict[str, float]] = {}
        self.t = 0.0
        self.frame = 0

    # ------------------------------------------------------------------ main

    def update(self, obs: FrameObs) -> list[Event]:
        self.t, self.frame = obs.t, obs.frame
        events: list[Event] = []
        seen = set()
        self._present = {self.alias.get(po.track_id, po.track_id) for po in obs.persons}   # never stitch onto these
        for po in obs.persons:
            events += self._update_person(po)
            seen.add(self.alias.get(po.track_id, po.track_id))
        events += self._update_products(obs.products)
        events += self._check_releases()
        events += self._check_conceals()
        events += self._check_gone(seen)
        return events

    def flush(self) -> list[Event]:
        """End of stream: everyone still visible is treated as gone now."""
        self.t += max(self.r.person_lost_s, self.r.exit_confirm_s, self.r.conceal_confirm_s) + 1e-3
        return self._check_releases() + self._check_conceals() + self._check_gone(set())

    # --------------------------------------------------------------- persons

    def _ev(self, type_: EventType, pid: int, **kw) -> Event:
        return Event(type=type_, t=kw.pop("t", self.t), person_id=pid, frame=self.frame, **kw)

    def _update_person(self, po: PersonObs) -> list[Event]:
        events = []
        ps = self.people.get(self.alias.get(po.track_id, po.track_id))
        if ps is None:
            ps = self._stitch(po)
        if ps is None:
            ps = PersonState(po.track_id, self.t, self.t, po)
            self.people[po.track_id] = ps
            events.append(self._ev(EventType.ENTER, po.track_id))
        if ps.done and not ps.exited:
            # Lost inside the store (occlusion) and now back: resume the same visit.
            ps.done = False
            self.log.append(f"{self.t:.1f}s person {ps.pid} seen again after being lost: resumed")
        if ps.done:
            return events
        ps.t_last, ps.last = self.t, po

        fx, fy = po.foot_point(self.r.foot_point)
        exit_zone = self.store.zone_at(fx, fy, "exit")
        ps.in_exit_zone = exit_zone is not None
        if not ps.in_exit_zone:
            ps.been_inside = True

        # Register dwell -> PAY start / end.
        reg = self.store.zone_at(fx, fy, "register")
        if reg is not None:
            if ps.register_since is None:
                ps.register_since = self.t
            ps.register_last_in = self.t
            if not ps.register_open and self.t - ps.register_since >= self.r.register_dwell_s:
                ps.register_open = True
                events.append(self._ev(EventType.PAY, ps.pid, zone=reg.name,
                                       meta={"phase": "start", "t_start": ps.register_since,
                                             "source": "register_dwell"}))
        elif ps.register_last_in is not None and self.t - ps.register_last_in > self.r.register_leave_s:
            events += self._close_register(ps)

        # Wrists in shelf/cooler zones -> reaches.
        wrists = dict(po.hands(self.r.kpt_conf, self.r.hand_extend))
        for name in ("left", "right"):
            p = wrists.get(name)
            z = self.store.zone_at(p[0], p[1], "shelf", "cooler") if p else None
            prev = ps.wrist_zone.get(name)
            zname = z.name if z else None
            if zname != prev:
                if prev is not None:
                    for rch in reversed(ps.reaches):
                        if rch.zone == prev and rch.wrist == name and rch.t_end is None:
                            rch.t_end = self.t
                            break
                if zname is not None:
                    ps.reaches.append(Reach(zname, name, self.t))
            ps.wrist_zone[name] = zname
        # Keep reach history short.
        horizon = self.t - max(self.r.reach_window_s, self.r.crowd_window_s) - 5.0
        ps.reaches = [rc for rc in ps.reaches if rc.t_end is None or rc.t_end >= horizon]
        return events

    def _stitch(self, po: PersonObs) -> PersonState | None:
        """Trackers hand out a new id after an occlusion or a missed detection. A new id that appears away
        from the door, soon after and close to where someone was lost inside the store, is that person
        (position and time only, no appearance features). Otherwise the visit splits: the first half never
        exits, so its basket is never reconciled."""
        if self.r.stitch_dist <= 0:
            return None
        fx, fy = po.foot_point(self.r.foot_point)
        if self.store.zone_at(fx, fy, "exit") is not None:
            return None                                      # someone new walking in
        best, best_d = None, math.inf
        for ps in self.people.values():
            if (ps.exited or ps.pid in self._present or ps.t_last >= self.t
                    or self.t - ps.t_last > self.r.stitch_s):
                continue
            lx, ly = ps.last.foot_point(self.r.foot_point)
            h = max(ps.last.bbox[3] - ps.last.bbox[1], 1.0)
            d = math.hypot(fx - lx, fy - ly)
            if d <= self.r.stitch_dist * h and d < best_d:
                best, best_d = ps, d
        if best is not None:
            self.alias[po.track_id] = best.pid
            self.log.append(f"{self.t:.1f}s track {po.track_id} stitched to person {best.pid} "
                            f"({self.t - best.t_last:.1f}s, {best_d:.0f}px after losing them)")
        return best

    def _close_register(self, ps: PersonState) -> list[Event]:
        events = []
        if ps.register_open:
            events.append(self._ev(EventType.PAY, ps.pid, t=ps.register_last_in, zone="register",
                                   meta={"phase": "end", "t_start": ps.register_since,
                                         "t_end": ps.register_last_in, "source": "register_dwell"}))
            reg = self.store.zones_of("register")
            if reg:
                events[-1].zone = reg[0].name
        ps.register_since = ps.register_last_in = None
        ps.register_open = False
        return events

    def _check_gone(self, seen: set[int]) -> list[Event]:
        events = []
        for ps in self.people.values():
            if ps.done or ps.pid in seen:
                continue
            long_enough = ps.t_last - ps.t_first >= self.r.min_store_time_s
            at_door = ps.in_exit_zone and ps.been_inside and long_enough
            if self.t - ps.t_last < (self.r.exit_confirm_s if at_door else self.r.person_lost_s):
                continue
            ps.done = True
            events += self._close_register(ps)
            # Concealments that were still being confirmed when they walked out: the item never
            # reappeared, so they stand.
            for h in ps.conceal_pending:
                events.append(self._conceal_event(ps, h))
            ps.conceal_pending = []
            if at_door:
                ps.exited = True
                held = [h.category for h in ps.held.values() if h.confirmed and not h.released]
                events.append(self._ev(EventType.EXIT, ps.pid, t=ps.t_last,
                                       zone=self._exit_zone_name(),
                                       meta={"held_items": held, "t_detected": self.t}))
            else:
                self.log.append(f"{self.t:.1f}s person {ps.pid} track lost inside store (no exit)")
            for pid in list(ps.held):
                self.owner.pop(pid, None)
        return events

    def _exit_zone_name(self) -> str | None:
        z = self.store.zones_of("exit")
        return z[0].name if z else None

    # -------------------------------------------------------------- products

    def _nearest_wrist(self, c: tuple[float, float]) -> tuple[PersonState | None, str | None, float]:
        best, best_w, best_d = None, None, math.inf
        for ps in self.people.values():
            if ps.done or ps.t_last != self.t:
                continue
            for name, (wx, wy) in ps.last.wrists(self.r.kpt_conf):
                d = math.hypot(c[0] - wx, c[1] - wy)
                if d < best_d:
                    best, best_w, best_d = ps, name, d
        return best, best_w, best_d

    def _owner_near(self, owner: PersonState, c: tuple[float, float]) -> bool | None:
        """Is the product at c in the owner's hand? None = can't tell (no wrist visible)."""
        if owner.t_last != self.t:
            return None
        wrists = owner.last.wrists(self.r.kpt_conf)
        if not wrists:
            x1, y1, x2, y2 = owner.last.bbox
            pad = self.r.hold_radius_px
            return True if (x1 - pad <= c[0] <= x2 + pad and y1 - pad <= c[1] <= y2 + pad) else None
        return any(math.hypot(c[0] - wx, c[1] - wy) <= self.r.hold_radius_px for _, (wx, wy) in wrists)

    def _update_products(self, products: list[ProductObs]) -> list[Event]:
        events = []
        for pr in products:
            c = pr.center
            self.prod_seen[pr.track_id] = self.t
            self.prod_first.setdefault(pr.track_id, c)
            self.prod_center[pr.track_id] = c
            for z in self.store.zones_of("shelf", "cooler"):
                if z.distance(*c) <= self.r.zone_pad_px:
                    self.prod_in_zone.setdefault(pr.track_id, {})[z.name] = self.t
            ps, wrist, d = self._nearest_wrist(c)
            near = ps is not None and d <= self.r.hold_radius_px
            owner_id = self.owner.get(pr.track_id)

            if owner_id is not None:
                owner = self.people[owner_id]
                h = owner.held[pr.track_id]
                owner_near = self._owner_near(owner, c)
                if owner_near is None:
                    # Owner's hands not visible (occluded): keep it theirs, don't start a release.
                    h.t_near = self.t
                    continue
                if owner_near:
                    h.handoff.clear()
                    self._touch(owner, h, pr)
                elif near and ps.pid != owner_id and h.confirmed:
                    # Possibly handed to someone else: only after several frames near their hand
                    # with the owner's hands visibly elsewhere.
                    h.handoff[ps.pid] = h.handoff.get(ps.pid, 0) + 1
                    if h.handoff[ps.pid] >= self.r.handoff_frames:
                        self.log.append(f"{self.t:.1f}s item {pr.track_id} handed {owner_id}->{ps.pid}")
                        del owner.held[pr.track_id]
                        self.owner[pr.track_id] = ps.pid
                        ps.held[pr.track_id] = h
                        h.handoff.clear()
                        self._touch(ps, h, pr)
                continue

            # Not held yet. Stock sitting on a shelf is ignored until it leaves the zone in a hand.
            if not near or self.store.zone_at(c[0], c[1], "shelf", "cooler") is not None:
                continue
            h = HeldItem(pr.track_id, pr.category, self.t, wrist=wrist)
            self.owner[pr.track_id] = ps.pid
            ps.held[pr.track_id] = h
            self._touch(ps, h, pr, count=False)
        # Confirm pending holds.
        for ps in self.people.values():
            for h in list(ps.held.values()):
                if not h.confirmed and h.frames >= self.r.hold_min_frames:
                    events += self._confirm(ps, h)
        return events

    def _touch(self, ps: PersonState, h: HeldItem, pr: ProductObs, count: bool = True) -> None:
        if count:
            h.frames += 1
        h.confs.append(pr.conf)
        h.t_near = self.t
        c = pr.center
        z = self.store.zone_at(c[0], c[1], "shelf", "cooler")
        if z is None:
            # The HOLDING hand inside a merch zone counts too (item held right at the shelf edge). Only that hand:
            # a resting hand next to a gondola is often inside its polygon, and must not turn a concealment by
            # the other hand into a put-back.
            ws = ps.last.wrists(self.r.kpt_conf)
            holder = min(ws, key=lambda w: math.hypot(w[1][0] - c[0], w[1][1] - c[1]))[0] if ws else h.wrist
            h.near_zone = ps.wrist_zone.get(holder) if holder else None
        else:
            h.near_zone = z.name
        x1, y1, x2, y2 = ps.last.torso_box(self.r.torso_margin, self.r.kpt_conf)
        h.near_torso = x1 <= c[0] <= x2 and y1 <= c[1] <= y2
        fx, fy = ps.last.foot_point(self.r.foot_point)
        h.at_register = self.store.zone_at(fx, fy, "register") is not None

    def _confirm(self, ps: PersonState, h: HeldItem) -> list[Event]:
        h.confirmed = True
        # Coming back from an occlusion? (a) a track this person already picked, or (b) a
        # same-category item whose concealment we were still confirming, or (c) a same-category
        # item recently lost from their hand. Then it's the same item, not a new pick.
        if h.product_id in ps.picked_tracks:
            h.picked = True
            self.log.append(f"{self.t:.1f}s person {ps.pid} {h.category} (track {h.product_id}) re-acquired")
            return []
        for other in ps.conceal_pending:
            if other.category == h.category:
                ps.conceal_pending.remove(other)
                h.picked = other.picked
                self.log.append(f"{self.t:.1f}s person {ps.pid} {h.category} reappeared: concealment cancelled")
                return []
        for other in list(ps.held.values()):
            if (other is not h and other.confirmed and not other.released
                    and other.category == h.category and self.t - other.t_near <= self.r.relink_s
                    and other.t_near <= h.t_first):
                del ps.held[other.product_id]
                self.owner.pop(other.product_id, None)
                h.picked = other.picked
                self.log.append(f"{self.t:.1f}s person {ps.pid} {h.category} re-linked after occlusion")
                return []
        reach = self._recent_reach(ps, h.t_first, h.wrist)
        if reach is None:
            self.log.append(f"{self.t:.1f}s person {ps.pid} holding {h.category} with no reach (brought in / re-acquired)")
            return []
        # The product itself must have come from that shelf: seen in (or right at) the zone
        # shortly before, and actually moved (a hand brushing past a still item isn't a pick).
        seen_in_zone = self.prod_in_zone.get(h.product_id, {}).get(reach.zone)
        first = self.prod_first.get(h.product_id)
        cur = self.prod_center.get(h.product_id, first)
        moved = first is not None and math.dist(first, cur) >= self.r.min_pick_move_px
        if seen_in_zone is None or h.t_first - seen_in_zone > self.r.reach_window_s + 1.0 or not moved:
            self.log.append(f"{self.t:.1f}s person {ps.pid} holding {h.category}: not seen leaving {reach.zone} "
                            f"(in zone: {seen_in_zone is not None}, moved: {moved}) -> no pick")
            return []
        h.picked = True
        ps.picked_tracks.add(h.product_id)
        others = self._crowd(ps, reach)
        conf = sum(h.confs) / len(h.confs)
        return [self._ev(EventType.PICK, ps.pid, t=h.t_first, item=h.category, zone=reach.zone,
                         confidence=round(conf, 3),
                         candidates=[ps.pid] + others if others else [],
                         meta={"product_track": h.product_id, "t_confirmed": self.t})]

    def _recent_reach(self, ps: PersonState, t: float, wrist: str | None = None) -> Reach | None:
        """Latest reach (by the holding hand, if known) that ended within reach_window_s of t."""
        reaches = [rc for rc in ps.reaches if rc.wrist == wrist] or ps.reaches
        best = None
        for rc in reaches:
            end = rc.t_end if rc.t_end is not None else t
            if rc.t_start <= t and t - end <= self.r.reach_window_s:
                if best is None or end > (best.t_end or t):
                    best = rc
        return best

    def _crowd(self, ps: PersonState, reach: Reach) -> list[int]:
        w = self.r.crowd_window_s
        r_end = reach.t_end if reach.t_end is not None else self.t
        out = []
        for q in self.people.values():
            if q.pid == ps.pid or q.done:
                continue
            for rc in q.reaches:
                q_end = rc.t_end if rc.t_end is not None else self.t
                if rc.zone == reach.zone and rc.t_start <= r_end + w and q_end >= reach.t_start - w:
                    out.append(q.pid)
                    break
        return out

    def _check_releases(self) -> list[Event]:
        events = []
        for ps in self.people.values():
            for h in list(ps.held.values()):
                # Released = we kept seeing the person for release_s after the item left their
                # hand. If the person vanished too (walked out, occluded), the item is still theirs.
                if ps.t_last - h.t_near < self.r.release_s:
                    continue
                if not h.confirmed:           # never really held
                    del ps.held[h.product_id]
                    self.owner.pop(h.product_id, None)
                    continue
                if h.released or ps.done:
                    continue
                h.released = True
                del ps.held[h.product_id]
                self.owner.pop(h.product_id, None)
                still_visible = self.prod_seen.get(h.product_id, -1) > h.t_near
                if h.near_zone is not None:
                    events.append(self._ev(EventType.PUT_BACK, ps.pid, t=h.t_near, item=h.category,
                                           zone=h.near_zone, meta={"product_track": h.product_id}))
                elif still_visible:
                    self.log.append(f"{h.t_near:.1f}s person {ps.pid} set down {h.category} (still visible)")
                elif h.near_torso and not h.at_register and h.picked:
                    # Vanished at the torso: CONCEAL once it has stayed gone for conceal_confirm_s.
                    ps.conceal_pending.append(h)
                elif h.at_register:
                    self.log.append(f"{h.t_near:.1f}s person {ps.pid} set down {h.category} at register")
                else:
                    self.log.append(f"{h.t_near:.1f}s person {ps.pid} lost sight of {h.category} (no event)")
        return events

    def _conceal_event(self, ps: PersonState, h: HeldItem) -> Event:
        conf = sum(h.confs) / len(h.confs)
        return self._ev(EventType.CONCEAL, ps.pid, t=h.t_near, item=h.category, confidence=round(0.8 * conf, 3),
                        meta={"product_track": h.product_id})

    def _check_conceals(self) -> list[Event]:
        events = []
        for ps in self.people.values():
            keep = []
            for h in ps.conceal_pending:
                if self.prod_seen.get(h.product_id, -1) > h.t_near:
                    self.log.append(f"{self.t:.1f}s person {ps.pid} {h.category} visible again: not concealed")
                elif self.t - h.t_near >= self.r.conceal_confirm_s:
                    events.append(self._conceal_event(ps, h))
                else:
                    keep.append(h)
            ps.conceal_pending = keep
        return events
