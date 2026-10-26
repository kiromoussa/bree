"""Observation gate between the tracker and the event engine: keep ghosts, staff and fixtures out of the ledger.

What real footage from one store showed (REAL-UNLABELLED, docs/ultra/theft.md): the tracker hands the engine a few dozen
ids for every handful of real people. Low-confidence boxes flicker on product racks and dark displays, staff behind the
counter are tracked like customers, and a card terminal on the counter is a "phone" in 99% of frames. Each new id is an
ENTER and every one of them is a visit the ledger has to reconcile. The gate removes them *before* the engine, causally
(it only looks at the past), so it works the same live and in replay:

  person tracks   hidden until they have lived `min_age_s` and reached `min_conf` once (flicker); hidden for good when
                  they sit in an `ignore` zone, are a static low-confidence box (`static_*`: a ghost on a rack, not a
                  person standing still, who is detected with higher confidence), or stand in a `staff` zone for most of
                  their life (staff are not shoppers: employee theft is a different problem and is not scored here).
  product tracks  dropped when they have not moved (`product_static_s`, `product_static_px`): a fixture, not an item
                  in a hand.

Everything is a counter in `stats`, so a run can say what the gate removed and why. Zones are polygons in the camera's
pixels (see bree.theft.station). A gate with default settings and no zones still applies the age / confidence / static rules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import FrameObs
from bree.events.zones import Zone


@dataclass
class GateConfig:
    enabled: bool = True
    min_age_s: float = 1.0           # a person track is invisible to the engine until it has lived this long...
    min_conf: float = 0.45           # ...and reached this detection confidence at least once
    static_s: float = 6.0            # a track this old whose box centre has stayed within static_px...
    static_px: float = 40.0
    static_max_conf: float = 0.55    # ...and never got past this confidence is a ghost on a fixture
    staff_frac: float = 0.6          # share of a track's frames with its feet in a staff zone that makes it staff
    staff_min_s: float = 1.0         # (judged once the track is at least this old)
    product_static_s: float = 3.0    # a product track this old that never moved more than product_static_px
    product_static_px: float = 12.0  #   is a fixture
    foot_point: str = "bottom"


@dataclass
class _Track:
    t_first: float
    t_last: float
    max_conf: float = 0.0
    frames: int = 0
    staff_frames: int = 0
    ignore_frames: int = 0
    lo: np.ndarray = field(default_factory=lambda: np.array([1e9, 1e9]))
    hi: np.ndarray = field(default_factory=lambda: np.array([-1e9, -1e9]))
    verdict: str | None = None       # None = undecided, "ok" = let through for good, else the reason it is hidden


class ObsGate:
    def __init__(self, cfg: GateConfig | None = None, staff: list[Zone] | None = None, ignore: list[Zone] | None = None):
        self.cfg = cfg or GateConfig()
        self.staff, self.ignore = staff or [], ignore or []
        self.persons: dict[int, _Track] = {}
        self.products: dict[int, _Track] = {}
        self.stats: Counter = Counter()      # why tracks were hidden, plus raw / passed counts

    # ------------------------------------------------------------------ main

    def update(self, obs: FrameObs) -> FrameObs:
        if not self.cfg.enabled:
            return obs
        keep = [p for p in obs.persons if self._person_ok(p, obs.t)]
        keep_q = [q for q in obs.products if self._product_ok(q, obs.t)]
        return FrameObs(obs.frame, obs.t, keep, keep_q)

    def _person_ok(self, p, t: float) -> bool:
        c = self.cfg
        tr = self.persons.get(p.track_id)
        if tr is None:
            tr = self.persons[p.track_id] = _Track(t, t)
            self.stats["tracks_raw"] += 1
        tr.t_last = t
        tr.frames += 1
        tr.max_conf = max(tr.max_conf, p.conf)
        cx, cy = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
        tr.lo, tr.hi = np.minimum(tr.lo, (cx, cy)), np.maximum(tr.hi, (cx, cy))
        fx, fy = p.foot_point(c.foot_point)
        tr.staff_frames += any(z.contains(fx, fy) for z in self.staff)
        tr.ignore_frames += any(z.contains(fx, fy) for z in self.ignore)
        if tr.verdict not in (None, "ok"):
            return False
        age = t - tr.t_first
        if tr.verdict is None:
            if tr.ignore_frames / tr.frames >= 0.5 and age >= 0.3:
                return self._hide(tr, "ignore_zone")
            if age >= c.staff_min_s and tr.staff_frames / tr.frames >= c.staff_frac:
                return self._hide(tr, "staff")
            if age >= c.static_s and float(np.max(tr.hi - tr.lo)) <= c.static_px and tr.max_conf < c.static_max_conf:
                return self._hide(tr, "static_ghost")
            if age >= c.min_age_s and tr.max_conf >= c.min_conf:
                tr.verdict = "ok"
                self.stats["tracks_passed"] += 1
            else:
                return False
        # an established track can still turn out to be a ghost on a fixture, or staff
        if age >= c.static_s and float(np.max(tr.hi - tr.lo)) <= c.static_px and tr.max_conf < c.static_max_conf:
            return self._hide(tr, "static_ghost", was_passed=True)
        # (staff:) (stood in the shop first, then walked behind the counter)
        if age >= c.staff_min_s * 5 and tr.staff_frames / tr.frames >= max(c.staff_frac, 0.8):
            return self._hide(tr, "staff", was_passed=True)
        return True

    def _hide(self, tr: _Track, why: str, was_passed: bool = False) -> bool:
        tr.verdict = why
        self.stats["hidden_" + why] += 1
        if was_passed:
            self.stats["tracks_passed"] -= 1
        return False

    def _product_ok(self, q, t: float) -> bool:
        c = self.cfg
        tr = self.products.get(q.track_id)
        cx, cy = q.center
        if tr is None:
            tr = self.products[q.track_id] = _Track(t, t)
            self.stats["products_raw"] += 1
        tr.t_last = t
        tr.lo, tr.hi = np.minimum(tr.lo, (cx, cy)), np.maximum(tr.hi, (cx, cy))
        if tr.verdict is not None:
            return tr.verdict == "ok"
        if t - tr.t_first >= c.product_static_s:
            if float(np.max(tr.hi - tr.lo)) <= c.product_static_px:
                tr.verdict = "static"
                self.stats["products_static"] += 1
                return False
            tr.verdict = "ok"
        return True

    def status(self, track_id: int) -> str:
        """"pass", "pending" or the reason a person track is hidden (debug overlays)."""
        tr = self.persons.get(track_id)
        return "pending" if tr is None or tr.verdict is None else "pass" if tr.verdict == "ok" else tr.verdict
