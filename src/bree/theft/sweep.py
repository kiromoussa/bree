"""Bulk-grab / sweep detection at shelves and coolers, from pose alone (no product detector needed).

A normal shopper reaches in, takes one or two things and leaves. Someone clearing a shelf, or loading a bag with
whatever is within reach, moves differently: the arm travels along the shelf in one fast stroke, or the hand goes in and out
of the same fixture again and again. Both are visible in the wrist track even when no product is recognised, which is the
case on this store's cameras (COCO classes do not see snacks).

Per tracked person and per shelf / cooler zone (hand = wrist pushed 0.5 forearm past the wrist, like the event engine):
  sweep    inside a `window_s` window the hand covers at least `sweep_len` body heights inside the zone at at least
           `sweep_speed` body heights per second, mostly in one direction (net / path >= `straightness`): a swipe
  repeat   `repeat_n` separate reach-ins into the same zone within `repeat_window_s` (a reach-in = the hand enters the
           zone after being clearly out of it, `out_bh` body heights away)
  both     the two together, or a sweep with both hands in the zone at once

Each hit is a `SweepEvent` with a score in [0, 1]; the same person and zone cannot fire again for `cooldown_s`. This is a
*review* cue ("someone took a lot from this fixture quickly"), never proof: a person buying four drinks at the cooler
reaches in four times. Lengths are in body heights (person box height) so they do not depend on camera distance. Keypoints
are the weak point on real footage (far people have wrist confidence under 0.3): `stats` counts how many person-frames had
a usable hand.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from bree.events.observations import FrameObs, PersonObs
from bree.events.zones import StoreConfig


@dataclass
class SweepConfig:
    kpt_conf: float = 0.5            # wrist keypoints below this are noise on this footage (trails jump across the frame)
    hand_extend: float = 0.5
    zone_pad_px: float = 25.0
    window_s: float = 2.0
    sweep_len: float = 1.0           # body heights of hand path inside the zone within the window
    sweep_speed: float = 1.0         # body heights per second
    straightness: float = 0.55
    out_bh: float = 0.3              # a hand must get this many body heights away from the zone before the next entry is a new
    #                                  reach-in (wrist keypoints jitter: a hand resting on the edge of a counter is one reach, not twenty)
    repeat_n: int = 5
    repeat_window_s: float = 20.0
    cooldown_s: float = 20.0
    min_body_px: float = 60.0        # people smaller than this (box height) are too small for wrist tracking
    max_body_move: float = 0.6       # the person's feet may move at most this many body heights in the window: someone
    #                                  walking past a fixture with a swinging arm is not grabbing from it
    max_jump_bh: float = 0.6         # a wrist that moves more than this between two samples is a keypoint glitch: ignored
    repeat_at_register: bool = False  # repeated reach-ins by someone standing in a register zone are a customer paying and
    #                                  pointing at the counter display (first real-footage false item, REAL-UNLABELLED): off by
    #                                  default. A fast straight sweep still fires there.


@dataclass
class SweepEvent:
    person_id: int
    zone: str
    t: float
    t_start: float
    kind: str                        # sweep | repeat | both
    score: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"person_id": self.person_id, "zone": self.zone, "t": round(self.t, 2), "t_start": round(self.t_start, 2),
                "kind": self.kind, "score": round(self.score, 3), "detail": self.detail}


@dataclass
class _Hand:
    pts: deque = field(default_factory=deque)        # (t, x, y, zone or None)
    armed: dict = field(default_factory=dict)        # zone -> True once the hand has been clearly outside since the last entry
    last_in: dict = field(default_factory=dict)      # zone -> last time inside
    entries: dict = field(default_factory=dict)      # zone -> deque of entry times


class SweepDetector:
    def __init__(self, store: StoreConfig, cfg: SweepConfig | None = None):
        self.cfg = cfg or SweepConfig()
        self.zones = store.zones_of("shelf", "cooler")
        self.registers = store.zones_of("register")
        self.hands: dict[tuple[int, str], _Hand] = {}
        self.body: dict[int, deque] = {}                 # track id -> (t, x, y) of the box foot point
        self.cool: dict[tuple[int, str], float] = {}
        self.stats = {"person_frames": 0, "with_hand": 0, "events": 0, "repeat_skipped_at_register": 0}

    def _zone_of(self, x: float, y: float):
        best, bd = None, self.cfg.zone_pad_px
        for z in self.zones:
            d = z.distance(x, y)
            if d <= bd:
                best, bd = z, d
        return best

    def update(self, obs: FrameObs) -> list[SweepEvent]:
        c = self.cfg
        out = []
        for p in obs.persons:
            self.stats["person_frames"] += 1
            h = p.bbox[3] - p.bbox[1]
            if h < c.min_body_px:
                continue
            bd = self.body.setdefault(p.track_id, deque())
            bd.append((obs.t, *p.foot_point()))
            while bd and obs.t - bd[0][0] > c.window_s:
                bd.popleft()
            hands = p.hands(c.kpt_conf, c.hand_extend)
            if hands:
                self.stats["with_hand"] += 1
            inside_now: dict[str, set] = {}
            for name, (x, y) in hands:
                hd = self.hands.setdefault((p.track_id, name), _Hand())
                if hd.pts and math.hypot(x - hd.pts[-1][1], y - hd.pts[-1][2]) > c.max_jump_bh * h:
                    continue                                  # glitch: keep the last good point
                z = self._zone_of(x, y)
                for zz in self.zones:               # hysteresis: far from a zone re-arms it
                    if zz.distance(x, y) > c.out_bh * h:
                        hd.armed[zz.name] = True
                hd.pts.append((obs.t, x, y, z.name if z else None))
                while hd.pts and obs.t - hd.pts[0][0] > max(c.window_s, c.repeat_window_s):
                    hd.pts.popleft()
                if z is not None:
                    inside_now.setdefault(z.name, set()).add(name)
                    if hd.armed.get(z.name, True):
                        hd.armed[z.name] = False
                        hd.entries.setdefault(z.name, deque()).append(obs.t)
                    hd.last_in[z.name] = obs.t
                    at_reg = any(r.contains(*p.foot_point()) for r in self.registers)
                    ev = self._check(p, name, hd, z.name, h, obs.t, len(inside_now[z.name]), at_reg)
                    if ev:
                        out.append(ev)
        self.stats["events"] += len(out)
        return out

    def _check(self, p: PersonObs, hand: str, hd: _Hand, zone: str, body_h: float, t: float, hands_in: int,
               at_register: bool = False):
        c = self.cfg
        key = (p.track_id, zone)
        if t - self.cool.get(key, -1e9) < c.cooldown_s:
            return None
        bd = self.body.get(p.track_id) or [(t, 0, 0)]
        xy_b = np.array([(x, y) for _, x, y in bd])
        body_move = float(np.linalg.norm(np.diff(xy_b, axis=0), axis=1).sum()) / body_h if len(xy_b) > 1 else 0.0
        moving = body_move > c.max_body_move
        # sweep: path of the hand inside the zone over the last window
        seg = [(tt, x, y) for tt, x, y, zn in hd.pts if zn == zone and t - tt <= c.window_s]
        sweep = None
        if len(seg) >= 3 and not moving:
            xy = np.array([(x, y) for _, x, y in seg])
            path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) / body_h
            net = float(np.linalg.norm(xy[-1] - xy[0])) / body_h
            span = max(seg[-1][0] - seg[0][0], 1e-3)
            if path >= c.sweep_len and path / span >= c.sweep_speed and net / max(path, 1e-6) >= c.straightness:
                sweep = {"path_bh": round(path, 2), "speed_bh_s": round(path / span, 2), "straight": round(net / path, 2)}
        ent = [e for e in hd.entries.get(zone, ()) if t - e <= c.repeat_window_s]
        repeat = {"reach_ins": len(ent), "within_s": round(t - ent[0], 1)} if len(ent) >= c.repeat_n else None
        if repeat and at_register and not c.repeat_at_register:
            repeat = None
            self.stats["repeat_skipped_at_register"] += 1
        if not sweep and not repeat:
            return None
        kind = "both" if sweep and repeat or (sweep and hands_in > 1) else "sweep" if sweep else "repeat"
        score = min(1.0, 0.45 + (0.25 if sweep else 0) + (0.25 if repeat else 0) + 0.1 * (hands_in > 1)
                    + (0.05 * (len(ent) - c.repeat_n) if repeat else 0))
        self.cool[key] = t
        return SweepEvent(p.track_id, zone, t, ent[0] if repeat else seg[0][0], kind, score,
                          {"hand": hand, **(sweep or {}), **(repeat or {})})
