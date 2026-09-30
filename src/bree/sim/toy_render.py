"""Rendered 2D toy clips with ground truth. TOY DATA: it proves the pipeline runs
end to end (pixels -> detections -> tracks -> events -> ledger -> alerts); it
does not measure real-world accuracy.

The scene uses the zone layout of configs/store_gas_station_small.yaml.
People are coloured blobs with orange hands; products are coloured squares
(palette in bree/detect/toy.py). Each scenario is a small script:

    a = Actor("A", color=0, t0=0.0, start=DOOR)
    a.walk(COOLER_SPOT).reach("right", item="s1").walk(REGISTER_SPOT).wait(6).leave()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bree.detect.toy import HAND_COLOR, PERSON_COLORS, PRODUCT_COLORS

W, H, FPS = 1280, 720, 15
BODY_W, BODY_H, HEAD_R = 44, 130, 14
HAND_R = 7
SPEED = 170.0           # px / s walking

DOOR = (85, 690)
DOOR_OUT = (60, 640)
LANE_Y = 575
REGISTER_SPOT = (1120, 610)
REGISTER_SPOT_2 = (1215, 640)   # second in the queue

BG, SHELF, COOLER, COUNTER, DOORC = (215, 215, 215), (150, 160, 170), (235, 222, 205), (140, 140, 140), (200, 200, 235)


@dataclass
class Item:
    id: str
    category: str
    pos: tuple[float, float]
    sku: str


@dataclass
class _Seg:
    t0: float
    t1: float
    kind: str                     # walk | wait | reach | conceal
    a: tuple[float, float]        # foot start
    b: tuple[float, float]        # foot end
    hand: str | None = None
    target: tuple[float, float] | None = None


class Actor:
    def __init__(self, name: str, color: int, t0: float, start=DOOR):
        self.name, self.color, self.t_start = name, color, t0
        self.t = t0
        self.foot = start
        self.segs: list[_Seg] = []
        self.item_events: list[tuple[float, str, str, str | None]] = []   # (t, item_id, action, hand)
        self.t_exit: float | None = None
        self.picked: list[str] = []
        self.paid: list[str] = []
        self.concealed: list[str] = []
        self.put_back: list[str] = []
        self.visible_hold: dict[str, str] = {}

    # --- script verbs -------------------------------------------------------
    def walk(self, to, via_lane: bool = False):
        pts = [to]
        if via_lane:
            pts = [(self.foot[0], LANE_Y), (to[0], LANE_Y), to]
        for p in pts:
            d = float(np.hypot(p[0] - self.foot[0], p[1] - self.foot[1]))
            dur = max(d / SPEED, 1 / FPS)
            self.segs.append(_Seg(self.t, self.t + dur, "walk", self.foot, p))
            self.t += dur
            self.foot = p
        return self

    def wait(self, dur: float):
        self.segs.append(_Seg(self.t, self.t + dur, "wait", self.foot, self.foot))
        self.t += dur
        return self

    def reach(self, hand: str, target, take: str | None = None, put: str | None = None, dur: float = 1.6):
        """Hand goes out to `target` and back. Takes / puts an item at the far point."""
        self.segs.append(_Seg(self.t, self.t + dur, "reach", self.foot, self.foot, hand, target))
        mid = self.t + dur / 2
        if take:
            self.item_events.append((mid, take, "attach", hand))
            self.picked.append(take)
        if put:
            self.item_events.append((mid, put, "place", None))
            self.put_back.append(put)
        self.t += dur
        return self

    def conceal(self, item: str, hand: str, dur: float = 1.2):
        """Hand moves to the waistband / pocket; item disappears there."""
        self.segs.append(_Seg(self.t, self.t + dur, "conceal", self.foot, self.foot, hand))
        self.item_events.append((self.t + dur * 0.55, item, "hide", None))
        self.concealed.append(item)
        self.t += dur
        return self

    def leave(self):
        self.walk((self.foot[0], LANE_Y)) if self.foot[1] < LANE_Y - 5 else None
        self.walk((DOOR_OUT[0] + 40, LANE_Y))
        self.walk(DOOR_OUT)
        self.t_exit = self.t
        return self

    # --- state ---------------------------------------------------------------
    def state(self, t: float):
        """(foot, {hand: pos}) at time t, or None if not in the scene."""
        if t < self.t_start or (self.t_exit is not None and t > self.t_exit):
            return None
        seg = next((s for s in self.segs if s.t0 <= t <= s.t1), None)
        if seg is None:
            foot = self.foot if t > self.t else self.segs[0].a
            return foot, self._rest(foot)
        u = (t - seg.t0) / max(seg.t1 - seg.t0, 1e-6)
        foot = (seg.a[0] + (seg.b[0] - seg.a[0]) * u, seg.a[1] + (seg.b[1] - seg.a[1]) * u)
        hands = self._rest(foot)
        if seg.kind in ("reach", "conceal"):
            rest = hands[seg.hand]
            if seg.kind == "reach":
                goal = seg.target
            else:  # pocket / waistband: body centre at hip height
                goal = (foot[0] + (-6 if seg.hand == "left" else 6), foot[1] - BODY_H * 0.5)
            k = 1 - abs(2 * u - 1)          # out and back
            hands[seg.hand] = (rest[0] + (goal[0] - rest[0]) * k, rest[1] + (goal[1] - rest[1]) * k)
        return foot, hands

    @staticmethod
    def _rest(foot):
        y = foot[1] - 62
        return {"left": (foot[0] - 30, y), "right": (foot[0] + 30, y)}


@dataclass
class Scenario:
    name: str
    description: str
    actors: list[Actor]
    items: list[Item]
    payments: list[dict] = field(default_factory=list)   # POS messages (stream time)
    thieves: dict[str, list[str]] = field(default_factory=dict)  # actor -> stolen categories

    @property
    def duration(self) -> float:
        return max(a.t_exit or a.t for a in self.actors) + 3.0


def _stock() -> list[Item]:
    items = []
    cooler = [("soda_bottle", "COKE-20OZ"), ("energy_drink", "REDBULL-12OZ"), ("water_bottle", "WATER-1L"),
              ("soda_bottle", "PEPSI-20OZ"), ("energy_drink", "MONSTER-16OZ")]
    for i, x in enumerate(range(240, 1080, 42)):
        cat, sku = cooler[i % len(cooler)]
        items.append(Item(f"c{i}", cat, (x, 95), sku))
    for i, y in enumerate(range(255, 450, 38)):
        items.append(Item(f"a{i}", "candy_bar", (372, y), "SNICKERS"))
        items.append(Item(f"b{i}", "chips_bag", (540, y), "LAYS-CLASSIC"))
        items.append(Item(f"d{i}", "lighter", (812, y), "BIC-LIGHTER"))
    return items


def _item(items, iid) -> Item:
    return next(i for i in items if i.id == iid)


def _pay(t: float, skus: list[str], txn: str) -> dict:
    return {"t": round(t, 2), "terminal": "pos_1", "txn_id": txn, "items": [{"sku": s, "qty": 1} for s in skus]}


def scenarios() -> list[Scenario]:
    out = []

    def cooler_spot(item: Item, side="right"):
        return (item.pos[0] - 30 if side == "right" else item.pos[0] + 30, 300)

    # 1. Normal: soda from the cooler, pays, leaves with it in hand.
    items = _stock(); s = _item(items, "c0")
    a = Actor("A", 0, 0.5).walk(cooler_spot(s), via_lane=True).reach("right", s.pos, take="c0")
    a.walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [s.sku], "T1")]
    a.leave()
    out.append(Scenario("normal_pay", "picks a soda, pays at the register, walks out with it", [a], items, pays))

    # 2. Walkout: energy drink straight out the door.
    items = _stock(); e = _item(items, "c1")
    a = Actor("A", 1, 0.5).walk(cooler_spot(e), via_lane=True).reach("right", e.pos, take="c1").leave()
    out.append(Scenario("walkout", "picks an energy drink and walks out without paying", [a], items,
                        thieves={"A": ["energy_drink"]}))

    # 3. Conceal + partial pay: pays for the soda, pockets the candy bar.
    items = _stock(); s = _item(items, "c3"); c = _item(items, "a2")
    a = Actor("A", 2, 0.5).walk(cooler_spot(s), via_lane=True).reach("right", s.pos, take="c3")
    a.walk((455, 330), via_lane=True).reach("left", c.pos, take="a2").wait(0.6).conceal("a2", "left")
    a.walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [s.sku], "T1")]
    a.leave()
    out.append(Scenario("conceal_partial_pay", "pays for a soda, pockets a candy bar", [a], items, pays,
                        thieves={"A": ["candy_bar"]}))

    # 4. Put-back: takes chips, changes mind, puts them back, buys candy instead.
    items = _stock(); ch = _item(items, "b1"); c = _item(items, "a3")
    a = Actor("A", 3, 0.5).walk((455, 300), via_lane=True).reach("right", ch.pos, take="b1").wait(1.5)
    a.reach("right", ch.pos, put="b1").wait(0.5).walk((455, 370)).reach("left", c.pos, take="a3")
    a.walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [c.sku], "T1")]
    a.leave()
    out.append(Scenario("put_back", "takes chips, puts them back, buys a candy bar", [a], items, pays))

    # 5. Crowded cooler: two people reach into the cooler at the same moment; both pay.
    items = _stock(); s1 = _item(items, "c5"); s2 = _item(items, "c7")
    # B comes in 6 s after A (strangers, not a group); both reach at the same instant.
    a = Actor("A", 0, 0.5).walk(cooler_spot(s1), via_lane=True)
    b = Actor("B", 1, 6.5).walk(cooler_spot(s2, "left"), via_lane=True)
    t_reach = max(a.t, b.t) + 0.5
    a.wait(t_reach - a.t).reach("right", s1.pos, take="c5")
    b.wait(t_reach - b.t).reach("left", s2.pos, take="c7")
    a.walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [s1.sku], "T1")]
    a.leave()
    b.wait(max(0.0, a.t_exit - 3.0 - b.t)).walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays.append(_pay(b.t - 2.0, [s2.sku], "T2"))
    b.leave()
    out.append(Scenario("crowd_cooler", "two shoppers reach into the cooler together, both pay", [a, b], items, pays))

    # 6. Lingerer: browses, touches a shelf, takes nothing, leaves.
    items = _stock()
    a = Actor("A", 2, 0.5).walk((725, 330), via_lane=True).wait(2.0).reach("right", (812, 330)).wait(3.0)
    a.walk((995, 400), via_lane=True).wait(4.0).leave()
    out.append(Scenario("linger", "browses and touches a shelf, buys nothing", [a], items))

    # 7. Pocket then pay: puts candy in pocket while shopping, then pays for it.
    items = _stock(); c = _item(items, "a1")
    a = Actor("A", 3, 0.5).walk((455, 300), via_lane=True).reach("left", c.pos, take="a1").wait(0.5)
    a.conceal("a1", "left").walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [c.sku], "T1")]
    a.leave()
    out.append(Scenario("pocket_then_pay", "pockets a candy bar while shopping, then pays for it", [a], items, pays))

    # 8. Group: two friends come in together, each takes a drink, one pays for both.
    items = _stock(); s1 = _item(items, "c10"); s2 = _item(items, "c13")
    a = Actor("A", 0, 0.5).walk(cooler_spot(s1), via_lane=True).reach("right", s1.pos, take="c10")
    b = Actor("B", 1, 2.0).walk(cooler_spot(s2), via_lane=True).reach("right", s2.pos, take="c13")
    a.walk(REGISTER_SPOT, via_lane=True).wait(6.0)
    pays = [_pay(a.t - 2.0, [s1.sku, s2.sku], "T1")]
    b.walk((900, 560)).wait(max(0.0, a.t - b.t))
    a.leave(); b.leave()
    out.append(Scenario("group_one_pays", "two friends, one pays for both drinks", [a, b], items, pays))
    return out


# ------------------------------------------------------------------ rendering

def _background(store) -> np.ndarray:
    img = np.full((H, W, 3), BG, np.uint8)
    fills = {"shelf": SHELF, "cooler": COOLER, "register": BG, "exit": DOORC}
    for z in store.zones:
        cv2.fillPoly(img, [z.polygon.astype(np.int32)], fills[z.kind])
    cv2.rectangle(img, (1000, 400), (1270, 418), COUNTER, -1)       # the counter itself
    return img


def render(sc: Scenario, store, out_dir: Path, noise: float = 3.0, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bg = _background(store)
    video = out_dir / f"toy_{sc.name}.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    events = sorted((ev for a in sc.actors for ev in ((t, iid, act, hand, a) for t, iid, act, hand in a.item_events)),
                    key=lambda e: e[0])
    where = {it.id: ("shelf", it.pos) for it in sc.items}
    n = int(sc.duration * FPS)
    noise_bank = [rng.normal(0, noise, (H, W, 3)).astype(np.int16) for _ in range(8)] if noise else []
    for f in range(n):
        t = f / FPS
        for (te, iid, act, hand, actor) in events:
            if te <= t and (te > t - 1 / FPS):
                if act == "attach":
                    where[iid] = ("hand", (actor, hand))
                elif act == "place":
                    where[iid] = ("shelf", _item(sc.items, iid).pos)
                elif act == "hide":
                    where[iid] = ("hidden", None)
        img = bg.copy()
        states = {a.name: a.state(t) for a in sc.actors}
        for a in sc.actors:
            st = states[a.name]
            if st is None:
                continue
            (fx, fy), _ = st
            col = PERSON_COLORS[a.color]
            cv2.rectangle(img, (int(fx - BODY_W / 2), int(fy - BODY_H)), (int(fx + BODY_W / 2), int(fy)), col, -1)
            cv2.circle(img, (int(fx), int(fy - BODY_H - HEAD_R + 2)), HEAD_R, col, -1)
        for it in sc.items:
            kind, val = where[it.id]
            if kind == "hidden":
                continue
            if kind == "hand":
                actor, hand = val
                st = states[actor.name]
                if st is None:
                    continue
                hx, hy = st[1][hand]
                cx, cy = hx, hy + 12
            else:
                cx, cy = val
            cv2.rectangle(img, (int(cx - 7), int(cy - 9)), (int(cx + 7), int(cy + 9)), PRODUCT_COLORS[it.category], -1)
        for a in sc.actors:
            st = states[a.name]
            if st is None:
                continue
            for hx, hy in st[1].values():
                cv2.circle(img, (int(hx), int(hy)), HAND_R, HAND_COLOR, -1)
        if noise:
            img = np.clip(img.astype(np.int16) + noise_bank[f % len(noise_bank)], 0, 255).astype(np.uint8)
        cv2.putText(img, f"TOY DATA  {sc.name}  t={t:5.1f}s", (W - 420, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
        vw.write(img)
    vw.release()

    (out_dir / f"toy_{sc.name}.payments.jsonl").write_text("".join(json.dumps(p) + "\n" for p in sc.payments))
    cats = {it.id: it.category for it in sc.items}
    truth = {
        "clip": sc.name, "description": sc.description, "toy_data": True, "fps": FPS, "frames": n,
        "people": [{"name": a.name, "t_enter": round(a.t_start, 2), "t_exit": round(a.t_exit, 2),
                    "picked": [cats[i] for i in a.picked], "put_back": [cats[i] for i in a.put_back],
                    "concealed": [cats[i] for i in a.concealed],
                    "thief": a.name in sc.thieves, "stolen": sc.thieves.get(a.name, [])}
                   for a in sc.actors],
    }
    (out_dir / f"toy_{sc.name}.truth.json").write_text(json.dumps(truth, indent=2))
    return {"video": str(video), "truth": truth}


def render_all(store, out_dir: str | Path) -> list[dict]:
    return [render(sc, store, Path(out_dir), seed=i) for i, sc in enumerate(scenarios())]
