"""Event engine: per-frame observations -> store events, one test per event type."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bree.detect.toy import synth_keypoints
from bree.events.engine import EventEngine
from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.events.types import EventType as E
from bree.events.zones import load_store_config

ROOT = Path(__file__).resolve().parents[1]
FPS = 15.0


@pytest.fixture
def store():
    return load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")


class Sim:
    """Tiny scripted world: one or more people (foot point + hands) and products."""

    def __init__(self, store):
        self.engine = EventEngine(store)
        self.f = 0
        self.events = []

    def step(self, people: dict[int, tuple], products: list[tuple] = (), n: int = 1):
        """people: pid -> (foot_x, foot_y, [hand (x, y), ...]); products: (pid, x, y, category)."""
        for _ in range(n):
            persons = []
            for pid, (fx, fy, hands) in people.items():
                box = (fx - 22, fy - 158, fx + 22, fy)
                persons.append(PersonObs(pid, box, 0.9, synth_keypoints(box, hands)))
            prods = [ProductObs(tid, (x - 7, y - 9, x + 7, y + 9), cat, 0.85) for tid, x, y, cat in products]
            self.events += self.engine.update(FrameObs(self.f, self.f / FPS, persons, prods))
            self.f += 1
        return self

    def of(self, type_):
        return [e for e in self.events if e.type == type_]


def rest(fx, fy):
    return [(fx - 30, fy - 62), (fx + 30, fy - 62)]


def test_enter_and_exit(store):
    s = Sim(store)
    s.step({1: (85, 690, rest(85, 690))}, n=5)             # appears at the door
    for x in range(85, 600, 20):                            # walks in
        s.step({1: (x, 575, rest(x, 575))})
    s.step({1: (600, 575, rest(600, 575))}, n=30)
    for x in range(600, 60, -20):                           # walks back out
        s.step({1: (x, 640, rest(x, 640))})
    s.step({}, n=30)                                        # gone
    assert len(s.of(E.ENTER)) == 1 and len(s.of(E.EXIT)) == 1
    assert s.of(E.EXIT)[0].zone == "door"


def test_track_lost_inside_store_is_not_an_exit(store):
    s = Sim(store)
    s.step({1: (600, 575, rest(600, 575))}, n=45).step({}, n=40)
    assert s.of(E.EXIT) == []
    assert any("lost inside store" in m for m in s.engine.log)


def _pick_sequence(s, pid=1, fx=420, fy=300, item=(7, 450, 95, "soda_bottle")):
    """Stand at the cooler, reach up with the right hand, come back holding the item."""
    tid, ix, iy, cat = item
    hand_rest = (fx + 30, fy - 62)
    s.step({pid: (fx, fy, rest(fx, fy))}, [(tid, ix, iy, cat)], n=5)
    path = [(hand_rest[0] + (ix - hand_rest[0]) * u, hand_rest[1] + (iy - hand_rest[1]) * u)
            for u in np.linspace(0, 1, 8)]
    for hx, hy in path:                                     # hand out, item still on the shelf
        s.step({pid: (fx, fy, [(fx - 30, fy - 62), (hx, hy)])}, [(tid, ix, iy, cat)])
    for hx, hy in reversed(path):                           # hand back, item in hand
        s.step({pid: (fx, fy, [(fx - 30, fy - 62), (hx, hy)])}, [(tid, hx, hy + 12, cat)])
    s.step({pid: (fx, fy, rest(fx, fy))}, [(tid, hand_rest[0], hand_rest[1] + 12, cat)], n=10)
    return hand_rest


def test_pick_from_cooler(store):
    s = Sim(store)
    _pick_sequence(s)
    picks = s.of(E.PICK)
    assert len(picks) == 1
    assert picks[0].zone == "cooler_bank" and picks[0].item == "soda_bottle" and picks[0].person_id == 1
    assert picks[0].candidates == []


def test_holding_own_item_without_reach_is_not_a_pick(store):
    s = Sim(store)
    s.step({1: (600, 575, rest(600, 575))}, [(3, 630, 525, "soda_bottle")], n=30)
    assert s.of(E.PICK) == []


def test_put_back(store):
    s = Sim(store)
    hand_rest = _pick_sequence(s)
    fx, fy, (ix, iy) = 420, 300, (450, 95)
    for u in np.linspace(0, 1, 8):                          # carry it back up to the cooler
        hx, hy = hand_rest[0] + (ix - hand_rest[0]) * u, hand_rest[1] + (iy - hand_rest[1]) * u
        s.step({1: (fx, fy, [(fx - 30, fy - 62), (hx, hy)])}, [(7, hx, hy + 12, "soda_bottle")])
    for u in np.linspace(1, 0, 8):                          # hand comes back empty, item stays
        hx, hy = hand_rest[0] + (ix - hand_rest[0]) * u, hand_rest[1] + (iy - hand_rest[1]) * u
        s.step({1: (fx, fy, [(fx - 30, fy - 62), (hx, hy)])}, [(7, ix, iy + 12, "soda_bottle")])
    s.step({1: (fx, fy, rest(fx, fy))}, [(7, ix, iy + 12, "soda_bottle")], n=15)
    assert len(s.of(E.PUT_BACK)) == 1
    assert s.of(E.PUT_BACK)[0].zone == "cooler_bank"


def test_conceal_near_torso(store):
    s = Sim(store)
    _pick_sequence(s)
    fx, fy = 420, 300
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [(7, fx + 6, fy - 53, "soda_bottle")], n=5)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [], n=15)   # vanished at the waistband
    assert len(s.of(E.CONCEAL)) == 1
    assert s.of(E.CONCEAL)[0].item == "soda_bottle"


def test_item_lost_away_from_torso_is_not_conceal(store):
    s = Sim(store)
    hand_rest = _pick_sequence(s)
    fx, fy = 420, 300
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 70, fy - 140)])}, [(7, fx + 70, fy - 128, "soda_bottle")], n=5)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 70, fy - 140)])}, [], n=15)
    assert s.of(E.CONCEAL) == []


def test_no_conceal_at_register(store):
    s = Sim(store)
    fx, fy = 1120, 610
    s.step({1: (fx, fy, rest(fx, fy))}, n=10)
    s.engine.people[1].held.clear()
    # item appears in hand at the register with no reach -> not a pick; then vanishes at the torso
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [(9, fx + 6, fy - 53, "candy_bar")], n=8)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [], n=15)
    assert s.of(E.CONCEAL) == []


def test_register_dwell_emits_pay_start_and_end(store):
    s = Sim(store)
    s.step({1: (1120, 610, rest(1120, 610))}, n=int(5 * FPS))
    s.step({1: (700, 575, rest(700, 575))}, n=20)
    pays = s.of(E.PAY)
    assert [p.meta["phase"] for p in pays] == ["start", "end"]
    assert pays[1].meta["t_end"] - pays[1].meta["t_start"] == pytest.approx(5 - 1 / FPS, abs=0.1)


def test_short_register_pass_is_not_a_visit(store):
    s = Sim(store)
    s.step({1: (1120, 610, rest(1120, 610))}, n=int(1.0 * FPS))
    s.step({1: (700, 575, rest(700, 575))}, n=20)
    assert s.of(E.PAY) == []


def test_crowded_pick_lists_both_candidates(store):
    s = Sim(store)
    fa, fb, fy = 420, 564, 300
    ia, ib = (450, 95), (534, 95)
    ra, rb = (fa + 30, fy - 62), (fb - 30, fy - 62)
    prods = [(1, *ia, "soda_bottle"), (2, *ib, "soda_bottle")]
    s.step({1: (fa, fy, rest(fa, fy)), 2: (fb, fy, rest(fb, fy))}, prods, n=5)
    lerp = lambda a, b, u: (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
    for u in np.linspace(0, 1, 8):
        ha, hb = lerp(ra, ia, u), lerp(rb, ib, u)
        s.step({1: (fa, fy, [(fa - 30, fy - 62), ha]), 2: (fb, fy, [hb, (fb + 30, fy - 62)])}, prods)
    for u in np.linspace(1, 0, 8):
        ha, hb = lerp(ra, ia, u), lerp(rb, ib, u)
        s.step({1: (fa, fy, [(fa - 30, fy - 62), ha]), 2: (fb, fy, [hb, (fb + 30, fy - 62)])},
               [(1, ha[0], ha[1] + 12, "soda_bottle"), (2, hb[0], hb[1] + 12, "soda_bottle")])
    s.step({1: (fa, fy, rest(fa, fy)), 2: (fb, fy, rest(fb, fy))},
           [(1, ra[0], ra[1] + 12, "soda_bottle"), (2, rb[0], rb[1] + 12, "soda_bottle")], n=10)
    picks = s.of(E.PICK)
    assert sorted(p.person_id for p in picks) == [1, 2]
    assert all(sorted(p.candidates) == [1, 2] for p in picks)


def test_exit_reports_items_in_hand(store):
    s = Sim(store)
    hand_rest = _pick_sequence(s)
    for x in range(420, 60, -25):
        s.step({1: (x, 640, rest(x, 640))}, [(7, x + 30, 640 - 50, "soda_bottle")])
    s.step({}, n=30)
    ex = s.of(E.EXIT)
    assert len(ex) == 1 and ex[0].meta["held_items"] == ["soda_bottle"]
