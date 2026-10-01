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
    s.step({}, n=60)                                        # gone (exit confirmed after 3 s)
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
    assert s.of(E.CONCEAL) == []                             # not yet: confirming it stays gone
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [], n=15)
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
    s.step({}, n=60)
    ex = s.of(E.EXIT)
    assert len(ex) == 1 and ex[0].meta["held_items"] == ["soda_bottle"]


# ------------------------------------------ regressions from the adversarial review

def _walk_in_holding_own_drink(s, n=20):
    """Customer comes in with their own drink at chest height (never picked here)."""
    fx, fy = 600, 575
    hand = (fx + 6, fy - 100)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), hand])}, [(9, hand[0], hand[1] + 12, "soda_bottle")], n=n)
    return fx, fy, hand


def test_own_drink_detection_dropout_is_not_conceal(store):
    s = Sim(store)
    fx, fy, hand = _walk_in_holding_own_drink(s)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), hand])}, [], n=10)                          # 0.67 s dropout
    s.step({1: (fx, fy, [(fx - 30, fy - 62), hand])}, [(9, hand[0], hand[1] + 12, "soda_bottle")], n=20)
    assert s.of(E.CONCEAL) == [] and s.of(E.PICK) == []


def test_picked_item_dropout_then_reappears_is_not_conceal_or_second_pick(store):
    s = Sim(store)
    _pick_sequence(s)
    fx, fy = 420, 300
    at_chest = (fx + 6, fy - 100)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), at_chest])}, [(7, at_chest[0], at_chest[1] + 12, "soda_bottle")], n=5)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), at_chest])}, [], n=12)                       # 0.8 s unseen
    s.step({1: (fx, fy, [(fx - 30, fy - 62), at_chest])}, [(7, at_chest[0], at_chest[1] + 12, "soda_bottle")], n=30)
    assert s.of(E.CONCEAL) == [] and len(s.of(E.PICK)) == 1


def test_set_down_but_still_visible_is_not_conceal(store):
    s = Sim(store)
    _pick_sequence(s)
    fx, fy = 420, 300
    ledge = (fx + 6, fy - 53)
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 6, fy - 65)])}, [(7, *ledge, "soda_bottle")], n=5)
    s.step({1: (fx, fy, [(fx - 30, fy - 140), (fx + 30, fy - 140)])}, [(7, *ledge, "soda_bottle")], n=40)
    assert s.of(E.CONCEAL) == []


def test_own_phone_after_touching_shelf_is_not_a_pick(store):
    s = Sim(store)
    fx, fy = 725, 330
    s.step({1: (fx, fy, [(fx - 30, fy - 62), (812, 300)])}, n=10)                         # hand on shelf_C edge
    for k in range(10):                                                                     # phone out of pocket
        s.step({1: (fx, fy, [(fx - 30, fy - 62), (fx + 30, fy - 62)])}, [(5, fx + 30, fy - 50, "phone_accessory")])
    assert s.of(E.PICK) == []


def test_brief_occlusion_at_door_is_not_an_exit(store):
    s = Sim(store)
    s.step({1: (600, 575, rest(600, 575))}, n=45)          # 3 s in the store
    s.step({1: (100, 600, rest(100, 600))}, n=5)          # standing in the door zone
    s.step({}, n=24)                                       # fully occluded 1.6 s
    s.step({1: (300, 575, rest(300, 575))}, n=20)          # back in the store
    assert s.of(E.EXIT) == []


def test_passer_by_does_not_inherit_item(store):
    s = Sim(store)
    fx, fy, hand = _walk_in_holding_own_drink(s)
    item = (9, hand[0], hand[1] + 12, "soda_bottle")
    # B stands right in front for 2 frames; A's wrists are momentarily not detected
    for _ in range(2):
        s.step({1: (fx, fy, []), 2: (fx + 20, fy + 5, [(hand[0] + 5, hand[1] + 5), (fx + 50, fy - 57)])}, [item])
    assert s.engine.owner[9] == 1


def _walk_in(s, pid, to_x=600, y=400):
    s.step({pid: (85, 690, rest(85, 690))}, n=5)
    for x in range(85, to_x, 20):
        s.step({pid: (x, y, rest(x, y))})


def test_new_track_id_after_occlusion_continues_the_visit(store):
    s = Sim(store)
    _walk_in(s, 1)
    s.step({1: (600, 400, rest(600, 400))}, n=10)
    s.step({}, n=30)                                        # lost for 2 s (detector miss)
    s.step({2: (610, 400, rest(610, 400))}, n=10)           # tracker hands out a new id, same place
    for x in range(610, 60, -20):
        s.step({2: (x, 640, rest(x, 640))})
    s.step({}, n=60)
    assert [e.person_id for e in s.of(E.ENTER)] == [1]
    assert [e.person_id for e in s.of(E.EXIT)] == [1]


def test_new_track_at_the_door_is_a_new_person(store):
    s = Sim(store)
    _walk_in(s, 1)
    s.step({1: (600, 400, rest(600, 400))}, n=5)
    s.step({}, n=10)
    s.step({2: (85, 690, rest(85, 690))}, n=5)              # someone walking in
    assert sorted(e.person_id for e in s.of(E.ENTER)) == [1, 2]


def test_never_stitch_onto_someone_still_visible(store):
    s = Sim(store)
    _walk_in(s, 1)
    s.step({1: (600, 400, rest(600, 400)), 2: (620, 400, rest(620, 400))}, n=5)
    assert sorted(e.person_id for e in s.of(E.ENTER)) == [1, 2]


def test_stitching_can_be_switched_off(store):
    from bree.events.engine import EngineRules
    s = Sim(store)
    s.engine.r = EngineRules(stitch_dist=0)
    _walk_in(s, 1)
    s.step({}, n=30)
    s.step({2: (610, 400, rest(610, 400))}, n=5)
    assert sorted(e.person_id for e in s.of(E.ENTER)) == [1, 2]


def test_hand_point_extends_past_the_wrist():
    k = np.zeros((17, 3), np.float32)
    k[7] = (100, 100, 1)   # left elbow
    k[9] = (100, 140, 1)   # left wrist
    p = PersonObs(1, (0, 0, 200, 300), 1.0, k)
    assert dict(p.hands(0.3, 0.0))["left"] == (100.0, 140.0)
    assert dict(p.hands(0.3, 0.5))["left"] == (100.0, 160.0)


def test_contained_boxes_are_dropped_as_duplicates():
    from bree.detect.yolo import contained
    b = np.array([[0, 0, 100, 200], [10, 10, 90, 120], [300, 0, 400, 200]], float)
    assert contained(b, np.ones(3, bool), 0.85).tolist() == [False, True, False]


def test_resting_hand_in_a_shelf_zone_does_not_turn_a_conceal_into_a_put_back(store):
    from bree.events.engine import EngineRules
    s = Sim(store)
    s.engine.r = EngineRules(hand_extend=0.0)
    _pick_sequence(s)
    fx, fy = 420, 300
    left_in_shelf_a = (380, 238)                           # resting next to the gondola, inside its polygon
    s.step({1: (fx, fy, [left_in_shelf_a, (fx + 6, fy - 65)])}, [(7, fx + 6, fy - 53, "soda_bottle")], n=5)
    s.step({1: (fx, fy, [left_in_shelf_a, (fx + 6, fy - 65)])}, [], n=30)
    assert s.of(E.PUT_BACK) == [] and len(s.of(E.CONCEAL)) == 1


def test_no_stitch_when_two_lost_people_could_match(store):
    s = Sim(store)
    _walk_in(s, 1, to_x=600)
    _walk_in(s, 3, to_x=640)
    s.step({1: (600, 400, rest(600, 400)), 3: (640, 400, rest(640, 400))}, n=5)
    s.step({}, n=30)                                        # both lost
    s.step({2: (620, 400, rest(620, 400))}, n=5)            # new track between them: ambiguous
    assert 2 in [e.person_id for e in s.of(E.ENTER)]


def test_no_stitch_when_someone_visible_stands_there(store):
    s = Sim(store)
    _walk_in(s, 1, to_x=600)
    _walk_in(s, 3, to_x=300)
    s.step({1: (600, 400, rest(600, 400)), 3: (300, 400, rest(300, 400))}, n=5)
    s.step({3: (610, 400, rest(610, 400))}, n=30)           # 1 lost; 3 now stands where 1 was
    s.step({3: (610, 400, rest(610, 400)), 2: (605, 400, rest(605, 400))}, n=5)
    assert 2 in [e.person_id for e in s.of(E.ENTER)]
