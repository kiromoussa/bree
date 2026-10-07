"""Shelf events -> shoppers -> ledger, on SYNTHETIC scripted scenes (geometry only): bree.track.associate,
bree.track.floor, bree.events.shelf."""
import numpy as np
import pytest

from bree.calib.camera import from_layout
from bree.events.shelf import run_ledger, store_events
from bree.events.types import EventType, LineItem, Payment
from bree.track.associate import associate
from bree.track.floor import FloorConfig, FloorTracker, counter_zones
from bree.track.scenes import FPS, Path2D, ScriptedTrack, detections, layout, run_scene, summarise

LAY = layout()
SLOT = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-8")          # face (0.6, 0.8, 0.0), normal +x
ACROSS = next(s for s in LAY["slots"] if s["id"] == "G1L-S1-8")


def person(pid, x, z, t0=0.0, t1=20.0, **kw):
    p = ScriptedTrack(pid, Path2D([(t0, x, z), (t1, x, z)]), np.random.default_rng(pid), sigma_m=0.0, gap_p=0.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def shelf(slot, t, kind="take", **kw):
    return {"camera_id": "rail", "t": t, "t_start": t - 0.4, "t_end": t + 0.4, "kind": kind, "slot_id": slot["id"], "sku_id": slot["skuId"], "sku_conf": 0.9,
            "count": 1, "source": "shelf_diff", "hand_px": None, "point_3d": slot["face"], "point_sigma_m": 0.05, "evidence": {"before": None, "after": None, "frames": []}, **kw}


# ---------------------------------------------------------------- association

def test_scripted_scenes_right_shopper_rate():
    s = summarise([run_scene(i) for i in range(80)])
    assert s["all"]["right_shopper_rate"] >= 0.9
    assert s["clear_cases"]["right_shopper_rate"] >= 0.95
    assert s["all"]["wrong_and_sure"] <= 0.01 * s["all"]["events"]
    sh = s["by_case"]["shoulder"]
    assert sh["marked_uncertain"] >= 0.85 * sh["events"]          # nobody could tell: say so


def test_two_shoppers_at_one_shelf():
    a, b = person(1, 1.1, 0.0), person(2, 1.1, 1.0)
    r = associate([shelf(SLOT, 5.0)], [a, b], LAY)[0]
    assert r.person_id == 1 and not r.uncertain
    close = associate([shelf(SLOT, 5.0)], [a, person(2, 1.1, 0.2)], LAY)[0]
    assert close.uncertain and {p for p, _ in close.candidates} == {1, 2}


def test_other_side_of_the_gondola_is_not_a_candidate():
    here, across = person(1, 1.35, 0.5), person(2, -0.75, 0.0)       # the one across is nearer in a straight line
    r = associate([shelf(SLOT, 5.0)], [here, across], LAY)[0]
    assert r.person_id == 1 and 2 not in {p for p, _ in r.candidates}
    assert associate([shelf(ACROSS, 5.0)], [here, across], LAY)[0].person_id == 2


def test_nobody_in_reach_stays_unassigned():
    r = associate([shelf(SLOT, 5.0)], [person(1, 3.5, 2.0), person(2, 1.1, 0.0, t0=10.0)], LAY)[0]
    assert r.person_id is None and not r.candidates


def test_two_takes_in_one_moment_are_solved_together():
    other = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-10")       # 0.6 m along
    a, b = person(1, 1.1, 0.25), person(2, 1.1, 0.75)                     # a is the nearer one for both slots on its own
    alone = associate([shelf(other, 5.0)], [a, b], LAY)[0]
    both = associate([shelf(SLOT, 5.0), shelf(other, 5.1)], [a, b], LAY)
    assert both[0].person_id == 1 and both[1].person_id == 2
    assert alone.uncertain or alone.person_id == 2


def test_one_person_cannot_take_from_two_far_places_at_once():
    far = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-16")          # 2.4 m along
    a = person(1, 1.1, 0.0)
    r = associate([shelf(SLOT, 5.0), shelf(far, 5.2)], [a], LAY)
    assert r[0].person_id == 1 and r[1].person_id is None


def test_identity_doubt_reaches_the_event_and_the_ledger():
    a = person(1, 1.1, 0.0, uncertain="ids 1 and 2 may have been swapped", t_uncertain=3.0)
    r = associate([shelf(SLOT, 2.0), shelf(SLOT, 5.0)], [a], LAY)
    assert not r[0].uncertain and r[1].uncertain          # only from the moment of the doubt on
    events, _ = store_events([shelf(SLOT, 5.0)], [a], LAY)
    pick = next(e for e in events if e.type == EventType.PICK)
    assert pick.meta["identity_uncertain"] and pick.sku == SLOT["skuId"] and pick.meta["slot"]["id"] == SLOT["id"]


def test_staff_are_not_candidates():
    r = associate([shelf(SLOT, 5.0)], [person(1, 1.1, 0.0, staff=True)], LAY)[0]
    assert r.person_id is None


# ---------------------------------------------------------------- floor tracker

def track(paths, seed=0, t_end=None, cfg=None, **det_kw):
    cams = {c["id"]: from_layout(c) for c in LAY["cameras"]}
    scene = {"paths": paths, "t_end": t_end or max(p.pts[-1][0] for p in paths.values()) + 3.0}
    dets = detections(scene, cams, np.random.default_rng(seed), **det_kw)
    tr = FloorTracker(cams, layout=LAY, cfg=cfg)
    for f in range(len(dets["TRACK-A"])):
        tr.update(f / FPS, {c: dets[c][f] for c in dets})
    tr.finish(scene["t_end"])
    return tr


def test_one_identity_per_shopper_from_door_to_exit():
    paths = {1: Path2D([(0.5, 0.0, 6.5), (4.0, 1.7, 3.5), (7.0, 1.2, 0.0), (10.0, 1.2, 0.0), (13.0, 1.7, 3.5), (16.0, 0.3, 7.0)]),
             2: Path2D([(3.0, -0.3, 6.5), (6.5, -1.7, 3.5), (9.5, -1.2, -1.0), (12.0, -1.2, -1.0), (15.0, -1.7, 3.5), (18.5, -0.2, 7.0)])}
    tr = track(paths, second_box_p=0.05)
    people = tr.people()
    assert len(people) == 2 and all(p.born == "door" and p.state == "exited" for p in people)
    assert tr.counts.get("births_inside", 0) == 0
    got = people[0].at(8.0)[0]
    assert np.linalg.norm(got - paths[people[0].id].at(8.0)) < 0.3           # ids are given in door order


def test_lost_person_takes_their_identity_back():
    paths = {1: Path2D([(0.5, 0.0, 6.5), (4.0, 1.7, 3.5), (7.0, 1.2, 0.0), (14.0, 1.2, 0.0)])}
    cams = {c["id"]: from_layout(c) for c in LAY["cameras"]}
    dets = detections({"paths": paths, "t_end": 14.0}, cams, np.random.default_rng(0), miss_p=0.0, second_box_p=0.0)
    tr = FloorTracker(cams, layout=LAY)
    for f in range(len(dets["TRACK-A"])):
        hidden = 80 <= f < 105                                                # nobody sees them for 2.5 s
        tr.update(f / FPS, {c: ([] if hidden else dets[c][f]) for c in dets})
    people = tr.people()
    assert len(people) == 1 and tr.counts.get("hand_backs", 0) == 1 and people[0].uncertain is None


def test_tailgating_at_the_door_is_two_people():
    paths = {1: Path2D([(0.5, 0.2, 6.4), (3.0, 1.5, 3.5), (6.0, 1.5, 3.5), (8.5, 0.3, 7.1)]),         # leaves at 8.5 s
             2: Path2D([(8.9, -0.2, 6.5), (12.0, -1.5, 3.5), (15.0, -1.5, 3.5)])}                     # walks in 0.4 s later
    tr = track(paths, miss_p=0.0, second_box_p=0.0)
    people = sorted(tr.people(), key=lambda p: p.id)
    assert len(people) == 2 and people[0].state == "exited" and people[0].path[-1][0] < 8.9


def test_two_people_passing_very_close_are_marked_uncertain():
    paths = {1: Path2D([(0.5, 0.3, 6.5), (3.0, 1.7, 4.0), (8.0, 1.7, -2.0)]), 2: Path2D([(0.5, 3.0, 6.0), (3.0, 1.75, -2.0), (8.0, 1.75, 4.0)])}
    tr = track(paths, miss_p=0.0, second_box_p=0.0, t_end=8.0)
    assert all(p.uncertain for p in tr.people())
    off = track(paths, miss_p=0.0, second_box_p=0.0, t_end=8.0, cfg=FloorConfig(encounter_m=0.0))
    assert tr.counts["encounter_marks"] >= 1 and "encounter_marks" not in off.counts


def test_scripted_scenes_through_the_floor_tracker():
    s = summarise([run_scene(i, tracker=True) for i in range(40)])
    assert s["identity"]["ids_per_shopper"] <= 1.1
    assert s["all"]["right_shopper_rate"] >= 0.9
    assert s["all"]["wrong_and_sure"] <= 0.01 * s["all"]["events"]
    assert s["identity"]["of_those_not_marked_uncertain"] <= 0.02 * s["identity"]["ids"]


def test_counter_zones_follow_the_counter():
    staff, customer = counter_zones(LAY)            # counter at x 4.1 to 4.9 running along z, nearer the +x wall
    assert staff[0] == pytest.approx(4.9) and customer[1] == pytest.approx(4.1) and customer[0] == pytest.approx(2.9)


# ---------------------------------------------------------------- to the ledger

def visit(pid, pay: bool):
    """Door -> shelf (take at 6 s) -> [counter] -> out."""
    pts = [(0.5, 0.0, 6.5), (3.0, 1.7, 3.5), (5.0, 1.1, 0.0), (7.5, 1.1, 0.0), (9.5, 1.7, 3.5)]
    pts += [(11.5, 3.6, 4.0), (15.0, 3.6, 4.0), (17.5, 0.0, 7.0)] if pay else [(12.0, 0.0, 7.0)]
    return ScriptedTrack(pid, Path2D(pts), np.random.default_rng(pid), sigma_m=0.02, gap_p=0.0)


def receipt(t, *skus):
    return Payment(t=t, terminal="pos_1", txn_id=f"T{t}", items=[LineItem(sku=s, category=s) for s in skus])


def test_take_and_leave_without_paying_is_review_and_a_conceal_cue_makes_it_an_alert():
    thief = visit(1, pay=False)
    events, assocs = store_events([shelf(SLOT, 6.0)], [thief], LAY)
    assert assocs[0].person_id == 1 and [e.type for e in events] == [EventType.ENTER, EventType.PICK, EventType.EXIT]
    alerts, _ = run_ledger(events, [], LAY)
    assert [a.tier for a in alerts] == ["review"] and alerts[0].unpaid_items[0].sku == SLOT["skuId"]
    events, _ = store_events([shelf(SLOT, 6.0)], [thief], LAY, conceal=[{"t": 8.0, "point_3d": [1.1, 1.0, 0.0], "sku_id": SLOT["skuId"], "conf": 0.6}])
    alerts, _ = run_ledger(events, [], LAY)
    assert [a.tier for a in alerts] == ["alert"]


def test_take_and_pay_is_no_alert_and_a_put_back_empties_the_basket():
    shopper = visit(1, pay=True)
    events, _ = store_events([shelf(SLOT, 6.0)], [shopper], LAY)
    assert EventType.PAY in [e.type for e in events]
    alerts, ledger = run_ledger(events, [receipt(14.0, SLOT["skuId"])], LAY)
    assert not alerts and not ledger.people[1].unpaid
    events, _ = store_events([shelf(SLOT, 6.0), shelf(SLOT, 7.0, kind="put")], [visit(2, pay=False)], LAY)
    assert [e.type for e in events if e.type in (EventType.PICK, EventType.PUT_BACK)] == [EventType.PICK, EventType.PUT_BACK]
    assert not run_ledger(events, [], LAY)[0]


def test_paying_for_one_thing_and_not_the_other_is_flagged_for_the_other():
    other = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-9")
    events, _ = store_events([shelf(SLOT, 5.8), shelf(other, 6.6)], [visit(1, pay=True)], LAY)
    alerts, _ = run_ledger(events, [receipt(14.0, SLOT["skuId"])], LAY)
    assert len(alerts) == 1 and [u.sku for u in alerts[0].unpaid_items] == [other["skuId"]]


def test_uncertain_take_names_the_other_candidate():
    a, b = visit(1, pay=False), visit(2, pay=False)
    b.path = [(t, x, z + 0.2) for t, x, z in b.path]
    events, assocs = store_events([shelf(SLOT, 6.0)], [a, b], LAY)
    pick = next(e for e in events if e.type == EventType.PICK)
    assert assocs[0].uncertain and pick.meta["uncertain"] and set(pick.candidates) == {1, 2}


def test_engine_flag_turns_the_single_view_pick_rule_off():
    from bree.events.engine import EngineRules, EventEngine
    from bree.events.types import Event
    eng = EventEngine.__new__(EventEngine)
    eng.r = EngineRules(picks_from_shelf_events=True)
    evs = [Event(EventType.PICK, 1.0, 1), Event(EventType.CONCEAL, 2.0, 1), Event(EventType.PUT_BACK, 3.0, 1)]
    assert [e.type for e in eng._without_picks(evs)] == [EventType.CONCEAL]
    eng.r = EngineRules()
    assert len(eng._without_picks(evs)) == 3


# ---------------------------------------------------------------- which shelf events reach the ledger

def test_slot_watch_alone_is_not_a_pick_and_an_unconfirmed_put_is_not_a_put_back():
    far = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-14")         # 6 facings along: not where the item came from
    evs = [shelf(SLOT, 5.6, cue="slot_state"), shelf(SLOT, 6.0), shelf(far, 7.0, kind="put")]
    events, assocs = store_events(evs, [visit(1, pay=False)], LAY)
    assert [e.type for e in events if e.type in (EventType.PICK, EventType.PUT_BACK)] == [EventType.PICK]
    assert "slot watch alone" in assocs[0].why and "put not confirmed" in assocs[2].why
    events, _ = store_events([evs[1], {**evs[2], "source": "both"}], [visit(1, pay=False)], LAY)      # the item was seen in the hand going in
    assert [e.type for e in events if e.type in (EventType.PICK, EventType.PUT_BACK)] == [EventType.PICK, EventType.PUT_BACK]


def test_a_put_back_where_they_took_it_returns_that_item_whatever_it_is_called():
    events, _ = store_events([shelf(SLOT, 6.0), shelf(SLOT, 7.0, kind="put", sku_id="another_name")], [visit(1, pay=False)], LAY)
    assert [e.sku for e in events if e.type == EventType.PUT_BACK] == [SLOT["skuId"]]
    assert not run_ledger(events, [], LAY)[0]


def test_a_put_that_fits_two_people_goes_to_the_one_who_took_from_there():
    a, b = person(1, 1.1, 0.0), person(2, 1.1, 0.25)
    for first in (1, 2):                              # whoever the geometry prefers, the taker gets the put
        taker = person(first, 1.1, 0.0, t1=5.0)       # stood alone at the shelf for the take
        took = dict(shelf(SLOT, 3.0))
        _, assocs = store_events([took, shelf(SLOT, 12.0, kind="put", source="both")], [a, b] if first == 1 else [b, a], LAY)
        assert assocs[1].person_id == assocs[0].person_id, (first, assocs[1].why, taker.id)


def test_a_count_above_one_is_one_pick():
    events, _ = store_events([shelf(SLOT, 6.0, count=2)], [visit(1, pay=False)], LAY)
    assert sum(e.type == EventType.PICK for e in events) == 1


def test_a_party_walks_together_strangers_who_enter_together_do_not():
    from bree.events.shelf import parties
    walk = lambda pid, dz, split: ScriptedTrack(pid, Path2D([(0.0, 0.0, 6.5 + dz), (4.0, 1.7, 3.5 + dz), (30.0, (9.0 if split else 1.7), 3.5 + dz)]),      # noqa: E731
                                                np.random.default_rng(pid), sigma_m=0.02, gap_p=0.0)
    def paths(*ts):
        for p in ts:
            p.path = [(round(t, 1), *p.at(t)[0]) for t in np.arange(0.0, 30.0, 0.1)]
        return list(ts)
    assert parties(paths(walk(1, 0.0, False), walk(2, 0.5, False))) == {1: [2], 2: [1]}
    assert parties(paths(walk(1, 0.0, False), walk(2, 0.5, True))) == {1: [], 2: []}


def test_a_put_undoes_only_the_reading_it_names():
    """One reach read by two cameras ("eids" a and b). The first camera sees its place as before again (an arm that
    covered the slot went away): the take stands, the other camera still reads it. Both cameras see it: returned."""
    other = next(s for s in LAY["slots"] if s["id"] == "G1R-S1-9")
    take = shelf(SLOT, 6.0, eids=["a", "b"])
    kinds = lambda evs: [e.type for e in evs if e.type in (EventType.PICK, EventType.PUT_BACK)]      # noqa: E731
    events, assocs = store_events([take, shelf(SLOT, 7.0, kind="put", undoes=["a"])], [visit(1, pay=False)], LAY)
    assert kinds(events) == [EventType.PICK] and "still read elsewhere" in assocs[1].why
    assert len(run_ledger(events, [], LAY)[0]) == 1
    # the second camera's put is read at the neighbouring facing and names another product: it still returns this take
    events, _ = store_events([take, shelf(SLOT, 7.0, kind="put", undoes=["a"]), shelf(other, 7.2, kind="put", undoes=["b"])], [visit(1, pay=False)], LAY)
    assert kinds(events) == [EventType.PICK, EventType.PUT_BACK] and [e.sku for e in events if e.type == EventType.PUT_BACK] == [SLOT["skuId"]]
    assert not run_ledger(events, [], LAY)[0]
    # a put that nobody stands near still returns the take of the shopper who was given it
    events, _ = store_events([take, shelf(SLOT, 11.0, kind="put", undoes=["a", "b"])], [visit(1, pay=False)], LAY)
    assert [(e.type, e.person_id) for e in events if e.type == EventType.PUT_BACK] == [(EventType.PUT_BACK, 1)]


def test_pixels_changing_at_the_pay_point_with_no_item_seen_are_not_a_pick():
    x, _, z = SLOT["face"]
    lay = {**LAY, "poi": {"register": [x + 0.5, 0.0, z + 0.2]}}           # the payer stands half a metre in front of this slot
    events, assocs = store_events([shelf(SLOT, 6.0)], [visit(1, pay=False)], lay)
    assert not [e for e in events if e.type == EventType.PICK] and "where the payer stands" in assocs[0].why
    for evs, where in (([shelf(SLOT, 6.0, source="both")], lay), ([shelf(SLOT, 6.0)], LAY)):      # the item seen in a hand, or no pay point here
        assert [e.type for e in store_events(evs, [visit(1, pay=False)], where)[0] if e.type == EventType.PICK] == [EventType.PICK]


def test_a_product_put_into_another_slot_goes_back_and_goods_on_the_counter_do_not():
    other = next(s for s in LAY["slots"] if s["id"] != SLOT["id"] and s["fixtureId"] == SLOT["fixtureId"] and 0.25 < abs(s["face"][2] - SLOT["face"][2]) < 0.6 and s["normal"] == SLOT["normal"])
    came = {**shelf(other, 7.0, kind="put", source="both"), "sku_id": SLOT["skuId"], "arrived": True, "item_in": True}      # read from the picture
    events, _ = store_events([shelf(SLOT, 6.0), came], [visit(1, pay=False)], LAY)
    assert [(e.type, e.sku) for e in events if e.type in (EventType.PICK, EventType.PUT_BACK)] == [(EventType.PICK, SLOT["skuId"]), (EventType.PUT_BACK, SLOT["skuId"])]
    assert not run_ledger(events, [], LAY)[0]
    x, _, z = other["face"]
    events, assocs = store_events([shelf(SLOT, 6.0), came], [visit(1, pay=False)], {**LAY, "poi": {"register": [x + 0.5, 0.0, z + 0.2]}})
    assert not [e for e in events if e.type == EventType.PUT_BACK] and "goods on the counter" in assocs[1].why


def test_a_track_can_be_taken_out_of_a_list_that_holds_another_with_the_same_numbers():
    from bree.track.floor import Track
    a, b = Track(1, np.zeros(4), 0.0, 0.0), Track(2, np.zeros(4), 0.0, 0.0)      # a dataclass == on numpy fields raised here
    tracks = [a, b]
    tracks.remove(b)
    assert tracks == [a] and a != b
