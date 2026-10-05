"""Store-wide closed-world identity (MultiCamIdentity closed_world=True, StoreEvents with rules.closed_world on
several cameras): one pool, births only at the door, hand-off by floor position (+ appearance), deaths at the
exit or the timeout, uncertain marks kept on the store-wide identity. Scripted scenes, no model weights."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bree.calib.bench import _feat, _look
from bree.events.observations import FrameObs, PersonObs
from bree.events.types import Event, EventType as E, LineItem, Payment
from bree.events.zones import merge_stores
from bree.ledger import build_ledger
from bree.track.multicam import MultiCamIdentity, StoreEvents, homography, to_floor
from test_multicam import FPS, VIEWS, _stores, camera_frames

IMG = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
# Camera A sees floor x in [0, 6] m and the door (x < 1 m); camera B sees x in [4, 10] m. 2 m of overlap.
H = {"A": homography(IMG, [(0, 0), (6, 0), (6, 4), (0, 4)]), "B": homography(IMG, [(4, 0), (10, 0), (10, 4), (4, 4)])}
X0 = {"A": 0.0, "B": 4.0}
WARM = 6.0                                    # past the 5 s warm-up


def pool(**kw) -> MultiCamIdentity:
    door = lambda p: to_floor(H["A"], *p.foot_point("bottom"))[0] < 1.0      # noqa: E731
    return MultiCamIdentity(H, closed_world=True, entry={"A": door}, **kw)


def person(cam, tid, x, y_m=2.0, feat=None) -> PersonObs:
    px, py = (x - X0[cam]) / 6 * 1280, y_m / 4 * 720
    return PersonObs(tid, (px - 20, py - 150, px + 20, py), 0.9, reid=feat)


def see(mc, t, **cams):
    """see(mc, t, A=[(track id, x), ...], B=[...]); a third tuple item is the y in metres, a fourth the features."""
    for cam, people in cams.items():
        mc.update(cam, FrameObs(0, t, [person(cam, *p) for p in people]))


def gid(mc, cam, tid):
    return mc.local_to_global.get((cam, tid))


def enter(mc, t0=0.0, **who):
    """enter(mc, a=(track id in A, x to walk to), ...): each appears at the door, then stands at x until WARM."""
    see(mc, t0, A=[(tid, 0.5, 1.0 + k) for k, (tid, x) in enumerate(who.values())])
    t = t0 + 1.0
    while t <= max(WARM, t0 + 1.0):
        see(mc, t, A=[(tid, x, 1.0 + k) for k, (tid, x) in enumerate(who.values())])
        t += 0.5
    return t


# ------------------------------------------------------------------ births


def test_births_only_at_the_door_and_a_track_inside_is_a_flagged_missed_entry():
    mc = pool()
    see(mc, 0.0, A=[(1, 0.5)])
    assert mc.people[gid(mc, "A", 1)].born == "entrance"
    see(mc, 1.0, B=[(9, 8.0)])                                       # during warm-up: someone who was already inside
    assert mc.people[gid(mc, "B", 9)].born == "warmup"
    for t in np.arange(1.5, 8.0, 0.5):
        see(mc, t, A=[(1, 2.0)], B=[(9, 8.0)])
    see(mc, 8.0, A=[(1, 2.0)], B=[(9, 8.0), (10, 9.5, 0.5)])         # nobody is unaccounted for, far from anyone
    g = mc.people[gid(mc, "B", 10)]
    assert g.born == "missed_entry" and mc.cw_stats["births_missed_entry"] == 1
    assert mc.occupancy() == 2                                       # a brand-new track inside is not counted yet...
    for t in (9.0, 10.5):
        see(mc, t, A=[(1, 2.0)], B=[(9, 8.0), (10, 9.5, 0.5)])
    assert mc.occupancy() == 3                                       # ...until it has lasted 2 s
    [ev] = mc.remap("B", [Event(E.ENTER, 8.0, 10)])
    assert ev.meta["born"] == "missed_entry" and ev.meta["missed_entry"] is True and ev.person_id == g.gid


def test_no_door_camera_at_all_means_no_door_births():
    mc = MultiCamIdentity(H, closed_world=True)
    for t in (10.0, 11.0, 12.5):
        see(mc, t, A=[(1, 2.0)])
    see(mc, 20.0, B=[(2, 9.0)])                                      # track 1 is gone: it is them
    assert gid(mc, "B", 2) == gid(mc, "A", 1) and mc.people[1].born == "warmup"
    see(mc, 21.0, B=[(2, 9.0), (3, 7.0, 0.5)])
    assert mc.people[gid(mc, "B", 3)].born == "no_door"


# ------------------------------------------------------------------ hand-off


def test_walking_through_the_overlap_is_one_identity():
    mc = pool()
    t = 0.0
    for x in np.arange(0.5, 9.0, 0.1):                               # 1.5 m/s
        t += 0.1 / 1.5
        see(mc, t, **({"A": [(7, x)]} if x <= 6 else {}), **({"B": [(3, x)]} if x >= 4 else {}))
    assert gid(mc, "A", 7) == gid(mc, "B", 3) and len(mc.people) == 1
    assert mc.cw_stats["handoffs_live"] == 1 and mc.cw_stats["uncertain_marks"] == 0
    assert [e.type for e in mc.remap("B", [Event(E.ENTER, t, 3), Event(E.PICK, t, 3)])] == [E.PICK]


def test_a_long_blind_gap_is_still_the_same_person():
    """Lost at the shelf, seen 40 s later at the far end by the other camera. Position and time alone call that a
    new person (the visit splits); one person came in and nobody left, so the closed world says it is them."""
    for closed in (False, True):
        mc = pool() if closed else MultiCamIdentity(H)
        t = enter(mc, a=(1, 3.0))
        see(mc, t + 40.0, B=[(5, 9.5)])
        assert (gid(mc, "B", 5) == gid(mc, "A", 1)) is closed
        assert len(mc.people) == (1 if closed else 2)
    assert mc.cw_stats["assigned_lost"] == 1 and mc.occupancy() == 1


def test_someone_another_camera_is_watching_elsewhere_is_not_a_candidate():
    mc = pool()
    t = enter(mc, a=(1, 3.0), b=(2, 3.5))
    see(mc, t, A=[(1, 3.0)])                                         # b walks off unseen, a stays in view of A
    see(mc, t + 8.0, A=[(1, 3.0)], B=[(5, 9.0)])
    assert gid(mc, "B", 5) == gid(mc, "A", 2) != gid(mc, "A", 1)
    assert mc.cw_stats["uncertain_marks"] == 0                       # a is visibly somewhere else: no doubt


def test_two_lost_at_once_are_both_placed_and_both_marked():
    """Simultaneous occlusion: two shoppers vanish in camera A and two tracks start in camera B. No new people;
    without appearance the call is too close, so both identities are marked, and the mark follows them to
    every camera."""
    mc = pool()
    t = enter(mc, a=(1, 5.0), b=(2, 5.3))
    see(mc, t + 6.0, B=[(5, 8.0, 1.0), (6, 8.3, 2.0)])
    assert {gid(mc, "B", 5), gid(mc, "B", 6)} == {gid(mc, "A", 1), gid(mc, "A", 2)} and len(mc.people) == 2
    assert all(g.uncertain_until > t + 6.0 for g in mc.people.values()) and mc.cw_stats["uncertain_marks"] >= 2
    see(mc, t + 30.0, A=[(8, 2.0)])                                  # one of them, later, back in camera A
    [ev] = mc.remap("A", [Event(E.PICK, t + 30.0, 8, item="soda_bottle")])
    assert "could be person" in ev.meta["identity_uncertain"] and ev.meta["identity_uncertain_until"] > t + 30.0


def test_appearance_settles_who_is_who_after_they_trade_places():
    rng = np.random.default_rng(0)
    la, lb = _look(rng), _look(rng)
    mc = pool(reid=True)
    for t in np.arange(0.0, WARM + 0.1, 0.5):
        x = 0.5 if t == 0 else 5.0
        see(mc, t, A=[(1, x, 1.0, _feat(rng, la, t)), (2, x + 0.3, 2.0, _feat(rng, lb, t))])
    t = WARM + 6.0                                                   # they reappear in B at each other's side
    see(mc, t, B=[(5, 8.3, 2.0, _feat(rng, la, t)), (6, 8.0, 1.0, _feat(rng, lb, t))])
    assert gid(mc, "B", 5) == gid(mc, "A", 1) and gid(mc, "B", 6) == gid(mc, "A", 2)
    assert mc.cw_stats["uncertain_marks"] == 0
    # Without appearance the same scene is a coin toss that is said out loud.
    blind = pool()
    for t in np.arange(0.0, WARM + 0.1, 0.5):
        x = 0.5 if t == 0 else 5.0
        see(blind, t, A=[(1, x, 1.0), (2, x + 0.3, 2.0)])
    see(blind, WARM + 6.0, B=[(5, 8.3, 2.0), (6, 8.0, 1.0)])
    assert blind.cw_stats["uncertain_marks"] >= 2


def test_a_second_box_on_someone_is_not_a_person_until_it_lasts():
    mc = pool()
    t = enter(mc, a=(1, 3.0))
    see(mc, t, A=[(1, 3.0), (2, 3.2)])                               # detector doubles the box
    assert gid(mc, "A", 2) is None and mc.remap("A", [Event(E.PICK, t, 2)]) == []
    assert mc.cw_stats["second_box_frames_ignored"] == 1 and len(mc.people) == 1
    see(mc, t + 0.5, A=[(1, 3.0)])                                   # it went away: never a person
    assert len(mc.people) == 1
    for k in range(6):                                               # one that stays 2 s is a missed entry after all
        see(mc, t + 1.0 + 0.5 * k, A=[(1, 3.0), (3, 3.2)])
    assert mc.people[gid(mc, "A", 3)].born == "missed_entry"


# ------------------------------------------------------------------ deaths


def test_exit_ends_the_identity_everywhere_and_forgets_the_features():
    rng = np.random.default_rng(1)
    la = _look(rng)
    mc = pool(reid=True)
    for t in np.arange(0.0, WARM + 0.1, 0.5):
        see(mc, t, A=[(1, 0.5 if t < 5 else 0.6, 1.0, _feat(rng, la, t))])
    g = mc.people[gid(mc, "A", 1)]
    assert g.gallery
    assert [e.type for e in mc.remap("A", [Event(E.EXIT, WARM, 1)])] == [E.EXIT]
    assert g.exited and g.gallery == [] and mc.occupancy() == 0 and mc.cw_stats["exits"] == 1
    assert mc.remap("A", [Event(E.EXIT, WARM, 1)]) == []             # a second door camera reporting the same exit
    see(mc, WARM + 20.0, B=[(5, 9.0)])                               # nobody is left to be: not the one who left
    assert gid(mc, "B", 5) != g.gid and mc.people[gid(mc, "B", 5)].born == "missed_entry"


def test_an_exit_is_dropped_when_another_camera_still_sees_them():
    mc = pool()
    t = enter(mc, a=(1, 0.8))
    for k in range(8):                                               # A lost them in the door zone; B sees them walk on
        see(mc, t + 2.0 + 0.5 * k, B=[(5, 4.5 + 0.3 * k)])
    assert gid(mc, "B", 5) == gid(mc, "A", 1)
    assert mc.remap("A", [Event(E.EXIT, t - 0.5, 1)]) == []
    assert not mc.people[gid(mc, "A", 1)].exited and mc.cw_stats["exits_vetoed"] == 1


def test_timeout_drops_an_identity_that_never_left():
    mc = pool(timeout_s=60.0)
    t = enter(mc, a=(1, 3.0), b=(2, 3.5))
    for k in range(1, 15):
        see(mc, t + 5.0 * k, A=[(2, 3.5, 2.0)])                      # a is never seen again
    a = mc.people[gid(mc, "A", 1)]
    assert a.timed_out and mc.cw_stats["timeouts"] == 1 and mc.occupancy() == 1
    see(mc, t + 80.0, A=[(2, 3.5, 2.0)], B=[(9, 9.0)])               # a timed-out identity is not handed out again
    assert gid(mc, "B", 9) != a.gid


def test_someone_walking_in_is_not_handed_the_identity_of_someone_who_just_walked_out():
    mc = pool()
    t = enter(mc, a=(1, 0.6))
    see(mc, t + 1.5, A=[(2, 0.5)])                                   # 1.5 s after a was last seen at the door
    assert gid(mc, "A", 2) != gid(mc, "A", 1) and mc.people[gid(mc, "A", 2)].born == "entrance"


# ------------------------------------------------------------------ whole store, one ledger


def _run_closed(via_register: bool, payments):
    stores = {n: replace(s, rules={**s.rules, "closed_world": True}) for n, s in _stores().items()}
    se = StoreEvents(stores)
    led = build_ledger(merge_stores(list(stores.values())))
    frames, events = camera_frames(via_register), []
    pending = sorted(payments, key=lambda p: p.t)
    for f in range(len(frames["door"])):
        while pending and pending[0].t <= f / FPS:
            led.on_payment(pending.pop(0))
        for cam in VIEWS:
            for ev in se.update(cam, frames[cam][f]):
                events.append(ev)
                led.on_event(ev)
        led.tick(f / FPS)
    for ev in se.flush():
        events.append(ev)
        led.on_event(ev)
    led.finalize()
    return se, led, events


def test_three_cameras_closed_world_one_visit_one_ledger():
    se, led, events = _run_closed(True, [Payment(14.0, "pos_1", [LineItem(sku="COKE-20OZ")], txn_id="T1")])
    assert se.identity.closed_world and not any(e.r.closed_world for e in se.engines.values())
    enters = [e for e in events if e.type == E.ENTER]
    assert len(enters) == 1 and enters[0].meta["born"] == "entrance" and enters[0].meta["camera"] == "door"
    assert {(e.type, e.meta["camera"]) for e in events} >= {(E.PICK, "cooler"), (E.PAY, "register"), (E.EXIT, "door")}
    assert {e.person_id for e in events} == {1} and list(led.people) == [1]
    assert led.people[1].reconciled and led.alerts == []
    s = se.identity.cw_stats
    assert s["births_entrance"] == 1 and s["births_missed_entry"] == 0 and s["exits"] == 1 and s["uncertain_marks"] == 0
    assert se.identity.occupancy() == 0 and s["handoffs_live"] + s["assigned_lost"] >= 4


def test_three_cameras_closed_world_walkout_is_one_alert():
    se, led, events = _run_closed(False, [])
    assert list(led.people) == [1] and [e.type for e in events].count(E.EXIT) == 1
    assert len(led.alerts) == 1 and [i.category for i in led.alerts[0].unpaid_items] == ["soda_bottle"]
    assert not any("identity_uncertain" in e.meta for e in events)


# ------------------------------------------------------------------ audit 2026-10-05


def test_tailgating_at_the_door_is_two_people_and_the_exit_is_delivered():
    """b walks in 0.2 s (and 0.4 s) after a's track ended in the door zone. b must not get a's identity, and
    a's EXIT (emitted later, timed at a's last sighting) must not be vetoed because "a" was seen since."""
    for gap in (0.2, 0.4, 0.6, 1.0):
        mc = pool()
        t = enter(mc, a=(1, 0.6)) - 0.5                              # a's last frame, in the door zone
        ga = gid(mc, "A", 1)
        for k in range(8):                                           # b appears at the door and walks in
            see(mc, t + gap + 0.5 * k, A=[(2, 0.5 + 0.4 * k)])
        assert gid(mc, "A", 2) != ga and mc.people[gid(mc, "A", 2)].born == "entrance", gap
        assert [e.type for e in mc.remap("A", [Event(E.EXIT, t, 1)])] == [E.EXIT], gap
        assert mc.people[ga].exited and mc.cw_stats["exits_vetoed"] == 0 and mc.cw_stats["births_entrance"] == 2
        [pick] = mc.remap("A", [Event(E.PICK, t + 4.0, 2, item="soda_bottle")])
        assert pick.person_id != ga                                  # b's basket is b's


def test_two_door_cameras_on_one_person_are_still_one_identity():
    """The door rule must not split someone two cameras watch at the door at the same moment."""
    door = lambda H_: (lambda p: to_floor(H_, *p.foot_point("bottom"))[0] < 5.0)      # noqa: E731
    mc = MultiCamIdentity(H, closed_world=True, entry={"A": door(H["A"]), "B": door(H["B"])})
    for k in range(5):
        see(mc, 0.1 * k, A=[(1, 4.5)], B=[(7, 4.5)])
    assert gid(mc, "A", 1) == gid(mc, "B", 7) and len(mc.people) == 1


def test_a_track_seen_again_after_its_exit_is_placed_again():
    mc = pool()
    t = enter(mc, a=(1, 0.6))
    assert [e.type for e in mc.remap("A", [Event(E.EXIT, t, 1)])] == [E.EXIT]
    old = gid(mc, "A", 1)
    see(mc, t + 5.0, A=[(1, 3.0)])                                   # the same local track id, inside the store
    new = gid(mc, "A", 1)
    assert new != old and mc.people[old].exited and not mc.people[new].exited
    assert mc.cw_stats["tracks_seen_after_exit"] == 1 and mc.occupancy() == 0   # counted once it has lasted 2 s
    see(mc, t + 8.0, A=[(1, 3.0)])
    assert mc.occupancy() == 1
    [pick] = mc.remap("A", [Event(E.PICK, t + 8.0, 1, item="soda_bottle")])
    assert pick.person_id == new


def test_to_floor_is_none_above_the_horizon_and_cut_feet_do_not_move_the_person():
    import math

    from bree.calib.camera import from_layout
    cam = from_layout({"position": [0, 2.5, 0], "yaw": 0.0, "pitch": math.radians(-10), "hfov": 60, "resolution": [1280, 720]})
    Hc = cam.floor_homography()                                      # looks down -z, 10 degrees below level
    px, depth = cam.project([[0.5, 0.0, -5.0]])
    assert depth[0] > 0 and to_floor(Hc, *px[0]) == pytest.approx((0.5, -5.0), abs=1e-6)
    assert to_floor(Hc, 640.0, 0.0) is None                          # top of the frame: above the horizon
    mc = MultiCamIdentity(H, frame_sizes={"A": (1280, 720)})
    mc.update("A", FrameObs(0, 0.0, [PersonObs(1, (600, 300, 640, 500), 0.9)]))
    before = mc.people[1].pos
    mc.update("A", FrameObs(1, 0.1, [PersonObs(1, (600, 400, 640, 720), 0.9)]))     # box cut by the frame bottom
    assert mc.people[1].pos == before and mc.people[1].t_last == 0.1
