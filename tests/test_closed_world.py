"""Closed-world identity (rules.closed_world): people are only created at the door, a track that starts anywhere
else is one of the people already inside, identities end at the exit or after the timeout, and a call that is
too close keeps one identity per track but caps theft alerts at review. Scripted scenes, no model weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bree.events.engine import EngineRules, EventEngine
from bree.events.observations import FrameObs, PersonObs
from bree.events.types import Event, EventType as E, LineItem, Payment
from bree.events.zones import load_store_config
from bree.ledger import build_ledger
from test_reid import BLUE, FPS, GREEN, GREY, RED, World

ROOT = Path(__file__).resolve().parents[1]
DOOR = (85, 690)                    # inside the example store's exit zone
A, B = (600, 600, RED, BLUE), (760, 600, GREEN, GREY)
WARM = int(6 * FPS)                 # frames past the 5 s warm-up


def world(reid=False, **rules):
    return World(reid=reid, closed_world=True, **rules)


def born(w):
    return {e.person_id: e.meta.get("born") for e in w.events if e.type == E.ENTER}


def walk_in(w, people: dict[int, tuple]):
    """Everyone appears at the door, then stands at their spot until the warm-up is over."""
    w.step({tid: (*DOOR, p[2], p[3]) for tid, p in people.items()}, n=3)
    return w.step(people, n=WARM)


# ------------------------------------------------------------------ births


def test_default_is_off_and_timeout_is_an_hour():
    r = EngineRules()
    assert r.closed_world is False and r.closed_world_timeout_s == 3600.0


def test_a_track_that_starts_inside_is_the_lost_person_not_a_new_one():
    """Far away, long gone and in other clothes: the open-world rules call that a new person. Closed world:
    one person came in and nobody left, so it is them."""
    w = walk_in(world(reid=True), {1: A}).step({}, n=int(20 * FPS))
    w.step({2: (1100, 300, GREEN, GREY)}, n=5)
    assert born(w) == {1: "entrance"} and w.engine.alias == {2: 1}
    assert w.engine.occupancy() == 1
    assert any("looks different" in m for m in w.engine.log)      # still said out loud, and held against alerts
    assert w.engine.people[1].uncertain_until > w.engine.t


def test_new_person_at_the_door_is_a_birth_even_when_someone_is_lost():
    w = walk_in(world(), {1: A}).step({}, n=30)
    w.step({2: (*DOOR, GREEN, GREY)}, n=5)
    assert born(w) == {1: "entrance", 2: "entrance"} and w.engine.alias == {}
    assert w.engine.occupancy() == 2


def test_entrance_zone_kind_creates_people_but_is_not_an_exit(tmp_path):
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "store_gas_station_small.yaml").read_text())
    cfg["zones"].append({"name": "side_entry", "kind": "entrance", "polygon": [[1100, 150], [1280, 150], [1280, 300], [1100, 300]]})
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(cfg))
    store = load_store_config(tmp_path / "s.yaml")
    eng = EventEngine(store, EngineRules.from_dict({**store.rules, "closed_world": True}))
    box = lambda x, y: (x - 22, y - 158, x + 22, y)   # noqa: E731
    ev = []
    for f in range(WARM + 40):
        ev += eng.update(FrameObs(f, f / FPS, [PersonObs(7, box(1190, 250))] if WARM <= f < WARM + 5 else []))
    ev += eng.flush()
    assert [e.meta["born"] for e in ev if e.type == E.ENTER] == ["entrance"]
    assert not [e for e in ev if e.type == E.EXIT]


def test_warm_up_lets_people_already_inside_appear_anywhere():
    w = world().step({1: A, 2: B}, n=5)
    assert born(w) == {1: "warmup", 2: "warmup"}


def test_missed_entry_is_flagged_and_counted_not_fatal():
    w = world().step({}, n=WARM).step({1: A}, n=5)                 # nobody inside, nobody lost
    enter = next(e for e in w.events if e.type == E.ENTER)
    assert enter.meta["born"] == "missed_entry" and enter.meta["missed_entry"] is True
    assert w.engine.cw_stats["births_missed_entry"] == 1 and w.engine.occupancy() == 1
    assert any("missed entry" in m for m in w.engine.log)


def test_no_door_in_the_config_falls_back_to_first_appearance():
    store = load_store_config(ROOT / "configs" / "merl_overhead.yaml")
    eng = EventEngine(store, EngineRules.from_dict({**store.rules, "closed_world": True}))
    ev = []
    for f in range(WARM + 110):                                    # track 1 for 3 s, a 3 s gap, then track 2 elsewhere
        tid = 1 if WARM <= f < WARM + 45 else 2 if f >= WARM + 90 else None
        x = 300 if tid == 1 else 700
        ev += eng.update(FrameObs(f, f / FPS, [PersonObs(tid, (x - 40, 300, x + 40, 460))] if tid else []))
    enters = [e for e in ev if e.type == E.ENTER]
    assert [e.meta["born"] for e in enters] == ["no_door"] and "missed_entry" not in enters[0].meta
    assert eng.alias == {2: 1}


def test_frame_border_counts_as_the_door_when_configured():
    store = load_store_config(ROOT / "configs" / "no_zones.yaml")          # 1280x720, no zones at all
    eng = EventEngine(store, EngineRules.from_dict({"closed_world": True, "entry_border_px": 30}))
    ev = []
    for f in range(WARM + 10):
        ppl = [PersonObs(1, (0, 300, 44, 458)), PersonObs(2, (600, 300, 644, 458))] if f >= WARM else []
        ev += eng.update(FrameObs(f, f / FPS, ppl))
    assert {e.person_id: e.meta["born"] for e in ev} == {1: "entrance", 2: "missed_entry"}


# ------------------------------------------------------------------ assignment


def test_two_lost_at_once_reappear_swapped_appearance_sorts_them_out():
    w = walk_in(world(reid=True), {1: A, 2: B}).step({}, n=int(3 * FPS))
    w.step({3: (A[0], A[1], GREEN, GREY), 4: (B[0], B[1], RED, BLUE)}, n=5)      # traded places while unseen
    assert w.engine.alias == {3: 2, 4: 1} and len(born(w)) == 2
    assert w.engine.cw_stats["uncertain_marks"] == 0


def test_two_lost_at_once_same_clothes_one_identity_each_and_both_uncertain():
    a, b = (600, 600, RED, BLUE), (760, 600, RED, BLUE)
    w = walk_in(world(reid=True), {1: a, 2: b}).step({}, n=int(3 * FPS))
    w.step({3: a, 4: b}, n=5)
    assert sorted(w.engine.alias.values()) == [1, 2] and len(born(w)) == 2       # nobody new, nobody doubled
    assert all(w.engine.people[p].uncertain_until > w.engine.t for p in (1, 2))


def test_without_appearance_a_swap_cannot_be_seen_so_it_is_marked_uncertain():
    w = walk_in(world(reid=False), {1: A, 2: B}).step({}, n=int(3 * FPS))
    w.step({3: (A[0], A[1], GREEN, GREY), 4: (B[0], B[1], RED, BLUE)}, n=5)
    assert w.engine.alias == {3: 1, 4: 2}                                        # position says so; it is wrong
    assert all(w.engine.people[p].uncertain_until > w.engine.t for p in (1, 2))  # and the engine knows it may be
    assert any("identity uncertain" in m for m in w.engine.log)


def test_assignment_is_solved_jointly_not_greedily():
    """Track 3 is nearest to person 1, but track 4 can only be person 1. Greedy (track 3 first) would give
    3 -> 1 and push 4 onto person 2, 310 px away; the joint solution is 4 -> 1, 3 -> 2."""
    a, b = (600, 600, RED, BLUE), (750, 600, RED, BLUE)
    w = walk_in(world(), {1: a, 2: b}).step({}, n=int(0.5 * FPS))
    w.step({3: (660, 600, RED, BLUE), 4: (440, 600, RED, BLUE)}, n=3)
    assert w.engine.alias == {3: 2, 4: 1}


def test_one_new_track_two_lost_people_takes_the_better_one_only():
    w = walk_in(world(reid=True), {1: A, 2: B}).step({}, n=int(3 * FPS))
    w.step({3: (680, 600, GREEN, GREY)}, n=5)
    assert w.engine.alias == {3: 2} and w.engine.occupancy() == 2


def test_more_new_tracks_than_lost_people_extra_one_is_a_missed_entry():
    w = walk_in(world(), {1: A}).step({}, n=30)
    w.step({2: A, 3: (1000, 600, GREEN, GREY)}, n=3)
    assert w.engine.alias == {2: 1} and born(w)[3] == "missed_entry"


def test_a_flicker_that_never_came_through_the_door_is_not_a_person_to_come_back_to():
    w = world(reid=True).step({}, n=WARM).step({1: A}, n=3).step({}, n=30)  # 0.2 s false detection, then nothing
    assert w.engine.people[1].gallery == []                        # nothing to match it to later: forgotten at once
    w.step({2: B}, n=3)
    assert w.engine.alias == {} and w.engine.cw_stats["births_missed_entry"] == 2


def test_second_box_on_a_visible_person_is_not_a_new_person():
    near = (A[0] + 30, A[1], RED, BLUE)
    w = walk_in(world(), {1: A}).step({1: A, 2: near}, n=10)       # nobody is lost, a second box sits on person 1
    assert born(w) == {1: "entrance"} and w.engine.alias == {} and w.engine.occupancy() == 1
    w.step({2: near}, n=5)                                         # the first track ends: the box carries the identity on
    assert w.engine.alias == {2: 1} and born(w) == {1: "entrance"}
    w = walk_in(world(), {1: A}).step({1: A, 2: near}, n=int(2.5 * FPS))
    assert born(w)[2] == "missed_entry"                            # it stayed: a real person whose entry was missed


# ------------------------------------------------------------------ deaths, memory


def test_identity_survives_a_long_absence_until_the_timeout():
    w = walk_in(world(reid=True, closed_world_timeout_s=60.0), {1: A}).step({}, n=int(30 * FPS))
    assert w.engine.people[1].gallery                               # still needed: they have not left
    w.step({2: A}, n=3)
    assert w.engine.alias == {2: 1}                                 # 30 s: far past every open-world stitch window
    w.step({}, n=int(62 * FPS))
    p = w.engine.people[1]
    assert p.timed_out and p.gallery == [] and w.engine.occupancy() == 0
    assert w.engine.cw_stats["timeouts"] == 1 and not [e for e in w.events if e.type == E.EXIT]
    w.step({3: A}, n=3)
    assert born(w)[3] == "missed_entry"                             # a timed-out identity is not revived


def test_exit_ends_the_identity_and_clears_features():
    w = walk_in(world(reid=True), {1: A, 2: B})
    w.step({1: (*DOOR, RED, BLUE), 2: B}, n=10).step({2: B}, n=int(4 * FPS))
    assert [e.person_id for e in w.events if e.type == E.EXIT] == [1]
    assert w.engine.people[1].gallery == [] and w.engine.occupancy() == 1
    w.step({}, n=30).step({3: A}, n=3)                              # the only candidate left is person 2
    assert w.engine.alias == {3: 2}


def test_lost_shopper_keeps_the_basket_and_the_receipt_lands_on_it():
    """Same case as the re-ID register test, with appearance off: closed world alone carries the basket."""
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    reg = next(z for z in store.zones if z.kind == "register")
    rx, ry = float(np.mean(reg.polygon[:, 0])), float(np.mean(reg.polygon[:, 1]))
    w, ledger = walk_in(world(), {1: (300, 400, RED, BLUE)}), build_ledger(store)
    w.events.append(Event(type=E.PICK, t=w.f / FPS, person_id=1, item="soda_bottle", zone="cooler_bank", confidence=0.9))
    w.step({}, n=int(15 * FPS)).step({2: (rx, ry, RED, BLUE)}, n=int(4 * FPS))
    t_pay = w.f / FPS - 1.0
    w.step({}, n=int(3 * FPS))
    for ev in w.events:
        ledger.on_event(ev)
    ledger.on_payment(Payment(terminal="pos_1", t=t_pay, items=[LineItem(category="soda_bottle")]))
    assert [li.category for li in ledger.people[1].paid] == ["soda_bottle"] and list(ledger.people) == [1]


# ------------------------------------------------------------------ uncertainty -> ledger


def _theft(meta):
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    L = build_ledger(store)
    alerts = L.replay([Event(E.ENTER, 0, 1), Event(E.PICK, 5, 1, item="energy_drink", zone="cooler_bank", confidence=0.9),
                       Event(E.CONCEAL, 8, 1, item="energy_drink", confidence=0.7),
                       Event(E.EXIT, 20, 1, zone="door", meta={"held_items": [], **meta})])
    return alerts, L


def test_alert_on_an_uncertain_identity_is_downgraded_to_review_with_the_reason_logged():
    alerts, _ = _theft({})
    assert [a.tier for a in alerts] == ["alert"]
    why = "track 4 could be person 1 or 2 (scores 0.60 vs 0.58)"
    alerts, L = _theft({"identity_uncertain": why, "identity_uncertain_until": 3600.0})
    assert [a.tier for a in alerts] == ["review"]
    assert any("identity uncertain" in r and why in r for r in alerts[0].reasons)
    log = "\n".join(L.people[1].log)                               # this is what ledger_log.txt is written from
    assert "alert downgraded to review: identity uncertain" in log and why in log
    assert L.decisions[-1]["tier"] == "review" and L.decisions[-1]["alert_eligible"] is False


def test_uncertainty_that_ran_out_before_the_exit_does_not_downgrade():
    alerts, _ = _theft({"identity_uncertain": "old", "identity_uncertain_until": 10.0})
    assert [a.tier for a in alerts] == ["alert"]


def test_uncertain_identity_reaches_the_ledger_on_the_exit_event():
    w = walk_in(world(), {1: A, 2: B}).step({}, n=int(3 * FPS)).step({3: A, 4: B}, n=5)
    w.step({3: (*DOOR, RED, BLUE), 4: B}, n=10).step({4: B}, n=int(4 * FPS))
    ex = next(e for e in w.events if e.type == E.EXIT)
    assert "could be person" in ex.meta["identity_uncertain"] and ex.meta["identity_uncertain_until"] > ex.t


def test_pipeline_writes_the_downgrade_reason_to_ledger_log(tmp_path):
    """ledger_log.txt is the per-person audit log joined; the missed-entry note lands there too."""
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    L = build_ledger(store)
    L.on_event(Event(E.ENTER, 1.0, 9, meta={"born": "missed_entry", "missed_entry": True}))
    assert any("entry not seen" in line for line in L.people[9].log)


# ------------------------------------------------------------------ scope, regression guard


def test_multi_camera_stores_run_one_store_wide_pool(tmp_path):
    """Several cameras: no per-camera closed world (a person walking in from another camera's area is not
    someone lost here). The pool lives in MultiCamIdentity (tests/test_multicam_closed_world.py);
    `closed_world: false` in the handoff settings keeps the plain floor-plan handoff."""
    import yaml
    from bree.track.multicam import StoreEvents
    cfg = yaml.safe_load((ROOT / "configs" / "store_gas_station_small.yaml").read_text())
    cfg["rules"]["closed_world"] = True
    cfg["camera"]["floor_points"] = [[212, 690, 0, 0], [1068, 690, 6, 0], [930, 300, 6, 5], [350, 300, 0, 5]]
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(cfg))
    store = load_store_config(tmp_path / "s.yaml")
    assert EventEngine(store).r.closed_world and EventEngine(store)._floor is not None    # one camera: on, floor metres
    se = StoreEvents({"a": store, "b": store})
    assert not any(e.r.closed_world or e.r.stitch_dist for e in se.engines.values())
    assert se.identity.closed_world and set(se.identity.entry) == {"a", "b"}
    assert all("store-wide" in e.log[0] for e in se.engines.values())
    se = StoreEvents({"a": store, "b": store}, {"closed_world": False})
    assert not se.identity.closed_world and not any(e.r.closed_world for e in se.engines.values())
    assert all(e.r.stitch_dist > 0 for e in se.engines.values())


def test_scripted_store_does_not_regress():
    """Scripted multi-shopper store (SYNTHETIC, seeded): closed world must keep cutting splits and silent false
    merges versus the open-world rules, and must not get worse than tests/fixtures/reid_guard.json."""
    from bree.eval.reid_bench import synthetic_store_rows
    ref = json.loads((ROOT / "tests/fixtures/reid_guard.json").read_text())["closed_world_synthetic_store"]
    rows = synthetic_store_rows(episodes=ref["episodes"], seed=ref["seed"])["rows"]
    off = rows["off"]
    for name in ("closed_world", "closed_world_reid"):
        r, want = rows[name], ref[name]
        assert r["splits"] <= want["splits"] and r["splits"] < off["splits"], (name, r)
        assert r["false_merges_silent"] <= want["false_merges_silent"] < off["false_merges_silent"], (name, r)
        assert r["false_merges"] <= want["false_merges"], (name, r)
        assert r["clean_visit_rate"] >= want["clean_visit_rate"] - 1e-9, (name, r)
    assert {k: off[k] for k in ref["off"]} == ref["off"]              # flag off: the old behaviour, exactly
