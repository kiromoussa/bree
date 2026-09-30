"""Ledger: baskets, payment attribution, reconciliation, scoring, holds."""
from __future__ import annotations

import pytest

from bree.events.types import Catalog, Event, EventType as E, LineItem, Payment
from bree.ledger.ledger import Ledger, LedgerConfig

CATALOG = Catalog({"COKE": "soda", "PEPSI": "soda", "SNICKERS": "candy", "REDBULL": "energy", "CHIPS": "chips"})
TERMINALS = {"pos_1": "register", "tap_1": "cooler"}
KINDS = {"register": "register", "cooler": "cooler", "shelf_A": "shelf", "door": "exit"}


def ledger(**cfg) -> Ledger:
    return Ledger(CATALOG, LedgerConfig(**cfg), TERMINALS, KINDS)


def enter(t, p):
    return Event(E.ENTER, t, p)


def pick(t, p, item, zone="shelf_A", conf=0.9, candidates=()):
    return Event(E.PICK, t, p, item=item, zone=zone, confidence=conf, candidates=list(candidates))


def put_back(t, p, item, zone="shelf_A"):
    return Event(E.PUT_BACK, t, p, item=item, zone=zone)


def conceal(t, p, item, conf=0.7):
    return Event(E.CONCEAL, t, p, item=item, confidence=conf)


def visit(t0, t1, p):
    """A register visit as the vision engine reports it: start, then end."""
    return [Event(E.PAY, t0 + 2.5, p, zone="register", meta={"phase": "start", "t_start": t0}),
            Event(E.PAY, t1, p, zone="register", meta={"phase": "end", "t_start": t0, "t_end": t1})]


def leave(t, p, held=()):
    return Event(E.EXIT, t, p, zone="door", meta={"held_items": list(held)})


def pos(t, *skus, terminal="pos_1"):
    return Payment(t, terminal, [LineItem(sku=s) for s in skus], txn_id=f"T{t}")


# --------------------------------------------------------------- basics

def test_paid_for_everything_no_alert():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda", "cooler"), *visit(20, 30, 1), leave(35, 1, ["soda"])]
    assert L.replay(ev, [pos(25, "COKE")]) == []
    assert L.people[1].reconciled


def test_walkout_with_item_in_hand_alerts():
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "energy", "cooler"), leave(20, 1, ["energy"])])
    assert len(alerts) == 1
    a = alerts[0]
    assert a.tier == "alert" and a.person_id == 1
    assert [i.category for i in a.unpaid_items] == ["energy"]
    # 0.5*0.9 + 0.2 held + 0.15 no register = 0.80
    assert a.confidence == pytest.approx(0.80)
    assert not a.visited_register


def test_single_uncorroborated_pick_is_never_an_alert():
    """A missed put-back looks exactly like this: must not accuse anyone."""
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "candy", conf=1.0), leave(20, 1)])
    assert len(alerts) == 1 and alerts[0].tier == "review"
    assert alerts[0].confidence == pytest.approx(0.65)


def test_partial_payment_with_conceal_alerts_on_concealed_item():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda", "cooler"), pick(8, 1, "candy"), conceal(9, 1, "candy"),
          *visit(20, 30, 1), leave(35, 1, ["soda"])]
    alerts = L.replay(ev, [pos(25, "COKE")])
    assert len(alerts) == 1
    assert [i.category for i in alerts[0].unpaid_items] == ["candy"]
    assert alerts[0].unpaid_items[0].concealed
    assert alerts[0].tier == "alert"


def test_conceal_then_pay_is_fine():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "candy"), conceal(6, 1, "candy"), *visit(20, 30, 1), leave(35, 1)]
    assert L.replay(ev, [pos(25, "SNICKERS")]) == []


def test_put_back_removes_item():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "chips"), put_back(8, 1, "chips"), leave(20, 1)]
    assert L.replay(ev) == []
    assert L.basket_of(1) == {}


def test_put_back_of_unknown_item_is_ignored():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda"), put_back(8, 1, "candy"), *visit(20, 30, 1), leave(35, 1)]
    assert L.replay(ev, [pos(25, "COKE")]) == []
    assert "ignored" in " ".join(L.people[1].log)


def test_conceal_without_seen_pick_adds_item():
    L = ledger()
    alerts = L.replay([enter(0, 1), conceal(9, 1, "candy", conf=0.8), leave(20, 1)])
    assert len(alerts) == 1
    it = alerts[0].unpaid_items[0]
    assert it.category == "candy" and it.concealed
    # 0.5*0.8 + 0.35 + 0.15 = 0.90
    assert alerts[0].confidence == pytest.approx(0.90)


def test_lingerer_without_picks_no_alert():
    L = ledger()
    assert L.replay([enter(0, 1), leave(300, 1)]) == []


def test_paid_more_than_seen_is_not_a_problem():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda"), *visit(20, 30, 1), leave(35, 1)]
    assert L.replay(ev, [pos(25, "COKE", "PEPSI", "SNICKERS")]) == []


def test_sku_match_before_category_match():
    L = ledger()
    ev = [enter(0, 1),
          Event(E.PICK, 5, 1, sku="COKE", zone="cooler", confidence=0.9),
          Event(E.PICK, 6, 1, sku="PEPSI", zone="cooler", confidence=0.9),
          *visit(20, 30, 1), leave(35, 1)]
    alerts = L.replay(ev, [pos(25, "PEPSI")])
    assert len(alerts) == 1
    assert alerts[0].unpaid_items[0].sku == "COKE"


def test_payment_clears_least_suspicious_item_first():
    """Two sodas, one pocketed, paid for one: the pocketed one is the unpaid one."""
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda", "cooler"), pick(6, 1, "soda", "cooler"), conceal(7, 1, "soda"),
          *visit(20, 30, 1), leave(35, 1, ["soda"])]
    alerts = L.replay(ev, [pos(25, "COKE")])
    assert len(alerts) == 1
    assert alerts[0].unpaid_items[0].concealed


# ------------------------------------------------------ payment attribution

def test_pos_payment_before_visit_is_recorded_is_held_then_matched():
    L = ledger()
    L.on_event(enter(0, 1))
    L.on_event(pick(5, 1, "soda"))
    L.on_payment(pos(22, "COKE"))                 # arrives while nobody's visit is known yet
    assert len(L.unassigned_payments) == 1
    for e in visit(20, 30, 1):
        L.on_event(e)
    assert L.unassigned_payments == []
    assert [li.category for li in L.people[1].paid] == ["soda"]


def test_queue_payment_goes_to_person_at_the_counter_not_person_who_just_left():
    L = ledger()
    ev = [enter(0, 1), enter(1, 2), pick(5, 1, "soda"), pick(6, 2, "candy"),
          *visit(20, 30, 1),                    # person 1 at counter 20-30
          Event(E.PAY, 27.5, 2, zone="register", meta={"phase": "start", "t_start": 25}),  # 2 queued at 25
          Event(E.PAY, 40, 2, zone="register", meta={"phase": "end", "t_start": 25, "t_end": 40}),
          leave(45, 1), leave(50, 2)]
    # Person 1 pays at 28 (both present: head of queue wins); person 2 pays at 32
    # (person 1 left at 30, only within slack -> person 2 standing there wins).
    alerts = L.replay(ev, [pos(28, "COKE"), pos(32, "SNICKERS")])
    assert alerts == []
    assert [li.category for li in L.people[1].paid] == ["soda"]
    assert [li.category for li in L.people[2].paid] == ["candy"]


def test_cooler_tap_attributed_to_person_who_picked_there():
    L = ledger()
    ev = [enter(0, 1), enter(25, 2), pick(10, 1, "energy", "cooler"), pick(40, 2, "energy", "cooler"),
          leave(20, 1, ["energy"]), leave(60, 2, ["energy"])]
    alerts = L.replay(ev, [pos(9, "REDBULL", terminal="tap_1")])
    assert [a.person_id for a in alerts] == [2]


def test_unmatched_payment_expires_to_orphans():
    L = ledger(payment_expiry_s=60)
    L.on_payment(pos(10, "COKE"))
    L.on_event(enter(100, 1))
    assert L.unassigned_payments == [] and len(L.orphan_payments) == 1


def test_dwell_mode_register_visit_pays_for_earlier_picks_only():
    L = ledger(payment_mode="dwell")
    ev = [enter(0, 1), pick(5, 1, "soda"), *visit(20, 30, 1), pick(32, 1, "candy"), conceal(33, 1, "candy"),
          leave(40, 1)]
    alerts = L.replay(ev)
    assert len(alerts) == 1
    assert [i.category for i in alerts[0].unpaid_items] == ["candy"]


# ----------------------------------------------------------- groups / crowd

def test_group_one_person_pays_for_both():
    L = ledger()
    ev = [enter(0, 1), enter(2, 2), pick(10, 1, "soda"), pick(12, 2, "soda"),
          *visit(20, 30, 1), leave(35, 1), leave(36, 2)]
    assert L.replay(ev, [pos(25, "COKE", "PEPSI")]) == []
    assert L.people[1].group == [1, 2] == L.people[2].group


def test_group_hold_waits_for_mate_then_decides():
    L = ledger()
    for e in [enter(0, 1), enter(2, 2), pick(10, 2, "soda"), leave(35, 2)]:
        L.on_event(e)
    assert L.tick(50) == []                       # past grace but mate 1 still inside -> hold
    assert 2 in L.pending
    for e in visit(40, 55, 1):
        L.on_event(e)
    L.on_payment(pos(50, "COKE"))
    L.on_event(leave(60, 1))
    assert L.tick(70) == []                       # 1 paid for 2's soda
    assert not L.pending


def test_strangers_entering_together_still_caught_if_nothing_covers_it():
    L = ledger()
    ev = [enter(0, 1), enter(2, 2), pick(10, 1, "soda"), pick(12, 2, "energy", conf=0.9),
          conceal(13, 2, "energy"), *visit(20, 30, 1), leave(35, 1), leave(36, 2)]
    alerts = L.replay(ev, [pos(25, "COKE")])
    assert len(alerts) == 1
    assert [i.category for i in alerts[0].unpaid_items] == ["energy"]


def test_group_hold_times_out():
    L = ledger(max_hold_s=60)
    for e in [enter(0, 1), enter(2, 2), pick(10, 2, "energy"), conceal(11, 2, "energy"), leave(35, 2)]:
        L.on_event(e)
    assert L.tick(90) == []
    alerts = L.tick(96)
    assert len(alerts) == 1 and alerts[0].person_id == 2


def test_crowded_pick_resolved_by_other_candidates_surplus():
    """1 and 2 reach into the cooler together. Vision credits both sodas to 1;
    2 pays for a soda they were never seen taking -> that soda was theirs."""
    L = ledger()
    ev = [enter(0, 1), enter(30, 2),
          pick(40, 1, "soda", "cooler", candidates=[1, 2]), pick(40.5, 1, "soda", "cooler", candidates=[1, 2]),
          *visit(50, 60, 1), *visit(62, 70, 2), leave(65, 1), leave(75, 2)]
    alerts = L.replay(ev, [pos(55, "COKE"), pos(66, "PEPSI")])
    assert alerts == []


def test_crowded_pick_unresolved_is_downweighted():
    L = ledger()
    ev = [enter(0, 1), enter(30, 2), pick(40, 1, "soda", "cooler", conf=0.9, candidates=[1, 2]),
          leave(65, 1, ["soda"]), leave(75, 2)]
    alerts = L.replay(ev)
    # (0.45 + 0.2 + 0.15) * 0.5 = 0.40 -> review, not alert
    assert len(alerts) == 1 and alerts[0].tier == "review"
    assert alerts[0].confidence == pytest.approx(0.40)


def test_crowded_pick_waits_for_other_candidate():
    L = ledger()
    for e in [enter(0, 1), enter(30, 2), pick(40, 1, "soda", "cooler", candidates=[1, 2]), leave(45, 1)]:
        L.on_event(e)
    assert L.tick(60) == [] and 1 in L.pending     # waiting on candidate 2
    for e in visit(50, 58, 2):
        L.on_event(e)
    L.on_payment(pos(55, "COKE", "COKE"))            # 2 pays for two sodas
    L.on_event(leave(62, 2))
    assert L.tick(80) == []
    assert not L.pending


def test_mutual_ambiguity_does_not_deadlock():
    L = ledger()
    ev = [enter(0, 1), enter(30, 2),
          pick(40, 1, "soda", "cooler", candidates=[1, 2]), pick(40.2, 2, "soda", "cooler", candidates=[2, 1]),
          leave(60, 1, ["soda"]), leave(61, 2, ["soda"])]
    alerts = L.replay(ev)
    # each: (0.45 + 0.2 + 0.15) * 0.5 = 0.40 -> review; and nobody waits for max_hold
    assert {a.person_id for a in alerts} == {1, 2}
    assert all(a.tier == "review" and a.latency_s < 30 for a in alerts)


def test_group_alert_names_the_person_holding_the_unpaid_item():
    L = ledger()
    ev = [enter(0, 1), enter(1, 2), pick(10, 2, "energy"), conceal(11, 2, "energy"), leave(30, 1), leave(31, 2)]
    alerts = L.replay(ev)
    assert len(alerts) == 1 and alerts[0].person_id == 2 and alerts[0].group == [1, 2]


# --------------------------------------------------------------- timing

def test_alert_emitted_after_grace_period():
    L = ledger(exit_grace_s=5)
    for e in [enter(0, 1), pick(5, 1, "energy"), conceal(6, 1, "energy"), leave(20, 1)]:
        assert L.on_event(e) == []
    assert L.tick(24.9) == []
    alerts = L.tick(25.0)
    assert len(alerts) == 1 and alerts[0].latency_s == pytest.approx(5.0)


def test_late_pos_message_within_grace_prevents_alert():
    L = ledger(exit_grace_s=5)
    for e in [enter(0, 1), pick(5, 1, "soda"), conceal(6, 1, "soda"), *visit(10, 18, 1), leave(20, 1)]:
        L.on_event(e)
    L.on_payment(pos(17, "COKE"))   # POS message delayed until after the exit
    assert L.tick(30) == []


def test_replay_decision_times_are_realistic():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "energy"), conceal(6, 1, "energy"), leave(20, 1),
          enter(1000, 2), leave(1100, 2)]
    alerts = L.replay(ev)
    assert alerts[0].t_emitted == pytest.approx(25.0)


def test_audit_log_explains_decision():
    L = ledger()
    L.replay([enter(0, 1), pick(5, 1, "energy", "cooler"), leave(20, 1, ["energy"])])
    log = "\n".join(L.people[1].log)
    for word in ("pick energy", "exit", "reconciled"):
        assert word in log


def test_second_exit_after_reconciliation_is_ignored_and_replay_terminates():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "energy"), conceal(6, 1, "energy"), leave(20, 1), leave(200, 1),
          enter(300, 2), leave(400, 2)]
    alerts = L.replay(ev)
    assert len(alerts) == 1 and not L.pending
    assert "second exit" in " ".join(L.people[1].log)


# ------------------------------------------ regressions from the adversarial review

def test_several_uncorroborated_picks_stay_review():
    """Missed put-backs on 2-3 items must not add up to an alert."""
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "candy", conf=1.0), pick(10, 1, "chips", conf=1.0),
                       pick(15, 1, "soda", conf=1.0), leave(60, 1)])
    assert len(alerts) == 1 and alerts[0].tier == "review"


def test_category_mismatch_with_receipt_is_review_not_alert():
    """Vision says soda, POS says Red Bull: most likely misrecognition."""
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "soda", "cooler", conf=1.0), *visit(20, 30, 1),
                       leave(35, 1, ["soda"])], [pos(25, "REDBULL")])
    assert len(alerts) == 1 and alerts[0].tier == "review"


def test_unknown_sku_covers_an_open_item():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "candy", conf=1.0), *visit(20, 30, 1), leave(35, 1, ["candy"])]
    assert L.replay(ev, [pos(25, "TWIX")]) == []


def test_low_confidence_conceal_is_ignored_and_conceal_weight_scales():
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "soda"), conceal(9, 1, "soda", conf=0.05), leave(30, 1)])
    assert alerts[0].tier == "review" and not alerts[0].unpaid_items[0].concealed
    L = ledger()
    weak = L.replay([enter(0, 1), pick(5, 1, "soda"), conceal(9, 1, "soda", conf=0.35), leave(30, 1)])
    L = ledger()
    strong = L.replay([enter(0, 1), pick(5, 1, "soda"), conceal(9, 1, "soda", conf=0.8), leave(30, 1)])
    assert weak[0].confidence < strong[0].confidence


def test_put_back_keeps_the_concealed_item():
    """Pick two, pocket one, put one back: the pocketed one is still theirs."""
    L = ledger()
    alerts = L.replay([enter(0, 1), pick(5, 1, "candy"), pick(6, 1, "candy"), conceal(7, 1, "candy"),
                       put_back(8, 1, "candy"), leave(30, 1)])
    assert len(alerts) == 1 and alerts[0].tier == "alert" and alerts[0].unpaid_items[0].concealed


def test_walkout_thief_cannot_claim_strangers_receipt_to_clear_concealed_item():
    L = ledger()
    ev = [enter(0, 1), pick(5, 1, "soda"), conceal(6, 1, "soda"), leave(40, 1),
          enter(10, 2), pick(12, 2, "energy", "cooler"), *visit(20, 30, 2),
          enter(11, 3), pick(13, 3, "chips"), *visit(22, 32, 3), leave(60, 2), leave(61, 3)]
    alerts = L.replay(ev, [pos(26, "COKE"), pos(31, "CHIPS")])
    assert any(a.person_id == 1 and a.tier == "alert" for a in alerts)


def test_dwell_mode_does_not_clear_concealed_items():
    L = ledger(payment_mode="dwell")
    alerts = L.replay([enter(0, 1), pick(5, 1, "candy"), conceal(6, 1, "candy"), *visit(20, 23, 1), leave(25, 1)])
    assert len(alerts) == 1


def test_own_conceal_resolves_crowded_pick():
    L = ledger()
    alerts = L.replay([enter(0, 1), enter(30, 2), pick(40, 1, "candy", candidates=[1, 2]),
                       conceal(41, 1, "candy"), leave(60, 1), leave(62, 2)])
    assert len(alerts) == 1 and alerts[0].tier == "alert" and alerts[0].person_id == 1


def test_reused_track_id_starts_a_new_visit():
    L = ledger()
    alerts = L.replay([enter(0, 1), leave(10, 1), enter(100, 1), pick(105, 1, "candy"),
                       conceal(106, 1, "candy"), leave(120, 1)])
    assert len(alerts) == 1 and alerts[0].t_exit == 120 and len(L.archived) == 1


def test_malformed_pay_meta_does_not_crash():
    L = ledger()
    L.on_event(Event(E.PAY, 20, 1, zone="register", meta={"t_start": 20, "t_end": None}))
    L.on_event(Event(E.PAY, 21, 2, zone="register", meta={"phase": "start", "t_start": None}))
    L.on_payment(pos(22, "COKE"))
