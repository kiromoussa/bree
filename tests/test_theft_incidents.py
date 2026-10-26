"""One incident shape, a transparent priority score, and a queue sorted by it."""
from __future__ import annotations

from bree.alerts.types import Alert, UnpaidItem
from bree.theft.incidents import Incident, PriorityModel, from_alert, from_sweep, from_tamper, rank
from bree.theft.sweep import SweepEvent
from bree.theft.tamper import TamperEvent


def alert(conceal=False, held=False, conf=0.6, t=100.0):
    return Alert("A1", 3, "review", conf, t, t + 5, [UnpaidItem("energy", None, t - 30, "cooler", 0.9, conceal, held, [], 0.6)],
                 ["r"], visited_register=False)


def test_conversions_and_bands():
    assert from_alert(alert(conceal=True)).band == "R1"
    assert from_alert(alert(held=True)).band == "R2"
    assert from_alert(alert()).band == "R3"
    t = from_tamper(TamperEvent("covered", "start", 50, 40, "high", {"std": 2}), "ch07")
    assert t.kind == "tamper" and t.camera == "ch07" and "covered" in t.headline
    s = from_sweep(SweepEvent(1, "cooler", 10, 8, "both", 0.9, {"hand": "left"}), "ch12")
    assert s.kind == "bulk_grab" and s.severity == "medium"


def test_priority_orders_concealment_over_plain_pick_and_explains_itself():
    a, b, c = from_alert(alert(conceal=True, conf=0.8), "ch07"), from_alert(alert(conf=0.45), "ch07"), \
        from_tamper(TamperEvent("moved", "start", 5, 0, "high", {}), "ch10")
    q = rank([b, c, a])
    assert q[0] is c or q[0] is a
    assert q.index(a) < q.index(b)
    assert all(0 <= i.priority <= 100 and i.priority_why for i in q)


def test_tampered_camera_discounts_item_evidence_and_closed_store_raises_it():
    base = rank([from_alert(alert(conceal=True), "ch07")])[0].priority
    low = rank([from_alert(alert(conceal=True), "ch07")], tampered={"ch07": [(0, 500)]})[0].priority
    high = rank([from_alert(alert(conceal=True), "ch07")], closed=True)[0].priority
    assert low < base < high


def test_fit_moves_base_towards_labels_and_discounts_noisy_cameras():
    items = [(from_alert(alert(), "ch12"), False) for _ in range(12)]
    m = PriorityModel().fit(items)
    k = ("theft_review", "R3")
    assert m.base[k] < PriorityModel().base[k]
    assert m.camera_noise["ch12"] == 1.0
    assert m.score(from_alert(alert(), "ch12")).priority < PriorityModel().score(from_alert(alert(), "ch12")).priority
