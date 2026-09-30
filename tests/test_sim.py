"""Event-level simulator: ground-truth invariants and ledger sanity under perfect vision."""
from __future__ import annotations

from pathlib import Path

from bree.eval.metrics import run_event_bench
from bree.events.types import EventType as E
from bree.events.zones import load_store_config
from bree.sim.events_sim import SimConfig, VisionNoise, build_world, observe

ROOT = Path(__file__).resolve().parents[1]
STORE = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")


def world(hours=10, seed=5):
    return build_world(STORE.catalog.sku_to_category, SimConfig(hours=hours), seed)


def test_deterministic():
    a, b = world(), world()
    assert [(p.kind, round(p.t_exit, 3)) for p in a.people] == [(p.kind, round(p.t_exit, 3)) for p in b.people]


def test_ground_truth_invariants():
    w = world(hours=20)
    kinds = {p.kind for p in w.people}
    assert {"normal", "thief_walkout", "linger"} <= kinds
    for p in w.people:
        assert p.t_enter < p.t_exit
        for it in p.items:
            assert p.t_enter <= it.t_pick <= p.t_exit + 1e-6
        if p.kind in ("normal", "group", "pocket_then_pay", "cooler_tap", "linger", "browse_putback"):
            assert not p.stolen, p.kind
        if p.kind == "thief_walkout":
            assert p.stolen and p.register is None


def test_register_queue_is_fifo_by_arrival():
    w = world(hours=30)
    visits = sorted((p.register[0], p.payments[0].t) for p in w.people
                    if p.register and p.payments and p.payments[0].terminal == "pos_1")
    pay_times = [t for _, t in visits]
    assert pay_times == sorted(pay_times)


def test_perfect_vision_observation_is_faithful():
    w = world()
    obs = observe(w, VisionNoise().scaled(0), 1)
    picks = sum(1 for e in obs.events if e.type == E.PICK)
    assert picks == sum(len(p.items) for p in w.people)
    assert sum(1 for e in obs.events if e.type == E.EXIT) == len(w.people)
    assert all(len(ids) == 1 for ids in obs.id_map.values())


def test_ledger_nearly_perfect_with_perfect_vision():
    r = run_event_bench(STORE, SimConfig(hours=40), VisionNoise().scaled(0), seed=3)
    assert r["basket_exact_match"] == 1.0
    assert r["precision"] >= 0.9
    assert r["false_alerts_per_hour"] <= 0.15
    assert r["recall_alert_or_review"] >= 0.9


def test_noise_makes_things_worse_not_better():
    clean = run_event_bench(STORE, SimConfig(hours=40), VisionNoise().scaled(0), seed=3)
    noisy = run_event_bench(STORE, SimConfig(hours=40), VisionNoise().scaled(2), seed=3)
    assert noisy["recall"] < clean["recall"]
    assert noisy["basket_exact_match"] < clean["basket_exact_match"]


def test_reproducible_across_processes():
    """Same seed -> same numbers in a fresh interpreter (guards against set/hash ordering)."""
    import subprocess, sys
    code = ("from pathlib import Path;from bree.eval.metrics import run_event_bench;"
            "from bree.events.zones import load_store_config;from bree.sim.events_sim import SimConfig, VisionNoise;"
            f"s=load_store_config(r'{ROOT / 'configs' / 'store_gas_station_small.yaml'}');"
            "r=run_event_bench(s, SimConfig(hours=15), VisionNoise(), 4);"
            "print(r['precision'], r['recall'], r['false_alerts'], r['events'])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(h), "PATH": ""}).stdout for h in (1, 2, 3)}
    assert len(outs) == 1 and outs.pop().strip()
