"""POS-less tiers: the ledger never raises an alert, and each review item lands in an evidence band."""
from __future__ import annotations

from bree.events.types import Catalog, Event, EventType as E
from bree.ledger.ledger import Ledger, LedgerConfig
from bree.theft.tiers import BANDS, POSLESS_LEDGER, band_of
from bree.theft.eval_sim import posless_bench

CATALOG = Catalog({"RB": "energy", "CH": "chips"})
KINDS = {"register": "register", "cooler": "cooler", "door": "exit"}


def ledger(**cfg) -> Ledger:
    return Ledger(CATALOG, LedgerConfig(**{**POSLESS_LEDGER, **cfg}), {"pos_1": "register"}, KINDS)


def pick(t, p, item):
    return Event(E.PICK, t, p, item=item, zone="cooler", confidence=0.9)


def leave(t, p, held=()):
    return Event(E.EXIT, t, p, zone="door", meta={"held_items": list(held)})


def visit(t0, t1, p):
    return [Event(E.PAY, t0 + 2.5, p, zone="register", meta={"phase": "start", "t_start": t0}),
            Event(E.PAY, t1, p, zone="register", meta={"phase": "end", "t_start": t0, "t_end": t1})]


def test_walkout_with_concealment_is_review_not_alert():
    L = ledger()
    out = L.replay([Event(E.ENTER, 0, 1), pick(5, 1, "energy"), Event(E.CONCEAL, 8, 1, item="energy", confidence=0.8),
                    leave(20, 1)])
    assert [a.tier for a in out] == ["review"]
    assert band_of(out[0]) == "R1"
    assert any("never raises an alert" in r for r in out[0].reasons)


def test_same_story_alerts_with_a_pos_ledger():
    L = Ledger(CATALOG, LedgerConfig(), {"pos_1": "register"}, KINDS)
    out = L.replay([Event(E.ENTER, 0, 1), pick(5, 1, "energy"), Event(E.CONCEAL, 8, 1, item="energy", confidence=0.8),
                    leave(20, 1)])
    assert out[0].tier == "alert"


def test_carried_out_is_r2_and_plain_pick_r3():
    L = ledger()
    a = L.replay([Event(E.ENTER, 0, 1), pick(5, 1, "energy"), leave(20, 1, ["energy"])])
    assert band_of(a[0]) == "R2"
    L = ledger()
    b = L.replay([Event(E.ENTER, 0, 2), pick(5, 2, "chips"), leave(20, 2)])
    assert b and band_of(b[0]) == "R3"


def test_register_visit_covers_picks_in_dwell_mode():
    L = ledger()
    assert L.replay([Event(E.ENTER, 0, 1), pick(5, 1, "energy"), *visit(20, 30, 1), leave(35, 1, ["energy"])]) == []


def test_bands_are_ordered_and_bench_runs_small():
    assert BANDS == ("R1", "R2", "R3")
    from bree.events.zones import load_store_config
    from pathlib import Path
    store = load_store_config(Path(__file__).resolve().parents[1] / "configs" / "store_gas_station_small.yaml")
    r = posless_bench(store, hours=10, seed=3)
    assert r["label"] == "SIMULATED" and set(r["bands"]) == set(BANDS)
    assert all(b["review_items"] >= 0 for b in r["bands"].values())
