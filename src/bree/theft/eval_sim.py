"""Simulator checks for the theft package (labelled SIMULATED everywhere it is reported).

`posless_bench`: the same simulated store hours as `bree sim`, run with the POS-less ledger (dwell payment mode,
no alert tier) and scored per evidence band: how many review items per hour a reviewer would get, how many of them are
a real simulated thief, and how many of the simulated thieves show up in some band.
"""
from __future__ import annotations

from collections import Counter

from bree.events.zones import StoreConfig
from bree.eval.metrics import apply_retractions
from bree.ledger import build_ledger
from bree.sim.events_sim import SimConfig, VisionNoise, build_world, observe
from bree.theft.tiers import BANDS, POSLESS_LEDGER, band_of


def posless_bench(store: StoreConfig, hours: float = 200.0, seed: int = 2, noise_scale: float = 1.0,
                  ledger_overrides: dict | None = None) -> dict:
    world = build_world(store.catalog.sku_to_category, SimConfig(hours=hours, payment_mode="dwell"), seed)
    obs = observe(world, VisionNoise().scaled(noise_scale), seed + 10_000)
    ledger = build_ledger(store, **{**POSLESS_LEDGER, **(ledger_overrides or {})})
    alerts = apply_retractions(ledger.replay(obs.events, obs.payments))
    people = {p.pid: p for p in world.people}
    thieves = {pid for pid, p in people.items() if p.stolen}
    per = {b: {"items": 0, "thief": 0} for b in BANDS}
    caught: dict[str, set] = {b: set() for b in BANDS}
    for a in alerts:
        b = band_of(a)
        if b is None:
            continue
        pid = obs.exit_owner.get(a.person_id)
        per[b]["items"] += 1
        if pid in thieves:
            per[b]["thief"] += 1
            caught[b].add(pid)
    any_band = set().union(*caught.values())
    cum = set()
    out_bands = {}
    for b in BANDS:                      # cumulative: what a reviewer sees if they work R1, then R1+R2, ...
        cum |= caught[b]
        n = sum(per[k]["items"] for k in BANDS[:BANDS.index(b) + 1])
        t = sum(per[k]["thief"] for k in BANDS[:BANDS.index(b) + 1])
        out_bands[b] = {"review_items": per[b]["items"], "per_hour": round(per[b]["items"] / hours, 3),
                        "precision": round(per[b]["thief"] / per[b]["items"], 3) if per[b]["items"] else None,
                        "cum_per_hour": round(n / hours, 3),
                        "cum_precision": round(t / n, 3) if n else None,
                        "cum_thief_recall": round(len(cum) / len(thieves), 3) if thieves else None}
    kinds = Counter(people[t].kind for t in thieves)
    return {"label": "SIMULATED", "hours": hours, "seed": seed, "noise_scale": noise_scale, "thieves": len(thieves),
            "honest": len(people) - len(thieves), "bands": out_bands,
            "thief_recall_any_band": round(len(any_band) / len(thieves), 3) if thieves else None,
            "thief_kinds": dict(kinds)}
