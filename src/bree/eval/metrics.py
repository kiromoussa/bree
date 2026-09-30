"""Event-level benchmark: simulate -> observe with vision noise -> ledger -> score against truth.

Definitions (person level; a "thief" is a true person who left with at least one
item they did not pay for):
  precision        = alerts on thieves / all alerts                   ("alert" tier only)
  recall           = thieves with an alert / all thieves
  false alerts/hr  = alerts on honest people / simulated store-open hours
  review/hr        = "review"-tier flags on honest people / hour (manager queue, nobody confronted)
  basket accuracy  = share of people whose reconciled basket (categories, after put-backs)
                     exactly equals what they really left with; plus item-level P/R
  decision latency = alert emitted - true exit (camera time), p50 / p95
"""
from __future__ import annotations

import time
from dataclasses import replace
from collections import Counter

import numpy as np

from bree.events.zones import StoreConfig
from bree.ledger import build_ledger
from bree.sim.events_sim import SimConfig, VisionNoise, World, build_world, observe


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def evaluate(world: World, obs, ledger, alerts, hours: float) -> dict:
    people = {p.pid: p for p in world.people}
    thieves = {pid for pid, p in people.items() if p.stolen}
    # Map each ledger decision (keyed by an observed id that exited) to true people.
    def truth_of(obs_ids):
        return {obs.exit_owner[i] for i in obs_ids if i in obs.exit_owner}

    def owner_of(alert_pid, group):
        t = truth_of([alert_pid])
        return next(iter(t)) if t else None

    flagged = {"alert": set(), "review": set()}
    fa_alert, fa_review, alert_tp_items, alert_items = 0, 0, 0, 0
    latencies = []
    for a in alerts:
        pid = owner_of(a.person_id, a.group)
        if pid is None:
            # alert on a track we can't map to a real exit (e.g. swapped IDs): it's a false alert
            fa_alert += a.tier == "alert"
            fa_review += a.tier == "review"
            continue
        flagged[a.tier].add(pid)
        true_p = people[pid]
        if pid in thieves:
            latencies.append(a.t_emitted - true_p.t_exit)
            stolen = Counter(i.category for i in true_p.stolen)
            for it in a.unpaid_items:
                alert_items += 1
                if stolen[it.category] > 0:
                    stolen[it.category] -= 1
                    alert_tp_items += 1
        else:
            fa_alert += a.tier == "alert"
            fa_review += a.tier == "review"
            alert_items += len(a.unpaid_items)

    n_alerts = sum(1 for a in alerts if a.tier == "alert")
    tp = len(flagged["alert"] & thieves)
    recall = tp / len(thieves) if thieves else None
    precision = tp / n_alerts if n_alerts else None
    tp_any = len((flagged["alert"] | flagged["review"]) & thieves)

    # Theft recall by type (who gets caught, who doesn't).
    by_kind = {}
    for kind in sorted({people[t].kind for t in thieves}):
        ids = {t for t in thieves if people[t].kind == kind}
        by_kind[kind] = {"thieves": len(ids), "alerted": len(ids & flagged["alert"]),
                         "alert_or_review": len(ids & (flagged["alert"] | flagged["review"]))}
    fa_by_kind = Counter()
    for pid in flagged["alert"] - thieves:
        fa_by_kind[people[pid].kind] += 1

    # Basket accuracy: ledger basket (after put-backs) vs true basket, per true person.
    exact, n_b, item_tp, item_pred, item_true = 0, 0, 0, 0, 0
    for pid, p in people.items():
        recs = [ledger.people[i] for i in obs.id_map[pid] if i in ledger.people]
        if not recs:
            continue
        pred = Counter(it.category for r in recs for it in r.basket)
        true = Counter(i.category for i in p.basket)
        n_b += 1
        exact += pred == true
        item_tp += sum((pred & true).values())
        item_pred += sum(pred.values())
        item_true += sum(true.values())

    honest_people = len(people) - len(thieves)
    return {
        "people": len(people), "thieves": len(thieves), "honest": honest_people,
        "store_hours": hours,
        "alerts": n_alerts,
        "precision": None if precision is None else round(precision, 3),
        "recall": None if recall is None else round(recall, 3),
        "recall_alert_or_review": round(tp_any / len(thieves), 3) if thieves else None,
        "false_alerts": fa_alert,
        "false_alerts_per_hour": round(fa_alert / hours, 4),
        "false_alerts_per_1000_honest_visitors": round(1000 * fa_alert / max(honest_people, 1), 2),
        "reviews_on_honest_per_hour": round(fa_review / hours, 4),
        "flagged_item_precision": round(alert_tp_items / alert_items, 3) if alert_items else None,
        "basket_exact_match": round(exact / n_b, 3) if n_b else None,
        "basket_item_precision": round(item_tp / item_pred, 3) if item_pred else None,
        "basket_item_recall": round(item_tp / item_true, 3) if item_true else None,
        "decision_latency_s_p50": _pct(latencies, 50),
        "decision_latency_s_p95": _pct(latencies, 95),
        "recall_by_theft_type": by_kind,
        "false_alerts_by_shopper_type": dict(fa_by_kind),
    }


def threshold_sweep(world: World, obs, ledger, thresholds=(0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)) -> list[dict]:
    """Precision / recall / false alerts per hour if the alert threshold were set elsewhere."""
    people = {p.pid: p for p in world.people}
    thieves = {pid for pid, p in people.items() if p.stolen}
    rows = []
    for thr in thresholds:
        tp, fp = set(), 0
        for d in ledger.decisions:
            if d["confidence"] < thr:
                continue
            pid = obs.exit_owner.get(d["person_id"])
            if pid in thieves:
                tp.add(pid)
            else:
                fp += 1
        n = len(tp) + fp
        rows.append({"threshold": thr, "precision": round(len(tp) / n, 3) if n else None,
                     "recall": round(len(tp) / len(thieves), 3) if thieves else None,
                     "false_alerts_per_hour": round(fp / world.hours, 4)})
    return rows


def run_event_bench(store: StoreConfig, sim: SimConfig, noise: VisionNoise, seed: int,
                    ledger_overrides: dict | None = None, sweep: bool = False) -> dict:
    t0 = time.perf_counter()
    world = build_world(store.catalog.sku_to_category, sim, seed)
    obs = observe(world, noise, seed + 10_000)
    t1 = time.perf_counter()
    ledger = build_ledger(store, payment_mode=sim.payment_mode, **(ledger_overrides or {}))
    alerts = ledger.replay(obs.events, obs.payments)
    t2 = time.perf_counter()
    out = evaluate(world, obs, ledger, alerts, sim.hours)
    out["sim_notes"] = dict(obs.notes)
    out["crowd_pairs"] = getattr(world, "crowd_pairs", 0)
    out["events"] = len(obs.events)
    out["ledger_events_per_s"] = round(len(obs.events) / max(t2 - t1, 1e-9))
    out["wall_s"] = {"simulate": round(t1 - t0, 2), "ledger": round(t2 - t1, 2)}
    if sweep:
        out["threshold_sweep"] = threshold_sweep(world, obs, ledger)
    return out


NOISE_KNOBS = {  # knob -> fields of VisionNoise it controls
    "missed_picks": ["p_pick_detected", "pick_conf_mean"],
    "false_picks_on_touch": ["p_false_pick_on_touch"],
    "missed_putbacks": ["p_putback_detected"],
    "missed_conceals": ["p_conceal_detected"],
    "false_conceals": ["p_false_conceal"],
    "category_confusion": ["p_category_confusion"],
    "crowd_misattribution": ["p_crowd_candidates", "p_crowd_swap"],
    "held_at_exit_errors": ["p_visible_at_exit", "p_false_held_at_exit"],
    "missed_register_visits": ["p_register_visit_detected"],
    "missed_exits": ["p_exit_detected"],
    "id_switches": ["p_id_switch"],
    "id_swaps": ["p_id_swap"],
    "pos_drops_and_jitter": ["p_pos_dropped", "pos_jitter_s"],
}


def noise_ablation(store: StoreConfig, sim: SimConfig, seed: int, ledger_overrides: dict | None = None) -> list[dict]:
    """Perfect vision + ONE error source at its baseline rate: which errors hurt most?"""
    perfect, base = VisionNoise().scaled(0), VisionNoise()
    rows = []
    ref = run_event_bench(store, sim, perfect, seed, ledger_overrides)
    rows.append({"error_source": "none (perfect vision)", "precision": ref["precision"], "recall": ref["recall"],
                 "false_alerts_per_hour": ref["false_alerts_per_hour"]})
    for knob, fields in NOISE_KNOBS.items():
        n = perfect
        for f in fields:
            n = replace(n, **{f: getattr(base, f)})
        r = run_event_bench(store, sim, n, seed, ledger_overrides)
        rows.append({"error_source": knob, "precision": r["precision"], "recall": r["recall"],
                     "false_alerts_per_hour": r["false_alerts_per_hour"]})
    return rows
