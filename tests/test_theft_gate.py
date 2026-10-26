"""Observation gate: flicker, ghosts on fixtures and staff never reach the event engine."""
from __future__ import annotations

import numpy as np

from bree.events.observations import FrameObs, PersonObs, ProductObs
from bree.events.zones import Zone
from bree.theft.gate import GateConfig, ObsGate


def person(i, x, y, conf=0.8, w=80, h=200):
    return PersonObs(i, (x - w / 2, y - h, x + w / 2, y), conf)


def run(gate, tracks, seconds, fps=10.0):
    """tracks: fn(t) -> list[PersonObs]. Returns the ids the gate passed at each frame."""
    seen = []
    for k in range(int(seconds * fps)):
        t = k / fps
        seen.append({p.track_id for p in gate.update(FrameObs(k, t, tracks(t))).persons})
    return seen


def test_flicker_never_reaches_the_engine_but_a_person_does():
    g = ObsGate(GateConfig())
    # id 1 lives 0.5 s at conf 0.35 (flicker); id 2 stays 4 s at conf 0.8
    seen = run(g, lambda t: ([person(1, 500, 400, 0.35)] if t < 0.5 else []) + ([person(2, 900, 700)] if t >= 0.5 else []), 5)
    assert not any(1 in s for s in seen)
    assert 2 in seen[-1] and 2 not in seen[5]          # held back for min_age_s, then passed
    assert g.stats["tracks_raw"] == 2 and g.stats["tracks_passed"] == 1


def test_low_confidence_track_never_passes_however_long():
    g = ObsGate(GateConfig(static_s=1000))
    seen = run(g, lambda t: [person(1, 500 + 300 * (t % 4) / 4, 400, 0.38)], 10)
    assert not any(seen)


def test_static_low_conf_box_is_a_ghost_but_a_still_confident_person_is_not():
    g = ObsGate(GateConfig())
    seen = run(g, lambda t: [person(1, 300, 300, 0.5), person(2, 800, 600, 0.9)], 12)
    assert 2 in seen[-1]
    assert 1 not in seen[-1] and g.stats["hidden_static_ghost"] == 1


def test_staff_zone_hides_staff_only():
    staff = Zone("behind_counter", "shelf", np.array([[1000, 0], [1400, 0], [1400, 900], [1000, 900]], float))
    g = ObsGate(GateConfig(), staff=[staff])
    seen = run(g, lambda t: [person(1, 1200, 600), person(2, 600, 600 + 10 * np.sin(t))], 5)
    assert 2 in seen[-1] and 1 not in seen[-1]
    assert g.stats["hidden_staff"] == 1


def test_ignore_zone_and_disabled_gate():
    ign = Zone("poster", "shelf", np.array([[0, 0], [400, 0], [400, 400], [0, 400]], float))
    g = ObsGate(GateConfig(), ignore=[ign])
    assert not any(run(g, lambda t: [person(1, 200, 300, 0.9)], 3))
    off = ObsGate(GateConfig(enabled=False))
    assert all(1 in s for s in run(off, lambda t: [person(1, 200, 300, 0.1)], 1))


def test_static_product_is_dropped_but_moving_one_is_kept():
    g = ObsGate(GateConfig())
    kept = []
    for k in range(60):
        t = k / 10
        obs = FrameObs(k, t, [], [ProductObs(1, (100, 100, 140, 160), "phone_accessory", 0.9),
                                  ProductObs(2, (200 + 20 * t, 300, 240 + 20 * t, 360), "soda_bottle", 0.7)])
        kept.append({q.track_id for q in g.update(obs).products})
    assert 1 not in kept[-1] and 2 in kept[-1] and g.stats["products_static"] == 1
