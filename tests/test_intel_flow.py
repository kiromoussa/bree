"""People-flow analytics on synthetic detections (SIMULATED: boxes are generated, no model, no video)."""
from datetime import datetime
from pathlib import Path

import numpy as np

from bree.intel import aggregate as ag
from bree.intel import evaluate as ev
from bree.intel.flow import FlowEngine, IntelConfig, run_detections
from bree.intel.flowtrack import FlowConfig, FlowTracker
from bree.intel.layout import load_layout

LAYOUT = Path(__file__).resolve().parents[1] / "configs" / "intel_trumanrd_ch07.yaml"
FPS = 6.0


def box(x, y, h=420.0, conf=0.9):
    """A person box whose foot (bottom centre) is at (x, y), in 1280x720 pixels."""
    return {"bbox": [x - h * 0.2, y - h, x + h * 0.2, y], "conf": conf}


def walk(path, t0=0.0, conf=0.9, h=420.0):
    """frames [(t, [box])] for a person following straight segments between waypoints [(x, y, seconds)]."""
    frames = []
    for (xa, ya, _), (xb, yb, dur) in zip(path[:-1], path[1:]):
        n = max(int(dur * FPS), 1)
        for k in range(n):
            f = k / n
            frames.append((xa + (xb - xa) * f, ya + (yb - ya) * f))
    return [(t0 + i / FPS, [box(x, y, h, conf)]) for i, (x, y) in enumerate(frames)]


def merge(*tracks):
    ts = {}
    for tr in tracks:
        for t, bs in tr:
            ts.setdefault(round(t, 3), []).extend(bs)
    return [{"t": t, "persons": ts[t]} for t in sorted(ts)]


def engine_run(frames, **kw):
    lay = load_layout(LAYOUT, (1280, 720))
    return run_detections(frames, lay, IntelConfig(**kw), datetime(2026, 10, 25, 16, 0, 0))


def door(recs):
    return [(r["dir"], r["via"]) for r in recs if r["type"] == "door"]


def test_layout_scales_to_video_size():
    a, b = load_layout(LAYOUT, (1280, 720)), load_layout(LAYOUT, (2560, 1440))
    assert np.allclose(b.door.a, a.door.a * 2)
    assert a.door.signed(*a.door.inside) > 0
    assert any(z.kind == "register" for z in a.zones)
    assert a.to_plan(*a.door.a) is not None


def test_entry_then_exit_counts_and_occupancy():
    enter = walk([(370, 150, 0), (370, 330, 2.0), (370, 560, 2.0)], 0.0)
    pause = [(4.0 + i / FPS, [box(370, 560)]) for i in range(1, 7)]
    leave = walk([(370, 560, 0), (370, 330, 2.0), (370, 150, 2.0)], 5.0)
    recs, eng = engine_run(merge(enter, pause, leave))
    assert door(recs) == [("in", "line"), ("out", "line")]
    assert eng.entries == 1 and eng.exits == 1
    assert eng.occupancy == 0
    tr = [r for r in recs if r["type"] == "track"]
    assert len(tr) == 1 and tr[0]["entered"] and tr[0]["exited"] and tr[0]["role"] == "customer"


def test_pacing_on_the_line_is_not_a_crossing():
    # jitter of +-8 px around the line for 10 s: inside the margin, no event
    frames = [(i / FPS, [box(370, 332 + (8 if i % 2 else -8))]) for i in range(60)]
    recs, eng = engine_run(merge(frames))
    assert door(recs) == []


def test_single_frame_jump_across_the_line_is_not_a_crossing():
    frames = [(i / FPS, [box(370, 520)]) for i in range(30)]
    frames[10] = (10 / FPS, [box(370, 200)])          # one wild box (another person, partial body)
    recs, eng = engine_run(merge(frames))
    assert door(recs) == []


def test_flicker_ghost_never_becomes_a_track():
    frames = [{"t": i / FPS, "persons": [box(900, 500, conf=0.34)] if i % 3 == 0 else []} for i in range(60)]
    recs, eng = engine_run(frames)
    assert not [r for r in recs if r["type"] == "track"]


def test_track_survives_a_short_occlusion():
    path = walk([(300, 500, 0), (700, 600, 4.0)])
    path = [f for f in path if not (2.0 < f[0] < 3.2)]    # hidden behind a rack for 1.2 s
    tk = FlowTracker(FlowConfig())
    for t, bs in path:
        tk.step(t, bs)
    assert len(tk.finish()) == 1


def test_staff_behind_counter_is_not_a_customer():
    staff = [(i / FPS, [box(1075, 450, h=300, conf=0.45)]) for i in range(60)]
    recs, eng = engine_run(merge(staff))
    tr = [r for r in recs if r["type"] == "track"]
    assert len(tr) == 1 and tr[0]["role"] == "staff" and tr[0]["outcome"] is None
    s = [r for r in recs if r["type"] == "sample"]
    assert max(r["customers"] for r in s) == 0 and max(r["staff"] for r in s) == 1


def test_register_visit_and_queue():
    reg = (760, 600)
    a = [(i / FPS, [box(*reg)]) for i in range(60)]                          # stands 10 s at the till
    b = [(i / FPS, [box(reg[0] - 40, reg[1] + 60)]) for i in range(20, 60)]  # second person joins
    recs, eng = engine_run(merge(a, b))
    z = [r for r in recs if r["type"] == "zone" and r["kind"] == "register"]
    assert z and max(r["still_s"] for r in z) > 5
    assert max(r["queue"] for r in recs if r["type"] == "sample") >= 2
    outcomes = {r["outcome"] for r in recs if r["type"] == "track"}
    assert "registered" in outcomes


def test_browse_without_register_then_exit_is_left_without_register():
    in_ = walk([(370, 150, 0), (370, 330, 2.0), (370, 480, 1.0), (120, 560, 2.0)], 0.0)
    stand = [(5.0 + i / FPS, [box(120, 560)]) for i in range(1, 37)]
    back = walk([(120, 560, 0), (370, 480, 2.0), (370, 330, 1.0), (370, 150, 2.0)], 11.0)
    recs, eng = engine_run(merge(in_, stand, back))
    tr = [r for r in recs if r["type"] == "track"]
    assert tr[0]["outcome"] == "left_without_register" and tr[0]["browse_s"] >= 4


def test_hourly_and_daily_aggregates():
    enter = walk([(370, 150, 0), (370, 330, 2.0), (370, 560, 2.0)])
    recs, eng = engine_run(merge(enter))
    recs = [{"type": "segment", "video": "x.mp4", "t0": 0.0, "t1": 300.0, "ts": "2026-10-25T16:00:00"}] + recs
    h = ag.hourly(recs)
    assert list(h) == ["2026-10-25T16:00"]
    assert h["2026-10-25T16:00"]["entries"] == 1 and h["2026-10-25T16:00"]["coverage_s"] == 300.0
    assert h["2026-10-25T16:00"]["entries_per_hour"] == 12.0
    d = ag.daily(h)
    assert d["2026-10-25"]["entries"] == 1
    assert "1 entries" in ag.digest(d["2026-10-25"])


def test_event_matching_is_one_to_one_and_directional():
    sys_ev = [{"t": 10.0, "dir": "in", "via": "line"}, {"t": 11.0, "dir": "in", "via": "line"}, {"t": 30.0, "dir": "out", "via": "lost"}]
    lab = [{"t": 10.5, "dir": "in"}, {"t": 30.5, "dir": "in"}]
    tp, fp, fn = ev.match(sys_ev, lab, 4.0)
    assert len(tp) == 1 and len(fp) == 2 and len(fn) == 1


def test_score_splits():
    recs = {"s1": [{"type": "door", "dir": "in", "t": 5.0, "via": "line"}]}
    labels = {"segments": {"s1": {"split": "dev", "events": [{"t": 5.5, "dir": "in"}, {"t": 50.0, "dir": "out"}]}}}
    r = ev.score(recs, labels)
    assert r["dev"]["true_pos"] == 1 and r["dev"]["missed"] == 1 and r["dev"]["recall"] == 0.5


def _samples(qs, staff=1, dt=5.0):
    return [{"type": "sample", "t": i * dt, "ts": None, "in_view": q + staff, "customers": q, "staff": staff, "queue": q, "occupancy": None}
            for i, q in enumerate(qs)]


def test_rule_anomalies():
    from bree.intel.anomaly import rule_anomalies
    recs = _samples([0, 3, 3, 3, 3, 3, 3, 3, 0])                      # 3+ in the queue for 30 s
    kinds = [a["kind"] for a in rule_anomalies(recs)]
    assert "long_queue" in kinds and "unattended_counter" not in kinds
    recs = _samples([1] * 20, staff=0)                                # 95 s of a customer with nobody behind the counter
    from bree.intel.anomaly import RuleConfig
    assert "unattended_counter" not in [a["kind"] for a in rule_anomalies(recs)]          # off by default
    assert "unattended_counter" in [a["kind"] for a in rule_anomalies(recs, RuleConfig(unattended_s=60.0))]
    recs = [{"type": "door", "dir": "in", "t": float(i), "via": "line"} for i in range(8)]
    assert "door_imbalance" in [a["kind"] for a in rule_anomalies(recs)]
    recs = [{"type": "zone", "kind": "browse", "zone": "snack_rack", "still_s": 400.0, "t_in": 5.0, "ts": None}]
    assert "loitering" in [a["kind"] for a in rule_anomalies(recs)]


def test_baseline_needs_history_and_flags_outliers():
    from bree.intel.anomaly import Baseline

    def hour(day, v):
        return {"hour": f"2026-09-{day:02d}T16:00", "coverage_s": 3600.0, "entries_per_hour": v, "queue_mean": None, "register_time_median_s": None}
    mondays = [hour(d, v) for d, v in zip((7, 14, 21, 28), (20, 22, 19, 21))]      # four Mondays
    base = Baseline.fit(mondays)
    assert base.score({"hour": "2026-10-05T16:00", "coverage_s": 3600.0, "entries_per_hour": 21, "queue_mean": None, "register_time_median_s": None}) == []
    hi = base.score({"hour": "2026-10-05T16:00", "coverage_s": 3600.0, "entries_per_hour": 60, "queue_mean": None, "register_time_median_s": None})
    assert hi and hi[0]["kind"] == "pattern_high"
    assert base.score({"hour": "2026-10-06T16:00", "coverage_s": 3600.0, "entries_per_hour": 60, "queue_mean": None, "register_time_median_s": None}) == []   # a Tuesday: no history
    assert Baseline.fit(mondays[:2]).score({"hour": "2026-10-05T16:00", "coverage_s": 3600.0, "entries_per_hour": 60, "queue_mean": None, "register_time_median_s": None}) == []
    assert base.score({"hour": "2026-10-05T16:00", "coverage_s": 60.0, "entries_per_hour": 60, "queue_mean": None, "register_time_median_s": None}) == []
