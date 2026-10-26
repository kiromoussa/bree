"""Operator product: signal contract, metrics, adapter, notifier, digest, dashboard server. All synthetic."""
import datetime as dt
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from bree.events.types import Catalog
from bree.events.zones import StoreConfig, Zone
from bree.product import adapt, app as appmod, digest, metrics, notify, sim
from bree.product.signals import Signal, SignalLog, day_bounds
from bree.product.station import Station, load_station

TZ = "America/Chicago"


def mk(ts, kind, value=1.0, camera="c1", zone=None, **meta):
    return Signal(ts, kind, value, "s", camera, zone, "simulated", meta)


def test_signal_validation_and_roundtrip(tmp_path):
    log = SignalLog(tmp_path, TZ)
    with pytest.raises(ValueError):
        mk(1, "nonsense").validate()
    with pytest.raises(ValueError):
        mk(1, "dwell").validate()                  # needs a zone
    with pytest.raises(ValueError):
        mk(1, "shelf", 0.5, zone="z", state="weird").validate()
    t0 = day_bounds(dt.date(2026, 9, 14), log.tz)[0]
    log.append([mk(t0 + 5, "door_in"), mk(t0 + 86400 + 5, "door_in")])
    assert [p.name for p in sorted(tmp_path.glob("*.jsonl"))] == ["signals-20260914.jsonl", "signals-20260915.jsonl"]
    assert len(log.read(t0, t0 + 86400)) == 1
    with open(tmp_path / "signals-20260914.jsonl", "a") as f:
        f.write('{"v": 1, "ts": \n')                # half-written line is skipped
    assert len(log.read()) == 2
    assert log.latest_ts() == t0 + 86400 + 5


def test_bad_batch_writes_nothing(tmp_path):
    log = SignalLog(tmp_path, TZ)
    with pytest.raises(ValueError):
        log.append([mk(1, "door_in"), mk(2, "bogus")])
    assert log.read() == []


def test_metrics_basics(tmp_path):
    tz = SignalLog(tmp_path, TZ).tz
    t0 = day_bounds(dt.date(2026, 9, 14), tz)[0] + 9 * 3600
    sigs = [mk(t0 + i, "door_in") for i in (1, 2, 3)] + [mk(t0 + 4000, "door_in"), mk(t0 + 5, "door_out")]
    h = metrics.traffic_by_hour(sigs, tz)
    assert [(x["hour"], x["in"], x["out"]) for x in h] == [(9, 3, 1), (10, 1, 0)]
    assert metrics.busiest_hour(h)["hour"] == 9
    assert metrics.busiest_hour([]) is None
    # two cameras: store occupancy is the max, not the sum
    occ = [mk(t0, "occupancy", 2, "a"), mk(t0 + 1, "occupancy", 3, "b")]
    assert metrics.occupancy_series(occ)[0]["n"] == 3
    # queue episodes merge close samples; the far one is its own episode
    q = [mk(t0 + i * 10, "queue", v) for i, v in enumerate([0, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3])]
    eps = metrics.queue_episodes(q)
    assert len(eps) == 2 and eps[0]["peak"] == 3 and eps[0]["start"] == t0 + 10
    assert metrics.queue_summary([])["peak"] is None
    # dwell: p90 needs five visits
    d = [mk(t0 + i, "dwell", 10 + i, zone="z") for i in range(4)]
    assert metrics.dwell_stats(d)["z"]["p90_s"] is None
    assert metrics.dwell_stats(d + [mk(t0, "dwell", 99, zone="z")])["z"]["p90_s"] is not None


def test_shelf_issue_episodes_and_slow_reporting_camera():
    s = [mk(i * 300, "shelf", v, zone="z", state=st) for i, (v, st) in
         enumerate([(0.9, "ok"), (0.1, "low"), (0.1, "low"), (0.9, "ok"), (0.5, "door_open")])]
    eps = metrics.shelf_issues(s)
    assert [(e["state"], e["seconds"]) for e in eps] == [("low", 600.0), ("door_open", 0.0)]
    up = metrics.camera_uptime(s, 0, 1200)
    assert up["c1"]["share"] == 1.0 and up["c1"]["gap_s"] == 300.0     # 5-minute reports are not downtime


def test_notifier_for_s_cooldown_and_reset():
    got = []

    class Keep(notify.Sink):
        def send(self, n):
            got.append(n)
    rule = notify.Rule("q", "queue", ">=", 3, for_s=60, cooldown_s=600, scope="store", message="{value:.0f} waiting")
    n = notify.Notifier([rule], [Keep()])
    n.feed([mk(0, "queue", 3), mk(30, "queue", 3)])
    assert not got                                   # not long enough
    n.feed([mk(70, "queue", 3)])
    assert len(got) == 1 and got[0].message == "3 waiting"
    n.feed([mk(100, "queue", 4)])
    assert len(got) == 1                             # cooldown
    n.feed([mk(120, "queue", 0), mk(800, "queue", 3), mk(870, "queue", 3)])
    assert len(got) == 2                             # dropped below, restarted, fired again


def test_notifier_shelf_state_and_long_gap_resets():
    rule = notify.Rule("open", "shelf", for_s=180, cooldown_s=0, state="door_open")
    n = notify.Notifier([rule], [])
    out = n.feed([mk(0, "shelf", 0.5, zone="z", state="door_open"), mk(100, "shelf", 0.5, zone="z", state="ok"),
                  mk(300, "shelf", 0.5, zone="z", state="door_open"), mk(400, "shelf", 0.5, zone="z", state="door_open")])
    assert out == []                                  # 100 s of open, never 180
    out = n.feed([mk(5000, "shelf", 0.5, zone="z", state="door_open"), mk(5200, "shelf", 0.5, zone="z", state="door_open")])
    assert len(out) == 1 and out[0].message                # the 4600 s silence did not count as "open"


def test_webhook_sink_never_sends(tmp_path):
    sink = notify.WebhookSink("https://example.invalid/hook", tmp_path / "out.jsonl")
    sink.send(notify.Notification(1.0, "r", "info", "hello"))
    rows = [json.loads(x) for x in (tmp_path / "out.webhook.jsonl").read_text().splitlines()]
    assert rows[0]["sent"] is False and rows[0]["would_post_to"].startswith("https://")


def test_broken_sink_does_not_stop_others():
    got = []

    class Bad(notify.Sink):
        def send(self, n):
            raise RuntimeError("boom")

    class Ok(notify.Sink):
        def send(self, n):
            got.append(n)
    n = notify.Notifier([notify.Rule("a", "queue", ">=", 1)], [Bad(), Ok()])
    n.feed([mk(0, "queue", 2)])
    assert len(got) == 1


def _frames(path, tracks):
    """tracks: {id: [(t, x, y, conf), ...]} -> frames.jsonl with small boxes whose foot is (x, y)."""
    by_t = {}
    for tid, pts in tracks.items():
        for t, x, y, c in pts:
            by_t.setdefault(t, []).append({"id": tid, "bbox": [x - 20, y - 80, x + 20, y], "conf": c})
    with open(path, "w") as f:
        for i, t in enumerate(sorted(by_t)):
            f.write(json.dumps({"frame": i, "t": t, "persons": by_t[t], "products": [], "events": []}) + "\n")


def test_adapter_filters_flicker_ghosts_and_roles(tmp_path):
    ts = [i * 0.5 for i in range(200)]                                   # 100 s
    tracks = {
        1: [(t, 100 + t * 4, 400, 0.9) for t in ts[:60]],               # real person walking away from the door
        2: [(t, 500, 300, 0.9) for t in ts[:3]],                        # 1 s flicker
        3: [(t, 700, 300, 0.5) for t in ts],                           # stationary low-confidence ghost
        4: [(t, 600, 300, 0.2) for t in ts[:40]],                       # low confidence
    }
    _frames(tmp_path / "frames.jsonl", tracks)
    door = Zone("door", "exit", np.array([[0, 300], [200, 300], [200, 500], [0, 500]], float))
    sc = StoreConfig("s", "c1", (1000, 600), 2.0, [door], {}, Catalog({}))
    cfg = adapt.AdaptConfig(store="s", camera="c1", role="door")
    sigs, rep = adapt.adapt_run(tmp_path / "frames.jsonl", 1000.0, cfg, sc)
    assert rep["kept"] == 1 and rep["ghost"] == 1 and rep["short"] == 1 and rep["low_conf"] == 1
    occ = {s.ts - 1000.0: s.value for s in sigs if s.kind == "occupancy"}
    assert occ[10.0] == 1 and occ[60.0] == 0
    assert sum(s.kind == "door_in" for s in sigs) == 1
    assert sum(s.kind == "door_out" for s in sigs) == 0                  # ended away from the door
    # an aisle camera must never produce door counts
    sigs, _ = adapt.adapt_run(tmp_path / "frames.jsonl", 1000.0, adapt.AdaptConfig(role="aisle"), sc)
    assert not any(s.kind in ("door_in", "door_out") for s in sigs)
    with pytest.raises(ValueError):
        adapt.adapt_run(tmp_path / "frames.jsonl", 0, adapt.AdaptConfig(role="ceiling"), sc)


def test_segment_start_parses_recorder_name():
    tz = dt.timezone.utc
    assert adapt.segment_start("x/20261025_182851.mp4", tz) == dt.datetime(2026, 10, 25, 18, 28, 51, tzinfo=tz).timestamp()
    with pytest.raises(ValueError):
        adapt.segment_start("clip.mp4")


@pytest.fixture(scope="module")
def demo_log(tmp_path_factory):
    d = tmp_path_factory.mktemp("sig")
    return sim.populate_log(d, dt.date(2026, 9, 14), 8, seed=1, tz=TZ)


def test_sim_is_labelled_and_daily_digest_renders(demo_log):
    tz = demo_log.tz
    day = dt.date(2026, 9, 15)
    a, b = day_bounds(day, tz)
    sigs = demo_log.read(a, b)
    assert {s.source for s in sigs} == {"simulated"}
    rep = metrics.day_report(sigs, a, b, tz)
    html = digest.daily_digest(rep, store_name="T <b>", day=day, tz=tz, tz_assumed=True)
    assert "SIMULATED DATA" in html and "T &lt;b&gt;" in html and "<script" not in html
    assert rep["door"]["in"] > 100 and rep["busiest"]["hour"] in range(24)


def test_empty_day_digest_does_not_crash(tmp_path):
    log = SignalLog(tmp_path, TZ)
    d = dt.date(2026, 9, 15)
    html = digest.daily_digest(metrics.day_report([], *day_bounds(d, log.tz), log.tz), store_name="x", day=d, tz=log.tz)
    assert "No camera data for this day" in html
    days = [(d + dt.timedelta(days=i), metrics.day_report([], *day_bounds(d + dt.timedelta(days=i), log.tz), log.tz)) for i in range(7)]
    assert "No camera data this week" in digest.weekly_digest(days, store_name="x", tz=log.tz)


def test_weekly_digest_compares_with_previous_week(demo_log):
    tz = demo_log.tz
    end = dt.date(2026, 9, 21)
    days = []
    for i in range(6, -1, -1):
        d = end - dt.timedelta(days=i)
        a, b = day_bounds(d, tz)
        days.append((d, metrics.day_report(demo_log.read(a, b), a, b, tz)))
    html = digest.weekly_digest(days, store_name="W", tz=tz, prev_total_in=100)
    assert "than the week before" in html and "Busiest day" in html


def _get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


@pytest.fixture()
def server(demo_log, tmp_path):
    from bree.review.store import ReviewStore
    rs = ReviewStore(tmp_path / "review")
    rs.add_alert({"alert_id": "A1", "tier": "alert", "confidence": 0.8, "unpaid_items": [{"category": "chips"}],
                  "reasons": ["chips not paid"]}, camera="c1", event_ts=demo_log.latest_ts() - 100)
    app = appmod.ProductApp(demo_log, Station(name="Test", tz=TZ), rs, notif_file=tmp_path / "n.jsonl", replay=True)
    srv = appmod.serve(app, "127.0.0.1", 0, None, monitor_every_s=None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app, rs
    srv.shutdown()


def test_server_api_and_pages(server):
    base, app, rs = server
    code, body, h = _get(base + "/")
    assert code == 200 and b"BREE store view" in body and "default-src 'self'" in h["Content-Security-Policy"]
    s = json.loads(_get(base + "/api/state")[1])
    assert s["replay"] and list(s["sources"]) == ["simulated"] and s["pending_review"] == 1
    assert json.loads(_get(base + "/api/days")[1])
    d = json.loads(_get(base + "/api/day?date=2026-09-15")[1])
    assert d["door"]["in"] > 0 and "shelf_now" not in d
    assert _get(base + "/api/day?date=garbage")[0] == 400
    assert b"Daily digest" in _get(base + "/digest/daily?date=2026-09-15")[1]
    assert b"Weekly digest" in _get(base + "/digest/weekly?end=2026-09-21")[1]
    assert _get(base + "/clips/A1")[0] == 404          # no clip stored
    assert _get(base + "/nope")[0] == 404


def test_review_decision_flow_and_csrf(server):
    base, app, rs = server
    post = lambda body, ct="application/json": _get(base + "/api/review/decide", json.dumps(body).encode(), {"Content-Type": ct})
    assert post({"id": "A1", "decision": "not_theft", "reviewer": "sam"}, "text/plain")[0] == 415
    assert post({"id": "A1", "decision": "bogus", "reviewer": "sam"})[0] == 400
    assert post({"id": "A1", "decision": "not_theft", "reviewer": ""})[0] == 400
    assert post({"id": "nope", "decision": "not_theft", "reviewer": "sam"})[0] == 400
    assert post({"id": "A1", "decision": "confirmed_theft", "reviewer": "sam"})[0] == 200
    r = json.loads(_get(base + "/api/review")[1])
    assert r["pending"] == [] and r["recent"][0]["decision"] == "confirmed_theft"
    assert json.loads(_get(base + "/api/state")[1])["pending_review"] == 0


def test_token_required_off_localhost(demo_log):
    app = appmod.ProductApp(demo_log, Station(tz=TZ), None, replay=True)
    with pytest.raises(ValueError):
        appmod.serve(app, "0.0.0.0", 0, None, monitor_every_s=None)
    srv = appmod.serve(app, "127.0.0.1", 0, "s3cret", monitor_every_s=None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert _get(base + "/api/state")[0] == 401
        assert _get(base + "/api/state", headers={"Authorization": "Bearer wrong"})[0] == 401
        assert _get(base + "/api/state", headers={"Authorization": "Bearer s3cret"})[0] == 200
        code, _, h = _get(base + "/?token=s3cret")
        assert code == 200 and "bree_token=s3cret" in h["Set-Cookie"]
    finally:
        srv.shutdown()


def test_monitor_notifies_new_alert_once_and_not_history(demo_log, tmp_path):
    from bree.review.store import ReviewStore
    rs = ReviewStore(tmp_path / "r")
    rs.add_alert({"alert_id": "OLD", "tier": "alert", "confidence": 0.9, "unpaid_items": []}, camera="c1", event_ts=1.0)
    app = appmod.ProductApp(demo_log, Station(tz=TZ), rs, notif_file=tmp_path / "n.jsonl", replay=True)
    assert app.monitor.tick() == []                  # history is not announced
    rs.add_alert({"alert_id": "NEW", "tier": "alert", "confidence": 0.9, "unpaid_items": []}, camera="c1", event_ts=2.0)
    fired = app.monitor.tick()
    assert [f.rule for f in fired] == ["basket_flagged"] and app.monitor.tick() == []
    assert len(app.api_notifications()) == 1


def test_station_yaml_rule_override(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("name: X\ntz: America/Chicago\ncameras: {c1: {label: Door, role: door}}\nrules:\n- {name: long_queue, threshold: 5}\n")
    st = load_station(p)
    assert st.label("c1") == "Door" and st.label("zz") == "zz"
    r = next(r for r in st.rules if r.name == "long_queue")
    assert r.threshold == 5 and r.for_s == 60      # other fields kept
