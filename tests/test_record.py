"""Segment recorder against the mock platform: finished files only, status, circuit breaker on stub streams."""
from __future__ import annotations

import json
import threading
import time

import pytest

from bree.ingest.imou import ImouClient, ImouCredentials
from bree.ingest.imou_mock import APP_ID, APP_SECRET, MockImou
from bree.ingest.record import CircuitBreaker, Recorder, probe_duration


@pytest.fixture
def mock():
    with MockImou(channels={0: "ok", 1: "ok", 2: "stub"}, live_seconds=10) as m:
        yield m


def _rec(mock, tmp_path, channels, **kw):
    c = ImouClient(ImouCredentials(APP_ID, APP_SECRET, "or"), base_url=mock.base_url)
    return Recorder(c, mock.serial, "st", channels, tmp_path / "real" / "st", tmp_path / "status.json", seg_seconds=8, stream="sub",
                    log=lambda m: None, min_bytes=2000, **kw)


def test_records_a_finished_segment_and_never_leaves_part_files(mock, tmp_path):
    r = _rec(mock, tmp_path, [0])
    name = r.record_one(0)
    f = tmp_path / "real" / "st" / "ch00" / name
    assert f.exists() and 6.5 <= probe_duration(f) <= 9
    assert not list((tmp_path / "real" / "st").glob("ch*/*.part"))
    assert (tmp_path / "real" / "st" / "ch00").stat().st_mode & 0o077 == 0


def test_stub_segment_is_rejected_and_cleaned_up(mock, tmp_path):
    from bree.ingest.imou import StreamStaleError
    r = _rec(mock, tmp_path, [2])
    with pytest.raises(StreamStaleError):
        r.record_one(2)
    assert not list((tmp_path / "real" / "st").glob("ch*/*"))


def test_breaker_trips_pauses_everyone_and_backs_off():
    now = [1000.0]
    b = CircuitBreaker(max_stubs=3, cooldown_s=100, max_cooldown_s=400, clock=lambda: now[0])
    for _ in range(3):
        b.record(False)
    assert b.open and 99 <= b.wait_s() <= 100
    now[0] += 101
    assert not b.open
    b.record(False)                      # the probe is a stub again: longer pause
    assert b.open and 199 <= b.wait_s() <= 200
    now[0] += 201
    b.record(True)                       # a good segment resets everything
    assert not b.open and b.trips == 0 and b.stubs == 0


def test_run_loop_trips_on_a_stub_channel_and_stops_fetching(mock, tmp_path):
    r = _rec(mock, tmp_path, [2], hours=0.01, breaker=CircuitBreaker(max_stubs=2, cooldown_s=600))
    import bree.ingest.record as rec
    th = threading.Thread(target=r.run)
    th.start()
    t0 = time.time()
    while time.time() - t0 < 40 and not r.breaker.open:
        time.sleep(0.5)
    n = mock.methods().count("getLiveStreamInfo")
    time.sleep(4)
    assert r.breaker.open and mock.methods().count("getLiveStreamInfo") == n          # paused: no more API traffic
    r.stop.set()
    th.join(30)
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["state"] == "stopped" and st["breaker_trips"] >= 1 and st["channels"]["2"]["stubs"] >= 2
    assert "mock-secret" not in json.dumps(st) and "http" not in json.dumps(st)
