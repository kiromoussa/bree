"""Imou client + live source against the local mock platform (no network, no credentials)."""
from __future__ import annotations

import json
import pytest

from bree.ingest import imou
from bree.ingest.imou import (DeviceOfflineError, HealthSink, ImouClient, ImouConfigError, ImouCredentials, ImouSource,
                              StationClock, StampedFrame, StreamStaleError, StreamUnavailable, parse_imou_url)
from bree.ingest.imou_mock import APP_ID, APP_SECRET, MockImou

CREDS = ImouCredentials(APP_ID, APP_SECRET, "or")


@pytest.fixture(scope="module")
def mock():
    with MockImou(channels={0: "ok", 1: "ok", 5: "offline", 7: "stub"}, live_seconds=6) as m:
        yield m


@pytest.fixture
def client(mock):
    mock.quota_exhausted, mock.device_online = False, True
    mock.set_channel(7, "stub")
    mock.log.clear()
    return ImouClient(CREDS, base_url=mock.base_url)


def test_signing_matches_the_platform_test_vector():
    got = imou.sign("test123456789test123456789", 1706511734, "f5a1ae2d-c09c-4d39-a744-83a5c2c653c2")
    assert got == "xjhCQBoJ9hRDsCjyDcHjtDNzRZ3ZJezcawsfWeiaoxU="


def test_url_parsing_and_rejections():
    t = parse_imou_url("imou://AH040CBPAZ893C5/7?stream=main&clock_offset=-7200")
    assert (t.serial, t.channel, t.stream, t.params["clock_offset"]) == ("AH040CBPAZ893C5", 7, "main", "-7200")
    assert parse_imou_url("imou://ABC123/0").stream == "sub"
    for bad in ("imou://ABC123", "imou://ABC123/x", "imou:///3", "imou://ABC123/1?stream=hd", "imou://user:pw@ABC123/1", "rtsp://a/1"):
        with pytest.raises(ImouConfigError):
            parse_imou_url(bad)


def test_credentials_repr_hides_secret_and_missing_is_clear(tmp_path):
    assert "mock-secret" not in repr(CREDS) and "mock-secret" not in str(CREDS)
    with pytest.raises(ImouConfigError, match="IMOU_APP_ID"):
        imou.load_credentials({}, env_file=tmp_path / "none.env")
    f = tmp_path / "x.env"
    f.write_text("# c\nIMOU_APP_ID=a\nIMOU_APP_SECRET='s'\nIMOU_DC=fk\n")
    c = imou.load_credentials({}, env_file=f)
    assert (c.app_id, c.app_secret, c.dc) == ("a", "s", "fk")


def test_only_read_only_methods_are_allowed(client, mock):
    for m in ("unbindDevice", "deleteDevice", "controlPTZ", "bindDevice", "modifyDeviceName"):
        with pytest.raises(ImouConfigError):
            client.call(m, {})
    assert mock.log == []                       # refused before anything went on the wire
    with pytest.raises(ImouConfigError):
        client.bind_device("MOCKSERIAL001", "x")


def test_bad_secret_is_rejected_and_not_echoed(mock):
    bad = ImouClient(ImouCredentials(APP_ID, "wrong-secret-xyz", "or"), base_url=mock.base_url)
    with pytest.raises(imou.ImouAuthError) as e:
        bad.token()
    assert "wrong-secret-xyz" not in str(e.value)


def test_device_online_lists_channels(client, mock):
    st = client.device_online(mock.serial)
    assert st.online and st.channels == {0: True, 1: True, 5: False, 7: True}
    mock.device_online = False
    assert not client.device_online(mock.serial).online


def test_token_is_cached_and_renewed_when_the_platform_expires_it(client, mock):
    client.device_online(mock.serial)
    client.device_online(mock.serial)
    assert mock.methods().count("accessToken") == 1
    mock.expire_token()
    assert client.device_online(mock.serial).online              # TK1002 -> renew -> retry, invisible to the caller
    assert mock.methods().count("accessToken") == 2


def test_bind_needs_explicit_opt_in_and_right_code(mock):
    c = ImouClient(CREDS, base_url=mock.base_url, allow_bind=True)
    with pytest.raises(imou.ImouError) as e:
        c.bind_device(mock.serial, "wrong")
    assert e.value.code == "DV1007"
    c.bind_device(mock.serial, mock.device_code)


def test_live_url_picks_requested_stream_and_scrubs_addresses(client, mock):
    u = client.live_url(mock.serial, 0, "sub")
    assert u.endswith("/live/index.m3u8")
    assert imou.scrub(f"failed to open {u} now") == "failed to open <url> now"
    mock.quota_exhausted = True
    assert client.live_url(mock.serial, 0, "main").endswith("/stub/index.m3u8")   # the platform does not tell us: the source must notice


def test_source_streams_stamped_frames_and_reports_health(client, mock, tmp_path):
    hp = tmp_path / "h.json"
    health = HealthSink(hp, "st", mock.serial, 0, every_s=0)
    src = ImouSource(f"imou://{mock.serial}/0?clock_offset=-7200", client=client, health=health, max_frames=25)
    assert (src.width, src.height) == (320, 240) and src.live and abs(src.fps - 10) < 1
    frames = list(src)
    assert len(frames) == 25 and isinstance(frames[0], StampedFrame)
    assert frames[0].image.shape == (240, 320, 3)
    assert frames[-1].t > frames[0].t
    assert abs((frames[0].station_ts - frames[0].wall) + 7200) < 1e-6      # clock offset applied
    h = json.loads(hp.read_text())
    assert h["state"] == "stopped" and h["frames"] == 25 and h["clock_offset_s"] == -7200.0
    assert "mock-secret" not in hp.read_text() and "http" not in hp.read_text()


def test_source_reconnects_with_a_fresh_address_when_the_stream_ends(client, mock):
    # live clip is 6 s = 60 frames; asking for 90 forces one re-open (and so one more live_url call)
    src = ImouSource(f"imou://{mock.serial}/0", client=client, max_frames=90, min_run_s=2, sleep=lambda s: None)
    n = sum(1 for _ in src)
    assert n == 90
    assert mock.methods().count("getLiveStreamInfo") >= 2


def test_offline_device_and_channel_fail_at_startup(client, mock):
    with pytest.raises(DeviceOfflineError, match="channel 5"):
        ImouSource(f"imou://{mock.serial}/5", client=client)
    mock.device_online = False
    with pytest.raises(DeviceOfflineError, match="offline"):
        ImouSource(f"imou://{mock.serial}/0", client=client)


def test_stub_stream_trips_the_circuit_breaker_and_stops_calling_the_api(client, mock, tmp_path):
    health = HealthSink(tmp_path / "h.json", "st", mock.serial, 7, every_s=0)
    src = ImouSource(f"imou://{mock.serial}/7", client=client, health=health, max_stub_retries=3, min_run_s=8,
                     sleep=lambda s: None)
    mock.log.clear()
    with pytest.raises(StreamStaleError, match="Not retrying"):
        for _ in src:
            pass
    assert mock.methods().count("getLiveStreamInfo") <= 3        # bounded: never a retry storm on a metered API
    h = json.loads((tmp_path / "h.json").read_text())
    assert h["state"] == "stale_stream" and h["stub_opens"] >= 3 and "http" not in h["last_error"]


def test_quota_exhaustion_from_the_platform_is_a_stale_error_not_a_retry_loop(client, mock):
    # A platform that answers the API call itself with the flow-allowance code.
    orig = mock._handle
    mock._handle = lambda m, b: {"code": "11003", "msg": "Insufficient flow resources"} if m == "getLiveStreamInfo" else orig(m, b)
    try:
        with pytest.raises(StreamStaleError) as e:
            ImouSource(f"imou://{mock.serial}/0", client=client)
        assert e.value.code == "11003"
    finally:
        mock._handle = orig


def test_unreachable_platform_is_a_clean_unavailable_error():
    c = ImouClient(CREDS, base_url="http://127.0.0.1:9", timeout=1)
    with pytest.raises(StreamUnavailable) as e:
        c.token()
    assert "mock-secret" not in str(e.value)


def test_open_source_dispatches_imou_urls(mock, monkeypatch):
    from bree.ingest.source import open_source
    monkeypatch.setattr(imou, "load_credentials", lambda *a, **k: ImouCredentials(APP_ID, APP_SECRET, "or", mock.base_url))
    s = open_source(f"imou://{mock.serial}/1?stream=main", max_frames=3)
    assert isinstance(s, ImouSource) and len(list(s)) == 3
