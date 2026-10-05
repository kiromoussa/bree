"""Camera node first pass: trigger, pre-roll bursts, wire format, spool, node -> hub over real HTTP.
Everything here is synthetic drawings; nothing needs a Pi, a camera or model weights."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from bree.edge.capture import BurstRecorder, FileSource
from bree.edge.hub import Hub
from bree.edge.node import build, load_node_config, pack_burst
from bree.edge.trigger import TriggerConfig, ZoneTrigger
from bree.edge.uplink import Spool, Uplink, pack, unpack

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 640, 480, 10
ZONE = [[200, 40], [600, 40], [600, 200], [200, 200]]
SKIN = (120, 150, 210)   # BGR


def frame(hand: tuple[int, int] | None = None, gain: float = 1.0, seed: int = 0, color=SKIN, r=18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 110, np.float32)
    img[40:200, 200:600] = 150                               # the shelf
    img += rng.normal(0, 2, img.shape)
    if hand:
        cv2.circle(img, hand, r, color, -1)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def lores(img):
    return cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)


def hand_path(i: int) -> tuple[int, int] | None:
    """Frames 0-29 empty, 30-49 a hand goes into the shelf and back out, then empty."""
    if not 30 <= i < 50:
        return None
    k = 1 - abs((i - 30) / 10 - 1)                           # 0 -> 1 -> 0
    return 400, int(300 - 200 * k)


def trig(**kw):
    return ZoneTrigger({"shelf": np.array(ZONE)}, (W, H), TriggerConfig(**kw))


# ------------------------------------------------------------------ trigger
def test_hand_entering_the_zone_triggers_and_releases():
    tr = trig()
    on = [bool(tr.update(lores(frame(hand_path(i), seed=i)), i / FPS)) for i in range(80)]
    assert not any(on[:30])                                  # sensor noise alone never fires
    assert any(on[34:46])                                    # hand inside the shelf polygon
    assert not any(on[60:])                                  # released, and no ghost left in the background


def test_a_dark_glove_triggers_too():
    tr = trig()
    on = [bool(tr.update(lores(frame(hand_path(i), seed=i, color=(30, 30, 30))), i / FPS)) for i in range(60)]
    assert any(on[34:46])


def test_movement_outside_the_zone_does_not_trigger():
    tr = trig()
    on = [bool(tr.update(lores(frame((60 + 5 * i, 350), seed=i)), i / FPS)) for i in range(60)]
    assert not any(on)


def test_lights_changing_does_not_trigger():
    tr = trig()
    on = [bool(tr.update(lores(frame(gain=1.0 if i < 20 else 1.5, seed=i)), i / FPS)) for i in range(60)]
    assert not any(on)


def test_small_skin_blob_passes_where_a_small_grey_one_does_not():
    def run(color, skin):
        tr = trig(min_blob_px=120, skin=skin)
        return any(tr.update(lores(frame(hand_path(i), seed=i, color=color, r=9)), i / FPS) for i in range(60))
    assert run(SKIN, True) and not run(SKIN, False) and not run((40, 40, 40), True)


def test_unknown_trigger_setting_is_an_error():
    with pytest.raises(ValueError):
        TriggerConfig.from_dict({"min_blob": 3})


# ------------------------------------------------------------------ ring buffer and bursts
def test_burst_has_pre_roll_during_and_post_frames_with_tags():
    rec = BurstRecorder("cam7", fps=10, pre_roll_s=2.0, post_roll_s=1.0, max_burst_s=30)
    out = []
    for i in range(100):
        out += rec.push(i / 10, i, {"shelf"} if 50 <= i < 60 else set())
    assert len(out) == 1
    b = out[0]
    assert (b.camera_id, b.zones, b.trigger_id, b.part, b.t_trigger) == ("cam7", ["shelf"], 1, 0, 5.0)
    assert [f.main for f in b.frames] == list(range(30, 70))          # 2 s before, 1 s active, 1 s after
    assert [f.phase for f in b.frames] == ["pre"] * 20 + ["during"] * 10 + ["post"] * 10
    assert b.frames[25].zones == ["shelf"] and b.frames[0].zones == []
    assert all(x.t < y.t for x, y in zip(b.frames, b.frames[1:]))


def test_long_trigger_is_cut_into_parts_without_losing_frames():
    rec = BurstRecorder("c", fps=10, pre_roll_s=1.0, post_roll_s=0.5, max_burst_s=3.0)
    out = []
    for i in range(200):
        out += rec.push(i / 10, i, {"a"} if 20 <= i < 120 else set())
    out += rec.flush()
    assert len(out) > 2 and {b.trigger_id for b in out} == {1}
    assert [b.part for b in out] == list(range(len(out)))
    assert [f.main for b in out for f in b.frames] == list(range(10, 125))


# ------------------------------------------------------------------ wire format and spool
def test_pack_unpack_roundtrip_and_rejects_garbage():
    h, blobs = unpack(pack({"a": 1, "frames": [{}, {}]}, [b"xx", b"yyy"]))
    assert h["a"] == 1 and h["sizes"] == [2, 3] and blobs == [b"xx", b"yyy"]
    good = pack({"a": 1}, [b"abc"])
    for bad in (b"", b"nope", good[:-1], good + b"z", b"BREE1\xff\xff\xff\xff{}"):
        with pytest.raises(ValueError):
            unpack(bad)


def test_spool_is_oldest_first_survives_restart_and_drops_oldest_when_full(tmp_path):
    s = Spool(tmp_path, max_bytes=250)
    for i in range(3):
        s.put(bytes([i]) * 100)
    assert s.dropped == 1 and len(s) == 2 and s.oldest().read_bytes()[0] == 1
    (tmp_path / "junk.tmp").write_bytes(b"half written")
    s2 = Spool(tmp_path, max_bytes=250)
    assert len(s2) == 2 and s2.bytes == 200 and not list(tmp_path.glob("*.tmp"))
    s2.put(b"z" * 10)
    assert sorted(p.name for p in tmp_path.glob("*.msg"))[-1] == "000000000003.msg"


# ------------------------------------------------------------------ node -> hub, real HTTP on localhost
@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "clip.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(90):
        vw.write(frame(hand_path(i), seed=i))
    vw.release()
    return p


def node_for(clip, tmp_path, port, **node):
    cfg = tmp_path / "node.yaml"
    cfg.write_text(json.dumps({
        "camera": {"id": "shelf_cam_1", "resolution": [W, H]},
        "zones": [{"name": "shelf_A", "kind": "shelf", "polygon": ZONE},
                  {"name": "door", "kind": "exit", "polygon": [[0, 400], [100, 400], [100, 480], [0, 480]]}],
        "node": {"source": str(clip), "fps": FPS, "hub_url": f"http://127.0.0.1:{port}", "token": "s3cret",
                 "spool_dir": str(tmp_path / "spool"), "heartbeat_s": 0.0, **node}}))
    return build(load_node_config(cfg))


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_node_to_hub_end_to_end(clip, tmp_path):
    got = []
    hub = Hub(tmp_path / "hub", token="s3cret", on_burst=lambda d, m: got.append((d, m)))
    httpd = hub.serve("127.0.0.1", 0)
    port = httpd.server_address[1]
    node = node_for(clip, tmp_path, port, lowres_uplink_fps=2)
    node.run()
    assert node.triggers == 1 and len(node.uplink.spool) >= 2          # nothing sent yet: all in the spool
    first = node.uplink.spool.oldest().read_bytes()
    assert node.uplink.drain(10)
    hub.q.join()
    httpd.shutdown()

    (burst_dir, meta), = got
    frames = meta["frames"]
    assert meta["camera_id"] == "shelf_cam_1" and meta["zones"] == ["shelf_A"] and meta["size"] == [W, H]
    assert {f["phase"] for f in frames} == {"pre", "during", "post"}
    assert frames[0]["t"] <= meta["t_trigger"] - 1.9                  # 2 s of pre-roll made it
    assert all(f["zones"] == ["shelf_A"] for f in frames if f["phase"] == "during")
    jpgs = sorted(burst_dir.glob("frame_*.jpg"))
    assert len(jpgs) == len(frames) and cv2.imread(str(jpgs[0])).shape == (H, W, 3)   # full resolution
    assert list((tmp_path / "hub" / "shelf_cam_1" / "lowres").glob("*/frame_0000.jpg"))
    hb = [json.loads(l) for l in (tmp_path / "hub" / "shelf_cam_1" / "heartbeats.jsonl").read_text().splitlines()]
    assert hb[-1]["boot_id"] == node.uplink.boot_id and "trigger_ms" in hb[-1] and "spool_files" in hb[-1]
    assert hub.health()["nodes"]["shelf_cam_1"]["stale"] is False
    assert node.uplink.clock_offset_s is not None and abs(node.uplink.clock_offset_s) < 1.0   # same machine

    # a retry of a message the hub already has is acknowledged, not stored twice
    n = len(list((tmp_path / "hub" / "shelf_cam_1").rglob("burst.json")))
    code, reply, _ = hub.handle("POST", "/v1/burst", {"Authorization": "Bearer s3cret"}, first)
    assert code == 200 and reply.get("duplicate") and len(list((tmp_path / "hub" / "shelf_cam_1").rglob("burst.json"))) == n


def test_hub_down_then_up_nothing_is_lost(clip, tmp_path):
    port = free_port()
    node = node_for(clip, tmp_path, port)
    node.run()
    n = len(node.uplink.spool)
    assert n >= 1 and not node.uplink.step() and node.uplink.errors == 1 and len(node.uplink.spool) == n
    node.uplink._not_before = 0                                        # skip the back-off wait in the test
    hub = Hub(tmp_path / "hub", token="s3cret")
    httpd = hub.serve("127.0.0.1", port)
    assert node.uplink.drain(10) and node.uplink.sent == n
    httpd.shutdown()
    assert len(list((tmp_path / "hub").rglob("burst.json"))) == n


def test_busy_hub_pushes_back_and_wrong_token_keeps_the_evidence(clip, tmp_path):
    gate = threading.Event()
    hub = Hub(tmp_path / "hub", token="s3cret", queue_max=1, on_burst=lambda d, m: gate.wait(10))
    httpd = hub.serve("127.0.0.1", 0)
    node = node_for(clip, tmp_path, httpd.server_address[1], max_burst_s=1.0)     # several short parts
    node.run()
    n = len(node.uplink.spool)
    assert n >= 3
    for _ in range(n):
        node.uplink.step()
        node.uplink._not_before = 0
    assert node.uplink.busy >= 1 and 0 < len(node.uplink.spool) < n    # 429: kept on the node
    gate.set()
    assert node.uplink.drain(10)
    httpd.shutdown()

    bad = Uplink(f"http://127.0.0.1:{httpd.server_address[1]}", "shelf_cam_1", Spool(tmp_path / "s2"), token="wrong")
    assert hub.handle("POST", "/v1/burst", {"Authorization": "Bearer wrong"}, b"x")[0] == 401
    assert hub.handle("POST", "/v1/burst", {"Authorization": "Bearer s3cret"}, b"x")[0] == 400
    auth = {"Authorization": "Bearer s3cret"}
    ok = {"camera_id": "c", "boot_id": "b", "kind": "burst", "seq": 0, "frames": []}
    for field in ("camera_id", "boot_id", "kind"):                     # no name may walk out of the hub folder
        for name in ("../etc", "..", ".", ".x", ""):
            assert hub.handle("POST", "/v1/burst", auth, pack({**ok, field: name}, []))[0] == 400, (field, name)
    assert hub.handle("POST", "/v1/heartbeat", auth, json.dumps({"camera_id": ".."}).encode())[0] == 400
    assert not (tmp_path / "heartbeats.jsonl").exists() and {p.name for p in (tmp_path / "hub").iterdir()} == {"shelf_cam_1"}
    assert bad.pressure() == 0.0


def test_back_pressure_thins_the_burst():
    rec = BurstRecorder("c", fps=10, pre_roll_s=1.0, post_roll_s=1.0)
    out = []
    for i in range(60):
        out += rec.push(i / 10, np.zeros((8, 8, 3), np.uint8), {"a"} if 20 <= i < 30 else set())

    class Src:
        fps, size = 10.0, (8, 8)
        encode = staticmethod(lambda main, q: b"j")
    full, half, tight = (pack_burst(out[0], Src, 85, p)[0] for p in (0.0, 0.6, 0.95))
    assert len(full["frames"]) == 30 and full["thinned"] == "none"
    assert len(half["frames"]) == 20 and sum(f["phase"] == "during" for f in half["frames"]) == 10
    assert [f["phase"] for f in tight["frames"]] == ["during"] * 10


def test_example_config_loads():
    cfg = load_node_config(ROOT / "deploy" / "pi" / "node.example.yaml")
    assert cfg.source == "picamera2" and cfg.zones and cfg.main_size


@pytest.mark.vision
def test_hub_feeds_the_pipeline(tmp_path):
    """A toy clip through node -> hub -> the existing pipeline (toy colour backend). TOY DATA."""
    video = ROOT / "data" / "toy" / "toy_normal_pay.mp4"
    if not video.exists():
        pytest.skip("toy clips missing (make demo)")
    from bree.edge.hub import pipeline_feed
    store = ROOT / "configs" / "store_gas_station_small.yaml"
    hub = Hub(tmp_path / "hub", on_burst=pipeline_feed({"cam_main": str(store)}, backend="toy"))
    httpd = hub.serve("127.0.0.1", 0)
    raw = store.read_text() + f"\nnode:\n  source: {video}\n  fps: 15\n  hub_url: http://127.0.0.1:{httpd.server_address[1]}\n" \
                              f"  spool_dir: {tmp_path / 'spool'}\n"
    (tmp_path / "node.yaml").write_text(raw)
    node = build(load_node_config(tmp_path / "node.yaml"))
    node.run()
    assert node.triggers >= 1 and node.uplink.drain(30)
    hub.q.join()
    httpd.shutdown()
    assert not hub.errors
    events = [json.loads(l) for p in (tmp_path / "hub").rglob("pipeline/events.jsonl") for l in p.read_text().splitlines()]
    assert any(e["type"] == "pick" for e in events)


def test_hub_time_comes_from_the_boot_that_took_the_frame(tmp_path):
    hub = Hub(tmp_path / "hub")
    def send(seq, boot_header, sent_mono):
        msg = pack({"camera_id": "c1", "boot_id": "aaa", "seq": seq, "clock": "monotonic", "fps": 10,
                    "frames": [{"t": 100.0, "zones": ["z"], "phase": "during"}]}, [b"jpeg"])
        assert hub.handle("POST", "/v1/burst", {"X-Boot": boot_header, "X-Sent-Mono": str(sent_mono)}, msg)[0] == 200
        return json.loads((tmp_path / "hub" / "c1" / "burst" / f"aaa_{seq:010d}" / "burst.json").read_text())["frames"][0]
    assert send(0, "bbb", 5.0)["hub_time"] is None            # spooled by boot aaa, sent after a reboot: unknown
    now = time.time()
    assert abs(send(1, "aaa", 101.0)["hub_time"] - (now - 1.0)) < 0.5   # frame taken 1 s before it was sent
    assert abs(send(2, "bbb", 5.0)["hub_time"] - (now - 1.0)) < 0.5     # old boot's offset is reused


# ------------------------------------------------------------------ findings from the independent review
def raw_server(reply: bytes):
    """A hub that answers every request with these exact bytes and hangs up."""
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            with c:
                c.settimeout(1)
                try:
                    c.recv(65536)
                except OSError:
                    pass
                c.sendall(reply)
    threading.Thread(target=loop, daemon=True).start()
    return srv


@pytest.mark.parametrize("reply", [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 500\r\n\r\n{\"hub",   # body cut short
    b"garbage that is not HTTP\r\n\r\n",                                                         # bad status line
    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",                                           # heartbeat reply without hub_time
    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n[]",                                           # JSON, wrong shape
], ids=["short_body", "bad_status_line", "no_hub_time", "not_a_dict"])
def test_malformed_hub_reply_backs_off_and_keeps_the_spool(tmp_path, reply):
    srv = raw_server(reply)
    up = Uplink(f"http://127.0.0.1:{srv.getsockname()[1]}", "c1", Spool(tmp_path / "spool"), heartbeat_s=0.0, timeout_s=2)
    up.submit({"kind": "burst", "frames": [{}]}, [b"jpeg"])
    before = sorted(p.name for p in (tmp_path / "spool").iterdir())
    assert up.step() is False                                # no exception out of step()
    assert up.errors == 1 and up.sent == 0 and up._not_before > time.monotonic()
    assert sorted(p.name for p in (tmp_path / "spool").iterdir()) == before and len(up.spool) == 1
    srv.close()


def test_sender_thread_survives_an_unexpected_error(tmp_path):
    up = Uplink("http://127.0.0.1:1", "c1", Spool(tmp_path / "spool"))
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("anything")
    up._step = boom
    stop = threading.Event()
    th = threading.Thread(target=up.run, args=(stop,), daemon=True)
    th.start()
    time.sleep(0.7)
    assert th.is_alive() and len(calls) >= 2 and up.errors == len(calls)
    stop.set()
    th.join(2)
    assert not th.is_alive()


def test_spool_sets_aside_a_stray_file_and_still_starts(tmp_path):
    (tmp_path / "notes.msg").write_bytes(b"not ours")
    s = Spool(tmp_path)
    s.put(b"a")
    assert len(s) == 1 and s.oldest().name == "000000000000.msg" and (tmp_path / "notes.bad").exists()


def burst_msg(seq, cam="c1"):
    return pack({"camera_id": cam, "boot_id": "aaa", "seq": seq, "kind": "burst", "fps": 10,
                 "frames": [{"t": 1.0, "zones": ["z"], "phase": "during"}]}, [b"jpeg"])


def wait_for(cond, s=5.0):
    end = time.time() + s
    while not cond() and time.time() < end:
        time.sleep(0.02)
    return cond()


def test_stored_bursts_reach_the_pipeline_after_a_restart_and_after_a_lost_queue_race(tmp_path):
    out = tmp_path / "hub"
    h1 = Hub(out)                                            # a hub with no pipeline stores two bursts, then "stops"
    assert all(h1.handle("POST", "/v1/burst", {}, burst_msg(i))[0] == 200 for i in (0, 1))
    got = []
    h2 = Hub(out, on_burst=lambda d, m: got.append(d.name))  # restart with a pipeline: the scan finds both
    assert wait_for(lambda: len(got) == 2) and got == ["aaa_0000000000", "aaa_0000000001"]
    assert all((out / "c1" / "burst" / n / ".done").exists() for n in got)

    class Racy:                                              # the queue that was full a moment after the check
        full = staticmethod(lambda: False)
        def put_nowait(self, item):
            raise __import__("queue").Full
    real, h2.q = h2.q, Racy()
    code, reply, _ = h2.handle("POST", "/v1/burst", {}, burst_msg(2))
    h2.q = real
    assert code == 200 and reply == {"stored": True, "queued": False}
    assert wait_for(lambda: len(got) == 3) and got[-1] == "aaa_0000000002"
    time.sleep(1.2)
    assert len(got) == 3                                     # each burst reaches the pipeline once


def test_hub_health_reports_pipeline_errors(tmp_path):
    def bad(d, m):
        raise RuntimeError("pipeline broke")
    hub = Hub(tmp_path / "hub", on_burst=bad)
    assert hub.handle("POST", "/v1/burst", {}, burst_msg(0))[0] == 200
    assert wait_for(lambda: hub.health()["errors"] == 1)
    assert "pipeline broke" in hub.health()["last_error"]


def test_brightness_flicker_on_an_empty_scene_does_not_trigger():
    """SYNTHETIC: every frame times a random gain in 0.9..1.1, nobody in view."""
    def active(norm):
        rng = np.random.default_rng(3)
        tr = trig(gain_norm=norm)
        return sum(bool(tr.update(lores(frame(gain=1 + rng.uniform(-0.1, 0.1), seed=i)), i / FPS)) for i in range(150))
    assert active(True) == 0


def test_a_hand_still_triggers_under_flicker():
    rng = np.random.default_rng(4)
    tr = trig()
    on = [bool(tr.update(lores(frame(hand_path(i), gain=1 + rng.uniform(-0.1, 0.1), seed=i)), i / FPS)) for i in range(80)]
    assert not any(on[:30]) and any(on[34:46]) and not any(on[60:])


def test_a_node_stuck_on_raises_the_alarm_in_its_heartbeat(clip, tmp_path):
    node = node_for(clip, tmp_path, free_port(), stuck_window_s=2.0)
    node.trigger.update = lambda lo, t: {"shelf_A"}          # a trigger that never lets go (shake, flicker)
    node.run()
    st = node.status()
    assert st["stuck_on"] is True and st["active_share"] > 0.9
    hub = Hub(tmp_path / "hub", token="s3cret")
    hub.handle("POST", "/v1/heartbeat", {"Authorization": "Bearer s3cret"},
               json.dumps({"camera_id": "shelf_cam_1", "boot_id": "b", **st}).encode())
    assert hub.health()["nodes"]["shelf_cam_1"]["stuck_on"] is True
    (tmp_path / "quiet").mkdir()
    quiet = node_for(clip, tmp_path / "quiet", free_port(), stuck_window_s=2.0)
    quiet.run()
    assert quiet.status()["stuck_on"] is False               # one normal reach in 9 s is not stuck


def test_burst_source_reads_a_cameras_bursts_as_one_stream_with_gaps(clip, tmp_path):
    """What the hub stored comes back as frames in time order, each once, at full resolution, on the node's
    media clock: the form bree.pipeline.CameraInput takes as a source (several node cameras, one ledger)."""
    from bree.edge.hub import BurstSource
    hub = Hub(tmp_path / "hub", token="s3cret")
    httpd = hub.serve("127.0.0.1", 0)
    node = node_for(clip, tmp_path, httpd.server_address[1], max_burst_s=1.0)      # several parts per trigger
    node.run()
    assert node.uplink.drain(10) and node.bursts >= 2
    httpd.shutdown()
    src = BurstSource(tmp_path / "hub" / "shelf_cam_1")
    frames = list(src)
    sent = sum(len(json.loads(p.read_text())["frames"]) for p in (tmp_path / "hub" / "shelf_cam_1" / "burst").glob("*/burst.json"))
    assert len(frames) == len(src) == sent < 90                      # only the frames around the trigger left the node
    assert src.fps == FPS and not src.live and frames[0].image.shape == (H, W, 3)
    ts = [f.t for f in frames]
    assert ts == sorted(ts) and len(set(ts)) == len(ts) and 0 <= ts[0] and ts[-1] <= 90 / FPS
    assert [f.index for f in frames] == list(range(len(frames)))
    assert len(BurstSource(tmp_path / "hub" / "nobody")) == 0        # a camera that never triggered: an empty stream


# ------------------------------------------------------------------ audit 2026-10-05
def test_pre_roll_is_a_time_span_not_a_frame_count():
    """A source slower than the configured rate must not give a longer pre-roll; a faster one is capped by
    the frame count (RAM), so its pre-roll is shorter."""
    def pre_roll(real_fps):
        rec = BurstRecorder("c", fps=10, pre_roll_s=2.0, post_roll_s=0.5)
        out = []
        for i in range(int(20 * real_fps)):
            out += rec.push(i / real_fps, i, {"a"} if 10.0 <= i / real_fps < 11.0 else set())
        return out[0].t_trigger - out[0].frames[0].t
    assert pre_roll(10) == pytest.approx(2.0)
    assert pre_roll(5) == pytest.approx(2.0)                         # was 4.0 s as a frame count
    assert pre_roll(20) == pytest.approx(1.0)                        # 20 frames is the RAM bound


def test_hub_deletes_bursts_older_than_the_retention(tmp_path):
    import os
    out = tmp_path / "hub"
    hub = Hub(out, on_burst=lambda d, m: (d / "burst.mp4").write_bytes(b"x"), retain_s=3600.0)
    assert all(hub.handle("POST", "/v1/burst", {}, burst_msg(i))[0] == 200 for i in (0, 1, 2))
    hub.q.join()
    old, edge, new = (out / "c1" / "burst" / f"aaa_{i:010d}" for i in (0, 1, 2))
    now = time.time()
    os.utime(old / "burst.json", (now - 3601, now - 3601))
    os.utime(edge / "burst.json", (now - 3599, now - 3599))
    assert hub.sweep(now) == 1
    assert not old.exists() and (edge / "frame_0000.jpg").exists() and (new / "burst.mp4").exists()
    assert hub.health()["bursts_deleted_by_retention"] == 1 and hub.error_count == 0
    # a hub restarted on old bursts sweeps at start; one the pipeline never saw is counted as an error
    os.utime(edge / "burst.json", (now - 7200, now - 7200))
    (new / ".done").unlink()
    os.utime(new / "burst.json", (now - 7200, now - 7200))
    h2 = Hub(out, on_burst=lambda d, m: None, retain_s=3600.0)
    assert wait_for(lambda: h2.swept == 2) and not list(out.rglob("frame_*.jpg"))
    assert h2.error_count == 1 and "retention" in h2.errors[-1]
    assert Hub(tmp_path / "keep", retain_s=None).sweep(now + 1e9) == 0


def test_burst_source_never_mixes_hub_time_and_media_time(tmp_path):
    from bree.edge.hub import BurstSource
    from bree.pipeline import CameraInput, run_store
    for cam in ("c1", "c2"):
        for seq, hub_time in ((0, None), (1, 1.7e9)):
            d = tmp_path / cam / "burst" / f"b_{seq}"
            d.mkdir(parents=True)
            (d / "burst.json").write_text(json.dumps({"fps": 10, "size": [4, 4], "frames": [{"t": 50.0, "hub_time": hub_time}]}))
    src = BurstSource(tmp_path / "c1")
    assert len(src) == 1 and src.dropped_no_hub_time == 1 and src.wall_clock and not src.t0_given
    assert [t - src.t0 for t, _ in src.rows] == [0.0]                # was [50.0, 1700000000.0] on one axis
    with pytest.raises(ValueError, match="same t0"):
        run_store([CameraInput(c, BurstSource(tmp_path / c), None) for c in ("c1", "c2")], None, tmp_path / "o")
