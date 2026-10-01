"""Shadow mode: POS export folder, would-be alert log, labels + summary, review API."""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bree.alerts.types import Alert, UnpaidItem
from bree.dashboard.server import serve
from bree.events.types import Catalog, Event, EventType as E
from bree.ledger.ledger import Ledger
from bree.ledger.payments import FolderPayments
from bree.shadow import ShadowLog, load_shadow_config, would_be_record

ROOT = Path(__file__).resolve().parents[1]


def _pay(t, sku="COKE", terminal="pos_1"):
    return json.dumps({"terminal": terminal, "t": t, "items": [{"sku": sku}]})


# ------------------------------------------------------------------ POS folder

def test_pos_folder_skips_old_files_and_reads_new_ones(tmp_path):
    (tmp_path / "old.jsonl").write_text(_pay(1) + "\n")
    src = FolderPayments(tmp_path, every_s=0)
    assert src.poll(100) == []                                          # receipts from before start
    (tmp_path / "batch1.jsonl").write_text(_pay(5) + "\n" + _pay(9, "PEPSI") + "\n")
    (tmp_path / "notes.txt").write_text(_pay(6) + "\n")                 # not an export file
    assert [p.t for p in src.poll(6)] == [5]                            # released by stream time
    assert [p.items[0].sku for p in src.poll(10)] == ["PEPSI"]
    assert src.poll(100) == []                                          # never read twice


def test_pos_folder_appends_partial_lines_lists_and_terminal_filter(tmp_path):
    src = FolderPayments(tmp_path, every_s=0, terminals={"pos_1": "register"})
    f = tmp_path / "live.jsonl"
    f.write_text(_pay(1) + "\n" + _pay(2)[:15])                          # POS still writing line 2
    assert [p.t for p in src.poll(10)] == [1]
    with f.open("a") as fh:
        fh.write(_pay(2)[15:])                                          # finished, no trailing newline
    assert src.poll(10) == []                                           # still growing at this scan
    assert [p.t for p in src.poll(10)] == [2]                           # unchanged since: line is complete
    (tmp_path / "b.json").write_text(json.dumps([json.loads(_pay(3)), json.loads(_pay(4, terminal="other_cam"))]))
    assert src.poll(10) == []                                           # no newline: waits one scan
    assert [p.t for p in src.poll(10)] == [3]                           # other terminal filtered out
    (tmp_path / "bad.jsonl").write_text("garbage\n")
    src.poll(10)
    assert len(src.errors) == 1


def test_pos_folder_replay_reads_existing(tmp_path):
    (tmp_path / "day.jsonl").write_text(_pay(1) + "\n")
    assert [p.t for p in FolderPayments(tmp_path, every_s=0, skip_existing=False).poll(5)] == [1]


# ------------------------------------------------------------------ would-be alert log

def _ledger_alert() -> Alert:
    """A real ledger decision: picks a soda, pockets it, walks out."""
    led = Ledger(Catalog({"COKE": "soda"}), terminal_zones={"pos_1": "register"},
                 zone_kinds={"shelf_A": "shelf", "register": "register", "door": "exit"})
    alerts = led.replay([Event(E.ENTER, 0, 1),
                         Event(E.PICK, 2, 1, item="soda", zone="shelf_A", confidence=0.9),
                         Event(E.CONCEAL, 4, 1, item="soda", confidence=0.8),
                         Event(E.EXIT, 10, 1, zone="door")])
    assert len(alerts) == 1
    return alerts[0]


def _log_with_alert(tmp_path) -> tuple[ShadowLog, dict]:
    log = ShadowLog(tmp_path)
    a = _ledger_alert()
    clip = tmp_path / "cam1" / "S1" / "alerts" / f"{a.alert_id}.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"0123456789")
    a.clip_path = str(clip)
    rec = would_be_record(a, "cam1", "S1", tmp_path)
    log.add_alert(rec)
    return log, rec


def test_would_be_record_has_everything_a_reviewer_needs(tmp_path):
    log, rec = _log_with_alert(tmp_path)
    [back] = log.alerts()
    assert back == rec
    assert rec["id"] == "cam1-S1-A00001" and rec["camera"] == "cam1" and rec["person_id"] == 1
    assert rec["tier"] in ("alert", "review") and rec["confidence"] >= 0.4
    assert rec["concealment_seen"] is True
    assert rec["basket"] == ["soda"] and [i["category"] for i in rec["unpaid"]] == ["soda"]
    assert rec["clip"] == "cam1/S1/alerts/A00001.mp4"                    # relative to the output folder
    assert any("conceal" in line for line in rec["audit_log"])
    assert log.clip_file(rec["id"]) == (tmp_path / rec["clip"]).resolve()
    assert log.clip_file("nope") is None


def test_late_receipt_retraction_is_folded_into_the_would_be_alert(tmp_path):
    """Pocket-then-pay; the POS export with their receipt lands a minute after the decision."""
    from bree.events.types import LineItem, Payment
    led = Ledger(Catalog({"COKE": "soda"}), terminal_zones={"pos_1": "register"},
                 zone_kinds={"shelf_A": "shelf", "register": "register", "door": "exit"})
    alerts = led.replay([Event(E.ENTER, 0, 1), Event(E.PICK, 2, 1, item="soda", zone="shelf_A", confidence=0.9),
                         Event(E.CONCEAL, 4, 1, item="soda", confidence=0.8),
                         Event(E.PAY, 12.5, 1, zone="register", meta={"phase": "start", "t_start": 10}),
                         Event(E.PAY, 20, 1, zone="register", meta={"phase": "end", "t_start": 10, "t_end": 20}),
                         Event(E.EXIT, 25, 1, zone="door")],
                        [Payment(15, "pos_1", [LineItem(sku="COKE")], txn_id="T1", t_received=80)])
    assert [a.tier for a in alerts] == ["review", "retracted"]
    log = ShadowLog(tmp_path)
    for a in alerts:
        log.add_alert(would_be_record(a, "cam1", "S1", tmp_path))
    assert len(log.alerts()) == 2                                        # both records kept in the jsonl
    [row] = log.labelled()
    assert row["id"] == "cam1-S1-A00001" and row["tier"] == "review"
    assert row["retraction"]["retracts"] == row["id"] and row["retraction"]["tier"] == "retracted"
    assert "late receipt" in row["retraction"]["reasons"][0]
    s = log.summary()
    assert s["would_be_alerts"] == 1 and s["retracted"] == 1 and s["review"]["total"] == 1
    httpd = serve(None, "127.0.0.1", 0, review=log)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert b"RETRACTED" in _req(base + "/review").read()
        d = json.loads(_req(base + "/api/review").read())
        assert [a["retraction"]["tier"] for a in d["alerts"]] == ["retracted"] and d["summary"]["retracted"] == 1
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------ labels

def test_labels_append_only_latest_wins_and_summary(tmp_path):
    log = ShadowLog(tmp_path)
    for i, tier in enumerate(["alert", "alert", "alert", "review"]):
        log.add_alert({"id": f"x{i}", "tier": tier, "clip": None})
    log.add_label("x0", "false_alert", "kiro")
    log.add_label("x0", "real_theft", "operator", "on camera 2 too")     # changed mind: latest wins
    log.add_label("x1", "false_alert", "kiro")
    log.add_label("x2", "unsure", "kiro")
    with pytest.raises(ValueError):
        log.add_label("x3", "maybe", "kiro")
    with pytest.raises(ValueError):
        log.add_label("x3", "real_theft", "  ")                          # reviewer required
    with pytest.raises(ValueError):
        log.add_label("missing", "real_theft", "kiro")
    assert len(log.labels_path.read_text().splitlines()) == 4             # every click kept
    assert log.latest_labels()["x0"]["reviewer"] == "operator"
    s = log.summary()
    assert s["would_be_alerts"] == 4 and s["labelled"] == 3
    assert s["alert"] == {"total": 3, "real_theft": 1, "false_alert": 1, "unsure": 1, "unlabelled": 0,
                          "precision": 0.5}
    assert s["review"]["unlabelled"] == 1 and s["review"]["precision"] is None


# ------------------------------------------------------------------ review API

def _req(url, data=None, ctype="application/json", headers=None):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": ctype, **(headers or {})})
    return urllib.request.urlopen(req, timeout=5)


def test_review_api(tmp_path):
    log, rec = _log_with_alert(tmp_path)
    httpd = serve(None, "127.0.0.1", 0, review=log)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert httpd.server_address[0] == "127.0.0.1"
        assert b"BREE shadow review" in _req(base + "/review").read()
        d = json.loads(_req(base + "/api/review").read())
        assert d["alerts"][0]["id"] == rec["id"] and d["alerts"][0]["label"] is None
        body = json.dumps({"id": rec["id"], "label": "false_alert", "reviewer": "kiro", "note": "own drink"}).encode()
        assert json.loads(_req(base + "/api/review/label", body).read())["label"] == "false_alert"
        d = json.loads(_req(base + "/api/review").read())
        assert d["alerts"][0]["label"]["note"] == "own drink" and d["summary"]["all"]["false_alert"] == 1
        with pytest.raises(urllib.error.HTTPError) as e:                 # cross-site form post
            _req(base + "/api/review/label", body, ctype="application/x-www-form-urlencoded")
        assert e.value.code == 415
        with pytest.raises(urllib.error.HTTPError) as e:
            _req(base + "/api/review/label", json.dumps({"id": rec["id"], "label": "x", "reviewer": "k"}).encode())
        assert e.value.code == 400
        assert _req(base + "/clips/" + rec["id"]).read() == b"0123456789"
        part = _req(base + "/clips/" + rec["id"], headers={"Range": "bytes=2-4"})
        assert part.status == 206 and part.read() == b"234"
        for bad in ("/clips/nope", "/clips/..%2F..%2Fetc%2Fpasswd", "/state"):
            with pytest.raises(urllib.error.HTTPError):
                _req(base + bad)
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------ config + end to end

def test_example_config_loads():
    cfg = load_shadow_config(ROOT / "configs" / "shadow_example.yaml")
    assert cfg.cameras[0].rtsp_url.startswith("rtsp://") and Path(ROOT / cfg.cameras[0].store).exists()
    assert cfg.record_raw["enabled"] is False                            # raw (unblurred) recording off by default


@pytest.mark.vision
def test_shadow_end_to_end_on_toy_clip(tmp_path):
    """Toy clip as the 'camera', its POS receipts dropped in the export folder."""
    pytest.importorskip("ultralytics")
    from bree.events.zones import load_store_config
    from bree.shadow import run_shadow
    from bree.sim.toy_render import render, scenarios
    store_path = ROOT / "configs" / "store_gas_station_small.yaml"
    sc = next(s for s in scenarios() if s.name == "conceal_partial_pay")
    render(sc, load_store_config(store_path), tmp_path / "toy", seed=0)
    pos = tmp_path / "pos"
    pos.mkdir()
    shutil.copy(tmp_path / "toy" / f"toy_{sc.name}.payments.jsonl", pos / "export.jsonl")
    cfg_path = tmp_path / "shadow.yaml"
    cfg_path.write_text(json.dumps({     # JSON is valid YAML
        "cameras": [{"name": "cam1", "rtsp_url": str(tmp_path / "toy" / f"toy_{sc.name}.mp4"),
                     "store": str(store_path)}],
        "pos_export_dir": str(pos), "output_dir": str(tmp_path / "out"), "backend": "toy", "review_port": 0}))
    log = run_shadow(load_shadow_config(cfg_path), once=True)
    recs = log.alerts()
    assert recs, "the conceal + partial pay clip should produce a would-be alert"
    r = recs[0]
    assert r["concealment_seen"] and r["paid"]                           # the receipt reached the ledger
    assert log.clip_file(r["id"]) is not None
    assert not list((tmp_path / "out").rglob("annotated.mp4"))           # nothing but flagged clips on disk
    assert not (tmp_path / "out" / "raw").exists()
