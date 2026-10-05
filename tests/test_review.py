"""Review feedback loop on SYNTHETIC alerts: store, decisions, label export, metrics, owner report,
retention, reviewer page API, retrain hook."""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

from bree.review.__main__ import ingest
from bree.review.labels import export_labels, load_manifest, retrain_hook
from bree.review.metrics import metrics, owner_report
from bree.review.server import serve
from bree.review.store import ReviewStore, frames_from_log
from bree.review.synthetic import BASE, populate

DAY = 86400


def _alert(i, item="soda_bottle", zone="cooler_bank", **kw):
    return {"alert_id": f"A{i}", "tier": "alert", "confidence": 0.8, "person_id": i, "visited_register": False,
            "unpaid_items": [{"category": item, "sku": None, "zone": zone, "score": 0.8}], "reasons": [], **kw}


def _small(tmp_path):
    """Four alerts, decisions chosen so every number below can be checked by hand."""
    s = ReviewStore(tmp_path / "s", retention_days=30, now=BASE)
    for i, zone in enumerate(["cooler_bank", "cooler_bank", "snack_aisle", "snack_aisle", "snack_aisle"]):
        s.add_alert(_alert(i, zone=zone), camera="cam1" if i < 3 else "cam2", event_ts=BASE + i * 100, now=BASE + i * 100)
    s.decide("A0", "ann", "confirmed_theft", time_to_decision_s=4, now=BASE + 60)
    s.decide("A0", "bob", "confirmed_theft", time_to_decision_s=6, now=BASE + 600)
    s.decide("A1", "ann", "not_theft", time_to_decision_s=8, now=BASE + 100 + 120)
    s.decide("A1", "bob", "confirmed_theft", now=BASE + 900)                       # disagreement
    s.decide("A2", "ann", "wrong_item", "chips", time_to_decision_s=10, now=BASE + 200 + 300)
    s.decide("A3", "ann", "confirmed_theft", now=BASE + 300 + 30)
    s.decide("A3", "ann", "not_theft", now=BASE + 300 + 40)                        # changed mind: latest counts
    return s                                                                       # A4 never reviewed


def test_metrics_by_hand(tmp_path):
    m = metrics(_small(tmp_path))
    o = m["overall"]
    assert (o["sent"], o["reviewed"], o["review_rate"], o["not_reviewed"]) == (5, 4, 0.8, 1)
    assert (o["confirmed_theft"], o["wrong_item"], o["not_theft"], o["disputed"]) == (1, 1, 1, 1)
    assert o["precision"] == round(2 / 3, 3) and o["item_precision"] == round(1 / 3, 3)
    assert o["median_event_to_decision_s"] == 90.0            # first decisions after 60, 120, 300, 30 s
    assert o["median_view_s"] == 7.0                          # 4, 6, 8, 10
    assert m["by_zone"]["cooler_bank"]["precision"] == 1.0 and m["by_zone"]["snack_aisle"]["precision"] == 0.5
    assert m["by_camera"]["cam2"] == {**m["by_camera"]["cam2"], "sent": 2, "reviewed": 1, "precision": 0.0}
    assert list(m["by_week"]) == ["2026-W38"]
    # two alerts seen by ann and bob: one agreement. po = 0.5, pe = 0.5*1 + 0.5*0 = 0.5 -> kappa 0
    assert m["agreement"] == {"alerts_with_two_reviewers": 2, "same_decision": 1, "percent_agreement": 0.5, "cohens_kappa": 0.0}


def test_decide_validation_and_ingest_dedup(tmp_path):
    s = _small(tmp_path)
    assert s.add_alert(_alert(0)) is None                                          # same id: not stored twice
    for args in (("A0", "ann", "maybe"), ("A0", " ", "unclear"), ("A0", "ann", "wrong_item"), ("nope", "ann", "unclear")):
        with pytest.raises(ValueError):
            s.decide(*args)
    with pytest.raises(ValueError):
        s.decide("A0", "ann", "unclear", time_to_decision_s=float("inf"))
    assert s.decide("A4", "cy", "confirmed_theft", "3")["corrected_item"] is None  # a correction only goes with wrong_item
    assert "3" not in s.known_items() and s.known_items() == ["chips", "soda_bottle"]
    assert len(s.alerts()[3]["reviews"]) == 1 and s.alerts()[3]["decision"] == "not_theft"
    rows = sqlite3.connect(s.dir / "review.sqlite").execute("SELECT COUNT(*) FROM reviews WHERE alert_id='A3'").fetchone()
    assert rows[0] == 2                                                            # history is kept


def test_shadow_record_retraction_and_privacy(tmp_path):
    s = ReviewStore(tmp_path / "s")
    clip = tmp_path / "shadow" / "cam1" / "alerts" / "A1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    rec = {"id": "cam1-s1-A1", "retracts": None, "time": "2026-09-20T10:00:00+0000", "camera": "cam1", "person_id": 3,
           "tier": "review", "confidence": 0.6, "clip": "cam1/alerts/A1.mp4", "visited_register": True,
           "unpaid": [{"category": "chips", "sku": "CHP1", "zone": "snack_aisle", "score": 0.6, "reid": [0.1, 0.2],
                       "vector": ["IDVEC"], "track_signature": "sig", "crop": "person.jpg"}],
           "paid": [{"category": "soda_bottle", "vector": ["IDVEC"]}], "reasons": ["capped at review: identity uncertain (swap)"],
           "reid_embedding": [1, 2, 3]}
    assert s.add_alert(rec, base_dir=tmp_path / "shadow",
                       frames=[{"t": 1.0, "event": "pick", "face_crop": "x.jpg", "appearance_feature": [1], "emb": [1],
                                "color_hist": [2], "appearance": [3], "gait_signature": [4], "vector": ["IDVEC"],
                                "track_signature": "sig", "crop": "person.jpg",
                                "products": [{"id": 1, "bbox": [0, 0, 1, 1], "cat": "chips", "vector": ["IDVEC"]}]}]) == "cam1-s1-A1"
    a = s.alerts()[0]
    assert (a["camera"], a["zone"], a["predicted_item"], a["register_match"]) == ("cam1", "snack_aisle", "CHP1", "receipt_missing_item")
    assert a["identity_uncertain"] and a["clip_path"] == str(clip.resolve()) and s.clip_file("cam1-s1-A1") == clip.resolve()
    assert a["frames"] == [{"t": 1.0, "event": "pick", "image": None,              # only the listed fields are kept
                            "products": [{"id": 1, "bbox": [0, 0, 1, 1], "cat": "chips"}]}]
    assert a["items"] == [{"category": "chips", "sku": "CHP1", "zone": "snack_aisle", "score": 0.6}]
    assert a["register"]["paid_items"] == [{"category": "soda_bottle"}]
    dump = sqlite3.connect(s.dir / "review.sqlite").execute("SELECT * FROM alerts").fetchone()
    assert not any(k in json.dumps(dump) for k in ("reid", "embedding", "face", "feature", "emb", "hist", "gait", "vector", "IDVEC",
                                                      "signature", "crop", "person.jpg"))
    assert a["event_ts"] == 1789898400.0                                           # 10:00 +0000, whatever the local zone
    s.add_alert({"id": "cam1-s1-A2", "retracts": "cam1-s1-A1", "tier": "retracted"})
    assert len(s.alerts()) == 1 and s.alerts()[0]["retracted_tier"] == "retracted"
    assert s.add_alert({"id": "cam1-s1-A9", "retracts": "cam1-s1-A7", "tier": "retracted"}) is None
    assert len(s.alerts()) == 1                                                    # an orphan retraction is not an alert
    for raw in ("raw/cam1/seg.mp4", "RAW/seg.mp4", "cam1/raw_segments/seg.mp4", "raw/alerts/seg.mp4",
                "rawclips/a.mp4", "segments_unblurred/a.mp4", "cam1/a.mp4", "cam1/alerts/a.jpg"):   # not pixelated, by folder name
        with pytest.raises(ValueError):
            s.add_alert({**rec, "id": "x", "clip": raw}, base_dir=tmp_path / "shadow")


def test_end_to_end_synthetic(tmp_path):
    store, now = populate(tmp_path / "s", n=24, seed=1)
    alerts = store.alerts()
    assert len(alerts) == 25 and all(a["clip_path"] for a in alerts)
    m = metrics(store)
    o = m["overall"]
    assert o["reviewed"] == sum(1 for a in alerts if a["reviews"]) - 2 > 0         # the two staged alerts are left out
    assert o["reviewed"] == sum(o[d] for d in ("confirmed_theft", "not_theft", "wrong_item", "unclear", "disputed"))
    assert (m["alerts_recorded"], m["staged_alerts"], m["retracted_alerts"]) == (25, 2, 0)
    assert sum(b["sent"] for b in m["by_week"].values()) == sum(b["sent"] for b in m["by_camera"].values()) == 23
    assert 0 < o["precision"] < 1 and 3 <= o["median_view_s"] <= 9

    man = export_labels(store, now=now)
    c = man["counts"]
    agreed = [a for a in alerts if a["decision"] in ("confirmed_theft", "wrong_item")]
    assert c["detector"] == len({a["alert_id"] for a in agreed} - {"SYN_DUP"}) > 0
    assert c["pick"] > 0 and c["conceal"] > 0 and c["new_this_export"] == len(man["examples"])
    assert c["duplicates_skipped"] >= 2                                            # SYN_DUP: same frames under a second id
    root = store.dir / "datasets" / "review"
    assert load_manifest(root / "manifest.json")["classes"] == man["classes"]
    for e in man["examples"]:
        assert "reviewer_one" not in json.dumps(e)                                 # counts only, no names
        if e["task"] == "detector":
            cid, cx, cy, w, h = (root / e["labels"]).read_text().splitlines()[0].split()
            assert man["classes"][int(cid)] == e["class"] and all(0 < float(v) < 1 for v in (cx, cy, w, h))
            assert (root / e["image"]).is_file() and e["pseudo_boxes"] == 6
            assert e["corrected"] == (e["class"] != e["predicted_class"])
        else:
            z = np.load(root / e["window"])
            assert z["kps"].shape == (e["n_frames"], 17, 3) and e["label"] in (0, 1)
    again = export_labels(store, now=now)
    assert again["counts"]["new_this_export"] == 0 and again["classes"] == man["classes"]

    weeks = [owner_report(store, d, tmp_path / "rep") for d in ("2026-09-14", "2026-09-21", "2026-09-28")]
    assert sum(w["alerts_sent"] for w in weeks) == 25
    st = [t for w in weeks for t in w["staged_tests"]["tests"]]
    assert {t["test_id"]: t["caught"] for t in st} == {"T1": True, "T2": True, "T3": False}
    w0 = weeks[0]
    assert w0["customer_alerts"]["precision"] == m["by_week"]["2026-W38"]["precision"]   # one rule in both places
    assert w0["customer_alerts"]["sent"] == w0["alerts_sent"] - w0["alerts_from_staged_tests"]
    md = (tmp_path / "rep" / "owner_report_2026-09-14.md").read_text()
    assert "Staged: " in md and "MISSED" in md and chr(0x2014) not in md and chr(0x2013) not in md
    assert json.loads((tmp_path / "rep" / "owner_report_2026-09-14.json").read_text()) == json.loads(json.dumps(w0))


def test_changed_decision_drops_the_example(tmp_path):
    store, now = populate(tmp_path / "s", n=6, seed=3)
    man = export_labels(store, now=now)
    e = next(e for e in man["examples"] if e["task"] == "detector" and e["alert_id"] != "SYN0000")
    root = store.dir / "datasets" / "review"
    for r in next(a for a in store.alerts() if a["alert_id"] == e["alert_id"])["reviews"]:
        store.decide(e["alert_id"], r["reviewer_id"], "not_theft", now=now)
    man2 = export_labels(store, now=now)
    assert e["key"] not in {x["key"] for x in man2["examples"]} and man2["counts"]["removed_this_export"] >= 1
    assert not (root / e["image"]).exists() and not (root / e["labels"]).exists()


def test_retention_deletes_media_keeps_decisions(tmp_path):
    store, now = populate(tmp_path / "s", n=6, seed=2, retention_days=30)
    export_labels(store, now=now)
    root = store.dir / "datasets" / "review"
    assert store.purge(now)["alerts_purged"] == 0 and list(root.glob("*/*.npz"))
    before = metrics(store)["overall"]
    late = ReviewStore(tmp_path / "s", now=now + 40 * DAY)                         # opening the store purges
    rows = late.alerts()
    assert all(a["purged_ts"] and a["clip_path"] is None and a["frames"] == [] for a in rows)
    assert not list((tmp_path / "s").rglob("*.mp4")) and not list((tmp_path / "s").rglob("*.jpg"))
    assert not list(root.glob("*/*.npz")) and json.loads((root / "manifest.json").read_text())["examples"] == []
    assert metrics(late)["overall"] == before                                      # decisions and counts survive
    assert late.next_for("someone_new") == (None, 0)                               # nothing left to show
    with pytest.raises(ValueError):
        ReviewStore(tmp_path / "s", retention_days=0)


def test_review_page_api(tmp_path):
    store, _ = populate(tmp_path / "s", n=3, seed=0, retention_days=36500)
    httpd = serve(store, port=0)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(body, ctype="application/json"):
        req = urllib.request.Request(url + "/api/decide", data=body, headers={"Content-Type": ctype})
        return json.loads(urllib.request.urlopen(req).read())

    try:
        page = urllib.request.urlopen(url + "/").read().decode()
        assert "http://" not in page and "https://" not in page                    # nothing loaded from outside
        d = json.loads(urllib.request.urlopen(url + "/api/next?reviewer=cat").read())
        aid = d["alert"]["alert_id"]
        assert d["left"] == 4 and d["alert"]["clip_path"] is True and "frames" not in d["alert"]
        assert urllib.request.urlopen(f"{url}/clips/{aid}").read()[4:8] == b"ftyp"
        req = urllib.request.Request(f"{url}/clips/{aid}", headers={"Range": "bytes=0-9"})
        assert len(urllib.request.urlopen(req).read()) == 10
        rec = post(json.dumps({"alert_id": aid, "reviewer": "cat", "decision": "wrong_item", "corrected_item": "chips",
                               "conceal_seen": True, "view_s": 3.5}).encode())
        assert rec["decision"] == "wrong_item"
        mine = next(r for a in store.alerts() if a["alert_id"] == aid for r in a["reviews"] if r["reviewer_id"] == "cat")
        assert (mine["corrected_item"], mine["conceal_seen"], mine["pick_seen"], mine["time_to_decision_s"]) == ("chips", 1, None, 3.5)
        assert json.loads(urllib.request.urlopen(url + "/api/next?reviewer=cat").read())["left"] == 3
        for body, ctype, code in ((json.dumps({"alert_id": aid, "reviewer": "cat", "decision": "bogus"}).encode(), "application/json", 400),
                                  (json.dumps({"alert_id": aid, "reviewer": "", "decision": "unclear"}).encode(), "application/json", 400),
                                  (b"alert_id=x", "application/x-www-form-urlencoded", 415)):
            with pytest.raises(urllib.error.HTTPError) as e:
                post(body, ctype)
            assert e.value.code == code
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/clips/../review.sqlite")
        with pytest.raises(urllib.error.HTTPError) as e:                           # not an object
            post(b"[1]")
        assert e.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as e:                           # view time must be finite
            post(json.dumps({"alert_id": aid, "reviewer": "cat", "decision": "unclear", "view_s": 1e999}).encode())
        assert e.value.code == 400
        size = len(urllib.request.urlopen(f"{url}/clips/{aid}").read())
        bad = urllib.request.urlopen(urllib.request.Request(f"{url}/clips/{aid}", headers={"Range": "bytes=abc-"}))
        assert bad.status == 200 and len(bad.read()) == size                       # malformed range: whole file
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(f"{url}/clips/{aid}", headers={"Range": f"bytes={size}-"}))
        assert e.value.code == 416
        tail = urllib.request.urlopen(urllib.request.Request(f"{url}/clips/{aid}", headers={"Range": "bytes=-7"}))
        assert tail.status == 206 and len(tail.read()) == 7
        # back: a named alert comes back with this reviewer's decision, and the queue count is unchanged
        again = json.loads(urllib.request.urlopen(f"{url}/api/next?reviewer=cat&alert={aid}").read())
        assert (again["alert"]["alert_id"], again["alert"]["my_decision"], again["left"]) == (aid, "wrong_item", 3)
        # another Host name (DNS rebinding) is refused on every route
        for path, data in (("/api/next?reviewer=cat", None), ("/", None), (f"/clips/{aid}", None), ("/api/decide", b"{}")):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(url + path, data=data, headers={
                    "Host": "evil.example", "Content-Type": "application/json"}))
            assert e.value.code == 403
        port = httpd.server_address[1]
        assert urllib.request.urlopen(urllib.request.Request(url + "/", headers={"Host": f"localhost:{port}"})).status == 200
        # a negative or huge Content-Length is answered at once, not read
        for length in ("-1", "99999999", "abc"):
            with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
                c.sendall(f"POST /api/decide HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
                          f"Content-Length: {length}\r\n\r\n".encode())
                assert c.recv(64).startswith(b"HTTP/1.0 400")
    finally:
        httpd.shutdown()


def test_retrain_hook(tmp_path):
    store, now = populate(tmp_path / "s", n=6, seed=2)
    seen = tmp_path / "seen.txt"
    cmd = f"{sys.executable} -c \"import sys,os;open(r'{seen}','w').write(sys.argv[1]+'|'+os.environ['BREE_REVIEW_MANIFEST'])\""
    wait = retrain_hook(store, min_new=10_000, command=cmd, now=now)
    assert not wait["ready"] and not wait["ran"] and not seen.exists()
    r = retrain_hook(store, min_new=1, command=cmd, now=now)
    assert r["ran"] and r["returncode"] == 0 and r["new_total"] == sum(r["new"].values()) > 0
    assert seen.read_text() == f"{r['manifest']}|{r['manifest']}"
    assert retrain_hook(store, min_new=1, command=cmd, now=now)["new_total"] == 0   # already used
    failed = retrain_hook(ReviewStore(tmp_path / "s2"), min_new=0, command=f"{sys.executable} -c \"raise SystemExit(3)\"")
    assert failed["returncode"] == 3 and not (tmp_path / "s2" / "datasets" / "review" / "retrain_state.json").exists()
    gone = retrain_hook(ReviewStore(tmp_path / "s2"), min_new=0, command="/no/such/train_command --x")
    assert (gone["ran"], gone["returncode"]) == (False, None) and "FileNotFoundError" in gone["error"]
    assert not (tmp_path / "s2" / "datasets" / "review" / "retrain_state.json").exists()


def test_frames_from_log(tmp_path):
    kp = [[1.0, 2.0, 0.9]] * 17
    rows = [{"frame": i, "t": i / 10, "persons": [{"id": 7, "kpts": kp}, {"id": 8, "kpts": kp}], "products": [],
             "events": [{"type": "pick", "t": 2.0, "person_id": 7, "item": "chips"}] if i == 20 else []} for i in range(40)]
    log = tmp_path / "frames.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fr = frames_from_log(log, {"person_id": 7, "group": []})
    assert {f["event"] for f in fr} == {"pick"} and (fr[0]["t"], fr[-1]["t"], len(fr)) == (0.4, 3.0, 27)
    assert frames_from_log(log, {"person_id": 9, "group": []}) == []


def test_page_keyboard_flow_in_chrome(tmp_path):
    """Key 3 must not type a "3" into the item field, Enter there submits, and text left in the
    field is not stored with another decision. Needs node and BREE_PLAYWRIGHT (else skipped)."""
    pw = os.environ.get("BREE_PLAYWRIGHT")
    if not pw or not shutil.which("node"):
        pytest.skip("set BREE_PLAYWRIGHT to node_modules/playwright to run the browser test")
    store, _ = populate(tmp_path / "s", n=6, seed=0, retention_days=36500)
    httpd = serve(store, port=0)
    try:
        script = os.path.join(os.path.dirname(__file__), "..", "scripts", "review", "page_keys.mjs")
        run = subprocess.run(["node", script, f"http://127.0.0.1:{httpd.server_address[1]}"],
                             capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        r = json.loads(run.stdout)
    finally:
        httpd.shutdown()
    assert (r["afterKey3"], r["focused"], r["outside"], r["tooEarly"]) == ("", "item", [], False)
    mine = {a["alert_id"]: x for a in store.alerts() for x in a["reviews"] if x["reviewer_id"] == "kb"}
    assert (mine[r["first"]]["decision"], mine[r["first"]]["corrected_item"]) == ("wrong_item", "chips")
    assert (mine[r["second"]]["decision"], mine[r["second"]]["corrected_item"]) == ("confirmed_theft", None)
    assert "3chips" not in store.known_items() and "leftover" not in store.known_items()
    # key 1 held for 30 repeats: one decision. Five fast taps of key 2: one decision.
    assert (r["leftBeforeHold"], r["leftAfterHold"], r["leftAfterTaps"]) == (6, 5, 4)
    assert len({r["first"], r["second"], r["third"], r["fourth"]}) == 4 and r["afterBack"] == r["fourth"]
    rows = sqlite3.connect(store.dir / "review.sqlite").execute(
        "SELECT alert_id, decision FROM reviews WHERE reviewer_id='kb' ORDER BY id").fetchall()
    assert rows == [(r["first"], "wrong_item"), (r["second"], "confirmed_theft"), (r["third"], "not_theft"),
                    (r["third"], "unclear"), (r["fourth"], "wrong_item")]          # back: third decided again
    assert r["backShows"] and mine[r["third"]]["decision"] == "unclear"
    assert r["heldKey3"] == ""                                                     # repeats of 3 are not typed
    assert r["unlistedFirstEnter"]["moved"] is False and "not in the list" in r["unlistedFirstEnter"]["hint"]
    assert mine[r["fourth"]]["corrected_item"] == "chpis"                          # stored after the second Enter


def test_retention_reaches_a_dataset_outside_the_store(tmp_path):
    store, now = populate(tmp_path / "s", n=4, seed=2, retention_days=30)
    out = tmp_path / "elsewhere" / "ds"
    man = export_labels(store, out, now=now)
    assert len(man["examples"]) > 0 and list(out.rglob("*.npz")) and list(out.rglob("*.jpg"))
    assert out.resolve() in store.dataset_dirs()
    res = ReviewStore(tmp_path / "s", now=now + 60 * DAY).purge(now + 60 * DAY)    # the open already purged
    assert res["dataset_examples_deleted"] == 0
    assert not [p for p in out.rglob("*") if p.suffix in (".npz", ".jpg", ".txt")]
    assert json.loads((out / "manifest.json").read_text())["examples"] == []
    shutil.rmtree(out)                                                             # a deleted folder does not break purge
    ReviewStore(tmp_path / "s", now=now + 61 * DAY)


def test_export_survives_bad_boxes(tmp_path):
    img = tmp_path / "frames" / "f.jpg"
    img.parent.mkdir()
    img.write_bytes(b"not really a jpg, the export only copies it")
    s = ReviewStore(tmp_path / "s", now=BASE)
    frame = {"t": 1.0, "event": "pick", "item": "soda_bottle", "image": str(img), "width": 100, "height": 80}
    products = [{"id": 1, "bbox": [10, 10, 20, 20]},                               # no category
                {"id": 2, "bbox": [30, 30, 20, 20], "cat": "chips"},               # inverted
                {"id": 3, "bbox": [90, 70, 140, 120], "cat": "chips"}]             # runs out of the frame
    for i, box in enumerate([[50, 40, 120, 90], [60, 40, 50, 30], [200, 200, 220, 220]]):
        s.add_alert(_alert(i), camera="cam1", event_ts=BASE, now=BASE, frames=[{**frame, "item_box": box, "products": products}])
        s.decide(f"A{i}", "ann", "confirmed_theft", now=BASE + 5)
    man = export_labels(s, now=BASE + 10)
    assert man["counts"]["detector"] == 1 and man["counts"]["skipped"] == {"detector_bad_box": 2}
    e = man["examples"][0]
    assert e["bbox"] == [50.0, 40.0, 100.0, 80.0] and e["pseudo_boxes"] == 1       # clipped to 100 x 80
    lines = (s.dir / "datasets" / "review" / e["labels"]).read_text().splitlines()
    assert lines == ["0 0.750000 0.750000 0.500000 0.500000", "1 0.950000 0.937500 0.100000 0.125000"]


def test_ingest_rerun_same_folder_and_clip_lookup(tmp_path, monkeypatch, capsys):
    run = tmp_path / "repo" / "out" / "runA"
    (run / "alerts").mkdir(parents=True)
    (run / "alerts" / "A00001.mp4").write_bytes(b"x")
    rec = {"alert_id": "A00001", "person_id": 1, "tier": "alert", "confidence": 0.7, "retracts": None,
           "unpaid_items": [{"category": "chips", "zone": "snack_aisle", "score": 0.7}], "reasons": [],
           "clip_path": "out/runA/alerts/A00001.mp4"}                              # relative to where the run started
    retract = {"alert_id": "A00002", "tier": "retracted", "retracts": "A00001"}
    (run / "alerts.jsonl").write_text(json.dumps(rec) + "\n" + json.dumps(retract) + "\n")
    monkeypatch.chdir(tmp_path)                                                    # not the folder the run started in
    s = ReviewStore(tmp_path / "s")
    r1 = ingest(s, run / "alerts.jsonl", camera="cam1")
    assert (r1["new"], r1["clips_missing"], r1["warnings"]) == (1, 0, [])
    a = s.alerts()[0]
    assert a["clip_path"] == str((run / "alerts" / "A00001.mp4").resolve()) and a["retracted_tier"] == "retracted"
    assert ingest(s, run / "alerts.jsonl", camera="cam1")["new"] == 0              # the same file again: nothing new
    # a second run into the same folder name, a different alert under the same number, and no clip
    (run / "alerts.jsonl").write_text(json.dumps({**rec, "confidence": 0.9, "clip_path": "out/runA/alerts/gone.mp4"}) + "\n")
    r2 = ingest(s, run / "alerts.jsonl", camera="cam1")
    assert (r2["new"], r2["clips_missing"], len(r2["warnings"])) == (1, 1, 2)
    assert "same name" in r2["warnings"][0] and "not found" in r2["warnings"][1]
    assert len(s.alerts()) == 2 and len({x["alert_id"] for x in s.alerts()}) == 2


def test_staged_test_across_the_week_boundary(tmp_path):
    s = ReviewStore(tmp_path / "s", retention_days=3650, now=BASE)
    edge = BASE + 7 * DAY                                                          # Monday 00:00 UTC, next week
    s.add_staged_test("LATE", edge - 60, "cam1", item="chips")                     # one minute before midnight
    s.add_staged_test("ANYCAM", BASE + DAY)                                        # no camera named
    s.add_alert(_alert(1), camera="cam1", event_ts=edge + 90, now=edge + 90)       # caused by LATE
    s.add_alert(_alert(2), camera="cam2", event_ts=edge + 3600, now=edge + 3600)   # a customer
    s.add_alert(_alert(3), camera="cam9", event_ts=BASE + DAY + 10, now=BASE + DAY + 10)
    w1, w2 = owner_report(s, "2026-09-14"), owner_report(s, "2026-09-21")
    assert {t["test_id"]: t["caught"] for t in w1["staged_tests"]["tests"]} == {"LATE": True, "ANYCAM": True}
    assert (w1["alerts_sent"], w1["alerts_from_staged_tests"], w1["customer_alerts"]["sent"]) == (1, 1, 0)
    assert (w2["alerts_sent"], w2["alerts_from_staged_tests"], w2["customer_alerts"]["sent"]) == (2, 1, 1)
    assert w2["staged_tests"]["staged"] == 0


def test_corrupt_manifest_does_not_block_the_store(tmp_path):
    store, now = populate(tmp_path / "s", n=4, seed=2, retention_days=30)
    good = export_labels(store, tmp_path / "good", now=now)
    export_labels(store, now=now)
    mf = store.dir / "datasets" / "review" / "manifest.json"
    assert not list(mf.parent.glob("*.tmp"))                                       # written through a temp file, renamed
    mf.write_text(mf.read_text()[:200])                                            # an interrupted write
    late = ReviewStore(tmp_path / "s", now=now + 60 * DAY)                         # opens, and purges what it can
    res = late.purge(now + 60 * DAY)
    assert [e["folder"] for e in res["dataset_errors"]] == [str(mf.parent.resolve())]
    assert "JSONDecodeError" in res["dataset_errors"][0]["error"]
    assert json.loads((tmp_path / "good" / "manifest.json").read_text())["examples"] == [] and len(good["examples"]) > 0
    assert late.next_for("x") == (None, 0) and all(a["purged_ts"] for a in late.alerts())


def test_staged_alert_with_no_logged_test_is_not_a_customer_alert(tmp_path):
    s = ReviewStore(tmp_path / "s", retention_days=3650, now=BASE)
    for i in range(4):
        s.add_alert(_alert(i), camera="cam1", event_ts=BASE + 1000 * (i + 1), now=BASE, staged_test_id="GHOST" if i == 0 else None)
        s.decide(f"A{i}", "ann", "confirmed_theft" if i == 0 else "not_theft", now=BASE + 9000)
    rep = owner_report(s, "2026-09-14", tmp_path / "rep")
    assert (rep["alerts_sent"], rep["alerts_from_staged_tests"], rep["customer_alerts"]["sent"]) == (4, 1, 3)
    assert rep["customer_alerts"]["precision"] == 0.0 and rep["staged_alerts_with_no_logged_test"] == ["GHOST"]
    assert "GHOST" in (tmp_path / "rep" / "owner_report_2026-09-14.md").read_text()
    m = metrics(s)
    assert (m["staged_alerts"], m["overall"]["sent"], m["overall"]["precision"]) == (1, 3, 0.0)


def test_retractions_end_to_end(tmp_path):
    run = tmp_path / "runB"
    run.mkdir()
    rec = lambda n: {"alert_id": f"A0000{n}", "person_id": n, "tier": "alert", "confidence": 0.7, "retracts": None,  # noqa: E731
                     "unpaid_items": [{"category": "chips", "zone": "snack_aisle", "score": 0.7}], "reasons": []}
    (run / "alerts.jsonl").write_text("".join(json.dumps(rec(n)) + "\n" for n in (1, 2, 3)))
    s = ReviewStore(tmp_path / "s")
    assert ingest(s, run / "alerts.jsonl", camera="cam1")["new"] == 3
    for n in (1, 2, 3):
        s.decide(next(a["alert_id"] for a in s.alerts() if f"-A0000{n}-" in a["alert_id"]), "ann", "not_theft" if n == 3 else "confirmed_theft")
    # a later file with only the retractions: 1 withdrawn, 2 lowered to review, 7 was never ingested
    (run / "alerts.jsonl").write_text("".join(json.dumps({"alert_id": f"A0001{n}", "tier": tier, "retracts": f"A0000{n}"}) + "\n"
                                              for n, tier in ((1, "retracted"), (2, "review"), (7, "retracted"))))
    r = ingest(s, run / "alerts.jsonl", camera="cam1")
    assert (r["new"], r["retractions"], len(r["warnings"])) == (0, 2, 1) and "A00007" in r["warnings"][0]
    by = {a["alert_id"].split("-")[1]: a for a in s.alerts()}
    assert len(by) == 3 and (by["A00001"]["retracted_tier"], by["A00002"]["retracted_tier"], by["A00003"]["retracted_tier"]) == ("retracted", "review", None)
    left = [a for a in (s.next_for("bob"),)]
    assert s.next_for("bob")[1] == 2 and left[0][0]["alert_id"] != by["A00001"]["alert_id"]   # the withdrawn one is not queued
    m = metrics(s)
    assert (m["alerts_recorded"], m["retracted_alerts"], m["overall"]["sent"], m["overall"]["precision"]) == (3, 1, 2, 0.5)
    httpd = serve(s, port=0)
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        d = json.loads(urllib.request.urlopen(f"{url}/api/next?reviewer=bob").read())
        assert d["alert"]["retracted_tier"] == "review" and "by a late receipt" in urllib.request.urlopen(url + "/").read().decode()
    finally:
        httpd.shutdown()


def test_corrected_item_spelling_and_unknown_classes(tmp_path):
    img = tmp_path / "frames" / "f.jpg"
    img.parent.mkdir()
    s = ReviewStore(tmp_path / "s", now=BASE)
    s.add_alert(_alert(9, item="chips"), camera="cam1", event_ts=BASE, now=BASE)   # the pipeline has named "chips"
    for i, typed in enumerate(["Chips ", "CHIPS", "chpis", "Energy  Drink"]):
        img.write_bytes(b"frame %d" % i)
        s.add_alert(_alert(i), camera="cam1", event_ts=BASE, now=BASE, frames=[
            {"t": 1.0, "item": "soda_bottle", "image": str(img), "width": 100, "height": 80, "item_box": [10 + i, 10, 30, 30]}])
        assert s.decide(f"A{i}", "ann", "wrong_item", typed, now=BASE + 5)["corrected_item"] == ["chips", "chips", "chpis", "energy_drink"][i]
        export_labels(s, now=BASE + 10)                                            # read the frame before it is overwritten
    man = export_labels(s, now=BASE + 10)
    assert man["classes"] == ["chips", "chpis", "energy_drink"] and man["counts"]["unknown_classes"] == 2
    assert {e["class"]: e["class_known"] for e in man["examples"]} == {"chips": True, "chpis": False, "energy_drink": False}


def test_two_stores_cannot_share_a_dataset_folder(tmp_path):
    a, now = populate(tmp_path / "a", n=4, seed=2)
    b, _ = populate(tmp_path / "b", n=4, seed=3)
    shared = tmp_path / "shared"
    n = len(export_labels(a, shared, now=now)["examples"])
    with pytest.raises(ValueError, match="another review store"):
        export_labels(b, shared, now=now)
    kept = json.loads((shared / "manifest.json").read_text())
    assert n > 0 and len(kept["examples"]) == n and kept["store_id"] == a.store_id != b.store_id
    assert all((shared / (e.get("image") or e["window"])).is_file() for e in kept["examples"])
    assert shared.resolve() not in b.dataset_dirs()                                # and B's retention does not reach into it


def test_purged_keypoints_are_not_left_in_the_database_file(tmp_path):
    """Audit 2026-10-05: after a purge the keypoints of purged alerts were still readable in the bytes of
    review.sqlite (freed pages). The marker is a keypoint value that appears nowhere else."""
    marker = b"7777.125"
    frames = [{"t": 0.1 * k, "kpts": [[7777.125, 7777.125, 0.9]] * 17} for k in range(40)]
    s = ReviewStore(tmp_path / "s", retention_days=30, now=BASE)
    for i in range(4):
        s.add_alert(_alert(i), camera="cam1", event_ts=BASE + (0 if i < 2 else 20 * DAY), frames=frames, now=BASE)
    db = tmp_path / "s" / "review.sqlite"
    full = db.read_bytes().count(marker)                             # a few markers straddle a page boundary
    assert 0.99 * 4 * 40 * 17 * 2 <= full <= 4 * 40 * 17 * 2
    assert s.purge(BASE + 35 * DAY)["alerts_purged"] == 2
    live = sum(json.dumps(a["frames"]).count(marker.decode()) for a in s.alerts())
    s.close()
    assert live == 2 * 40 * 17 * 2 and 0.99 * live <= db.read_bytes().count(marker) <= live
    assert not list((tmp_path / "s").glob("review.sqlite-*"))        # no journal or WAL left holding the old rows


def test_ingest_of_a_missing_file_is_one_line_not_a_traceback(tmp_path):
    with pytest.raises(SystemExit, match="no alerts file"):
        ingest(ReviewStore(tmp_path / "s"), str(tmp_path / "nope" / "alerts.jsonl"))
