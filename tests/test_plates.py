"""Fuel drive-off module on SYNTHETIC plates and SCRIPTED timelines: reader plumbing, pump-zone
tracker, the drive-off rule, the alert and review path, plate retention."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bree.alerts.writer import AlertSink
from bree.plates import synth
from bree.plates.pump import DriveOffConfig, DriveOffMonitor, FuelSale, IouTracker, PumpZone
from bree.plates.reader import ClassicalDetector, PlatePipeline, PlateRead, _iou, edit_distance, vote
from bree.plates.retention import PlateStore
from bree.plates.scenarios import SCENARIOS, run
from bree.review.store import ReviewStore

T0 = 1_800_000_000.0
HOUR, DAY = 3600.0, 86400.0
ZONE = PumpZone("P1", [(0, 100), (200, 100), (200, 200), (0, 200)], "forecourt")
BOX = (50, 60, 150, 150)          # bottom centre (100, 150) is inside ZONE
PLATE = "7XYZ123"


class StubReader:
    """Always reads PLATE with the given confidence."""

    def __init__(self, conf=0.95, text=PLATE):
        self.conf, self.text, self.calls = conf, text, 0

    def read(self, frame, roi=None):
        self.calls += 1
        return [PlateRead(self.text, self.conf, [self.conf] * len(self.text), (60, 100, 100, 120))]


def _drive_off(mon, t_leave=40.0, until=200.0, pay_at=None):
    """A car fuels at P1 from 10 s to 30 s, leaves at t_leave, optionally pays at pay_at."""
    frame = np.zeros((240, 320, 3), np.uint8)
    out, t = [], 0.0
    mon.on_sale(FuelSale("P1", 10.0, None, txn_id="T1"))
    while t <= until:
        if t == 30.0:
            mon.on_sale(FuelSale("P1", 10.0, 30.0, amount=40.0, gallons=11.5, txn_id="T1", grade="regular"))
        if pay_at is not None and t == pay_at:
            out += mon.on_payment(t, txn_id="T1")
        out += mon.update(t, [(7, BOX)] if 5.0 <= t <= t_leave else [], frame, camera_id="forecourt")
        t += 0.5
    return out


# ---------------------------------------------------------------- synthetic plates, reader plumbing
def test_synthetic_plate_and_crop():
    rng = np.random.default_rng(0)
    text = synth.random_text(rng)
    assert 6 <= len(text.replace(" ", "")) <= 7 and text.replace(" ", "").isalnum()
    s = synth.plate_crop(rng, 96, "night")
    assert s.image.ndim == 3 and 96 <= s.image.shape[1] <= 96 * 1.3 and s.text.isalnum()
    assert synth.plate_crop(rng, 96, "day").image.mean() > 3 * synth.plate_crop(rng, 96, "night_noisy").image.mean()


def test_edit_distance_and_vote():
    assert edit_distance("ABC123", "ABC123") == 0 and edit_distance("ABC123", "A8C12") == 2
    r = lambda t, c: PlateRead(t, c, [c] * len(t), (0, 0, 1, 1))  # noqa: E731
    assert vote([r("ABC123", 0.9), r("A8C123", 0.6), r("ABC123", 0.8), r("ABC12", 0.99)])[0] == "ABC123"
    assert vote([]) == ("", 0.0) and vote([r("ABC123", 0.2)]) == ("", 0.0)


def test_classical_localiser_finds_synthetic_plates():
    rng, det, hit = np.random.default_rng(1), ClassicalDetector(), 0
    for _ in range(10):
        s, car = synth.scene(rng, 96, "day")
        x1, y1 = car[0], car[1]
        crop = s.image[max(0, car[1]):car[3], max(0, car[0]):car[2]]
        hit += any(_iou((b[0] + max(0, x1), b[1] + max(0, y1), b[2] + max(0, x1), b[3] + max(0, y1)), s.box) >= 0.4
                   for b, _ in det.detect(crop))
    assert hit >= 8


def test_pipeline_roi_offsets_and_ordering():
    class Det:
        def detect(self, img):
            return [((10, 10, 70, 40), 0.9), ((0, 0, 10, 5), 0.9)]      # the second is under min_plate_w

    class Ocr:
        def recognise(self, crop):
            return "ABC123", [0.9] * 6, None

    reads = PlatePipeline(Det(), Ocr()).read(np.zeros((200, 300, 3), np.uint8), roi=(100, 50, 250, 150))
    assert len(reads) == 1 and reads[0].box == (110, 60, 170, 90) and reads[0].text == "ABC123"
    assert PlatePipeline(Det(), Ocr()).read(np.zeros((200, 300, 3), np.uint8), roi=(400, 400, 500, 500)) == []


def test_iou_tracker_keeps_ids():
    tr = IouTracker()
    a = tr.update(0.0, [(0, 0, 100, 100), (300, 0, 400, 100)])
    b = tr.update(0.5, [(305, 0, 405, 100), (4, 0, 104, 100)])
    assert {i for i, _ in a} == {i for i, _ in b} and dict(b)[a[0][0]] == (4, 0, 104, 100)
    assert tr.update(10.0, [(4, 0, 104, 100)])[0][0] not in {i for i, _ in a}      # gap too long: a new track


# ---------------------------------------------------------------- the drive-off rule on scripted timelines
@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_scripted_timeline(sc):
    r = run(sc)
    assert r["ok"], (sc.what, [(a.unpaid_items[0].zone, a.reasons) for a in r["alerts"]], len(r["retractions"]))
    assert all(a.tier == "review" for a in r["alerts"])


def test_alert_waits_for_the_grace_period():
    mon = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=60))
    alerts = _drive_off(mon, t_leave=40.0)
    assert len(alerts) == 1 and 100.0 <= alerts[0].t_emitted <= 101.0 and alerts[0].t_exit == 40.0
    assert _drive_off(DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=60)), pay_at=99.0) == []
    assert _drive_off(DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=60)), pay_at=35.0) == []


def test_uncertain_when_two_vehicles_share_the_zone():
    mon, t = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=10)), 0.0
    mon.on_sale(FuelSale("P1", 10.0, 30.0, amount=20.0, txn_id="T1"))
    out = []
    while t <= 80:
        out += mon.update(t, [(1, BOX), (2, (20, 80, 120, 180))] if 5 <= t <= 40 else [], camera_id="forecourt")
        t += 0.5
    assert len(out) == 1 and any("identity uncertain" in r for r in out[0].reasons) and out[0].confidence < 0.6


def test_drive_off_goes_through_the_alert_and_review_path(tmp_path):
    plates = PlateStore(tmp_path / "plates", now=T0)
    mon = DriveOffMonitor([ZONE], reader=StubReader(), plate_store=plates, out_dir=tmp_path, fps=2,
                          config=DriveOffConfig(grace_s=30), wall0=T0)
    alerts = _drive_off(mon)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.tier == "review" and a.unpaid_items[0].category == "fuel" and a.unpaid_items[0].zone == "P1"
    assert mon.reader.calls == 3                       # stops reading once three frames agree
    # The plate text is in the plate store only: not in the alert, its JSON, or the review store.
    assert PLATE not in json.dumps(a.to_dict())
    sink = AlertSink(tmp_path)
    sink.emit(a)
    sink.close()
    assert PLATE not in (tmp_path / "alerts.jsonl").read_text()
    review = ReviewStore(tmp_path / "review", now=T0)
    assert review.add_alert(a.to_dict(), camera="forecourt", event_ts=T0 + a.t_exit, now=T0 + a.t_emitted) == a.alert_id
    row = review.alerts()[0]
    assert row["tier"] == "review" and row["predicted_item"] == "fuel:regular" and row["zone"] == "P1"
    assert row["clip_path"] and row["clip_path"].endswith(f"alerts/{a.alert_id}.mp4")
    assert PLATE not in json.dumps(row)
    (rid,) = plates.for_alert(a.alert_id)
    rec = plates.get(rid, "manager", "police report", now=T0 + 100)
    assert rec["plate_text"] == PLATE and rec["pump_id"] == "P1" and rec["event_ts"] == T0 + 40.0
    assert rec["crop_path"] and (tmp_path / "plates" / "crops" / f"{rid}.jpg").is_file()


def test_low_confidence_plate_is_not_stored(tmp_path):
    plates = PlateStore(tmp_path / "plates", now=T0)
    mon = DriveOffMonitor([ZONE], reader=StubReader(conf=0.3), plate_store=plates, config=DriveOffConfig(grace_s=30), wall0=T0)
    (a,) = _drive_off(mon)
    assert plates.count() == 0 and any(r.startswith("plate not read") for r in a.reasons) and a.confidence == 0.6


def test_paid_visit_leaves_no_plate_anywhere(tmp_path):
    plates = PlateStore(tmp_path / "plates", now=T0)
    mon = DriveOffMonitor([ZONE], reader=StubReader(), plate_store=plates, config=DriveOffConfig(grace_s=30), wall0=T0)
    assert _drive_off(mon, pay_at=35.0, until=600.0) == []
    assert plates.count() == 0 and not mon.open and not mon.closed and not mon.sales and not mon.pending


def test_late_payment_retracts_and_deletes_the_plate(tmp_path):
    plates = PlateStore(tmp_path / "plates", now=T0)
    review = ReviewStore(tmp_path / "review", now=T0)
    mon = DriveOffMonitor([ZONE], reader=StubReader(), plate_store=plates, config=DriveOffConfig(grace_s=30), wall0=T0)
    alert, retraction = _drive_off(mon, pay_at=150.0)
    assert plates.count() == 0 and not list((tmp_path / "plates" / "crops").iterdir())
    assert retraction.retracts == alert.alert_id and retraction.tier == "retracted"
    review.add_alert(alert.to_dict(), now=T0)
    assert review.add_alert(retraction.to_dict(), now=T0) == alert.alert_id
    assert review.alerts()[0]["retracted_tier"] == "retracted"
    assert _drive_off(mon, until=10.0) == []            # nothing left over to fire twice


def _play(mon, cars, msgs, until, reader_frame=True):
    """cars: [(track_id, t_in, t_out, hidden or None)], all in BOX. msgs: {t: callable(mon) -> alerts or None}."""
    frame, out, t = (np.zeros((240, 320, 3), np.uint8) if reader_frame else None), [], 0.0
    while t <= until:
        if t in msgs:
            out += msgs[t](mon) or []
        seen = [(tid, BOX) for tid, a, b, hid in cars if a <= t <= b and not (hid and hid[0] <= t < hid[1])]
        out += mon.update(t, seen, frame, camera_id="forecourt")
        t += 0.5
    return out


class TimedReader:
    """Reads nothing before `t_from` (the first car's plate is unreadable), then `text`."""

    def __init__(self, mon_ref, t_from, text):
        self.mon_ref, self.t_from, self.text = mon_ref, t_from, text

    def read(self, frame, roi=None):
        if self.mon_ref[0].now < self.t_from - 0.5:
            return []
        return [PlateRead(self.text, 0.97, [0.97] * len(self.text), (60, 100, 100, 120))]


def test_next_car_in_the_same_spot_is_not_the_drive_off(tmp_path):
    """The verifier's probe: thief 5 to 40 s, unreadable plate, unpaid sale A. The next car takes the same spot
    at 60 s (inside rejoin_s), fuels, pays its own sale B and leaves at 150 s."""
    plates, ref = PlateStore(tmp_path / "plates", now=T0), [None]
    mon = ref[0] = DriveOffMonitor([ZONE], reader=TimedReader(ref, 60.0, "PAY1001"), plate_store=plates, wall0=T0)
    alerts = _play(mon, [(1, 5, 40, None), (2, 60, 150, None)], {
        10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, None, txn_id="A")),
        35.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 35.0, amount=40.0, txn_id="A")),
        65.0: lambda m: m.on_sale(FuelSale("P1", 65.0, None, txn_id="B")),
        120.0: lambda m: m.on_sale(FuelSale("P1", 65.0, 120.0, amount=30.0, txn_id="B")),
        130.0: lambda m: m.on_payment(130.0, txn_id="B"),
    }, until=400.0)
    (a,) = alerts
    assert a.t_exit == 40.0 and 160.0 <= a.t_emitted <= 161.0          # the thief's times, not the second car's
    assert plates.count() == 0 and any(r.startswith("plate not read") for r in a.reasons)


def test_second_car_plate_is_never_stored_even_without_its_own_sale(tmp_path):
    """The next car parks in the spot and buys no fuel. Both plates readable: the plates differ, so the visit is
    split and the alert carries the first car's plate and leave time."""
    class TwoPlates:
        def read(self, frame, roi=None):
            text = PLATE if ref[0].now < 50 else "PAY1001"
            return [PlateRead(text, 0.97, [0.97] * len(text), (60, 100, 100, 120))]
    plates, ref = PlateStore(tmp_path / "plates", now=T0), [None]
    mon = ref[0] = DriveOffMonitor([ZONE], reader=TwoPlates(), plate_store=plates, wall0=T0)
    (a,) = _play(mon, [(1, 5, 40, None), (2, 60, 150, None)],
                 {10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 35.0, amount=40.0, txn_id="A"))}, until=400.0)
    assert a.t_exit == 40.0
    (rid,) = plates.for_alert(a.alert_id)
    assert plates.get(rid, "test", "check")["plate_text"] == PLATE

    # Same, first plate unreadable: the limit. The alert waits for the second car, names the gap, stores no plate.
    plates2, ref2 = PlateStore(tmp_path / "plates2", now=T0), [None]
    mon2 = ref2[0] = DriveOffMonitor([ZONE], reader=TimedReader(ref2, 60.0, "PAY1001"), plate_store=plates2, wall0=T0)
    (b,) = _play(mon2, [(1, 5, 40, None), (2, 60, 150, None)],
                 {10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 35.0, amount=40.0, txn_id="A"))}, until=400.0)
    assert plates2.count() == 0 and any("not seen in the zone from 40.0s to 60.0s" in r for r in b.reasons)


@pytest.mark.parametrize("cars", [
    [(1, 5, 90, (32, 38))],                    # hidden for 6 s while fuel is flowing
    [(1, 5, 90, (20, 26))],
    [(1, 5, 50, None), (2, 50.5, 90, None)],   # track id changes while fuel is flowing
    [(1, 5, 15, None), (2, 15.5, 90, None)],
    [(1, 5, 12, None), (2, 80, 90, None)],     # hidden 68 s, longer than rejoin_s
], ids=["hidden_32_38", "hidden_20_26", "id_switch_50", "id_switch_15", "hidden_longer_than_rejoin"])
def test_car_still_at_the_pump_is_not_a_drive_off(cars):
    """Sale 10 to 85 s (still dispensing through every gap), paid at 88 s, grace 20 s. No alert at any point."""
    mon = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=20))
    out = _play(mon, cars, {
        10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, None, txn_id="T1")),
        85.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 85.0, amount=40.0, txn_id="T1")),
        88.0: lambda m: m.on_payment(88.0, txn_id="T1"),
    }, until=200.0, reader_frame=False)
    assert out == []


def test_gap_after_fuelling_still_waits_and_alert_names_it():
    """Sale 10 to 30 s. Hidden 32 to 38 s, leaves at 60 s unpaid: one alert, timed from 60 s, gap named."""
    mon = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=20))
    (a,) = _play(mon, [(1, 5, 60, (32, 38))],
                 {10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 30.0, amount=40.0, txn_id="T1"))}, 200.0, False)
    assert a.t_exit == 60.0 and 80.0 <= a.t_emitted <= 81.0
    assert any("not seen in the zone from 31.5s to 38.0s" in r for r in a.reasons)


def test_sales_without_txn_id_merge_and_settle_by_pump():
    """A controller with no transaction ids: start message, end message, payment by pump. One sale, no alert."""
    mon = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=20))
    out = _play(mon, [(1, 5, 40, None)], {
        10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, None)),
        30.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 30.0, amount=40.0)),
        35.0: lambda m: m.on_payment(35.0, pump_id="P1"),
    }, until=200.0, reader_frame=False)
    assert out == [] and not mon.sales and not mon.pending
    mon = DriveOffMonitor([ZONE], config=DriveOffConfig(grace_s=20))           # and unpaid: exactly one alert
    out = _play(mon, [(1, 5, 40, None)], {
        10.0: lambda m: m.on_sale(FuelSale("P1", 10.0, None)),
        30.0: lambda m: m.on_sale(FuelSale("P1", 10.0, 30.0, amount=40.0)),
    }, until=200.0, reader_frame=False)
    assert len(out) == 1


def test_monitor_purges_an_idle_plate_store(tmp_path):
    plates = PlateStore(tmp_path, retention_hours=1, now=T0)
    plates.add("A1", PLATE, 0.95, event_ts=T0, now=T0)
    mon = DriveOffMonitor([ZONE], plate_store=plates, wall0=T0)
    mon.tick(1800.0)
    assert plates.db.execute("SELECT COUNT(*) FROM plates").fetchone()[0] == 1
    mon.tick(2 * HOUR)                                  # no write, no lookup, no reopen: the monitor's own timer
    assert plates.db.execute("SELECT COUNT(*) FROM plates").fetchone()[0] == 0


# ---------------------------------------------------------------- retention
def test_plate_expires_after_default_72_hours(tmp_path):
    s = PlateStore(tmp_path, now=T0)
    assert s.retention_hours == 72 and s.confirmed_retention_days == 30
    rid = s.add("A1", PLATE, 0.9, event_ts=T0, pump_id="P1", crop_jpg=b"jpg", now=T0)
    crop = tmp_path / "crops" / f"{rid}.jpg"
    assert s.get(rid, "manager", "review", now=T0 + 71 * HOUR)["plate_text"] == PLATE and crop.is_file()
    assert s.purge(now=T0 + 72 * HOUR)["plates_deleted"] == 1
    assert s.count() == 0 and not crop.exists() and s.get(rid, "manager", "review", now=T0 + 73 * HOUR) is None
    assert PLATE.encode() not in (tmp_path / "plates.sqlite").read_bytes()      # secure_delete zeroed it


def test_retention_is_configurable_and_purges_on_open(tmp_path):
    s = PlateStore(tmp_path, retention_hours=1, now=T0)
    s.add("A1", PLATE, 0.9, event_ts=T0, now=T0)
    s.close()
    assert PlateStore(tmp_path, now=T0 + 2 * HOUR).count() == 0          # reopened later: gone, setting remembered
    assert PlateStore(tmp_path, now=T0).retention_hours == 1
    with pytest.raises(ValueError):
        PlateStore(tmp_path, retention_hours=0)


def test_review_decisions_extend_or_delete(tmp_path):
    plates, review = PlateStore(tmp_path / "p", now=T0), ReviewStore(tmp_path / "r", now=T0)
    for aid in ("A1", "A2", "A3"):
        review.add_alert({"alert_id": aid, "tier": "review", "confidence": 0.8, "person_id": 1,
                          "unpaid_items": [{"category": "fuel", "zone": "P1", "score": 0.8}], "reasons": []},
                         event_ts=T0, now=T0)
        plates.add(aid, PLATE, 0.9, event_ts=T0, now=T0)
    review.decide("A1", "ann", "confirmed_theft", now=T0 + HOUR)
    review.decide("A2", "ann", "not_theft", now=T0 + HOUR)
    assert plates.sync_reviews(review, now=T0 + HOUR) == {"confirmed": 1, "deleted": 1}
    assert plates.for_alert("A2") == []
    plates.purge(now=T0 + 29 * DAY)                    # unreviewed A3 is long gone, confirmed A1 is still there
    assert plates.for_alert("A3") == [] and len(plates.for_alert("A1")) == 1
    plates.purge(now=T0 + 30 * DAY)
    assert plates.count() == 0


def test_every_lookup_is_logged_without_the_plate_text(tmp_path):
    s = PlateStore(tmp_path, now=T0)
    rid = s.add("A1", PLATE, 0.9, event_ts=T0, now=T0)
    with pytest.raises(ValueError):
        s.get(rid, "", "police report")
    with pytest.raises(ValueError):
        s.get(rid, "manager", " ")
    s.get(rid, "manager", "police report 24-1187", now=T0 + 60)
    s.purge(now=T0 + 80 * HOUR)
    log = s.access_log()
    assert [(r["who"], r["action"]) for r in log] == [("system", "write"), ("manager", "read"), ("system", "delete")]
    assert PLATE not in json.dumps(log)


def test_orphan_crops_are_deleted(tmp_path):
    s = PlateStore(tmp_path, now=T0)
    (tmp_path / "crops" / "left_over.jpg").write_bytes(b"x")
    assert s.purge(now=T0)["orphan_crops_deleted"] == 1


# ---------------------------------------------------------------- open models (skipped when not installed)
@pytest.mark.vision
def test_open_reader_reads_a_synthetic_drive_off(tmp_path):
    pytest.importorskip("fast_plate_ocr")
    pytest.importorskip("open_image_models")
    from bree.plates.reader import make_reader
    try:
        reader = make_reader()
    except Exception as e:                              # weights not cached and no network
        pytest.skip(f"plate models unavailable: {e}")
    rng, ok = np.random.default_rng(3), 0
    for _ in range(10):
        s = synth.plate_crop(rng, 128, "day")
        ok += reader.read_crop(s.image).text == s.text
    assert ok >= 8
    plates = PlateStore(tmp_path / "plates", now=T0)
    r = run(next(s for s in SCENARIOS if s.name == "drive_off"), reader=reader, render=True, plate_store=plates,
            out_dir=tmp_path)
    assert r["ok"] and r["plates"] and r["plates"][0][2] is not None and r["plates"][0][2] <= 1
