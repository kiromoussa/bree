"""Onboarding: role rules, zone drafts, station folder, fleet health. Synthetic frames and a fake detector (no GPU)."""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import pytest
import yaml

from bree.events.zones import load_store_config
from bree.fleet.registry import CameraEntry, Fleet, Station
from bree.ingest.imou import HealthSink
from bree.onboard import drafts
from bree.onboard.discover import FolderDiscovery
from bree.onboard.roles import suggest_role
from bree.onboard.run import onboard
from bree.onboard.sheet import contact_sheet, pixelate_people
from bree.onboard.stats import Det, image_stats, compute_stats, static_fraction


# ---------------------------------------------------------------- roles

def _f(**kw):
    base = {"sharpness": 5000, "contrast": 0.25, "brightness": 0.45, "edge_density": 0.2}
    base.update(kw)
    return base


def test_each_role_fires_on_its_own_evidence_and_explains_itself():
    r = suggest_role(29, _f(posgear_mean_count=7, posgear_static=0.95, posgear_frac=1.0, person_frac=0.6))
    assert r.role == "register" and any("counter equipment" in e for e in r.evidence) and not r.needs_review
    assert suggest_role(10, _f(fridge_mean_count=4, drinks_frac=0.9, drinks_static=0.9)).role == "cooler"
    r = suggest_role(20, _f(vehicle_mean_area=0.05, vehicle_frac=1.0, edge_density=0.1))
    assert r.role == "outdoor" and r.confidence > 0.8
    assert suggest_role(7, _f(sky_frac=0.15, vehicle_frac=0.7, vehicle_mean_area=0.01, person_frac=0.6, person_mean_height=0.35)).role == "door"
    assert suggest_role(3, _f(goods_mean_count=8, person_frac=0.3)).role == "aisle"


def test_obscured_and_unreachable_and_unknown():
    assert suggest_role(24, _f(sharpness=200, contrast=0.03, brightness=0.2)).role == "obscured"
    assert suggest_role(24, _f(brightness=0.02)).role == "obscured"
    off = suggest_role(1, {}, online=False)
    assert off.role == "unknown" and "offline" in off.evidence[0] and off.needs_review
    assert suggest_role(2, {}, error="stub stream").role == "unknown"
    blank = suggest_role(5, _f(edge_density=0.15))            # nothing distinctive: it must not pretend
    assert blank.needs_review


def test_close_calls_are_flagged_for_review():
    r = suggest_role(12, _f(sky_frac=0.1, vehicle_frac=1.0, vehicle_mean_area=0.019, person_frac=1.0, person_mean_height=0.28))
    assert r.needs_review and "close call" in r.note


# ---------------------------------------------------------------- stats

def test_image_stats_tell_blank_from_busy():
    blank = np.full((360, 640, 3), 120, np.uint8)
    rng = np.random.default_rng(0)
    busy = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    assert image_stats([blank])["sharpness"] < 1 and image_stats([busy])["sharpness"] > 1000


def test_static_fraction_separates_fixtures_from_movers():
    fixed = Det("keyboard", 0.9, (0.1, 0.1, 0.2, 0.2))
    frames = [[fixed, Det("keyboard", 0.9, (0.5 + 0.1 * i, 0.5, 0.6 + 0.1 * i, 0.6))] for i in range(4)]
    assert 0.45 < static_fraction(frames, ("keyboard",)) < 0.55


# ---------------------------------------------------------------- station folder end to end

def _clip(path, colour, seconds=4, fps=5, size=(320, 180)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(colour)
    for i in range(seconds * fps):
        img = np.full((size[1], size[0], 3), colour, np.uint8)
        img[20:90, 200:300] = rng.integers(0, 255, (70, 100, 3), dtype=np.uint8)
        w.write(img)
    w.release()


class FakeScene:
    """Pretends: channel with colour 60 has counter equipment and a person; 200 has fridges; others nothing."""

    def __call__(self, img):
        c = int(img[0, 0, 0])
        if abs(c - 60) < 8:
            return [Det("keyboard", 0.9, (0.3, 0.6, 0.5, 0.7)), Det("tv", 0.9, (0.3, 0.3, 0.5, 0.55)), Det("keyboard", 0.9, (0.3, 0.6, 0.5, 0.7)),
                    Det("person", 0.8, (0.6, 0.3, 0.7, 0.9)), Det("tv", 0.9, (0.3, 0.3, 0.5, 0.55))]
        if abs(c - 200) < 8:
            return [Det("refrigerator", 0.9, (0.0, 0.0, 0.3, 0.6)), Det("refrigerator", 0.9, (0.3, 0.0, 0.6, 0.6)), Det("bottle", 0.8, (0.1, 0.2, 0.15, 0.3))]
        return []


@pytest.fixture
def footage(tmp_path):
    root = tmp_path / "real" / "st1"
    for ch, colour in ((3, 60), (4, 200), (5, 128)):
        d = root / f"ch{ch:02d}"
        d.mkdir(parents=True)
        for k in range(2):
            _clip(d / f"20261025_18{k}000.mp4", colour)
    return root


def test_onboard_writes_a_station_folder(footage, tmp_path):
    fleet = Fleet(tmp_path / "fleet")
    logs = []
    rep = onboard(FolderDiscovery(footage, min_bytes=0), FakeScene(), fleet, "st1", "SERIAL1", clock_offset_s=-7200, log=logs.append)
    d = fleet.dir("st1")
    roles = {c["channel"]: c["suggestion"]["role"] for c in rep["cameras"]}
    assert roles[3] == "register" and roles[4] == "cooler"
    assert (d / "contact_sheet.jpg").exists() and (d / "stills" / "ch03.jpg").exists()
    assert os.stat(d).st_mode & 0o077 == 0 and os.stat(d / "stills").st_mode & 0o077 == 0        # private folders
    st = fleet.load("st1")
    assert st.clock_offset_s == -7200 and [c.channel for c in st.cameras] == [3, 4, 5]
    assert "stub" not in json.dumps(rep) and "limits" in rep
    # drafts load as normal store configs and carry their provenance
    reg = load_store_config(d / "cameras" / "ch03.yaml")
    assert [z.kind for z in reg.zones] == ["register"] and reg.resolution == (320, 180)
    assert "DRAFT" in (d / "cameras" / "ch03.yaml").read_text()
    cool = load_store_config(d / "cameras" / "ch04.yaml")
    assert cool.zones[0].kind == "cooler"
    sh = yaml.safe_load((d / "shadow.yaml").read_text())
    assert any("imou://SERIAL1/3" in c["rtsp_url"] and "clock_offset=-7200" in c["rtsp_url"] for c in sh["cameras"])
    assert "password" not in (d / "shadow.yaml").read_text().lower()


def test_rerun_keeps_a_role_a_human_set(footage, tmp_path):
    fleet = Fleet(tmp_path / "fleet")
    onboard(FolderDiscovery(footage, min_bytes=0), FakeScene(), fleet, "st1", "S", log=lambda m: None)
    fleet.set_role("st1", 4, "aisle")
    onboard(FolderDiscovery(footage, min_bytes=0), FakeScene(), fleet, "st1", "S", log=lambda m: None)
    c = fleet.load("st1").camera(4)
    assert (c.role, c.role_source) == ("aisle", "human")
    with pytest.raises(ValueError):
        fleet.set_role("st1", 4, "kitchen")


def test_bad_station_ids_are_refused(tmp_path):
    for bad in ("../x", "A B", "", "x" * 50):
        with pytest.raises(ValueError):
            Fleet(tmp_path).dir(bad)


# ---------------------------------------------------------------- privacy helpers

def test_pixelation_destroys_head_detail_only():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)
    out = pixelate_people(img, [(0.4, 0.2, 0.6, 0.9)])
    head = (slice(int(0.2 * 400), int(0.2 * 400) + 20), slice(int(0.4 * 600), int(0.6 * 600)))
    assert np.abs(out[head].astype(int) - img[head].astype(int)).mean() > 20          # head area changed
    assert np.array_equal(out[300:, :100], img[300:, :100])                             # elsewhere untouched
    assert contact_sheet([("a", img), ("b", img)], cols=3).shape[1] == 3 * 560


# ---------------------------------------------------------------- fleet health

def test_heartbeat_age_decides_ok_stale_down(tmp_path):
    fleet = Fleet(tmp_path)
    st = Station("s", "SER", cameras=[CameraEntry(1, "door"), CameraEntry(2, "aisle"), CameraEntry(3, "cooler", enabled=False), CameraEntry(4, "aisle")])
    fleet.save(st)
    now = time.time()
    for ch, age, state in ((1, 2, "ok"), (2, 100, "ok")):
        s = HealthSink(fleet.health_path("s", ch), "s", "SER", ch, every_s=0)
        s.update(True, state=state, last_frame_wall=now - age)
    h = fleet.station_health(st, now)
    states = {c["channel"]: c["state"] for c in h["cameras"]}
    assert states == {1: "ok", 2: "stale", 3: "disabled", 4: "never_seen"} and h["overall"] == "degraded"
    states = {c["channel"]: c["state"] for c in fleet.station_health(st, now + 1000)["cameras"]}
    assert states[1] == "down" and states[2] == "down"
    s = HealthSink(fleet.health_path("s", 4), "s", "SER", 4, every_s=0)
    s.update(True, state="stale_stream", last_error="stub", last_frame_wall=now)
    assert [c for c in fleet.station_health(st, now)["cameras"] if c["channel"] == 4][0]["state"] == "stale_stream"


def test_check_reports_missing_pieces_without_reading_secrets(tmp_path):
    fleet = Fleet(tmp_path / "f")
    cred = tmp_path / "imou.env"
    cred.write_text("IMOU_APP_SECRET=topsecret\n")
    os.chmod(cred, 0o644)
    st = Station("s", "SER", credentials_file=str(cred), cameras=[CameraEntry(1, "door")])
    fleet.save(st)
    res = fleet.check(st)
    names = {r["name"]: r for r in res}
    assert not names["credentials file private"]["ok"] and not names["clock offset known"]["ok"]
    assert not names["ch01 store config"]["ok"] and "topsecret" not in json.dumps(res)
