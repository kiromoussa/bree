"""Regression guard on the SYNTHETIC calibration / slot / store-wide identity bench (bree/calib/bench.py), on the
recommended 45 camera layout (tests/fixtures/calib_layout.json). Seeded, so the reference numbers in
tests/fixtures/calib_guard.json repeat exactly; regenerate them with
`PYTHONPATH=src .venv/bin/python tests/test_calib_bench.py`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bree.calib.bench import FIXTURE, calibration_rows, identity_rows, slot_rows
from bree.calib.slots import SlotMap

pytest.importorskip("cv2")
GUARD = Path(__file__).resolve().parent / "fixtures" / "calib_guard.json"
LAYOUT = json.loads(FIXTURE.read_text())
SM = SlotMap(LAYOUT)
SLOT_ROW = "default model"
ID_KEYS = ("shoppers", "tracks", "handoffs_live", "handoffs_live_ok", "handoffs_gap", "handoffs_gap_ok", "false_merges",
           "false_merges_silent", "splits", "clean_visits", "visits_uncertain", "exits_ok", "exits_wrong")


def measure(episodes: int = 25, seed: int = 0) -> dict:
    overhead = [c["id"] for c in LAYOUT["cameras"] if c["kind"] == "overhead"]
    cal = calibration_rows(LAYOUT["cameras"], SM, LAYOUT["store"], seed)["rows"]["click_2px_lens_known"]
    [slot] = slot_rows(LAYOUT["cameras"], SM, LAYOUT["store"], seed, only=SLOT_ROW)["rows"].values()
    out = {"episodes": episodes, "seed": seed, "calibration": cal, "slots": slot}
    for n in (2, 4):
        rows = identity_rows(LAYOUT["cameras"], SM, LAYOUT["store"], tuple(overhead[:n]), episodes, seed)["rows"]
        out[f"identity_{n}_cameras"] = {name: {k: r[k] for k in ID_KEYS} | {"handoff_success": r["handoff_success"]}
                                        for name, r in rows.items()}
    return out


@pytest.fixture(scope="module")
def got():
    ref = json.loads(GUARD.read_text())
    return measure(ref["episodes"], ref["seed"]), ref


def test_every_layout_camera_calibrates_within_the_noise_model(got):
    cal, ref = got[0]["calibration"], got[1]["calibration"]
    assert cal["calibrated"] == cal["cameras"] == 45
    assert cal["position_error_m"]["p90"] <= max(ref["position_error_m"]["p90"], 0.02)       # NoiseModel.cam_pos_m
    assert cal["rotation_error_deg"]["median"] <= max(ref["rotation_error_deg"]["median"], 0.2)   # NoiseModel.cam_rot_deg
    assert cal["reprojection_rms_px"]["max"] < 5.0                     # the default threshold of `calibrate.py check`


def test_two_views_beat_one_view_on_the_same_slots(got):
    s, ref = got[0]["slots"], got[1]["slots"]
    assert s["slots_two_view"] == ref["slots_two_view"] and s["slots_with_a_view"] == ref["slots_with_a_view"]
    assert s["two_view_exact"] >= ref["two_view_exact"] - 1e-9
    assert s["two_view_exact"] > s["one_view_hand_exact_same_slots"] + 0.05
    assert s["two_view_exact"] > s["two_view_plain_nearest_exact"]     # weighting by the error model earns its keep
    assert s["two_view_right_sku"] >= ref["two_view_right_sku"] - 1e-9


@pytest.mark.parametrize("n", (2, 4))
def test_store_wide_closed_world_does_not_regress(got, n):
    rows, ref = got[0][f"identity_{n}_cameras"], got[1][f"identity_{n}_cameras"]
    off = rows["off"]
    assert off == ref["off"] and rows["reid"] == ref["reid"]           # flags off: the old behaviour, exactly
    for name in ("closed_world", "closed_world_reid"):
        r, want = rows[name], ref[name]
        assert r["splits"] <= want["splits"] and r["splits"] < off["splits"], (name, r)
        assert r["false_merges_silent"] <= want["false_merges_silent"] < off["false_merges_silent"], (name, r)
        assert r["handoff_success"] >= want["handoff_success"] - 1e-9 and r["handoff_success"] > off["handoff_success"], (name, r)
        assert r["exits_wrong"] == 0, (name, r)
    best, reid = rows["closed_world_reid"], rows["reid"]
    assert best["false_merges_silent"] < reid["false_merges_silent"] and best["clean_visits"] > reid["clean_visits"]
    assert best["handoff_success"] > 0.9


if __name__ == "__main__":
    GUARD.write_text(json.dumps(measure(), indent=1))
    print(f"wrote {GUARD}")
