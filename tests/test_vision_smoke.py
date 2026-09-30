"""Vision smoke tests: backends produce sane output and the pipeline runs end to end.
These need torch/ultralytics and model weights, so they are marked `vision`."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.vision


@pytest.fixture(scope="module")
def store():
    from bree.events.zones import load_store_config
    return load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")


def _frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    assert ok
    return img


def test_yolo_backend_finds_people_with_keypoints(store):
    video = ROOT / "data" / "real" / "vtest.avi"
    if not video.exists():
        pytest.skip("data/real/vtest.avi missing (make data)")
    from bree.cli import make_backend
    persons, products = make_backend("yolo", store)(_frame(video, 150))
    assert len(persons) >= 4                                  # ~7 people visible in this frame
    assert persons.keypoints.shape == (len(persons), 17, 3)
    assert (persons.keypoints[:, 9:11, 2] > 0.3).mean() > 0.5  # most wrists found


def test_toy_backend_and_tracker(store):
    from bree.detect.toy import ToyBackend
    from bree.sim.toy_render import render, scenarios
    from bree.track.bytetrack import Tracker
    sc = next(s for s in scenarios() if s.name == "walkout")
    out = ROOT / "out" / "test_toy"
    render(sc, store, out, seed=0)
    video = out / "toy_walkout.mp4"
    backend, tracker = ToyBackend(), Tracker(15)
    ids = set()
    for i in range(0, 60):
        persons, products = backend(_frame(video, i))
        p, q = tracker.update(persons, products, (720, 1280))
        ids |= {x.track_id for x in p}
    assert len(ids) == 1                                      # one shopper, one stable ID


def test_pipeline_end_to_end_on_toy_clip(store, tmp_path):
    from bree.detect.toy import ToyBackend
    from bree.pipeline import run_pipeline
    from bree.sim.toy_render import render, scenarios
    sc = next(s for s in scenarios() if s.name == "walkout")
    render(sc, store, tmp_path, seed=0)
    s = run_pipeline(str(tmp_path / "toy_walkout.mp4"), store, ToyBackend(), tmp_path / "run", verbose=False)
    assert s.events.get("pick") == 1 and s.events.get("exit") == 1
    assert len(s.alerts) == 1 and s.alerts[0]["tier"] == "alert"
    assert Path(s.alerts[0]["clip_path"]).exists()
    for f in ("frames.jsonl", "events.jsonl", "alerts.jsonl", "annotated.mp4", "summary.json", "ledger_log.txt"):
        assert (tmp_path / "run" / f).exists()


def test_head_pixelation_changes_head_region_only():
    from bree.alerts.annotate import blur_heads, head_box
    from bree.detect.toy import synth_keypoints
    from bree.events.observations import PersonObs
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    box = (100, 50, 160, 250)
    p = PersonObs(1, box, 0.9, synth_keypoints(box, []))
    out = blur_heads(img.copy(), [p])
    x1, y1, x2, y2 = head_box(p)
    assert not np.array_equal(out[y1:y2, x1:x2], img[y1:y2, x1:x2])
    assert np.array_equal(out[260:, :], img[260:, :])
