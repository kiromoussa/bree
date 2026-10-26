"""Real-camera perception pieces (bree/perception): slicing, merging, profile filters, ghost learning, track gate,
lens model, label loader. Synthetic inputs only; no footage."""
from __future__ import annotations

import numpy as np
import pytest

from bree.perception.evalcounts import match_frame, summarize
from bree.perception.ghost import find_ghosts, ghost_polygon
from bree.perception.lens import distort_points, fit_k1, undistort_points
from bree.perception.profile import IgnoreRegion, PerceptionProfile, keep_mask, load_profile, save_profile
from bree.perception.sliced import SliceSpec, grid_windows, merge_boxes, plan_windows, touches_cut


def test_grid_windows_cover_frame_with_overlap():
    wins = grid_windows(2560, 1440, 3, 2, 0.2)
    assert len(wins) == 6
    assert min(w[0] for w in wins) == 0 and max(w[2] for w in wins) == 2560
    assert min(w[1] for w in wins) == 0 and max(w[3] for w in wins) == 1440
    row = sorted(w for w in wins if w[1] == 0)
    assert row[0][2] > row[1][0]            # neighbours overlap


def test_plan_windows_counts():
    spec = SliceSpec(2, 2, rois=[[0.0, 0.0, 0.5, 0.5]])
    assert len(plan_windows((1440, 2560, 3), spec)) == 1 + 4 + 1 == spec.n_windows
    assert plan_windows((100, 200, 3), SliceSpec()) == [(0, 0, 200, 100)]


def test_merge_drops_duplicates_and_contained_partials():
    boxes = np.array([[0, 0, 100, 200], [2, 2, 102, 198], [10, 120, 90, 200], [300, 0, 400, 200]], float)
    confs = np.array([0.9, 0.8, 0.5, 0.7])
    assert merge_boxes(boxes, confs).tolist() == [0, 3]


def test_touches_cut_only_on_interior_edges():
    win = (1000, 0, 1640, 640)
    assert touches_cut(np.array([1002, 100, 1100, 300.0]), win, (1440, 2560))      # cut by the window's left edge
    assert not touches_cut(np.array([1100, 0, 1200, 300.0]), win, (1440, 2560))    # top edge is the frame edge


def test_keep_mask_filters():
    p = PerceptionProfile(person_conf=0.3, min_box_h_frac=0.05, edge_margin_frac=0.05, edge_conf=0.6,
                          ignore=[IgnoreRegion("g", [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])])
    shape = (1000, 1000)
    boxes = np.array([[480, 400, 520, 600],      # centre inside the ignore region
                      [100, 100, 140, 200],      # fine
                      [100, 100, 110, 120],      # too short
                      [5, 300, 45, 400],         # at the edge, conf 0.4 < 0.6
                      [5, 600, 45, 700]], float)  # at the edge, conf 0.9
    confs = np.array([0.5, 0.5, 0.5, 0.4, 0.9])
    assert keep_mask(p, boxes, confs, shape).tolist() == [False, True, False, False, True]


def test_profile_roundtrip(tmp_path):
    p = PerceptionProfile(camera="ch14", role="aisle", imgsz=960, slices=SliceSpec(2, 2, 0.25),
                          ignore=[IgnoreRegion("a", [[0, 0], [1, 0], [1, 1]])], min_track_age_s=1.0,
                          new_track_thresh=0.5, provenance={"label_kind": "REAL-UNLABELLED"})
    save_profile(p, tmp_path / "p.yaml")
    q = load_profile(tmp_path / "p.yaml")
    assert q == p


def test_profile_rejects_unknown_role(tmp_path):
    (tmp_path / "p.yaml").write_text("role: spaceship\n")
    with pytest.raises(ValueError):
        load_profile(tmp_path / "p.yaml")


def test_ghost_found_and_confident_person_kept():
    rng = np.random.default_rng(0)
    shape = (1000, 2000)
    dets = []
    for i in range(30):
        b, c = [[1700, 100, 1760, 240]], [0.35 + 0.05 * rng.random()]      # static low-confidence ghost
        b.append([800 + 3 * i, 300, 900 + 3 * i, 600]); c.append(0.85)     # a person who stands still, confident
        dets.append((np.array(b, float) + rng.normal(0, 1, (2, 4)), np.array(c)))
    g = find_ghosts(dets, shape)
    assert len(g) == 1 and g[0].box[0] > 0.8 and g[0].frac_frames == 1.0
    poly = np.array(ghost_polygon(g[0]))
    assert poly[:, 0].min() < g[0].box[0] and poly[:, 0].max() > g[0].box[2]


def test_ghost_ignores_rare_detections():
    dets = [(np.array([[10, 10, 50, 90.0]]), np.array([0.3])) if i < 3 else (np.zeros((0, 4)), np.zeros(0)) for i in range(30)]
    assert find_ghosts(dets, (200, 200)) == []


def test_track_gate_confirms_after_min_age():
    from bree.events.observations import PersonObs
    from bree.perception.gate import TrackGate
    g = TrackGate(fps=10, min_age_s=0.5)       # 5 frames
    seen = []
    for f in range(7):
        out = g.filter([PersonObs(1, (0, 0, 1, 1), 0.9, None)] + ([PersonObs(2, (0, 0, 1, 1), 0.9, None)] if f < 3 else []))
        seen.append(sorted(p.track_id for p in out))
    assert seen[:4] == [[], [], [], []] and seen[4] == [1] and seen[6] == [1]    # id 2 vanished before confirming


def test_lens_roundtrip_and_fit():
    shape = (1440, 2560)
    pts = np.array([[100, 100], [2400, 200], [1280, 720], [300, 1300]], float)
    for k in (-0.3, -0.1, 0.05):
        assert np.allclose(distort_points(undistort_points(pts, k, shape), k, shape), pts, atol=1e-3)
    # Straight lines seen through k1=-0.25 barrel distortion: fitting recovers k1.
    true = -0.25
    lines = []
    for y in (200, 500, 900, 1250):
        u = np.stack([np.linspace(100, 2460, 12), np.full(12, y)], 1)
        lines.append(distort_points(u, true, shape))
    k, res = fit_k1(lines, shape)
    assert abs(k - true) < 0.02 and res < 1.0


def test_match_and_summarize():
    shape = (100, 200)
    boxes = np.array([[90, 40, 110, 60], [10, 10, 20, 30]], float)     # centres (0.5, 0.5) and (0.075, 0.2)
    s = match_frame([(0.5, 0.5), (0.9, 0.9)], boxes, shape)
    assert (s.n_label, s.n_pred, s.tp) == (2, 2, 1)
    r = summarize([s, match_frame([], np.zeros((0, 4)), shape)])
    assert r["recall"] == 0.5 and r["precision"] == 0.5 and r["exact_count"] == 1.0 and r["empty_label_frames"] == 1


def test_label_loader_normalises_all_channels():
    from bree.perception.labels import DATA_DIR, load_labels
    if not (DATA_DIR / "real" / "labels").exists():
        pytest.skip("no real labels on this machine")
    chans = {l.channel for l in load_labels()}
    assert chans <= {7, 10, 12, 14, 15, 29} and len(chans) >= 4
    assert all(l.split in ("dev", "heldout") for l in load_labels())


@pytest.mark.vision
def test_calibrate_smoke_on_synthetic_clip(tmp_path):
    """The calibration script runs end to end on a tiny synthetic clip and writes a loadable profile."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    from bree.perception.calibrate import calibrate
    path = str(tmp_path / "clip.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    rng = np.random.default_rng(1)
    for _ in range(30):
        w.write(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    w.release()
    prof, report = calibrate(path, "cam_test", "aisle", samples=12, device="cpu")
    save_profile(prof, tmp_path / "p.yaml")
    assert load_profile(tmp_path / "p.yaml").camera == "cam_test"
    assert report["selection"]["label_kind"].startswith("REAL-UNLABELLED") and set(report["candidates"]) >= {"s640"}


def test_fixture_mask_and_finder():
    from bree.perception.ghost import find_fixtures
    from bree.perception.profile import fixture_mask
    dets = [(np.array([[1000, 700, 1060, 760.0]]), np.array([0.9 if i % 2 else 0.3])) for i in range(30)]
    fx = find_fixtures(dets, (1440, 2560))
    assert len(fx) == 1
    p = PerceptionProfile(fixtures=[IgnoreRegion("t", __import__("bree.perception.ghost", fromlist=["x"]).ghost_polygon(fx[0], 0.2))])
    boxes = np.array([[1000, 700, 1060, 760.0], [200, 200, 260, 260.0]])
    assert fixture_mask(p, boxes, (1440, 2560)).tolist() == [False, True]
