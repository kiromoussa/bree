import numpy as np

from bree.conceal import Track, Video, eer, frame_scores, pr_auc, roc_auc, rule_scores, windows


def _track(n, wrist_at_torso):
    kps = np.zeros((n, 17, 3), np.float32)
    kps[:, :, 2] = 1.0
    kps[:, 5], kps[:, 6] = (40, 20, 1), (60, 20, 1)      # shoulders
    kps[:, 11], kps[:, 12] = (42, 60, 1), (58, 60, 1)    # hips
    kps[:, 9] = kps[:, 10] = (50, 40, 1) if wrist_at_torso else (100, 0, 1)
    return Track(np.arange(n), kps, np.tile(np.array([30, 0, 70, 100], np.float32), (n, 1)))


def test_rule_counts_wrist_in_torso_over_window():
    assert rule_scores(_track(30, True))[-1] == 1.0
    assert rule_scores(_track(30, False))[-1] == 0.0
    assert rule_scores(_track(30, True), win=24)[11] == 12 / 24   # causal: only frames seen so far


def test_frame_scores_take_max_over_people():
    v = Video("1_1", "1", "test", 30, {1: _track(30, True), 2: _track(30, False)}, np.zeros(30, np.int8))
    assert frame_scores(v, rule_scores)[-1] == 1.0


def test_windows_are_causal_and_padded():
    f = np.arange(5, dtype=np.float32)[:, None]
    w = windows(f, win=3)
    assert w.shape == (5, 3, 1)
    assert w[0, :, 0].tolist() == [0, 0, 0] and w[4, :, 0].tolist() == [2, 3, 4]


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert eer(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_nan_keypoints_become_undetected():
    from bree.conceal import clean_kps
    k = np.ones((2, 17, 3), np.float32)
    k[0, 3, 0] = np.nan
    c = clean_kps(k)
    assert not np.isnan(c).any() and c[0, 3].tolist() == [0, 0, 0] and c[1, 3, 2] == 1


def test_pr_auc_ignores_order_of_ties():
    y1, y2 = np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert pr_auc(y1, s) == pr_auc(y2, s) == 0.5


def test_split_gaps_cuts_tracks_at_missing_frames():
    from bree.conceal import split_gaps
    t = _track(6, True)
    t.frames = np.array([0, 1, 2, 10, 11, 12])
    parts = split_gaps({7: t})
    assert sorted(len(p.frames) for p in parts.values()) == [3, 3]


def test_triggers_need_half_a_second_at_the_video_fps():
    from bree.conceal import trigger_frames
    s, f = np.ones(20), np.arange(20)
    assert trigger_frames(s, f, 0.5, fps=15) == [7]      # 8 frames = 0.5 s at 15 fps
    assert trigger_frames(s, f, 0.5, fps=10) == [4]      # 5 frames at 10 fps


def test_metrics_mask_drops_empty_frames():
    from bree.conceal import metrics
    y = np.array([0, 0, 1, 1]); s = np.array([0.0, 0.6, 0.5, 0.9]); m = np.array([False, True, True, True])
    r = metrics(y, s, m)
    assert r["n_frames"] == 3 and r["all_frames_auc_roc"] > r["auc_roc"]
