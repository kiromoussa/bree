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
