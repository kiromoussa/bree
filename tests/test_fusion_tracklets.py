"""bree.fusion.tracklets: single-camera association of cached detections (synthetic boxes)."""
import numpy as np

from bree.fusion import tracklets as T


def _det(x, y, conf=0.9, colour=0):
    app = np.zeros(104)
    app[colour] = 0.5
    app[52 + colour] = 0.5
    return {"bbox": [x, y, x + 80, y + 200], "conf": conf, "app": app.tolist()}


def test_two_people_crossing_keep_their_ids_by_colour():
    frames = []
    for i in range(60):
        t = 1000.0 + i * 0.2
        # A walks right, B walks left, they cross at the middle; A is colour 3, B colour 20
        frames.append((t, [_det(100 + i * 10, 300, colour=3), _det(700 - i * 10, 310, colour=20)]))
    tls = T.track_camera(frames, "x")
    assert len(tls) == 2
    for tl in tls:
        cols = {int(np.argmax(a[:52])) for a in tl.app}
        assert len(cols) == 1


def test_gap_and_low_confidence_rules():
    frames = []
    for i in range(30):
        t = 50.0 + i * 0.2
        persons = [_det(200 + i * 3, 100)] if not (10 <= i < 14) else []     # 0.8 s without a box: same track
        persons += [_det(900, 50, conf=0.3)]                                # a flicker that never gets confident: no track
        frames.append((t, persons))
    tls = T.track_camera(frames, "x")
    assert len(tls) == 1 and len(tls[0].t) >= 24


def test_static_track_is_marked():
    cfg = T.TrackerConfig()
    frames = [(10.0 + i * 0.2, [_det(500, 200)]) for i in range(150)]      # 30 s without moving
    tls = T.track_camera(frames, "x", cfg)
    assert len(tls) == 1 and tls[0].static


def test_at_and_colour_summary():
    frames = [(10.0 + i * 0.2, [_det(300 + i, 200)]) for i in range(20)]
    tl = T.track_camera(frames, "x")[0]
    assert tl.at(10.41) == 2 and tl.at(30.0) is None
    c = T.summary_colour(tl, (2560, 1440))
    assert c is not None and abs(c[:52].sum() - 0.5) < 1e-6
    assert T.app_distance(c, c) < 1e-6
