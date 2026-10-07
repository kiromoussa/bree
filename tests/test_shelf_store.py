"""Shelf events without a person box (bree/shelf/diff.py), store-wide floor tracks (bree/track/floor.py) and the
repeat filter of the store runner (bree/shelf/store.py). The join of shelf events and people is bree/events/shelf.py,
tested in tests/test_association.py. SYNTHETIC pictures and boxes with known truth."""
from __future__ import annotations

import numpy as np
import pytest

from bree.calib.camera import Camera
from bree.shelf.diff import ShelfDiff, slot_boxes
from bree.shelf.store import one_act_per_reach
from bree.track.floor import FloorConfig, FloorTracker, floor_point

cv2 = pytest.importorskip("cv2")


def look_at(pos, target, f=800.0, res=(1280, 720)) -> Camera:
    fwd = np.subtract(target, pos) / np.linalg.norm(np.subtract(target, pos))
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return Camera(f, res[0] / 2, res[1] / 2, np.stack([right, np.cross(fwd, right), fwd]), np.asarray(pos, float), res)


def slot(i, x, sku):      # a shelf along x at z = 0, facing +z
    return {"id": f"S{i}", "skuId": sku, "fixtureId": "G1", "position": [x, 1.0, -0.2], "size": [0.12, 0.2, 0.4],
            "face": [x, 1.0, 0.0], "normal": [0, 0, 1]}


SLOTS = [slot(i, 0.2 * i, f"sku{i}") for i in range(5)]
CAM = look_at([0.4, 1.3, 2.5], [0.4, 1.0, 0.0])


def shelf_picture(cam, present):
    img = np.full((cam.resolution[1], cam.resolution[0], 3), 90, np.uint8)
    boxes, keep = slot_boxes(cam, SLOTS, 1.0, 9.0)
    for b, i in zip(boxes, keep):
        if present[i]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (30 + 40 * i, 200, 250 - 40 * i), -1)
    return img


def play(sd, img, n):
    return [e for _ in range(n) for e in sd.update(img)]


def test_take_then_put_back_is_read_from_the_shelf_alone():
    sd = ShelfDiff("cam", CAM, SLOTS, 10.0)
    full, missing = shelf_picture(CAM, [True] * 5), shelf_picture(CAM, [True, True, False, True, True])
    assert play(sd, full, 20) == []
    arm = full.copy()
    cv2.rectangle(arm, (500, 0), (900, 720), (10, 10, 10), -1)       # something big in front: never an event
    assert play(sd, arm, 10) == []
    evs = play(sd, missing, 10)
    assert [(e["kind"], e["slot_id"], e["sku_id"], e["source"]) for e in evs] == [("take", "S2", "sku2", "shelf_diff")]
    assert 1.9 <= evs[0]["t_start"] <= 2.1 and evs[0]["point_3d"] == [0.4, 1.0, 0.0]      # when the shelf was last as before
    assert play(sd, missing, 20) == []                               # reported once
    back = play(sd, full, 10)
    assert [(e["kind"], e["slot_id"]) for e in back] == [("put", "S2")] and back[0]["undoes"] == evs[0]["eids"]      # the put names the take it undoes


def test_a_person_standing_still_is_not_a_shelf_event():
    sd = ShelfDiff("cam", CAM, SLOTS, 10.0)
    full = shelf_picture(CAM, [True] * 5)
    play(sd, full, 5)
    person = full.copy()
    cv2.rectangle(person, (300, 100), (700, 720), (200, 180, 160), -1)
    assert play(sd, person, 30) == [] and play(sd, full, 10) == []


OVER = {"o1": look_at([0.0, 3.0, 6.0], [2.0, 0.0, 1.0], f=500.0), "o2": look_at([6.0, 3.0, 0.0], [2.0, 0.0, 2.0], f=500.0)}


def box_of(cam, x, z, height=1.7, legs_hidden=False):
    pts = np.array([[x + dx, y, z + dz] for dx in (-0.2, 0.2) for dz in (-0.2, 0.2) for y in ((0.9 if legs_hidden else 0.0), height)])
    px, _ = cam.project(pts)
    return [px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max(), 0.9]


def test_floor_point_uses_the_head_when_a_shelf_hides_the_legs():
    cam = OVER["o1"]
    whole, _, _ = floor_point(cam, box_of(cam, 2.0, 2.0)[:4], FloorConfig())
    cut, _, _ = floor_point(cam, box_of(cam, 2.0, 2.0, legs_hidden=True)[:4], FloorConfig())
    assert np.linalg.norm(whole - [2.0, 2.0]) < 0.35 and np.linalg.norm(cut - [2.0, 2.0]) < 0.45


def walk(tracker, path, t0=0.0, hidden=(), fps=10.0):
    ids = []
    for i, (x, z) in enumerate(path):
        t = t0 + i / fps
        boxes = {c: np.array([] if any(a <= t < b for a, b in hidden) else [box_of(cam, x, z)]).reshape(-1, 5) for c, cam in OVER.items()}
        ids += [v for got in tracker.update(t, boxes).values() for v in got if v is not None]
    return ids


def test_one_identity_through_a_gap_and_an_exit_at_the_door():
    tr = FloorTracker(OVER, door_xz=[0.0, 4.0])
    path = [(0.3 + 0.08 * i, 3.8 - 0.05 * i) for i in range(50)] + [(4.3 - 0.08 * i, 1.3 + 0.05 * i) for i in range(50)]
    ids = walk(tr, path, hidden=[(2.0, 4.0)])
    assert set(ids) == {1}
    for i in range(30):
        tr.update(10.0 + i / 10, {c: np.zeros((0, 5)) for c in OVER})
    assert [(k.id, k.state) for k in tr.people()] == [(1, "exited")]


def test_two_people_who_stood_on_one_spot_keep_their_own_identity_by_clothing_colour():
    red, blue = [1.0] + [0.0] * 51, [0.0] * 10 + [1.0] + [0.0] * 41
    tr = FloorTracker(OVER, door_xz=[0.0, 4.0])
    a = [(0.3 + 0.07 * i, 3.8 - 0.06 * i) for i in range(30)]                   # in at the door, to (2.3, 2.0)
    b = [(0.3 + 0.07 * i, 3.8 - 0.02 * i) for i in range(30)] + [(2.3, 3.2 - 0.06 * i) for i in range(20)]   # to the same spot, later
    # both stand on that spot for 2 s, then walk off in opposite directions
    pa = a + [(2.3, 2.0)] * 40 + [(2.3 + 0.07 * i, 2.0) for i in range(30)]
    pb = [None] * 20 + b + [(2.3, 2.0)] * 10 + [(2.3 - 0.07 * i, 2.0) for i in range(30)]
    seen = {"red": [], "blue": []}
    for i in range(len(pa)):
        dets = [(name, xz, col) for name, path, col in (("red", pa, red), ("blue", pb, blue)) if i < len(path) and (xz := path[i]) is not None]
        boxes = {c: [{"bbox": box_of(cam, *xz)[:4], "conf": 0.9, "app": col * 2} for _, xz, col in dets] for c, cam in OVER.items()}
        got = tr.update(i / 10, boxes)
        for j, (name, _, _) in enumerate(dets):
            seen[name].append(got["o1"][j])
    first, last = {n: next(v for v in ids if v is not None) for n, ids in seen.items()}, {n: ids[-1] for n, ids in seen.items()}
    assert first["red"] != first["blue"] and last == first                       # each ends with the identity it came in with
    assert all(k.uncertain is None for k in tr.people())                         # and the meeting leaves no doubt mark
    assert tr.counts.get("swaps_put_right", 0) + tr.counts.get("meetings_told_apart", 0) >= 1 and tr.counts.get("encounter_marks", 0) >= 1


def test_one_reach_read_twice_is_one_take_and_two_shoppers_stay_two():
    ev = lambda t, x, sku, **kw: {"kind": "take", "t": t, "t_start": t, "point_3d": [x, 1.0, 0.0], "sku_id": sku, "source": "shelf_diff", "cameras": ["a"], **kw}  # noqa: E731
    acts = [ev(5.0, 0.2, "sku1"), ev(5.6, 0.6, "sku3", source="both", detector_frames=6), ev(6.0, 0.4, "sku2"), ev(20.0, 0.2, "sku1")]
    got = one_act_per_reach(acts, [1, 1, 2, 1])
    assert [(g["t"], g["sku_id"], g["by"], g["repeats"]) for g in got] == [(5.0, "sku3", 1, 1), (6.0, "sku2", 2, 0), (20.0, "sku1", 1, 0)]
    assert len(one_act_per_reach(acts[:2], [None, None])) == 2                    # nobody attached: nothing is merged


def shifted(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), img.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def test_a_shifted_picture_is_put_back_in_place_before_comparing():
    sd = ShelfDiff("cam", CAM, SLOTS, 10.0)
    full, missing = shelf_picture(CAM, [True] * 5), shelf_picture(CAM, [True, True, False, True, True])
    play(sd, full, 10)
    assert play(sd, shifted(full, 6, -4), 30) == []                  # 3 px and 2 px at working scale: no event from the shift alone
    assert abs(sd.shift[0] - 3) < 0.5 and abs(sd.shift[1] + 2) < 0.5
    evs = play(sd, shifted(missing, 6, -4), 10)
    assert [(e["kind"], e["slot_id"]) for e in evs] == [("take", "S2")]
    assert evs[0]["t_start"] <= evs[0]["t_end"] and evs[0]["point_sigma_m"] == 0.06


def test_a_large_brightness_change_is_undone_and_reported():
    sd = ShelfDiff("cam", CAM, SLOTS, 10.0)
    full, missing = shelf_picture(CAM, [True] * 5), shelf_picture(CAM, [True, True, False, True, True])
    play(sd, full, 10)
    dim = lambda im: cv2.convertScaleAbs(im, alpha=0.5)              # outside the 0.8 to 1.25 band of a lamp flicker
    assert play(sd, dim(full), 30) == []
    assert [s["status"] for s in sd.status] == ["brightness_changed"]
    assert [(e["kind"], e["slot_id"]) for e in play(sd, dim(missing), 10)] == [("take", "S2")]
    assert [(e["kind"], e["slot_id"]) for e in play(sd, dim(full), 10)] == [("put", "S2")]


def test_a_picture_that_changed_everywhere_is_reported_and_a_new_reference_is_taken():
    from bree.shelf.events import unreliable_windows
    sd = ShelfDiff("cam", CAM, SLOTS, 10.0)
    full = shelf_picture(CAM, [True] * 5)
    play(sd, full, 10)
    other = cv2.resize(np.random.default_rng(0).integers(0, 255, (45, 80, 3), dtype=np.uint8), full.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    assert play(sd, other, 30) == []                                 # nothing is read while the camera is unreliable
    assert [s["status"] for s in sd.status] == ["unreliable", "reference_reset"]
    assert unreliable_windows(sd.status) == {"cam": [(1.0, 2.0)]}
    assert play(sd, other, 30) == [] and len(sd.status) == 2         # the new picture is the shelf now


def test_a_swap_put_right_while_one_of_the_two_has_another_meeting_open_does_not_crash():
    from bree.track.floor import Track
    tr = FloorTracker(OVER, door_xz=[0.0, 4.0])
    hist = lambda k: np.eye(52)[[k, k + 1]] * 5.0      # noqa: E731  a clothing colour: 2 x 52 histogram sums
    a, b, c = (Track(i + 1, np.array([float(i), 2.0, 0.0, 0.0]), 0.0, 9.0, state="live", app=hist(10 * i), app_n=5) for i in range(3))
    n = tr.cfg.app_frames
    tr.meetings = [{"t0": 1.0, "a": a, "b": b, "fa": [hist(10)] * n, "fb": [hist(0)] * n, "marks": []},      # a now looks like b: swapped
                   {"t0": 1.0, "a": a, "b": c, "fa": [], "fb": [], "marks": []}]
    tr._colours(1.0 + tr.cfg.app_wait_s + 1.0, {})
    assert (a.id, b.id) == (2, 1) and tr.meetings == [] and tr.counts.get("swaps_put_right") == 1


def feed(tr, frames, t0=0.0):
    """frames: per frame {camera: [(x, z, colour)]} -> every id given out."""
    ids = []
    for i, f in enumerate(frames):
        boxes = {c: [{"bbox": box_of(cam, x, z)[:4], "conf": 0.9, "app": col * 2} for x, z, col in f.get(c, [])] for c, cam in OVER.items()}
        ids += [v for got in tr.update(t0 + i / 10, boxes).values() for v in got if v is not None]
    return ids


RED, BLUE = [1.0] + [0.0] * 51, [0.0] * 10 + [1.0] + [0.0] * 41


def test_a_second_track_of_a_tracked_person_is_not_a_new_person_and_takes_over_when_theirs_is_lost():
    tr = FloorTracker(OVER, door_xz=[0.0, 4.0], cfg=FloorConfig(gate_m=0.6))      # as tight as a keypoint placement makes the gate
    come = [{c: [(0.3 + 0.07 * i, 3.8 - 0.06 * i, RED)] for c in OVER} for i in range(30)]        # in at the door, to (2.3, 2.0)
    # for 3 s the second camera places them about 0.9 m off (too far for one track, further than a second box), then only that camera sees them
    apart = [{"o1": [(2.3, 2.0, RED)], "o2": [(2.85, 2.0, RED)]}] * 30
    alone = [{"o2": [(2.85, 2.0, RED)]}] * 30
    ids = feed(tr, come + apart + alone)
    assert set(ids) == {1} and [k.id for k in tr.people()] == [1]
    assert tr.counts.get("second_track_frames", 0) > 0 and tr.counts.get("births_inside", 0) == 0 and tr.counts.get("hand_backs", 0) == 1


def test_far_from_where_they_were_lost_and_in_other_clothes_is_somebody_else_when_the_gate_is_on():
    def run(cfg):
        tr = FloorTracker(OVER, door_xz=[0.0, 4.0], cfg=cfg)
        come = [{c: [(0.3 + 0.07 * i, 3.8 - 0.06 * i, RED)] for c in OVER} for i in range(30)]    # red, lost at (2.3, 2.0)
        other = [{}] * 20 + [{c: [(4.6, 0.6, BLUE)] for c in OVER}] * 40                          # blue, 2.7 m away, 2 s later, not at the door
        feed(tr, come + other)
        return sorted(k.id for k in tr.people())
    assert run(None) == [1]                              # position alone: the lost person, back
    assert run(FloorConfig(other_m=2.0)) == [1, 2]       # an entry nobody saw; id 1 stays lost


def test_an_act_can_be_timed_by_the_readings_no_put_took_back():
    ev = lambda t, kind="take", **kw: {"kind": kind, "t": t, "t_start": t, "point_3d": [0.2, 1.0, 0.0], "sku_id": "sku1", "source": "both", "cameras": ["a"], **kw}  # noqa: E731
    acts = [ev(22.4, eids=["a"]), ev(25.9, eids=["b"]), ev(26.6, eids=["c"]), ev(27.6, "put", undoes=["a"])]
    assert [g["t"] for g in one_act_per_reach(acts, [1, 1, 1, 1], standing=False) if g["kind"] == "take"] == [22.4, 26.6]       # 4.2 s after the first reading: a second take
    assert [g["t"] for g in one_act_per_reach(acts, [1, 1, 1, 1]) if g["kind"] == "take"] == [25.9]


def test_a_lens_term_bends_the_picture_and_the_line_of_sight_undoes_it():
    import dataclasses
    cam = look_at([0.0, 2.9, 0.0], [3.0, 0.0, 3.0], f=520.0, res=(1920, 1080))
    lens = dataclasses.replace(cam, k_div=0.035)
    X = np.array([[9.0, 0.0, -1.0], [1.0, 0.9, 4.0], [3.0, 0.0, 3.0]])
    px, z = lens.project(X)
    assert np.linalg.norm(px[0] - cam.project(X)[0][0]) > 15 and np.allclose(px[2], cam.project(X)[0][2], atol=1e-6)      # bent off the axis, not on it
    for p, x in zip(px, X):                              # the ray through the bent pixel passes through the point
        d = lens.ray(*p)
        assert np.linalg.norm(np.cross(d, x - lens.C)) < 1e-6
    far, zf = lens.project(np.array([[-40.0, 0.0, 30.0]]))      # far off the axis, where the lens formula folds back: stays out of the frame
    assert not lens.in_frame(far, zf)[0]
    assert Camera.from_dict(lens.to_dict()).k_div == 0.035 and "k_div" not in cam.to_dict()


def test_two_standing_readings_of_one_camera_at_two_slots_are_two_takes():
    ev = lambda t, x, sku, eid: {"kind": "take", "t": t, "t_start": t, "point_3d": [x, 1.0, 0.0], "sku_id": sku, "source": "shelf_diff", "cameras": ["a"], "eids": [eid]}  # noqa: E731
    acts = [ev(17.2, 0.2, "sku1", "a:188:S1"), ev(20.5, 0.6, "sku3", "a:245:S3")]
    assert len(one_act_per_reach(acts, [1, 1])) == 1                                             # as before: one reach
    assert [g["sku_id"] for g in one_act_per_reach(acts, [1, 1], two_slots=2.0)] == ["sku1", "sku3"]
    assert len(one_act_per_reach([acts[0], ev(18.0, 0.6, "sku3", "a:190:S3")], [1, 1], two_slots=2.0)) == 1       # the same moment: one reach read at two slots
    assert len(one_act_per_reach([acts[0], ev(20.5, 0.6, "sku3", "b:245:S3")], [1, 1], two_slots=2.0)) == 1       # another camera: its slot may be off
    put = {"kind": "put", "t": 21.0, "t_start": 21.0, "point_3d": [0.2, 1.0, 0.0], "sku_id": "sku1", "source": "shelf_diff", "cameras": ["a"], "undoes": ["a:188:S1"]}
    assert len([g for g in one_act_per_reach([*acts, put], [1, 1, 1], two_slots=2.0) if g["kind"] == "take"]) == 1      # the first was an arm over the slot
