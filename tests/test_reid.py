"""Body re-ID (bree/track/reid.py): features never see the head, fusion separates clothing, the engine and the
multi-camera handoff use it only to veto / relink, and features are forgotten when the visit ends.
The last test is the regression guard on real footage (MOT16, needs weights + data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bree.detect.toy import synth_keypoints
from bree.events.engine import EngineRules, EventEngine
from bree.events.observations import FrameObs, PersonObs
from bree.events.types import EventType as E
from bree.events.zones import load_store_config
from bree.track import reid as R
from bree.track.multicam import MultiCamIdentity, homography

ROOT = Path(__file__).resolve().parents[1]
FPS = 15.0
RED, BLUE, GREEN, GREY = (40, 40, 220), (220, 60, 40), (60, 200, 60), (128, 128, 128)   # BGR


class HueEmbedder:
    """Stand-in for the DINOv2 embedder: a unit vector from the crop's mean colour (no model weights in unit tests)."""

    def __call__(self, crops):
        e = np.stack([c.reshape(-1, 3).mean(0) - 127.0 for c in crops]).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-6)


def scene(people: dict[int, tuple], head=(200, 180, 160)):
    """Image + PersonObs list. people: id -> (foot_x, foot_y, shirt BGR, trousers BGR)."""
    img = np.full((720, 1280, 3), 90, np.uint8)
    obs = []
    for tid, (fx, fy, shirt, legs) in people.items():
        box = (fx - 22, fy - 158, fx + 22, fy)
        x1, y1, x2, y2 = map(int, box)
        img[y1:y1 + 36, x1:x2] = head
        img[y1 + 36:y1 + 98, x1:x2] = shirt
        img[y1 + 98:y2, x1:x2] = legs
        obs.append(PersonObs(tid, box, 0.9, synth_keypoints(box, [])))
    return img, obs


def feats(shirt, legs, head=(200, 180, 160), t=0.0):
    ext = R.ReidExtractor(HueEmbedder())
    img, obs = scene({1: (600, 400, shirt, legs)}, head)
    ext.update(img, obs, t)
    return obs[0].reid


# ------------------------------------------------------------------ features


def test_head_pixels_never_reach_a_feature():
    a, b = feats(RED, BLUE, head=(200, 180, 160)), feats(RED, BLUE, head=(0, 255, 0))
    assert np.array_equal(a.emb, b.emb) and np.array_equal(a.colors, b.colors)


def test_head_below_the_shoulder_line_is_greyed_out():
    img, obs = scene({1: (600, 400, RED, BLUE)})
    kp = obs[0].keypoints
    kp[:5, 1] += 60                                        # bowed head: head keypoints now inside the body crop
    crop = R.body_crop(img, obs[0].bbox, kp)
    y = int(kp[0, 1] - R.body_parts(obs[0].bbox, kp, img.shape[:2])["body"][1])
    assert (crop[y, crop.shape[1] // 2] == 127).all()


def test_colours_survive_a_lighting_change_and_tell_clothes_apart():
    img, obs = scene({1: (600, 400, RED, BLUE)})
    base = R.part_colors(img, obs[0].bbox, obs[0].keypoints)
    dim = R.part_colors((img * 0.7).astype(np.uint8), obs[0].bbox, obs[0].keypoints)
    img2, obs2 = scene({1: (600, 400, GREEN, GREY)})
    other = R.part_colors(img2, obs2[0].bbox, obs2[0].keypoints)
    assert R._bhatt(base[0], dim[0]) > 0.9 and R._bhatt(base[0], other[0]) < 0.1
    assert R._bhatt(base[1], dim[1]) > 0.9 and R._bhatt(base[1], other[1]) < 0.1


def test_fusion_scores_same_clothes_above_different_clothes():
    same = R.match_prob([feats(RED, BLUE)], feats(RED, BLUE, t=1.0))
    diff = R.match_prob([feats(RED, BLUE)], feats(GREEN, GREY, t=1.0))
    assert same > 0.5 and diff < 0.01
    assert R.match_prob([], feats(RED, BLUE)) is None


def test_bag_and_shape_cues():
    assert R.carries_bag((0, 0, 100, 200), np.array([[20, 80, 60, 140]])) is True
    assert R.carries_bag((0, 0, 100, 200), np.array([[300, 80, 360, 140]])) is False
    assert R.carries_bag((0, 0, 100, 200), None) is None
    assert R.body_shape(synth_keypoints((0, 0, 44, 158), [])) is not None
    assert R.body_shape(None) is None


def test_gallery_forgets_old_features():
    g = []
    R.remember(g, R.ReidFeatures(0.0), 0.0, max_age_s=100)
    R.remember(g, R.ReidFeatures(90.0), 90.0, max_age_s=100)
    assert len(g) == 2
    R.remember(g, None, 150.0, max_age_s=100)
    assert [f.t for f in g] == [90.0]


def test_logistic_fit_separates():
    from bree.eval.reid_bench import fit_logistic
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0.8, 0.1, (200, 2)), rng.normal(0.2, 0.1, (200, 2))])
    y = np.arange(400) < 200
    w, b = fit_logistic(X, y)
    assert (((X @ w + b) > 0) == y).mean() > 0.98


# ------------------------------------------------------------------ engine stitching


class World:
    def __init__(self, reid=True, **rules):
        store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
        self.engine = EventEngine(store, EngineRules.from_dict({**store.rules, "reid": reid, **rules}))
        self.ext = R.ReidExtractor(HueEmbedder())
        self.f, self.events = 0, []

    def step(self, people: dict[int, tuple], n: int = 1):
        for _ in range(n):
            img, obs = scene(people)
            self.ext.update(img, obs, self.f / FPS)
            self.events += self.engine.update(FrameObs(self.f, self.f / FPS, obs, []))
            self.f += 1
        return self

    def enters(self):
        return sorted(e.person_id for e in self.events if e.type == E.ENTER)


def test_same_clothes_stitch_different_clothes_do_not():
    w = World().step({1: (600, 400, RED, BLUE)}, n=20).step({}, n=30)
    w.step({2: (610, 400, RED, BLUE)}, n=5)
    assert w.enters() == [1] and w.engine.alias == {2: 1}
    w = World().step({1: (600, 400, RED, BLUE)}, n=20).step({}, n=30)
    w.step({2: (610, 400, GREEN, GREY)}, n=5)              # same place and time, another person
    assert w.enters() == [1, 2]
    assert any("looks different" in m for m in w.engine.log)


def test_flag_off_keeps_position_only_behaviour():
    w = World(reid=False).step({1: (600, 400, RED, BLUE)}, n=20).step({}, n=30)
    w.step({2: (610, 400, GREEN, GREY)}, n=5)
    assert w.enters() == [1] and w.engine.people[1].gallery == []


def test_appearance_settles_which_of_two_lost_people():
    both = {1: (600, 400, RED, BLUE), 3: (640, 400, GREEN, GREY)}
    w = World().step(both, n=20).step({}, n=30)
    w.step({2: (620, 400, GREEN, GREY)}, n=5)              # between them: the position guard alone skips this
    assert w.engine.alias == {2: 3}


def test_two_lost_people_in_the_same_clothes_stay_ambiguous():
    both = {1: (600, 400, RED, BLUE), 3: (640, 400, RED, BLUE)}
    w = World().step(both, n=20).step({}, n=30)
    w.step({2: (620, 400, RED, BLUE)}, n=5)
    assert w.engine.alias == {} and 2 in w.enters()


def test_lost_at_the_shelf_relinked_at_the_register_by_appearance():
    """Shelf to register: lost for 20 s, reappears 500 px away. Position-only stitching cannot link this."""
    for reid, expect in ((True, {2: 1}), (False, {})):
        w = World(reid=reid).step({1: (300, 400, RED, BLUE)}, n=20).step({}, n=int(20 * FPS))
        w.step({2: (800, 400, RED, BLUE)}, n=5)
        assert w.engine.alias == expect


def test_relink_needs_a_walkable_distance():
    w = World(reid_max_speed=0.1).step({1: (300, 400, RED, BLUE)}, n=20).step({}, n=int(5 * FPS))
    w.step({2: (1100, 400, RED, BLUE)}, n=5)               # 800 px in 5 s at 0.1 body heights/s: not possible
    assert w.engine.alias == {}


def test_features_are_forgotten_when_the_visit_ends_or_the_window_passes():
    w = World(reid_long_s=5.0).step({1: (600, 400, RED, BLUE)}, n=20)
    assert w.engine.people[1].gallery
    w.step({}, n=int(11 * FPS))                            # lost for longer than any stitch window
    assert w.engine.people[1].gallery == []
    assert w.ext.tracks == {}                              # the extractor's per-track state is gone too


def test_features_are_not_serialised():
    from bree.pipeline import _obs_to_json
    img, obs = scene({1: (600, 400, RED, BLUE)})
    R.ReidExtractor(HueEmbedder()).update(img, obs, 0.0)
    assert obs[0].reid is not None
    text = json.dumps(_obs_to_json(FrameObs(0, 0.0, obs, []), []))
    assert "reid" not in text and "emb" not in text


# ------------------------------------------------------------------ cross-camera handoff


def _cams(**kw):
    img = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
    HA = homography(img, [(0, 0), (6, 0), (6, 4), (0, 4)])
    HB = homography(img, [(4, 0), (10, 0), (10, 4), (4, 4)])
    return MultiCamIdentity({"A": HA, "B": HB}, **kw)


def _p(tid, floor_x, cam_x0, f, y_px=600):
    px = (floor_x - cam_x0) / 6 * 1280
    return PersonObs(tid, (px - 20, y_px - 150, px + 20, y_px), 0.9, reid=f)


def test_handoff_rejects_someone_who_looks_different():
    red, green = feats(RED, BLUE), feats(GREEN, GREY)
    for reid, same in ((True, False), (False, True)):
        mc = _cams(reid=reid)
        mc.update("A", FrameObs(0, 0.0, [_p(1, 5.0, 0, red)]))
        mc.update("B", FrameObs(0, 0.5, [_p(9, 5.2, 4, green)]))       # right place and time, other clothes
        assert (mc.local_to_global[("A", 1)] == mc.local_to_global[("B", 9)]) is same


def test_handoff_picks_the_matching_person_of_two_nearby():
    red, green = feats(RED, BLUE), feats(GREEN, GREY)
    mc = _cams(reid=True)
    mc.update("A", FrameObs(0, 0.0, [_p(1, 5.0, 0, red), _p(2, 5.3, 0, green, y_px=560)]))
    mc.update("B", FrameObs(0, 0.2, [_p(9, 5.1, 4, green, y_px=590)]))    # nearer to the red person
    assert mc.local_to_global[("B", 9)] == mc.local_to_global[("A", 2)]


def test_handoff_between_cameras_that_do_not_overlap():
    """Cooler camera to register camera with a 10 s unseen walk: linked by appearance if walkable."""
    red = feats(RED, BLUE)
    mc = _cams(reid=True)
    mc.update("A", FrameObs(0, 0.0, [_p(1, 1.0, 0, red)]))
    mc.update("B", FrameObs(0, 10.0, [_p(9, 9.0, 4, red)]))            # 8 m in 10 s
    assert mc.local_to_global[("B", 9)] == mc.local_to_global[("A", 1)]
    mc = _cams(reid=True)
    mc.update("A", FrameObs(0, 0.0, [_p(1, 1.0, 0, red)]))
    mc.update("B", FrameObs(0, 1.0, [_p(9, 9.0, 4, red)]))             # 8 m in 1 s: not the same person
    assert mc.local_to_global[("B", 9)] != mc.local_to_global[("A", 1)]


def test_register_payment_lands_on_the_shopper_who_picked():
    """Shelf to register to exit: a pick under one tracker id, the register visit under another. With the
    relink the receipt clears that shopper's basket; without it the pick is left on a person who never pays."""
    from bree.events.types import Event, LineItem, Payment
    from bree.ledger import build_ledger
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    reg = next(z for z in store.zones if z.kind == "register")
    rx, ry = np.mean([p[0] for p in reg.polygon]), np.mean([p[1] for p in reg.polygon])
    paid = {}
    for reid in (True, False):
        w = World(reid=reid)
        ledger = build_ledger(store)
        w.step({1: (300, 400, RED, BLUE)}, n=20)
        w.events.append(Event(type=E.PICK, t=w.f / FPS, person_id=1, item="soda_bottle", zone="cooler", confidence=0.9))
        w.step({}, n=int(15 * FPS))
        w.step({2: (rx, ry, RED, BLUE)}, n=int(4 * FPS))               # new tracker id standing at the register
        t_pay = w.f / FPS - 1.0
        w.step({}, n=int(3 * FPS))
        for ev in w.events:
            ledger.on_event(ev)
        ledger.on_payment(Payment(terminal="pos_1", t=t_pay, items=[LineItem(category="soda_bottle")]))
        paid[reid] = [li.category for li in ledger.people[1].paid]
    assert paid == {True: ["soda_bottle"], False: []}


# ------------------------------------------------------------------ regression guard (real footage)


@pytest.mark.vision
def test_reid_does_not_regress_on_mot16():
    """Runs the quick benchmark (two MOT16 sequences) and compares with tests/fixtures/reid_guard.json.
    Fails if IDF1 drops or false merges rise, with re-ID on, versus the recorded numbers or versus the
    position-only guard measured in the same run."""
    from bree.eval import reid_bench as B
    if not all((B.MOT / s / "gt/gt.txt").exists() for s in B.QUICK_SEQS) or not (ROOT / "models/yolo26s.pt").exists():
        pytest.skip("MOT16 or model weights not present (make data / make setup)")
    ref = json.loads((ROOT / "tests/fixtures/reid_guard.json").read_text())
    res = B.run(quick=True, toy=False)
    on, off = res["tracking"]["guard_reid"], res["tracking"]["guard"]
    assert on["idf1"] >= ref["idf1"] - ref["idf1_tolerance"], (on["idf1"], ref)
    assert on["false_merges"] <= ref["false_merges"], (on["false_merges"], ref)
    assert on["idf1"] >= off["idf1"] - ref["idf1_tolerance"] and on["false_merges"] <= off["false_merges"]
    assert res["reid"]["fused"]["rank1"] >= ref["rank1"] - ref["rank1_tolerance"]
    # Flags off = the position-only guard, unchanged by closed-world identity (which is off unless a store asks for it).
    same = ref["position_only_guard_same_run"]
    assert abs(off["idf1"] - same["idf1"]) <= ref["idf1_tolerance"] and off["false_merges"] == same["false_merges"], off
    assert "closed_world" not in off and res["tracking"]["closed_world"]["closed_world"]["assigned"] > 0
