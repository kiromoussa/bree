"""The concealment cue on a made-up scene: one camera, one or two people, hand-written detector looks."""
import numpy as np
import pytest

from bree.calib.camera import Camera
from bree.concealment.cue import ConcealConfig, analyse, conceal_cues, sightings, sweeps

FPS = 10.0
LAYOUT = {"slots": [], "skus": [], "poi": {"register": [8.0, 0, 8.0]}}


class Person:
    def __init__(self, pid, path, staff=False):
        self.id, self.path, self.staff = pid, path, staff

    def at(self, t):
        if t < self.path[0][0] - 0.5 or t > self.path[-1][0] + 0.5:
            return None
        ts = np.array([p[0] for p in self.path])
        i = int(np.argmin(abs(ts - t)))
        return np.array(self.path[i][1:3]), float(abs(ts[i] - t))


def camera():      # 3 m up at the origin side, looking along +z and 30 degrees down
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    R = np.array([[1.0, 0, 0], [0, -c, -s], [0, -s, c]])      # rows: right, down, forward
    return Camera(1000.0, 960.0, 540.0, R, np.array([0.0, 3.0, -4.0]), (1920, 1080))


def box_at(cam, x, y, z, half=20):
    (u, v), _ = cam.project([[x, y, z]])[0][0], None
    return [u - half, v - half, u + half, v + half]


def looks_for(cam, steps):
    """steps: [(t, [(x, y, z, sku)] items, [(x, y, z)] hands)]"""
    return {"CAM": [{"f": int(round(t * FPS)), "items": [[*box_at(cam, x, y, z), 0.9, sku, 0.9, True, False] for x, y, z, sku in items],
                     "hands": [[*box_at(cam, x, y, z), 0.9] for x, y, z in hands]} for t, items, hands in steps]}


def stand(pid, x=0.0, z=0.0, t0=0.0, t1=20.0):
    return Person(pid, [(round(t, 1), x, z) for t in np.arange(t0, t1 + 0.05, 0.1)])


def take(t=2.0, sku="soda", pt=(0.0, 1.2, 0.6)):
    return {"t": t, "kind": "take", "sku_id": sku, "point_3d": list(pt)}


def scene(item_until, hands_until=12.0):
    """A person takes an item at 2 s; the item is seen in the hand (at the body) until item_until, the hand until hands_until."""
    cam = camera()
    steps = [(t, [(0.2, 1.0, 0.0, "soda")] if t <= item_until else [], [(0.2, 0.95, 0.0)]) for t in np.arange(2.4, hands_until, 0.2)]
    return cam, looks_for(cam, steps)


def test_item_and_hand_go_to_the_person_under_the_line_of_sight():
    cam, looks = scene(6.0)
    held, hands = sightings(looks, {"CAM": cam}, [stand(1), stand(2, x=3.0)], FPS, ConcealConfig(), LAYOUT)
    assert set(held) == {1} and set(hands) == {1}
    assert abs(held[1][0].y - 1.0) < 0.05 and held[1][0].off < 0.25


def test_item_gone_while_the_hand_is_still_seen_is_a_cue():
    cam, looks = scene(item_until=5.0)
    cues, score = conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, model=None)
    assert len(cues) == 1 and cues[0]["person_id"] == 1 and cues[0]["sku_id"] == "soda" and abs(cues[0]["t"] - 5.3) < 0.31
    assert score[1]["score"] >= 0.9 and score[1]["concealed"] == 1


def test_item_seen_to_the_end_is_no_cue():
    cam, looks = scene(item_until=12.0)
    cues, score = conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, model=None)
    assert cues == [] and score[1]["concealed"] == 0


def test_one_false_sighting_later_does_not_hide_it_and_one_missed_sighting_does_not_make_one():
    cam, looks = scene(item_until=5.0)
    looks["CAM"][-3]["items"] = [[*box_at(cam, 0.2, 1.0, 0.0), 0.9, "soda", 0.9, True, False]]       # one stray sighting near the end
    assert len(conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, model=None)[0]) == 1
    cam, looks = scene(item_until=12.0)
    for r in looks["CAM"][10:14]:
        r["items"] = []                                                                             # four looks in a row without the item
    assert conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, model=None)[0] == []


def test_a_put_back_is_not_a_concealment():
    cam, looks = scene(item_until=5.0)
    acts = [take(), {"t": 5.2, "kind": "put", "sku_id": "soda", "point_3d": [0.0, 1.2, 0.6]}]
    takes = analyse(looks, {"CAM": cam}, [stand(1)], acts, [1, 1], LAYOUT, FPS)
    assert [t.where for t in takes] == ["shelf"] and takes[0].p == 0.0


def test_no_take_no_cue_and_staff_never_score():
    cam, looks = scene(item_until=5.0)
    assert conceal_cues(looks, {"CAM": cam}, [stand(1)], [], [], LAYOUT, FPS, model=None)[0] == []
    staff = stand(1)
    staff.staff = True
    assert conceal_cues(looks, {"CAM": cam}, [staff], [take()], [1], LAYOUT, FPS, model=None) == ([], {})


def test_window_ends_at_the_register_zone():
    cam, looks = scene(item_until=5.0)
    p = stand(1)
    p.path = [(t, x, z) if t < 5.2 else (t, 8.0, 8.0) for t, x, z in p.path]       # at the pay point from 5.2 s on
    takes = analyse(looks, {"CAM": cam}, [p], [take()], [1], LAYOUT, FPS)
    assert takes[0].stop_why == "register" and takes[0].feats["empty_bins"] <= 1


def test_item_never_seen_in_the_hand_is_not_scored():
    cam, looks = scene(item_until=0.0)
    takes = analyse(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS)
    assert takes[0].where == "not_seen"


def test_stock_on_its_own_slot_is_not_a_held_item():
    cam, looks = scene(item_until=12.0)
    layout = {**LAYOUT, "skus": [{"id": "soda", "size": [0.07, 0.12, 0.07]}],
              "slots": [{"id": "S1", "skuId": "soda", "position": [0.2, 1.0, 0.1], "face": [0.2, 1.0, 0.0], "size": [0.07, 0.12, 0.2]}]}
    held, _ = sightings(looks, {"CAM": cam}, [stand(1)], FPS, ConcealConfig(), layout)
    assert held == {}


def test_shelf_sweep():
    acts = [take(2.0, pt=(0, 1, 0)), take(4.0, pt=(0.3, 1, 0)), take(6.0, pt=(0.6, 1, 0)), take(30.0, pt=(5, 1, 0))]
    assert sweeps(acts, [1, 1, 1, 2], ConcealConfig()) == {1: 6.0}
    assert sweeps(acts[:2] + acts[3:], [1, 1, 1], ConcealConfig()) == {}


def test_hand_crop_is_centred_and_padded():
    from bree.concealment.where import SIZE, crop
    im = np.zeros((100, 200, 3), np.uint8)
    im[40:60, 90:110] = 255
    c = crop(im, [90, 40, 110, 60])
    assert c.shape == (SIZE, SIZE, 3) and c[SIZE // 2, SIZE // 2].min() == 255 and c[2, 2].max() == 0
    assert crop(im, [0, 0, 10, 10])[0, 0].tolist() == [114, 114, 114]


def test_model_file_scores_like_its_coefficients():
    from bree.concealment.cue import score_take
    m = {"features": ["cusum"], "log": [], "mean": [5.0], "scale": [5.0], "coef": [2.0], "intercept": 0.0}
    assert score_take({"cusum": 5.0}, m, ConcealConfig()) == pytest.approx(0.5)
    assert score_take({"cusum": 20.0}, m, ConcealConfig()) > 0.99


def test_plain_takes_do_not_add_up_to_a_shopper_score():
    cam, looks = scene(item_until=12.0)          # the item is seen to the end: no evidence
    m = {"features": ["cusum"], "log": [], "mean": [10.0], "scale": [10.0], "coef": [2.0], "intercept": 0.0}      # p = 0.12 at a cusum of 0
    cues, score = conceal_cues(looks, {"CAM": cam}, [stand(1)], [take(1.0), take(1.2, pt=(3, 1, 0)), take(1.4, pt=(6, 1, 0)), take(1.6, pt=(9, 1, 0))], [1, 1, 1, 1], LAYOUT, FPS, model=m)      # far apart: no sweep
    assert cues == [] and score[1]["score"] <= 0.15         # only "left without passing the register zone" is in it


def _record(**kw):
    return {"alert_id": "A1", "person_id": 1, "tier": "review", "group": [], "reasons": [], "audit_log": [],
            "unpaid_items": [{"category": "soda", "concealed": True}], **kw}


def test_tier_rule_raises_only_unpaid_concealed_and_sure_of_identity():
    from bree.concealment.tier import retier
    high, low = {1: {"score": 0.8}}, {1: {"score": 0.2}}
    assert retier([_record()], high)[0]["tier"] == "alert"
    assert retier([_record()], low)[0]["tier"] == "review"                                                # score under the bar
    assert retier([_record(unpaid_items=[{"category": "soda", "concealed": False}])], high)[0]["tier"] == "review"      # nothing concealed
    unsure = _record(audit_log=["  46.8s identity uncertain: ids 6 and 4 were within 0.2 m"])
    assert retier([unsure], high)[0]["tier"] == "review" and retier([unsure], high, ignore_identity=True)[0]["tier"] == "alert"
    assert retier([], high) == [] and retier([_record(unpaid_items=[])], high)[0]["tier"] == "review"     # never makes a record: no unpaid item, no alert
    rec = _record()
    assert retier([rec], None)[0]["tier"] == "alert" and rec["tier"] == "review"                          # no scores given: the mark alone; input unchanged


def test_another_product_seen_on_them_says_nothing():
    """Shelf stock seen past the person (another product) after the hide: with own_product it neither hides the
    concealment nor counts as "no item"; without it every such look is the item still in the hand."""
    cam = camera()
    steps = [(t, [(0.2, 1.0, 0.0, "soda" if t <= 5.0 else "chips")] if t <= 5.0 or int(round(t * 5)) % 2 else [], [(0.2, 0.95, 0.0)]) for t in np.arange(2.4, 14.0, 0.2)]
    looks = looks_for(cam, steps)
    assert len(conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, ConcealConfig(own_product=True), model=None)[0]) == 1
    assert conceal_cues(looks, {"CAM": cam}, [stand(1)], [take()], [1], LAYOUT, FPS, ConcealConfig(own_product=False), model=None)[0] == []


def test_window_runs_past_a_take_of_another_product_when_asked():
    cam, looks = scene(item_until=5.0)
    acts = [take(), take(t=6.0, sku="chips")]
    assert conceal_cues(looks, {"CAM": cam}, [stand(1)], acts, [1, 1], LAYOUT, FPS, ConcealConfig(past_next=False), model=None)[0] == []      # 4.5 s window: too few moments
    assert [c["sku_id"] for c in conceal_cues(looks, {"CAM": cam}, [stand(1)], acts, [1, 1], LAYOUT, FPS, ConcealConfig(past_next=True), model=None)[0]] == ["soda"]
