"""Shelf events in the shared contract (bree/shelf/events.py, hand.py): the shelf comparison fused with an item seen
in a hand, without any person box. SYNTHETIC pictures and a stand-in detector with known truth."""
from __future__ import annotations

import numpy as np
import pytest

from bree.shelf.diff import DiffConfig, slot_boxes
from bree.shelf.events import ShelfCamera, ShelfConfig, fuse_views
from bree.shelf.hand import skin_hand
from test_shelf_store import CAM, SLOTS, shelf_picture

cv2 = pytest.importorskip("cv2")

CONTRACT = {"camera_id", "t", "t_start", "t_end", "kind", "slot_id", "sku_id", "sku_conf", "count", "source", "hand_px", "point_3d", "point_sigma_m", "evidence"}
LAYOUT = {"slots": SLOTS, "skus": [], "fixtures": []}
BOXES, _ = slot_boxes(CAM, SLOTS, 1.0, 9.0)
W, H = BOXES[2][2] - BOXES[2][0], BOXES[2][3] - BOXES[2][1]


class HeldItem:
    """Stand-in SKU detector: reports the box the test drew for the item in the hand, as `sku`."""
    names = {i: f"sku{i}" for i in range(5)}

    def __init__(self, sku=2):
        self.box, self.sku = None, sku

    def __call__(self, image, rois=None):
        if self.box is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        return np.array([self.box], float), np.array([0.8]), np.array([self.sku])


def take(sc, det, slot=2, steps=10, colour=(110, 200, 170)):
    """Slot `slot` goes empty while its item moves down and out of the picture in a hand. Returns all events."""
    present = [i != slot for i in range(5)]
    full, missing, out = shelf_picture(CAM, [True] * 5), shelf_picture(CAM, present), []
    for _ in range(20):
        out += sc.update(full)
    x0, y0 = BOXES[slot][:2]
    for k in range(1, steps + 1):
        img = missing.copy()
        det.box = [x0, y0 + 30 * k, x0 + W, y0 + 30 * k + H]
        cv2.rectangle(img, (int(det.box[0]), int(det.box[1])), (int(det.box[2]), int(det.box[3])), colour, -1)
        out += sc.update(img)
    det.box = None
    for _ in range(60):
        out += sc.update(missing)
    return out + sc.finish()


def test_shelf_change_with_the_item_seen_in_a_hand_is_one_event_from_both_cues(tmp_path):
    det = HeldItem()
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0, ShelfConfig(diff=DiffConfig(evidence_dir=str(tmp_path))), detector=det)
    evs = take(sc, det)
    assert len(evs) == 1 and CONTRACT <= set(evs[0])
    e = evs[0]
    assert (e["kind"], e["slot_id"], e["sku_id"], e["source"], e["count"]) == ("take", "S2", "sku2", "both", 1)
    assert 1.9 <= e["t"] <= 2.4 and e["point_3d"] == [0.4, 1.0, 0.0] and e["hand_px"][1] > BOXES[2][1]
    before, after = cv2.imread(e["evidence"]["before"]), cv2.imread(e["evidence"]["after"])
    assert before.shape == after.shape and np.abs(before.astype(int) - after.astype(int)).max() > 60


def test_without_a_detector_the_shelf_alone_still_reports_the_take():
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0)
    evs = take(sc, HeldItem())
    assert [(e["kind"], e["slot_id"], e["source"]) for e in evs] == [("take", "S2", "shelf_diff")]


def test_detector_names_the_neighbouring_slot_when_the_patch_is_between_two():
    """The shelf comparison says S2, the detector sees sku3 in the hand and S3 is a candidate: S3 it is."""
    det = HeldItem(sku=3)
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0, detector=det)
    e = {"camera_id": "cam", "t": 2.0, "t_start": 2.0, "t_end": 2.5, "kind": "take", "slot_id": "S2", "sku_id": "sku2", "sku_conf": 0.5, "count": 1,
         "source": "shelf_diff", "hand_px": [0, 0], "point_3d": SLOTS[2]["face"], "point_sigma_m": None, "evidence": {"before": None, "after": None, "frames": [20, 30]},
         "slots": [("S2", 0.4), ("S3", 0.35)]}
    cx, cy = (BOXES[2][0] + BOXES[2][2]) / 2, (BOXES[2][1] + BOXES[2][3]) / 2
    sc.cue.tracks.append({"sku": "sku3", "obs": [(22, cx + 20, cy + 30, 0.7), (24, cx + 20, cy + 80, 0.7)]})
    got = sc._confirm(e)
    assert (got["slot_id"], got["sku_id"], got["source"], got["t"], got["t_item"]) == ("S3", "sku3", "both", 2.0, 2.2)


def test_skin_blob_is_a_hand_point_only_inside_the_changed_mask():
    img = np.full((200, 200, 3), 90, np.uint8)
    cv2.circle(img, (60, 60), 12, (120, 150, 210), -1)        # skin tone (BGR)
    cv2.circle(img, (150, 150), 12, (120, 150, 210), -1)
    changed = np.zeros((100, 100), np.uint8)
    changed[15:45, 15:45] = 1
    got = skin_hand(img, changed, 0.5)
    assert len(got) == 1 and abs(got[0][0] - 60) < 6 and abs(got[0][1] - 60) < 6


def test_fuse_views_lets_a_detector_confirmed_view_decide_the_slot():
    a = {"camera_id": "a", "kind": "take", "t": 5.0, "t_start": 5.0, "t_end": 5.4, "slot_id": "S1", "sku_id": "sku1", "point_3d": [0.2, 1, 0],
         "slots": [("S1", 0.5), ("S2", 0.45)], "source": "shelf_diff", "sku_conf": 0.5}
    b = {**a, "camera_id": "b", "t": 5.6, "t_start": 5.3, "slot_id": "S2", "sku_id": "sku2", "point_3d": [0.4, 1, 0], "slots": [("S2", 0.5), ("S1", 0.45)], "source": "both"}
    other = {**a, "camera_id": "b", "t": 30.0, "t_start": 30.0, "t_end": 30.5}
    got = fuse_views([a, b, other], LAYOUT)
    assert [g["cameras"] for g in got] == [["a", "b"], ["b"]]
    assert (got[0]["slot_id"], got[0]["sku_id"], got[0]["source"], got[0]["t"]) == ("S2", "sku2", "both", 5.6)


def test_slot_watch_counts_units_from_the_row_position_of_the_front_item():
    """A camera looking along the shelf sees the next item of the row beside the front one: the detection that fits
    row position k says k items are gone. Take of one, take of two more, put of one."""
    from bree.shelf.diff import ShelfDiff
    from bree.shelf.slots import SlotWatch
    from test_shelf_store import look_at
    slots = [{**s, "depthCount": 4} for s in SLOTS]
    sd = ShelfDiff("cam", look_at([2.2, 1.3, 0.9], [0.4, 1.0, -0.2]), slots, 10.0)
    sd.update(np.zeros((720, 1280, 3), np.uint8))
    w = SlotWatch(sd)
    row = w.rows[2]
    look = lambda f, k: w.update(f, np.array([row[k] + [2, -1, 1, 2]]), ["sku2"])
    assert look(0, 0) == [] and look(1, 0) == [] and w.state[2] == 0          # learned, no event
    assert look(2, 0) == [] and look(4, 0) == []
    assert look(10, 1) == []                                                   # one look is not enough
    look(12, 1)
    (e,) = look(14, 1)
    assert (e["kind"], e["slot_id"], e["sku_id"], e["count"], e["source"], e["row"]) == ("take", "S2", "sku2", 1, "shelf_diff", [0, 1])
    assert CONTRACT <= set(e) and 0.1 <= e["t"] <= 1.0
    assert look(20, 3) == []
    look(22, 3)
    (e,) = look(24, 3)
    assert (e["kind"], e["count"]) == ("take", 2)
    look(30, 2)
    look(32, 2)
    (e,) = look(34, 2)
    assert (e["kind"], e["count"], e["t_end"]) == ("put", 1, 3.0)
    assert w.update(40, np.array([row[2] + [200, 0, 200, 0]]), ["sku2"]) == []          # an item of this SKU elsewhere (in a hand)


def test_a_second_take_of_the_same_facing_within_seconds_is_dropped_as_a_repeat():
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0)
    ev = lambda t, kind="take", slot=2: {"camera_id": "cam", "t": t, "t_start": t, "t_end": t + 0.5, "kind": kind, "slot_id": f"S{slot}", "sku_id": f"sku{slot}",
                                         "sku_conf": 1.0, "count": 1, "source": "shelf_diff", "point_3d": SLOTS[slot]["face"], "slots": [(f"S{slot}", 1.0)], "evidence": {"frames": []}}
    sc._step([ev(1.0)])
    sc._step([ev(3.0)])                      # the same facing again 2 s later: one act read twice
    sc._step([ev(3.5, slot=4)])              # another slot 0.4 m away: its own event
    sc._step([ev(20.0)])                     # long after: a new take
    sc._step([ev(26.0, "put"), ev(27.0)])    # a put in between (more than transient_s after the take): the next take counts
    got = sorted((e["t"], e["kind"], e["slot_id"]) for e in sc.finish())
    assert got == [(1.0, "take", "S2"), (3.5, "take", "S4"), (20.0, "take", "S2"), (26.0, "put", "S2"), (27.0, "take", "S2")]
    assert len(sc.dropped) == 1


def test_evidence_crops_are_written_when_a_folder_is_given(tmp_path):
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0, ShelfConfig(diff=DiffConfig(evidence_dir=str(tmp_path))))
    full, missing = shelf_picture(CAM, [True] * 5), shelf_picture(CAM, [True, True, False, True, True])
    evs = [e for im in [full] * 10 + [missing] * 50 for e in sc.update(im)] + sc.finish()
    assert len(evs) == 1 and all((tmp_path / evs[0]["evidence"][k].split("/")[-1]).exists() for k in ("before", "after"))


def test_a_take_is_timed_by_the_item_in_the_hand_when_the_slot_was_hidden_for_long():
    """Somebody stands in front of a slot for six seconds before an item leaves it: the take is when the item is first
    seen in a hand (less the usual lead), not when the slot was first hidden."""
    from types import SimpleNamespace
    sc = ShelfCamera("cam", CAM, LAYOUT, 10.0)
    s = LAYOUT["slots"][0]
    b = sc.diff.boxes[0] / sc.cfg.diff.scale
    u, v = float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2
    sc.cue = SimpleNamespace(tracks=[{"sku": s["skuId"], "obs": [(64, u, v, 0.8), (66, u + 5, v, 0.8)]}], hands=[])
    ev = lambda t0, t1: {"camera_id": "cam", "t": t0, "t_start": t0, "t_end": t1, "kind": "take", "slot_id": s["id"], "sku_id": s["skuId"],      # noqa: E731
                         "sku_conf": 0.9, "count": 1, "source": "shelf_diff", "point_3d": s["face"], "evidence": {"before": None, "after": None, "frames": []}}
    assert sc._confirm(ev(0.5, 6.5))["t"] == pytest.approx(6.1)        # hidden from 0.5 s, item in the hand at 6.4 s
    sc.cue.tracks[0].pop("used")
    assert sc._confirm(ev(6.0, 6.2))["t"] == 6.0                       # seen after the window: the window's start stands
    # a put: the item's track ends at the slot coming from further away (it went in), or starts there and leaves
    sc.cue.tracks = [{"sku": s["skuId"], "obs": [(60, u + 60, v, 0.8), (64, u + 2, v, 0.8)]}]
    assert sc._confirm({**ev(6.2, 6.6), "kind": "put"})["item_in"] is True
    sc.cue.tracks = [{"sku": s["skuId"], "obs": [(60, u + 2, v, 0.8), (64, u + 60, v, 0.8)]}]
    assert sc._confirm({**ev(6.2, 6.6), "kind": "put"})["item_in"] is False


def _reading(cam, slot, after, t=10.0, **kw):
    s = next(x for x in SLOTS if x["id"] == slot)
    return {"camera_id": cam, "t": t, "t_start": t, "t_end": t + 1.0, "kind": "take", "slot_id": slot, "sku_id": s.get("skuId"), "source": "shelf_diff",
            "point_3d": list(s["face"]), "slots": [(slot, 0.6)], "eids": [f"{cam}:{slot}"], "after": after, **kw}


def test_a_product_that_newly_stands_in_a_slot_is_a_put_not_a_take():
    from bree.shelf.events import arrivals
    a, b = SLOTS[2]["id"], SLOTS[0]["id"]
    planned = SLOTS[2].get("skuId")
    evs = [_reading("c1", a, [["foreign_sku", 0.9, False, 4]]),          # a product of another slot arrived: put
           _reading("c2", a, []),                                       # the same act through another camera: put
           _reading("c1", a, [[planned, 0.9, True, 2]], t=30.0),        # the slot's own product behind the one taken: take
           _reading("c1", b, [["foreign_sku", 0.9, True, 3]], t=50.0),  # it stood there from the start (planogram out of date): take
           _reading("c1", b, [["foreign_sku", 0.9, False, 0]], t=70.0), # never seen in a hand: one cue alone, take
           _reading("c1", b, None, t=90.0)]                             # an older run without the reading: unchanged
    got = arrivals(evs, LAYOUT)
    assert [e["kind"] for e in got] == ["put", "put", "take", "take", "take", "take"]
    assert got[0]["sku_id"] == got[1]["sku_id"] == "foreign_sku" and got[0]["item_in"] and "eids" not in got[0]
    assert evs[0]["kind"] == "take"       # the stored readings are not changed
