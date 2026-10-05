"""Hand and held-item detector: labels, held-out split, weight versions, the tile merge with a hand class, score maths.
No model weights needed."""
from __future__ import annotations

import numpy as np

from bree.train import backend as be
from bree.train import dataset as ds
from bree.train.eval_hands import summarise


def test_hand_boxes_become_labels_of_the_hand_class():
    m = {"annotations": [{"sku": "fizzo_cola", "kind": "hand", "geo": "can", "bbox": [10, 10, 40, 60], "full": [10, 10, 40, 60], "vis_px": 900, "px_eff": 30}],
         "hands": [{"bbox": [35, 40, 60, 70], "vis_px": 400, "reaching": True, "holding": True, "px_eff": 28},
                   {"bbox": [0, 0, 4, 4], "vis_px": 9, "reaching": False, "holding": False}]}       # too small to be a label
    anns = ds.frame_anns(m, {"can": 0.6})
    assert [(a["sku"], a["kind"]) for a in anns] == [("fizzo_cola", "hand"), (ds.HAND, "body_hand")]
    assert anns[1]["reaching"] and anns[1]["holding"]


def test_heldout_seeds_are_never_train_or_val():
    got = {s: ds.split_from(s, 3000) for s in range(2000, 3010)}
    assert all(got[s] == "test" for s in range(3000, 3010)) and "test" not in {got[s] for s in range(2000, 3000)}
    assert {"train", "val"} == {got[s] for s in range(2000, 2042)}
    assert ds.split_from(1000) == ds.split_of(1000)


def test_weights_are_picked_by_version_name(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "DEFAULT_WEIGHTS", tmp_path / "sim_sku.pt")
    for n in ("sim_sku.pt", "sim_sku_hands_v2.pt"):
        (tmp_path / n).write_bytes(b"x")
    monkeypatch.delenv("BREE_SKU_WEIGHTS", raising=False)
    assert be.sku_weights().name == "sim_sku.pt"                      # the old weights stay the default
    assert be.sku_weights("sim_sku_hands_v2").name == "sim_sku_hands_v2.pt"
    monkeypatch.setenv("BREE_SKU_WEIGHTS", "sim_sku_hands_v2")
    assert be.sku_weights().name == "sim_sku_hands_v2.pt"


def test_detect_keeps_a_hand_that_overlaps_the_item_it_holds():
    class R:      # one Ultralytics result: an item and a hand box nearly on top of each other
        orig_shape = (640, 640)

        class boxes:
            xyxy = conf = cls = None

            def __len__(self):
                return 2
        boxes = boxes()

    class T:
        def __init__(self, a):
            self.a = np.array(a, np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self.a
    R.boxes.xyxy, R.boxes.conf, R.boxes.cls = T([[100, 100, 200, 200], [105, 105, 200, 200]]), T([0.9, 0.8]), T([0, 1])

    class Model:
        names = {0: "fizzo_cola", 1: "hand"}

        def predict(self, tiles, **kw):
            return [R for _ in tiles]
    det = be.SkuDetector.__new__(be.SkuDetector)
    det.model, det.names, det.hand_cls = Model(), Model.names, 1
    det.device, det.conf, det.batch, det.tile, det.overlap = "cpu", 0.25, 16, 640, 128
    (ib, ic, ik), (hb, hc) = det.detect(np.zeros((640, 640, 3), np.uint8))
    assert ib.tolist() == [[100, 100, 200, 200]] and ik.tolist() == [0] and hb.tolist() == [[105, 105, 200, 200]]
    assert len(det(np.zeros((640, 640, 3), np.uint8))[0]) == 1       # the old call returns items only
    (ib, _, _), (hb, _) = det.detect(np.zeros((640, 640, 3), np.uint8), np.array([[300, 300, 400, 400]]))
    assert len(ib) == 0 and len(hb) == 0                              # nothing centred in the region asked for


def test_summary_counts_right_sku_at_20_px_and_reaching_hands():
    item = lambda px, pred: {"sku": "a", "kind": "hand", "vis": 0.9, "px_eff": px, "short_side": 20, "camera_kind": "shelf",   # noqa: E731
                             "detected": pred is not None, "pred": pred}
    hand = lambda reach, hit: {"reaching": reach, "holding": False, "short_side": 40, "px_eff": 30, "camera_kind": "shelf",    # noqa: E731
                               "iou50": hit, "iou30": True, "centre": True}
    s = summarise([item(30, "a"), item(30, "b"), item(30, None), item(12, None)], [hand(True, True), hand(True, False), hand(False, False)], [])
    assert s["in_hand_20px_plus"]["sku_right"] == {"n": 3, "k": 1, "rate": 0.3333} and s["in_hand_20px_plus"]["found"]["k"] == 2
    assert s["hand_recall_reaching"]["iou50"] == {"n": 2, "k": 1, "rate": 0.5} and s["hand_recall_all_labelled"]["iou50"]["n"] == 3
