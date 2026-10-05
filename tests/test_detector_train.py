"""Sim-trained SKU detector: dataset builder, tile merge, evaluation maths, backend wiring.
No model weights needed except the test marked `vision`
(skipped when data/synth/weights/sim_sku.pt is absent)."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from bree.detect.base import Detections
from bree.train import dataset as ds
from bree.train.backend import DEFAULT_WEIGHTS, SimSkuBackend, merge_tiles
from bree.train.evaluate import confusion, curve, match, min_px
from bree.train.sim_eval_sku import product_recognition


def test_split_is_by_seed_and_stable():
    got = {s: ds.split_of(s) for s in range(1000, 1240)}
    assert set(got.values()) == {"train", "val", "test"}
    assert got == {s: ds.split_of(s) for s in range(1000, 1240)}
    n = {k: sum(v == k for v in got.values()) for k in ds.SPLITS}
    assert n["train"] > n["val"] and n["train"] > n["test"]


@pytest.mark.parametrize("size", [640, 1080, 1520, 1920, 2688])
def test_tiles_cover_the_frame_with_overlap(size):
    o = ds.tile_origins(size)
    assert o[0] == 0 and o[-1] == size - ds.TILE
    assert all(b - a <= ds.TILE - ds.OVERLAP for a, b in zip(o, o[1:]))


def test_tile_labels_keep_a_cut_box_only_if_half_is_inside():
    anns = [{"bbox": [600, 100, 700, 200]}, {"bbox": [630, 300, 730, 400]}, {"bbox": [10, 10, 60, 60]}]
    got = ds.tile_labels(anns, 0, 0)
    assert [b for _, b in got] == [[10, 10, 60, 60]]                     # 40% and 10% inside: dropped
    assert [b for _, b in ds.tile_labels(anns, 512, 0)] == [[88, 100, 188, 200], [118, 300, 218, 400]]


def test_visible_share_and_label_rule():
    ref = {"box": 0.8}
    a = {"geo": "box", "full": [0, 0, 100, 100], "vis_px": 4000, "bbox": [0, 0, 50, 100]}
    assert ds.visible_share(a, ref) == pytest.approx(0.5)
    assert ds.keep(a, 0.5) and not ds.keep(a, 0.2)
    assert not ds.keep({**a, "bbox": [0, 0, 4, 100]}, 0.5) and not ds.keep({**a, "vis_px": 10}, 0.5)
    assert ds.visible_share({**a, "full": None}, ref) == 0.0             # part of it is behind the camera


def _fake_frames(root, seeds):
    (root).mkdir()
    skus = [{"id": "a_red", "geo": "box", "family": ["a_red", "a_blue"]}, {"id": "a_blue", "geo": "box", "family": ["a_red", "a_blue"]}]
    (root / "skus.json").write_text(json.dumps(skus))
    for seed in seeds:
        d = root / f"s{seed}"
        d.mkdir()
        img = np.full((900, 1300, 3), 90, np.uint8)
        anns = []
        for k, (x, y) in enumerate([(100, 100), (700, 200), (1100, 700)]):
            cv2.rectangle(img, (x, y), (x + 60, y + 80), (0, 0, 255) if k % 2 == 0 else (255, 0, 0), -1)
            anns.append({"id": k + 1, "sku": skus[k % 2]["id"], "kind": "hand" if k == 2 else "shelf", "geo": "box", "slot": None,
                         "bbox": [x, y, x + 60, y + 80], "vis_px": 4800, "full": [x, y, x + 60, y + 80], "px_eff": 30.0, "px_item": 30.0})
        cv2.imwrite(str(d / "0_CAM.jpg"), img)
        (d / "0_CAM.json").write_text(json.dumps({"seed": seed, "capture": 0, "params": {}, "width": 1300, "height": 900, "t": 1.0,
                                                  "camera": {"id": "CAM", "kind": "shelf"}, "annotations": anns}))
        (d / "done").write_text("")


def test_build_writes_yolo_and_coco_without_seed_leak(tmp_path):
    seeds = list(range(1000, 1012))
    _fake_frames(tmp_path / "frames", seeds)
    meta = ds.build(tmp_path / "frames", tmp_path / "out")
    sp = meta["splits"]
    assert sorted(sp["train"]["seeds"] + sp["val"]["seeds"] + sp["test"]["seeds"]) == seeds
    assert all(ds.split_of(s) == k for k in ds.SPLITS for s in sp[k]["seeds"])
    assert "a_red" in (tmp_path / "out" / "data.yaml").read_text()
    lab = next((tmp_path / "out" / "labels" / "train").glob("*.txt")).read_text().split()
    assert int(lab[0]) in (0, 1) and all(0 <= float(v) <= 1 for v in lab[1:5])
    coco = json.loads((tmp_path / "out" / "coco_train.json").read_text())
    assert coco["annotations"] and coco["annotations"][0]["attributes"]["px_eff"] == 30.0
    assert sp["train"]["boxes_by_kind"].get("hand", 0) >= len(sp["train"]["seeds"])     # item-in-hand tiles are always kept
    assert len(list((tmp_path / "out" / "test_frames").glob("*.json"))) == len(sp["test"]["seeds"])


def test_merge_tiles_drops_duplicates_and_cut_fragments():
    boxes = np.array([[100, 100, 200, 200], [102, 101, 199, 202], [170, 100, 200, 200], [400, 400, 450, 450]], np.float32)
    b, c, k = merge_tiles(boxes, np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 2, 1, 3]), np.array([False, False, True, False]))
    assert b.tolist() == [[100, 100, 200, 200], [400, 400, 450, 450]] and k.tolist() == [1, 3]
    # a fragment of another class is a different item: kept
    b, _, k = merge_tiles(boxes[[0, 2]], np.array([0.9, 0.7]), np.array([1, 2]), np.array([False, True]))
    assert len(b) == 2


def test_match_curve_min_px_and_confusion():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], np.float32)
    pred = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [41, 0, 51, 10]], np.float32)
    assert match(gt, pred, np.array([0.9, 0.5, 0.8])).tolist() == [0, -1, 2]       # one prediction per item
    recs = ([{"px_eff": 12, "sku": "a", "pred": None, "detected": False}] * 40 + [{"px_eff": 22, "sku": "a", "pred": "b", "detected": True}] * 10
            + [{"px_eff": 22, "sku": "a", "pred": "a", "detected": True}] * 30 + [{"px_eff": 35, "sku": "a", "pred": "a", "detected": True}] * 40
            + [{"px_eff": 50, "sku": "b", "pred": "b", "detected": True}] * 5)
    rows = {r["px"]: r for r in curve(recs)}
    assert rows["10-15"]["found_rate"] == 0 and rows["20-25"]["sku_right_rate"] == 0.75 and rows["30-40"]["sku_right_rate"] == 1
    assert min_px(curve(recs), 0.9) == 30 and min_px(curve(recs), 0.7) == 20        # the 5-item bucket is too small to count
    c = confusion(recs, [{"id": "a", "family": ["a", "b"]}, {"id": "b", "family": ["a", "b"]}])["a/b"]
    assert c["rows"]["a"] == [70, 10, 0, 40] and c["confused_within_family"] == 10 and c["confused_within_family_rate"] == pytest.approx(10 / 85, abs=1e-4)


def test_backend_runs_products_around_people_only():
    class Base:
        last_bags = None

        def __call__(self, image):
            return Detections(np.array([[100, 100, 200, 300]], np.float32), np.array([0.9], np.float32), ["person"]), Detections.empty()

    class Sku:
        names = {0: "fizzo_cola", 1: "not_in_store"}

        def detect(self, image, rois=None):
            self.rois = rois
            return (np.array([[150, 150, 170, 190], [160, 160, 180, 200]], np.float32), np.array([0.8, 0.7], np.float32), np.array([0, 1])), \
                (np.array([[140, 180, 160, 200]], np.float32), np.array([0.6], np.float32))

    sku = Sku()
    backend = SimSkuBackend(Base(), sku, {"fizzo_cola": "fizzo_cola"})
    persons, products = backend(np.zeros((480, 640, 3), np.uint8))
    assert backend.last_hands[0].tolist() == [[140, 180, 160, 200]]
    assert len(persons) == 1 and products.labels == ["fizzo_cola"] and sku.rois.tolist() == [[75, 50, 225, 350]]

    class Nobody(Base):
        def __call__(self, image):
            return Detections.empty(True), Detections.empty()
    sku.rois = "not called"
    _, products = SimSkuBackend(Nobody(), sku)(np.zeros((480, 640, 3), np.uint8))
    assert len(products) == 0 and sku.rois == "not called"


def test_product_recognition_block(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [
        {"type": "pick", "t": 5.2, "item": "a", "zone": "G1"}, {"type": "pick", "t": 9.0, "item": "b", "zone": "G1"}, {"type": "enter", "t": 1.0}]))
    (tmp_path / "pipeline" / "frames_c.jsonl").write_text(json.dumps({"products": [1, 2, 3]}) + "\n" + json.dumps({"products": []}) + "\n")
    truth = [{"t": 5.0, "skuId": "a", "slotId": "s1"}, {"t": 9.5, "skuId": "a", "slotId": "s1"}, {"t": 30.0, "skuId": "a", "slotId": "s1"}]
    pr = product_recognition(truth, tmp_path, {"slots": [{"id": "s1", "fixtureId": "G1"}]})
    assert (pr["picks_matched"], pr["picks_sku_right"], pr["product_boxes"], pr["frames"]) == (2, 1, 3, 2)


@pytest.mark.vision
def test_trained_detector_reads_a_simulated_test_frame():
    frames = sorted((DEFAULT_WEIGHTS.parents[1] / "sku" / "test_frames").glob("*.json"))
    if not DEFAULT_WEIGHTS.exists() or not frames:
        pytest.skip("no trained weights or simulated test frames here (make sku-data sku-train)")
    from bree.train.backend import SkuDetector
    m = json.loads(frames[0].read_text())
    boxes, confs, cls = SkuDetector(device="cpu")(cv2.imread(str(frames[0].with_suffix(".jpg"))))
    hit = match(np.array([a["bbox"] for a in m["annotations"]], np.float32).reshape(-1, 4), boxes, confs)
    assert len(boxes) and (hit >= 0).mean() > 0.3


def test_review_manifest_feeds_the_finetune_dataset(tmp_path):
    """The review store's label export (SYNTHETIC alerts) is what scripts/train/finetune_real.py --manifest reads:
    same classes, alerts never split across train and val, unknown classes left out, tiles cut without error."""
    import importlib.util
    from pathlib import Path

    import yaml

    from bree.review.labels import export_labels
    from bree.review.synthetic import populate
    store, now = populate(tmp_path / "review", 30, 0)
    man = export_labels(store, now=now)
    det = [e for e in man["examples"] if e["task"] == "detector"]
    assert det
    y = ds.manifest_to_yolo(man["path"], tmp_path / "yolo")
    d = yaml.safe_load(y.read_text())
    assert list(d["names"].values()) == man["classes"]
    side = {}
    for split in ("train", "val"):
        for f in (tmp_path / "yolo" / "images" / split).iterdir():
            assert (tmp_path / "yolo" / "labels" / split / f"{f.stem}.txt").read_text().strip()
            side.setdefault(next(e["alert_id"] for e in det if e["key"] == f.stem), set()).add(split)
    assert all(len(s) == 1 for s in side.values()) and {"train", "val"} == set().union(*side.values())
    assert len(side) == len({e["alert_id"] for e in det if e.get("class_known", True)})
    spec = importlib.util.spec_from_file_location("finetune_real", Path(__file__).resolve().parents[1] / "scripts/train/finetune_real.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    tiles = ft.tile_yolo_dataset(y, tmp_path / "tiles")
    assert list((tmp_path / "tiles" / "labels" / "train").glob("*.txt")) and yaml.safe_load(tiles.read_text())["names"] == d["names"]


def test_build_without_a_train_scene_says_so(tmp_path):
    """Audit 2026-10-05: SKU_SEEDS=1000:1001 died with KeyError 'can' in visible_share."""
    seeds = [s for s in range(1000, 1040) if ds.split_of(s) != "train"][:2]
    _fake_frames(tmp_path / "frames", seeds)
    with pytest.raises(SystemExit, match="no train scenes"):
        ds.build(tmp_path / "frames", tmp_path / "out")
    assert [ds.split_of(s) for s in range(1000, 1004)].count("train") >= 1      # the README's smoke range has a train seed
    assert {ds.split_of(s) for s in range(1000, 1004)} == {"train", "val", "test"}
