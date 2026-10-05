#!/usr/bin/env python
"""Fine-tune the sim-trained SKU detector on REAL labelled frames, once they exist.

    PYTHONPATH=src .venv/bin/python scripts/train/finetune_real.py --data data/real_sku/data.yaml
    .venv/bin/python scripts/train/finetune_real.py --manifest out/review/datasets/review/manifest.json

`--manifest` takes the label export of the review store (bree.review.labels: reviewer-confirmed boxes from
alerts) and turns it into the same kind of dataset first (bree.train.dataset.manifest_to_yolo: split by
alert, unknown classes left out). The review store's retrain hook can call this script directly:
    .venv/bin/python -m bree.review --store out/review retrain --min-new 50 \
        --command ".venv/bin/python scripts/train/finetune_real.py --manifest"

`--data` is a normal YOLO dataset of whole camera frames (images/{train,val}, labels/{train,val},
names = the store's real SKUs). Label a few hundred frames from the installed cameras: shelf views,
items in hands, items on the counter (any labelling tool that exports YOLO boxes).

What it does:
  1. cuts frames and labels into 640 px tiles at native resolution, like the simulated set (bree.train.dataset)
  2. starts from the sim-trained weights (the backbone has learned small packaged goods on shelves; the
     class head is rebuilt when the real SKU list differs from the simulator's invented one)
  3. trains with the first `--freeze` layers frozen and a low learning rate, so a few hundred frames are enough
  4. reports mAP on the real val tiles before (only meaningful when the class names match) and after
  5. writes data/synth/weights/real_sku.pt (+ .json record); use it with BREE_SKU_WEIGHTS=...

The sim-trained numbers say nothing about real accuracy. The real val split here is the first real number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from bree.train.backend import DEFAULT_WEIGHTS, sku_weights
from bree.train.dataset import TILE, manifest_to_yolo, tile_labels, tile_origins
from bree.train.train import train


def tile_yolo_dataset(data_yaml: Path, out: Path) -> Path:
    """Whole-frame YOLO dataset -> 640 px tile dataset. Returns the new data.yaml."""
    d = yaml.safe_load(data_yaml.read_text())
    root = Path(d.get("path", data_yaml.parent))
    root = root if root.is_absolute() else data_yaml.parent / root
    for split in ("train", "val"):
        img_dir = root / d[split]
        lab_dir = Path(str(img_dir).replace("images", "labels"))
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for f in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            lab = lab_dir / f"{f.stem}.txt"
            anns = []
            for line in lab.read_text().splitlines() if lab.exists() else []:
                c, cx, cy, bw, bh = line.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                anns.append({"cls": int(c), "bbox": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]})
            for y in tile_origins(h):
                for x in tile_origins(w):
                    labs = tile_labels(anns, x, y)
                    if not labs and anns:      # keep empty tiles only for frames with no labels at all (background)
                        continue
                    crop = img[y:y + TILE, x:x + TILE]
                    th, tw = crop.shape[:2]
                    name = f"{f.stem}_{x}_{y}"
                    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out / "labels" / split / f"{name}.txt").write_text("\n".join(
                        f"{a['cls']} {(b[0] + b[2]) / 2 / tw:.6f} {(b[1] + b[3]) / 2 / th:.6f} {(b[2] - b[0]) / tw:.6f} {(b[3] - b[1]) / th:.6f}" for a, b in labs))
    names = d["names"] if isinstance(d["names"], dict) else dict(enumerate(d["names"]))
    y = out / "data.yaml"
    y.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": names}, sort_keys=False))
    return y


def main(argv=None) -> None:
    from ultralytics import YOLO
    from bree.hw import detect_hardware
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="YOLO data.yaml of real labelled whole frames")
    src.add_argument("--manifest", help="manifest.json of a review-store label export (python -m bree.review ... export)")
    ap.add_argument("--base", default=None, help="starting weights (default: the sim-trained detector)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--freeze", type=int, default=10, help="layers frozen from the input side (10 = the backbone)")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=detect_hardware().device)
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS.with_name("real_sku.pt")))
    ap.add_argument("--note", default="fine-tuned on REAL labelled frames, starting from the sim-trained detector")
    a = ap.parse_args(argv)
    # A review export holds head-pixelated frames under the store's retention rule. The copies made here
    # (whole frames, tiles) go in a temp folder and are deleted when training ends, so nothing outlives the store's purge.
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="bree_review_ft_")) if a.manifest else None
    try:
        _run(a, Path(a.data) if a.data else manifest_to_yolo(a.manifest, tmp / "yolo"))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _run(a, data: Path) -> None:
    from ultralytics import YOLO
    base = sku_weights(a.base)
    tiles = tile_yolo_dataset(data, data.parent / (data.parent.name + "_tiles"))
    names = list(yaml.safe_load(tiles.read_text())["names"].values())
    before = None
    if names == list(YOLO(str(base)).names.values()):      # same SKU list: the sim detector can be scored as is
        r = YOLO(str(base)).val(data=str(tiles), split="val", imgsz=TILE, device=a.device, plots=False, verbose=False)
        before = {"map50": float(r.box.map50), "map50_95": float(r.box.map)}
    rec = train(str(tiles), str(base), a.epochs, a.batch, a.device, Path(a.out).stem, Path(a.out), note=a.note,
                freeze=a.freeze, lr0=a.lr0, optimizer="AdamW", warmup_epochs=1)
    r = YOLO(a.out).val(data=str(tiles), split="val", imgsz=TILE, device=a.device, plots=False, verbose=False)
    rec["val_before_finetune"] = before
    rec["val_after_finetune"] = {"map50": float(r.box.map50), "map50_95": float(r.box.map)}
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
