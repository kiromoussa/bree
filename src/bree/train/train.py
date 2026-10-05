"""Train the SKU detector on the SIMULATED tiles (bree.train.dataset) with the repo's detector stack.

    python -m bree.train.train --data data/synth/sku/data.yaml --epochs 6

Ultralytics YOLO, 640 px tiles at native camera resolution. Writes the run under data/synth/runs/<name>,
copies the best weights to data/synth/weights/sim_sku.pt and records exactly what was trained and for how
long in data/synth/weights/sim_sku.json.

Fine-tuning on real labelled frames uses the same function (scripts/train/finetune_real.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from bree.train.backend import DEFAULT_WEIGHTS, ROOT

# Augmentation: the SKU is in the label art, so colour shifts stay small and nothing is mirrored;
# scale stays modest so "pixels across the item" keeps its meaning.
AUG = dict(hsv_h=0.01, hsv_s=0.4, hsv_v=0.35, fliplr=0.0, flipud=0.0, scale=0.25, translate=0.1, mosaic=0.5, close_mosaic=2, erasing=0.0)


def train(data: str, base: str, epochs: int, batch: int, device: str, name: str, out_weights: Path = DEFAULT_WEIGHTS,
          imgsz: int = 640, fraction: float = 1.0, note: str = "SIMULATED store frames only", epoch_val: bool = False, **overrides) -> dict:
    """epoch_val False: no validation pass after each epoch (on MPS it costs about as much as a fifth of an epoch),
    so the kept weights are the last epoch's; the val split is scored once at the end."""
    from ultralytics import YOLO
    import yaml
    model = YOLO(base)
    t0 = time.time()
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=str(ROOT / "data" / "synth" / "runs"),
                name=name, exist_ok=True, workers=4, patience=epochs, plots=False, fraction=fraction, seed=0, val=epoch_val,
                warmup_epochs=overrides.pop("warmup_epochs", 1), **{**AUG, **overrides})
    wall = time.time() - t0
    best = Path(model.trainer.best if Path(model.trainer.best).exists() else model.trainer.last)
    final = getattr(model.trainer, "metrics", None) or {}
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_weights)
    d = yaml.safe_load(Path(data).read_text())
    n_train = len(list((Path(d["path"]) / d["train"]).glob("*.jpg")))
    rec = {"data_note": note, "base_weights": Path(base).name, "data": str(data), "classes": len(d["names"]),
           "train_images": int(n_train * fraction), "imgsz": imgsz, "epochs": epochs, "batch": batch, "device": device,
           "train_wall_s": round(wall, 1), "train_wall_min": round(wall / 60, 1),
           "validated_every_epoch": epoch_val, "val_split_at_end": {k: round(float(v), 4) for k, v in final.items() if "mAP" in k},
           "augmentation": {**AUG, **overrides},
           "run_dir": str(best.parents[1]), "weights": str(out_weights)}
    out_weights.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv=None) -> None:
    from bree.cli import _weights
    from bree.hw import detect_hardware
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synth/sku/data.yaml")
    ap.add_argument("--base", default=_weights("yolo26n.pt"), help="starting weights (COCO-pretrained YOLO26 nano by default)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--epoch-val", action="store_true", help="validate after every epoch and keep the best epoch (slower)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=detect_hardware().device)
    ap.add_argument("--name", default="sim_sku")
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the train split to use (smoke test)")
    ap.add_argument("--out", default=str(DEFAULT_WEIGHTS))
    a = ap.parse_args(argv)
    print(json.dumps(train(a.data, a.base, a.epochs, a.batch, a.device, a.name, Path(a.out), fraction=a.fraction, epoch_val=a.epoch_val), indent=1))


if __name__ == "__main__":
    main()
