"""Simulator frames (scripts/train/render_synth.mjs) -> a YOLO / COCO detection dataset. SIMULATED DATA.

    python -m bree.train.dataset --frames data/synth/frames --out data/synth/sku

Split is by scene seed (hash of the seed), so no scene, shopper or lighting draw is shared between
train, val and test. It is still ONE store model and ONE set of package art: the test split measures
held-out scenes, not a held-out store.

Each 4MP frame is cut into 640 px tiles at native resolution (a can that is 25 px wide stays 25 px),
after sensor effects the renderer does not have: blur, noise, white balance, gain, JPEG compression.
Test frames are also kept whole (same effects) for the pixels-vs-accuracy evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TILE, OVERLAP = 640, 128
MIN_VIS_PX, MIN_SIDE, MIN_VIS = 30, 5, 0.25      # a box is a label only if this much of the item shows
SPLITS = ("train", "val", "test")


def split_of(seed: int) -> str:
    """70 / 15 / 15 by a hash of the seed (stable when more seeds are rendered later)."""
    h = zlib.crc32(str(seed).encode()) % 20
    return "train" if h < 14 else "val" if h < 17 else "test"


def tile_origins(size: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    """Tile starts covering [0, size) with at least `overlap` px shared, the last tile flush with the edge."""
    if size <= tile:
        return [0]
    n = -(-(size - overlap) // (tile - overlap))
    return [round(i * (size - tile) / (n - 1)) for i in range(n)]


def visible_share(ann: dict, ref: dict[str, float]) -> float:
    """Share of the item that shows: visible pixels over the box it would fill unhidden, divided by what an
    unhidden item of that shape reaches (`ref`, the 95th percentile per shape; a bottle never fills its box)."""
    if not ann["full"]:
        return 0.0
    x0, y0, x1, y1 = ann["full"]
    return min(1.0, ann["vis_px"] / max((x1 - x0) * (y1 - y0), 1.0) / ref[ann["geo"]])


def keep(ann: dict, vis: float) -> bool:
    x0, y0, x1, y1 = ann["bbox"]
    return ann["vis_px"] >= MIN_VIS_PX and min(x1 - x0, y1 - y0) >= MIN_SIDE and vis >= MIN_VIS


def sensor(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Camera effects the WebGL render lacks. Returns the image and the draw (for the record)."""
    p = {"blur_sigma": float(rng.choice([0, 0, rng.uniform(0.3, 1.6)])), "motion_px": int(rng.choice([0, 0, 0, rng.integers(3, 8)])),
         "noise_sigma": float(rng.uniform(0, 7)), "wb": [float(v) for v in rng.uniform(0.93, 1.07, 3)], "gain": float(rng.uniform(0.85, 1.15)),
         "jpeg_q": int(rng.integers(55, 96))}
    out = img.astype(np.float32)
    if p["blur_sigma"]:
        out = cv2.GaussianBlur(out, (0, 0), p["blur_sigma"])
    if p["motion_px"]:
        k = np.zeros((p["motion_px"], p["motion_px"]), np.float32)
        k[p["motion_px"] // 2, :] = 1 / p["motion_px"]
        out = cv2.filter2D(out, -1, k)
    out = out * (np.array(p["wb"], np.float32) * p["gain"])
    if p["noise_sigma"]:
        out += rng.normal(0, p["noise_sigma"], out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_q"]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), p


def tile_labels(anns: list[dict], x0: int, y0: int, tile: int = TILE) -> list[tuple[dict, list[float]]]:
    """Boxes of a tile (tile pixels). A box cut by the tile edge stays if at least half of it is inside."""
    out = []
    for a in anns:
        bx0, by0, bx1, by1 = a["bbox"]
        cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x0 + tile), min(by1, y0 + tile)
        if cx1 - cx0 < MIN_SIDE or cy1 - cy0 < MIN_SIDE:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (bx1 - bx0) * (by1 - by0):
            continue
        out.append((a, [cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0]))
    return out


def build(frames_dir, out_dir, tiles_per_frame: int = 4, background_share: float = 0.05, limit: int | None = None) -> dict:
    frames_dir, out = Path(frames_dir), Path(out_dir)
    skus = json.loads((frames_dir / "skus.json").read_text())
    names = [s["id"] for s in skus]
    cls = {n: i for i, n in enumerate(names)}
    metas = sorted(p for p in frames_dir.glob("s*/*.json") if (p.parent / "done").exists())[:limit]
    if not metas:
        raise SystemExit(f"no rendered frames under {frames_dir}")
    loaded = [(p, json.loads(p.read_text())) for p in metas]
    # unhidden reference per shape, from train scenes only
    ratios: dict[str, list[float]] = {}
    for _, m in loaded:
        if split_of(m["seed"]) == "train":
            for a in m["annotations"]:
                if a["full"]:
                    x0, y0, x1, y1 = a["full"]
                    ratios.setdefault(a["geo"], []).append(a["vis_px"] / max((x1 - x0) * (y1 - y0), 1.0))
    ref = {g: float(np.percentile(v, 95)) for g, v in ratios.items()}
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (out / "test_frames").mkdir(exist_ok=True)
    coco = {sp: {"info": {"description": "BREE store simulator, SIMULATED", "split": sp}, "images": [], "annotations": [],
                 "categories": [{"id": i, "name": n} for i, n in enumerate(names)]} for sp in SPLITS}
    stats = {sp: {"seeds": set(), "frames": 0, "tiles": 0, "boxes": 0, "kinds": Counter(), "classes": Counter()} for sp in SPLITS}
    for path, m in loaded:
        sp = split_of(m["seed"])
        stem = f"s{m['seed']}_{path.stem}"
        rng = np.random.default_rng(zlib.crc32(stem.encode()))
        img, effects = sensor(cv2.imread(str(path.with_suffix(".jpg"))), rng)
        anns = []
        for a in m["annotations"]:
            v = visible_share(a, ref)
            if keep(a, v):
                anns.append({**a, "vis": round(v, 3)})
        st = stats[sp]
        st["seeds"].add(m["seed"])
        st["frames"] += 1
        if sp == "test":
            cv2.imwrite(str(out / "test_frames" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "test_frames" / f"{stem}.json").write_text(json.dumps(
                {"seed": m["seed"], "camera": m["camera"], "width": m["width"], "height": m["height"], "effects": effects,
                 "params": m["params"], "annotations": anns}))
        tiles = [(x, y, tile_labels(anns, x, y)) for y in tile_origins(m["height"]) for x in tile_origins(m["width"])]
        hard = [t for t in tiles if any(a["kind"] in ("hand", "counter") for a, _ in t[2])]     # item in hand: always kept
        rest = [t for t in tiles if t not in hard and t[2]]
        pyr = random.Random(zlib.crc32(stem.encode()))
        pyr.shuffle(rest)
        chosen = hard + rest[:max(0, tiles_per_frame - len(hard))]
        empty = [t for t in tiles if not t[2]]
        if empty and pyr.random() < background_share * tiles_per_frame:
            chosen.append(pyr.choice(empty))
        for x, y, labs in chosen:
            name = f"{stem}_{x}_{y}"
            crop = img[y:y + TILE, x:x + TILE]
            h, w = crop.shape[:2]
            cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            img_id = len(coco[sp]["images"])
            coco[sp]["images"].append({"id": img_id, "file_name": f"{name}.jpg", "width": w, "height": h})
            for a, (bx0, by0, bx1, by1) in labs:
                lines.append(f"{cls[a['sku']]} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} {(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}")
                coco[sp]["annotations"].append({"id": len(coco[sp]["annotations"]), "image_id": img_id, "category_id": cls[a["sku"]],
                                                "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": (bx1 - bx0) * (by1 - by0), "iscrowd": 0,
                                                "attributes": {"kind": a["kind"], "px_eff": a["px_eff"], "vis": a["vis"]}})
                st["kinds"][a["kind"]] += 1
                st["classes"][a["sku"]] += 1
            (out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
            st["tiles"] += 1
            st["boxes"] += len(labs)
    for sp in SPLITS:
        (out / f"coco_{sp}.json").write_text(json.dumps(coco[sp]))
    (out / "data.yaml").write_text(f"# BREE store simulator SKU detection, SIMULATED (python -m bree.train.dataset)\npath: {out.resolve()}\n"
                                   "train: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    seeds = {sp: sorted(stats[sp]["seeds"]) for sp in SPLITS}
    assert not (set(seeds["train"]) & set(seeds["val"]) or set(seeds["train"]) & set(seeds["test"]) or set(seeds["val"]) & set(seeds["test"])), "seed leak between splits"
    meta = {"source": str(frames_dir), "data": "SIMULATED (browser store simulator copy)", "tile": TILE, "overlap": OVERLAP,
            "label_rule": {"min_visible_px": MIN_VIS_PX, "min_side_px": MIN_SIDE, "min_visible_share": MIN_VIS},
            "visible_share_reference": ref, "skus": skus,
            "splits": {sp: {"seeds": seeds[sp], "frames": stats[sp]["frames"], "tiles": stats[sp]["tiles"], "boxes": stats[sp]["boxes"],
                            "boxes_by_kind": dict(stats[sp]["kinds"]), "boxes_by_class": dict(sorted(stats[sp]["classes"].items()))} for sp in SPLITS}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def manifest_to_yolo(manifest_path, out, val_share: float = 0.2, pseudo: bool = True) -> Path:
    """Review-store label export (bree.review.labels, format "bree-review-labels" version 1) -> a whole-frame
    YOLO dataset that scripts/train/finetune_real.py reads. Returns the data.yaml.

    Detector examples only. Split by `alert_id` (a stable hash, so an alert never changes side and its frames
    stay together), as the manifest asks. Examples with `class_known: false` (a typed item name the pipeline
    has never used: a typo or a new product) are left out; map or fix them in the review store first.
    `pseudo=False` keeps only line 1 of each label file, the box a reviewer confirmed."""
    import hashlib
    import shutil

    import yaml

    from bree.review.labels import load_manifest
    mp = Path(manifest_path)
    m = load_manifest(mp)
    ex = [e for e in m["examples"] if e["task"] == "detector" and e.get("class_known", True)]
    if not ex:
        raise ValueError(f"{mp}: no detector examples with a known class (the pipeline must save evidence frames first)")
    alerts = sorted({e["alert_id"] for e in ex}, key=lambda a: hashlib.sha1(a.encode()).hexdigest())
    n_val = min(max(1, round(val_share * len(alerts))), len(alerts) - 1) if len(alerts) > 1 else 0
    val = set(alerts[:n_val])
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    for e in ex:
        split = "val" if e["alert_id"] in val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        src = mp.parent / e["image"]
        shutil.copy(src, out / "images" / split / f"{e['key']}{src.suffix}")
        lines = (mp.parent / e["labels"]).read_text().splitlines()
        (out / "labels" / split / f"{e['key']}.txt").write_text("\n".join(lines if pseudo else lines[:1]))
    y = out / "data.yaml"
    # ponytail: one alert only = no val split, the same frames are used for both; fine for a smoke run, not for a number.
    y.write_text(yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val" if val else "images/train",
                                 "names": dict(enumerate(m["classes"]))}, sort_keys=False))
    return y


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="data/synth/frames")
    ap.add_argument("--out", default="data/synth/sku")
    ap.add_argument("--tiles-per-frame", type=int, default=4, help="labelled tiles kept per frame (tiles with an item in a hand are always kept)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (smoke test)")
    a = ap.parse_args(argv)
    meta = build(a.frames, a.out, a.tiles_per_frame, limit=a.limit)
    for sp, s in meta["splits"].items():
        print(f"{sp}: {len(s['seeds'])} seeds, {s['frames']} frames, {s['tiles']} tiles, {s['boxes']} boxes, by kind {s['boxes_by_kind']}")
    print(f"wrote {a.out}/data.yaml, coco_*.json, meta.json, test_frames/")


if __name__ == "__main__":
    main()
