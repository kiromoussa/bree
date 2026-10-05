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


HAND = "hand"                                    # extra detector class when the frames carry hand boxes (render_hands.mjs)
HARD_KINDS = ("hand", "counter", "body_hand")    # item in a hand, item on the counter, a hand: tiles with these come first


def split_from(seed: int, heldout_from: int | None = None) -> str:
    """Default: split_of. With `heldout_from`: seeds from there up are the test split (rendered apart, never
    trained on), an eighth of the rest is val."""
    if heldout_from is None:
        return split_of(seed)
    return "test" if seed >= heldout_from else "val" if zlib.crc32(str(seed).encode()) % 8 == 0 else "train"


def frame_anns(m: dict, ref: dict[str, float]) -> list[dict]:
    """Labels of one frame: items by the visible-share rule, hands (class HAND, kind body_hand) by visible pixels."""
    anns = []
    for a in m["annotations"]:
        v = visible_share(a, ref)
        if keep(a, v):
            anns.append({**a, "vis": round(v, 3)})
    for h in m.get("hands", []):
        x0, y0, x1, y1 = h["bbox"]
        if h["vis_px"] >= MIN_VIS_PX and min(x1 - x0, y1 - y0) >= MIN_SIDE:
            anns.append({**h, "sku": HAND, "kind": "body_hand", "vis": 1.0, "px_eff": h.get("px_eff", 0.0)})
    return anns


def _frame(job) -> dict:
    """One frame -> its tiles on disk. Returns what the index needs (runs in a worker process)."""
    path, out, sp, cls, ref, tiles_per_frame, background_share, max_hard, focus = job
    path, out = Path(path), Path(out)
    m = json.loads(path.read_text())
    stem = f"s{m['seed']}_{path.stem}"
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    img, effects = sensor(cv2.imread(str(path.with_suffix(".jpg"))), rng)
    anns = frame_anns(m, ref)
    if sp == "test":
        cv2.imwrite(str(out / "test_frames" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "test_frames" / f"{stem}.json").write_text(json.dumps(
            {"seed": m["seed"], "camera": m["camera"], "width": m["width"], "height": m["height"], "effects": effects,
             "chosen_for": m.get("chosen_for"), "params": m["params"], "annotations": anns}))
    tiles = [(x, y, tile_labels(anns, x, y)) for y in tile_origins(m["height"]) for x in tile_origins(m["width"])]
    hard = [t for t in tiles if any(a["kind"] in HARD_KINDS for a, _ in t[2])]     # item in hand: always kept
    rest = [t for t in tiles if t not in hard and t[2]]
    pyr = random.Random(zlib.crc32(stem.encode()))
    pyr.shuffle(rest)
    if max_hard is not None and len(hard) > max_hard:
        hard = pyr.sample(hard, max_hard)
    chosen = hard + rest[:max(0, tiles_per_frame - len(hard))]
    empty = [t for t in tiles if not t[2]]
    if empty and pyr.random() < background_share * tiles_per_frame:
        chosen.append(pyr.choice(empty))
    # crops at any offset around a held item or a raised hand (the pipeline also crops around a point, not only on the grid)
    targets = [a for a in anns if a["kind"] in ("hand", "counter") or a.get("reaching")]
    for a in pyr.sample(targets, min(focus, len(targets))):
        cx, cy = (a["bbox"][0] + a["bbox"][2]) / 2, (a["bbox"][1] + a["bbox"][3]) / 2
        x = int(min(max(cx - pyr.uniform(80, TILE - 80), 0), max(m["width"] - TILE, 0)))
        y = int(min(max(cy - pyr.uniform(80, TILE - 80), 0), max(m["height"] - TILE, 0)))
        chosen.append((x, y, tile_labels(anns, x, y)))
    rec = {"sp": sp, "seed": m["seed"], "tiles": []}
    for x, y, labs in chosen:
        name = f"{stem}_{x}_{y}"
        crop = img[y:y + TILE, x:x + TILE]
        h, w = crop.shape[:2]
        cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / sp / f"{name}.txt").write_text("\n".join(
            f"{cls[a['sku']]} {(b[0] + b[2]) / 2 / w:.6f} {(b[1] + b[3]) / 2 / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}" for a, b in labs))
        rec["tiles"].append((name, w, h, [(a["sku"], b, {"kind": a["kind"], "px_eff": a["px_eff"], "vis": a["vis"],
                                                         **({"held": a["kind"] == "hand"} if a["kind"] != "body_hand" else
                                                            {"reaching": a["reaching"], "holding": a["holding"]})}) for a, b in labs]))
    return rec


def build(frames_dir, out_dir, tiles_per_frame: int = 4, background_share: float = 0.05, limit: int | None = None,
          heldout_from: int | None = None, max_hard: int | None = None, focus: int = 0, workers: int = 1, stride: int = 1) -> dict:
    """frames_dir: one folder or a list of them (render_synth.mjs or render_hands.mjs output)."""
    dirs = [Path(d) for d in (frames_dir if isinstance(frames_dir, (list, tuple)) else [frames_dir])]
    frames_dir, out = dirs[0], Path(out_dir)
    skus = json.loads((frames_dir / "skus.json").read_text())
    metas = sorted(p for d in dirs for p in d.glob("s*/*.json") if (p.parent / "done").exists())[:limit][::stride]
    if not metas:
        raise SystemExit(f"no rendered frames under {frames_dir}")
    # unhidden reference per shape, from train scenes only
    ratios: dict[str, list[float]] = {}
    split, hands = {}, False
    for p in metas:
        m = json.loads(p.read_text())
        split[p] = split_from(m["seed"], heldout_from)
        hands |= "hands" in m
        if split[p] == "train":
            for a in m["annotations"]:
                if a["full"]:
                    x0, y0, x1, y1 = a["full"]
                    ratios.setdefault(a["geo"], []).append(a["vis_px"] / max((x1 - x0) * (y1 - y0), 1.0))
    if not ratios:      # a short --seeds range can leave the train split empty (the split is by seed)
        seeds = sorted({int(p.parent.name[1:]) for p in metas})
        raise SystemExit(f"no train scenes among seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds): the visible-share "
                         "reference comes from train scenes only. Render more seeds; a smoke run that has all three "
                         "splits is `make sku-data SKU_SEEDS=1000:1004 SYNTH=out/synth_smoke`.")
    names = [s["id"] for s in skus] + ([HAND] if hands else [])
    cls = {n: i for i, n in enumerate(names)}
    ref = {g: float(np.percentile(v, 95)) for g, v in ratios.items()}
    sim_copy = frames_dir / "sim_copy.json"      # which simulator version rendered the frames (render_synth.mjs)
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    (out / "test_frames").mkdir(exist_ok=True)
    coco = {sp: {"info": {"description": "BREE store simulator, SIMULATED", "split": sp}, "images": [], "annotations": [],
                 "categories": [{"id": i, "name": n} for i, n in enumerate(names)]} for sp in SPLITS}
    stats = {sp: {"seeds": set(), "frames": 0, "tiles": 0, "boxes": 0, "kinds": Counter(), "classes": Counter()} for sp in SPLITS}
    jobs = [(str(p), str(out), split[p], cls, ref, tiles_per_frame, background_share, max_hard, focus) for p in metas]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            recs = list(ex.map(_frame, jobs, chunksize=8))
    else:
        recs = [_frame(j) for j in jobs]
    for rec in recs:
        sp, st = rec["sp"], stats[rec["sp"]]
        st["seeds"].add(rec["seed"])
        st["frames"] += 1
        for name, w, h, labs in rec["tiles"]:
            img_id = len(coco[sp]["images"])
            coco[sp]["images"].append({"id": img_id, "file_name": f"{name}.jpg", "width": w, "height": h})
            for sku, (bx0, by0, bx1, by1), attrs in labs:
                coco[sp]["annotations"].append({"id": len(coco[sp]["annotations"]), "image_id": img_id, "category_id": cls[sku],
                                                "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": (bx1 - bx0) * (by1 - by0), "iscrowd": 0,
                                                "attributes": attrs})
                st["kinds"][attrs["kind"]] += 1
                st["classes"][sku] += 1
            st["tiles"] += 1
            st["boxes"] += len(labs)
    for sp in SPLITS:
        (out / f"coco_{sp}.json").write_text(json.dumps(coco[sp]))
    (out / "data.yaml").write_text(f"# BREE store simulator SKU detection, SIMULATED (python -m bree.train.dataset)\npath: {out.resolve()}\n"
                                   "train: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    seeds = {sp: sorted(stats[sp]["seeds"]) for sp in SPLITS}
    assert not (set(seeds["train"]) & set(seeds["val"]) or set(seeds["train"]) & set(seeds["test"]) or set(seeds["val"]) & set(seeds["test"])), "seed leak between splits"
    meta = {"source": [str(d) for d in dirs] if len(dirs) > 1 else str(frames_dir), "data": "SIMULATED (browser store simulator copy)", "tile": TILE, "overlap": OVERLAP,
            "simulator": json.loads(sim_copy.read_text()) if sim_copy.exists() else "not recorded (rendered before 2026-10-05)",
            "label_rule": {"min_visible_px": MIN_VIS_PX, "min_side_px": MIN_SIDE, "min_visible_share": MIN_VIS},
            "visible_share_reference": ref, "skus": skus, "classes": names,
            "split_rule": "hash of the seed (split_of)" if heldout_from is None else f"seeds from {heldout_from} up are the held-out test split",
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
    ap.add_argument("--frames", nargs="+", default=["data/synth/frames"], help="one or more rendered frame folders")
    ap.add_argument("--out", default="data/synth/sku")
    ap.add_argument("--tiles-per-frame", type=int, default=4, help="labelled tiles kept per frame (tiles with an item in a hand are always kept)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (smoke test)")
    ap.add_argument("--heldout-from", type=int, default=None, help="seeds from here up are the test split (default: split by a hash of the seed)")
    ap.add_argument("--max-hard", type=int, default=None, help="at most this many grid tiles with a hand or a held item per frame")
    ap.add_argument("--focus", type=int, default=0, help="extra crops per frame at a random offset around a held item or a raised hand")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (captures 0.3 s apart look alike)")
    a = ap.parse_args(argv)
    meta = build(a.frames, a.out, a.tiles_per_frame, limit=a.limit, heldout_from=a.heldout_from, max_hard=a.max_hard, focus=a.focus, workers=a.workers, stride=a.stride)
    for sp, s in meta["splits"].items():
        print(f"{sp}: {len(s['seeds'])} seeds, {s['frames']} frames, {s['tiles']} tiles, {s['boxes']} boxes, by kind {s['boxes_by_kind']}")
    print(f"wrote {a.out}/data.yaml, coco_*.json, meta.json, test_frames/")


if __name__ == "__main__":
    main()
