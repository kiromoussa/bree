"""Evaluate the sim-trained SKU detector on the held-out SIMULATED test scenes.

    python -m bree.train.evaluate --data data/synth/sku --out results

1. mAP50 and mAP50-95 per class on the test tiles (Ultralytics val).
2. On whole test frames, through the same tiled inference the pipeline backend uses: for every
   ground-truth item, was it found (IoU >= 0.5) and was the SKU right, bucketed by pixels across a
   6.6 cm can at the item (the simulator's effective-pixel metric, without its lens edge term, which
   the render does not have). This is the curve that checks the simulator's 20 / 25 / 30 / 40 px thresholds.
3. Confusion between near-identical variants (SKUs of the same shape and size).
4. Speed per 640 px tile and per whole frame.

Everything here is SIMULATED data from one store model: it bounds what the cameras can resolve, it is
not accuracy on real footage.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from bree.train.backend import SkuDetector, _iou_matrix, sku_weights

EDGES = [0, 10, 15, 20, 25, 30, 40, 60, 100, float("inf")]
SIM_THRESHOLDS = (20, 25, 30, 40)
GROUPS = {   # name -> filter on a ground-truth record
    "shelf_front_clear": lambda r: r["kind"] == "shelf" and r["vis"] >= 0.8,
    "shelf_front_partly_hidden": lambda r: r["kind"] == "shelf" and r["vis"] < 0.8,
    "backstock": lambda r: r["kind"] == "backstock",
    "in_hand_or_on_counter": lambda r: r["kind"] in ("hand", "counter"),
}
GROUP_TITLE = {"shelf_front_clear": "Shelf, front item, at least 80% visible", "shelf_front_partly_hidden": "Shelf, front item, 25 to 80% visible",
               "backstock": "Shelf, item behind the front one", "in_hand_or_on_counter": "In a hand or on the counter"}


def bucket_label(i: int) -> str:
    return f"{EDGES[i]:g}+" if EDGES[i + 1] == float("inf") else f"{EDGES[i]:g}-{EDGES[i + 1]:g}"


def match(gt: np.ndarray, pred: np.ndarray, confs: np.ndarray, iou: float = 0.5) -> np.ndarray:
    """Prediction index per ground-truth box (-1 = missed). Best-first, one prediction per item, any class."""
    out = np.full(len(gt), -1, int)
    if not len(gt) or not len(pred):
        return out
    m = _iou_matrix(gt, pred)
    for j in np.argsort(-confs):
        col = np.where(out == -1, m[:, j], -1.0)
        i = int(col.argmax())
        if col[i] >= iou:
            out[i] = j
    return out


def frame_records(det: SkuDetector, frames_dir: Path, limit: int | None = None) -> tuple[list[dict], dict]:
    recs, n_pred, n_fp, frames = [], 0, 0, 0
    for jf in sorted(frames_dir.glob("*.json"))[:limit]:
        m = json.loads(jf.read_text())
        img = cv2.imread(str(jf.with_suffix(".jpg")))
        boxes, confs, cls = det(img)
        gt = np.array([a["bbox"] for a in m["annotations"]], np.float32).reshape(-1, 4)
        hit = match(gt, boxes, confs)
        frames += 1
        n_pred += len(boxes)
        n_fp += len(boxes) - int((hit >= 0).sum())
        for a, j in zip(m["annotations"], hit):
            x0, y0, x1, y1 = a["bbox"]
            recs.append({"sku": a["sku"], "kind": a["kind"], "vis": a["vis"], "px_eff": a["px_eff"], "short_side": min(x1 - x0, y1 - y0),
                         "camera_kind": m["camera"]["kind"], "detected": bool(j >= 0),
                         "pred": det.names[int(cls[j])] if j >= 0 else None, "conf": float(confs[j]) if j >= 0 else None})
    return recs, {"frames": frames, "predictions": n_pred, "unmatched_predictions": n_fp,
                  "box_precision": round(1 - n_fp / n_pred, 4) if n_pred else None}


def curve(recs: list[dict], key: str = "px_eff") -> list[dict]:
    rows = []
    for i in range(len(EDGES) - 1):
        b = [r for r in recs if EDGES[i] <= r[key] < EDGES[i + 1]]
        det = sum(r["detected"] for r in b)
        ok = sum(r["pred"] == r["sku"] for r in b)
        rows.append({"px": bucket_label(i), "lo": EDGES[i], "n": len(b), "found": det, "sku_right": ok,
                     "found_rate": round(det / len(b), 4) if b else None, "sku_right_rate": round(ok / len(b), 4) if b else None,
                     "sku_right_of_found": round(ok / det, 4) if det else None})
    return rows


def min_px(rows: list[dict], target: float, min_n: int = 30) -> float | None:
    """Lowest bucket edge from which every bucket (with enough items) has the SKU right at `target` or better."""
    good = None
    for r in reversed(rows):
        if r["n"] < min_n:
            continue
        if r["sku_right_rate"] >= target:
            good = r["lo"]
        else:
            break
    return good


def confusion(recs: list[dict], skus: list[dict]) -> dict:
    """Per family of same-shape, same-size SKUs: counts[true][predicted] over found items ("other" = outside the family)."""
    out = {}
    for fam in sorted({tuple(s["family"]) for s in skus if len(s["family"]) > 1}):
        cols = list(fam) + ["other", "missed"]
        mat = {t: Counter() for t in fam}
        for r in recs:
            if r["sku"] in mat:
                mat[r["sku"]]["missed" if not r["detected"] else r["pred"] if r["pred"] in fam else "other"] += 1
        found = sum(v for t in fam for k, v in mat[t].items() if k != "missed")
        right = sum(mat[t][t] for t in fam)
        swapped = sum(v for t in fam for k, v in mat[t].items() if k in fam and k != t)
        out["/".join(fam)] = {"columns": cols, "rows": {t: [mat[t][c] for c in cols] for t in fam}, "found": found, "sku_right": right,
                              "confused_within_family": swapped, "confused_within_family_rate": round(swapped / found, 4) if found else None}
    return out


def speed(weights: Path, frame_shape=(2688, 1520), n_tile: int = 40, n_frame: int = 8) -> dict:
    import torch
    out = {}
    rng = np.random.default_rng(0)
    tile = rng.integers(0, 255, (640, 640, 3), np.uint8)
    frame = rng.integers(0, 255, (*frame_shape, 3), np.uint8)
    for device in (["mps"] if torch.backends.mps.is_available() else []) + (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]:
        det = SkuDetector(weights, device=device)
        for _ in range(3):
            det.model.predict(tile, imgsz=640, device=device, verbose=False)
        t0 = time.perf_counter()
        for _ in range(n_tile):
            det.model.predict(tile, imgsz=640, device=device, verbose=False)
        t_tile = (time.perf_counter() - t0) / n_tile
        det(frame)
        t0 = time.perf_counter()
        for _ in range(n_frame):
            det(frame)
        t_frame = (time.perf_counter() - t0) / n_frame
        out[device] = {"ms_per_640_tile": round(1000 * t_tile, 1), "tiles_per_s": round(1 / t_tile, 1),
                       "ms_per_frame": round(1000 * t_frame, 1), "frame": f"{frame_shape[1]}x{frame_shape[0]}",
                       "tiles_per_frame": len(det.tiles(*frame_shape)), "frames_per_s": round(1 / t_frame, 2)}
    return out


def plot(curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#e4e3dc"
    fig, axes = plt.subplots(1, len(curves), figsize=(4.3 * len(curves), 3.9), sharey=True, facecolor="#fcfcfb")
    for ax, (g, rows) in zip(np.atleast_1d(axes), curves.items()):
        rows = [r for r in rows if r["n"] >= 30]
        x = range(len(rows))
        ax.set_facecolor("#fcfcfb")
        for t in SIM_THRESHOLDS:     # the simulator's thresholds sit on bucket edges
            pos = next((i for i, r in enumerate(rows) if r["lo"] == t), None)
            if pos is not None:
                ax.axvline(pos - 0.5, color=grid, lw=1, zorder=0)
                ax.text(pos - 0.5, 1.03, f"{t}", color=muted, fontsize=8, ha="center")
        ax.plot(x, [r["found_rate"] for r in rows], color=blue, lw=2, marker="o", ms=5, label="item found")
        ax.plot(x, [r["sku_right_rate"] for r in rows], color=orange, lw=2, marker="o", ms=5, label="found with the right SKU")
        ax.set_xticks(list(x), [f"{r['px']}\nn={r['n']}" for r in rows], fontsize=7.5, color=muted)
        ax.set_ylim(0, 1.09)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        ax.grid(axis="y", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, labelcolor=muted)
        ax.set_title(GROUP_TITLE[g], fontsize=10, color=ink, loc="left", pad=16)
        ax.set_xlabel("pixels across a 6.6 cm can at the item", fontsize=8.5, color=muted)
    np.atleast_1d(axes)[0].set_ylabel("share of ground-truth items", fontsize=8.5, color=muted)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=ink)
    fig.suptitle("SIMULATED test scenes: SKU detector accuracy by pixels on the item (grey lines: simulator thresholds 20, 25, 30, 40 px)",
                 fontsize=10.5, color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown(res: dict) -> str:
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"     # noqa: E731
    t, v = res["training"], res["val"]
    L = ["# Sim-trained SKU detector: results on held-out SIMULATED scenes", "",
         "All numbers below are on simulated frames from the browser store simulator (one store model, invented package art). "
         "They are not accuracy on real footage.", "",
         f"Trained: {t.get('base_weights')} fine-tuned for {t.get('epochs')} epochs on {t.get('train_images')} tiles of {t.get('imgsz')} px, "
         f"batch {t.get('batch')}, device {t.get('device')}, {t.get('train_wall_min')} minutes wall time.", "",
         f"Test split: {res['dataset']['test']['frames']} frames from {len(res['dataset']['test']['seeds'])} scene seeds never used in training "
         f"({res['dataset']['train']['frames']} train frames from {len(res['dataset']['train']['seeds'])} seeds).", "",
         "## mAP on the test tiles", "", f"All classes: mAP50 {v['map50']:.3f}, mAP50-95 {v['map50_95']:.3f} "
         f"({v['images']} tiles, {v['instances']} boxes).", "", "| SKU | boxes | mAP50 | mAP50-95 |", "|---|---|---|---|"]
    L += [f"| {c['name']} | {c['instances']} | {c['ap50']:.3f} | {c['ap50_95']:.3f} |" for c in v["per_class"]]
    L += ["", "## Accuracy by pixels across the item (whole frames, tiled inference, confidence 0.25, IoU 0.5)", "",
          f"{res['frames']['frames']} test frames, {res['frames']['items']} labelled items, box precision {pct(res['frames']['box_precision'])}. "
          "Pixels = pixels across a 6.6 cm can at the item, the simulator's effective-pixel metric without its lens edge term.", ""]
    for g, rows in res["curves"].items():
        L += [f"### {GROUP_TITLE[g]}", "", "| px across a can | items | found | found with the right SKU | right SKU when found |", "|---|---|---|---|---|"]
        L += [f"| {r['px']} | {r['n']} | {pct(r['found_rate'])} | {pct(r['sku_right_rate'])} | {pct(r['sku_right_of_found'])} |" for r in rows if r["n"]]
        L.append("")
    L += ["## Minimum pixels", "", "Lowest bucket edge from which every higher bucket (30+ items) has the right SKU at the target rate:", "",
          "| items | 80% | 90% | 95% |", "|---|---|---|---|"]
    L += [f"| {GROUP_TITLE[g]} | {m['0.8']} | {m['0.9']} | {m['0.95']} |" for g, m in res["min_px"].items()]
    L += ["", res["recommendation"], "", "## Near-identical variants (same shape and size, only the label art differs)", ""]
    for fam, c in res["confusion"].items():
        L += [f"**{fam}**: {c['found']} found, {c['confused_within_family']} given a sibling's SKU ({pct(c['confused_within_family_rate'])}).", "",
              "| true \\ predicted | " + " | ".join(c["columns"]) + " |", "|---|" + "---|" * len(c["columns"])]
        L += [f"| {t} | " + " | ".join(str(x) for x in row) + " |" for t, row in c["rows"].items()]
        L.append("")
    L += ["## Speed", "", "| device | ms per 640 px tile | tiles per s | ms per whole frame | frames per s |", "|---|---|---|---|---|"]
    L += [f"| {d} | {s['ms_per_640_tile']} | {s['tiles_per_s']} | {s['ms_per_frame']} ({s['frame']}, {s['tiles_per_frame']} tiles) | {s['frames_per_s']} |"
          for d, s in res["speed"].items()]
    return "\n".join(L) + "\n"


def recommend(curves: dict, mins: dict, by_side: list[dict], min_n: int = 30) -> str:
    """The pixel threshold this evaluation supports, in words. Rule: the lowest bucket edge from which every
    bucket with enough items has the right SKU 90% of the time, for clearly visible shelf items."""
    rows = [r for r in curves["shelf_front_clear"] if r["n"] >= min_n]
    at = ", ".join(f"{r['px']} px {100 * r['sku_right_rate']:.1f}%" for r in rows if r["lo"] in (10, 15) + SIM_THRESHOLDS)
    side = ", ".join(f"{r['px']} px {100 * r['sku_right_rate']:.1f}%" for r in by_side if r["n"] >= min_n and r["lo"] < 30)
    rec90 = mins["shelf_front_clear"]["0.9"]
    if rec90 is None:
        head = "No pixel bucket reaches 90% right SKU for clearly visible shelf items (simulated)."
    elif rec90 <= rows[0]["lo"]:
        head = ("Minimum pixels (simulated): no lower limit shows up for clearly visible shelf items. Every bucket with "
                f"{min_n} or more items has the right SKU 90% of the time or better, so the simulator's default of "
                f"{SIM_THRESHOLDS[0]} px across a can holds on this data and nothing here supports raising it.")
    else:
        head = (f"Recommended minimum (simulated): {rec90:g} px across a can for a clearly visible shelf item to get the right "
                "SKU 90% of the time with this detector.")
    return (f"{head} Right-SKU rate by pixels across a can, clearly visible shelf items: {at}. By the short side of the item's own "
            f"box in pixels, all items: {side}.")


def evaluate(data_dir, out_dir, weights=None, device=None, limit: int | None = None, skip_val: bool = False) -> dict:
    from ultralytics import YOLO
    from bree.hw import detect_hardware
    data_dir, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = sku_weights(weights)
    device = device or detect_hardware().device
    meta = json.loads((data_dir / "meta.json").read_text())
    train_rec = json.loads(w.with_suffix(".json").read_text()) if w.with_suffix(".json").exists() else {}
    val = {}
    if not skip_val:
        r = YOLO(str(w)).val(data=str(data_dir / "data.yaml"), split="test", imgsz=640, batch=32, device=device, plots=False, verbose=False,
                             project=str(data_dir.parent / "runs"), name="eval_test", exist_ok=True)
        counts = meta["splits"]["test"]["boxes_by_class"]
        val = {"map50": float(r.box.map50), "map50_95": float(r.box.map), "images": meta["splits"]["test"]["tiles"], "instances": meta["splits"]["test"]["boxes"],
               "per_class": [{"name": r.names[int(c)], "instances": counts.get(r.names[int(c)], 0), "ap50": float(r.box.ap50[i]), "ap50_95": float(r.box.ap[i])}
                             for i, c in enumerate(r.box.ap_class_index)]}
    recs, fstats = frame_records(SkuDetector(w, device=device), data_dir / "test_frames", limit)
    curves = {g: curve([r for r in recs if f(r)]) for g, f in GROUPS.items()}
    mins = {g: {str(t): min_px(rows, t) for t in (0.8, 0.9, 0.95)} for g, rows in curves.items()}
    recommendation = recommend(curves, mins, curve(recs, "short_side"))
    res = {"data": "SIMULATED (browser store simulator copy); not real footage", "weights": str(w), "training": train_rec,
           "dataset": {sp: {k: meta["splits"][sp][k] for k in ("seeds", "frames", "tiles", "boxes", "boxes_by_kind")} for sp in meta["splits"]},
           "val": val, "frames": {**fstats, "items": len(recs), "conf": 0.25, "iou": 0.5},
           "curves": curves, "curve_all_items": curve(recs), "curve_by_box_short_side_all_items": curve(recs, "short_side"),
           "min_px": mins, "recommendation": recommendation, "confusion": confusion(recs, meta["skus"]), "speed": speed(w)}
    (out / "sku_detector.json").write_text(json.dumps(res, indent=1))
    plot(curves, out / "px_vs_accuracy.png")
    if val:
        (out / "sku_detector.md").write_text(markdown(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synth/sku")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N test frames (smoke test)")
    ap.add_argument("--report-only", action="store_true", help="rewrite the recommendation and the .md from the saved sku_detector.json (no inference)")
    a = ap.parse_args(argv)
    if a.report_only:
        jf = Path(a.out) / "sku_detector.json"
        res = json.loads(jf.read_text())
        res["recommendation"] = recommend(res["curves"], res["min_px"], res["curve_by_box_short_side_all_items"])
        jf.write_text(json.dumps(res, indent=1))
        (Path(a.out) / "sku_detector.md").write_text(markdown(res))
    else:
        res = evaluate(a.data, a.out, a.weights, a.device, a.limit)
    print(markdown(res))
    print(f"wrote {a.out}/sku_detector.json, sku_detector.md, px_vs_accuracy.png")


if __name__ == "__main__":
    main()
