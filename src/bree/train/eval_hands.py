"""Evaluate the hand and held-item detector on held-out SIMULATED frames, next to the first detector.

    python -m bree.train.eval_hands --data data/synth/hands/sku --weights sim_sku sim_sku_hands_v2

Whole held-out frames (seeds never trained on, bree.train.dataset --heldout-from) through the same tiled
inference the pipeline uses, on the whole frame: no person box is used anywhere.
1. Items in a hand: found (IoU >= 0.5) and right SKU, by pixels across a 6.6 cm can at the item, and at 20 px or more.
2. Hands: recall on reaching arms (IoU >= 0.5, IoU >= 0.3, and "a predicted hand box holds the true hand centre",
   which is what the shelf event's hand_px needs), precision.
3. Shelf items, so a regression against the first detector shows.
4. mAP per class on the held-out tiles and speed, for weights that have the dataset's classes.

Everything here is SIMULATED data from one store model. It is not accuracy on real footage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bree.train.backend import SkuDetector, _iou_matrix, sku_weights
from bree.train.dataset import HAND
from bree.train.evaluate import GROUPS, confusion, curve, match, speed

ACCEPT = {"in_hand_sku_right_20px": 0.85, "hand_recall_reach": 0.9}
ITEM_GROUPS = {"in_hand": lambda r: r["kind"] == "hand", "on_counter": lambda r: r["kind"] == "counter",
               **{g: f for g, f in GROUPS.items() if g != "in_hand_or_on_counter"}}


def rate(rows: list[dict], key: str) -> dict:
    n = len(rows)
    k = sum(bool(r[key]) for r in rows)
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None}


def run(det: SkuDetector, frames_dir: Path, limit: int | None = None, second_look: int = 0) -> tuple[list[dict], list[dict], dict]:
    """Item records, hand records and counts over the held-out frames."""
    items, hands, n_frames, hand_pred, hand_fp, item_pred, item_fp = [], [], 0, 0, 0, 0, 0
    for jf in sorted(frames_dir.glob("*.json"))[:limit]:
        m = json.loads(jf.read_text())
        (ib, ic, ik), (hb, hc) = det.detect(cv2.imread(str(jf.with_suffix(".jpg"))), second_look=second_look)
        n_frames += 1
        gi = [a for a in m["annotations"] if a["kind"] != "body_hand"]
        gh = [a for a in m["annotations"] if a["kind"] == "body_hand"]
        hit = match(np.array([a["bbox"] for a in gi], np.float32).reshape(-1, 4), ib, ic)
        item_pred += len(ib)
        item_fp += len(ib) - int((hit >= 0).sum())
        for a, j in zip(gi, hit):
            x0, y0, x1, y1 = a["bbox"]
            items.append({"sku": a["sku"], "kind": a["kind"], "vis": a["vis"], "px_eff": a["px_eff"], "short_side": min(x1 - x0, y1 - y0),
                          "camera_kind": m["camera"]["kind"], "state": a.get("shopper_state"), "detected": bool(j >= 0),
                          "pred": det.names[int(ik[j])] if j >= 0 else None, "conf": float(ic[j]) if j >= 0 else None})
        g = np.array([a["bbox"] for a in gh], np.float32).reshape(-1, 4)
        iou = _iou_matrix(g, hb) if len(g) and len(hb) else np.zeros((len(g), len(hb)))
        hit5, hit3 = match(g, hb, hc, 0.5), match(g, hb, hc, 0.3)
        hand_pred += len(hb)
        hand_fp += int((iou.max(0) < 0.3).sum()) if len(g) and len(hb) else len(hb)
        for i, a in enumerate(gh):
            x0, y0, x1, y1 = a["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hands.append({"reaching": a["reaching"], "holding": a["holding"], "short_side": min(x1 - x0, y1 - y0), "px_eff": a["px_eff"],
                          "camera_kind": m["camera"]["kind"], "chosen_for": m.get("chosen_for"),
                          "iou50": bool(hit5[i] >= 0), "iou30": bool(hit3[i] >= 0),
                          "centre": bool(len(hb) and ((hb[:, 0] <= cx) & (cx <= hb[:, 2]) & (hb[:, 1] <= cy) & (cy <= hb[:, 3])).any())})
    return items, hands, {"frames": n_frames, "item_predictions": item_pred, "item_box_precision": round(1 - item_fp / item_pred, 4) if item_pred else None,
                          "hand_predictions": hand_pred, "hand_box_precision_iou30": round(1 - hand_fp / hand_pred, 4) if hand_pred else None}


def summarise(items: list[dict], hands: list[dict], skus: list[dict]) -> dict:
    held = [r for r in items if r["kind"] == "hand"]
    held20 = [{**r, "ok": r["pred"] == r["sku"]} for r in held if r["px_eff"] >= 20]
    reach = [h for h in hands if h["reaching"]]
    side = lambda lo, hi: [h for h in reach if lo <= h["short_side"] < hi]     # noqa: E731
    return {"in_hand_by_px": curve(held),
            "in_hand_20px_plus": {"found": rate(held20, "detected"), "sku_right": rate(held20, "ok"),
                                  "by_camera_kind": {k: rate([r for r in held20 if r["camera_kind"] == k], "ok") for k in sorted({r["camera_kind"] for r in held20})},
                                  "by_visible_share": {name: rate([r for r in held20 if lo <= r["vis"] < hi], "ok")
                                                       for name, lo, hi in (("25 to 50%", 0.25, 0.5), ("50 to 80%", 0.5, 0.8), ("80%+", 0.8, 1.01))}},
            "in_hand_variant_confusion": confusion(held, skus),
            "groups_20px_plus": {g: rate([{**r, "ok": r["pred"] == r["sku"]} for r in items if f(r) and r["px_eff"] >= 20], "ok") for g, f in ITEM_GROUPS.items()},
            "hand_recall_reaching": {k: rate(reach, k) for k in ("iou50", "iou30", "centre")},
            "hand_recall_reaching_by_short_side": {name: rate(side(lo, hi), "iou50") for name, lo, hi in
                                                   (("5-15 px", 5, 15), ("15-30 px", 15, 30), ("30-60 px", 30, 60), ("60+ px", 60, 1e9))},
            "hand_recall_all_labelled": {k: rate(hands, k) for k in ("iou50", "iou30", "centre")}}


def tile_map(weights: Path, data_dir: Path, device: str) -> dict:
    from ultralytics import YOLO
    r = YOLO(str(weights)).val(data=str(data_dir / "data.yaml"), split="test", imgsz=640, batch=16, device=device, plots=False, verbose=False,
                               project=str(data_dir.parent / "runs"), name="eval_test", exist_ok=True)
    return {"map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
            "per_class": [{"name": r.names[int(c)], "ap50": round(float(r.box.ap50[i]), 4), "ap50_95": round(float(r.box.ap[i]), 4)}
                          for i, c in enumerate(r.box.ap_class_index)]}


def acceptance(weights: dict) -> dict:
    out = {}
    for name, w in weights.items():
        got = {"in_hand_sku_right_20px": w["in_hand_20px_plus"]["sku_right"]["rate"], "hand_recall_reach": w["hand_recall_reaching"]["iou50"]["rate"]}
        for k, t in ACCEPT.items():
            out[f"{name}: {k}"] = f"{'met' if got[k] is not None and got[k] >= t else 'NOT met'}: {got[k]} against {t}"
    return out


def markdown(res: dict) -> str:
    pct = lambda d: "n/a" if not d or d["rate"] is None else f"{100 * d['rate']:.1f}% ({d['k']} of {d['n']})"     # noqa: E731
    names = list(res["weights"])
    L = ["# Hand and held-item detector: held-out SIMULATED frames", "",
         "All numbers are on simulated frames from the browser store simulator (one store model, invented package art, "
         "block-shaped hands). They are not accuracy on real footage.", "",
         f"Held-out split: {res['dataset']['test']['frames']} whole frames from seeds {res['dataset']['test']['seeds'][0]} to "
         f"{res['dataset']['test']['seeds'][-1]} ({len(res['dataset']['test']['seeds'])} scenes), none used for training. "
         f"Tiled inference on the whole frame, confidence {res['conf']}, no person box.", "",
         "| measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    row = lambda title, f: L.append(f"| {title} | " + " | ".join(f(res["weights"][n]) for n in names) + " |")     # noqa: E731
    row("item in a hand, 20 px or more: right SKU (target 85%)", lambda w: pct(w["in_hand_20px_plus"]["sku_right"]))
    row("item in a hand, 20 px or more: found", lambda w: pct(w["in_hand_20px_plus"]["found"]))
    row("hand on a reaching arm found, IoU 0.5 (target 90%)", lambda w: pct(w["hand_recall_reaching"]["iou50"]))
    row("hand on a reaching arm found, IoU 0.3", lambda w: pct(w["hand_recall_reaching"]["iou30"]))
    row("a hand box holds the true hand centre (reaching arm)", lambda w: pct(w["hand_recall_reaching"]["centre"]))
    row("every labelled hand found, IoU 0.5", lambda w: pct(w["hand_recall_all_labelled"]["iou50"]))
    row("hand boxes that are a hand (IoU 0.3)", lambda w: "n/a" if w["counts"]["hand_box_precision_iou30"] is None else f"{100 * w['counts']['hand_box_precision_iou30']:.1f}%")
    for g in ITEM_GROUPS:
        if g != "in_hand":
            row(f"{g.replace('_', ' ')}, 20 px or more: right SKU", lambda w, g=g: pct(w["groups_20px_plus"][g]))
    row("item boxes that are an item", lambda w: f"{100 * w['counts']['item_box_precision']:.1f}%")
    L += ["", "## Item in a hand: right SKU by pixels across a 6.6 cm can", "", "| px | items | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for i, r0 in enumerate(res["weights"][names[0]]["in_hand_by_px"]):
        if r0["n"]:
            L.append(f"| {r0['px']} | {r0['n']} | " + " | ".join(f"{100 * res['weights'][n]['in_hand_by_px'][i]['sku_right_rate']:.1f}%" for n in names) + " |")
    for n in names:
        w = res["weights"][n]
        L += ["", f"## {n}", "", "Item in a hand at 20 px or more, right SKU, by camera kind: " + ", ".join(f"{k} {pct(v)}" for k, v in w["in_hand_20px_plus"]["by_camera_kind"].items()) + ".",
              "By share of the item that shows: " + ", ".join(f"{k} {pct(v)}" for k, v in w["in_hand_20px_plus"]["by_visible_share"].items()) + ".",
              "Reaching hand found (IoU 0.5) by the short side of the hand box: " + ", ".join(f"{k} {pct(v)}" for k, v in w["hand_recall_reaching_by_short_side"].items()) + ".",
              "Near-identical variants held in a hand, given a sibling's SKU: " + ", ".join(f"{fam} {c['confused_within_family']} of {c['found']} found" for fam, c in w["in_hand_variant_confusion"].items() if c["found"]) + "."]
        if w.get("tiles"):
            t = w["tiles"]
            L += ["", f"Held-out tiles: mAP50 {t['map50']:.3f}, mAP50-95 {t['map50_95']:.3f}.", "", "| class | AP50 | AP50-95 |", "|---|---|---|"]
            L += [f"| {c['name']} | {c['ap50']:.3f} | {c['ap50_95']:.3f} |" for c in t["per_class"]]
        if w.get("speed"):
            L += ["", "| device | ms per 640 px tile | ms per whole frame | frames per s |", "|---|---|---|---|"]
            L += [f"| {d} | {s['ms_per_640_tile']} | {s['ms_per_frame']} ({s['frame']}, {s['tiles_per_frame']} tiles) | {s['frames_per_s']} |" for d, s in w["speed"].items()]
    L += ["", "## Acceptance", ""] + [f"- {k}: {v}" for k, v in res["acceptance"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    from bree.hw import detect_hardware
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synth/hands/sku")
    ap.add_argument("--weights", nargs="+", default=["sim_sku", "sim_sku_hands_v2"], help="version names or files; the last one is judged")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N held-out frames (smoke test)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--second-look", type=int, default=0, help="also score the last weights with a tile centred on each of the N best hands")
    ap.add_argument("--report-only", action="store_true", help="rewrite the .md from the saved hand_detector.json (no inference)")
    ap.add_argument("--quick", action="store_true", help="skip the tile mAP and the speed test")
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        res = json.loads((out / "hand_detector.json").read_text())
        res["acceptance"] = acceptance(res["weights"])
        (out / "hand_detector.json").write_text(json.dumps(res, indent=1))
        (out / "hand_detector.md").write_text(markdown(res))
        print(markdown(res))
        return
    device = a.device or detect_hardware().device
    meta = json.loads((data / "meta.json").read_text())
    res = {"data": "SIMULATED (browser store simulator copy); not real footage",
           "dataset": {sp: {k: meta["splits"][sp][k] for k in ("seeds", "frames", "tiles", "boxes", "boxes_by_kind")} for sp in meta["splits"]}, "weights": {}, "conf": a.conf}
    for name in a.weights:
        w = sku_weights(name)
        det = SkuDetector(w, device=device, conf=a.conf)
        items, hands, counts = run(det, data / "test_frames", a.limit)
        r = {"file": str(w), "training": json.loads(w.with_suffix(".json").read_text()) if w.with_suffix(".json").exists() else {},
             "counts": counts, **summarise(items, hands, meta["skus"])}
        if not a.quick:
            if list(det.names.values()) == meta["classes"]:
                r["tiles"] = tile_map(w, data, device)
            r["speed"] = speed(w)
        res["weights"][Path(name).stem] = r
    if a.second_look:
        items, hands, counts = run(det, data / "test_frames", a.limit, a.second_look)
        res["weights"][f"{Path(a.weights[-1]).stem} + second look"] = {"file": str(w), "counts": counts, **summarise(items, hands, meta["skus"])}
    res["acceptance"] = acceptance(res["weights"])
    (out / "hand_detector.json").write_text(json.dumps(res, indent=1))
    (out / "hand_detector.md").write_text(markdown(res))
    print(markdown(res))
    print(f"wrote {out}/hand_detector.json and hand_detector.md")


if __name__ == "__main__":
    main()
