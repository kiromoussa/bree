"""Label export: reviewer decisions -> training examples, a manifest, dedup, and the retrain hook.

Which decisions become which examples (alerts where two reviewers disagree are skipped):

  detector   one example per stored frame of the predicted item that has an `image` and an
             `item_box`. Class = the predicted item (its category) when the decision is confirmed_theft, the
             reviewer's corrected item when it is wrong_item. not_theft and unclear give none
             (the reviewer did not check the item).
  pick       one keypoint window per alert from frames tagged event "pick". Label = the reviewer's
             "pick seen" answer if given (label_source "reviewer"); else 1 when the decision is
             confirmed_theft (label_source "derived": a confirmed theft means the item was taken).
  conceal    same, frames tagged "conceal". Label only from the reviewer's "concealment seen"
             answer. Never derived: a theft can be a plain walk-out.

Boxes are clipped to the image. A reviewed box with no area left is skipped and counted under
`skipped.detector_bad_box`; an unreviewed box with no area or no category is left out of the file.

Layout of the dataset folder (default `<store>/datasets/review/`; any other folder is recorded in the
store so retention reaches it too):
  manifest.json
  detector/images/<key>.jpg       head-pixelated frame, copied as is
  detector/labels/<key>.txt       YOLO lines "class_id cx cy w h" (normalised). Line 1 is the reviewed
                                  box. The other lines are the detector's own unreviewed boxes
                                  (pseudo-labels), so the rest of the shelf is not taught as background.
  pick/<key>.npz, conceal/<key>.npz    kps float32 (T, 17, 3) = x, y, conf in pixels; t float32 (T,)

manifest.json (format "bree-review-labels", version 1):
  classes        class names; class_id is the index. Append-only across exports, ids never shift.
  store_id       the review store this dataset belongs to. Another store's export into the same
                 folder is refused (it would delete this store's examples).
  retention_days, created_ts, updated_ts
  counts         detector / pick / conceal totals, unknown_classes, new_this_export,
                 removed_this_export, duplicates_skipped, skipped (reason -> count)
  examples[]     key (sha1 of the content, also the dedup key), task, alert_id, camera, zone,
                 decision, label_source, reviewers (count, no names), event_ts,
                 expires_ts (event_ts + retention; the store deletes the example after it), added_ts
      detector:  image, labels, class, class_id, class_known, predicted_class, corrected,
                 bbox (pixels, xyxy), image_size [w, h], pseudo_boxes
                 class_known is false when a reviewer typed an item name the pipeline has never
                 named in any alert of this store (a typo, or a product the detector does not have
                 yet). counts.unknown_classes counts them. Drop or map those before training.
      pick / conceal:  window, label (0 or 1), n_frames

Every export rebuilds the example list from the store, so a changed decision or a purged alert
drops out. `added_ts` is kept for examples that were already there.

There is no train / test split in the manifest. Split by `alert_id` (never by example): one alert
gives a detector frame, a pick window and a conceal window of the same person and moment, and the
same evidence must not land on both sides.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bree.review.store import write_json

FORMAT, VERSION = "bree-review-labels", 1
TASKS = ("detector", "pick", "conceal")


def _yolo(box, w, h) -> str:
    x1, y1, x2, y2 = box
    return f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def _clip(box, w, h) -> list[float] | None:
    """The box clipped to the image, or None when it is malformed or has no area left."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    x1, x2, y1, y2 = max(0.0, x1), min(float(w), x2), max(0.0, y1), min(float(h), y2)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _agreed(revs: list[dict], field: str):
    """The reviewers' shared answer for `field`, or None if nobody answered or they differ."""
    vals = {r[field] for r in revs if r[field] is not None}
    return vals.pop() if len(vals) == 1 else None


def export_labels(store, out_dir: str | Path | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    out = Path(out_dir) if out_dir else store.dir / "datasets" / "review"
    mf = out / "manifest.json"
    old = json.loads(mf.read_text()) if mf.exists() else {"classes": [], "examples": [], "created_ts": now}
    if old.get("store_id", store.store_id) != store.store_id:
        raise ValueError(f"{out} holds the label export of another review store ({old['store_id']}). "
                         "Export to a different folder: exporting here would delete that store's examples.")
    out.mkdir(parents=True, exist_ok=True)
    store.register_dataset(out)              # so retention deletes expired examples here as well
    seen = store.pipeline_items()
    old_by_key = {e["key"]: e for e in old["examples"]}
    classes: list[str] = list(old["classes"])
    examples: dict[str, dict] = {}
    skipped: Counter = Counter()
    dups = 0

    def class_id(name: str) -> int:
        if name not in classes:
            classes.append(name)
        return classes.index(name)

    def add(key: str, a: dict, task: str, source: str, extra: dict) -> bool:
        nonlocal dups
        if key in examples:
            dups += 1
            return False
        examples[key] = {"key": key, "task": task, "alert_id": a["alert_id"], "camera": a["camera"],
                         "zone": a["zone"], "decision": a["decision"], "label_source": source,
                         "reviewers": len(a["reviews"]), "event_ts": a["event_ts"],
                         "expires_ts": a["event_ts"] + store.retention_days * 86400,
                         "added_ts": old_by_key.get(key, {}).get("added_ts", now), **extra}
        return True

    for a in store.alerts():
        if a["decision"] is None or a["purged_ts"]:
            continue
        if a["decision"] == "disputed":
            skipped["reviewers_disagree"] += 1
            continue
        revs, frames = a["reviews"], a["frames"]

        # ---- detector
        # The detector works on categories: use the predicted item's category, not its SKU.
        cat = next((i.get("category") for i in a["items"]
                    if a["predicted_item"] in (i.get("sku"), i.get("category"))), a["predicted_item"])
        cls = {"confirmed_theft": cat, "wrong_item": _agreed(revs, "corrected_item")}.get(a["decision"])
        if a["decision"] == "wrong_item" and cls is None:
            skipped["reviewers_disagree_on_item"] += 1
        if cls:
            for f in frames:
                if f.get("item") != cat or not f.get("item_box"):
                    continue
                if not f.get("image") or not Path(f["image"]).is_file() or not f.get("width") or not f.get("height"):
                    skipped["detector_no_clean_frame"] += 1
                    continue
                box = _clip(f["item_box"], f["width"], f["height"])
                if box is None:
                    skipped["detector_bad_box"] += 1
                    continue
                data = Path(f["image"]).read_bytes()
                key = hashlib.sha1(data + f"{cls}{[round(v) for v in box]}".encode()).hexdigest()[:20]
                pseudo = [(p["cat"], b) for p in f.get("products", [])
                          if p.get("cat") and (b := _clip(p.get("bbox"), f["width"], f["height"])) and b != box]
                img, lab = f"detector/images/{key}.jpg", f"detector/labels/{key}.txt"
                if add(key, a, "detector", "reviewer", {
                        "image": img, "labels": lab, "class": cls, "class_id": class_id(cls), "class_known": cls in seen,
                        "predicted_class": f.get("item"), "corrected": a["decision"] == "wrong_item",
                        "bbox": box, "image_size": [f["width"], f["height"]], "pseudo_boxes": len(pseudo)}):
                    (out / img).parent.mkdir(parents=True, exist_ok=True)
                    (out / lab).parent.mkdir(parents=True, exist_ok=True)
                    (out / img).write_bytes(data)
                    lines = [f"{class_id(cls)} {_yolo(box, f['width'], f['height'])}"]
                    lines += [f"{class_id(c)} {_yolo(b, f['width'], f['height'])}" for c, b in pseudo]
                    (out / lab).write_text("\n".join(lines) + "\n")

        # ---- pick / conceal keypoint windows
        for task, field in (("pick", "pick_seen"), ("conceal", "conceal_seen")):
            win = sorted((f for f in frames if f.get("event") == task and f.get("kpts")), key=lambda f: f["t"])
            if not win:
                continue
            answered = [r[field] for r in revs if r[field] is not None]
            label, source = _agreed(revs, field), "reviewer"
            if label is None and answered:
                skipped[f"reviewers_disagree_on_{task}"] += 1
                continue
            if label is None and task == "pick" and a["decision"] == "confirmed_theft":
                label, source = 1, "derived"
            if label is None:
                skipped[f"{task}_no_label"] += 1
                continue
            kps = np.asarray([f["kpts"] for f in win], np.float32)
            key = hashlib.sha1(task.encode() + kps.tobytes() + bytes([int(label)])).hexdigest()[:20]
            rel = f"{task}/{key}.npz"
            if add(key, a, task, source, {"window": rel, "label": int(label), "n_frames": len(win)}):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / rel, kps=kps, t=np.asarray([f["t"] for f in win], np.float32))

    removed = [e for k, e in old_by_key.items() if k not in examples]
    for e in removed:                        # decision changed, or the alert passed retention
        for k in ("image", "labels", "window"):
            if e.get(k):
                (out / e[k]).unlink(missing_ok=True)
    ex = list(examples.values())
    manifest = {
        "format": FORMAT, "version": VERSION, "created_ts": old.get("created_ts", now), "updated_ts": now,
        "store_id": store.store_id, "retention_days": store.retention_days, "classes": classes,
        "counts": {**{t: sum(1 for e in ex if e["task"] == t) for t in TASKS},
                   "unknown_classes": sum(1 for e in ex if e.get("class_known") is False),
                   "new_this_export": sum(1 for k in examples if k not in old_by_key),
                   "removed_this_export": len(removed), "duplicates_skipped": dups, "skipped": dict(skipped)},
        "examples": ex,
    }
    write_json(mf, manifest)
    manifest["path"] = str(mf)
    return manifest


def retrain_hook(store, out_dir: str | Path | None = None, min_new: int = 1, command: str | None = None,
                 now: float | None = None) -> dict:
    """What a training job calls. Exports the labels, counts examples added since the last
    successful retrain (per task) and, when at least `min_new` are new and `command` is given,
    runs `<command> <manifest.json>` (also in env BREE_REVIEW_MANIFEST). The command's exit code
    0 marks those examples as used. This module never trains anything itself.

    Returns {"manifest", "new": {task: n}, "new_total", "ready", "ran", "returncode", "error"}.
    A command that cannot be started (not found, not executable) gives ran false and the reason
    in "error"; the examples stay unused."""
    m = export_labels(store, out_dir, now)
    state_path = Path(m["path"]).with_name("retrain_state.json")
    used = set(json.loads(state_path.read_text())["used_keys"]) if state_path.exists() else set()
    fresh = [e for e in m["examples"] if e["key"] not in used]
    res = {"manifest": m["path"], "new": {t: sum(1 for e in fresh if e["task"] == t) for t in TASKS},
           "new_total": len(fresh), "ready": len(fresh) >= min_new, "ran": False, "returncode": None, "error": None}
    if res["ready"] and command:
        try:
            proc = subprocess.run([*shlex.split(command), m["path"]],
                                  env={**os.environ, "BREE_REVIEW_MANIFEST": m["path"]})
        except (OSError, ValueError) as e:       # command not found / not executable / unbalanced quotes
            res["error"] = f"{type(e).__name__}: {e}"
            return res
        res.update(ran=True, returncode=proc.returncode)
        if proc.returncode == 0:
            write_json(state_path, {"used_keys": sorted(e["key"] for e in m["examples"]),
                                    "last_retrain_ts": m["updated_ts"]})
    return res


def load_manifest(path: str | Path) -> dict:
    """For the training side: read and check a manifest. Paths in it are relative to its folder."""
    m = json.loads(Path(path).read_text())
    if m.get("format") != FORMAT or m.get("version") != VERSION:
        raise ValueError(f"{path}: not a {FORMAT} v{VERSION} manifest")
    return m

