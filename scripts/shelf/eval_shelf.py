"""Shelf events against simulator truth on TRAIN-seed clips (SIMULATED). Scoring only reads truth/; the detector side
(bree.shelf.events.run_clip) reads video, calibration.json, layout.json and clip.json.

    .venv/bin/python scripts/shelf/eval_shelf.py 4903 4904 4905 --name heldout
    .venv/bin/python scripts/shelf/eval_shelf.py 4900 --no-detector --score-only

Unit of the per-camera table: one (truth pick, item camera) pair where the camera saw the picked item at --min-px
or more (truth `cameras[].px`). A pair is found when that camera reported a shelf event of the same kind within
--tol-s of the truth time and within --tol-m of the truth slot. Precision: share of a camera's events that match
any truth act (a take, or the put of a put-back) in time and place, whatever its pixel size. Strict precision
allows one event per act and camera; `repeats` are the further events of a camera for an act it already matched.
slot_or_adj: the right slot, or the facing next to it (0.1 m or less) holding the same SKU.
Idle cameras: rendered item cameras that are in no truth act's camera list; their events are all false.
    --tag _allcams   reads clip_<seed>_allcams (a clip rendered with every item camera of the layout)
Writes out/shelf/eval_<name>.json and .md; shelf events are cached in out/shelf/events/clip_<seed>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITEM_KINDS = ("shelf", "cooler", "checkout")


def truth_acts(clip: Path) -> list[dict]:
    """One act per take and per put-back: kind, t, slot, sku, place, cameras {id: px}."""
    acts = []
    for line in (clip / "truth" / "events.jsonl").read_text().splitlines():
        r = json.loads(line)
        base = {"slot": r["slotId"], "sku": r["skuId"], "p": r["slotFace"], "zone": r["zone"], "shopper": r["shopper"],
                "cams": {c["id"]: c["px"] for c in r["cameras"]}}
        acts.append({**base, "kind": "take", "t": r["t"]})
        if r.get("tPutBack") is not None:
            acts.append({**base, "kind": "put", "t": r["tPutBack"]})
    return acts


def near(e: dict, a: dict, tol_s: float, tol_m: float) -> bool:
    return (e["kind"] == a["kind"] and e.get("point_3d") is not None and min(abs(e["t"] - a["t"]), abs(e["t_start"] - a["t"])) <= tol_s
            and float(np.linalg.norm(np.subtract(e["point_3d"], a["p"]))) <= tol_m)


def score_clip(clip: Path, events: list[dict], kinds: dict, rendered: list[str], min_px: float, tol_s: float, tol_m: float, slots: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(pairs, event rows). A truth act takes at most one event per camera (the closest in time, right slot first)."""
    acts, used, pairs, matched = truth_acts(clip), set(), [], set()
    for ai, a in enumerate(acts):
        n_same = sum(1 for b in acts if b["kind"] == a["kind"] and b["slot"] == a["slot"] and abs(b["t"] - a["t"]) <= 1.0)
        for cam, px in a["cams"].items():
            if cam not in rendered or kinds.get(cam) not in ITEM_KINDS:
                continue
            cands = [(e["slot_id"] != a["slot"], abs(e["t"] - a["t"]), i) for i, e in enumerate(events)
                     if e["camera_id"] == cam and i not in used and near(e, a, tol_s, tol_m)]
            row = {"clip": clip.name, "camera": cam, "cam_kind": kinds[cam], "act": a["kind"], "t": a["t"], "slot": a["slot"], "sku": a["sku"], "px": px,
                   "scored": px >= min_px, "found": bool(cands)}
            if cands:
                i = min(cands)[2]
                used.add(i)
                matched.add((ai, cam))
                e = events[i]
                adj = bool(slots) and e["sku_id"] == a["sku"] and float(np.linalg.norm(np.subtract(slots[e["slot_id"]]["face"], a["p"]))) <= 0.1
                row.update(event=i, right_slot=e["slot_id"] == a["slot"], slot_or_adj=e["slot_id"] == a["slot"] or adj, right_sku=e["sku_id"] == a["sku"], right_count=e["count"] == n_same,
                           dt=round(e["t"] - a["t"], 3), source=e["source"], got_slot=e["slot_id"])
            pairs.append(row)
    rows = []
    for i, e in enumerate(events):
        hit = i in used or any(near(e, a, tol_s, tol_m) for a in acts)
        rows.append({"clip": clip.name, "camera": e["camera_id"], "cam_kind": kinds[e["camera_id"]], "kind": e["kind"], "t": e["t"], "slot": e["slot_id"],
                     "source": e["source"], "true": hit, "strict": i in used, "count": e["count"],
                     "repeat": i not in used and any((ai, e["camera_id"]) in matched and near(e, a, tol_s, tol_m) for ai, a in enumerate(acts))})
    return pairs, rows


def idle_cameras(clip: Path, kinds: dict, rendered: list[str]) -> list[str]:
    seen = {c for a in truth_acts(clip) for c in a["cams"]}
    return [c for c in rendered if kinds.get(c) in ITEM_KINDS and c not in seen]


def store_level(clip: Path, events: list[dict], layout: dict, kinds: dict, rendered: list[str], min_px: float, tol_s: float, tol_m: float) -> list[dict]:
    """After fuse_views: one row per truth take that some rendered item camera saw at min_px or more."""
    from bree.shelf.events import fuse_views
    fused, out, used = fuse_views([dict(e) for e in events], layout), [], set()
    for a in truth_acts(clip):
        if a["kind"] != "take" or not any(px >= min_px and c in rendered and kinds.get(c) in ITEM_KINDS for c, px in a["cams"].items()):
            continue
        cands = [(e["slot_id"] != a["slot"], abs(e["t"] - a["t"]), i) for i, e in enumerate(fused) if i not in used and near(e, a, tol_s, tol_m)]
        row = {"clip": clip.name, "t": a["t"], "slot": a["slot"], "zone": a["zone"], "found": bool(cands)}
        if cands:
            i = min(cands)[2]
            used.add(i)
            row.update(right_slot=fused[i]["slot_id"] == a["slot"], right_sku=fused[i]["sku_id"] == a["sku"], views=len(fused[i]["cameras"]))
        out.append(row)
    return out, len(fused), sum(1 for i, e in enumerate(fused) if i in used or any(near(e, a, tol_s, tol_m) for a in truth_acts(clip)))


def table(pairs: list[dict], rows: list[dict], act: str = "take") -> list[dict]:
    out = []
    for kind in (*ITEM_KINDS, "all"):
        p = [r for r in pairs if r["scored"] and r["act"] == act and kind in (r["cam_kind"], "all")]
        e = [r for r in rows if r["kind"] == act and kind in (r["cam_kind"], "all")]
        f = [r for r in p if r["found"]]
        dt = np.abs([r["dt"] for r in f]) if f else np.array([np.nan])
        frac = lambda k: round(sum(bool(r.get(k)) for r in p) / len(p), 3) if p else None
        out.append({"camera_kind": kind, "act": act, "pairs": len(p), "recall": frac("found"), "right_slot": frac("right_slot"), "slot_or_adj": frac("slot_or_adj"), "right_sku": frac("right_sku"),
                    "right_count": frac("right_count"), "events": len(e), "precision": round(sum(r["true"] for r in e) / len(e), 3) if e else None,
                    "strict_precision": round(sum(r["strict"] for r in e) / len(e), 3) if e else None, "repeats": sum(r["repeat"] for r in e),
                    "dt_median_s": round(float(np.median(dt)), 2) if f else None, "dt_p90_s": round(float(np.percentile(dt, 90)), 2) if f else None})
    return out


def markdown(res: dict) -> str:
    cols = ["camera_kind", "act", "pairs", "recall", "right_slot", "slot_or_adj", "right_sku", "right_count", "events", "precision", "strict_precision", "repeats", "dt_median_s", "dt_p90_s"]
    lines = [f"# Shelf events, {res['name']} (SIMULATED, TRAIN seeds {', '.join(res['clips'])})", "",
             f"Detector: {res['detector']}. A pair is one truth pick and one item camera that saw the item at {res['min_px']} px or more. "
             f"Match: same camera, same kind, within {res['tol_s']} s and {res['tol_m']} m. right_slot, right_sku and right_count are shares of all pairs (a miss counts as wrong).", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in res["per_camera"]]
    s = res["store"]
    lines += ["", f"Store level (views fused, bree.shelf.events.fuse_views): {s['picks']} picks, recall {s['recall']}, right slot {s['right_slot']}, "
              f"right SKU {s['right_sku']}; {s['events']} fused events, {s['true_events']} match a truth act.", "",
              f"Idle cameras (item cameras that see no truth act): {res['idle']['cameras']} cameras, {res['idle']['minutes']} camera minutes, {res['idle']['events']} events, "
              f"{res['idle']['per_camera_minute']} per camera per minute. Events with count above 1: {res['count_above_1']}. Camera status records: {res['status']}.", "",
              f"Wall time {res['seconds']} s for {res['camera_frames']} camera frames ({res['jobs']} processes)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--name", default="run")
    ap.add_argument("--root", default=str(ROOT / "data/synth/bench/train"))
    ap.add_argument("--out", default=str(ROOT / "out/shelf"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--no-detector", action="store_true", help="shelf comparison alone (no SKU detector, no hand cue)")
    ap.add_argument("--score-only", action="store_true", help="reuse cached shelf events")
    ap.add_argument("--tag", default="", help="clip folder is clip_<seed><tag>")
    ap.add_argument("--min-px", type=float, default=20.0)
    ap.add_argument("--tol-s", type=float, default=3.0)
    ap.add_argument("--tol-m", type=float, default=0.6)
    a = ap.parse_args()
    from bree.shelf.events import run_clip
    manifest = json.loads((ROOT / "scripts/bench/manifest.json").read_text())
    lo, hi = (int(v) for v in manifest["splits"]["train"]["seeds"].split(":"))
    assert all(lo <= s < hi for s in a.seeds), "TRAIN seeds only (scripts/bench/manifest.json)"
    out = Path(a.out)
    (out / "events").mkdir(parents=True, exist_ok=True)
    pairs, rows, store, n_fused, n_fused_true, secs, frames = [], [], [], 0, 0, 0.0, 0
    idle_n, idle_min, idle_ev, status_n = 0, 0.0, 0, {}
    for seed in a.seeds:
        clip = Path(a.root) / f"clip_{seed}{a.tag}"
        meta = json.loads((clip / "clip.json").read_text())
        kinds = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
        cache = out / "events" / f"clip_{seed}{a.tag}{'_nodet' if a.no_detector else ''}.jsonl"
        health = cache.with_suffix(".status.jsonl")
        if not (a.score_only and cache.exists()):
            t0, st = time.time(), []
            evs = run_clip(clip, jobs=a.jobs, detector=not a.no_detector, verbose=True, status=st)
            secs += time.time() - t0
            frames += meta["frames"] * sum(1 for c in meta["cameras"] if kinds[c] in ITEM_KINDS)
            cache.write_text("".join(json.dumps(e) + "\n" for e in evs))
            health.write_text("".join(json.dumps(x) + "\n" for x in st))
        evs = [json.loads(line) for line in cache.read_text().splitlines()]
        for x in (json.loads(line) for line in health.read_text().splitlines()) if health.exists() else []:
            status_n[x["status"]] = status_n.get(x["status"], 0) + 1
        layout = json.loads((clip / "layout.json").read_text())
        p, r = score_clip(clip, evs, kinds, meta["cameras"], a.min_px, a.tol_s, a.tol_m, {x["id"]: x for x in layout["slots"]})
        idle = idle_cameras(clip, kinds, meta["cameras"])
        idle_n, idle_min, idle_ev = idle_n + len(idle), idle_min + len(idle) * meta["frames"] / meta["fps"] / 60, idle_ev + sum(e["camera_id"] in idle for e in evs)
        s, nf, nt = store_level(clip, evs, layout, kinds, meta["cameras"], a.min_px, a.tol_s, a.tol_m)
        pairs, rows, store, n_fused, n_fused_true = pairs + p, rows + r, store + s, n_fused + nf, n_fused_true + nt
        print(f"clip {seed}: {len(evs)} shelf events, {sum(x['scored'] and x['act'] == 'take' for x in p)} scored pick pairs, "
              f"{sum(x['scored'] and x['act'] == 'take' and x['found'] for x in p)} found", flush=True)
    frac = lambda k: round(sum(bool(r.get(k)) for r in store) / len(store), 3) if store else None
    res = {"name": a.name, "simulated": True, "clips": [str(s) for s in a.seeds], "detector": "off (shelf comparison alone)" if a.no_detector else "sim_sku on moving regions",
           "min_px": a.min_px, "tol_s": a.tol_s, "tol_m": a.tol_m, "jobs": a.jobs, "seconds": round(secs, 1), "camera_frames": frames,
           "per_camera": table(pairs, rows, "take") + table(pairs, rows, "put"),
           "idle": {"cameras": idle_n, "minutes": round(idle_min, 1), "events": idle_ev, "per_camera_minute": round(idle_ev / idle_min, 2) if idle_min else None},
           "count_above_1": sum(r["count"] > 1 for r in rows), "status": status_n,
           "store": {"picks": len(store), "recall": frac("found"), "right_slot": frac("right_slot"), "right_sku": frac("right_sku"), "events": n_fused, "true_events": n_fused_true},
           "pairs": pairs, "events": rows, "store_picks": store}
    (out / f"eval_{a.name}.json").write_text(json.dumps(res, indent=1))
    (out / f"eval_{a.name}.md").write_text(markdown(res))
    print(markdown(res))


if __name__ == "__main__":
    main()
