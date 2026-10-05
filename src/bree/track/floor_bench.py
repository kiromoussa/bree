"""Measure the store-wide floor tracker (bree.track.floor) and the shelf event association (bree.track.associate)
on SIMULATED clips. Ground truth is read here and only here, for scoring.

    python -m bree.track.floor_bench dev                      # the 6 DEV clips of the benchmark
    python -m bree.track.floor_bench dev --clips 7001 -v
    python -m bree.track.floor_bench door                     # data/synth/clip_5001_door (TRACK-2, TRACK-3)
    python -m bree.track.floor_bench dev --write results/assoc_dev.json

Person boxes come from a pipeline run's frame logs (frames_<camera>.jsonl: boxes and pose keypoints found in the
pixels by the person model), so the detector does not have to run again. Default: the recorded baseline run.

Identity score, same rule as bree.sim.bench: a store-wide id belongs to the shopper most of its boxes sit on
(IoU 0.3 with the true box, at least 5 boxes); ids per shopper is how many ids a shopper got.
Association score: each true pick is turned into a shelf event in the shared contract (true time and slot, nothing
else) and given to the association with the tracks the tracker made from pixels; right is the id whose boxes sit on
the true shopper around that time.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bree.calib.camera import Camera, from_layout
from bree.track.floor import FloorConfig, FloorTracker

ROOT = Path(__file__).resolve().parents[3]
PEOPLE_KINDS = ("overhead", "entrance")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _iou(a, b) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    return w * h / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h) if w > 0 and h > 0 else 0.0


def load_boxes(pipe: Path, cams: list[str], n: int, min_conf: float = 0.0) -> dict[str, list[list[dict]]]:
    """Per camera, per frame: [{"bbox", "conf", "kpts"}] from a run's frame logs."""
    out = {}
    for cam in cams:
        frames: list[list[dict]] = [[] for _ in range(n)]
        for r in _jsonl(pipe / f"frames_{cam}.jsonl"):
            if r["frame"] < n:
                frames[r["frame"]] = [p for p in r["persons"] if p.get("conf", 1.0) >= min_conf]
        out[cam] = frames
    return out


def run_tracker(cams: dict[str, Camera], layout: dict, boxes: dict[str, list[list[dict]]], fps: float, cfg: FloorConfig | None = None):
    """-> (tracker, ids[camera][frame] = store-wide id of each box)."""
    tracker = FloorTracker(cams, layout=layout, cfg=cfg)
    n = max(len(v) for v in boxes.values())
    ids: dict[str, list[list]] = {c: [] for c in boxes}
    for f in range(n):
        got = tracker.update(f / fps, {c: boxes[c][f] for c in boxes})
        for c in boxes:
            ids[c].append(got[c])
    tracker.finish(n / fps)
    return tracker, ids


def score_identity(boxes, ids, truth_frames: dict, names: list[str]) -> tuple[dict, dict]:
    seen: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for cam, frames in boxes.items():
        for f, ps in enumerate(frames):
            gt = (truth_frames.get((cam, f)) or {}).get("persons", [])
            for p, gid in zip(ps, ids[cam][f]):
                if gid is None:
                    continue
                seen[gid]                                  # an id that never sits on a person still counts as made
                best = max(gt, key=lambda g: _iou(g["bbox"], p["bbox"]), default=None)
                if best and _iou(best["bbox"], p["bbox"]) >= 0.3:
                    seen[gid].append((f, best["shopper"]))
    main = {g: Counter(s for _, s in obs).most_common(1)[0][0] for g, obs in seen.items() if len(obs) >= 5}
    per = {n: sorted(g for g, s in main.items() if s == n) for n in names}
    mixed = [g for g, obs in seen.items() if len(obs) >= 5 and (c := Counter(s for _, s in obs if s != "clerk"))
             and c.most_common(1)[0][1] < 0.9 * sum(c.values())]
    return {"ids": len(seen), "ids_on_nobody": sum(1 for g in seen if g not in main), "ids_on_clerk": sum(1 for s in main.values() if s == "clerk"),
            "ids_per_shopper": {n: len(v) for n, v in per.items()}, "never_tracked": [n for n, v in per.items() if not v],
            "ids_covering_two_shoppers": len(mixed), "mixed_ids": mixed}, seen


def shopper_of(seen, gid, f, win=25):
    obs = seen.get(gid) or []
    near = [s for ff, s in obs if abs(ff - f) <= win]
    votes = Counter(near or [s for _, s in obs])
    return votes.most_common(1)[0][0] if votes else None


def bench_clip(clip: Path, pipe: Path, cfg: FloorConfig | None = None, verbose: bool = False, min_conf: float = 0.0,
               chain_out: Path | None = None, oracle_conceal: bool = False) -> dict:
    from bree.sim.bench import load_calibration
    from bree.track.associate import associate
    meta, layout = json.loads((clip / "clip.json").read_text()), json.loads((clip / "layout.json").read_text())
    fps = float(meta["fps"])
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    cams = {c: v for c, v in load_calibration(clip).items() if kind[c] in PEOPLE_KINDS and c in meta["cameras"]}
    boxes = load_boxes(pipe, sorted(cams), int(meta["frames"]), min_conf)
    tracker, ids = run_tracker(cams, layout, boxes, fps, cfg)
    tf = {(r["camera"], r["frame"]): r for r in _jsonl(clip / "truth" / "frames.jsonl")}
    names = [s["shopper"] for s in json.loads((clip / "truth" / "shoppers.json").read_text())]
    ident, seen = score_identity(boxes, ids, tf, names)
    # how far the track of a shopper is from where the shopper really stands
    pos = {r["frame"]: {s["shopper"]: (s["x"], s["z"]) for s in r["shoppers"]} for r in _jsonl(clip / "truth" / "tracks.jsonl")}
    err, covered, total = [], 0, 0
    main = {g: Counter(s for _, s in obs).most_common(1)[0][0] for g, obs in seen.items() if obs}
    paths = {k.id: {round(t * fps): (x, z) for t, x, z in k.path} for k in tracker.people()}
    for f, row in pos.items():
        for s, xz in row.items():
            total += 1
            d = [np.hypot(p[f][0] - xz[0], p[f][1] - xz[1]) for g, p in paths.items() if main.get(g) == s and f in p]
            if d:
                covered += 1
                err.append(min(d))
    # association: true picks as shelf events
    truth = _jsonl(clip / "truth" / "events.jsonl")
    slots = {s["id"]: s for s in layout["slots"]}
    shelf = []
    for g in truth:
        shelf.append({"camera_id": "truth", "t": g["t"], "t_start": g["t"] - 0.4, "t_end": g["t"] + 0.4, "kind": "take", "slot_id": g["slotId"],
                      "sku_id": g["skuId"], "sku_conf": 1.0, "count": 1, "source": "shelf_diff", "hand_px": None,
                      "point_3d": slots[g["slotId"]]["face"], "point_sigma_m": 0.05, "evidence": {"before": None, "after": None, "frames": []}})
    for g in truth:
        if g.get("tPutBack") is not None:
            shelf.append({**shelf[truth.index(g)], "t": g["tPutBack"], "t_start": g["tPutBack"] - 0.4, "t_end": g["tPutBack"] + 0.4, "kind": "put"})
    res = associate(shelf, tracker.people(), layout, cams=cams)
    chain = None
    if chain_out is not None:        # the whole way to alerts, scored by the benchmark's own scorer
        from bree.events.shelf import load_payments, run_ledger, store_events, write_run
        from bree.sim.bench import score_clip
        cues = [{"t": g["tConceal"], "person_id": a.person_id, "sku_id": g["skuId"], "conf": 0.6, "source": "ORACLE"}
                for g, a in zip(truth, res) if g.get("tConceal") is not None and a.person_id is not None] if oracle_conceal else None
        events, _ = store_events(shelf, tracker.people(), layout, cams=cams, conceal=cues)
        alerts, _ = run_ledger(events, load_payments(clip / "register.jsonl"), layout)
        write_run(chain_out / clip.name, events, alerts, boxes, ids, fps)
        chain = score_clip(clip, chain_out / clip.name)
    rows = []
    for g, a in zip(truth, res[:len(truth)]):
        who = shopper_of(seen, a.person_id, round(g["t"] * fps)) if a.person_id is not None else None
        rows.append({"t": g["t"], "shopper": g["shopper"], "slot": g["slotId"], "person_id": a.person_id, "event_shopper": who, "uncertain": a.uncertain,
                     "candidates": a.candidates, "right": who == g["shopper"], "margin": a.margin, "why": a.why})
        if verbose and who != g["shopper"]:
            print("   pick", rows[-1])
    exits = sum(1 for k in tracker.people() if k.state == "exited")
    out = {"clip": meta["seed"], "cameras": sorted(cams), "shoppers": len(names), "identity": ident,
           "people_made": len(tracker.people()), "exits_seen": exits, "true_exits": sum(1 for s in json.loads((clip / "truth" / "shoppers.json").read_text()) if s["tExit"] <= meta["frames"] / fps),
           "uncertain_ids": sum(1 for k in tracker.people() if k.uncertain and not k.staff),
           "mixed_ids_not_marked_uncertain": sum(1 for k in tracker.people() if k.id in ident["mixed_ids"] and not k.uncertain), "counts": dict(tracker.counts),
           "position": {"shopper_frames": total, "covered": covered, "median_m": round(float(np.median(err)), 3) if err else None,
                        "p90_m": round(float(np.percentile(err, 90)), 3) if err else None},
           "association": rows, "chain": chain}
    if verbose:
        print("\n".join("   " + line for line in tracker.log))
    return out


def bench_door(clip: Path, pipe: Path, cfg: FloorConfig | None = None, verbose: bool = False) -> dict:
    """The older clip: no calibration.json and no true person boxes. Cameras come from layout.json, identity is scored by
    the true item boxes (an item in a hand names the shopper holding it: the id of the person box around it)."""
    meta, layout = json.loads((clip / "clip.json").read_text()), json.loads((clip / "layout.json").read_text())
    fps = float(meta["fps"])
    cams = {c["id"]: from_layout(c) for c in layout["cameras"] if c["id"] in meta["cameras"] and c["id"].startswith("TRACK")}
    boxes = load_boxes(pipe, sorted(cams), int(meta["frames"]))
    tracker, ids = run_tracker(cams, layout, boxes, fps, cfg)
    seen: dict[int, list] = defaultdict(list)
    for r in _jsonl(clip / "truth_frames.jsonl"):
        if r["camera"] not in boxes or r["frame"] >= len(boxes[r["camera"]]):
            continue
        for it in r["items"]:
            if it["kind"] != "hand":
                continue
            u, v = (it["bbox"][0] + it["bbox"][2]) / 2, (it["bbox"][1] + it["bbox"][3]) / 2
            inside = [g for p, g in zip(boxes[r["camera"]][r["frame"]], ids[r["camera"]][r["frame"]])
                      if g is not None and p["bbox"][0] - 10 <= u <= p["bbox"][2] + 10 and p["bbox"][1] - 10 <= v <= p["bbox"][3] + 10]
            if len(inside) == 1:
                seen[inside[0]].append((r["frame"], it["shopper"]))
    people = tracker.people()
    shoppers = sorted({json.loads(line)["shopper"] for line in (clip / "events.jsonl").read_text().splitlines() if line.strip()})
    main = {g: Counter(s for _, s in obs).most_common(1)[0][0] for g, obs in seen.items() if len(obs) >= 5}
    staff = [k.id for k in people if k.staff]
    out = {"clip": meta["seed"], "cameras": sorted(cams), "shoppers": len(shoppers), "people_made": len(people), "staff_ids": staff,
           "shopper_ids": len(people) - len(staff), "ids_per_shopper": round((len(people) - len(staff)) / len(shoppers), 2),
           "ids_by_item_in_hand": {s: sorted(g for g, m in main.items() if m == s) for s in shoppers},
           "ids_holding_two_shoppers_items": sum(1 for obs in seen.values() if len(obs) >= 5 and Counter(s for _, s in obs).most_common(1)[0][1] < 0.9 * len(obs)),
           "exits_seen": sum(1 for k in people if k.state == "exited"), "uncertain_ids": sum(1 for k in people if k.uncertain), "counts": dict(tracker.counts),
           "tracks": [{"id": k.id, "from_s": round(k.path[0][0], 1), "to_s": round(k.path[-1][0], 1), "state": k.state, "staff": k.staff, "born": k.born,
                       "uncertain": k.uncertain} for k in people if k.path]}
    if verbose:
        print("\n".join("   " + line for line in tracker.log))
    return out


def summary(clips: list[dict]) -> dict:
    per = [n for c in clips for n in c["identity"]["ids_per_shopper"].values()]
    rows = [r for c in clips for r in c["association"]]
    sure = [r for r in rows if r["person_id"] is not None and not r["uncertain"]]
    return {"clips": [c["clip"] for c in clips], "shoppers": len(per), "ids_per_shopper": round(sum(per) / max(sum(1 for n in per if n), 1), 3),
            "shoppers_with_one_id": sum(1 for n in per if n == 1), "shoppers_never_tracked": sum(1 for n in per if n == 0),
            "ids_covering_two_shoppers": sum(c["identity"]["ids_covering_two_shoppers"] for c in clips),
            "ids_on_clerk": sum(c["identity"]["ids_on_clerk"] for c in clips), "ids_on_nobody": sum(c["identity"]["ids_on_nobody"] for c in clips),
            "exits_seen": sum(c["exits_seen"] for c in clips), "true_exits": sum(c["true_exits"] for c in clips),
            "uncertain_ids": sum(c["uncertain_ids"] for c in clips), "mixed_ids_not_marked_uncertain": sum(c["mixed_ids_not_marked_uncertain"] for c in clips),
            "position_median_m": round(float(np.median([c["position"]["median_m"] for c in clips if c["position"]["median_m"] is not None])), 3),
            "position_covered": round(sum(c["position"]["covered"] for c in clips) / max(sum(c["position"]["shopper_frames"] for c in clips), 1), 3),
            "picks": len(rows), "picks_right_shopper": sum(r["right"] for r in rows),
            "picks_right_and_sure": sum(r["right"] for r in sure), "picks_wrong_and_sure": sum(not r["right"] for r in sure),
            "picks_marked_uncertain": sum(1 for r in rows if r["uncertain"]), "picks_uncertain_and_right": sum(r["right"] for r in rows if r["uncertain"]),
            "picks_unassigned": sum(1 for r in rows if r["person_id"] is None)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["dev", "door"])
    ap.add_argument("--clips", help="comma separated seeds")
    ap.add_argument("--boxes", help="folder of a pipeline run: <boxes>/clip_<seed>/pipeline/frames_<camera>.jsonl (dev) or the pipeline folder itself (door)")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--encounter-m", type=float, help="FloorConfig.encounter_m")
    ap.add_argument("--write")
    ap.add_argument("--chain", action="store_true", help="also make events and alerts from the true picks (as shelf events) and score them with bree.sim.bench")
    ap.add_argument("--oracle-conceal", action="store_true", help="with --chain: feed the true conceal times as conceal cues (upper bound, not a pipeline result)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    cfg = FloorConfig(encounter_m=args.encounter_m) if args.encounter_m is not None else None
    if args.what == "door":
        clip = ROOT / "data/synth/clip_5001_door"
        res = bench_door(clip, Path(args.boxes) if args.boxes else clip / "e2e/sim_eval/pipeline", cfg=cfg, verbose=args.verbose)
        print(json.dumps(res, indent=1))
    else:
        from bree.sim.bench import clip_dirs
        only = [int(s) for s in args.clips.split(",")] if args.clips else None
        base = Path(args.boxes) if args.boxes else ROOT / "out/bench/dev_baseline"
        clips = []
        for clip in clip_dirs("dev", only):
            c = bench_clip(clip, base / clip.name / "pipeline", cfg=cfg, verbose=args.verbose, min_conf=args.min_conf,
                           chain_out=ROOT / "out/assoc" / ("chain_conceal" if args.oracle_conceal else "chain") if args.chain else None, oracle_conceal=args.oracle_conceal)
            clips.append(c)
            a = c["association"]
            print(f"{c['clip']}: shoppers {c['shoppers']}, ids {c['identity']['ids']} (clerk {c['identity']['ids_on_clerk']}, nobody {c['identity']['ids_on_nobody']}), "
                  f"per shopper {list(c['identity']['ids_per_shopper'].values())}, two-shopper ids {c['identity']['ids_covering_two_shoppers']}, "
                  f"exits {c['exits_seen']}/{c['true_exits']}, pos median {c['position']['median_m']} m, "
                  f"picks right {sum(r['right'] for r in a)}/{len(a)} (uncertain {sum(r['uncertain'] for r in a)}, none {sum(r['person_id'] is None for r in a)})", flush=True)
        res = {"label": "SIMULATED (browser store simulator copy). Person boxes from " + str(base.relative_to(ROOT) if base.is_relative_to(ROOT) else base),
               "summary": summary(clips), "clips": clips}
        print(json.dumps(res["summary"], indent=1))
        if args.chain:
            from bree.sim.bench import aggregate
            sc = aggregate([c["chain"] for c in clips])["scorecard"]
            res["chain_scorecard"] = sc
            print("chain (true picks and put-backs as shelf events" + (", true conceal times as cues" if args.oracle_conceal else "") + "):")
            print(json.dumps({k: sc[k] for k in ("thieves", "stolen_items", "thefts_alerted", "thefts_alerted_or_reviewed", "theft_recall_alert_or_review", "alerts", "true_alerts",
                                                 "false_alerts_on_honest_shoppers", "reviews", "reviews_on_honest_shoppers", "pipeline_picks", "pick_recall") if k in sc}, indent=1))
    if args.write:
        Path(args.write).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
