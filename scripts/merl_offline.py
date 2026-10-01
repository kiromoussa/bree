"""Offline variants on cached MERL person tracks (scripts/measure_merl.py --save-tracks): no YOLO re-run.
Same definitions as measure_merl.py (reach recall on Reach To Shelf + Hand In Shelf instances +-0.5 s, false reach
runs per minute, visits = engine ENTERs lasting >= 2 s). Variants:
  hand_k     reach point = wrist + k * (wrist - elbow): the fingertips go deeper into a shelf than the wrist
  dedupe     drop a person box that lies >= 85% inside a larger one (duplicate detections of one person)
  stitch     engine stitching (stitch_s, stitch_dist)
Usage: python scripts/merl_offline.py {train,test} [--grid] -> results/merl_offline_<split>.json
  train --grid: try every variant; test: only the variants named in results/merl_offline_train.json["chosen"].
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bree.events.engine import EngineRules, EventEngine
from bree.events.observations import FrameObs, PersonObs
from bree.events.types import EventType
from bree.events.zones import Zone, load_store_config

SPLIT = sys.argv[1]
GRID = "--grid" in sys.argv
SHELF = Zone("shelf", "shelf", [(190, 30), (880, 30), (880, 300), (190, 300)])
STORE = load_store_config("configs/merl_overhead.yaml")
L_EL, R_EL, L_WR, R_WR = 7, 8, 9, 10


def load(f):
    d = np.load(f)
    rows, fps, stride = d["rows"], float(d["fps"]), int(d["stride"])
    frames = {}
    for r in rows:
        frames.setdefault(int(r[0]), []).append((int(r[1]), tuple(r[2:6]), r[6:].reshape(17, 3)))
    sampled = list(range(0, int(d["n_frames"]), stride))
    return frames, sampled, fps


def dedupe(people):
    def area(b):
        return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    keep = []
    for i, (tid, b, k) in enumerate(people):
        inside = False
        for j, (_, c, _) in enumerate(people):
            if j != i and area(c) > area(b):
                ix = max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
                inside |= ix >= 0.85 * max(area(b), 1)
        if not inside:
            keep.append((tid, b, k))
    return keep


def hand_points(k, kp):
    out = []
    for el, wr in ((L_EL, L_WR), (R_EL, R_WR)):
        if kp[wr, 2] < 0.3:
            continue
        p = kp[wr, :2]
        if k and kp[el, 2] >= 0.3:
            p = p + k * (p - kp[el, :2])
        out.append(p)
    return out


def evaluate(videos, hand_k, dd, stitch_s, stitch_dist, stitch_long_s=0.0, stitch_near=0.5):
    tot = dict(reach=0, found=0, false_runs=0, minutes=0.0, visits=[], split=0)
    for stem, (frames, sampled, fps), gt in videos:
        pad = 0.5 * fps
        rules = EngineRules.from_dict({**STORE.rules, "stitch_s": stitch_s, "stitch_dist": stitch_dist,
                                       "stitch_long_s": stitch_long_s, "stitch_near": stitch_near, "hand_extend": hand_k})
        eng, enters, hits = EventEngine(STORE, rules), [], []
        for fi in sampled:
            ppl = frames.get(fi, [])
            if dd:
                ppl = dedupe(ppl)
            hits.append(any(SHELF.contains(*p) for _, _, kp in ppl for p in hand_points(hand_k, kp)))
            obs = [PersonObs(tid, b, 1.0, kp) for tid, b, kp in ppl]
            enters += [e for e in eng.update(FrameObs(fi, fi / fps, obs, [])) if e.type == EventType.ENTER]
        eng.flush()
        visits = sum(1 for e in enters if (p := eng.people[e.person_id]).t_last - p.t_first >= 2.0)
        fr, h = np.array(sampled), np.array(hits)
        reach_gt = np.concatenate([gt[1], gt[3]])
        tot["reach"] += len(reach_gt)
        tot["found"] += sum(bool(h[(fr >= s - pad) & (fr <= e + pad)].any()) for s, e in reach_gt)
        runs, start, prev = [], None, None
        for f, x in zip(list(fr) + [None], list(h) + [False]):
            if x and start is None:
                start = f
            elif not x and start is not None:
                runs.append((start, prev))
                start = None
            prev = f
        shelf_gt = np.concatenate([gt[1], gt[2], gt[3]])
        tot["false_runs"] += sum(1 for s0, e0 in runs if e0 > s0 and
                                 not any(s0 <= e + pad and e0 >= s - pad for s, e in shelf_gt))
        tot["minutes"] += sampled[-1] / fps / 60 if sampled else 0
        tot["visits"].append(visits)
        tot["split"] += visits > 1
    n = len(videos)
    return {"reach_recall": tot["found"] / max(tot["reach"], 1), "reach_instances": tot["reach"],
            "false_reach_runs_per_min": tot["false_runs"] / max(tot["minutes"], 1e-9),
            "visits_per_video": float(np.mean(tot["visits"])), "videos_split": tot["split"] / n, "n_videos": n}


videos = []
for f in sorted(Path(f"out/merl_tracks/{SPLIT}").glob("*.npz")):
    tl = loadmat(f"data/merl/Labels_MERL_Shopping_Dataset/{f.stem}_label.mat")["tlabs"]
    videos.append((f.stem, load(f), {k + 1: np.asarray(tl[k][0]).reshape(-1, 2) for k in range(5)}))

base = dict(hand_k=0.0, dd=False, stitch_s=10.0, stitch_dist=1.0)
if GRID:
    combos = [dict(hand_k=k, dd=d, stitch_s=s, stitch_dist=r)
              for k, d, s, r in itertools.product((0.0, 0.25, 0.5), (False, True), (10.0, 20.0), (0.0, 1.0, 2.0))]
else:
    combos = [base, json.loads(Path("results/merl_offline_train.json").read_text())["chosen"]]
res = {"split": SPLIT, "n_videos": len(videos), "rows": []}
for c in combos:
    r = evaluate(videos, **c)
    res["rows"].append({**c, **r})
    print(c, {k: round(v, 3) for k, v in r.items()}, flush=True)
if GRID:
    # Pre-stated rule: keep reach recall within 1 point of the best, then fewest visits per video, then
    # fewest false reaches.
    best = max(r["reach_recall"] for r in res["rows"])
    ok = [r for r in res["rows"] if r["reach_recall"] >= best - 0.01]
    ch = min(ok, key=lambda r: (r["visits_per_video"], r["false_reach_runs_per_min"]))
    res["chosen"] = {k: ch[k] for k in ("hand_k", "dd", "stitch_s", "stitch_dist")}
    print("chosen", res["chosen"])
Path(f"results/merl_offline_{SPLIT}.json").write_text(json.dumps(res, indent=1))
