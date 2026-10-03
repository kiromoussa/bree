"""Re-ID benchmark (`make reid-bench`) -> results/reid_bench.json.

Data: MOT16 train (7 street sequences with identity ground truth; research licence, EVALUATION ONLY, already
used by scripts/eval_mot.py) and the repo's toy clips. No embedding or crop is written to disk: everything is
computed and scored in memory in this one process.

1. Re-ID retrieval: ground-truth person boxes sampled once a second (visibility >= 0.5, height >= 80 px).
   Query = one sample; gallery = samples of the same sequence at least 3 s away; rank-1 and mAP per cue and fused.
2. Fusion weights: logistic regression on stitch-candidate pairs (a new tracker id vs. tracks lost up to 60 s
   earlier that could have walked there), labelled by ground truth. Every reported number for a sequence uses
   weights fitted on the OTHER six (leave one sequence out). The weights fitted on all seven are printed for
   bree/track/reid.py.
3. Tracking: the tracker output replayed through the event engine's stitching: off, position guard only (the
   default before re-ID) and guard + re-ID. IDF1 / ID switches / MOTA (same protocol as scripts/eval_mot.py) and
   false merges: stitches that joined tracks of two different ground-truth people, the error that puts one
   shopper's basket on another and so creates false theft alerts.
4. Toy clips: the full pipeline with re-ID on vs off (alert scorecard must not change).

`--quick`: two short sequences only (the regression test, tests/test_reid.py). Not comparable to the full run.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bree.track import reid as R

ROOT = Path(__file__).resolve().parents[3]
MOT = ROOT / "data" / "mot16" / "train"
QUICK_SEQS = ("MOT16-10", "MOT16-13")   # the two short sequences where stitching decisions actually occur
VARIANTS = {   # engine rule overrides per row
    "no_stitch": {"stitch_dist": 0.0},
    "guard": {},                                             # position/time stitching + ambiguity guard
    "guard_reid_short": {"reid": True, "reid_long_s": 0.0},  # + appearance veto only
    "guard_reid": {"reid": True},                            # + appearance relink of longer gaps (engine default)
    # Exploratory rows (thresholds other than the defaults; defaults were not picked to maximise these):
    "guard_reid_reject_0.05": {"reid": True, "reid_reject": 0.05},
    "guard_reid_reject_0.5": {"reid": True, "reid_reject": 0.5},
    "reid_no_guard": {"reid": True, "stitch_ambiguous_skip": False},
}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of xywh boxes, (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = (np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None])).clip(0)
    ih = (np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None])).clip(0)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)


def perceive(seq: Path, backend, embedder, max_frames: int | None = None) -> dict:
    """Detector + tracker + re-ID features for one sequence, all kept in memory."""
    import cv2
    from scipy.optimize import linear_sum_assignment
    from bree.track.bytetrack import Tracker
    info = dict(l.strip().split("=") for l in (seq / "seqinfo.ini").read_text().splitlines() if "=" in l)
    fps = float(info["frameRate"])
    gt_all = np.loadtxt(seq / "gt/gt.txt", delimiter=",")
    gt_all = gt_all[(gt_all[:, 7] == 1) & (gt_all[:, 6] == 1)]
    tracker, ext = Tracker(fps), R.ReidExtractor(embedder)
    frames, samples, track_gt = [], [], defaultdict(Counter)
    for fi, img in enumerate(sorted((seq / "img1").glob("*.jpg"))[:max_frames], start=1):
        im = cv2.imread(str(img))
        persons, _ = tracker.update(*backend(im), im.shape[:2])
        ext.update(im, persons, fi / fps, backend.last_bags)
        g = gt_all[gt_all[:, 0] == fi]
        hb = np.array([[p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]] for p in persons]).reshape(-1, 4)
        iou = iou_matrix(g[:, 2:6], hb)
        for r, c in zip(*linear_sum_assignment(-iou)):
            if iou[r, c] >= 0.5:
                track_gt[persons[c].track_id][int(g[r, 1])] += 1
        frames.append((fi, fi / fps, persons, g[:, 1].astype(int).tolist(), g[:, 2:6]))
        if fi % max(1, round(fps)) == 0:        # re-ID samples from ground-truth boxes, once a second
            keep = g[(g[:, 8] >= 0.5) & (g[:, 5] >= 80)]
            if len(keep):
                boxes = np.stack([keep[:, 2], keep[:, 3], keep[:, 2] + keep[:, 4], keep[:, 3] + keep[:, 5]], 1)
                boxes = boxes.clip(0, [im.shape[1], im.shape[0], im.shape[1], im.shape[0]])
                kps = backend.pose(im, boxes)
                crops = [R.body_crop(im, b, k) for b, k in zip(boxes, kps)]
                ok = [i for i, c in enumerate(crops) if c is not None]
                embs = embedder([crops[i] for i in ok])
                for e, i in zip(embs, ok):
                    samples.append((int(keep[i, 1]), fi / fps, R.ReidFeatures(
                        fi / fps, e, R.part_colors(im, boxes[i], kps[i]), R.body_shape(kps[i]),
                        R.carries_bag(boxes[i], backend.last_bags))))
    return {"name": seq.name, "fps": fps, "frames": frames, "samples": samples,
            "track_gt": {t: c.most_common(1)[0][0] for t, c in track_gt.items()}, "ms": ext.ms}


# ------------------------------------------------------------------ calibration


def candidate_pairs(seq: dict, max_gap: float = 60.0, max_speed: float = 1.5):
    """(pair features, same person?) for every new track vs. each track lost before it that could have walked there."""
    first, last, gal = {}, {}, defaultdict(list)
    X, y = [], []
    for fi, t, persons, _, _ in seq["frames"]:
        present = {p.track_id for p in persons}
        for p in persons:
            if p.track_id not in first:
                first[p.track_id] = t
                a_gt = seq["track_gt"].get(p.track_id)
                if p.reid is not None and a_gt is not None and fi > 1:
                    fx, fy = p.foot_point()
                    for tid, (tl, q) in last.items():
                        gap = t - tl
                        b_gt = seq["track_gt"].get(tid)
                        if tid in present or not 0 < gap <= max_gap or b_gt is None or not gal[tid]:
                            continue
                        lx, ly = q.foot_point()
                        if np.hypot(fx - lx, fy - ly) > (1.0 + max_speed * gap) * max(q.bbox[3] - q.bbox[1], 1.0):
                            continue
                        X.append(R.pair_features(R.summarize(gal[tid]), p.reid))
                        y.append(a_gt == b_gt)
            last[p.track_id] = (t, p)
            R.remember(gal[p.track_id], p.reid, t, 1e9)
    return np.array(X, np.float32).reshape(-1, len(R.FEATURES)), np.array(y, bool)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 200) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton steps (tiny problem; no sklearn dependency)."""
    A = np.hstack([X.astype(np.float64), np.ones((len(X), 1))])
    w = np.zeros(A.shape[1])
    reg = l2 * np.eye(A.shape[1]); reg[-1, -1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ w).clip(-30, 30)))
        g = A.T @ (p - y) + reg @ w
        Hm = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-6 * np.eye(A.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1].astype(np.float32), float(w[-1])


class use_weights:
    """Temporarily swap the fusion weights in bree.track.reid (leave-one-sequence-out scoring)."""

    def __init__(self, w, b):
        self.new = (w, b)

    def __enter__(self):
        self.old = (R.WEIGHTS, R.BIAS)
        R.WEIGHTS, R.BIAS = self.new

    def __exit__(self, *a):
        R.WEIGHTS, R.BIAS = self.old


# ------------------------------------------------------------------ re-ID retrieval


def retrieval(samples, score, min_dt: float = 3.0) -> tuple[int, int, float]:
    """(queries, rank-1 hits, sum of average precision) for one sequence."""
    ids = np.array([s[0] for s in samples]); ts = np.array([s[1] for s in samples])
    n = hits = 0
    ap_sum = 0.0
    for i in range(len(samples)):
        gal = np.flatnonzero(np.abs(ts - ts[i]) >= min_dt)
        same = ids[gal] == ids[i]
        if not same.any():
            continue
        order = np.argsort(-score(i, gal))
        rel = same[order]
        n += 1
        hits += bool(rel[0])
        ap_sum += float((np.cumsum(rel)[rel] / (np.flatnonzero(rel) + 1)).mean())
    return n, hits, ap_sum


def retrieval_scores(samples) -> dict:
    """Scoring functions per cue: (query index, gallery indices) -> similarity."""
    feats = [s[2] for s in samples]
    E = np.stack([f.emb for f in feats]) if feats else np.zeros((0, 384))
    def color(i, gal):
        return np.array([np.mean([x for x in (R._bhatt(feats[i].colors[k], feats[j].colors[k]) for k in range(3))
                                  if x is not None] or [0.0]) for j in gal])
    def fused(i, gal):
        return np.array([R.pair_features(feats[i], feats[j]) @ R.WEIGHTS for j in gal])
    return {"embedding": lambda i, gal: E[gal] @ E[i], "colors": color, "fused": fused}


# ------------------------------------------------------------------ tracking replay


def replay(seq: dict, rules: dict):
    """Run the engine's stitching over the cached tracker output. Returns (motmetrics accumulator, stats)."""
    import motmetrics as mm
    from bree.events.engine import EngineRules, EventEngine
    from bree.events.observations import FrameObs
    from bree.events.zones import load_store_config
    st = load_store_config(ROOT / "configs" / "no_zones.yaml")
    eng = EventEngine(st, EngineRules.from_dict({**st.rules, **rules}))
    acc = mm.MOTAccumulator(auto_id=True)
    tgt = seq["track_gt"]
    last_tid: dict[int, int] = {}          # person id -> tracker id of their latest segment
    stats = Counter()
    for fi, t, persons, gids, gboxes in seq["frames"]:
        before = set(eng.alias)
        eng.update(FrameObs(fi, t, persons, []))
        for p in persons:
            pid = eng.alias.get(p.track_id, p.track_id)
            if p.track_id in eng.alias and p.track_id not in before:      # stitched on this frame
                a, b = tgt.get(p.track_id), tgt.get(last_tid.get(pid, pid))
                stats["stitches"] += 1
                stats["unscored" if a is None or b is None else "correct" if a == b else "false_merges"] += 1
            last_tid[pid] = p.track_id
        hb = np.array([[p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]] for p in persons]).reshape(-1, 4)
        iou = iou_matrix(gboxes, hb)
        d = 1 - iou
        d[iou < 0.5] = np.nan
        acc.update(gids, [eng.alias.get(p.track_id, p.track_id) for p in persons], d)
    return acc, stats


def tracking_rows(seqs: list[dict], weights: dict) -> dict:
    import motmetrics as mm
    out = {}
    for name, rules in VARIANTS.items():
        accs, stats = [], Counter()
        for s in seqs:
            with use_weights(*weights[s["name"]]):
                acc, st = replay(s, rules)
            accs.append(acc); stats += st
        summ = mm.metrics.create().compute_many(
            accs, names=[s["name"] for s in seqs], generate_overall=True,
            metrics=["mota", "idf1", "num_switches", "num_fragmentations", "recall", "precision"])
        o = {k: float(v) for k, v in summ.loc["OVERALL"].items()}
        scored = stats["correct"] + stats["false_merges"]
        o.update(stitches=stats["stitches"], correct_merges=stats["correct"], false_merges=stats["false_merges"],
                 unscored_stitches=stats["unscored"],
                 false_merge_rate=round(stats["false_merges"] / scored, 4) if scored else 0.0,
                 per_sequence_idf1={k: round(float(v), 4) for k, v in summ["idf1"].items() if k != "OVERALL"})
        out[name] = o
        print(f"[reid-bench] {name}: IDF1 {o['idf1']:.3f}  switches {o['num_switches']:.0f}  "
              f"stitches {o['stitches']}  false merges {o['false_merges']} ({o['false_merge_rate']:.1%})", flush=True)
    return out


# ------------------------------------------------------------------ MERL, toy clips, speed


MERL_VARIANTS = {"guard": {}, "guard_reid_short": {"reid": True, "reid_long_s": 0.0}, "guard_reid": {"reid": True}}


def merl_rows(embedder, variants: dict | None = None, split: str = "test", every: int = 1) -> dict:
    """MERL Shopping test subjects (overhead camera, ONE shopper per video; research licence, evaluation only):
    visits per video with the position guard vs guard + re-ID. Every extra visit is a split shopper; there is
    nobody to merge with, so this measures whether the appearance veto breaks correct stitches and whether the
    appearance relink repairs longer gaps. Uses the shipped weights (MERL plays no part in fitting them)."""
    import cv2
    from bree.detect.yolo import YoloBackend
    from bree.events.engine import EngineRules, EventEngine
    from bree.events.observations import FrameObs
    from bree.events.types import EventType
    from bree.events.zones import load_store_config
    from bree.hw import detect_hardware
    from bree.track.bytetrack import Tracker
    vids = sorted((ROOT / "data/merl/Videos_MERL_Shopping_Dataset").glob("*_crop.mp4"))
    subj = lambda v: int(v.name.split("_")[0])  # noqa: E731
    vids = [v for v in vids if (subj(v) >= 27 if split == "test" else subj(v) <= 20)][::every]
    if not vids:
        return {}
    hw = detect_hardware()
    backend = YoloBackend(str(ROOT / "models" / hw.pose_model), str(ROOT / "models" / hw.detect_model), {},
                          device=hw.device, imgsz=hw.imgsz, products=False)
    store = load_store_config(ROOT / "configs" / "merl_overhead.yaml")
    variants = variants or MERL_VARIANTS
    visits = {k: [] for k in variants}
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        tracker, ext = Tracker(fps / 2), R.ReidExtractor(embedder)
        engines = {k: EventEngine(store, EngineRules.from_dict({**store.rules, **r})) for k, r in variants.items()}
        enters = {k: [] for k in variants}
        fi = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if fi % 2 == 0:
                persons, _ = tracker.update(*backend(im), im.shape[:2])
                ext.update(im, persons, fi / fps, backend.last_bags)
                for k, eng in engines.items():
                    enters[k] += [e for e in eng.update(FrameObs(fi, fi / fps, persons, [])) if e.type == EventType.ENTER]
            fi += 1
        for k, eng in engines.items():
            eng.flush()
            visits[k].append(sum(1 for e in enters[k]
                                 if eng.people[e.person_id].t_last - eng.people[e.person_id].t_first >= 2.0))
        print(f"[reid-bench] MERL {v.name}: " + ", ".join(f"{k} {n[-1]}" for k, n in visits.items()), flush=True)
    return {k: {"videos": len(n), "visits_per_video": round(float(np.mean(n)), 3),
                "videos_split": round(float(np.mean([x > 1 for x in n])), 3)} for k, n in visits.items()}


def toy_rows() -> dict:
    """Full pipeline on the toy clips with re-ID off and on: the alert scorecard should not move."""
    import yaml
    from bree.commands import ensure_toy
    from bree.eval.toy_eval import run_toy_suite, score_rows
    ensure_toy()
    base = yaml.safe_load((ROOT / "configs" / "store_gas_station_small.yaml").read_text())
    out = {}
    for name, on in (("reid_off", False), ("reid_on", True)):
        cfg = {**base, "rules": {**(base.get("rules") or {}), "reid": on}}
        path = ROOT / "out" / f"reid_bench_store_{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(cfg))
        rows, summ = run_toy_suite(ROOT / "data" / "toy", ROOT / "out" / f"reid_bench_toy_{name}", path)
        out[name] = {**score_rows(rows), "pipeline_fps": round(float(np.mean([s.pipeline_fps for s in summ])), 1)}
    return out


def speed(embedder_cls=R.Embedder) -> dict:
    """ms per crop on CPU (ONNX Runtime), embedding and the cheap cues separately."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    box = (300.0, 100.0, 380.0, 340.0)
    out = {}
    for threads in (1, 4):
        e = embedder_cls(threads=threads)
        crop = [R.body_crop(img, box, None)]
        for _ in range(3):
            e(crop)
        t0 = time.perf_counter()
        for _ in range(20):
            e(crop)
        out[f"embedding_ms_per_crop_cpu_{threads}_thread"] = round((time.perf_counter() - t0) / 20 * 1000, 2)
    t0 = time.perf_counter()
    for _ in range(200):
        R.part_colors(img, box, None)
    out["colors_shape_ms_per_crop"] = round((time.perf_counter() - t0) / 200 * 1000, 3)
    return out


# ------------------------------------------------------------------ main


def run(quick: bool = False, toy: bool = True, max_frames: int | None = None) -> dict:
    from bree.detect.yolo import YoloBackend
    from bree.hw import detect_hardware
    hw = detect_hardware()
    backend = YoloBackend(str(ROOT / "models/yolo26s-pose.pt"), str(ROOT / "models/yolo26s.pt"), {}, device=hw.device,
                          imgsz=hw.imgsz, person_conf=0.3, products=False, dedupe_inside=0.85)
    embedder = R.Embedder()
    paths = sorted(p for p in MOT.iterdir() if (p / "gt/gt.txt").exists() and (not quick or p.name in QUICK_SEQS))
    seqs = []
    for p in paths:
        seqs.append(perceive(p, backend, embedder, max_frames))
        print(f"[reid-bench] perceived {p.name}: {len(seqs[-1]['frames'])} frames, {len(seqs[-1]['samples'])} re-ID samples", flush=True)

    pairs = {s["name"]: candidate_pairs(s) for s in seqs}
    def fit(names):
        X = np.concatenate([pairs[n][0] for n in names]); y = np.concatenate([pairs[n][1] for n in names])
        return fit_logistic(X, y)
    shipped = (R.WEIGHTS, R.BIAS)
    # Full run: each sequence scored with weights fitted on the others. Quick run: the shipped weights.
    loso = {s["name"]: shipped if quick else fit([n for n in pairs if n != s["name"]]) for s in seqs}
    all_w, all_b = fit(list(pairs))

    res = {"dataset": "MOT16 train (motchallenge.net), research licence, evaluation only", "quick": quick,
           "sequences": [s["name"] for s in seqs], "embedder": "DINOv2 ViT-S/14 (Apache-2.0), ONNX Runtime, 224x112 body crop",
           "calibration": {"pairs": int(sum(len(p[1]) for p in pairs.values())),
                           "positives": int(sum(p[1].sum() for p in pairs.values())), "features": list(R.FEATURES),
                           "weights_all_sequences": [round(float(v), 4) for v in all_w], "bias_all_sequences": round(all_b, 4),
                           "shipped_weights": [round(float(v), 4) for v in R.WEIGHTS], "shipped_bias": float(R.BIAS)}}

    tot = defaultdict(lambda: [0, 0, 0.0])
    for s in seqs:
        with use_weights(*loso[s["name"]]):
            for cue, fn in retrieval_scores(s["samples"]).items():
                for k, v in enumerate(retrieval(s["samples"], fn)):
                    tot[cue][k] += v
    res["reid"] = {cue: {"queries": n, "rank1": round(h / max(n, 1), 4), "mAP": round(ap / max(n, 1), 4)}
                   for cue, (n, h, ap) in tot.items()}
    res["reid"]["samples"] = sum(len(s["samples"]) for s in seqs)
    res["reid"]["identities"] = sum(len({x[0] for x in s["samples"]}) for s in seqs)
    print("[reid-bench] re-ID:", res["reid"], flush=True)

    res["tracking"] = tracking_rows(seqs, loso)
    ms = [m for s in seqs for m in s["ms"]]
    res["speed"] = {**speed(), "in_pipeline_ms_per_crop_all_cues": round(float(np.mean(ms)), 2) if ms else None}
    if toy and not quick:
        res["toy"] = toy_rows()
        res["merl_test"] = merl_rows(embedder)
    return res


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    res = run(quick=quick, toy="--no-toy" not in argv)
    out = ROOT / "results" / ("reid_bench_quick.json" if quick else "reid_bench.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"[reid-bench] wrote {out}")


if __name__ == "__main__":
    main()
