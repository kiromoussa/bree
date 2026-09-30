"""Concealment classifier vs the pose-only rule on PoseLift. Writes results/conceal_poselift.json
and models/conceal_poselift.pt (trained on all of PoseLift, Apache-2.0).

PoseLift's own Train split is normal-only, so supervised training needs labelled Test videos.
We never score a video the model was trained on:
  cv5      5 folds grouped by video (stratified by "has shoplifting"), 3 seeds; each fold trains on
           PoseLift Train (normal) + the other 4 folds' labelled videos.
  loco     leave one camera out: train on every other camera, test on the held-out camera.
The rule needs no training, so its numbers use the same held-out frames as the model.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bree.conceal import (THRESH, event_hit, frame_scores, video_triggers, load_poselift, metrics, model_track_scorer, rule_scores,
                          save_bundle, train_model)

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/poselift/PoseLift/Pickle_files")
SEEDS = (0, 1, 2)
import torch
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
vids = load_poselift(ROOT)
train_normal = [v for v in vids if v.split == "train"]
test = [v for v in vids if v.split == "test"]
print(f"PoseLift: {len(train_normal)} train (normal) videos, {len(test)} labelled test videos", flush=True)


def evaluate(held, bundle=None):
    y = np.concatenate([v.labels for v in held])
    sc = {"rule": rule_scores}
    if bundle is not None:
        sc["model"] = model_track_scorer(bundle)
    return {k: (y, np.concatenate([frame_scores(v, f) for v in held])) for k, f in sc.items()}, sc


def events(held, sc):
    """Per scorer and threshold: shoplifting clips with a trigger inside the labelled interval, and clean clips
    with any trigger. Counts, so folds can be pooled."""
    pos = [v for v in held if v.labels.any()]
    neg = [v for v in held if not v.labels.any()]
    return {k: {str(th): {"pos": len(pos), "pos_hit": sum(event_hit(v, f, th) for v in pos),
                          "neg": len(neg), "neg_triggered": sum(bool(video_triggers(v, f, th)) for v in neg)}
                for th in THRESH[k]} for k, f in sc.items()}


def folds(k=5, seed=0):
    rng = np.random.default_rng(seed)
    pos = [v for v in test if v.labels.any()]
    neg = [v for v in test if not v.labels.any()]
    out = [[] for _ in range(k)]
    for group in (pos, neg):
        for i, j in enumerate(rng.permutation(len(group))):
            out[i % k].append(group[j])
    return out


res = {"dataset": "PoseLift official (Google Drive, Apache-2.0)", "n_train_normal_videos": len(train_normal),
       "n_labelled_videos": len(test), "protocols": {}}

# 5-fold CV grouped by video
pooled = {s: {"rule": [[], []], "model": [[], []]} for s in SEEDS}
per_fold = []
for seed in SEEDS:
    for fi, held in enumerate(folds(5, seed)):
        names = {v.name for v in held}
        tr = train_normal + [v for v in test if v.name not in names]
        assert not names & {v.name for v in tr}
        ev, sc = evaluate(held, train_model(tr, seed=seed, device=DEV))
        for k, (y, s) in ev.items():
            pooled[seed][k][0].append(y)
            pooled[seed][k][1].append(s)
        per_fold.append({"seed": seed, "fold": fi, "held_out": sorted(names),
                         **{k: metrics(*ev[k]) for k in ev}, "events": events(held, sc)})
        print(seed, fi, {k: round(per_fold[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)
cv = {}
for k in ("rule", "model"):
    per_seed = [metrics(np.concatenate(pooled[s][k][0]), np.concatenate(pooled[s][k][1])) for s in SEEDS]
    cv[k] = {m: {"mean": float(np.mean([p[m] for p in per_seed])), "std": float(np.std([p[m] for p in per_seed]))}
             for m in ("auc_roc", "auc_pr", "eer")}
    cv[k]["n_frames"], cv[k]["n_pos"] = per_seed[0]["n_frames"], per_seed[0]["n_pos"]
ev_tot = {}
for pf in per_fold:
    for k, by_th in pf["events"].items():
        for th, c in by_th.items():
            acc = ev_tot.setdefault(k, {}).setdefault(th, {"pos": 0, "pos_hit": 0, "neg": 0, "neg_triggered": 0})
            for key in acc:
                acc[key] += c[key]
for k in ev_tot:
    for c in ev_tot[k].values():
        c["event_recall"] = c["pos_hit"] / max(c["pos"], 1)
        c["clean_clip_trigger_rate"] = c["neg_triggered"] / max(c["neg"], 1)
res["protocols"]["cv5_by_video"] = {"pooled_out_of_fold_over_seeds": cv, "events_pooled_over_folds_and_seeds": ev_tot,
                                    "per_fold": per_fold}

# leave one camera out
loco = []
for cam in sorted({v.cam for v in test}):
    held = [v for v in test if v.cam == cam]
    if not any(v.labels.any() for v in held):
        continue
    tr = [v for v in train_normal + test if v.cam != cam]
    ev, sc = evaluate(held, train_model(tr, seed=0, device=DEV))
    loco.append({"camera": cam, "n_videos": len(held), **{k: metrics(*ev[k]) for k in ev}})
    print("loco", cam, {k: round(loco[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)
res["protocols"]["leave_one_camera_out"] = loco

# final model on everything (for RetailS / UCF-Crime evaluation and the pipeline)
bundle = train_model(train_normal + test, seed=0, device=DEV)
Path("models").mkdir(exist_ok=True)
save_bundle(bundle, Path("models/conceal_poselift.pt"),
            {"trained_on": "PoseLift official, all Train + Test videos", "window": 24, "fps": 15})
res["final_model"] = "models/conceal_poselift.pt (all PoseLift videos; do not evaluate it on PoseLift)"

Path("results").mkdir(exist_ok=True)
Path("results/conceal_poselift.json").write_text(json.dumps(res, indent=1))
print(json.dumps(cv, indent=1))
