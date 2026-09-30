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

from bree.conceal import (frame_scores, load_poselift, metrics, model_track_scorer, rule_scores,
                          save_bundle, train_model)

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/poselift/PoseLift/Pickle_files")
SEEDS = (0, 1, 2)
vids = load_poselift(ROOT)
train_normal = [v for v in vids if v.split == "train"]
test = [v for v in vids if v.split == "test"]
print(f"PoseLift: {len(train_normal)} train (normal) videos, {len(test)} labelled test videos", flush=True)


def evaluate(held, bundle=None):
    y = np.concatenate([v.labels for v in held])
    r = np.concatenate([frame_scores(v, rule_scores) for v in held])
    out = {"rule": (y, r)}
    if bundle is not None:
        out["model"] = (y, np.concatenate([frame_scores(v, model_track_scorer(bundle)) for v in held]))
    return out


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
        ev = evaluate(held, train_model(tr, seed=seed))
        for k, (y, s) in ev.items():
            pooled[seed][k][0].append(y)
            pooled[seed][k][1].append(s)
        per_fold.append({"seed": seed, "fold": fi, "held_out": sorted(names),
                         **{k: metrics(*ev[k]) for k in ev}})
        print(seed, fi, {k: round(per_fold[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)
cv = {}
for k in ("rule", "model"):
    per_seed = [metrics(np.concatenate(pooled[s][k][0]), np.concatenate(pooled[s][k][1])) for s in SEEDS]
    cv[k] = {m: {"mean": float(np.mean([p[m] for p in per_seed])), "std": float(np.std([p[m] for p in per_seed]))}
             for m in ("auc_roc", "auc_pr", "eer")}
    cv[k]["n_frames"], cv[k]["n_pos"] = per_seed[0]["n_frames"], per_seed[0]["n_pos"]
res["protocols"]["cv5_by_video"] = {"pooled_out_of_fold_over_seeds": cv, "per_fold": per_fold}

# leave one camera out
loco = []
for cam in sorted({v.cam for v in test}):
    held = [v for v in test if v.cam == cam]
    if not any(v.labels.any() for v in held):
        continue
    tr = [v for v in train_normal + test if v.cam != cam]
    ev = evaluate(held, train_model(tr, seed=0))
    loco.append({"camera": cam, "n_videos": len(held), **{k: metrics(*ev[k]) for k in ev}})
    print("loco", cam, {k: round(loco[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)
res["protocols"]["leave_one_camera_out"] = loco

# final model on everything (for RetailS / UCF-Crime evaluation and the pipeline)
bundle = train_model(train_normal + test, seed=0)
Path("models").mkdir(exist_ok=True)
save_bundle(bundle, Path("models/conceal_poselift.pt"),
            {"trained_on": "PoseLift official, all Train + Test videos", "window": 24, "fps": 15})
res["final_model"] = "models/conceal_poselift.pt (all PoseLift videos; do not evaluate it on PoseLift)"

Path("results").mkdir(exist_ok=True)
Path("results/conceal_poselift.json").write_text(json.dumps(res, indent=1))
print(json.dumps(cv, indent=1))
