"""Concealment classifier vs the pose-only rule on PoseLift. Writes results/conceal_poselift.json
and models/conceal_poselift.pt (trained on all of PoseLift, Apache-2.0).

PoseLift's own Train split is normal-only, so supervised training needs labelled Test videos.
PoseLift clips are consecutive cuts of continuous recordings, so clips are grouped into incident chains
(bree.conceal.incident_chains) and a chain is never on both sides of a split:
  cv5      5 folds of chains (stratified by "chain has shoplifting"), 3 seeds; each fold trains on every
           clip (Train and Test) whose chain is not held out.
  loco     leave one camera out: train on every other camera, test on the held-out camera.
Frame metrics are on frames with at least one pose (empty frames score 0 for free); all-frame AUC is kept
for reference. Event counts are per unique clip, averaged over seeds. The rule needs no training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bree.conceal import (THRESH, event_hit, frame_scores, incident_chains, load_poselift, metrics,
                          model_track_scorer, occupied, rule_scores, save_bundle, train_model, video_triggers)

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/poselift/PoseLift/Pickle_files")
SEEDS = (0, 1, 2)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
vids = load_poselift(ROOT)
chain = incident_chains(vids)
test = [v for v in vids if v.split == "test"]
test_chains = sorted({chain[v.name] for v in test})
pos_chains = sorted({chain[v.name] for v in test if v.labels.any()})
print(f"PoseLift: {len(vids)} clips, {len(test)} labelled, {len(test_chains)} chains containing labelled clips "
      f"({len(pos_chains)} with shoplifting)", flush=True)


def scorers(bundle):
    return {"rule": rule_scores, **({"model": model_track_scorer(bundle)} if bundle else {})}


def frame_eval(held, sc):
    y = np.concatenate([v.labels for v in held])
    m = np.concatenate([occupied(v) for v in held])
    return {k: (y, np.concatenate([frame_scores(v, f) for v in held]), m) for k, f in sc.items()}


def event_eval(held, sc):
    pos = [v for v in held if v.labels.any()]
    neg = [v for v in held if not v.labels.any()]
    return {k: {str(th): {"pos": len(pos), "pos_hit": sum(event_hit(v, f, th) for v in pos),
                          "neg": len(neg), "neg_triggered": sum(bool(video_triggers(v, f, th)) for v in neg)}
                for th in THRESH[k]} for k, f in sc.items()}


def folds(k, seed):
    rng = np.random.default_rng(seed)
    out = [[] for _ in range(k)]
    for group in (pos_chains, [c for c in test_chains if c not in pos_chains]):
        for i, j in enumerate(rng.permutation(len(group))):
            out[i % k].append(group[j])
    return out


res = {"dataset": "PoseLift official (Google Drive, Apache-2.0)", "n_clips": len(vids), "n_labelled_clips": len(test),
       "n_chains_with_labelled_clips": len(test_chains), "chains": {v.name: chain[v.name] for v in vids},
       "protocols": {}}

pooled = {s: {} for s in SEEDS}
per_fold, events = [], {}
for seed in SEEDS:
    for fi, held_chains in enumerate(folds(5, seed)):
        held = [v for v in test if chain[v.name] in held_chains]
        tr = [v for v in vids if chain[v.name] not in held_chains]
        assert not {chain[v.name] for v in tr} & set(held_chains)
        sc = scorers(train_model(tr, seed=seed, device=DEV))
        fe = frame_eval(held, sc)
        for k, arrs in fe.items():
            acc = pooled[seed].setdefault(k, ([], [], []))
            for i, a in enumerate(arrs):
                acc[i].append(a)
        ev = event_eval(held, sc)
        for k, by in ev.items():
            for th, c in by.items():
                acc = events.setdefault(k, {}).setdefault(th, {"pos": 0, "pos_hit": 0, "neg": 0, "neg_triggered": 0})
                for key in acc:
                    acc[key] += c[key]
        per_fold.append({"seed": seed, "fold": fi, "held_out": sorted(v.name for v in held),
                         **{k: metrics(y, s, m) for k, (y, s, m) in fe.items()}, "events": ev})
        print(seed, fi, {k: round(per_fold[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)

cv = {}
for k in ("rule", "model"):
    per_seed = [metrics(*(np.concatenate(a) for a in pooled[s][k])) for s in SEEDS]
    cv[k] = {m: {"mean": float(np.mean([p[m] for p in per_seed])), "std": float(np.std([p[m] for p in per_seed]))}
             for m in ("auc_roc", "auc_pr", "eer", "all_frames_auc_roc")}
    cv[k]["n_frames"], cv[k]["n_pos"] = per_seed[0]["n_frames"], per_seed[0]["n_pos"]
    fa = [f[k]["auc_roc"] for f in per_fold if not np.isnan(f[k]["auc_roc"])]
    cv[k]["per_fold_auc_roc"] = {"mean": float(np.mean(fa)), "min": float(np.min(fa)), "max": float(np.max(fa))}
for k in events:   # counts were summed over seeds: report per unique clip (average over seeds)
    for c in events[k].values():
        for key in list(c):
            c[key] = c[key] / len(SEEDS)
        c["event_recall"] = c["pos_hit"] / max(c["pos"], 1)
        c["clean_clip_trigger_rate"] = c["neg_triggered"] / max(c["neg"], 1)
res["protocols"]["cv5_by_incident_chain"] = {"pooled_out_of_fold_over_seeds": cv,
                                             "events_per_unique_clip_mean_over_seeds": events, "per_fold": per_fold}

loco = []
for cam in sorted({v.cam for v in test}):
    held = [v for v in test if v.cam == cam]
    if not any(v.labels.any() for v in held):
        continue
    sc = scorers(train_model([v for v in vids if v.cam != cam], seed=0, device=DEV))
    fe = frame_eval(held, sc)
    loco.append({"camera": cam, "n_clips": len(held), **{k: metrics(y, s, m) for k, (y, s, m) in fe.items()},
                 "events": event_eval(held, sc)})
    print("loco", cam, {k: round(loco[-1][k]["auc_roc"], 3) for k in ("rule", "model")}, flush=True)
res["protocols"]["leave_one_camera_out"] = loco
res["protocols"]["leave_one_camera_out_mean_auc_roc"] = {
    k: float(np.nanmean([r[k]["auc_roc"] for r in loco])) for k in ("rule", "model")}

bundle = train_model(vids, seed=0, device=DEV)
Path("models").mkdir(exist_ok=True)
save_bundle(bundle, Path("models/conceal_poselift.pt"),
            {"trained_on": "PoseLift official, all Train + Test clips", "window": 24, "fps": 15})
res["final_model"] = "models/conceal_poselift.pt (all PoseLift clips; do not evaluate it on PoseLift)"
Path("results/conceal_poselift.json").write_text(json.dumps(res, indent=1))
print(json.dumps(cv, indent=1))
