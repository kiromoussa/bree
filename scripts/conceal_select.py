"""Pick the classifier variant using PoseLift only (leave one camera out, frames with a pose), never the
evaluation sets. Writes results/conceal_select.json. The winner becomes the default in conceal_experiment.py.
"""
import json
from pathlib import Path

import numpy as np
import torch

from bree.conceal import frame_scores, load_poselift, metrics, model_track_scorer, occupied, train_model

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = {"v1": dict(version=1, augment=0), "v1_aug2": dict(version=1, augment=2),
            "v2": dict(version=2, augment=0), "v2_aug2": dict(version=2, augment=2)}
SEEDS = (0, 1)
vids = load_poselift(Path("data/poselift/PoseLift/Pickle_files"))
test = [v for v in vids if v.split == "test"]
cams = sorted(c for c in {v.cam for v in test} if any(v.labels.any() for v in test if v.cam == c))
res = {"protocol": "leave one camera out on PoseLift, AUC-ROC on frames with a pose, mean over cameras and seeds",
       "seeds": SEEDS, "variants": {}}
for name, kw in VARIANTS.items():
    per = []
    for seed in SEEDS:
        for cam in cams:
            held = [v for v in test if v.cam == cam]
            sc = model_track_scorer(train_model([v for v in vids if v.cam != cam], seed=seed, device=DEV, **kw))
            y = np.concatenate([v.labels for v in held])
            m = np.concatenate([occupied(v) for v in held])
            s = np.concatenate([frame_scores(v, sc) for v in held])
            per.append({"seed": seed, "camera": cam, **metrics(y, s, m)})
    aucs = [p["auc_roc"] for p in per]
    res["variants"][name] = {**kw, "mean_auc_roc": float(np.nanmean(aucs)), "per_camera_seed": per,
                             "mean_excl_cam6": float(np.nanmean([p["auc_roc"] for p in per if p["camera"] != "6"]))}
    print(name, round(res["variants"][name]["mean_auc_roc"], 3), round(res["variants"][name]["mean_excl_cam6"], 3), flush=True)
res["winner"] = max(res["variants"], key=lambda k: res["variants"][k]["mean_auc_roc"])
Path("results/conceal_select.json").write_text(json.dumps(res, indent=1))
print("winner", res["winner"])
