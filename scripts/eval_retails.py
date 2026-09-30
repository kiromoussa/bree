"""Evaluate the PoseLift-trained concealment classifier and the pose-only rule on RetailS (EVALUATION ONLY:
RetailS has no stated license). Writes results/conceal_retails.json.

  staged test     624 staged shoplifting videos, same store/cameras, frame labels -> AUC-ROC / AUC-PR / EER
                  and recall at fixed thresholds.
  normal footage  a seeded sample of RetailS_train (all normal, real shoppers) -> would-be triggers per hour.
  (RetailS_test_realworld is the PoseLift test set, byte for byte, so it is skipped: the model trained on it.)

A trigger = one person track whose score stays >= threshold for >= 8 frames (0.5 s at 15 fps); triggers from
the same track closer than 10 s are merged. Thresholds are fixed up front, not tuned on RetailS.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bree.conceal import frame_scores, load_bundle, load_retails, metrics, model_track_scorer, rule_scores

R = Path("data/retails/RetailS")
FPS, MIN_RUN, MERGE_S = 15.0, 8, 10.0
THRESH = {"model": (0.5, 0.7, 0.9), "rule": (0.25, 0.5, 0.75)}
N_NORMAL = int(sys.argv[1]) if len(sys.argv) > 1 else 60

bundle = load_bundle(Path("models/conceal_poselift.pt"))
scorers = {"model": model_track_scorer(bundle), "rule": rule_scores}


def triggers(scores: np.ndarray, frames: np.ndarray, th: float) -> int:
    n, run, last_t = 0, 0, -1e9
    for s, f in zip(scores, frames):
        run = run + 1 if s >= th else 0
        if run == MIN_RUN:
            if f / FPS - last_t > MERGE_S:
                n += 1
            last_t = f / FPS
    return n


res = {"dataset": "RetailS (TeCSAR-UNCC), evaluation only, no license stated",
       "model": "models/conceal_poselift.pt (trained on all PoseLift)", "trigger_rule":
       f">= {MIN_RUN} consecutive frames over threshold per track, merged within {MERGE_S:.0f} s, {FPS:.0f} fps"}

staged = load_retails(R / "RetailS_test_staged/pose/test", R / "RetailS_test_staged/gt/test_frame_mask")
poselift_names = {p.stem for p in Path("data/poselift/PoseLift/Pickle_files/Test").glob("*.pkl")}
overlap = [v.name for v in staged if f"{int(v.name.split('_')[0])}_{int(v.name.split('_')[1])}" in poselift_names] \
    if all(len(v.name.split("_")) == 2 for v in staged) else []
res["staged"] = {"n_videos": len(staged), "names_overlapping_poselift_test": overlap}
y = np.concatenate([v.labels for v in staged])
for k, sc in scorers.items():
    s = np.concatenate([frame_scores(v, sc) for v in staged])
    res["staged"][k] = metrics(y, s)
    res["staged"][k]["recall_at"] = {str(t): float((s[y == 1] >= t).mean()) for t in THRESH[k]}
    res["staged"][k]["frame_fpr_at"] = {str(t): float((s[y == 0] >= t).mean()) for t in THRESH[k]}
    print("staged", k, {m: round(v, 3) for m, v in res["staged"][k].items() if isinstance(v, float)}, flush=True)

files = sorted((R / "RetailS_train/pose/train").glob("*.json"))
pick = sorted(np.random.default_rng(0).choice(len(files), min(N_NORMAL, len(files)), replace=False))
tmp = Path("out/retails_normal_sample")
tmp.mkdir(parents=True, exist_ok=True)
for f in tmp.glob("*.json"):
    f.unlink()
for i in pick:
    (tmp / files[i].name).symlink_to(files[i].resolve())
normal = load_retails(tmp, split="normal")
hours = sum(v.n_frames for v in normal) / FPS / 3600
res["normal"] = {"n_videos": len(normal), "hours": hours, "files": [v.name for v in normal],
                 "cameras": sorted({v.cam for v in normal})}
for k, sc in scorers.items():
    per_track = [(sc(t), t.frames) for v in normal for t in v.tracks.values()]
    res["normal"][k] = {"triggers_per_hour_at": {str(th): sum(triggers(s, f, th) for s, f in per_track) / hours
                                                  for th in THRESH[k]}}
    print("normal", k, res["normal"][k], flush=True)

Path("results/conceal_retails.json").write_text(json.dumps(res, indent=1))
print(f"normal sample: {len(normal)} videos, {hours:.1f} h")
