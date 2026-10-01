"""Train the variant picked by conceal_select.py (PoseLift only) and score it on the held-out sets, next to v1.
Uses cached perception tracks (out/dcsass_tracks, out/ucf_tracks); does not touch models/conceal_poselift.pt.
Writes models/conceal_poselift_<variant>.pt and results/conceal_variant.json.
Usage: python scripts/eval_variant.py [variant]   (default: the winner in results/conceal_select.json)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from bree.conceal import (event_hit, frame_scores, load_bundle, load_poselift, load_retails, metrics,   # noqa: E402
                          model_track_scorer, occupied, save_bundle, train_model)
from eval_ucf import to_video   # noqa: E402

sel = json.loads(Path("results/conceal_select.json").read_text())
name = sys.argv[1] if len(sys.argv) > 1 else sel["winner"]
kw = {k: sel["variants"][name][k] for k in ("version", "augment")}
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
path = Path(f"models/conceal_poselift_{name}.pt")
if not path.exists():
    b = train_model(load_poselift(Path("data/poselift/PoseLift/Pickle_files")), seed=0, device=DEV, **kw)
    save_bundle(b, path, {"trained_on": "PoseLift official, all clips", "variant": name, **kw})
models = {"v1": load_bundle(Path("models/conceal_poselift.pt")), name: load_bundle(path)}
res = {"variant": name, **kw, "selected_on": sel["protocol"], "sets": {}}

R = Path("data/retails/RetailS")
staged = load_retails(R / "RetailS_test_staged/pose/test", R / "RetailS_test_staged/gt/test_frame_mask")
y, m = np.concatenate([v.labels for v in staged]), np.concatenate([occupied(v) for v in staged])
pos = [v for v in staged if v.labels.any()]
res["sets"]["retails_staged"] = {k: {**metrics(y, np.concatenate([frame_scores(v, model_track_scorer(b)) for v in staged]), m),
                                     "caught_at_0.5": sum(event_hit(v, model_track_scorer(b), 0.5) for v in pos), "of": len(pos)}
                                 for k, b in models.items()}

D = Path("data/dcsass/DCSASS Dataset")
rows = [r for r in csv.reader(open(D / "Labels/Shoplifting.csv")) if len(r) == 3 and r[2] in ("0", "1")]
clips = [(to_video(Path(f"out/dcsass_tracks/{n}.npz")), int(l)) for n, _, l in rows if Path(f"out/dcsass_tracks/{n}.npz").exists()]
yc = np.array([l for _, l in clips])
res["sets"]["dcsass"] = {k: metrics(yc, np.array([frame_scores(v, model_track_scorer(b)).max(initial=0.0) for v, _ in clips]))
                         for k, b in models.items()}

U = Path("data/ucf_crime")
ann = {}
for line in (U / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").read_text().split("\n"):
    p = line.split()
    if len(p) >= 6 and p[1] == "Shoplifting":
        ann[Path(p[0]).stem] = [(int(p[2]), int(p[3]))] + ([(int(p[4]), int(p[5]))] if int(p[4]) >= 0 else [])
shop = [to_video(Path(f"out/ucf_tracks/{n}.npz"), iv) for n, iv in ann.items() if Path(f"out/ucf_tracks/{n}.npz").exists()]
yu, mu = np.concatenate([v.labels for v in shop]), np.concatenate([occupied(v) for v in shop])
res["sets"]["ucf_shoplifting"] = {k: metrics(yu, np.concatenate([frame_scores(v, model_track_scorer(b)) for v in shop]), mu)
                                  for k, b in models.items()}
Path("results/conceal_variant.json").write_text(json.dumps(res, indent=1))
for s, by in res["sets"].items():
    print(s, {k: round(v["auc_roc"], 3) for k, v in by.items()})
