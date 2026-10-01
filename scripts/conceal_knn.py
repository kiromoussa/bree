"""Unsupervised alternative to the concealment classifier: how far is a pose window from normal shopping?
Normal bank = windows from PoseLift's Train split only (all normal, Apache-2.0); no shoplifting labels are used to
fit anything. Score = mean distance to the k nearest normal windows (after standardising + PCA, fit on the bank).
Validation: PoseLift labelled clips (pick k / PCA size there). Held out, scored once with the chosen setting:
RetailS staged, DCSASS, UCF-Crime shoplifting (cached tracks). Threshold for triggers: the 99th percentile of the
bank's own leave-out scores. Writes results/conceal_knn.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from bree.conceal import (Track, frame_scores, load_poselift, load_retails, metrics, occupied,  # noqa: E402
                          track_features, windows)
from eval_ucf import to_video   # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STEP = 4                    # keep every 4th frame of the 24-frame window (6 x 97 features)
rng = np.random.default_rng(0)


def feats(t: Track) -> np.ndarray:
    w = windows(track_features(t, 2))[:, ::STEP]
    return w.reshape(len(w), -1).astype(np.float32)


class Knn:
    def __init__(self, bank: np.ndarray, k: int, dims: int):
        self.mu, self.sd = bank.mean(0), bank.std(0) + 1e-6
        x = torch.tensor((bank - self.mu) / self.sd)
        _, _, v = torch.pca_lowrank(x, q=dims, center=False)
        self.v = v
        self.bank = (x @ v).to(DEV)
        self.k = k

    def score(self, x: np.ndarray, leave_one_out: bool = False) -> np.ndarray:
        q = (torch.tensor((x - self.mu) / self.sd) @ self.v).to(DEV)
        out = []
        for i in range(0, len(q), 2048):
            d = torch.cdist(q[i:i + 2048], self.bank)
            kk = self.k + (1 if leave_one_out else 0)
            out.append(d.topk(kk, largest=False).values[:, (1 if leave_one_out else 0):].mean(1).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)


def scorer(model):
    return lambda t: model.score(feats(t))


vids = load_poselift(Path("data/poselift/PoseLift/Pickle_files"))
train = [v for v in vids if v.split == "train"]
test = [v for v in vids if v.split == "test"]
bank_all = np.concatenate([feats(t) for v in train for t in v.tracks.values() if len(t.frames)])
bank = bank_all[rng.choice(len(bank_all), min(40000, len(bank_all)), replace=False)]
print("bank windows", len(bank_all), "used", len(bank), flush=True)

res = {"bank": "PoseLift Train (normal only)", "bank_windows": int(len(bank)), "validation": [], "held_out": {}}
y = np.concatenate([v.labels for v in test])
m = np.concatenate([occupied(v) for v in test])
for k in (1, 5, 20):
    for dims in (16, 64):
        model = Knn(bank, k, dims)
        r = metrics(y, np.concatenate([frame_scores(v, scorer(model)) for v in test]), m)
        res["validation"].append({"k": k, "pca": dims, **r})
        print("val", k, dims, round(r["auc_roc"], 3), flush=True)
best = max(res["validation"], key=lambda r: r["auc_roc"])
res["chosen"] = {"k": best["k"], "pca": best["pca"]}
model = Knn(bank, best["k"], best["pca"])
th = float(np.percentile(model.score(bank[rng.choice(len(bank), 5000, replace=False)], leave_one_out=True), 99))
res["threshold_p99_normal"] = th

R = Path("data/retails/RetailS")
st = load_retails(R / "RetailS_test_staged/pose/test", R / "RetailS_test_staged/gt/test_frame_mask")
res["held_out"]["retails_staged"] = metrics(np.concatenate([v.labels for v in st]),
                                            np.concatenate([frame_scores(v, scorer(model)) for v in st]),
                                            np.concatenate([occupied(v) for v in st]))
D = Path("data/dcsass/DCSASS Dataset")
rows = [r for r in csv.reader(open(D / "Labels/Shoplifting.csv")) if len(r) == 3 and r[2] in ("0", "1")]
clips = [(to_video(Path(f"out/dcsass_tracks/{n}.npz")), int(l)) for n, _, l in rows if Path(f"out/dcsass_tracks/{n}.npz").exists()]
res["held_out"]["dcsass"] = metrics(np.array([l for _, l in clips]),
                                    np.array([frame_scores(v, scorer(model)).max(initial=0.0) for v, _ in clips]))
U = Path("data/ucf_crime")
ann = {}
for line in (U / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt").read_text().split("\n"):
    p = line.split()
    if len(p) >= 6 and p[1] == "Shoplifting":
        ann[Path(p[0]).stem] = [(int(p[2]), int(p[3]))] + ([(int(p[4]), int(p[5]))] if int(p[4]) >= 0 else [])
shop = [to_video(Path(f"out/ucf_tracks/{n}.npz"), iv) for n, iv in ann.items() if Path(f"out/ucf_tracks/{n}.npz").exists()]
res["held_out"]["ucf_shoplifting"] = metrics(np.concatenate([v.labels for v in shop]),
                                             np.concatenate([frame_scores(v, scorer(model)) for v in shop]),
                                             np.concatenate([occupied(v) for v in shop]))
Path("results/conceal_knn.json").write_text(json.dumps(res, indent=1))
print("chosen", res["chosen"], {k: round(v["auc_roc"], 3) for k, v in res["held_out"].items()})
