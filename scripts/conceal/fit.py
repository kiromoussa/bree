"""Fit the per-take concealment model on labelled rows (TRAIN-seed clips, SIMULATED) and write model.json.

  .venv/bin/python scripts/conceal/fit.py out/conceal/train/rows.jsonl [--features cusum,empty_cams] [--write]

A logistic regression on a few features of the take's window (bree.concealment.cue.takes_of), over the takes the rules
did not already settle (a put back, or no item ever seen in the hand). Prints leave-one-clip-out precision and recall:
each clip is scored by a model that never saw it. --write stores the model fitted on every row next to cue.py.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bree.concealment.cue import MODEL  # noqa: E402

LOG = ("empty_bins", "seen_before", "tail_s")


def design(rows, feats):
    return np.array([[math.log1p(r[k]) if k in LOG else r[k] for k in feats] for r in rows], float)


def logistic(X, y, C=1.0, steps=50):
    """L2-regularised logistic regression by Newton steps, classes weighted to balance (the intercept is not penalised)."""
    A = np.c_[np.ones(len(X)), X]
    sw = np.where(y == 1, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((1 - y).sum(), 1)))
    w, reg = np.zeros(A.shape[1]), np.r_[0.0, np.ones(X.shape[1])] / C
    for _ in range(steps):
        p = 1 / (1 + np.exp(-A @ w))
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(w))
        w = w - np.linalg.solve(H, A.T @ (sw * (p - y)) + reg * w)
    return w


def fit(rows, feats, C=1.0):
    X, y = design(rows, feats), np.array([r["label"] for r in rows])
    mean, scale = X.mean(0), np.where(X.std(0) > 1e-6, X.std(0), 1.0)
    w = logistic((X - mean) / scale, y, C)
    return {"features": list(feats), "log": [k for k in feats if k in LOG], "mean": mean.round(5).tolist(), "scale": scale.round(5).tolist(),
            "coef": w[1:].round(5).tolist(), "intercept": round(float(w[0]), 5)}


def predict(model, rows):
    X = (design(rows, model["features"]) - np.array(model["mean"])) / np.array(model["scale"])
    return 1 / (1 + np.exp(-(X @ np.array(model["coef"]) + model["intercept"])))


def pr(rows, p, thr):
    tp = sum(r["label"] and q >= thr for r, q in zip(rows, p)); fp = sum((not r["label"]) and q >= thr for r, q in zip(rows, p)); pos = sum(r["label"] for r in rows)
    return tp, fp, pos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--features", default="cusum")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    feats = a.features.split(",")
    allrows = [json.loads(x) for x in open(a.rows)]
    rows = [r for r in allrows if not r["where"]]
    print(f"{len(allrows)} takes, {sum(r['label'] for r in allrows)} followed by a concealment; settled by a rule: {len(allrows) - len(rows)} "
          f"({sum(r['label'] for r in allrows if r['where'])} of them followed by a concealment); fitted on {len(rows)} ({sum(r['label'] for r in rows)} positive)")
    loco = np.zeros(len(rows))
    for clip in sorted({r["clip"] for r in rows}):
        tr = [r for r in rows if r["clip"] != clip]
        te = [i for i, r in enumerate(rows) if r["clip"] == clip]
        if te and len({r["label"] for r in tr}) == 2:
            loco[te] = predict(fit(tr, feats, a.C), [rows[i] for i in te])
    pos_all = sum(r["label"] for r in allrows)
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9):
        tp, fp, _ = pr(rows, loco, thr)
        print(f"leave one clip out, p >= {thr}: true {tp}, false {fp}; precision {tp / max(tp + fp, 1):.3f}, recall {tp / max(pos_all, 1):.3f} of all {pos_all} takes followed by a concealment")
    model = fit(rows, feats, a.C)
    print(json.dumps(model))
    if a.write:
        model["fitted_on"] = {"rows": a.rows, "clips": sorted({r["clip"] for r in rows}), "takes": len(rows), "positive": int(sum(r["label"] for r in rows)), "note": "SIMULATED clips, TRAIN-range seeds"}
        MODEL.write_text(json.dumps(model, indent=1))
        print("wrote", MODEL)
