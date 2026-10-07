"""Train and score the "where the hand went" crop classifier on rendered hand frames (SIMULATED).

  .venv/bin/python scripts/conceal/train_where.py [--epochs 12]

Training frames: data/synth/hands/frames (seeds 2000 to 2043, scripts/train/render_hands.mjs). Held-out frames:
data/synth/hands/heldout_frames (seeds 3000 to 3011): whole scenes never trained on. Both are TRAIN-range seeds.
Labels come from the simulator's truth of each hand box: the reaching arm in reach / retract / put back / pay is "at
the shelf", every other hand is "at the body"; `holding` says an item is in it. Crops are cut around the truth hand box
with random shift and scale (the detector's boxes are not exact), random colour, brightness, blur, noise, mirror, and
random coloured patches pasted over part of the crop (a stand-in for the bags and clothing the simulator does not have).
Writes models/conceal_where.pt and results/conceal_where.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bree.concealment.where import CLASSES, SIZE, WEIGHTS, crop, net  # noqa: E402

BIG, BIG_CONTEXT = 96, 4.5      # stored crop: wider than the model's, so shift and scale can be drawn at training time
AT_SHELF = ("reach", "retract", "putback")


def label(h: dict) -> int:
    shelf = h["arm"] == 0 and (h["state"] in AT_SHELF or (h["state"] == "pay" and h["reaching"]))
    return (0 if shelf else 2) + int(bool(h["holding"]))


def build(frames: Path, cache: Path, min_px: int = 150):
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"], z["seed"]
    xs, ys, seeds = [], [], []
    for f in sorted(frames.glob("s*/*.json")):
        r = json.loads(f.read_text())
        hs = [h for h in r["hands"] if h["vis_px"] >= min_px and h["shopper"] != "clerk" and min(h["bbox"][2] - h["bbox"][0], h["bbox"][3] - h["bbox"][1]) >= 12]
        if not hs:
            continue
        im = cv2.imread(str(f.with_suffix(".jpg")))
        for h in hs:
            xs.append(crop(im, h["bbox"], BIG_CONTEXT, BIG)); ys.append(label(h)); seeds.append(r["seed"])
    x, y, s = np.stack(xs), np.array(ys), np.array(seeds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, seed=s)
    return x, y, s


def augment(x: np.ndarray, rng) -> np.ndarray:
    """One stored crop (BIG, BGR) -> one training crop (SIZE)."""
    k = rng.uniform(2.4, 3.8) / BIG_CONTEXT * BIG          # side in stored pixels
    c = BIG / 2 + rng.uniform(-0.3, 0.3, 2) * BIG / BIG_CONTEXT
    M = np.array([[SIZE / k, 0, -(c[0] - k / 2) * SIZE / k], [0, SIZE / k, -(c[1] - k / 2) * SIZE / k]])
    im = cv2.warpAffine(x, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    if rng.random() < 0.5:
        im = im[:, ::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-12, 12)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4); hsv[..., 2] *= rng.uniform(0.55, 1.35)
    im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    for _ in range(rng.integers(0, 3)):        # coloured patches away from the centre: clutter the simulator does not have
        w, h = rng.integers(8, 26, 2)
        x0, y0 = rng.integers(0, SIZE - w), rng.integers(0, SIZE - h)
        if abs(x0 + w / 2 - SIZE / 2) < 14 and abs(y0 + h / 2 - SIZE / 2) < 14:
            continue
        im[y0:y0 + h, x0:x0 + w] = rng.integers(0, 256, 3)
    if rng.random() < 0.3:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.5:
        im = np.clip(im.astype(np.float32) + rng.normal(0, rng.uniform(2, 10), im.shape), 0, 255).astype(np.uint8)
    return im


def centre(x: np.ndarray) -> np.ndarray:
    k = 3.0 / BIG_CONTEXT * BIG
    M = np.array([[SIZE / k, 0, -(BIG / 2 - k / 2) * SIZE / k], [0, SIZE / k, -(BIG / 2 - k / 2) * SIZE / k]])
    return cv2.warpAffine(x, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR)


def tens(batch) -> torch.Tensor:
    return torch.from_numpy(np.stack(batch)[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--score", help="score these weights on the held-out crops and write them as the model, no training (a run stopped early leaves models/conceal_where.tmp.pt)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.time()
    xtr, ytr, _ = build(ROOT / "data/synth/hands/frames", ROOT / "data/synth/conceal/where_train.npz")
    xte, yte, ste = build(ROOT / "data/synth/hands/heldout_frames", ROOT / "data/synth/conceal/where_heldout.npz")
    print(f"crops: train {len(ytr)} {np.bincount(ytr, minlength=4).tolist()}, held out {len(yte)} {np.bincount(yte, minlength=4).tolist()} ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = net().to(a.device)
    w = torch.tensor(len(ytr) / (4 * np.maximum(np.bincount(ytr, minlength=4), 1)), dtype=torch.float32).clamp(max=5).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    steps = a.epochs * (len(ytr) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=0.15)
    WEIGHTS_TMP = WEIGHTS.with_suffix(".tmp.pt")
    WEIGHTS.parent.mkdir(exist_ok=True)
    xt = tens([centre(x) for x in xte])
    if a.score:
        model.load_state_dict(torch.load(a.score, map_location="cpu"))
        torch.save(model.state_dict(), WEIGHTS_TMP)
    for ep in range(0 if a.score else a.epochs):
        model.train()
        order = rng.permutation(len(ytr))
        for i in range(0, len(order) - 127, 128):
            idx = order[i:i + 128]
            loss = torch.nn.functional.cross_entropy(model(tens([augment(xtr[j], rng) for j in idx]).to(a.device)), torch.from_numpy(ytr[idx]).to(a.device), weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pr = torch.cat([torch.softmax(model(xt[i:i + 512].to(a.device)), 1).cpu() for i in range(0, len(xt), 512)]).numpy()
        print(f"epoch {ep + 1}: loss {loss.item():.3f}, held-out accuracy {(pr.argmax(1) == yte).mean():.3f} ({time.time() - t0:.0f} s)", flush=True)
        torch.save(model.state_dict(), WEIGHTS_TMP)      # the last epoch is kept, not the best one: the held-out frames pick nothing
    WEIGHTS_TMP.replace(WEIGHTS)
    model.eval()
    with torch.no_grad():
        pr = torch.cat([torch.softmax(model(xt[i:i + 512].to(a.device)), 1).cpu() for i in range(0, len(xt), 512)]).numpy()
    pred = pr.argmax(1)
    conf = [[int(((yte == i) & (pred == j)).sum()) for j in range(4)] for i in range(4)]
    def two(truth, score, thr=0.5):
        tp = int((truth & (score >= thr)).sum()); fp = int((~truth & (score >= thr)).sum()); fn = int((truth & (score < thr)).sum())
        return {"true": tp, "false": fp, "missed": fn, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
    res = {"what": "SIMULATED. Hand crops from rendered frames, held-out scenes (seeds of data/synth/hands/heldout_frames), truth hand boxes, centre crop.",
           "classes": CLASSES, "train_crops": int(len(ytr)), "heldout_crops": int(len(yte)), "heldout_seeds": sorted(set(ste.tolist())),
           "accuracy": round(float((pred == yte).mean()), 3), "confusion_rows_truth_cols_predicted": conf,
           "at_shelf": two(yte < 2, pr[:, :2].sum(1)), "item_in_hand": two(yte % 2 == 1, pr[:, 1] + pr[:, 3]), "epochs": a.note or a.epochs,
           "at_shelf_by_threshold": {str(t): two(yte < 2, pr[:, :2].sum(1), t) for t in (0.3, 0.5, 0.7, 0.9)},
           "item_in_hand_by_threshold": {str(t): two(yte % 2 == 1, pr[:, 1] + pr[:, 3], t) for t in (0.3, 0.5, 0.7, 0.9)}, "weights": str(WEIGHTS.relative_to(ROOT))}
    (ROOT / "results/conceal_where.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
