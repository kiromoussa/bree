"""Can an open-vocabulary detector (YOLOE, text prompts, no training) see a product in the hand?
MERL labels "Inspect Product" (holding a product) intervals. For sampled frames we ask: is an object detected within
`R` x (forearm length) of a wrist? Positives: frames inside Inspect Product. Negatives: frames with no labelled action
(hands empty, shopper away from the shelf). Uses cached wrists (out/merl_tracks/<split>). Writes
results/merl_product_in_hand_<split>.json with the hit rate on positives / negatives per prompt set and threshold.
Usage: python scripts/merl_product_in_hand.py {train,test} [prompt_set] [conf]   (test: give the train-chosen ones)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
from ultralytics import YOLOE

SPLIT = sys.argv[1]
PROMPTS = {"generic": ["product package", "box", "bag", "bottle", "can"],
           "snacks": ["snack bag", "chip bag", "candy box", "cereal box", "food package"],
           "object": ["object held in hand"]}
CONFS = (0.05, 0.1, 0.2, 0.3)
R = 1.0                       # object centre within R forearm lengths of a wrist
PER_VIDEO = 40                # sampled frames per class per video
rng = np.random.default_rng(0)
sets = [sys.argv[2]] if len(sys.argv) > 2 else list(PROMPTS)
confs = [float(sys.argv[3])] if len(sys.argv) > 3 else list(CONFS)
model = YOLOE("models/yoloe-26s-seg.pt")
PE = {s: model.get_text_pe(PROMPTS[s]) for s in sets}    # text embeddings once per prompt set
res = {"split": SPLIT, "radius_forearms": R, "frames_per_class_per_video": PER_VIDEO, "rows": []}
counts = {(s, c): {"pos": 0, "pos_hit": 0, "neg": 0, "neg_hit": 0} for s in sets for c in confs}

for f in sorted(Path(f"out/merl_tracks/{SPLIT}").glob("*.npz")):
    d = np.load(f)
    rows, stride = d["rows"], int(d["stride"])
    wr = {}
    for r in rows:
        k = r[6:].reshape(17, 3)
        for w, e in ((9, 7), (10, 8)):
            if k[w, 2] >= 0.3 and k[e, 2] >= 0.3:
                wr.setdefault(int(r[0]), []).append((k[w, :2], max(np.linalg.norm(k[w, :2] - k[e, :2]), 10.0)))
    tl = loadmat(f"data/merl/Labels_MERL_Shopping_Dataset/{f.stem}_label.mat")["tlabs"]
    gt = {i + 1: np.asarray(tl[i][0]).reshape(-1, 2) for i in range(5)}
    lab = np.zeros(int(d["n_frames"]), int)
    for a in range(1, 6):
        for s, e in gt[a]:
            lab[s:e + 1] = a
    cand = [fi for fi in wr if fi < len(lab)]
    pos = [fi for fi in cand if lab[fi] == 4]                         # Inspect Product
    neg = [fi for fi in cand if lab[fi] == 0]                         # no labelled action
    pick = lambda xs: sorted(rng.choice(xs, min(PER_VIDEO, len(xs)), replace=False)) if xs else []
    cap = cv2.VideoCapture(f"data/merl/Videos_MERL_Shopping_Dataset/{f.stem}_crop.mp4")
    for kind, frames in (("pos", pick(pos)), ("neg", pick(neg))):
        for fi in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, im = cap.read()
            if not ok:
                continue
            for s in sets:
                model.set_classes(PROMPTS[s], PE[s])
                b = model.predict(im, conf=min(confs), verbose=False)[0].boxes
                cs = b.xywh[:, :2].cpu().numpy() if b is not None else np.zeros((0, 2))
                cf = b.conf.cpu().numpy() if b is not None else np.zeros(0)
                for c in confs:
                    hit = any(np.linalg.norm(cs[cf >= c] - w, axis=1).min(initial=1e9) <= R * fl for w, fl in wr[int(fi)])
                    counts[(s, c)][kind] += 1
                    counts[(s, c)][kind + "_hit"] += int(hit)
    print(f.stem, flush=True)

for (s, c), v in counts.items():
    res["rows"].append({"prompts": s, "conf": c, **v, "in_hand_recall": v["pos_hit"] / max(v["pos"], 1),
                        "empty_hand_false_rate": v["neg_hit"] / max(v["neg"], 1)})
    print(s, c, round(res["rows"][-1]["in_hand_recall"], 3), round(res["rows"][-1]["empty_hand_false_rate"], 3))
Path(f"results/merl_product_in_hand_{SPLIT}.json").write_text(json.dumps(res, indent=1))
