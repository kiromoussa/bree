#!/usr/bin/env python3
"""Head-cover check by eye: for sampled real frames, tile each detected person's upper body after the OLD routine
(left) and after bree.trust.privacy (right), so a reviewer can mark each head 'covered' or 'visible'. Output goes under the
data dir only (it shows people). Frames are sampled with a seed that differs from leak_eval.py.

    python scripts/trust/leak_crops.py --frames 70 --out ~/Projects/bree-data/trust/leak_crops
"""
import argparse, glob, json, os, random
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.annotate import blur_heads
from bree.trust.privacy import still_guard
from bree.trust.redact import make_person_backend, persons_of

T = 220   # tile size


def tile(img, p, label):
    x1, y1, x2, y2 = map(int, p.bbox)
    bh = y2 - y1
    y2c = int(y1 + max(0.45 * bh, 40))
    pad = int(0.15 * (x2 - x1)) + 10
    crop = img[max(0, y1 - 20):y2c, max(0, x1 - pad):x2 + pad]
    if crop.size == 0:
        crop = np.zeros((10, 10, 3), np.uint8)
    s = T / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
    out = np.zeros((T + 18, T, 3), np.uint8)
    out[18:18 + crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(out, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Projects/bree-data/real/trumanrd"))
    ap.add_argument("--channels", default="7,29,12,14,15,10")
    ap.add_argument("--frames", type=int, default=70)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path.home() / "Projects/bree-data/trust/leak_crops"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    rng = random.Random(a.seed)
    segs = [(int(c), s) for c in a.channels.split(",") for s in sorted(glob.glob(f"{a.data}/ch{int(c):02d}/*.mp4"))]
    backend = make_person_backend()
    tiles, index = [], []
    tries = 0
    while len({i["frame"] for i in index}) < a.frames and tries < a.frames * 6:
        tries += 1
        ch, seg = rng.choice(segs)
        cap = cv2.VideoCapture(seg)
        idx = rng.randrange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        persons = persons_of(backend, fr)
        if not persons:
            continue
        old = blur_heads(fr.copy(), persons)
        new = still_guard().protect(fr.copy(), persons)
        fid = f"ch{ch:02d}_{Path(seg).stem}_{idx}"
        for k, p in enumerate(persons):
            n = len(tiles) // 2 + 1
            tiles += [tile(old, p, f"#{n} old"), tile(new, p, f"#{n} new")]
            index.append({"n": n, "frame": fid, "ch": ch, "bbox_h": int(p.bbox[3] - p.bbox[1]), "conf": round(p.conf, 2)})
    per_sheet, cols = 18, 6      # 18 persons (36 tiles) per sheet: 6 pairs per row would be 12 tiles wide
    for s in range(0, len(tiles) // 2, per_sheet):
        rows = []
        chunk = tiles[2 * s:2 * (s + per_sheet)]
        for r in range(0, len(chunk), 2 * cols):
            row = chunk[r:r + 2 * cols]
            row += [np.zeros_like(tiles[0])] * (2 * cols - len(row))
            rows.append(np.hstack(row))
        cv2.imwrite(str(out / f"sheet_{s // per_sheet:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(len(index), "persons in", len({i['frame'] for i in index}), "frames")


if __name__ == "__main__":
    main()
