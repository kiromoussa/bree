"""INJECTED recall of the shelf-state monitor: a synthetic change is pasted into REAL cached frames (never into git).

Numbers from this script are INJECTED (synthetic change in real pixels), not REAL-GROUND-TRUTH: pasted items have no
real shading, and real restocks may look different. Frames and person boxes come from the cache a
`python -m bree.shelfstate run` or the real-footage sampler wrote (<cache>/<cam>_<segment>.npz / .boxes.json).

python scripts/shelfstate_inject_eval.py --cache ~/Projects/bree-data/shelf/cache --camera ch14 --segment 20261025_182857 \
    --regions configs/shelfstate/trumanrd_ch14.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bree.shelfstate.model import ShelfStateConfig, StateMonitor
from bree.shelfstate.regions import load_regions

T_INJ = 90   # seconds into the segment the change appears (reference is warm by then)


def free_spots(regions, boxes, shape, kinds, n, rng, size=(0.05, 0.05)):
    """Random normalised (x0, y0, x1, y1) boxes inside a region of `kinds` that no person box touched during the run."""
    h, w = shape
    bm = np.zeros((h, w), bool)
    for b in boxes:
        for x0, y0, x1, y1, *_ in np.asarray(b).reshape(-1, 5):
            bm[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = True
    bm = cv2.dilate(bm.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ok = np.zeros((h, w), bool)
    for r in regions.regions:
        if r.kind in kinds:
            ok |= r.mask((h, w))
    ok &= ~bm
    ok = cv2.erode(ok.astype(np.uint8), np.ones((int(size[1] * h) + 4, int(size[0] * w) + 4), np.uint8)).astype(bool)
    ys, xs = np.nonzero(ok)
    out = []
    for i in rng.permutation(len(ys))[: n * 50]:
        x, y = xs[i] / w, ys[i] / h
        if all(abs(x - o[0]) > 0.12 or abs(y - o[1]) > 0.12 for o in out):
            out.append((x, y, x + size[0], y + size[1]))
        if len(out) == n:
            break
    return out


def inject(frame, kind, box, rng):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
    f = frame.copy()
    if kind == "remove":        # the facings are gone: the shelf behind (a flat grey-brown) shows
        f[y0:y1, x0:x1] = np.median(frame[y0:y1, x0:x1].reshape(-1, 3), 0).astype(np.uint8) // 2 + 60
    elif kind == "item":        # an object turns up: a patch of a distant part of the same picture
        sx, sy = int(rng.integers(0, w - (x1 - x0))), int(rng.integers(0, h - (y1 - y0)))
        f[y0:y1, x0:x1] = frame[sy:sy + y1 - y0, sx:sx + x1 - x0]
    elif kind == "spill":       # a dark glossy puddle
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2 + 6, (y1 - y0) // 2 + 4), 0, 0, 360, 1, -1)
        f[m > 0] = (f[m > 0] * 0.55 + np.array([60, 40, 20]) * 0.45).astype(np.uint8)
    return f


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--regions")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    z = np.load(Path(a.cache) / f"{a.camera}_{a.segment}.npz")
    boxes = json.loads((Path(a.cache) / f"{a.camera}_{a.segment}.boxes.json").read_text())
    ts, frames = z["t"], z["frames"]
    regions = load_regions(a.regions)
    rng = np.random.default_rng(a.seed)
    res = {}
    for kind, kinds in (("remove", ("shelf", "cooler_door")), ("item", ("shelf", "cooler_door")), ("spill", ("floor",))):
        spots = free_spots(regions, boxes, frames.shape[1:3], kinds, a.n, rng)
        for gain, noise in ((1.0, 0.0), (1.0, 5.0), (1.3, 5.0), (0.7, 5.0)):
            hit = 0
            for sp in spots:
                m = StateMonitor(a.camera, regions, ShelfStateConfig())
                evs = []
                for t, f, b in zip(ts, frames, boxes):
                    g = inject(f, kind, sp, rng) if t >= T_INJ else f
                    g = np.clip(g.astype(np.float32) * gain + rng.normal(0, noise, g.shape), 0, 255).astype(np.uint8)
                    evs += m.update(float(t), g, np.array(b).reshape(-1, 5))
                hit += any(e.t >= T_INJ and e.box[0] < sp[2] and e.box[2] > sp[0] and e.box[1] < sp[3] and e.box[3] > sp[1] for e in evs)
            res[f"{kind} gain={gain} noise={noise}"] = f"{hit}/{len(spots)}"
            print(a.camera, kind, f"gain={gain} noise={noise}", f"{hit}/{len(spots)}", flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
