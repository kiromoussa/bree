#!/usr/bin/env python3
"""Measure how many faces survive head pixelation on REAL footage (REAL-UNLABELLED counts from detectors; the person
frames are then checked by eye, see docs/ultra/privacy_eval.md). Everything is written under the data dir, never to git.

    python scripts/trust/leak_eval.py --per-seg 6 --out ~/Projects/bree-data/trust/leak_eval

Compares the old routine (pose-keypoint heads only, bree.alerts.annotate.blur_heads) with bree.trust.privacy.Guard.
The audit uses the same face detector family at lower threshold and more scales, plus the pose model on the output.
"""
import argparse, glob, json, os, random
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.annotate import blur_heads
from bree.trust.privacy import Guard, scan_faces, still_guard
from bree.trust.redact import make_person_backend, persons_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Projects/bree-data/real/trumanrd"))
    ap.add_argument("--channels", default="7,29,12,14,15,10")
    ap.add_argument("--per-seg", type=int, default=6)
    ap.add_argument("--out", default=str(Path.home() / "Projects/bree-data/trust/leak_eval"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    rng = random.Random(a.seed)
    backend = make_person_backend()
    rows = []
    for ch in a.channels.split(","):
        for seg in sorted(glob.glob(f"{a.data}/ch{int(ch):02d}/*.mp4"))[:6]:
            cap = cv2.VideoCapture(seg)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for k in range(a.per_seg):
                idx = int((k + rng.random()) * n / a.per_seg)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                if not ok:
                    continue
                persons = persons_of(backend, fr)
                raw = scan_faces(fr, score=0.5, widths=(1280, 2560))
                old = blur_heads(fr.copy(), persons)
                new = still_guard().protect(fr.copy(), persons)
                row = {"ch": int(ch), "seg": Path(seg).name, "idx": idx, "persons": len(persons),
                       "raw_hits": len(raw), "old_hits": len(scan_faces(old, score=0.5, widths=(1280, 2560))),
                       "new_hits": len(scan_faces(new, score=0.5, widths=(1280, 2560))),
                       "new_hits_lowthr": len(scan_faces(new, score=0.3)),
                       "new_pose_heads": sum(1 for p in persons_of(backend, new)
                                             if p.keypoints is not None and (p.keypoints[:5, 2] >= 0.5).sum() >= 3)}
                rows.append(row)
                name = f"ch{int(ch):02d}_{Path(seg).stem}_{idx:05d}"
                if persons:
                    cv2.imwrite(str(out / "frames" / f"{name}_old.jpg"), cv2.resize(old, (1280, 720)))
                    cv2.imwrite(str(out / "frames" / f"{name}_new.jpg"), cv2.resize(new, (1280, 720)))
                print(row, flush=True)
    keys = ["persons", "raw_hits", "old_hits", "new_hits", "new_hits_lowthr", "new_pose_heads"]
    tot = {k: sum(r[k] for r in rows) for k in keys}
    tot["frames"] = len(rows)
    tot["frames_with_persons"] = sum(1 for r in rows if r["persons"])
    for k in ("raw_hits", "old_hits", "new_hits"):
        tot[f"frames_{k}"] = sum(1 for r in rows if r[k])
    (out / "result.json").write_text(json.dumps({"rows": rows, "totals": tot}, indent=1))
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
