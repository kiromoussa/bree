"""Cache the labelled frames of real segments as PNGs (full resolution) so detector variants can be compared fast.

Cache lives under <data>/perception/frames/chNN/ , outside every git repo: the frames contain people.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from bree.perception.labels import DATA_DIR, LabelledFrame, segment_path


def cache_path(lf: LabelledFrame, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "perception" / "frames" / f"ch{lf.channel:02d}" / f"{lf.segment}_{lf.t_s:07.2f}.png"


def extract(frames: list[LabelledFrame], data_dir: Path = DATA_DIR) -> int:
    """Write the missing PNGs; one decode pass per segment. Returns how many were written."""
    by_seg: dict[Path, list[LabelledFrame]] = {}
    for lf in frames:
        if not cache_path(lf, data_dir).exists():
            by_seg.setdefault(segment_path(lf, data_dir), []).append(lf)
    n = 0
    for seg, lfs in by_seg.items():
        if not seg.exists():
            continue
        cap = cv2.VideoCapture(str(seg))
        for lf in sorted(lfs, key=lambda x: x.t_s):
            cap.set(cv2.CAP_PROP_POS_MSEC, lf.t_s * 1000.0)
            ok, im = cap.read()
            if not ok:
                continue
            p = cache_path(lf, data_dir)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), im)
            n += 1
        cap.release()
    return n


def load(lf: LabelledFrame, data_dir: Path = DATA_DIR):
    p = cache_path(lf, data_dir)
    return cv2.imread(str(p)) if p.exists() else None
