"""Learn static ghost detections from a few minutes of footage.

A ghost is a low-confidence person box that sits in the same place frame after frame: a lit display, a poster, a
reflection in cooler glass. Real baseline runs on this store showed one per camera on several channels (a lottery
display above the counter, a candy rack, a cooler-door reflection), each producing dozens of tracks and fake 'enter'
events. We cluster box centres over sampled frames and mark clusters that recur often AND never get confident.

A person who really stands still (the counter clerk at the game machine) also recurs, but the detector is confident
about them, so the confidence ceiling keeps them. Ghost regions are proposals: `calibrate` writes them into the
profile with their statistics so a human can look at them, and nothing here proves a region is a ghost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GhostCluster:
    box: list[float]          # median box, normalised x0 y0 x1 y1
    frac_frames: float        # share of sampled frames with a detection here
    mean_conf: float
    max_conf: float
    n: int


def find_ghosts(dets: list[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int], radius: float = 0.03,
                min_frac: float = 0.5, max_mean_conf: float = 0.5, max_peak_conf: float = 0.65,
                min_frames: int = 12) -> list[GhostCluster]:
    """dets: per sampled frame (boxes xyxy px, confs). Returns the ghost clusters, most frequent first."""
    h, w = shape[:2]
    pts, confs, boxes, frame_idx = [], [], [], []
    for fi, (b, c) in enumerate(dets):
        for bb, cc in zip(b, c):
            pts.append([(bb[0] + bb[2]) / 2 / w, (bb[1] + bb[3]) / 2 / h])
            boxes.append([bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h])
            confs.append(cc); frame_idx.append(fi)
    if not pts:
        return []
    pts, confs, boxes, frame_idx = map(np.array, (pts, confs, boxes, frame_idx))
    n_frames = max(len(dets), 1)
    unassigned = np.ones(len(pts), bool)
    out: list[GhostCluster] = []
    for i in np.argsort(-confs):   # strongest first seeds the cluster
        if not unassigned[i]:
            continue
        member = unassigned & (np.hypot(*(pts - pts[i]).T) <= radius)
        unassigned &= ~member
        frames_hit = len(set(frame_idx[member].tolist()))
        if frames_hit < min_frames or frames_hit / n_frames < min_frac:
            continue
        mc, xc = float(confs[member].mean()), float(confs[member].max())
        if mc > max_mean_conf or xc > max_peak_conf:
            continue
        out.append(GhostCluster(np.median(boxes[member], 0).round(3).tolist(), round(frames_hit / n_frames, 3),
                                round(mc, 3), round(xc, 3), int(member.sum())))
    return sorted(out, key=lambda g: -g.frac_frames)


def find_fixtures(dets: list[tuple[np.ndarray, np.ndarray]], shape: tuple[int, int], radius: float = 0.02,
                  min_frac: float = 0.7, min_frames: int = 12) -> list[GhostCluster]:
    """Static product-class detections (a card terminal read as 'cell phone' in 99% of frames): same clustering as
    ghosts but at any confidence. They are fixtures, not picks. A bottle left on the counter for the whole clip is
    a fixture too: it only matters again if it moves, which is then a new detection elsewhere."""
    return find_ghosts(dets, shape, radius, min_frac, max_mean_conf=1.1, max_peak_conf=1.1, min_frames=min_frames)


def ghost_polygon(g: GhostCluster, pad: float = 0.5) -> list[list[float]]:
    """Rectangle around the ghost box, grown by `pad` of its size, as a normalised polygon."""
    x0, y0, x1, y1 = g.box
    pw, ph = (x1 - x0) * pad / 2, (y1 - y0) * pad / 2
    x0, y0, x1, y1 = max(0, x0 - pw), max(0, y0 - ph), min(1, x1 + pw), min(1, y1 + ph)
    return [[round(x0, 3), round(y0, 3)], [round(x1, 3), round(y0, 3)], [round(x1, 3), round(y1, 3)], [round(x0, 3), round(y1, 3)]]
