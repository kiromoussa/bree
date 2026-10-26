"""Where do the cameras stand relative to each other, from the people they both see?

The Manhattan fit (bree.fusion.manhattan) gives each camera's orientation and the lens; the plan frame (x, y metres)
then differs between cameras only by a translation and by the camera height (a scale). Nobody has to stand on marks:
if two cameras see the same person at the same moment, the difference of their two floor points IS the translation
between the cameras. False pairs (two different people) give scattered differences, true pairs the same one every time.

  placed()            floor points of a camera's person boxes with its (provisional) camera at the origin
  vote_translation()  for a camera pair and a range of clock offsets, histogram of the differences over all
                      simultaneous box pairs; the strongest mode is the translation and the offset between the two clocks
  refine()            weighted least squares with the pairs near the mode: translation and relative scale

Every output has the peak's share of all votes and a background estimate (the same count of random pairs), so a
pair of cameras that do not overlap shows up as "no peak". These are REAL-UNLABELLED consistency numbers: they say that
two cameras agree with each other, not that the metres are right (that rests on one anchor, the door width).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bree.calib.camera import Camera
from bree.fusion.plan import floor_xy
from bree.fusion.tracklets import Tracklet

PERSON_H = 1.7


def expected_height_px(cam: Camera, x: float, y: float, h: float = PERSON_H) -> float | None:
    """Height in pixels of a person of `h` metres standing at plan point (x, y) as this camera sees them."""
    px0, d0 = cam.project(np.array([[x, 0.0, y]]))
    px1, d1 = cam.project(np.array([[x, h, y]]))
    if d0[0] <= 1e-6 or d1[0] <= 1e-6:
        return None
    return float(np.hypot(*(px1[0] - px0[0])))


def foot_point(cam: Camera, box, mode: str = "feet", border: float = 4.0, height_tol: tuple[float, float] = (0.7, 1.35)):
    """Plan point (x, y) of a person box, or None if it should not be trusted.
    feet:   the bottom centre of the box on the floor; rejected if the box touches the picture bottom (feet cut off),
            or if its height is not that of a person of 1.5..1.9 m standing there (a cut, occluded or false box).
    centre: the box centre at hip height (about 0.95 m), for a camera that looks nearly straight down."""
    w, h = cam.resolution
    x0, y0, x1, y1 = box
    u = (x0 + x1) / 2
    if mode == "centre":
        v = (y0 + y1) / 2
        d = cam.ray(u, v)
        if d[1] > -1e-3:
            return None
        p = cam.C + d * ((0.95 - cam.C[1]) / d[1])
        return float(p[0]), float(p[2])
    if y1 >= h - border or x0 <= border or x1 >= w - border:
        return None
    p = floor_xy(cam, u, y1)
    if p is None:
        return None
    eh = expected_height_px(cam, *p)
    if eh is None or not (height_tol[0] <= (y1 - y0) / eh <= height_tol[1]):
        return None
    return p


@dataclass
class Placed:
    t: np.ndarray        # (N,) seconds
    xy: np.ndarray       # (N, 2) plan metres with the camera at its provisional position
    conf: np.ndarray
    tid: np.ndarray      # tracklet id of each point (-1 when from raw detections)


def placed_tracklets(cam: Camera, tracklets: list[Tracklet], mode: str = "feet", min_conf: float = 0.5, **kw) -> Placed:
    T, XY, C, I = [], [], [], []
    for tl in tracklets:
        if tl.static:
            continue
        for i in range(len(tl.t)):
            if tl.conf[i] < min_conf:
                continue
            p = foot_point(cam, tl.box[i], mode, **kw)
            if p is not None:
                T.append(tl.t[i]); XY.append(p); C.append(tl.conf[i]); I.append(tl.id)
    return Placed(np.array(T), np.array(XY).reshape(-1, 2), np.array(C), np.array(I, int))


@dataclass
class Peak:
    dt: float            # clock of B = clock of A + dt (seconds)
    shift: np.ndarray    # xy_A - xy_B at the peak (metres): add it to B's plan to land on A's
    votes: int
    background: float    # mean votes in a ring of cells around the peak, per cell
    share: float         # peak votes over all simultaneous pair votes at that offset
    n_pairs: int


def _pairs(a: Placed, b: Placed, dt: float, tol: float):
    """Index pairs (i, j) of points of A and B with |t_a - (t_b - dt)| <= tol."""
    tb = b.t - dt
    order = np.argsort(tb)
    tbs = tb[order]
    lo = np.searchsorted(tbs, a.t - tol)
    hi = np.searchsorted(tbs, a.t + tol)
    ii, jj = [], []
    for i, (l, h) in enumerate(zip(lo, hi)):
        for j in order[l:h]:
            ii.append(i); jj.append(j)
    return np.array(ii, int), np.array(jj, int)


def vote_translation(a: Placed, b: Placed, dts=np.arange(-8.0, 8.01, 0.4), cell: float = 0.5, span: float = 12.0, tol: float = 0.21,
                     max_scale_err: float = 0.0) -> tuple[Peak | None, list[tuple[float, int]]]:
    """Best (clock offset, translation) of B relative to A. For each offset dt the differences xy_A - xy_B of all
    simultaneous pairs go into a 2D histogram of `cell` metre cells over +-span; the (dt, cell) with the most votes wins
    (3 by 3 cells summed, so a mode straddling a cell edge is not lost). Returns the peak and the best votes per dt."""
    best, scan = None, []
    n = int(span / cell) * 2 + 1
    for dt in dts:
        ii, jj = _pairs(a, b, float(dt), tol)
        if len(ii) == 0:
            scan.append((float(dt), 0))
            continue
        d = a.xy[ii] - b.xy[jj]
        idx = np.floor(d / cell).astype(int) + n // 2
        ok = (idx >= 0).all(1) & (idx < n).all(1)
        H = np.zeros((n, n))
        np.add.at(H, (idx[ok, 0], idx[ok, 1]), 1)
        from scipy.ndimage import uniform_filter
        S = uniform_filter(H, 3, mode="constant") * 9
        k = np.unravel_index(int(np.argmax(S)), S.shape)
        v = int(S[k])
        scan.append((float(dt), v))
        if best is None or v > best[0]:
            m = (np.abs(d[:, 0] - ((k[0] - n // 2 + 0.5) * cell)) < 1.5 * cell) & (np.abs(d[:, 1] - ((k[1] - n // 2 + 0.5) * cell)) < 1.5 * cell)
            shift = d[m].mean(0) if m.any() else np.array([(k[0] - n // 2 + 0.5) * cell, (k[1] - n // 2 + 0.5) * cell])
            ring = (S.sum() - S[max(k[0] - 4, 0):k[0] + 5, max(k[1] - 4, 0):k[1] + 5].sum()) / max(S.size - 81, 1)
            best = (v, Peak(float(dt), shift, v, float(ring), v / max(int(ok.sum()), 1), int(ok.sum())))
    return (best[1] if best else None), scan
