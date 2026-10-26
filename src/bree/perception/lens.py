"""Wide-angle (barrel) distortion, handled on points rather than whole images.

Warping a 2560x1440 frame costs time and blurs small people, so BREE keeps the detector on the raw frame and maps
only the points that need geometry: zone polygons, floor points, box foot points. One-parameter division model
(Fitzgibbon 2001), radius normalised by the half diagonal, centred on the image centre by default:

    undistorted = centre + (distorted - centre) / (1 + k1 * r^2)

k1 < 0 is barrel distortion (straight lines bow outwards), which is what these fisheye-ish store cameras show.
`fit_k1` estimates k1 from points the user marks along edges that are straight in the room (shelf rails, floor
tile lines, door frames). It is a calibration aid: k1 from one marked line is rough, and no accuracy of the result
is measured on real footage yet (no ground-truth geometry exists for this store).
"""
from __future__ import annotations

import numpy as np


def _norm(shape, centre):
    h, w = shape[:2]
    c = np.array(centre if centre is not None else (w / 2, h / 2), float)
    return c, float(np.hypot(w, h) / 2)


def undistort_points(pts, k1: float, shape, centre=None) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    c, s = _norm(shape, centre)
    d = (pts - c) / s
    r2 = (d ** 2).sum(1, keepdims=True)
    return c + d / (1 + k1 * r2) * s


def distort_points(pts, k1: float, shape, centre=None, iters: int = 30) -> np.ndarray:
    """Inverse of undistort_points (fixed point iteration; fine for |k1| up to about 0.4)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    c, s = _norm(shape, centre)
    u = (pts - c) / s
    d = u.copy()
    for _ in range(iters):
        r2 = (d ** 2).sum(1, keepdims=True)
        d = u * (1 + k1 * r2)
    return c + d * s


def line_residual(lines, k1: float, shape) -> float:
    """Mean distance (pixels) of undistorted points from the best straight line through each marked polyline."""
    tot, n = 0.0, 0
    for line in lines:
        p = undistort_points(line, k1, shape)
        if len(p) < 3:
            continue
        q = p - p.mean(0)
        _, _, vt = np.linalg.svd(q, full_matrices=False)
        off = q @ vt[1]
        tot += float(np.abs(off).sum()); n += len(p)
    return tot / max(n, 1)


def fit_k1(lines, shape, lo: float = -0.6, hi: float = 0.2) -> tuple[float, float]:
    """(k1, residual_px) minimising line_residual over [lo, hi]; coarse grid then golden section."""
    grid = np.linspace(lo, hi, 33)
    best = grid[int(np.argmin([line_residual(lines, k, shape) for k in grid]))]
    a, b = max(lo, best - 0.025), min(hi, best + 0.025)
    g = (5 ** 0.5 - 1) / 2
    for _ in range(30):
        c1, c2 = b - g * (b - a), a + g * (b - a)
        if line_residual(lines, c1, shape) < line_residual(lines, c2, shape):
            b = c2
        else:
            a = c1
    k = (a + b) / 2
    return float(k), float(line_residual(lines, k, shape))


def undistort_polygon(poly_px, k1: float, shape) -> list[list[float]]:
    """A zone polygon drawn on the distorted frame, mapped to undistorted coordinates (edges re-straightened)."""
    return undistort_points(poly_px, k1, shape).round(1).tolist()
