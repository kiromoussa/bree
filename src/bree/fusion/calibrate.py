"""Floor calibration of a real camera from its empty-store picture, proposed automatically, corrected by a person.

A fixed ceiling camera sees a plane (the floor) with straight, regular structure: plank lines, accent tiles, door
mats, tape. From those a floor homography can be PROPOSED without anyone standing in the store with a tape:

  floor_lines()   straight edges on the floor part of the picture (LSD), most of them along two perpendicular
                  directions (the plank and tile grid)
  vanishing_points()  the two dominant directions found with a small RANSAC (angle-consistency, not pixel distance)
  rectify()       two perpendicular vanishing points and a principal point at the picture centre give the focal
                  length (f^2 = -(v1-c).(v2-c)), the camera tilt and so the floor homography up to ONE scale (the camera
                  height) and a rotation about the vertical: the "rectified" top-down view in arbitrary units
  accent_tiles()  the darker (or lighter) rectangles of the floor, as quadrilaterals; once rectified they are
                  rectangles of one known size, which fixes the scale, and they are landmarks for registering cameras
  fit_points()    the human step: floor points marked on a still with their plan coordinates in metres
                  (marks_3d-style rows [u, v, X_m, Y_m]) give the homography directly, with leave-one-out error

Everything here is a PROPOSAL with an error estimate: the result is a JSON a person can read, correct (move a
point, add a mark, change the tile size) and re-fit, and `overlay()` draws the result on the picture so a wrong one
is seen at a glance. Nothing is calibrated to ground truth; the cross-camera check (bree.fusion.registry,
people seen by two cameras at one moment) is the only independent evidence. Pinhole only: the radial term of a wide
lens is not estimated (the lines near the corners bend); use the middle of the picture.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from bree.track.multicam import homography


# ---------------------------------------------------------------- lines and vanishing points

def floor_lines(image: np.ndarray, roi: np.ndarray | None = None, min_len: float = 40.0) -> np.ndarray:
    """Straight segments (N, 4) [x0, y0, x1, y1] inside the floor region of interest (a polygon in pixels, default the
    lower 70 percent of the picture) from the line segment detector, after a light blur that suppresses plank texture."""
    import cv2
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    mask = np.zeros(g.shape, np.uint8)
    if roi is None:
        mask[int(g.shape[0] * 0.3):] = 255
    else:
        cv2.fillPoly(mask, [np.asarray(roi, np.int32)], 255)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(g)[0]
    if seg is None:
        return np.zeros((0, 4))
    seg = seg.reshape(-1, 4)
    keep = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= min_len
    seg = seg[keep]
    mid = ((seg[:, :2] + seg[:, 2:]) / 2).astype(int)
    inside = mask[np.clip(mid[:, 1], 0, g.shape[0] - 1), np.clip(mid[:, 0], 0, g.shape[1] - 1)] > 0
    return seg[inside]


def _homog_lines(seg: np.ndarray) -> np.ndarray:
    p0 = np.c_[seg[:, :2], np.ones(len(seg))]
    p1 = np.c_[seg[:, 2:], np.ones(len(seg))]
    return np.cross(p0, p1)


def _vp_support(seg: np.ndarray, v: np.ndarray, max_deg: float) -> np.ndarray:
    """Which segments point at vanishing point v (homogeneous): the angle between the segment and the line from its
    midpoint to v is small. Works for finite and infinite vanishing points."""
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    d = seg[:, 2:] - seg[:, :2]
    if abs(v[2]) < 1e-9:
        to = np.tile(v[:2], (len(seg), 1))
    else:
        to = v[:2] / v[2] - mid
    cosang = np.abs((d * to).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to, axis=1) + 1e-9)
    return cosang >= math.cos(math.radians(max_deg))


def vanishing_points(seg: np.ndarray, n: int = 2, iters: int = 3000, max_deg: float = 1.5, seed: int = 0):
    """The `n` dominant vanishing points of the segments: [(v homogeneous, support share of all segments)], most
    supported first. RANSAC over pairs of segments, weighted by segment length; each point removes its segments."""
    rng = np.random.default_rng(seed)
    lines = _homog_lines(seg)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    alive = np.ones(len(seg), bool)
    out = []
    for _ in range(n):
        idx = np.flatnonzero(alive)
        if len(idx) < 4:
            break
        best, best_score = None, 0.0
        p = length[idx] / length[idx].sum()
        for _ in range(iters):
            i, j = rng.choice(idx, 2, replace=False, p=p)
            v = np.cross(lines[i], lines[j])
            if np.linalg.norm(v) < 1e-9:
                continue
            v = v / np.linalg.norm(v)
            sup = _vp_support(seg[idx], v, max_deg)
            score = length[idx][sup].sum()
            if score > best_score:
                best, best_score, best_sup = v, score, idx[sup]
        if best is None:
            break
        # refine: least squares intersection of the supporting lines
        L = lines[best_sup] / np.linalg.norm(lines[best_sup][:, :2], axis=1, keepdims=True)
        w = length[best_sup]
        v = np.linalg.svd(L * w[:, None])[2][-1]
        out.append((v, float(length[best_sup].sum() / length.sum())))
        alive[best_sup] = False
        alive &= ~_vp_support(seg, v, max_deg)
    return out


def focal_from_vps(v1: np.ndarray, v2: np.ndarray, c: tuple[float, float]) -> float | None:
    """Focal length (pixels) from two vanishing points of perpendicular directions and the principal point."""
    p1 = v1[:2] / v1[2] - np.asarray(c)
    p2 = v2[:2] / v2[2] - np.asarray(c)
    f2 = -float(p1 @ p2)
    return math.sqrt(f2) if f2 > 0 else None


def rectify(f: float, c: tuple[float, float], v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """3x3 H with (a, b, 1) ~ H @ (u, v, 1): floor coordinates in units of the camera height, a along the direction
    of v1 and b along v2, right handed seen from above. Returns None-like (raises) if the geometry is degenerate."""
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    d1, d2 = Ki @ v1, Ki @ v2
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    n = np.cross(d1, d2)
    n = n / np.linalg.norm(n)
    if n[1] < 0:           # the floor normal points from the camera towards the floor: down, +y in the camera frame
        n = -n
    up = -n
    d1 = d1 - (d1 @ n) * n
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.cross(up, d1)         # (d1, d2, up) right handed: seen from above, d2 is d1 turned a quarter counter-clockwise
    H = np.vstack([d1 @ Ki, d2 @ Ki, n @ Ki])
    return H / H[2, 2] if abs(H[2, 2]) > 1e-9 else H


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(np.atleast_2d(pts)))] @ H.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- accent tiles

def accent_tiles(image: np.ndarray, roi: np.ndarray | None = None, delta: float = 14.0, min_area: float = 1500.0,
                 max_area_share: float = 0.08) -> list[np.ndarray]:
    """Floor tiles that differ from the plank colour, as convex quadrilaterals (4, 2) in pixels: the pixels of the
    floor region whose smoothed grey level is more than `delta` away from the median of the region (darker and
    lighter), connected components with four corners (cv2.approxPolyDP), convex, area between min_area and a share
    of the picture. The floor region defaults to the lower 70 percent and is meant to be corrected (roi)."""
    import cv2
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    s = cv2.GaussianBlur(g, (0, 0), 3)
    mask = np.zeros(g.shape, np.uint8)
    if roi is None:
        mask[int(g.shape[0] * 0.3):] = 255
    else:
        cv2.fillPoly(mask, [np.asarray(roi, np.int32)], 255)
    med = float(np.median(s[mask > 0]))
    quads = []
    for sign in (-1, 1):
        m = ((sign * (s.astype(np.float32) - med) > delta) & (mask > 0)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a = cv2.contourArea(c)
            if a < min_area or a > max_area_share * g.size:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            ap = cv2.approxPolyDP(hull, 0.035 * peri, True)
            if len(ap) == 4 and cv2.contourArea(ap) > 0.8 * a:
                quads.append(ap.reshape(4, 2).astype(float))
    return quads


def quad_rect_error(H: np.ndarray, quad: np.ndarray) -> float:
    """How far a quadrilateral is from a rectangle once put on the floor (mean angle error at the corners, degrees)."""
    p = apply_h(H, quad)
    err = []
    for i in range(4):
        a, b, c = p[i - 1], p[i], p[(i + 1) % 4]
        u, v = a - b, c - b
        ang = math.degrees(math.acos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1)))
        err.append(abs(ang - 90))
    return float(np.mean(err))


# ---------------------------------------------------------------- the human step

@dataclass
class Fit:
    H: np.ndarray                       # (u, v) -> (X_m, Y_m)
    rms_m: float
    loo_m: float | None                 # leave-one-out error: mark each point out, fit the rest, measure the miss
    per_point_m: list[float] = field(default_factory=list)


def fit_points(marks, min_points: int = 4) -> Fit:
    """Floor homography from marks [[u, v, X_m, Y_m], ...] (more than 4 are least squares). The error is in metres on
    the floor. A mark more than 3 times the median error is the first to check."""
    m = np.asarray([r[:4] for r in marks], float)
    if len(m) < min_points:
        raise ValueError(f"{len(m)} marks, need {min_points}")
    H = homography(m[:, :2], m[:, 2:4])
    err = np.linalg.norm(apply_h(H, m[:, :2]) - m[:, 2:4], axis=1)
    loo = None
    if len(m) > 4:
        e = []
        for i in range(len(m)):
            keep = np.arange(len(m)) != i
            Hi = homography(m[keep, :2], m[keep, 2:4])
            e.append(np.linalg.norm(apply_h(Hi, m[i:i + 1, :2])[0] - m[i, 2:4]))
        loo = float(np.sqrt(np.mean(np.square(e))))
    return Fit(H, float(np.sqrt(np.mean(err ** 2))), loo, [round(float(v), 3) for v in err])


def similarity(src, dst) -> np.ndarray:
    """3x3 similarity (rotation, one scale, translation, no reflection) taking src (N, 2) to dst (N, 2), least squares."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    s = (S * [1, d]).sum() / (a ** 2).sum()
    t = md - s * R @ ms
    return np.array([[s * R[0, 0], s * R[0, 1], t[0]], [s * R[1, 0], s * R[1, 1], t[1]], [0, 0, 1.0]])


def overlay(image: np.ndarray, H_floor_to_img: np.ndarray, x_range, y_range, step_m: float = 1.0,
            marks=None, regions: dict | None = None) -> np.ndarray:
    """Draw the metre grid (and the marks, and named plan regions) of a calibration on the picture. Lines are
    cyan every `step_m` with the metre value written at the grid ends; marks are red circles with their label."""
    import cv2
    out = image.copy()
    h, w = out.shape[:2]

    def pt(x, y):
        p = H_floor_to_img @ np.array([x, y, 1.0])
        return None if p[2] <= 1e-9 else (int(round(p[0] / p[2])), int(round(p[1] / p[2])))

    def line(a, b, col, th=2):
        A, B = pt(*a), pt(*b)
        if A and B and max(abs(A[0]), abs(A[1]), abs(B[0]), abs(B[1])) < 8 * max(w, h):
            cv2.line(out, A, B, col, th, cv2.LINE_AA)
    for x in np.arange(math.floor(x_range[0]), x_range[1] + 1e-6, step_m):
        line((x, y_range[0]), (x, y_range[1]), (255, 255, 0))
        p = pt(x, y_range[0])
        if p and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.putText(out, f"x{x:g}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    for y in np.arange(math.floor(y_range[0]), y_range[1] + 1e-6, step_m):
        line((x_range[0], y), (x_range[1], y), (0, 255, 255))
        p = pt(x_range[0], y)
        if p and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.putText(out, f"y{y:g}", p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    for name, poly in (regions or {}).items():
        P = [pt(*q) for q in poly]
        if all(P):
            cv2.polylines(out, [np.array(P, np.int32)], True, (0, 200, 0), 3, cv2.LINE_AA)
            cv2.putText(out, name, P[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2, cv2.LINE_AA)
    for r in marks or []:
        cv2.circle(out, (int(r[0]), int(r[1])), 10, (0, 0, 255), 3, cv2.LINE_AA)
        if len(r) > 4:
            cv2.putText(out, str(r[4]), (int(r[0]) + 12, int(r[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def propose(image: np.ndarray, roi=None) -> dict:
    """Everything the automatic step can say about one camera: vanishing points, focal length, tilt, accent tiles
    with their rectangle error under the proposed rectification. JSON-able."""
    h, w = image.shape[:2]
    c = (w / 2, h / 2)
    seg = floor_lines(image, roi)
    vps = vanishing_points(seg) if len(seg) >= 10 else []
    out = {"image_size": [w, h], "n_segments": int(len(seg)), "vps": [{"v": [round(float(x), 4) for x in v], "share": round(s, 3)} for v, s in vps]}
    if len(vps) >= 2:
        f = focal_from_vps(vps[0][0], vps[1][0], c)
        out["f_px"] = None if f is None else round(f, 1)
        if f:
            out["hfov_deg"] = round(math.degrees(2 * math.atan(w / 2 / f)), 1)
            H = rectify(f, c, vps[0][0], vps[1][0])
            out["H_rect"] = H.tolist()
            out["tiles"] = []
            for q in accent_tiles(image, roi):
                out["tiles"].append({"quad": q.round(1).tolist(), "rect_err_deg": round(quad_rect_error(H, q), 2)})
    return out
