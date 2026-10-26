"""Camera orientation and lens from one still of a store, with no marks: the Manhattan frame of its straight lines.

A store is built from three perpendicular directions (up, along the building, across it): shelf edges, door and
window frames, counter fronts, tile and plank lines, standing people. A straight line in space seen by a pinhole
camera is a straight line in the picture; its two end rays span a plane whose normal n is perpendicular to the line
direction d. If the three building directions are the axes (a1, a2, a3) of a rotation R, every such line has
|n . a_k| = 0 for one k. We look for the R (and the lens: focal length f, one radial term k, principal point at
the picture centre) that put the most line length on the three axes:

  lines -> undistorted rays (f, k) -> normals -> Manhattan frame by RANSAC (axis 1 from two lines, axis 2 from the
  lines perpendicular to axis 1, axis 3 their cross product) -> score = total length within `tol_deg` of an axis

and `fit_lens()` scans (f, k) for the best score. The frame gives the camera tilt, roll and heading relative to the
building (three of the six pose numbers) and the floor plane direction; the camera height and the 2D position stay
unknown and come from a metric anchor (bree.fusion.plan) and from people seen by two cameras (bree.fusion.registry).

All the cameras of one site are usually one lens model: `fit_lens(..., shared)` can pool them. A bad frame is possible
(a picture dominated by curved shelf edges or a person); the score and the share of explained line length are
reported so a person can reject it, and `overlay_axes()` draws the three families to look at.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    R: np.ndarray           # rows: the three building axes in the camera frame (x right, y down, z forward)
    score: float            # explained line length, share of all line length (0..1)
    f: float                # focal length, pixels of the image the lines came from
    k: float                # radial term (bree.calib.camera division model)
    vertical: int           # which row of R is the vertical direction
    n_lines: int = 0


def vertical_axis(axes: np.ndarray) -> int:
    """Which row of a Manhattan frame is the vertical. Turn every axis to point up in the picture (negative camera y);
    a camera looking down at pitch t has up = (0, -cos t, -sin t) and the horizontal forward axis
    (0, -sin t cos t, cos^2 t), so the vertical is the axis with the most negative z (it points back out of the picture plane)."""
    a = np.asarray(axes, float)
    a = a * np.where(a[:, 1] < 0, 1.0, -1.0)[:, None]
    return int(np.argmin(a[:, 2]))


def line_segments(image: np.ndarray, width: int = 1280, min_len: float = 25.0) -> np.ndarray:
    """Straight segments (N, 4) in pixels of the picture shrunk to `width`: LSD on a lightly blurred grey image."""
    import cv2
    h, w = image.shape[:2]
    s = width / w
    g = cv2.cvtColor(cv2.resize(image, (width, int(h * s)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    seg = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    if seg is None:
        return np.zeros((0, 4))
    seg = seg.reshape(-1, 4)
    return seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= min_len]


def rays(pts: np.ndarray, f: float, k: float, cx: float, cy: float) -> np.ndarray:
    """Unit rays of pixels under the division model of bree.calib.camera.Camera.ray."""
    x, y = (pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f
    if k:
        s = 2.0 / (1.0 + np.sqrt(np.maximum(1.0 - 4.0 * k * (x * x + y * y), 0.0)))
        x, y = x * s, y * s
    r = np.stack([x, y, np.ones_like(x)], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def plane_normals(seg: np.ndarray, f: float, k: float, cx: float, cy: float) -> np.ndarray:
    n = np.cross(rays(seg[:, :2], f, k, cx, cy), rays(seg[:, 2:], f, k, cx, cy))
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def manhattan(normals: np.ndarray, length: np.ndarray, tol_deg: float = 1.2, iters: int = 400, seed: int = 0):
    """(axes (3, 3), explained length share) from line plane normals and lengths, RANSAC then a refinement."""
    rng = np.random.default_rng(seed)
    tol = math.sin(math.radians(tol_deg))
    n = len(normals)
    p = length / length.sum()
    best = (0.0, None)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False, p=p)
        a1 = np.cross(normals[i], normals[j])
        if np.linalg.norm(a1) < 0.05:
            continue
        a1 = a1 / np.linalg.norm(a1)
        sup1 = np.abs(normals @ a1) < tol
        # axis 2: a direction perpendicular to a1 that many lines are perpendicular to. Lines not along a1 (but whose
        # direction is perpendicular to it) have normals with a component along a1; sample one such line's direction.
        cand = np.flatnonzero(~sup1)
        if len(cand) < 2:
            continue
        for _ in range(6):
            m = rng.choice(cand)
            # a line direction d perpendicular to normal n_m, and perpendicular to a1: d = n_m x a1 (if n_m is not along a1)
            a2 = np.cross(normals[m], a1)
            if np.linalg.norm(a2) < 0.05:
                continue
            a2 = a2 / np.linalg.norm(a2)
            a3 = np.cross(a1, a2)
            sup = sup1 | (np.abs(normals @ a2) < tol) | (np.abs(normals @ a3) < tol)
            score = float(length[sup].sum())
            if score > best[0]:
                best = (score, np.stack([a1, a2, a3]))
    if best[1] is None:
        return None, 0.0
    A = best[1]
    for _ in range(3):      # refine: each axis is the direction nearly perpendicular to its lines
        sup = np.stack([np.abs(normals @ a) < tol * 2 for a in A], 1)
        new = []
        for k in range(3):
            s = sup[:, k]
            if s.sum() < 3:
                new.append(A[k])
                continue
            w = length[s]
            u = np.linalg.svd(normals[s] * w[:, None])[2][-1]
            new.append(u if u @ A[k] > 0 else -u)
        U, _, Vt = np.linalg.svd(np.stack(new))
        A = U @ Vt
        if np.linalg.det(A) < 0:
            A[2] *= -1
    sup = np.zeros(n, bool)
    for a in A:
        sup |= np.abs(normals @ a) < tol
    return A, float(length[sup].sum() / length.sum())


def fit_frame(seg: np.ndarray, size: tuple[int, int], f: float, k: float, **kw) -> Frame | None:
    w, h = size
    cx, cy = w / 2, h / 2
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    A, share = manhattan(plane_normals(seg, f, k, cx, cy), length, **kw)
    if A is None:
        return None
    return Frame(A, share, f, k, vertical_axis(A), len(seg))


def fit_lens(seg: np.ndarray, size: tuple[int, int], f_over_w=np.linspace(0.35, 0.9, 12), ks=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
             **kw) -> tuple[Frame | None, list[tuple[float, float, float]]]:
    """Scan the lens: the (f, k) with the highest explained line length. Returns the best Frame and the scanned scores
    [(f/w, k, share)], which show how flat the optimum is (a flat optimum means the lens is not determined by this picture)."""
    w, _ = size
    best, scan = None, []
    for fw in f_over_w:
        for k in ks:
            fr = fit_frame(seg, size, float(fw * w), float(k), **kw)
            if fr is None:
                continue
            scan.append((round(float(fw), 3), float(k), round(fr.score, 4)))
            if best is None or fr.score > best.score:
                best = fr
    return best, scan


def overlay_axes(image: np.ndarray, seg: np.ndarray, fr: Frame, width: int = 1280, tol_deg: float = 1.5) -> np.ndarray:
    """The segments (of line_segments(image, width)) coloured by the axis they follow (red, green, blue = rows of R), grey otherwise."""
    import cv2
    h, w = image.shape[:2]
    s = width / w
    out = cv2.resize(image, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    nrm = plane_normals(seg, fr.f, fr.k, width / 2, out.shape[0] / 2)     # the frame was fitted on these same shrunk pixels
    tol = math.sin(math.radians(tol_deg))
    cols = [(0, 0, 255), (0, 200, 0), (255, 80, 0)]
    for sg, n in zip(seg, nrm):
        c = (140, 140, 140)
        for k in range(3):
            if abs(n @ fr.R[k]) < tol:
                c = cols[k]
        cv2.line(out, (int(sg[0]), int(sg[1])), (int(sg[2]), int(sg[3])), c, 2, cv2.LINE_AA)
    return out


def pooled_lens(segs: dict[str, np.ndarray], size: tuple[int, int], f_over_w, ks, **kw) -> tuple[tuple[float, float], list]:
    """One lens for all cameras of a site: the (f/w, k) with the largest summed explained share over the cameras.
    Returns ((f/w, k), [(f/w, k, mean share, {cam: share})]) for every scanned pair."""
    w, _ = size
    rows = []
    for fw in f_over_w:
        for k in ks:
            per = {}
            for cam, seg in segs.items():
                fr = fit_frame(seg, size, float(fw * w), float(k), **kw)
                per[cam] = 0.0 if fr is None else fr.score
            rows.append((round(float(fw), 4), round(float(k), 4), float(np.mean(list(per.values()))), per))
    best = max(rows, key=lambda r: r[2])
    return (best[0], best[1]), rows
