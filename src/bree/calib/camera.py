"""Camera calibration to the shared store floor plan.

Frame: the layout file's own (software/sim-prototype/LAYOUT_FORMAT.md): metres, right handed, y up, floor at
y = 0. A floor-plan point (x_m, y_m) of a store YAML is the layout point (x, 0, z) = (x_m, 0, y_m), as in
bree/sim/isaac_adapter.floor_points. A mark with a height is (x_m, h_m, y_m).

Model: pinhole, square pixels, principal point at the image centre, no lens distortion.
ponytail: no distortion term. The planned lenses are about 25 degrees wide, where it is small; add k1 (and
cv2.calibrateCamera on a checkerboard) if the reprojection error of real marks grows toward the frame edge.

  from_layout()  the design pose of a camera in the layout JSON (yaw, pitch, hfov)
  calibrate()    from marks clicked in the image: floor homography (4+ floor marks) and, with a known focal
                 length (lens spec or layout hfov) or 6+ marks, the full pose with cv2.solvePnP. Reports the
                 reprojection error.
  check()        marks against a stored calibration; flags a camera over the thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bree.track.multicam import homography


@dataclass
class Camera:
    f: float                       # focal length, pixels
    cx: float
    cy: float
    R: np.ndarray                  # 3x3: layout frame -> camera frame (x right, y down, z forward)
    C: np.ndarray                  # optical centre in the layout frame (x, up, z), metres
    resolution: tuple[int, int] = (0, 0)

    def project(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Layout points (N, 3) -> (pixels (N, 2), depth along the optical axis (N,)). Depth <= 0: behind."""
        xc = (np.asarray(X, float).reshape(-1, 3) - self.C) @ self.R.T
        z = xc[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return np.c_[self.f * xc[:, 0] / zs + self.cx, self.f * xc[:, 1] / zs + self.cy], z

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction (layout frame) of the line of sight through pixel (u, v)."""
        d = self.R.T @ np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        return d / np.linalg.norm(d)

    def in_frame(self, px, depth, margin: float = 0.0) -> np.ndarray:
        w, h = self.resolution
        px = np.asarray(px, float).reshape(-1, 2)
        return (np.asarray(depth) > 1e-6) & (px[:, 0] >= margin) & (px[:, 0] <= w - margin) \
            & (px[:, 1] >= margin) & (px[:, 1] <= h - margin)

    def floor_homography(self) -> np.ndarray:
        """3x3 H with floor (x_m, y_m) ~ H @ (u, v, 1): same contract as StoreConfig.floor_homography."""
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        G = K @ np.c_[self.R[:, 0], self.R[:, 2], -self.R @ self.C]      # floor (x, z, 1) -> image
        H = np.linalg.inv(G)
        return H / H[2, 2]

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.resolution[0] / 2 / self.f))

    def to_dict(self) -> dict:
        return {"f_px": round(float(self.f), 3), "cx": round(float(self.cx), 3), "cy": round(float(self.cy), 3),
                "R": [[round(float(v), 8) for v in row] for row in self.R],
                "position": [round(float(v), 4) for v in self.C],
                "resolution": [int(v) for v in self.resolution]}

    @staticmethod
    def from_dict(d: dict) -> "Camera":
        return Camera(float(d["f_px"]), float(d["cx"]), float(d["cy"]), np.asarray(d["R"], float),
                      np.asarray(d["position"], float), tuple(d["resolution"]))


def from_layout(cam: dict) -> Camera:
    """Design pose of a layout camera. three.js: yaw about +y, then pitch about the camera's x; looks down -z."""
    w, h = cam["resolution"]
    cy, sy, cp, sp = math.cos(cam["yaw"]), math.sin(cam["yaw"]), math.cos(cam["pitch"]), math.sin(cam["pitch"])
    right = np.array([cy, 0.0, -sy])
    fwd = np.array([-sy * cp, sp, -cy * cp])
    down = -np.cross(right, fwd)
    f = (w / 2) / math.tan(math.radians(cam["hfov"]) / 2)
    return Camera(f, w / 2, h / 2, np.stack([right, down, fwd]), np.asarray(cam["position"], float), (int(w), int(h)))


def marks_to_points(floor_points=None, marks_3d=None) -> np.ndarray:
    """(N, 5) rows [x_px, y_px, x_m, y_m, h_m] from `camera.floor_points` ([x_px, y_px, x_m, y_m], on the
    floor) and `camera.marks_3d` ([x_px, y_px, x_m, y_m, h_m], e.g. a shelf-edge corner at a known height)."""
    rows = [[*map(float, p[:4]), 0.0] for p in (floor_points or [])] + [[*map(float, p[:5])] for p in (marks_3d or [])]
    return np.asarray(rows, float).reshape(-1, 5)


def _layout_xyz(pts: np.ndarray) -> np.ndarray:
    return np.c_[pts[:, 2], pts[:, 4], pts[:, 3]]


def _errors(cam: Camera, pts: np.ndarray) -> np.ndarray:
    px, _ = cam.project(_layout_xyz(pts))
    return np.linalg.norm(px - pts[:, :2], axis=1)


def _floor_errors(H: np.ndarray, floor: np.ndarray) -> np.ndarray:
    p = np.c_[floor[:, :2], np.ones(len(floor))] @ H.T
    return np.linalg.norm(p[:, :2] / p[:, 2:3] - floor[:, 2:4], axis=1)


def _pose(obj: np.ndarray, img: np.ndarray, f: float, cx: float, cy: float, prior: Camera | None,
          res: tuple[int, int]) -> Camera | None:
    """Best pose for a given focal length. A flat set of marks (floor only) has two candidate poses; the one
    with the lower error wins, or the one nearer the design pose when the errors are close."""
    import cv2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    s = np.linalg.svd(obj - obj.mean(0), compute_uv=False)
    planar = s[2] < 1e-3 * max(s[0], 1e-9)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, None,
                                                 flags=cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    best = None
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, None, rv, tv)
        Rm = cv2.Rodrigues(rv)[0]
        cam = Camera(f, cx, cy, Rm, (-Rm.T @ tv).ravel(), res)
        px, z = cam.project(obj)
        if cam.C[1] <= 0 or (z <= 0).any():                 # under the floor, or marks behind the camera
            continue
        rms = float(np.sqrt(np.mean(np.sum((px - img) ** 2, axis=1))))
        # Near-tied flat solutions: prefer the one closest to where the layout says the camera hangs.
        key = (round(rms, 1) if prior is not None else rms,
               float(np.linalg.norm(cam.C - prior.C)) if prior is not None else 0.0)
        if best is None or key < best[0]:
            best = (key, cam, rms)
    return best[1] if best else None


def calibrate(points, resolution, f: float | None = None, prior: Camera | None = None) -> tuple[Camera | None, dict]:
    """Marks -> (Camera or None, report).

    points: (N, 5) rows [x_px, y_px, x_m, y_m, h_m] (see marks_to_points).
    f: focal length in pixels if known (lens spec, or the layout's hfov). None = estimated, which needs 6+ marks.
    prior: the design pose (from_layout); breaks the tie between the two poses a flat set of marks allows.
    Report keys: n, n_floor, method (homography / pnp / pnp+focal), rms_px, max_px (reprojection of every
    mark), floor_rms_m, floor_max_m (homography fit, floor marks only), floor_loo_rms_m (each floor mark
    predicted from the others, 5+ marks: with exactly 4 the fit is exact and proves nothing), and with a
    prior moved_m, turned_deg (estimated pose vs design pose)."""
    pts = np.asarray(points, float).reshape(-1, 5)
    w, h = resolution
    floor = pts[pts[:, 4] == 0]
    rep: dict = {"n": len(pts), "n_floor": len(floor), "method": None, "rms_px": None, "max_px": None,
                 "floor_rms_m": None, "floor_max_m": None, "floor_loo_rms_m": None}
    def spans_a_plane(p):                                    # not all on one line
        sv = np.linalg.svd(p - p.mean(0), compute_uv=False)
        return sv[1] > 1e-6 * max(sv[0], 1e-12)
    H = None
    if len(floor) >= 4 and spans_a_plane(floor[:, :2]) and spans_a_plane(floor[:, 2:4]):
        H = homography(floor[:, :2], floor[:, 2:4])
        if not np.isfinite(H).all() or np.linalg.cond(H) > 1e12:
            H = None
    if H is not None:
        e = _floor_errors(H, floor)
        rep.update(method="homography", floor_rms_m=float(np.sqrt(np.mean(e ** 2))), floor_max_m=float(e.max()))
        if len(floor) >= 5:
            loo = [_floor_errors(homography(np.delete(floor, i, 0)[:, :2], np.delete(floor, i, 0)[:, 2:4]),
                                 floor[i:i + 1])[0] for i in range(len(floor))]
            if np.isfinite(loo).all():
                rep["floor_loo_rms_m"] = float(np.sqrt(np.mean(np.square(loo))))
    cam = None
    obj, img = _layout_xyz(pts), np.ascontiguousarray(pts[:, :2])
    if len(pts) < 4 or not spans_a_plane(obj):
        pass                                                 # marks on one line fix no pose
    elif f is not None:
        cam = _pose(obj, img, f, w / 2, h / 2, prior, (w, h))
        rep["method"] = "pnp" if cam else rep["method"]
    elif len(pts) >= 6:
        from scipy.optimize import minimize_scalar

        def cost(logf):
            c = _pose(obj, img, math.exp(logf), w / 2, h / 2, prior, (w, h))
            return float(np.sqrt(np.mean(_errors(c, pts) ** 2))) if c else 1e9
        grid = np.log([(w / 2) / math.tan(math.radians(a) / 2) for a in np.geomspace(8, 140, 36)])
        costs = [cost(g) for g in grid]
        k = int(np.argmin(costs))
        if costs[k] < 1e9:
            lo, hi = grid[min(k + 1, len(grid) - 1)], grid[max(k - 1, 0)]      # the grid runs wide to narrow
            r = minimize_scalar(cost, bounds=(min(lo, hi), max(lo, hi)), method="bounded", options={"xatol": 1e-5})
            cam = _pose(obj, img, math.exp(r.x if r.fun <= costs[k] else grid[k]), w / 2, h / 2, prior, (w, h))
            rep["method"] = "pnp+focal" if cam else rep["method"]
    if cam is not None:
        e = _errors(cam, pts)
        rep.update(rms_px=float(np.sqrt(np.mean(e ** 2))), max_px=float(e.max()), hfov_deg=cam.hfov_deg)
        if prior is not None:
            cosang = (np.trace(cam.R @ prior.R.T) - 1) / 2
            rep.update(moved_m=float(np.linalg.norm(cam.C - prior.C)),
                       turned_deg=math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
    elif rep["method"] == "homography":
        # No pose: the pixel error is the homography's own (floor plan -> image).
        Hinv = np.linalg.inv(H)
        p = np.c_[floor[:, 2:4], np.ones(len(floor))] @ Hinv.T
        e = np.linalg.norm(p[:, :2] / p[:, 2:3] - floor[:, :2], axis=1)
        rep.update(rms_px=float(np.sqrt(np.mean(e ** 2))), max_px=float(e.max()))
    return cam, rep


def check(points, calibration: dict | None, resolution, max_px: float = 5.0, max_m: float = 0.15,
          f: float | None = None) -> dict:
    """Is this camera's calibration still good? With a stored `calibration` the marks are reprojected through
    it; without one the marks are fitted now. Returns the report plus `ok` and `problems` (plain sentences).
    A camera fails when the reprojection RMS is over `max_px`, the floor error (leave-one-out when there are
    5+ floor marks, else the fit) is over `max_m`, or there is nothing to check."""
    pts = np.asarray(points, float).reshape(-1, 5)
    cam, rep = calibrate(pts, resolution, f=f)
    problems = []
    if calibration:
        stored = Camera.from_dict(calibration)
        if len(pts):
            e = _errors(stored, pts)
            rep.update(method="stored", rms_px=float(np.sqrt(np.mean(e ** 2))), max_px=float(e.max()))
            floor = pts[pts[:, 4] == 0]
            if len(floor):
                fe = _floor_errors(stored.floor_homography(), floor)
                rep.update(floor_rms_m=float(np.sqrt(np.mean(fe ** 2))), floor_max_m=float(fe.max()),
                           floor_loo_rms_m=None)
    if len(pts) < 4:
        problems.append(f"{len(pts)} mark(s): at least 4 are needed to check anything")
    elif rep["rms_px"] is None:
        problems.append("no fit: fewer than 4 floor marks and no focal length for a pose")
    else:
        if rep["rms_px"] > max_px:
            problems.append(f"reprojection error {rep['rms_px']:.1f} px RMS is over {max_px:g} px")
        fm = rep["floor_loo_rms_m"] if rep["floor_loo_rms_m"] is not None else rep["floor_rms_m"]
        if fm is not None and fm > max_m:
            problems.append(f"floor error {fm:.2f} m is over {max_m:g} m")
        if not calibration and rep["method"] == "homography" and rep["n_floor"] == 4:
            problems.append("exactly 4 floor marks: the fit is exact by construction, add a fifth to check it")
    return {**rep, "ok": not problems, "problems": problems}
