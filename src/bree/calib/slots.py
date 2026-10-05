"""Where on the shelf did the hand go: triangulation from two calibrated cameras, one-view fallback.

Error model: the simulator's (bree/software/sim-prototype/js/cameras.js, `TRI` / `rayOf` / `pairError`). A
camera pins a point down across its line of sight, with a 1 sigma miss of
    s = sqrt((pixel sigma / pixels per metre)^2 + (range x pointing error)^2)   metres
in each direction perpendicular to the ray. `NoiseModel(pixel_px, cam_rot_deg)` are the simulator's `pxSigma`
and `calibDeg`; `cam_pos_m` adds an error on where the camera hangs (0 in the simulator). `sigma_px` is the
same s expressed in pixels, `triangulate` propagates it to a 3D covariance, and `perturbed` draws a
mis-calibrated camera for Monte Carlo runs.

How close `rms_m` is to the simulator's two-ray `pairError` (audit 2026-10-05): the same formula, but not
the same number. On the simulator's own per-slot output for recommended-47.json (471 slots it calls
localisable, the same camera pair per slot) rms_m / errCm has median 0.960, minimum 0.891, maximum 0.999:
within about 10%, and the simulator is the more cautious one. It applies a lens edge loss (edgeLoss 0.2) and
uses range where this code uses depth along the optical axis. tests/test_calib.py checks the formula at the
image centre with 5% tolerance, where the two agree; it does not see the gap off axis.
The prediction is first order and runs a little low: an independent Monte Carlo (40 layout slot pairs, 1,500
draws each) measured rms at 1.043 times rms_m at the median, range 1.000 to 1.091 with this module's
defaults and 0.991 to 1.121 with the simulator's. Read rms_m as about 4% low, up to about 12% off axis.
The slot gate and `assign` weights lean on it; the 0.7 to 1.4 band in the tests covers it.

  triangulate()   N-view linear triangulation + two Gauss-Newton steps on the reprojection error
  SlotMap         the layout's slots: front-face centre and outward normal of each. `assign` picks the most
                  likely slot for a measured hand, weighting each direction by how well it was measured.
  SlotLocator     buffers each person's wrist/elbow pixels per camera; for a PICK it returns the slot from
                  two or more views at the same moment, else from one view (line of sight meets the shelf
                  face). Used by bree.track.multicam.StoreEvents.

All 3D points are in the layout frame (bree/calib/camera.py).
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np

from bree.calib.camera import Camera


@dataclass
class NoiseModel:
    """Placeholders until measured on the real cameras: a pose model's wrist is a few pixels off at best, and
    the calibration terms are what scripts/calibrate.py leaves behind from hand-clicked marks."""
    pixel_px: float = 5.0          # 1 sigma, each image axis, on a hand keypoint (simulator: pxSigma)
    cam_pos_m: float = 0.02        # 1 sigma, each axis, on the camera position (not in the simulator's model)
    cam_rot_deg: float = 0.2       # 1 sigma, each axis, on the camera orientation (simulator: calibDeg)

    @staticmethod
    def sim(px_sigma: float = 1 / math.sqrt(12), calib_deg: float = 0.1) -> "NoiseModel":
        """The simulator's model and defaults: a point known to the nearest pixel, 0.1 degree pointing error."""
        return NoiseModel(px_sigma, 0.0, calib_deg)

    def sigma_px(self, cam: Camera, range_m: float) -> float:
        """Total 1 sigma image error of a point `range_m` away: pixel noise, plus what the camera's pose
        error moves the point by in the image."""
        rot = cam.f * math.radians(self.cam_rot_deg)
        pos = cam.f * self.cam_pos_m / max(range_m, 0.1)
        return math.sqrt(self.pixel_px ** 2 + rot ** 2 + pos ** 2)


def perturbed(cam: Camera, rng: np.random.Generator, noise: NoiseModel) -> Camera:
    """A draw of the camera as a noisy calibration would report it."""
    w = np.radians(rng.normal(0, noise.cam_rot_deg, 3)) if noise.cam_rot_deg else np.zeros(3)
    th = float(np.linalg.norm(w))
    Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    dR = np.eye(3) if th < 1e-12 else np.eye(3) + math.sin(th) / th * Kx + (1 - math.cos(th)) / th ** 2 * Kx @ Kx
    return replace(cam, R=dR @ cam.R, C=cam.C + rng.normal(0, noise.cam_pos_m, 3) if noise.cam_pos_m else cam.C)


def _linearise(views, X: np.ndarray, noise: NoiseModel):
    """Reprojection Jacobians and residuals of X in every view, each divided by that view's sigma_px.
    None if X is behind a camera."""
    J, r, sig = [], [], []
    for cam, (u, v) in views:
        xc = cam.R @ (X - cam.C)
        if xc[2] <= 1e-6:
            return None
        sg = noise.sigma_px(cam, float(np.linalg.norm(X - cam.C)))
        J.append(cam.f / xc[2] * np.array([[1, 0, -xc[0] / xc[2]], [0, 1, -xc[1] / xc[2]]]) @ cam.R / sg)
        r.append((np.array([cam.f * xc[0] / xc[2] + cam.cx, cam.f * xc[1] / xc[2] + cam.cy]) - (u, v)) / sg)
        sig.append(sg)
    return np.vstack(J), np.concatenate(r), sig


def triangulate(views: list[tuple[Camera, tuple[float, float]]], noise: NoiseModel | None = None) -> dict | None:
    """3D point seen at pixel uv by each camera. Returns {"xyz", "info", "sigma_m", "rms_px", "gate_px"} or None
    when the lines of sight are parallel or the point is behind a camera. `info` is the 3x3 inverse covariance
    of the position under the noise model (first order); sigma_m is the largest 1 sigma axis of the error and
    rms_m the expected 3D rms error (the simulator's `errM`)."""
    if len(views) < 2:
        return None
    noise = noise or NoiseModel()
    A = []
    for cam, (u, v) in views:
        P = np.c_[cam.R, -cam.R @ cam.C]
        x, y = (u - cam.cx) / cam.f, (v - cam.cy) / cam.f
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    _, s, vt = np.linalg.svd(np.asarray(A))
    if abs(vt[-1][3]) < 1e-12 or s[2] < 1e-9 * s[0]:
        return None
    X = vt[-1][:3] / vt[-1][3]
    for it in range(3):                                       # two Gauss-Newton steps, the last pass only measures
        lin = _linearise(views, X, noise)
        if lin is None:
            return None
        J, r, sig = lin
        info = J.T @ J
        if np.linalg.cond(info) > 1e12:
            return None
        cov = np.linalg.inv(info)
        if it < 2:
            X = X - cov @ J.T @ r
    res = (r.reshape(-1, 2) * np.array(sig)[:, None])
    return {"xyz": X, "info": info, "sigma_m": float(math.sqrt(np.linalg.eigvalsh(cov)[-1])),
            "rms_m": float(math.sqrt(np.trace(cov))),
            "rms_px": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), "gate_px": 4.0 * max(sig)}


class SlotMap:
    """Front-face centre and outward normal of every slot of a layout (LAYOUT_FORMAT.md: a gondola slot
    faces away from the gondola's centre plane; cooler, counter and back-bar slots face the fixture's front)."""

    def __init__(self, layout: dict):
        fx = {f["id"]: f for f in layout["fixtures"]}
        ids, skus, zones, front, normal = [], [], [], [], []
        for s in layout["slots"]:
            f = fx[s["fixtureId"]]
            n = np.array([math.sin(f["rotationY"]), 0.0, math.cos(f["rotationY"])])
            pos = np.asarray(s["position"], float)
            if f["type"] == "gondola" and (pos - np.asarray(f["position"], float)) @ n < 0:
                n = -n
            ids.append(s["id"]); skus.append(s.get("skuId")); zones.append(s["fixtureId"])
            front.append(pos + n * abs(n @ np.asarray(s["size"], float)) / 2)
            normal.append(n)
        self.ids, self.skus, self.fixtures = ids, skus, zones
        self.front, self.normal = np.asarray(front), np.asarray(normal)
        # Solid fixtures that block a line of sight (glass cooler doors, walls and the door do not count here).
        self.boxes, self.box_ids = [], []
        for f in layout["fixtures"]:
            if f["type"] in ("gondola", "counter", "lottery", "backbar"):
                w, h, d = f["size"]
                c, s_ = abs(math.cos(f["rotationY"])), abs(math.sin(f["rotationY"]))
                half = np.array([w * c + d * s_, h, w * s_ + d * c]) / 2
                self.boxes.append((np.asarray(f["position"], float) - half, np.asarray(f["position"], float) + half))
                self.box_ids.append(f["id"])

    def __len__(self) -> int:
        return len(self.ids)

    def nearest(self, X) -> tuple[int, float]:
        d = np.linalg.norm(self.front - np.asarray(X, float), axis=1)
        i = int(np.argmin(d))
        return i, float(d[i])

    def assign(self, X, info: np.ndarray, radius_m: float = 0.5, gap_m: float = 0.05,
               hand_m: float = 0.04) -> tuple[int, float] | None:
        """Most likely slot for a hand measured at X with inverse covariance `info` (from one view or several).
        The hand is taken to be on the line straight out from the slot's front centre, a gap g >= 0 in front
        of it (prior: g about `gap_m`). Score = squared error of X against that line in units of the
        measurement's own sigma, plus (g / gap_m)^2. With one view `info` says nothing along the line of
        sight, so the score is how far the line of sight passes from the slot's line; with two views the
        depth counts as much as the noise model says it is worth. `hand_m`: a hand is not exactly on that
        line (the arm comes in at an angle), so this much is added to the measurement's sigma in every
        direction; without it a very precise measurement would be trusted beyond what a hand can tell.
        Returns (slot, score)."""
        info = info - info @ np.linalg.inv(info + np.eye(3) / hand_m ** 2) @ info      # (info^-1 + hand_m^2 I)^-1
        r = np.asarray(X, float) - self.front
        cand = np.flatnonzero(np.linalg.norm(r, axis=1) <= radius_m)
        if not len(cand):
            return None
        r, n = r[cand], self.normal[cand]
        nN = n @ info
        g = np.clip((nN * r).sum(1) / ((nN * n).sum(1) + 1.0 / gap_m ** 2), 0.0, None)
        q = r - g[:, None] * n
        score = ((q @ info) * q).sum(1) + (g / gap_m) ** 2
        k = int(np.argmin(score))
        return int(cand[k]), float(score[k])

    def along_ray(self, cam: Camera, uv, tol_m: float = 0.12) -> tuple[int, float, np.ndarray] | None:
        """One view: where the line of sight through `uv` meets a shelf face. (slot, distance from the slot's
        front centre to that point, the point). The first face along the line that has a slot within `tol_m`
        wins; if none is that close, the closest slot overall.
        ponytail: a hand held in front of the shelf is pushed along the line of sight onto the face, so the
        slot shifts sideways by (gap to the face) x tan(angle off the face normal). Two views do not have this."""
        d = cam.ray(*uv)
        denom = self.normal @ d
        ok = denom < -1e-6                                           # the camera looks at the slot's front
        t = np.where(ok, ((self.front - cam.C) * self.normal).sum(1) / np.where(ok, denom, 1.0), -1.0)
        ok &= t > 0
        if not ok.any():
            return None
        hit = cam.C + t[:, None] * d
        dist = np.where(ok, np.linalg.norm(hit - self.front, axis=1), np.inf)
        close = dist <= tol_m
        if close.any():                                              # the nearest face; on it, the closest slot
            dist = np.where(close & (t <= t[close].min() + 0.3), dist, np.inf)
        i = int(np.argmin(dist))
        return i, float(dist[i]), hit[i]

    def visible(self, cam: Camera, points, margin_px: float = 20.0, max_range_m: float = 12.0) -> np.ndarray:
        """Geometry only, per slot: points[i] (the slot's front centre, or a hand at it) is in frame, seen
        from the front, in range, and no OTHER solid fixture blocks the line (product fronts sit a few cm
        inside their own fixture's box; from behind, the facing test already says no). No shoppers, no shelf lips."""
        P = np.asarray(points, float)
        px, z = cam.project(P)
        seg = P - cam.C
        rng_ = np.linalg.norm(seg, axis=1)
        ok = cam.in_frame(px, z, margin_px) & ((-seg * self.normal).sum(1) > 0.05 * rng_) & (rng_ <= max_range_m)
        seg = seg * (1 - 0.03 / np.maximum(rng_, 0.03))[:, None]      # stop 3 cm short of the point itself
        with np.errstate(divide="ignore", invalid="ignore"):
            own = np.asarray(self.fixtures)
            for (lo, hi), fid in zip(self.boxes, self.box_ids):
                t1, t2 = (lo - cam.C) / seg, (hi - cam.C) / seg
                tn, tf = np.fmin(t1, t2).max(1), np.fmax(t1, t2).min(1)
                ok &= ~((tn <= tf) & (tf >= 0) & (tn <= 1) & (own != fid))
        return ok


@dataclass
class _Hand:
    t: float
    wrist: tuple[float, float]
    elbow: tuple[float, float] | None


class SlotLocator:
    def __init__(self, cameras: dict[str, Camera], slots: SlotMap, noise: NoiseModel | None = None,
                 sync_s: float = 0.07, keep_s: float = 6.0, extend: float = 0.5, gap_m: float = 0.05):
        self.cameras, self.slots, self.noise = cameras, slots, noise or NoiseModel()
        self.sync_s, self.keep_s, self.extend, self.gap_m = sync_s, keep_s, extend, gap_m
        self.buf: dict[tuple[str, int, str], deque] = defaultdict(deque)     # (camera, person, hand) -> samples

    def observe(self, cam: str, t: float, person: int, hands: dict[str, tuple]) -> None:
        """hands: {"left" / "right": (wrist px, elbow px or None)} of one person in one calibrated camera."""
        if cam not in self.cameras:
            return
        for name, (w, e) in hands.items():
            q = self.buf[(cam, person, name)]
            q.append(_Hand(t, w, e))
            while q and t - q[0].t > self.keep_s:
                q.popleft()

    def _hand_3d(self, views: list[tuple[str, _Hand]]) -> dict | None:
        tri = triangulate([(self.cameras[c], h.wrist) for c, h in views], self.noise)
        if tri is None or tri["rms_px"] > tri["gate_px"]:
            return None                                              # lines of sight do not meet: not the same hand
        el = [(self.cameras[c], h.elbow) for c, h in views if h.elbow is not None]
        e3 = triangulate(el, self.noise) if self.extend and len(el) >= 2 else None
        if e3 is not None and e3["rms_px"] <= e3["gate_px"]:         # fingertips: past the wrist, along the forearm
            tri["xyz"] = tri["xyz"] + self.extend * (tri["xyz"] - e3["xyz"])
        return tri

    def locate(self, cam: str, person: int, hand: str | None, t0: float, t1: float, item_uv=None) -> dict | None:
        """Slot the hand reached during [t0, t1] as seen by `cam`, best source first:
          hand_2view  the hand seen by two or more cameras at the same moment (within sync_s): triangulated,
                      then SlotMap.assign. Of all such moments, the best-scoring one.
          item_1view  `item_uv`, the pixel where this camera first saw the item on the shelf, before the
                      pick: its line of sight meets the shelf face at the slot itself.
          hand_1view  the hand's line of sight at the middle of the reach: the first shelf face it meets,
                      then SlotMap.assign with that one view."""
        best, mid = None, (t0 + t1) / 2
        for name in ([hand] if hand else ["left", "right"]):
            for h in self.buf.get((cam, person, name), ()):
                if not (t0 - 1e-6 <= h.t <= t1 + 1e-6):
                    continue
                views = [(cam, h)]
                for other in self.cameras:
                    q = self.buf.get((other, person, name)) if other != cam else None
                    if q:
                        o = min(q, key=lambda s: abs(s.t - h.t))
                        if abs(o.t - h.t) <= self.sync_s:
                            views.append((other, o))
                tri = self._hand_3d(views) if len(views) >= 2 else None
                got = self.slots.assign(tri["xyz"], tri["info"], max(0.5, 3 * tri["sigma_m"]), self.gap_m) if tri else None
                if got is not None:
                    cand = (0, got[1], self._out(got[0], tri["xyz"], "hand_2view", [c for c, _ in views], h.t, tri["sigma_m"]))
                else:
                    uv = h.wrist if h.elbow is None else tuple(np.asarray(h.wrist) + self.extend * (np.asarray(h.wrist) - h.elbow))
                    hit = self.slots.along_ray(self.cameras[cam], uv)
                    lin = _linearise([(self.cameras[cam], uv)], hit[2], self.noise) if hit is not None else None
                    got = self.slots.assign(hit[2], lin[0].T @ lin[0], 0.5, self.gap_m) if lin is not None else None
                    if got is None:
                        continue
                    cand = (2, abs(h.t - mid), self._out(got[0], hit[2], "hand_1view", [cam], h.t, None))
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if (best is None or best[0] > 0) and item_uv is not None and cam in self.cameras:
            hit = self.slots.along_ray(self.cameras[cam], item_uv)
            if hit is not None:
                return self._out(hit[0], hit[2], "item_1view", [cam], t0, None)
        return best[2] if best else None

    def _out(self, i: int, X, source: str, cams: list[str], t: float, sigma) -> dict:
        return {"id": self.slots.ids[i], "sku": self.slots.skus[i], "fixture": self.slots.fixtures[i],
                "source": source, "views": len(cams), "cameras": cams, "xyz": [round(float(v), 3) for v in X],
                "sigma_m": None if sigma is None else round(sigma, 3),
                "dist_m": round(float(np.linalg.norm(self.slots.front[i] - np.asarray(X))), 3), "t": t}
