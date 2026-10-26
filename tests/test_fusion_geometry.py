"""bree.fusion: Manhattan frame and lens from lines, plan cameras, registration by simultaneous people (synthetic)."""
import numpy as np

from bree.calib.camera import Camera
from bree.fusion import manhattan as M
from bree.fusion import plan as P
from bree.fusion import registry as R
from bree.fusion.calibrate import fit_points, similarity, apply_h, order_quad


def _camera(heading="+x", yaw_jitter=0.0, k=0.3, pos=(0.0, 0.0), h=2.5):
    # a camera built from a known Manhattan frame: pitched 35 degrees down, slight roll
    pitch, roll = np.radians(35), np.radians(3)
    up = np.array([np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch), -np.sin(pitch)])
    fwd = np.array([-np.sin(roll) * np.sin(pitch), np.sin(pitch) * np.cos(roll), np.cos(pitch)])   # horizontal forward
    fwd = fwd - (fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    side = np.cross(up, fwd)
    axes = np.stack([up, fwd, side])
    return P.camera_from_axes(axes, heading, 768.0, k, (1280, 720), pos, h), axes


def _lines(cam: Camera, n=400, seed=0):
    """Short segments of 3D lines along the three building axes, projected with the camera (lens included)."""
    rng = np.random.default_rng(seed)
    segs = []
    while len(segs) < n:
        P0 = np.array([rng.uniform(-6, 6), rng.uniform(0, 2.5), rng.uniform(3, 14)])
        d = np.eye(3)[rng.integers(3)] * rng.uniform(0.4, 1.0)
        px, z = cam.project(np.stack([P0, P0 + d]))
        if (z > 0.5).all() and ((px > [0, 0]) & (px < [1280, 720])).all() and np.linalg.norm(px[1] - px[0]) > 25:
            segs.append([*px[0], *px[1]])
    return np.array(segs)


def test_manhattan_recovers_lens_and_frame():
    cam, axes = _camera()
    seg = _lines(cam, seed=1)
    best, scan = M.fit_lens(seg, (1280, 720), f_over_w=np.array([0.5, 0.6, 0.7]), ks=(0.0, 0.3, 0.5), iters=300)
    assert abs(best.f / 1280 - 0.6) < 1e-6 and best.k == 0.3          # the true lens is the optimum of the scan
    assert best.score > 0.9
    # the vertical axis found is the true up (up to sign)
    up = axes[M.vertical_axis(axes)]
    assert abs(abs(best.R[best.vertical] @ up) - 1) < 0.01


def test_vertical_axis_for_steep_and_shallow_cameras():
    for pitch in (20, 45, 70):
        p = np.radians(pitch)
        up = np.array([0, -np.cos(p), -np.sin(p)])
        fwd = np.array([0, -np.sin(p) * np.cos(p), np.cos(p) ** 2]); fwd /= np.linalg.norm(fwd)
        side = np.array([1.0, 0, 0])
        for perm in ([up, fwd, side], [side, up, fwd], [fwd, side, up]):
            axes = np.stack(perm)
            assert np.allclose(axes[M.vertical_axis(axes)], up)


def test_camera_from_axes_floor_roundtrip_and_heading():
    cam, _ = _camera("-y", pos=(3.0, 4.0))
    for x, y in [(3.5, 1.0), (2.0, -2.0), (4.0, 0.0)]:
        pix = P.to_pixel(cam, x, y)
        assert pix is not None
        got = P.floor_xy(cam, *pix)
        assert np.allclose(got, (x, y), atol=1e-6)
    # heading -y: the optical axis points towards decreasing y: nearer to the camera in y is lower in the picture
    a, b = P.to_pixel(cam, 3.0, 2.0), P.to_pixel(cam, 3.0, 0.0)
    assert b[1] < a[1]
    # layout frame is right handed (x, up, z): x cross up = z
    R_ = cam.R
    assert np.allclose(np.cross(R_[:, 0], R_[:, 1]), R_[:, 2], atol=1e-6)


def test_plan_roundtrip_and_region(tmp_path):
    cam, _ = _camera()
    plan = P.Plan("t", {"01": P.PlanCamera("01", cam, "aisle", "+x", None, 12.0)},
                  {"sales": {"kind": "floor", "poly": [[0, 0], [10, 0], [10, 8], [0, 8]]},
                   "cooler": {"kind": "cooler", "poly": [[8, 0], [10, 0], [10, 8], [8, 8]]}})
    plan.save(tmp_path / "p.json")
    back = P.Plan.load(tmp_path / "p.json")
    assert back.region_at(9, 3) == "cooler" and back.region_at(3, 3) == "sales" and back.region_at(20, 3) is None
    pix = P.to_pixel(cam, 6.0, 1.0)
    assert np.allclose(back.place("01", *pix), (6.0, 1.0), atol=1e-4)


def test_fit_points_and_similarity():
    H0 = np.array([[1.2, 0.1, 5.0], [0.05, 0.9, 3.0], [1e-4, 2e-4, 1.0]])
    uv = np.array([[100, 100], [900, 120], [880, 600], [120, 620], [500, 300], [300, 450]], float)
    xy = apply_h(H0, uv)
    fit = fit_points(np.c_[uv, xy])
    assert fit.rms_m < 1e-6 and fit.loo_m < 1e-4
    bad = np.c_[uv, xy]
    bad[4, 2] += 1.0
    assert max(fit_points(bad).per_point_m) > 0.3
    S = similarity(xy, xy * 2 + 7)
    assert np.allclose(apply_h(S, xy), xy * 2 + 7, atol=1e-6)
    q = order_quad(np.array([[10, 90], [90, 10], [10, 10], [90, 90]]))
    assert q.tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_vote_translation_recovers_shift_and_clock_offset():
    rng = np.random.default_rng(3)
    t = np.arange(0, 300, 0.2)
    walk = np.cumsum(rng.normal(0, 0.15, (len(t), 2)), 0) + [5.0, 5.0]
    walk2 = np.cumsum(rng.normal(0, 0.15, (len(t), 2)), 0) + [2.0, 8.0]
    shift = np.array([3.0, -2.0])
    dt = 1.6

    def mk(ts, xy):
        return R.Placed(np.asarray(ts), np.asarray(xy), np.ones(len(ts)), np.zeros(len(ts), int))
    # camera A sees both people; camera B (offset origin, clock ahead by dt) sees them, 20 percent of frames missed
    keep = rng.random(len(t)) > 0.2
    a = mk(np.r_[t, t], np.r_[walk, walk2])
    b = mk(np.r_[t[keep] + dt, t[keep] + dt], np.r_[(walk - shift)[keep], (walk2 - shift)[keep]] + rng.normal(0, 0.1, (2 * keep.sum(), 2)))
    pk, scan = R.vote_translation(a, b, dts=np.arange(-4, 4.01, 0.4))
    assert abs(pk.dt - dt) <= 0.21 and np.allclose(pk.shift, shift, atol=0.3)
    assert pk.votes > 5 * pk.background
