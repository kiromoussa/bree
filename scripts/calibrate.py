"""Calibrate a camera to the shared store floor plan, and check that a calibration still holds.

    # fit: marks in the store YAML -> floor homography, and the camera pose when there is enough to go on
    PYTHONPATH=src .venv/bin/python scripts/calibrate.py fit configs/store1_cooler_cam.yaml --hfov 25
    PYTHONPATH=src .venv/bin/python scripts/calibrate.py fit configs/store1_cooler_cam.yaml \\
        --layout ../bree/software/shared/layouts/recommended-47.json --camera-id COOLER-rod-1 --in-place
    # check: exit code 1 and one line per problem when a camera is over the thresholds
    PYTHONPATH=src .venv/bin/python scripts/calibrate.py check configs/store1_*.yaml --max-px 5 --max-m 0.15

Marks (in the store YAML, under `camera:`):
  floor_points  [[x_px, y_px, x_m, y_m], ...]        a pixel and the same spot on the floor plan (metres)
  marks_3d      [[x_px, y_px, x_m, y_m, h_m], ...]   the same with a height, e.g. a shelf-edge corner
What comes out:
  4+ floor marks                       floor homography (what the multi-camera handoff needs)
  4+ marks and a known lens            full pose (cv2.solvePnP). The lens: --hfov, `camera.hfov_deg` in
                                       the YAML, or the camera's entry in --layout
  6+ marks, lens unknown               pose and focal length together
  --layout and no marks                the design pose from the layout file, written as method "layout"
                                       (not verified against the image: check will say so)
`fit` prints the reprojection error and writes `camera.calibration` (focal length, rotation, position, the
errors, the method) into a copy of the YAML (`<name>.calibrated.yaml`, or --out, or --in-place; YAML comments
are not kept). With --layout and --camera-id it also reports how far the fitted pose is from the design pose,
and with no floor marks it writes floor points generated from the pose so the handoff has a floor mapping.

`check` reprojects the marks through the stored calibration (or fits them when there is none) and fails a
camera whose reprojection RMS is over --max-px or whose floor error is over --max-m. It compares marks with
a calibration: after a camera is bumped, click the marks again on a fresh frame (scripts/draw_zones.py)
and run check before fit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from bree.calib.camera import Camera, calibrate, check, from_layout, marks_to_points


def _layout_camera(layout: str | None, camera_id: str | None, fallback_id: str | None) -> Camera | None:
    if not layout:
        return None
    cams = {c["id"]: c for c in json.loads(Path(layout).read_text())["cameras"]}
    cid = camera_id or fallback_id
    if cid not in cams:
        raise SystemExit(f"camera {cid!r} is not in {layout} (pass --camera-id; it has {len(cams)} cameras)")
    return from_layout(cams[cid])


def _focal(args, cam_cfg: dict, prior: Camera | None, width: int) -> float | None:
    hfov = args.hfov or cam_cfg.get("hfov_deg")
    if hfov:
        return (width / 2) / math.tan(math.radians(float(hfov)) / 2)
    return prior.f if prior is not None else None


def _line(name: str, rep: dict) -> str:
    def v(k, unit, nd=2):
        return f"{rep[k]:.{nd}f} {unit}" if rep.get(k) is not None else "n/a"
    s = (f"{name}: {rep['method'] or 'no fit'}, {rep['n']} mark(s) ({rep['n_floor']} on the floor), "
         f"reprojection {v('rms_px', 'px RMS')} (max {v('max_px', 'px')}), floor {v('floor_rms_m', 'm RMS', 3)}"
         f", leave-one-out {v('floor_loo_rms_m', 'm', 3)}")
    if rep.get("moved_m") is not None:
        s += f", vs layout: {rep['moved_m']:.3f} m, {rep['turned_deg']:.2f} deg"
    return s


def fit(args) -> int:
    path = Path(args.store)
    raw = yaml.safe_load(path.read_text())
    cam_cfg = raw.setdefault("camera", {})
    res = tuple(cam_cfg.get("resolution", [1280, 720]))
    prior = _layout_camera(args.layout, args.camera_id, cam_cfg.get("id"))
    if prior is not None and tuple(prior.resolution) != res:
        raise SystemExit(f"resolution {list(res)} in {path} is not the layout camera's {list(prior.resolution)}")
    pts = marks_to_points(cam_cfg.get("floor_points"), cam_cfg.get("marks_3d"))
    cam, rep = calibrate(pts, res, f=_focal(args, cam_cfg, prior, res[0]), prior=prior)
    if cam is None and len(pts) == 0 and prior is not None:
        cam, rep = prior, {**rep, "method": "layout"}
    print(_line(cam_cfg.get("id", path.stem), rep))
    if rep["method"] is None:
        print("  nothing to write: 4+ floor marks, or 4+ marks and a lens (--hfov / --layout), are needed")
        return 1
    if cam is not None:
        cam_cfg["calibration"] = {**cam.to_dict(), "method": rep["method"], "n_marks": rep["n"],
                                  **{k: round(rep[k], 4) for k in ("rms_px", "max_px", "floor_rms_m", "floor_loo_rms_m",
                                                                    "moved_m", "turned_deg") if rep.get(k) is not None}}
        cam_cfg["hfov_deg"] = round(cam.hfov_deg, 3)
        if not cam_cfg.get("floor_points"):
            # No floor marks: give the handoff a floor mapping from the pose (a grid of floor spots in view).
            g = np.array([(x, 0.0, z) for x in np.arange(-15, 15.01, 0.5) for z in np.arange(-15, 15.01, 0.5)])
            px, depth = cam.project(g)
            ok = np.flatnonzero(cam.in_frame(px, depth) & (depth < 15))
            if len(ok) >= 4:
                keep = ok[np.linspace(0, len(ok) - 1, min(12, len(ok))).round().astype(int)]
                cam_cfg["floor_points"] = [[round(float(px[i, 0]), 1), round(float(px[i, 1]), 1),
                                            round(float(g[i, 0]), 3), round(float(g[i, 2]), 3)] for i in keep]
                print(f"  no floor marks: wrote {len(keep)} floor points from the pose")
    else:
        cam_cfg.pop("calibration", None)      # homography only: floor_points are the calibration
        print("  floor homography only (no lens given and fewer than 6 marks): no pose written")
    out = path if args.in_place else Path(args.out) if args.out else path.with_suffix(".calibrated.yaml")
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"  wrote {out}")
    return 0


def run_check(args) -> int:
    bad = 0
    for p in args.stores:
        cam_cfg = yaml.safe_load(Path(p).read_text()).get("camera", {})
        res = tuple(cam_cfg.get("resolution", [1280, 720]))
        pts = marks_to_points(cam_cfg.get("floor_points"), cam_cfg.get("marks_3d"))
        calib = cam_cfg.get("calibration")
        if calib and calib.get("method") == "layout":
            rep = {"method": "layout", "n": len(pts), "n_floor": len(pts), "ok": False, "rms_px": None, "max_px": None,
                   "floor_rms_m": None, "floor_loo_rms_m": None,
                   "problems": ["design pose from the layout file, never checked against the image: add marks and fit"]}
        else:
            hf = args.hfov or cam_cfg.get("hfov_deg")
            f = (res[0] / 2) / math.tan(math.radians(float(hf)) / 2) if hf else None
            rep = check(pts, calib, res, args.max_px, args.max_m, f=f)
        name = cam_cfg.get("id", Path(p).stem)
        print(("OK    " if rep["ok"] else "FAIL  ") + _line(name, rep))
        for why in rep["problems"]:
            print(f"      {name}: {why}")
        bad += not rep["ok"]
    print(f"{len(args.stores) - bad} of {len(args.stores)} camera(s) pass (max {args.max_px:g} px, {args.max_m:g} m)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="marks -> calibration, written into the store YAML")
    f.add_argument("store")
    f.add_argument("--layout", help="shared layout JSON: design pose and lens of the camera")
    f.add_argument("--camera-id", help="camera id in the layout (default: camera.id of the YAML)")
    f.add_argument("--hfov", type=float, help="horizontal field of view of the lens, degrees")
    f.add_argument("--out")
    f.add_argument("--in-place", action="store_true")
    c = sub.add_parser("check", help="flag cameras whose error is over the thresholds")
    c.add_argument("stores", nargs="+")
    c.add_argument("--max-px", type=float, default=5.0)
    c.add_argument("--max-m", type=float, default=0.15)
    c.add_argument("--hfov", type=float)
    args = ap.parse_args(argv)
    return fit(args) if args.cmd == "fit" else run_check(args)


if __name__ == "__main__":
    sys.exit(main())
