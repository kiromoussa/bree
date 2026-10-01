"""Draw a camera's zones (and floor points) on one frame and write its store YAML.

    # interactive (OpenCV window): a still, a video file (first frame) or an RTSP URL (one frame)
    PYTHONPATH=src .venv/bin/python scripts/draw_zones.py --source rtsp://user:pass@cam/stream1 \\
        --out configs/store1_register_cam.yaml --camera-id register_cam --preview zones.png
    # headless: polygons from JSON {"zones": [{"name", "kind", "polygon": [[x, y], ...]}],
    #                               "floor_points": [[x_px, y_px, x_m, y_m], ...]}   (floor optional)
    PYTHONPATH=src .venv/bin/python scripts/draw_zones.py --from-json zones.json --source still.png \\
        --out configs/store1_register_cam.yaml --preview zones.png

Window keys: left click = add a point (printed); Enter = close the polygon (then type its name and
kind in the terminal); u = undo the last point; f = floor-point mode on/off (each click then asks
for the spot's floor x y in metres); d = delete the last zone; q = save and quit; Esc = quit
without saving.

Everything but the zones, camera id/resolution and floor points (catalog, terminals,
product_classes, rules, ledger) is kept from --template (default: --out if it exists, else
configs/store_gas_station_small.yaml). The written file is loaded back with load_store_config.
No frame is written to disk except the --preview you ask for; it is the raw frame (people in it
are NOT pixelated), so keep it out of shared folders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from bree.events.zones import ZONE_KINDS, Zone, load_store_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "configs" / "store_gas_station_small.yaml"


def grab_frame(source: str) -> np.ndarray:
    img = cv2.imread(source) if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp") else None
    if img is not None:
        return img
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    frame = None
    for _ in range(5):                 # RTSP: the first frames can be grey before a keyframe
        ok, img = cap.read()
        if ok:
            frame = img
    cap.release()
    if frame is None:
        raise SystemExit("cannot read a frame from --source")   # not the URL: it may hold credentials
    return frame


def build_yaml(template: dict, zones: list[dict], floor_points: list | None, camera_id: str | None,
               resolution: tuple[int, int] | None) -> dict:
    out = dict(template)
    cam = dict(out.get("camera") or {})
    if camera_id:
        cam["id"] = camera_id
    if resolution:
        cam["resolution"] = list(resolution)
    cam.pop("floor_points", None)      # per camera pixels: never inherit another camera's marks
    if floor_points:
        cam["floor_points"] = [[float(v) for v in p] for p in floor_points]
    out["camera"] = cam
    px = lambda v: int(v) if float(v).is_integer() else round(float(v), 1)   # noqa: E731
    out["zones"] = [{"name": z["name"], "kind": z["kind"], "polygon": [[px(x), px(y)] for x, y in z["polygon"]]}
                    for z in zones]
    return out


def validate(zones: list[dict], floor_points: list | None, terminals: dict) -> list[str]:
    """Errors that make the YAML unusable; terminals pointing at missing zones are warnings."""
    errs, names = [], [z["name"] for z in zones]
    for z in zones:
        if z["kind"] not in ZONE_KINDS:
            errs.append(f"zone {z['name']!r}: kind must be one of {ZONE_KINDS}")
        if len(z["polygon"]) < 3:
            errs.append(f"zone {z['name']!r}: polygon needs 3+ points")
    errs += [f"zone name {n!r} used twice" for n in sorted({n for n in names if names.count(n) > 1})]
    if floor_points and (len(floor_points) < 4 or any(len(p) != 4 for p in floor_points)):
        errs.append("floor_points needs 4+ [x_px, y_px, x_m, y_m] rows")
    for term, zone in (terminals or {}).items():
        if zone not in names:
            print(f"warning: terminal {term!r} is in zone {zone!r}, which this camera does not have "
                  "(fine if another camera of the store sees it)", file=sys.stderr)
    return errs


def render(img: np.ndarray, zones: list[dict], floor_points: list | None = None,
           current: list | None = None) -> np.ndarray:
    from bree.alerts.annotate import draw_zones
    out = img.copy()
    draw_zones(out, SimpleNamespace(zones=[Zone(z["name"], z["kind"], np.asarray(z["polygon"], float))
                                           for z in zones if len(z["polygon"]) >= 3]))
    for z in zones:
        cv2.polylines(out, [np.asarray(z["polygon"], np.int32)], True, (255, 255, 255), 1)
    for x, y, xm, ym in floor_points or []:
        cv2.drawMarker(out, (int(x), int(y)), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(out, f"({xm:g},{ym:g})m", (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    if current:
        pts = np.asarray(current, np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(out, tuple(int(v) for v in p), 3, (0, 255, 255), -1)
    return out


def interactive(img: np.ndarray, zones: list[dict], floor_points: list) -> bool:
    """Edit `zones` / `floor_points` in place. True = save."""
    state = {"pts": [], "floor": False}

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if state["floor"]:
            state["ask"] = (x, y)             # asked in the main loop, not inside the GUI callback
        else:
            state["pts"].append([x, y])
            print(f"  point ({x}, {y})")

    win = "draw_zones (Enter close polygon, u undo, f floor mode, d delete zone, q save, Esc quit)"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        raise SystemExit("this OpenCV build has no GUI (opencv-python-headless?): use --from-json")
    cv2.setMouseCallback(win, on_mouse)
    print(f"zone kinds: {', '.join(ZONE_KINDS)}")
    while True:
        cv2.imshow(win, render(img, zones, floor_points, state["pts"]))
        k = cv2.waitKey(30) & 0xFF
        if state.get("ask"):
            x, y = state.pop("ask")
            try:
                xm, ym = (float(v) for v in input(f"floor point at pixel ({x}, {y}): x_m y_m = ").split())
                floor_points.append([x, y, xm, ym])
            except ValueError:
                print("  need two numbers, point ignored")
        if k in (13, 10) and len(state["pts"]) >= 3:
            name = input("zone name: ").strip()
            kind = input(f"kind ({'/'.join(ZONE_KINDS)}): ").strip()
            if name and kind in ZONE_KINDS:
                zones[:] = [z for z in zones if z["name"] != name] + [
                    {"name": name, "kind": kind, "polygon": state["pts"]}]
                print(f"zone {name} ({kind}): {state['pts']}")
                state["pts"] = []
            else:
                print("  need a name and a valid kind; keep clicking or press Enter again")
        elif k == ord("u") and state["pts"]:
            state["pts"].pop()
        elif k == ord("d") and zones:
            print(f"deleted zone {zones.pop()['name']}")
        elif k == ord("f"):
            state["floor"] = not state["floor"]
            print(f"floor-point mode {'on' if state['floor'] else 'off'} ({len(floor_points)} marked)")
        elif k == ord("q"):
            cv2.destroyAllWindows()
            return True
        elif k == 27:
            cv2.destroyAllWindows()
            return False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="image, video file, webcam index or rtsp:// URL (one frame is read)")
    ap.add_argument("--out", required=True, help="store YAML to write (updated in place if it exists)")
    ap.add_argument("--template", help="YAML to keep catalog/terminals/rules/... from (default: --out if it exists)")
    ap.add_argument("--camera-id", help="camera.id (default: the template's)")
    ap.add_argument("--from-json", help="zones (+ optional floor_points) as JSON instead of clicking")
    ap.add_argument("--keep-zones", action="store_true", help="start from the template's zones (same name replaces)")
    ap.add_argument("--preview", help="write the frame with the zones drawn on it to this image")
    args = ap.parse_args(argv)

    if not args.from_json and not args.source:
        ap.error("--source is required unless --from-json is given")
    out = Path(args.out)
    template_path = Path(args.template) if args.template else (out if out.exists() else DEFAULT_TEMPLATE)
    template = yaml.safe_load(template_path.read_text()) or {}
    zones = [dict(z) for z in template.get("zones", [])] if args.keep_zones else []
    floor = list((template.get("camera") or {}).get("floor_points") or []) if args.keep_zones else []
    img = grab_frame(args.source) if args.source else None

    if args.from_json:
        given = json.loads(Path(args.from_json).read_text())
        names = {z["name"] for z in given.get("zones", [])}
        zones = [z for z in zones if z["name"] not in names] + list(given.get("zones", []))
        floor = given.get("floor_points", floor)
    elif not interactive(img, zones, floor):
        print("not saved")
        return

    errs = validate(zones, floor, template.get("terminals", {}))
    if errs:
        raise SystemExit("not saved:\n  " + "\n  ".join(errs))
    res = (img.shape[1], img.shape[0]) if img is not None else None
    data = build_yaml(template, zones, floor, args.camera_id, res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Zones drawn with scripts/draw_zones.py (template: {template_path.name}).\n"
                   + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=110))
    store = load_store_config(out)               # the pipeline must accept what we wrote
    if store.floor_points:
        store.floor_homography()
    print(f"wrote {out}: {len(store.zones)} zone(s), "
          f"{len(store.floor_points or [])} floor point(s), resolution {list(store.resolution)}")
    if args.preview:
        if img is None:
            raise SystemExit("--preview needs --source (the frame to draw on)")
        prev = render(img, data["zones"], (data["camera"].get("floor_points")))
        if not cv2.imwrite(args.preview, prev):
            raise SystemExit(f"could not write {args.preview}")
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
