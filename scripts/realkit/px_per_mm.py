#!/usr/bin/env python
"""How many pixels does this camera put on a product? One frame, a ruler, two clicks.

    # click the two ends of something whose width you measured (a can is 66 mm, a US letter page is 216 mm wide)
    .venv/bin/python scripts/realkit/px_per_mm.py frame.jpg --mm 66
    # a video works too (the frame 2 s in is used), and so do typed points when there is no screen
    .venv/bin/python scripts/realkit/px_per_mm.py railA.mp4 --mm 216 --points 812,1040,1203,1051

It prints pixels per millimetre, how many pixels that puts across a 66 mm can, and how that compares with the two
numbers BREE plans with: 20 px across a can (0.3 px per mm, the layout research threshold) and 1.5 px per mm (what
PepsiCo's cooler patent says is needed to tell flavour variants apart; 2 px per mm to read text).

Measure where the product stands, not at the camera: put the ruler or the page on the shelf edge at the slot you
care about, flat against the shelf front. Do it at the nearest slot and at the farthest one.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

CAN_MM = 66.0            # width of a 12 oz can
WORKING_PX = 20.0        # BREE's working threshold: pixels across a can (research/camera-layouts.md)
VARIANT_PX_PER_MM = 1.5  # tell flavour variants apart (research/patents-technical.md, PepsiCo US 12,688,510)
TEXT_PX_PER_MM = 2.0     # read label text (same source)


def px_per_mm(p1, p2, mm: float) -> float:
    if mm <= 0:
        raise ValueError("the measured width must be over 0 mm")
    return math.dist(p1, p2) / mm


def verdict(ppm: float) -> list[str]:
    """Plain lines for a pixels-per-millimetre figure."""
    can = ppm * CAN_MM
    out = [f"{ppm:.2f} px per mm, so a 66 mm can is {can:.0f} px across."]
    out.append(f"Working threshold (20 px across a can, 0.30 px per mm): {'PASS' if can >= WORKING_PX else 'FAIL'}, {can / WORKING_PX:.1f} times the threshold.")
    out.append(f"Flavour variants (1.5 px per mm, 99 px across a can): {'PASS' if ppm >= VARIANT_PX_PER_MM else 'FAIL'}, {ppm / VARIANT_PX_PER_MM:.0%} of what is needed.")
    out.append(f"Label text (2 px per mm): {'PASS' if ppm >= TEXT_PX_PER_MM else 'FAIL'}.")
    if can < WORKING_PX:
        out.append("Too few pixels even for the working threshold: move the camera closer, use a longer lens, or record at a higher resolution.")
    elif ppm < VARIANT_PX_PER_MM:
        out.append(f"Enough to see that an item is there. To tell look-alike products apart from the picture you need {VARIANT_PX_PER_MM / ppm:.1f} times more pixels: "
                   "a longer lens (the High Quality Camera), a higher recording resolution, or a shorter distance.")
    else:
        out.append("Enough pixels to tell look-alike products apart, if the picture is also sharp and not blurred by motion.")
    return out


def first_frame(path: str, at_s: float = 2.0):
    """A picture file, or the frame `at_s` seconds into a video."""
    import cv2
    img = cv2.imread(path)
    if img is not None:
        return img
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, img = cap.read()
    if not ok:                      # shorter than at_s: take the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read a picture from {path}")
    return img


def click(img, prompts: list[str], max_side: int = 1000) -> list[tuple[float, float] | None]:
    """Show `img` and ask for one click per prompt. Returns full-resolution pixels, None for a skipped prompt.
    Keys: u undo, s skip this point (not visible), q or Esc give up."""
    import cv2
    k = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img.copy()
    pts: list[tuple[float, float] | None] = []
    win = "click (u undo, s skip, q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: pts.append((x / k, y / k)) if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(prompts) else None)
    while len(pts) < len(prompts):
        view = small.copy()
        for i, p in enumerate(pts):
            if p:
                cv2.circle(view, (int(p[0] * k), int(p[1] * k)), 5, (0, 255, 0), -1)
                cv2.putText(view, str(i + 1), (int(p[0] * k) + 7, int(p[1] * k) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        text = f"{len(pts) + 1}/{len(prompts)}: {prompts[len(pts)]}"
        cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("s"):
            pts.append(None)
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("stopped, nothing saved")
    cv2.destroyAllWindows()
    return pts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", help="a picture, or a video (the frame 2 s in is used)")
    ap.add_argument("--mm", type=float, help="the real width between the two points, millimetres")
    ap.add_argument("--inches", type=float, help="the same in inches")
    ap.add_argument("--points", help="x1,y1,x2,y2 in pixels of the picture, instead of clicking")
    a = ap.parse_args(argv)
    mm = a.mm if a.mm else a.inches * 25.4 if a.inches else None
    if not mm:
        ap.error("give the measured width: --mm 66 or --inches 8.5")
    if a.points:
        v = [float(x) for x in a.points.split(",")]
        if len(v) != 4:
            ap.error("--points needs four numbers: x1,y1,x2,y2")
        p1, p2 = v[:2], v[2:]
    else:
        p1, p2 = click(first_frame(a.frame), ["one end of the measured width", "the other end"])
        if p1 is None or p2 is None:
            return 1
    print(f"{Path(a.frame).name}: {math.dist(p1, p2):.0f} px over {mm:.0f} mm")
    print("\n".join(verdict(px_per_mm(p1, p2, mm))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
