"""The DVR's burned-in clock as a common time base for all channels.

All channels of one DVR show the same wall clock in the video ("10-25-2026 16:49:09"), but the recorder starts
each channel's stream at its own moment (HLS start latency differs by seconds between channels), so the file names
cannot be used to line cameras up to the second. Reading the overlay of a few frames of every segment gives, per
segment, the offset between the DVR clock and the recorder clock (the file name plus the frame time).

OCR is Tesseract (the system binary, called as a program; nothing is installed or sent anywhere) on the cropped,
upscaled, thresholded overlay. A reading is accepted only when several frames of one segment agree on the offset
to within a second. Cameras whose overlay is elsewhere pass their own crop box (fractions of the frame).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

OVERLAY_BOX = (0.72, 0.03, 0.98, 0.09)      # x0, y0, x1, y1 as fractions of the frame: top right of the indoor channels
PATTERN = re.compile(r"(\d{2})-(\d{2})-(\d{4})\s*(\d{2}):(\d{2}):(\d{2})")


def read_overlay(frame: np.ndarray, box=OVERLAY_BOX) -> datetime | None:
    import cv2
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, m = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)          # the text is white
    m = 255 - m                                                    # tesseract wants dark text on light
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "o.png"
        cv2.imwrite(str(f), m)
        out = subprocess.run(["tesseract", str(f), "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789-: "],
                             capture_output=True, text=True, timeout=20).stdout
    mt = PATTERN.search(out.replace("\n", " "))
    if not mt:
        return None
    mo, d_, y, H, M, S = (int(x) for x in mt.groups())
    try:
        return datetime(y, mo, d_, H, M, S)
    except ValueError:
        return None


def segment_offset(path: Path, n: int = 7, box=OVERLAY_BOX, tol_s: float = 1.5) -> tuple[float, int, float] | None:
    """(DVR clock minus recorder clock in seconds, readings used, spread) for one segment, from `n` frames spread
    over it; None when fewer than 3 readings agree. Recorder clock = file name time + frame time."""
    import cv2
    t0 = datetime.strptime(path.stem, "%Y%m%d_%H%M%S").timestamp()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.3
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    offs = []
    for i in np.linspace(int(nf * 0.05), int(nf * 0.95), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if not ok:
            continue
        dt = read_overlay(im, box)
        if dt is not None:
            offs.append(dt.timestamp() - (t0 + i / fps))
    cap.release()
    if len(offs) < 3:
        return None
    med = float(np.median(offs))
    good = [o for o in offs if abs(o - med) <= tol_s]
    if len(good) < 3:
        return None
    return float(np.median(good)), len(good), float(np.std(good))
