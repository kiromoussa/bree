"""Tools for making REAL-MODEL-LABELLED door-event labels from footage, locally (no images leave the data directory).

`door_activity` finds the moments when anything moves around the door, independent of the person detector (so a person the
detector misses can still be labelled), and `contact_sheet` tiles door-area crops of consecutive moments so a labeller
(a person or a vision model looking at the sheet) can write down who crossed the door threshold and in which direction.
The sheets contain people: keep them in the data directory, never in git or docs.
"""
from __future__ import annotations

import cv2
import numpy as np


def door_activity(video: str, roi: tuple[float, float, float, float], fps_out: float = 2.0, thresh: float = 1.2,
                  pad_s: float = 4.0, merge_s: float = 6.0) -> list[tuple[float, float]]:
    """Intervals (seconds) where the ROI (normalised x0,y0,x1,y1) changes. Frame difference at `fps_out`, against the previous
    sampled frame, in grey 160 px wide; `thresh` is the mean absolute difference in grey levels."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / fps_out)), 1)
    prev, act, i = None, [], 0
    while cap.grab():
        if i % step == 0:
            ok, im = cap.retrieve()
            h, w = im.shape[:2]
            x0, y0, x1, y1 = roi
            g = cv2.cvtColor(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(cv2.resize(g, (160, int(160 * g.shape[0] / g.shape[1]))), (5, 5), 0).astype(np.float32)
            if prev is not None:
                act.append((i / fps, float(np.abs(g - prev).mean())))
            prev = g
        i += 1
    cap.release()
    spans: list[list[float]] = []
    for t, a in act:
        if a >= thresh:
            if spans and t - pad_s - spans[-1][1] <= merge_s:
                spans[-1][1] = t + pad_s
            else:
                spans.append([max(t - pad_s, 0.0), t + pad_s])
    return [(a, b) for a, b in spans]


def contact_sheet(video: str, t0: float, t1: float, roi: tuple[float, float, float, float], per_s: float = 1.0,
                  cols: int = 5, cell=(400, 250), stamp: bool = True) -> list[np.ndarray]:
    """Sheets of ROI crops from t0 to t1, `per_s` crops per second, `cols` x 4 cells per sheet, each cell stamped with its time."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))       # one seek, then read forward (seeking per cell is slow on HEVC)
    want = {int(round((t0 + k / per_s) * fps)): t0 + k / per_s for k in range(int((t1 - t0) * per_s) + 1)}
    i, last = int(t0 * fps), max(want)
    cells = []
    while i <= last and cap.grab():
        if i in want:
            ok, im = cap.retrieve()
            if not ok:
                break
            h, w = im.shape[:2]
            x0, y0, x1, y1 = roi
            c = cv2.resize(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cell)
            if stamp:
                cv2.putText(c, f"{want[i]:6.1f}s", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cells.append(c)
        i += 1
    cap.release()
    if not cells:
        return []
    per = cols * 4
    sheets = []
    for s in range(0, len(cells), per):
        chunk = cells[s:s + per]
        chunk += [np.zeros_like(cells[0])] * (per - len(chunk))
        rows = [np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(4)]
        sheets.append(np.vstack(rows))
    return sheets
