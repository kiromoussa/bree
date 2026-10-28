"""Synchronised frame sheets for looking at (and labelling) several cameras at one moment.

A sheet is one picture: the frames of every camera at one store-clock time, each downsized, with the tracklet boxes
drawn and numbered ("<tracklet id>"), so a person can say which numbered box in which camera is the same person.
Sheets contain people: they are written under the data directory only, never into git, never sent anywhere.
Heads are pixelated by default (bree's own rule for anything saved); labellers work from clothing and position.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bree.fusion.tracklets import Tracklet, SEGMENT_FMT


def frame_at(station: Path, cam: str, t_store: float, offsets: dict[str, float] | None = None, width: int = 960):
    """The frame of camera `cam` nearest to DVR-clock time t_store (epoch seconds), shrunk to `width`. `offsets` maps a
    segment name (file stem) to the seconds that put its recorder clock on the DVR clock (clock_offsets.json).
    (None, 1.0) when no recorded segment covers it."""
    import cv2
    from datetime import datetime
    for p in sorted((station / f"ch{cam}").glob("*.mp4")):
        if p.stat().st_size < 1e6:
            continue
        t0 = datetime.strptime(p.stem, SEGMENT_FMT).timestamp() + (offsets or {}).get(p.stem, 0.0)
        if t0 <= t_store < t0 + 299.5:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.3
            if t_store - t0 > cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps:
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t_store - t0) * fps)))
            ok, im = cap.read()
            cap.release()
            if not ok:
                return None, 1.0
            s = width / im.shape[1]
            return cv2.resize(im, (width, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA), s
    return None, 1.0


def pixelate_boxes(im: np.ndarray, boxes, scale: float, frac: float = 0.2, block: int = 8) -> None:
    """Pixelate the head region (top `frac` of each person box) in place."""
    import cv2
    for b in boxes:
        x0, y0, x1, y1 = [int(v * scale) for v in b]
        y1h = y0 + int((y1 - y0) * frac)
        x0, y0 = max(x0, 0), max(y0, 0)
        roi = im[y0:y1h, x0:x1]
        if roi.size == 0:
            continue
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(w // block, 1), max(h // block, 1)), interpolation=cv2.INTER_LINEAR)
        im[y0:y1h, x0:x1] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def sheet(station: Path, t_store: float, cams: list[str], offsets: dict[str, dict[str, float]], tracklets: dict[str, list[Tracklet]],
          width: int = 640, cols: int = 3, labels: dict[tuple[str, int], str] | None = None, pixelate: bool = True) -> np.ndarray:
    import cv2
    tiles = []
    for cam in cams:
        im, s = frame_at(station, cam, t_store, offsets.get(cam), width)
        if im is None:
            im = np.zeros((int(width * 9 / 16), width, 3), np.uint8)
            s = 1.0
        boxes = []
        t_cam = t_store                       # tracklets are already on the DVR clock
        for tl in tracklets.get(cam, []):
            i = tl.at(t_cam, 0.25)
            if i is not None:
                boxes.append((tl, tl.box[i]))
        if pixelate:
            pixelate_boxes(im, [b for _, b in boxes], s * (width / width))
        for tl, b in boxes:
            x0, y0, x1, y1 = [int(v * s) for v in b]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), 2)
            lab = (labels or {}).get((cam, tl.id), "")
            cv2.putText(im, f"{tl.id}{(' ' + lab) if lab else ''}", (x0 + 2, max(y0 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"ch{cam}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(im)
    h = max(t.shape[0] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in row]))
    return np.vstack(rows)
