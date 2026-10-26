"""Evidence clips for review items, cut from a real segment with every detected head pixelated.

Privacy rules (same as bree.alerts): heads come from pose keypoints (nose / eyes / ears), falling back to the top of the
person box, and are pixelated in 5x5 blocks BEFORE the frame is scaled down or written. The pixelation uses every person
box in the perception cache (also the ones the gate hid), so a ghost that was actually a person is still covered. What
it cannot cover is a person the detector missed: clips are small (default 854 px wide) so a missed face is a few pixels
across. The raw segment is never copied. Clips go in an `alerts/` folder because that is the only place the review page
serves video from.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from bree.alerts.annotate import blur_heads, draw_zones
from bree.theft.cache import load_frames, load_info
from bree.theft.station import Station


def write_clip(video: str | Path, cache_dir: str | Path, out_path: str | Path, t0: float, t1: float,
               station: Station | None = None, label: str = "", width: int = 854, fps: float = 10.0) -> Path | None:
    info = load_info(cache_dir)
    t0, t1 = max(0.0, t0), min(info.duration_s, t1)
    if t1 <= t0:
        return None
    persons_at = {}
    for obs in load_frames(cache_dir):
        if obs.t > t1:
            break
        if obs.t >= t0:
            persons_at[obs.frame] = obs.persons
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    src_fps = info.fps
    i0 = int(t0 * src_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0 - (i0 % info.stride))      # land on a cached frame
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = int(info.height * width / info.width)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, h))
    i = i0 - (i0 % info.stride)
    n = 0
    while True:
        ok, img = cap.read()
        if not ok or i / src_fps > t1:
            break
        if i in persons_at:
            blur_heads(img, persons_at[i])
            if station is not None:
                draw_zones(img, station.store)
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{label}  t={i / src_fps:.0f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vw.write(img)
            n += 1
        i += 1
    cap.release()
    vw.release()
    return out if n else None
