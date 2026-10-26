"""Debug and evidence frames from a video segment plus its perception cache. Heads are pixelated in everything written.

`debug_frame` shows what the gate did: green = passed to the event engine, red = hidden (reason in the label),
grey = still pending. Zones outlined. Meant for looking at your own outputs; it is not an alert clip.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from bree.alerts.annotate import blur_heads, draw_zones
from bree.theft.cache import load_frames, load_info
from bree.theft.gate import GateConfig, ObsGate
from bree.theft.station import Station

COL = {"pass": (60, 200, 60), "pending": (170, 170, 170)}


def read_frame(video: str | Path, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def debug_frame(video: str | Path, cache_dir: str | Path, station: Station, t: float, width: int = 1280) -> np.ndarray:
    gate = ObsGate(GateConfig(**station.gate), station.staff, station.ignore)
    persons = []
    for obs in load_frames(cache_dir):
        gate.update(obs)
        persons = obs.persons
        if obs.t >= t:
            break
    img = read_frame(video, t)
    if img is None:
        raise IOError(f"no frame at {t}s in {video}")
    blur_heads(img, persons)
    draw_zones(img, station.store)
    for z in station.staff + station.ignore:
        cv2.polylines(img, [z.polygon.astype(np.int32)], True, (0, 140, 255), 3)
    for p in persons:
        st = gate.status(p.track_id)
        c = COL.get(st, (40, 40, 230))
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
        cv2.putText(img, f"{p.track_id} {st} {p.conf:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)
    s = width / img.shape[1]
    return cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
