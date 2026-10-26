"""Draft per-camera store configs and zone templates for a human to correct (never used blindly).

Output is a normal store YAML (loads with bree.events.zones.load_store_config) with `camera.role` and
`camera.draft: true`. Zones come from simple, inspectable cues and fall back to fixed templates:

  door      exit zone = the largest bright region in the upper part of the median frame (glass doors) plus a strip of
            floor below it; fallback: lower-left corner strip
  register  register zone = box around the fixed counter equipment (screens, keyboards, terminals) grown 25%
  cooler    cooler zone = box around refrigerator detections, grown 8%; fallback: top band
  aisle     two shelf zones = side bands of the frame (template only: aisles are not found, they are assumed)
  others    no zones (outdoor / obscured / backroom / unknown cameras do not feed the ledger)

Coordinates are pixels of `size`, the resolution of the stream the camera will be run on. The zones are NOT
validated against anything: correct them with scripts/draw_zones.py on the saved still.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

CAMERA_NOTES = {
    "door": "entrance/exit view: use for traffic counts, enter/exit events and door-to-register timing",
    "register": "counter view: register activity, queue and handover; body-based theft cues are weak from here",
    "cooler": "cooler doors: reach/pick near the doors, door-open and shelf-state cues",
    "aisle": "shelf aisle: traffic, dwell and shelf state; no entrance here, so no enter events from this camera",
    "outdoor": "outdoor: not used for in-store events (parking/pump traffic only)",
    "obscured": "picture unusable (blurred, covered or dark): fix the camera before using it",
    "backroom": "staff area: keep out of customer analytics",
    "unknown": "role not determined: look at the still and set it with `bree fleet set-role`",
}


def _poly(x0, y0, x1, y1, size) -> list[list[int]]:
    w, h = size
    x0, x1 = sorted((min(max(x0, 0), 1), min(max(x1, 0), 1)))
    y0, y1 = sorted((min(max(y0, 0), 1), min(max(y1, 0), 1)))
    return [[round(x0 * w), round(y0 * h)], [round(x1 * w), round(y0 * h)], [round(x1 * w), round(y1 * h)], [round(x0 * w), round(y1 * h)]]


def _union(boxes, grow: float):
    b = np.array(boxes, dtype=float)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    gw, gh = (x1 - x0) * grow, (y1 - y0) * grow
    return x0 - gw, y0 - gh, x1 + gw, y1 + gh


def bright_region(frames: list[np.ndarray]) -> tuple[float, float, float, float] | None:
    """Normalised box of the largest bright blob in the top 60% of the median frame (glass doors in daylight)."""
    if not frames:
        return None
    small = [cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA) for f in frames]
    med = np.median(np.stack(small), axis=0).astype(np.uint8)
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = (gray > 190).astype(np.uint8)
    mask[int(h * 0.6):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    if area < 0.03 * w * h:
        return None
    return x / w, y / h, (x + bw) / w, (y + bh) / h


def zones_for(role: str, size: tuple[int, int], dets: list[list], frames: list[np.ndarray]) -> tuple[list[dict], list[str]]:
    """(zones, notes on where each came from)."""
    zones, notes = [], []
    flat = [d for f in dets for d in f]
    if role == "door":
        r = bright_region(frames)
        if r:
            zones.append({"name": "door", "kind": "exit", "polygon": _poly(r[0], r[1], r[2], min(r[3] + 0.18, 1.0), size)})
            notes.append("door zone: largest bright region (glass) in the upper frame plus 18% of the height below it")
        else:
            zones.append({"name": "door", "kind": "exit", "polygon": _poly(0.0, 0.55, 0.3, 1.0, size)})
            notes.append("door zone: TEMPLATE (lower-left strip), no bright glass region found")
    elif role == "register":
        gear = [d.box for d in flat if d.name in ("keyboard", "laptop", "tv", "cell phone", "mouse") and d.conf >= 0.3]
        if gear:
            zones.append({"name": "register", "kind": "register", "polygon": _poly(*_union(gear, 0.25), size)})
            notes.append(f"register zone: box around {len(gear)} equipment detections, grown 25%")
        else:
            zones.append({"name": "register", "kind": "register", "polygon": _poly(0.5, 0.4, 1.0, 1.0, size)})
            notes.append("register zone: TEMPLATE (lower right)")
    elif role == "cooler":
        fr = [d.box for d in flat if d.name == "refrigerator" and d.conf >= 0.3]
        if fr:
            zones.append({"name": "cooler_bank", "kind": "cooler", "polygon": _poly(*_union(fr, 0.08), size)})
            notes.append(f"cooler zone: box around {len(fr)} refrigerator detections, grown 8%")
        else:
            zones.append({"name": "cooler_bank", "kind": "cooler", "polygon": _poly(0.0, 0.0, 1.0, 0.4, size)})
            notes.append("cooler zone: TEMPLATE (top band)")
    elif role == "aisle":
        zones.append({"name": "shelf_left", "kind": "shelf", "polygon": _poly(0.0, 0.1, 0.3, 0.95, size)})
        zones.append({"name": "shelf_right", "kind": "shelf", "polygon": _poly(0.7, 0.1, 1.0, 0.95, size)})
        notes.append("shelf zones: TEMPLATE side bands (aisle walls are assumed at the left and right edges)")
    return zones, notes


def store_yaml(station_id: str, serial: str, channel: int, role: str, size: tuple[int, int], zones: list[dict],
               notes: list[str], evidence: list[str], fps: float = 15.0, needs_review: bool = True) -> str:
    head = [f"# DRAFT camera config written by `bree onboard` for station {station_id}, channel {channel}.",
            f"# Role suggestion: {role}{' (NEEDS REVIEW)' if needs_review else ''}: {CAMERA_NOTES.get(role, '')}",
            *[f"#   evidence: {e}" for e in evidence], *[f"#   {n}" for n in notes],
            "# Zones are guesses from simple cues. Correct them on the saved still with scripts/draw_zones.py before",
            "# trusting any event from this camera. Remove `draft: true` once a person has checked it.", ""]
    doc = {"store": {"name": f"{station_id} ch{channel:02d}"},
           "camera": {"id": f"{station_id}_ch{channel:02d}", "role": role, "draft": True, "serial": serial, "channel": channel,
                      "resolution": [int(size[0]), int(size[1])], "fps": float(round(fps, 1))},
           "zones": zones, "terminals": {}, "product_classes": {}, "catalog": {}}
    return "\n".join(head) + yaml.safe_dump(doc, sort_keys=False)


def shadow_yaml(station_id: str, cameras: list[tuple[int, str, str]], out_dir: str) -> str:
    """cameras: (channel, source url, store yaml path). Credentials stay in the env file, not here."""
    lines = [f"# DRAFT shadow-mode config for station {station_id} (written by `bree onboard`).",
             "# Credentials: IMOU_APP_ID / IMOU_APP_SECRET come from ~/.config/bree/imou.env, never from this file.",
             "# Add pos_export_dir / pos_mapping once the POS export is known (docs/PILOT_RUNBOOK.md).", "cameras:"]
    for ch, url, store in cameras:
        lines += [f"  - name: {station_id}_ch{ch:02d}", f"    rtsp_url: \"{url}\"", f"    store: {store}"]
    lines += [f"output_dir: {out_dir}", "backend: yolo", "runtime: pytorch", "review_port: 0", ""]
    return "\n".join(lines)
