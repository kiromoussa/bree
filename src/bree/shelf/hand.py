"""Hand and held-item cue on an item camera, without a person box.

Gate: the pixels that differ from the shelf picture (bree.shelf.diff.ShelfDiff.changed, the same idea as the camera
node's trigger in bree.edge.trigger) and moved in the last frames. Only those regions are looked at.

  held item   the SKU detector (bree.train.backend.SkuDetector, sim-trained) on tiles around the moving regions; a
              detection counts as "in a hand" when its box is mostly changed pixels and something in it moved just now
              (stock on the shelf is neither). Detections of one SKU close in time and place form a track.
  hand        detector weights with a "hand" class (SkuDetector.hand_cls): its hand boxes are used. Otherwise a
              callable(image, changed_mask, scale) -> [(u, v, conf)] in full-resolution pixels. The default,
              `skin_hand`, is a skin-coloured moving blob (the fixed YCrCb box of bree.edge.trigger). It is weak on
              purpose: pass a real hand detector (a hand class, or wrist keypoints from a pose model run on the moving
              crop) as `hand=` to ShelfCamera and nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bree.edge.trigger import _SKIN_HI, _SKIN_LO
from bree.shelf.diff import slot_sigma


@dataclass
class HandConfig:
    every: int = 2               # run the SKU detector every this many frames (when something moved)
    conf: float = 0.3
    min_fg_px: int = 150         # changed region (working scale) worth a look
    roi_margin_px: int = 48      # full-resolution pixels around a changed region
    fg_frac: float = 0.4         # share of a detection box that differs from the shelf picture
    moving_frames: int = 3       # something in the box moved within this many frames
    linger_s: float = 1.0        # keep looking at a region this long after the last motion in it
    track_gap_s: float = 1.0
    track_px: float = 350.0      # full-resolution pixels between two sightings of one held item
    min_travel_px: float = 60.0  # a held item moves at least this far while it is seen
    min_obs: int = 3             # sightings before a track can be an event on its own
    near_slot: float = 2.5       # hand-only event: the track starts (take) or ends (put) within this many slot-box
    near_slot_min_px: float = 70.0   # diagonals of a slot of its SKU (and at least this many pixels)
    away: float = 1.5            # ... and is this many times farther from it at its other end
    skin_min_px: int = 12        # skin blob, working scale
    skin_max_px: int = 4000


def skin_hand(image: np.ndarray, changed: np.ndarray, scale: float, cfg: HandConfig | None = None) -> list[tuple[float, float, float]]:
    """Skin-coloured blobs inside the changed mask: [(u, v, conf)] in full-resolution pixels."""
    cfg = cfg or HandConfig()
    small = cv2.resize(image, (changed.shape[1], changed.shape[0]), interpolation=cv2.INTER_AREA)
    skin = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), _SKIN_LO, _SKIN_HI) & (changed * 255)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(skin, connectivity=8)
    return [(float(cent[k][0] / scale), float(cent[k][1] / scale), 0.5) for k in range(1, n) if cfg.skin_min_px <= stats[k][4] <= cfg.skin_max_px]


def sku_detector(weights=None, conf: float = 0.3):
    import torch
    from bree.train.backend import SkuDetector
    return SkuDetector(weights, device="mps" if torch.backends.mps.is_available() else "cpu", conf=conf)


class HandItemCue:
    """Keeps `tracks` (held items: {"sku", "obs": [(frame, u, v, conf)]}) and `hands` [(frame, u, v, conf)].
    diff: the camera's ShelfDiff (its masks are the gate, its slot boxes place a hand-only event)."""

    def __init__(self, diff, detector=None, hand=None, cfg: HandConfig | None = None):
        self.diff, self.det, self.cfg = diff, detector, cfg or HandConfig()
        self.hand = hand if hand is not None else (lambda im, ch, sc: skin_hand(im, ch, sc, self.cfg))
        self._hand_given, self._linger = hand is not None, None
        self.stock = None          # callable(sku, box) -> bool: the detection is shelf stock (set by ShelfCamera)
        self.last: tuple[int, np.ndarray, list[str]] | None = None      # (frame, item boxes, SKU names) of this frame's look
        self.tracks: list[dict] = []
        self.hands: list[tuple[int, float, float, float]] = []
        self.detector_frames = self.detector_tiles = 0
        self._by_sku: dict[str, list[int]] = {}
        for i, s in enumerate(diff.slots):
            if diff.readable[i] or diff.area[i] > 0:
                self._by_sku.setdefault(s.get("skuId"), []).append(i)

    def update(self, image: np.ndarray, f: int) -> None:
        c, d = self.cfg, self.diff
        self.last = None
        if d.changed is None or f % c.every:
            return
        sc = d.cfg.scale
        recent = (d.still < c.moving_frames).astype(np.uint8)
        rois, n = [], 0
        if recent.any() and int(d.changed.sum()) >= c.min_fg_px:
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(d.changed, np.ones((9, 9), np.uint8)), connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area >= c.min_fg_px and recent[y:y + h, x:x + w].any():
                rois.append([x / sc - c.roi_margin_px, y / sc - c.roi_margin_px, (x + w) / sc + c.roi_margin_px, (y + h) / sc + c.roi_margin_px])
        if rois:
            self._linger = (rois, f + int(c.linger_s * d.fps))
        elif self._linger and f <= self._linger[1]:
            rois = self._linger[0]         # the hand just left: look once more at where it was (the shelf behind it)
        else:
            return
        hand_cls = getattr(self.det, "hand_cls", None)
        if hand_cls is None or self._hand_given:
            for u, v, cf in self.hand(image, d.changed & recent, sc):
                self.hands.append((f, u, v, cf))
        if self.det is None:
            return
        rois = np.asarray(rois, float)
        self.detector_frames += 1
        self.detector_tiles += len(self.det.tiles(image.shape[0], image.shape[1], rois)) if hasattr(self.det, "tiles") else 0
        if hand_cls is not None:           # detector weights with a hand class: its hand boxes are the hand cue
            (boxes, confs, cls), (hb, hc) = self.det.detect(image, rois)
            if not self._hand_given:
                self.hands += [(f, float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2, float(cf)) for b, cf in zip(hb, hc)]
        else:
            boxes, confs, cls = self.det(image, rois)
        names = getattr(self.det, "names", None)
        self.last = (f, np.asarray(boxes, float).reshape(-1, 4), [names[int(k)] if names is not None else str(k) for k in cls])
        for b, cf, k in zip(boxes, confs, cls):
            x0, y0, x1, y1 = (int(max(v * sc, 0)) for v in b)
            if x1 <= x0 or y1 <= y0 or cf < c.conf:
                continue
            if d.changed[y0:y1, x0:x1].mean() < c.fg_frac or not recent[y0:y1, x0:x1].any():
                continue
            if self.stock is not None and self.stock(names[int(k)] if names is not None else str(k), b):
                continue           # an item standing in a slot (bree.shelf.slots), not in a hand
            self._add(f, names[int(k)] if names is not None else str(k), float(b[0] + b[2]) / 2, float(b[1] + b[3]) / 2, float(cf))

    def _add(self, f: int, sku: str, u: float, v: float, conf: float) -> None:
        c = self.cfg
        for tr in reversed(self.tracks):
            lf, lu, lv, _ = tr["obs"][-1]
            if tr["sku"] == sku and 0 < f - lf <= c.track_gap_s * self.diff.fps and np.hypot(u - lu, v - lv) <= c.track_px:
                tr["obs"].append((f, u, v, conf))
                return
        self.tracks.append({"sku": sku, "obs": [(f, u, v, conf)]})

    def closed(self, t: float) -> list[dict]:
        """Tracks that ended, and whose slot (if they point at one) shows no pending change: each returned once."""
        out, fps = [], self.diff.fps
        for tr in self.tracks:
            if tr.get("returned") or t < tr["obs"][-1][0] / fps + self.cfg.track_gap_s + 0.5:
                continue
            j = self._slot(tr)
            if j is not None and t != float("inf"):
                near = self.diff.ids == j[1]
                # wait until the slot is in plain view and unchanged (the shelf comparison had its chance) for a second
                if (self.diff.f - self.diff.last_same[near]).max() > 0 or self.diff.still[near].min() < fps:
                    continue
            tr["returned"] = True
            out.append(tr)
        return out

    def _slot(self, tr: dict) -> tuple[str, int] | None:
        """(kind, slot index): the slot of this SKU the track starts at (take) or ends at (put), if any."""
        c, d = self.cfg, self.diff
        if len(tr["obs"]) < c.min_obs or tr["sku"] not in self._by_sku:
            return None
        pts = np.array([o[1:3] for o in tr["obs"]])
        if np.ptp(pts, axis=0).max() < c.min_travel_px:
            return None            # it never moved: not an item in a hand
        idx = np.array(self._by_sku[tr["sku"]])
        b = d.boxes[idx] / d.cfg.scale
        centres, diag = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], np.hypot(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        first, last = np.array(tr["obs"][0][1:3]), np.array(tr["obs"][-1][1:3])
        d0, d1 = np.linalg.norm(centres - first, axis=1), np.linalg.norm(centres - last, axis=1)
        reach = np.maximum(c.near_slot * diag, c.near_slot_min_px)
        i0, i1 = int(np.argmin(d0 / reach)), int(np.argmin(d1 / reach))
        if d0[i0] <= reach[i0] and d1[i0] >= c.away * max(d0[i0], reach[i0] / 2):
            return "take", int(idx[i0])
        if d1[i1] <= reach[i1] and d0[i1] >= c.away * max(d1[i1], reach[i1] / 2):
            return "put", int(idx[i1])
        return None

    def as_event(self, tr: dict) -> dict | None:
        got = self._slot(tr)
        if got is None:
            return None
        kind, j = got
        d, fps, s = self.diff, self.diff.fps, self.diff.slots[got[1]]
        f, u, v, _ = tr["obs"][0] if kind == "take" else tr["obs"][-1]
        return {"camera_id": d.id, "t": round(f / fps, 3), "t_start": round(tr["obs"][0][0] / fps, 3), "t_end": round(tr["obs"][-1][0] / fps, 3),
                "kind": kind, "slot_id": s["id"], "sku_id": tr["sku"], "sku_conf": round(float(np.mean([o[3] for o in tr["obs"]])), 3),
                "count": 1, "source": "hand_item", "cue": "hand_item", "hand_px": [round(u, 1), round(v, 1)],
                "point_3d": [round(float(x), 3) for x in s["face"]], "point_sigma_m": slot_sigma(s),
                "evidence": {"before": None, "after": None, "frames": [tr["obs"][0][0], tr["obs"][-1][0]]},
                "slots": [(s["id"], 1.0)], "detector_frames": len(tr["obs"])}
