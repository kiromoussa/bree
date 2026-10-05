"""On-camera first pass: "something hand-sized moved into a shelf zone".

Runs on the node (Raspberry Pi Zero 2 W) on the low-resolution stream. No model: a running-average
background per pixel (brightness and colour), a frame-to-frame motion check, and a blob-size gate inside each shelf polygon,
with a skin-colour shortcut that lets a smaller blob through. Tuned for few misses: a false trigger
costs a few megabytes of upload, a miss loses the pick.

Only numpy and OpenCV, so the same file runs on the Pi (python3-opencv from apt) and on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np


@dataclass
class TriggerConfig:
    width: int = 320              # low-res frame width the trigger works at (height follows the aspect)
    diff_thr: int = 18            # grey levels away from the background to count as changed
    motion_thr: int = 10          # grey levels away from the previous frame to count as moving
    chroma_thr: int = 14          # Cr + Cb levels away (background or previous frame): colour change counts too
    min_blob_px: int = 30         # largest changed blob inside the zone, low-res pixels
    min_motion_px: int = 6        # moving pixels inside the zone, low-res pixels
    margin_px: int = 6            # grow each zone by this many low-res pixels (hands at the shelf's front edge)
    skin: bool = True             # a skin-coloured blob of half min_blob_px also triggers
    bg_alpha: float = 0.02        # background learning rate per frame where nothing changed
    bg_alpha_fg: float = 0.002    # ... where something did (a removed item is absorbed slowly, a hand is not)
    gain_norm: bool = True        # scale each frame's brightness to the background's before differencing (flicker)
    shake_px: int = 0             # > 0: a pixel only counts as changed if it differs from every background pixel
                                  # within this many low-res pixels (camera shake moves edges, a hand adds new colour)
    flood_frac: float = 0.6       # more of the frame changed than this = lights, not a hand: relearn, no trigger
    release_s: float = 0.5        # a zone stays active this long after the last hit
    warmup_frames: int = 5        # frames used to seed the background before any trigger

    @classmethod
    def from_dict(cls, d: dict | None) -> "TriggerConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d or {}) - names
        if unknown:
            raise ValueError(f"unknown trigger setting(s): {sorted(unknown)}")
        return cls(**(d or {}))


# Skin in YCrCb (the classic fixed box). Lighting and gloves break it, so it only ever lowers the
# blob size needed; it never blocks a trigger.
_SKIN_LO, _SKIN_HI = (0, 135, 85), (255, 180, 135)


class ZoneTrigger:
    """zones: {zone_id: polygon in pixels of `zones_size` (w, h)}. Feed low-res BGR frames in time order."""

    def __init__(self, zones: dict[str, np.ndarray], zones_size: tuple[int, int], cfg: TriggerConfig | None = None):
        self.cfg = cfg or TriggerConfig()
        self.zones = {k: np.asarray(p, dtype=np.float64) for k, p in zones.items()}
        self.zones_size = zones_size
        self._masks: dict[str, tuple[slice, slice, np.ndarray]] = {}
        self._shape: tuple[int, int] | None = None
        self._bg: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._n = 0
        self._last_hit: dict[str, float] = {}

    def lowres_size(self, full_w: int, full_h: int) -> tuple[int, int]:
        w = min(self.cfg.width, full_w)
        return w, max(2, round(full_h * w / full_w))

    def _build_masks(self, h: int, w: int) -> None:
        sx, sy = w / self.zones_size[0], h / self.zones_size[1]
        for zid, poly in self.zones.items():
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(poly * [sx, sy]).astype(np.int32)], 255)
            if self.cfg.margin_px > 0:      # reach a little past the drawn shelf: hands at the front edge count
                k = 2 * self.cfg.margin_px + 1
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            x, y, bw, bh = cv2.boundingRect(m)
            if not bw:
                raise ValueError(f"zone {zid!r} is outside the frame")
            self._masks[zid] = (slice(y, y + bh), slice(x, x + bw), m[y:y + bh, x:x + bw])
        self._shape = (h, w)

    def update(self, lores_bgr: np.ndarray, t: float) -> set[str]:
        """Zone ids that are active at time t (seconds, any monotonic clock)."""
        c = self.cfg
        h, w = lores_bgr.shape[:2]
        if self._shape != (h, w):
            self._build_masks(h, w)
            self._bg = self._prev = None
            self._n = 0
        ycc = cv2.cvtColor(lores_bgr, cv2.COLOR_BGR2YCrCb)
        cur = cv2.GaussianBlur(ycc, (5, 5), 0)
        self._n += 1
        if self._bg is None:
            self._bg, self._prev = cur.astype(np.float32), cur
            return set()
        bg8 = cv2.convertScaleAbs(self._bg)
        if c.gain_norm:
            # Lamp flicker and auto-exposure steps move every pixel by the same factor. Take the median
            # ratio background / frame on a coarse grid (the median ignores a shopper covering under half
            # the frame) and undo it. Outside 0.8 to 1.25 it is a real lighting change: leave it to the
            # flood check below, which relearns the background.
            g = float(np.median(bg8[::4, ::4, 0].astype(np.float32) / np.maximum(cur[::4, ::4, 0], 1)))
            if 0.8 <= g <= 1.25 and abs(g - 1) > 0.004:
                cur[:, :, 0] = cv2.convertScaleAbs(cur[:, :, 0], alpha=g)
        if c.shake_px > 0:
            k = np.ones((2 * c.shake_px + 1,) * 2, np.uint8)
            d = cv2.max(cv2.subtract(cv2.erode(bg8, k), cur), cv2.subtract(cur, cv2.dilate(bg8, k)))
            fg = (d[:, :, 0] > c.diff_thr) | (cv2.add(d[:, :, 1], d[:, :, 2]) > c.chroma_thr)
        else:
            fg = self._changed(cur, bg8, c.diff_thr)
        if self._n <= c.warmup_frames or fg.mean() > c.flood_frac:
            cv2.accumulateWeighted(cur, self._bg, 0.5)    # seeding, or the lights changed: relearn fast
            self._prev = cur
            return self._active(t)
        moving = self._changed(cur, self._prev, c.motion_thr)
        self._prev = cur
        fg8 = fg.astype(np.uint8)
        for zid, (ys, xs, mask) in self._masks.items():
            z = fg8[ys, xs] & mask
            n = int(cv2.countNonZero(z))
            if n < c.min_blob_px // 2:
                continue
            if int(cv2.countNonZero(moving[ys, xs].astype(np.uint8) & mask)) < c.min_motion_px:
                continue
            blob = n if n < c.min_blob_px else int(cv2.connectedComponentsWithStats(z, connectivity=8)[2][1:, 4].max())
            hit = blob >= c.min_blob_px
            if not hit and c.skin and blob >= c.min_blob_px // 2:
                skin = cv2.inRange(ycc[ys, xs], _SKIN_LO, _SKIN_HI) & z * 255
                hit = int(cv2.countNonZero(skin)) >= c.min_blob_px // 4
            if hit:
                self._last_hit[zid] = t
        cv2.accumulateWeighted(cur, self._bg, c.bg_alpha, mask=(~fg).astype(np.uint8))
        cv2.accumulateWeighted(cur, self._bg, c.bg_alpha_fg, mask=fg8)
        return self._active(t)

    def _changed(self, a: np.ndarray, b: np.ndarray, thr: int) -> np.ndarray:
        """Brightness moved by more than thr, or colour (Cr + Cb) by more than chroma_thr. Colour catches a
        hand that is as bright as the shelf behind it."""
        d = cv2.absdiff(a, b)
        return (d[:, :, 0] > thr) | (cv2.add(d[:, :, 1], d[:, :, 2]) > self.cfg.chroma_thr)

    def _active(self, t: float) -> set[str]:
        return {z for z, th in self._last_hit.items() if t - th <= self.cfg.release_s}
