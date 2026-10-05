"""Shelf events from one fixed camera, without a person box: what changed on the shelf once the hand is gone.

A camera that looks at a shelf sees the same picture until somebody takes or returns an item. `ShelfDiff` keeps a
reference picture, waits until a changed patch has stopped moving, and reports it as a shelf event (the shared
contract: camera_id, t, t_start, t_end, kind, slot_id, sku_id, sku_conf, count, source, hand_px, point_3d, ...).

Which slot: calibration and the planogram give, for every pixel, the slot whose front item is drawn there
(`slot_map`: the front item of every slot as a 3D box, nearest box wins, slots behind another fixture dropped). The
changed patch is compared with the visible pixels of each slot (intersection over union): the best one is the slot.
No person detector, no item detector, no ground truth.

Big changed regions (a person standing still, an open cooler door) are never events: an event is a patch about the
size of the visible part of a slot's front item. Take or put: a patch that goes back to the picture from before an
earlier take of that slot is a put.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bree.calib.camera import Camera

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class DiffConfig:
    scale: float = 0.5          # work at this share of the camera resolution
    diff_thr: int = 28          # a pixel differs from the reference (0 to 255, largest colour channel)
    move_thr: int = 14          # a pixel moved since the last frame
    still_frames: int = 5       # a patch must be still this long before it is read
    min_area_px: int = 20       # at working scale
    min_slot_px: int = 16       # a slot with fewer visible pixels (working scale) cannot be read by this camera
    max_slot_areas: float = 8.0 # a patch larger than this many of the largest visible slot is a person or a door
    min_slot_iou: float = 0.25  # changed patch against the visible pixels of a slot
    put_match: float = 14.0     # mean colour distance to the picture before an earlier take, to call it a put
    max_range_m: float = 9.0
    gain_norm: bool = True      # undo a frame-wide brightness step (lamp flicker, exposure) before comparing
    gain_range: tuple[float, float] = (0.25, 4.0)   # a brightness change inside this is undone; outside: camera unreliable
    gain_ok: tuple[float, float] = (0.8, 1.25)      # outside this a status record says the light has changed (it is still undone)
    shift_norm: bool = True     # undo a small shift of the whole picture (a camera that sways or was nudged)
    min_shift_px: float = 0.5   # working scale; a smaller shift is left alone
    max_shift_px: float = 12.0  # working scale; a larger one cannot be undone: camera unreliable
    big_change: float = 0.5     # this share of the picture unlike the reference: unreliable, nothing is read
    reset_still_s: float = 1.0  # unreliable and nothing moved this long: take a new reference picture
    evidence_dir: str | None = None   # write before/after crops of every event here (evidence.before / .after)


def slot_sigma(slot: dict) -> float:
    """How well point_3d (the slot face from the planogram) places the act: half the slot's width, at least 3 cm."""
    n, size = np.asarray(slot["normal"], float), np.asarray(slot["size"], float)
    return round(max(abs(float(size @ np.abs(np.cross(UP, n)))) / 2, 0.03), 3)


def item_corners(slot: dict, sku_size=None) -> np.ndarray:
    """The 8 corners of the front item of a slot (store frame). sku_size = (width, height, depth) from the catalogue;
    without it the slot's own block is used, cut to one item deep."""
    face, n, size = np.asarray(slot["face"], float), np.asarray(slot["normal"], float), np.asarray(slot["size"], float)
    side = np.cross(UP, n)
    if sku_size is not None:
        w, h, d = (float(v) for v in sku_size)
    else:
        w = abs(float(size @ np.abs(side)))
        h, d = float(size[1]), min(abs(float(size @ np.abs(n))), max(w, 0.06))
    return np.array([face + a * side * w / 2 + b * UP * h / 2 - k * n * d for a in (-1, 1) for b in (-1, 1) for k in (0, 1)])


def _blocked(C: np.ndarray, P: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Does the segment camera -> point pass through one of the axis-aligned boxes (lo, hi)?"""
    d = P - C
    for lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (lo - C) / d, (hi - C) / d
        t0, t1 = np.where(np.isnan(t0), -np.inf, t0), np.where(np.isnan(t1), np.inf, t1)
        near, far = np.minimum(t0, t1).max(), np.maximum(t0, t1).min()
        if near < far and far > 0 and near < 0.98:
            return True
    return False


def fixture_boxes(fixtures: list[dict] | None, kinds=("gondola", "counter", "backbar")) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned box of every fixture that hides what is behind it (rotationY in quarter turns)."""
    out = {}
    for f in fixtures or []:
        if f.get("type") not in kinds:
            continue
        s, p = np.asarray(f["size"], float), np.asarray(f["position"], float)
        if abs(np.sin(f.get("rotationY", 0.0))) > 0.5:
            s = s[[2, 1, 0]]
        out[f["id"]] = (p - s / 2 + 0.03, p + s / 2 - 0.03)
    return out


def slot_map(cam: Camera, slots: list[dict], scale: float = 0.5, max_range_m: float = 9.0, skus: dict | None = None,
             fixtures: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(id map, visible pixels per slot, box per slot). id map: (h, w) int32 at working scale, the index of the slot
    whose front item is drawn at that pixel, -1 where there is none. Boxes are x0, y0, x1, y1 of the whole item."""
    w, h = cam.resolution
    W, H = int(round(w * scale)), int(round(h * scale))
    ids = np.full((H, W), -1, np.int32)
    boxes = np.zeros((len(slots), 4))
    fx = fixture_boxes(fixtures)
    todo = []
    for i, s in enumerate(slots):
        face, n = np.asarray(s["face"], float), np.asarray(s["normal"], float)
        to_cam = cam.C - face
        dist = float(np.linalg.norm(to_cam))
        if dist > max_range_m or float(to_cam @ n) / dist < 0.08:
            continue
        px, z = cam.project(item_corners(s, (skus or {}).get(s.get("skuId"))))
        if (z <= 0.05).any():
            continue
        px = px * scale
        x0, y0, x1, y1 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
            continue
        if fx and _blocked(cam.C, face + 0.02 * n, [b for k, b in fx.items() if k != s.get("fixtureId")]):
            continue
        boxes[i] = x0, y0, x1, y1
        todo.append((dist, i, cv2.convexHull(np.round(px).astype(np.int32))))
    for _, i, hull in sorted(todo, key=lambda r: -r[0]):      # far first: the nearer item is drawn over it
        cv2.fillConvexPoly(ids, hull, i)
    area = np.bincount(ids[ids >= 0], minlength=len(slots)).astype(np.int64)
    return ids, area, boxes


def slot_boxes(cam: Camera, slots: list[dict], scale: float, max_range_m: float, skus: dict | None = None) -> tuple[np.ndarray, list[int]]:
    """Image boxes (x0, y0, x1, y1, at working scale) of the front item of every slot this camera can see."""
    _, area, boxes = slot_map(cam, slots, scale, max_range_m, skus)
    keep = [i for i in range(len(slots)) if area[i] > 0]
    return boxes[keep].reshape(-1, 4), keep


def _maxdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_, g_, r_ = cv2.split(cv2.absdiff(a, b))
    return cv2.max(cv2.max(b_, g_), r_)


class ShelfDiff:
    """Feed frames in time order with update(); read `changed` (pixels unlike the empty-of-people shelf) and `moved`
    after each call if another cue wants them (bree.shelf.hand)."""

    def __init__(self, camera_id: str, cam: Camera, slots: list[dict], fps: float, cfg: DiffConfig | None = None,
                 skus: dict | None = None, fixtures: list[dict] | None = None):
        self.id, self.cam, self.slots, self.fps, self.cfg = camera_id, cam, slots, fps, cfg or DiffConfig()
        c = self.cfg
        self.ids, self.area, self.boxes = slot_map(cam, slots, c.scale, c.max_range_m, skus, fixtures)
        self.readable = self.area >= c.min_slot_px
        self.max_area = c.max_slot_areas * float(self.area.max()) if len(self.area) else 0.0
        self.ref = self.prev = self.still = self.last_same = self.changed = self.moved = None
        self.taken: dict[int, tuple[tuple[int, int, int, int], np.ndarray, dict]] = {}     # slot index -> (box, picture before, take event)
        self.f = -1
        # camera health: records {"camera_id", "t", "kind": "status", "status": ..., ...} for the association step.
        # "unreliable" opens a window in which this camera reports nothing; "reference_reset" or "reliable" closes it.
        self.status: list[dict] = []
        self.unreliable, self.dead, self.quiet, self.gain_off = False, False, 0, False
        self.gain, self.shift, self.first_small = 1.0, np.zeros(2), None

    def _note(self, status: str, **detail) -> None:
        self.status.append({"camera_id": self.id, "t": round(self.f / self.fps, 3), "kind": "status", "status": status, **detail})

    def _small(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _set_ref(self, img: np.ndarray) -> None:
        self.ref, self.prev = img.copy(), img
        self.ref_small = self._small(img)
        self.win = cv2.createHanningWindow(self.ref_small.shape[::-1], cv2.CV_32F)
        # OpenCV reports a small constant offset for identical pictures of some sizes: measure it once and take it off
        self.bias = np.array(cv2.phaseCorrelate(self.ref_small * self.win, self.ref_small * self.win)[0])
        if self.first_small is None:
            self.first_small = self.ref_small

    def _align(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """Undo a shift of the whole picture against the reference. (picture, shift too large to undo)."""
        c, g = self.cfg, self._small(img)
        (dx, dy), _ = cv2.phaseCorrelate(self.ref_small * self.win, g * self.win)      # windowed copies: OpenCV windows its inputs in place
        est, last = 2.0 * (np.array([dx, dy]) - self.bias), self.shift
        self.shift = np.zeros(2)
        if float(np.hypot(*est)) < c.min_shift_px:
            return img, False
        k = self.gain if c.gain_range[0] <= self.gain <= c.gain_range[1] else 1.0
        move = lambda pic, d: cv2.warpAffine(pic, np.float32([[1, 0, -d[0]], [0, 1, -d[1]]]), pic.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        resid = lambda d: float(np.abs(move(g, d / 2) * k - self.ref_small).mean())
        r0 = float(np.abs(g * k - self.ref_small).mean())
        # a walking shopper can fool the estimate: a new shift is kept only when it makes the picture look clearly more
        # like the reference; the shift of the frame before is kept while it still helps at all
        if resid(est) <= 0.85 * r0:
            d = est
        elif last.any() and resid(last) < r0:
            d = last
        else:
            return img, False
        if float(np.hypot(*d)) > c.max_shift_px:
            return img, True
        self.shift = d
        return move(img, d), False

    def _gain(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """Undo a brightness change of the whole picture. (picture, change too large to undo)."""
        lo, hi = self.cfg.gain_range
        self.gain = g = float(np.median(self.ref[::8, ::8].astype(np.float32).sum(2) / np.maximum(img[::8, ::8].astype(np.float32).sum(2), 1)))
        if not lo <= g <= hi:
            return img, True
        return (cv2.convertScaleAbs(img, alpha=g) if abs(g - 1) > 0.01 else img), False

    def _health(self, raw: np.ndarray, img: np.ndarray, share: float, bad: bool) -> bool:
        """Keep the reference usable. True: nothing may be read from this frame."""
        c = self.cfg
        self.quiet = self.quiet + 1 if float(self.moved.mean()) < 0.002 else 0
        calm = self.quiet >= c.reset_still_s * self.fps
        if bad or share > c.big_change:
            if not self.unreliable:
                self.unreliable = True
                self._note("unreliable", changed_share=round(share, 3), gain=round(self.gain, 3), shift_px=[round(float(v) / c.scale, 1) for v in self.shift])
            if calm:        # the view has settled on a new picture: that is the shelf now. Earlier takes cannot be matched by a put any more
                self._set_ref(raw)
                self.taken.clear()
                self.last_same[:] = self.f
                self.unreliable, self.quiet = False, 0
                # how far the new picture sits from the one the calibration was made for
                (dx, dy), resp = cv2.phaseCorrelate(self.first_small * self.win, self.ref_small * self.win)
                off = 2.0 * float(np.hypot(dx - self.bias[0], dy - self.bias[1]))
                self._note("reference_reset", reason="picture changed", changed_share=round(share, 3), shift_px=round(off / c.scale, 1))
                if off > c.max_shift_px and resp > 0.3:      # slots would be read at the wrong place: stop until recalibrated
                    self.dead = True
                    self._note("needs_recalibration", shift_px=round(off / c.scale, 1))
            return True
        if self.unreliable:
            self.unreliable = False
            self._note("reliable")
        off = not c.gain_ok[0] <= self.gain <= c.gain_ok[1]
        if off != self.gain_off:       # the light is far from the reference picture's (still undone, but worth knowing)
            self.gain_off = off
            self._note("brightness_changed" if off else "brightness_back", gain=round(self.gain, 3))
        return False

    def update(self, image: np.ndarray) -> list[dict]:
        """Feed the next frame (BGR, full resolution). Returns the shelf events that became readable at this frame."""
        c, self.f = self.cfg, self.f + 1
        img = cv2.resize(image, (self.ids.shape[1], self.ids.shape[0]), interpolation=cv2.INTER_AREA) if c.scale != 1 else image
        if self.ref is None:
            self._set_ref(img)
            self.still = np.zeros(img.shape[:2], np.int32)
            self.last_same = np.zeros(img.shape[:2], np.int32)
            self.changed = np.zeros(img.shape[:2], np.uint8)
            self.moved = np.zeros(img.shape[:2], bool)
            return []
        if self.dead:
            return []
        raw, bad = img, False
        if c.gain_norm:
            img, bad = self._gain(img)
        if c.shift_norm:
            img, far = self._align(img)
            bad = bad or far
        self.moved = _maxdiff(img, self.prev) > c.move_thr
        self.prev = img
        self.still = np.where(cv2.dilate(self.moved.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0, 0, self.still + 1)
        changed = (_maxdiff(img, self.ref) > c.diff_thr).astype(np.uint8)
        if self.shift.any():        # the strip the shift pushed out of the picture has nothing to compare with
            mx, my = (int(np.ceil(abs(v))) + 2 for v in self.shift)
            changed[:my], changed[-my:], changed[:, :mx], changed[:, -mx:] = 0, 0, 0, 0
        self.changed = changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if self._health(raw, img, float(changed.mean()), bad):
            return []
        self.last_same[changed == 0] = self.f
        if not changed.any() or not self.readable.any():
            return []
        n, lab, stats, cent = cv2.connectedComponentsWithStats(cv2.dilate(changed, np.ones((5, 5), np.uint8)), connectivity=8)
        out = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < c.min_area_px or area > self.max_area:
                continue
            m = lab[y:y + h, x:x + w] == k
            if self.still[y:y + h, x:x + w][m].min() < c.still_frames:
                continue
            core = m & (changed[y:y + h, x:x + w] > 0)
            ev = self._read(img, (x, y, w, h), m, core, cent[k])
            if ev:
                if c.evidence_dir:
                    self._evidence(ev, img, (x, y, w, h))
                out.append(ev)
            # the patch is still and small: it is the shelf now, event or not (a moved price tag, an item pushed aside)
            self.ref[y:y + h, x:x + w][m] = img[y:y + h, x:x + w][m]
        return out

    def _read(self, img, box, m, core, cent) -> dict | None:
        c = self.cfg
        x, y, w, h = box
        hit = self.ids[y:y + h, x:x + w][m]
        cnt = np.bincount(hit[hit >= 0], minlength=len(self.slots))
        iou = np.where(self.readable, cnt / np.maximum(int(m.sum()) + self.area - cnt, 1), 0.0)
        order = np.argsort(-iou)[:5]
        best = int(order[0])
        t0 = float(np.median(self.last_same[y:y + h, x:x + w][core])) / self.fps
        # back to the picture before an earlier take of a slot under this patch: a put (checked first: the returned
        # item sits where the camera saw it leave, whatever the overlap says)
        for j in [int(j) for j in order if iou[j] > 0.05 and int(j) in self.taken]:
            (bx, by, bw, bh), before, take = self.taken[j]
            ax, ay, ex, ey = max(x, bx), max(y, by), min(x + w, bx + bw), min(y + h, by + bh)
            if ex <= ax or ey <= ay:
                continue
            d = img[ay:ey, ax:ex].astype(np.int16) - before[ay - by:ey - by, ax - bx:ex - bx].astype(np.int16)
            if np.abs(d).max(axis=2).mean() < c.put_match:
                del self.taken[j]
                # the put names the take it undoes: this camera saw this place go back to the picture from before that
                # take, whether an item was returned or an arm that had covered the slot went away
                return {**self._event("put", j, take["sku_conf"], take["slots"], t0, cent, int(m.sum())), "undoes": list(take["eids"])}
        if iou[best] < c.min_slot_iou:
            return None
        second = float(iou[order[1]]) if len(order) > 1 else 0.0
        cands = [(self.slots[int(j)]["id"], round(float(iou[j]), 3)) for j in order if iou[j] > 0]
        ev = self._event("take", best, float(iou[best] / (iou[best] + second + 1e-9)), cands, t0, cent, int(m.sum()))
        # ponytail: units from patch area over one item's visible area, capped by the slot's facings. Reads 1 for a
        # single-facing slot always; a depth-aware count (how far back the new front item sits) is the upgrade.
        ev["count"] = int(np.clip(round(float(core.sum()) / max(int(self.area[best]), 1)), 1, max(int(self.slots[best].get("facings", 1)), 1)))
        ev["eids"] = [f"{self.id}:{self.f}:{ev['slot_id']}"]      # a name for this reading, see "undoes" above
        self.taken[best] = (box, self.ref[y:y + h, x:x + w].copy(), ev)
        return ev

    def _evidence(self, ev: dict, img: np.ndarray, box, margin: int = 40) -> None:
        from pathlib import Path
        x, y, w, h = box
        y0, y1, x0, x1 = max(y - margin, 0), y + h + margin, max(x - margin, 0), x + w + margin
        d = Path(self.cfg.evidence_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, pic in (("before", self.ref), ("after", img)):
            path = d / f"{self.id}_{self.f:06d}_{ev['slot_id']}_{name}.jpg"
            cv2.imwrite(str(path), pic[y0:y1, x0:x1])
            ev["evidence"][name] = str(path)

    def _event(self, kind, j, conf, cands, t0, cent, area) -> dict:
        c, slot = self.cfg, self.slots[j]
        t0 = min(t0, (self.f - c.still_frames) / self.fps)
        return {"camera_id": self.id, "t": round(t0, 3), "t_start": round(t0, 3), "t_end": round((self.f - c.still_frames) / self.fps, 3),
                "kind": kind, "slot_id": slot["id"], "sku_id": slot.get("skuId"), "sku_conf": round(conf, 3), "count": 1, "source": "shelf_diff",
                "hand_px": [round(float(cent[0] / c.scale), 1), round(float(cent[1] / c.scale), 1)],
                "point_3d": [round(float(v), 3) for v in slot["face"]], "point_sigma_m": slot_sigma(slot),
                "evidence": {"before": None, "after": None, "frames": [int(round(t0 * self.fps)), self.f]},
                "slots": cands, "area_px": int(area / c.scale ** 2), "slot_px": int(self.area[j] / c.scale ** 2)}
