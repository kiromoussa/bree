"""Shelf and floor state from one fixed camera, no calibration, no product model.

What it does: keeps a person-free picture of the scene (the reference), compares every new frame with it cell by cell
with measures that do not care about brightness (normalised cross-correlation of the grey picture, plus chroma),
and reports a change only after it has (a) lasted `settle_s`, (b) stopped moving, (c) had no person on it for
`clear_s`. People are masked with detector boxes (`bree.shelfstate.frames.PersonDetector`) and with motion; cells
that flicker all the time (a TV, an LED sign) are learned and ignored. After an event the reference is replaced
there, so one change is one event.

Two pictures are kept: the adaptive reference (the shelf as it is now, person-free) and the golden picture (the
shelf as it was first seen, or as a person blessed it). Settled changes against the adaptive reference are the
transition events (stock_down, restock, floor_object, floor_spill, door_ajar, layout_change). The comparison of the
adaptive reference with the golden picture gives the state events (low_stock, empty_facing) and the drift figure.

Confidences are heuristic scores in 0..1 (how long, how big, how far past the threshold), not calibrated
probabilities. See docs/ultra/shelf.md for what was measured on real footage and with which labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from bree.shelfstate.evidence import write_evidence
from bree.shelfstate.frames import box_mask
from bree.shelfstate.regions import CameraRegions, whole_picture

CELL = 10   # pixels per cell at working size (640x360 -> 64x36 cells)


@dataclass
class ShelfStateConfig:
    # change measure
    flat_std: float = 3.0         # grey std (0..255) below which a cell has no texture to correlate
    ncc_thr: float = 0.45         # 1 - NCC above this: structure changed
    chroma_thr: float = 14.0      # distance of the cell's (a, b) / L above this (x100): colour changed; gain-invariant
    light_thr: float = 34.0       # Lab L step (after the gain field) above this: brightness of a flat cell changed
    gain_range: tuple[float, float] = (0.4, 2.5)
    # people and motion
    person_grow: float = 0.12     # grow person boxes by this share of their size
    person_dilate: int = 1        # cells of margin around a person mask
    move_thr: float = 0.35        # 1 - NCC against the previous frame above this: the cell is moving
    dynamic_ema: float = 0.97     # cells that move this often are screens or signs and are ignored
    dynamic_thr: float = 0.30
    # persistence
    settle_s: float = 12.0
    clear_s: float = 6.0          # no person box on the cell for this long
    still_s: float = 3.0          # no motion on the cell for this long
    cooldown_s: float = 20.0
    patience_s: float = 10.0
    min_cells: int = 2
    max_region_share: float = 0.85  # a change over more than this share of its region is not read (light, camera)
    warmup_frames: int = 16
    seed_min: int = 8             # person-free frames a cell needs before it has a reference
    seed_buffer: int = 16         # frames the reference is the median of
    ref_alpha: float = 0.03       # per frame blend of unchanged visible cells into the reference (lighting drift)
    # whole-frame sanity
    big_change: float = 0.45      # share of visible cells changed: light or camera, no events
    reset_s: float = 30.0         # unreliable this long: take a new reference
    # state events against the golden picture
    fill_hi: float = 0.42         # golden cell counts as stocked at or above this fill
    fill_lo: float = 0.18         # current cell counts as empty at or below this fill
    low_stock_frac: float = 0.35  # share of a shelf region's stocked cells that are empty
    low_stock_s: float = 120.0
    snapshot_s: float = 60.0
    # burned-in text (Dahua clock top right, 'IPC' bottom left) changes every second and is never a shelf event
    overlay_boxes: tuple = ((0.72, 0.0, 1.0, 0.09), (0.0, 0.90, 0.07, 1.0))
    evidence_dir: str | None = None


def blocks(a: np.ndarray, c: int = CELL) -> np.ndarray:
    h, w = a.shape[:2]
    return a.reshape(h // c, c, w // c, c, *a.shape[2:])


def cell_sum(a: np.ndarray) -> np.ndarray:
    return blocks(a).sum(axis=(1, 3))


def cell_mean(a: np.ndarray) -> np.ndarray:
    return blocks(a).mean(axis=(1, 3))


def ncc_cells(a: np.ndarray, b: np.ndarray, flat_std: float) -> np.ndarray:
    """1 - normalised cross-correlation per cell of two grey pictures (float32), weighted to 0 where both are flat.
    Gain and offset of either picture do not matter."""
    n = float(CELL * CELL)
    sa, sb = cell_sum(a), cell_sum(b)
    va = np.maximum(cell_sum(a * a) - sa * sa / n, 0) / n
    vb = np.maximum(cell_sum(b * b) - sb * sb / n, 0) / n
    cov = (cell_sum(a * b) - sa * sb / n) / n
    eps = flat_std ** 2
    ncc = cov / (np.sqrt(va * vb) + eps)
    w = np.clip((np.sqrt(np.maximum(va, vb)) - flat_std) / flat_std, 0, 1)
    return w * (1 - ncc)


def cell_std(g: np.ndarray) -> np.ndarray:
    n = float(CELL * CELL)
    s1 = cell_sum(g)
    return np.sqrt(np.maximum(cell_sum(g * g) - s1 * s1 / n, 0) / n)


def fill_map(bgr: np.ndarray) -> np.ndarray:
    """Per cell 0..1 'something stocked here': colourful and busy packaging is high, bare shelf, wall, floor low.
    A proxy that was only checked by eye on this store (docs/ultra/shelf.md), not a product recogniser."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = cell_mean(hsv[..., 1].astype(np.float32)) / 255.0
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 0.8)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    edge = cell_mean(np.minimum(np.hypot(gx, gy), 200.0)) / 200.0
    return np.clip(0.55 * sat / 0.45 + 0.45 * edge / 0.25, 0, 1)


def _prep(bgr: np.ndarray):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    return g, lab


def _chroma(lab_cells: np.ndarray) -> np.ndarray:
    """(a, b) of Lab cell means divided by lightness: colour with the brightness taken out (exposure steps scale both)."""
    L = np.maximum(lab_cells[..., 0], 30.0)
    return np.stack([(lab_cells[..., 1] - 128.0) / L, (lab_cells[..., 2] - 128.0) / L], -1)


def _dilate(m: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool) if k else m


@dataclass
class Event:
    camera_id: str
    t: float
    t_start: float
    kind: str
    region_id: str
    conf: float
    box: list[float]               # normalised x0, y0, x1, y1 of the changed cells
    detail: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)   # {"before": path, "after": path} (people pixelated)
    source: str = "shelfstate"

    def to_dict(self) -> dict:
        return {"camera_id": self.camera_id, "t": round(self.t, 1), "t_start": round(self.t_start, 1), "kind": self.kind,
                "region_id": self.region_id, "conf": round(self.conf, 2), "box": [round(v, 3) for v in self.box],
                "detail": self.detail, "evidence": self.evidence, "source": self.source}


class StateMonitor:
    """Feed frames in time order with `update(t, frame_bgr, person_boxes)`; get events back."""

    def __init__(self, camera_id: str, regions: CameraRegions | None = None, cfg: ShelfStateConfig | None = None,
                 size: tuple[int, int] = (640, 360)):
        self.cam = camera_id
        self.cfg = cfg or ShelfStateConfig()
        self.W, self.H = size
        self.gh, self.gw = self.H // CELL, self.W // CELL
        self.regions = regions or whole_picture(camera_id)
        pix, self.reg_list = self.regions.label_map((self.H, self.W))
        # region per cell: the region covering most pixels of the cell (-1 none)
        self.cell_region = np.full((self.gh, self.gw), -1, np.int32)
        for i in range(len(self.reg_list)):
            share = cell_mean((pix == i).astype(np.float32))
            self.cell_region[share > 0.5] = i
        self.ignore = np.array([self.reg_list[i].kind == "ignore" if i >= 0 else False
                                for i in self.cell_region.ravel()]).reshape(self.gh, self.gw)
        for x0, y0, x1, y1 in self.cfg.overlay_boxes:
            self.ignore[int(y0 * self.gh):int(np.ceil(y1 * self.gh)), int(x0 * self.gw):int(np.ceil(x1 * self.gw))] = True
        shape = (self.gh, self.gw)
        self.ref_g = np.zeros((self.H, self.W), np.float32)
        self.ref_lab = np.zeros((self.H, self.W, 3), np.float32)
        self.ref_bgr = np.zeros((self.H, self.W, 3), np.uint8)
        self.seed_n = np.zeros(shape, np.int32)
        self.buf: list = []
        self.seeded = np.zeros(shape, bool)
        self.run = np.zeros(shape, np.float32)
        self.since_person = np.full(shape, 1e3, np.float32)
        self.since_moving = np.full(shape, 1e3, np.float32)
        self.cool = np.zeros(shape, np.float32)
        self.dyn = np.zeros(shape, np.float32)
        self.score = np.zeros(shape, np.float32)
        self.changed = np.zeros(shape, bool)
        self.visible = np.zeros(shape, bool)
        self.prev_g: np.ndarray | None = None
        self.cur_g = np.zeros((self.H, self.W), np.float32)
        self.last_t: float | None = None
        self.last_persons = np.zeros((0, 5))
        self.n = 0
        self.unreliable_since: float | None = None
        self.status: dict = {"gain": 1.0, "unreliable": False, "changed_frac": 0.0}
        self.golden_bgr: np.ndarray | None = None
        self.golden_fill: np.ndarray | None = None
        self.low_since: dict[str, float] = {}
        self.low_open: set[str] = set()
        self.last_snapshot = -1e9
        self.snapshots: list[dict] = []
        self.person_seen: np.ndarray = np.zeros(shape, np.float32)   # seconds of person presence per cell, this run

    # ------------------------------------------------------------------ reference
    def _seed(self, g, lab, bgr, ok):
        """Reference of a cell = per-pixel median of the frames in which no person or motion was on it, so a person who
        stood on the cell in a few of the first frames (or one the detector missed for a few) is voted out."""
        self.buf.append((g, lab, bgr, ok))
        self.buf = self.buf[-self.cfg.seed_buffer:]
        self.seed_n += ok
        ready = (self.seed_n >= self.cfg.seed_min) & ~self.seeded & ~self.ignore
        if not ready.any() or self.n % 3:
            return
        ready &= np.sum([o for *_, o in self.buf], axis=0) >= self.cfg.seed_min // 2
        if not ready.any():
            return
        rp = np.kron(ready, np.ones((CELL, CELL), bool))
        valid = np.stack([np.kron(o, np.ones((CELL, CELL), bool))[rp] for *_, o in self.buf])      # (frames, pixels)
        for idx, ref in ((0, self.ref_g), (1, self.ref_lab), (2, self.ref_bgr)):
            st = np.stack([f[idx][rp] for f in self.buf]).astype(np.float32)                        # (frames, pixels[, 3])
            st = np.where(valid if st.ndim == 2 else valid[..., None], st, np.nan)
            ref[rp] = np.nanmedian(st, axis=0).astype(ref.dtype)
        self.seeded |= ready
        if self.golden_bgr is None and self.seeded[~self.ignore].mean() > 0.95:
            self.set_golden()

    def set_golden(self, bgr: np.ndarray | None = None):
        """The picture of the shelf that counts as 'stocked': the first person-free reference, or one passed in."""
        self.golden_bgr = (self.ref_bgr if bgr is None else bgr).copy()
        self.golden_fill = fill_map(self.golden_bgr)

    def _replace_ref(self, cells: np.ndarray, g, lab, bgr):
        px = np.kron(cells, np.ones((CELL, CELL), bool))
        self.ref_g[px], self.ref_lab[px], self.ref_bgr[px] = g[px], lab[px], bgr[px]
        self.run[cells] = 0
        self.cool[cells] = self.cfg.cooldown_s

    # ------------------------------------------------------------------ main step
    def update(self, t: float, bgr: np.ndarray, persons: np.ndarray | None = None) -> list[Event]:
        cfg = self.cfg
        dt = 1.0 if self.last_t is None else float(np.clip(t - self.last_t, 0.1, 5.0))
        self.last_t = t
        self.n += 1
        g, lab = _prep(bgr)
        self.cur_g = g
        persons = np.zeros((0, 5)) if persons is None else persons
        self.last_persons = persons
        pm = box_mask(persons, (self.H, self.W), cfg.person_grow)
        occ = _dilate(cell_mean(pm.astype(np.float32)) > 0.05, cfg.person_dilate)
        self.since_person = np.where(occ, 0, self.since_person + dt)
        self.person_seen += occ * dt

        # motion against the previous frame
        if self.prev_g is None:
            moving = np.zeros_like(occ)
        else:
            moving = ncc_cells(g, self.prev_g, cfg.flat_std) > cfg.move_thr
        self.prev_g = g
        self.dyn = cfg.dynamic_ema * self.dyn + (1 - cfg.dynamic_ema) * (moving & ~occ)
        dynamic = self.dyn > cfg.dynamic_thr
        self.since_moving = np.where(moving, 0, self.since_moving + dt)
        self.cool = np.maximum(self.cool - dt, 0)

        ok_seed = ~occ & ~moving & (self.since_moving >= 0)
        if not (self.seeded | self.ignore).all():
            self._seed(g, lab, bgr, ok_seed)
        elif self.buf:
            self.buf = []
        if self.n <= cfg.warmup_frames or not self.seeded.any():
            return []

        # change measure against the reference
        d_struct = ncc_cells(g, self.ref_g, cfg.flat_std)
        mc, mr = cell_mean(lab), cell_mean(self.ref_lab)
        ratio = (cell_mean(g) + 8.0) / (cell_mean(self.ref_g) + 8.0)
        stable = self.seeded & ~self.changed & ~occ & ~moving
        gain = self._gain_field(ratio, stable)
        dL = np.abs(mc[..., 0] / gain - mr[..., 0])
        dab = np.linalg.norm(_chroma(mc) - _chroma(mr), axis=-1) * 100
        # chroma shifts with a strong brightness change, so it only counts where the brightness is near the reference
        near = (gain > 0.7) & (gain < 1.4)
        chg = (d_struct > cfg.ncc_thr) | ((dab > cfg.chroma_thr) & near) | (dL > cfg.light_thr)
        self.score = np.maximum.reduce([d_struct / cfg.ncc_thr, dab / cfg.chroma_thr, dL / cfg.light_thr]).astype(np.float32)
        vis = self.seeded & ~occ & ~moving & ~dynamic & ~self.ignore
        self.changed, self.visible = chg & vis, vis

        vis_n = max(int(vis.sum()), 1)
        frac = float((chg & vis).sum()) / vis_n
        self.status = {"gain": round(float(np.median(gain[stable])) if stable.any() else 1.0, 3),
                       "unreliable": frac > cfg.big_change, "changed_frac": round(frac, 3)}
        if self.status["unreliable"]:
            self.unreliable_since = self.unreliable_since if self.unreliable_since is not None else t
            if t - self.unreliable_since >= cfg.reset_s and not (occ | moving).any():
                self._replace_ref(vis, g, lab, bgr)
                self.unreliable_since = None
            return self._status_events(t, frac)
        self.unreliable_since = None

        self.run = np.where(vis & chg, self.run + dt, np.where(vis, 0, self.run))
        # unchanged visible cells follow slow lighting drift
        keep = vis & ~chg & (self.since_person >= cfg.clear_s)
        if keep.any():
            px = np.kron(keep, np.ones((CELL, CELL), bool))
            a = cfg.ref_alpha
            self.ref_g[px] = (1 - a) * self.ref_g[px] + a * g[px]
            self.ref_lab[px] = (1 - a) * self.ref_lab[px] + a * lab[px]
        events = self._settled_events(t, bgr, g, lab)
        events += self._state_events(t, bgr)
        return events

    def _gain_field(self, ratio: np.ndarray, stable: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg.gain_range
        w = stable.astype(np.float32)
        if w.sum() < 8:
            return np.ones_like(ratio)
        r = np.clip(ratio, lo, hi)
        med = float(np.median(r[stable]))
        num = cv2.GaussianBlur(r * w, (0, 0), 3.0)
        den = cv2.GaussianBlur(w, (0, 0), 3.0)
        return np.clip(np.where(den > 0.05, num / np.maximum(den, 1e-3), med), lo, hi)

    # ------------------------------------------------------------------ events
    def _settled_events(self, t, bgr, g, lab) -> list[Event]:
        """One event per connected patch of changed cells. A patch fires when most of its cells have settled, and while
        nothing is moving or has just been covered by a person next to it (that would still be changing it)."""
        cfg = self.cfg
        calm = (self.since_person >= cfg.clear_s) & (self.since_moving >= cfg.still_s) & (self.cool <= 0)
        settled = (self.run >= cfg.settle_s) & calm & self.visible & self.changed
        if not settled.any():
            return []
        busy = ~calm & ~self.ignore & ~(self.dyn > self.cfg.dynamic_thr)
        m = cv2.morphologyEx(self.changed.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab_cc, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out: list[Event] = []
        done = np.zeros_like(settled)
        for i in range(1, n):
            patch = (lab_cc == i)
            comp = patch & self.changed
            k = int(comp.sum())
            if k == 0 or (comp & settled).sum() < 0.6 * k:
                continue
            if (_dilate(patch, 2) & busy).any() and float(self.run[comp].max()) < cfg.settle_s + cfg.patience_s:
                continue
            ridx = np.bincount(self.cell_region[comp] + 1, minlength=len(self.reg_list) + 1).argmax() - 1
            reg = self.reg_list[ridx] if ridx >= 0 else None
            p = {**(reg.params if reg else {})}
            if k < int(p.get("min_cells", cfg.min_cells)):
                continue
            share = k / max(int((self.cell_region == ridx).sum()), 1) if ridx >= 0 else 0.0
            done |= comp
            if share > cfg.max_region_share:
                continue
            ev = self._classify(t, comp, reg, bgr)
            if ev:
                if cfg.evidence_dir:
                    ev.evidence = write_evidence(cfg.evidence_dir, f"{self.cam}_{t:07.1f}_{ev.kind}", self.ref_bgr, bgr, ev.box, self.last_persons)
                out.append(ev)
        if done.any():
            self._replace_ref(_dilate(done, 1) & self.seeded & ~self.ignore, g, lab, bgr)
        return out

    def _classify(self, t, comp, reg, bgr) -> Event | None:
        cfg = self.cfg
        ys, xs = np.nonzero(comp)
        box = [xs.min() / self.gw, ys.min() / self.gh, (xs.max() + 1) / self.gw, (ys.max() + 1) / self.gh]
        f_before, f_after = float(fill_map(self.ref_bgr)[comp].mean()), float(fill_map(bgr)[comp].mean())
        d_fill = f_after - f_before
        run = float(self.run[comp].mean())
        area = int(comp.sum())
        kind_r = reg.kind if reg else "shelf"
        if kind_r == "ignore":
            return None
        score = float(self.score[comp].mean())
        sizes = min(1.0, area / 6.0)
        conf = float(np.clip(0.35 + 0.25 * min(run / (2 * cfg.settle_s), 1) + 0.2 * sizes + 0.2 * min((score - 1) / 1.5, 1), 0, 0.95))
        detail = {"area_cells": area, "fill_before": round(f_before, 2), "fill_after": round(f_after, 2),
                  "score": round(score, 2), "run_s": round(run, 1), "person_free_s": round(float(self.since_person[comp].min()), 1)}
        if kind_r == "floor":
            # medians: the rim of a patch has an edge whatever the patch is, its inside tells what it is
            tex_before, tex_after = float(np.median(cell_std(self.ref_g)[comp])), float(np.median(cell_std(self.cur_g)[comp]))
            d_fill = float(np.median(fill_map(bgr)[comp] - fill_map(self.ref_bgr)[comp]))
            detail["texture_before"], detail["texture_after"] = round(tex_before, 1), round(tex_after, 1)
            if d_fill < -0.08:
                kind = "floor_cleared"      # something that was on the floor is gone: an all-clear, not a hazard
            elif d_fill > 0.08 or tex_after > tex_before + 2 * cfg.flat_std:
                kind = "floor_object"       # a coloured or busy thing lies where the floor was bare
            else:
                kind = "floor_spill"        # bare floor that went darker or smoother: wet patch, film, shadow of something
        elif kind_r == "cooler_door":
            kind = "door_change" if abs(d_fill) < 0.08 else ("stock_down" if d_fill < 0 else "restock")
        else:
            kind = "layout_change" if abs(d_fill) < 0.06 else ("stock_down" if d_fill < 0 else "restock")
        return Event(self.cam, t, t - run, kind, reg.id if reg else "none", conf, [float(b) for b in box], detail)

    def _status_events(self, t, frac) -> list[Event]:
        return []

    def _state_events(self, t, bgr) -> list[Event]:
        cfg = self.cfg
        if self.golden_fill is None or t - self.last_snapshot < cfg.snapshot_s:
            return []
        self.last_snapshot = t
        cur = fill_map(self.ref_bgr)
        out: list[Event] = []
        snap = {"t": round(t, 1), "regions": {}}
        for i, reg in enumerate(self.reg_list):
            cells = (self.cell_region == i) & self.seeded
            if reg.kind not in ("shelf", "cooler_door") or cells.sum() < 4:
                continue
            stocked = cells & (self.golden_fill >= cfg.fill_hi)
            empty = stocked & (cur <= cfg.fill_lo)
            frac = float(empty.sum() / max(stocked.sum(), 1))
            snap["regions"][reg.id] = {"fill": round(float(cur[cells].mean()), 3), "golden_fill": round(float(self.golden_fill[cells].mean()), 3),
                                       "empty_frac": round(frac, 3), "stocked_cells": int(stocked.sum())}
            if stocked.sum() >= 6 and frac >= cfg.low_stock_frac:
                since = self.low_since.setdefault(reg.id, t)
                if t - since >= cfg.low_stock_s and reg.id not in self.low_open:
                    self.low_open.add(reg.id)
                    ys, xs = np.nonzero(empty)
                    box = [xs.min() / self.gw, ys.min() / self.gh, (xs.max() + 1) / self.gw, (ys.max() + 1) / self.gh]
                    out.append(Event(self.cam, t, since, "low_stock", reg.id, float(np.clip(0.4 + frac, 0, 0.9)), box,
                                     {"empty_frac": round(frac, 2), "stocked_cells": int(stocked.sum())}))
            else:
                self.low_since.pop(reg.id, None)
                if frac < cfg.low_stock_frac * 0.5:
                    self.low_open.discard(reg.id)
        self.snapshots.append(snap)
        return out
