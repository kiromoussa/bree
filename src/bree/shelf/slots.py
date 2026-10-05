"""Per-slot occupancy from the SKU detector and the planogram: which item of the row is at the front of a slot.

A slot holds a row of `depthCount` items of one SKU, front to back. Calibration and the planogram give the image box
of the item at every position of that row. The SKU detector (run on the moving regions by bree.shelf.hand) reports
stock items on the shelf with tight boxes, so the detection that fits "position k of slot S" says the k items in
front of it are gone. `SlotWatch` keeps that index per slot. When it moves and stays there, k went up by n: a take of
n units; down by n: a put of n units.

Unlike the pixel comparison (bree.shelf.diff) this reads the slot while an arm or a body is still in the picture, as
long as the new front item itself is in view, and it does not care about lighting. It is blind where the next item
sits right behind the front one in the image (a camera facing the shelf square on): there the pixel comparison is
the cue. No person box, no ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bree.shelf.diff import ShelfDiff, item_corners, slot_sigma


@dataclass
class SlotConfig:
    max_depth: int = 6          # row positions modelled per slot
    step: int = 2               # the front can move this many positions between two looks
    iou: float = 0.6            # a detection is the item at a row position: IoU with the predicted box
    iou_new: float = 0.7        # ... to move the front to another position
    old_max: float = 0.5        # and no other detection may fit the old front position this well any more
    margin: float = 0.18        # and the detection fits the new position this much better than the old one
    confirm: int = 3            # looks in a row at the new position, with the same box
    jitter: float = 0.06        # "the same box": corners within this share of the box size (plus 2 px)
    learn: int = 2              # looks to learn a slot's position the first time
    learn_iou: float = 0.45     # fit needed to learn (the predicted box is then corrected to the detector's box)
    front_prior: float = 0.05   # learning: per row position, how much a fit further back is marked down
    skip_zones: tuple = ("cooler",)   # behind a glass door the detector's boxes shift when the door swings (measured:
                                      # false takes on TRAIN clips 4900 and 4901); the pixel comparison reads these
    gone_iou: float = 0.3       # "the front item is not there": no detection fits its place this well
    gone_cover: float = 0.4     # "its place is in view": one detected item covers this share of it
    gone_looks: int = 0         # looks in a row; 0 = rule off (measured on TRAIN 4900 to 4902: 1 more take found, 11 false takes)
    min_seen: int = 3           # looks at a position before a move away from it is reported
    take_lead_s: float = 0.2    # the new front item is first seen about this long after the item left
    # ponytail: an empty slot (every item taken) has no detection and is not told apart from a hidden slot. Add an
    # "empty shelf" appearance check when a clip can empty a row.


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N, 4) with every box in b (M, 4)."""
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


class SlotWatch:
    def __init__(self, diff: ShelfDiff, skus: dict | None = None, cfg: SlotConfig | None = None):
        self.diff, self.cfg = diff, cfg or SlotConfig()
        self.rows: dict[int, np.ndarray] = {}        # slot index -> (K, 4) predicted boxes, full resolution
        self.by_sku: dict[str, list[int]] = {}
        for i in np.flatnonzero(diff.readable):
            s = diff.slots[int(i)]
            if s.get("zone") in self.cfg.skip_zones:
                continue
            n = np.asarray(s["normal"], float)
            depth = abs(float(np.asarray(s["size"], float) @ np.abs(n)))
            count = max(int(s.get("depthCount", 1)), 1)
            boxes = []
            for k in range(min(count, self.cfg.max_depth)):
                px, z = diff.cam.project(item_corners({**s, "face": np.asarray(s["face"], float) - n * k * depth / count}, (skus or {}).get(s.get("skuId"))))
                if (z <= 0.05).any():
                    break
                boxes.append([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()])
            if boxes:
                self.rows[int(i)] = np.asarray(boxes, float)
                self.by_sku.setdefault(s.get("skuId"), []).append(int(i))
        self.state: dict[int, int] = {}              # slot index -> row position at the front
        self.seen: dict[int, int] = {}               # frame the slot was last seen at its state
        self.looks: dict[int, int] = {}              # how many looks have seen it there
        self.gone: dict[int, tuple[int, int]] = {}   # slot index -> (looks in a row without its front item, first frame)
        self.pending: dict[int, tuple] = {}          # slot index -> (new position, looks, first frame, box)

    def update(self, f: int, boxes: np.ndarray, skus: list[str]) -> list[dict]:
        """One look: the detector's item boxes (full resolution) and SKU names at frame f. Returns shelf events."""
        c, out = self.cfg, []
        skus, boxes = np.asarray(skus), np.asarray(boxes, float).reshape(-1, 4)
        for sku in set(skus.tolist()) & set(self.by_sku):
            det = boxes[skus == sku]
            fits = {i: _iou(self.rows[i], det) for i in self.by_sku[sku]}          # (row positions, detections)
            # a detection that is the known front item of some slot cannot be the new front item of another
            owner = {i: m[self.state[i]] >= c.iou for i, m in fits.items() if i in self.state}
            self._learn(f, det, {i: m for i, m in fits.items() if i not in self.state},
                        np.any([own for own in owner.values()], axis=0) if owner else np.zeros(len(det), bool))
            for i, m in fits.items():
                cur = self.state.get(i)
                if cur is None or not m.size or i not in owner:        # not learned, or learned in this very look
                    continue
                known = np.zeros(len(det), bool)
                for j, own in owner.items():
                    if j != i:
                        known |= own
                free = np.where(known[None, :], 0.0, m)
                ks = list(range(max(cur - c.step, 0), min(cur + c.step, len(m) - 1) + 1))
                k, d = max(((k, int(free[k].argmax())) for k in ks), key=lambda kd: free[kd])
                if k == cur or free[k, d] - m[cur, d] < c.margin:
                    if m[cur].max() >= c.iou:          # the front item is where it was
                        self.seen[i], self.looks[i] = f, self.looks.get(i, 0) + 1
                        self.pending.pop(i, None)
                        self.gone.pop(i, None)
                    elif c.gone_looks and m[cur].max() < c.gone_iou and cur + 1 < len(self.rows[i]) and self._in_view(self.rows[i][cur], boxes):
                        # the front item is not there, yet items behind its place are in plain view: it was taken,
                        # and the next one of the row is hidden behind a neighbour (a camera looking along the shelf)
                        n, f0 = self.gone.get(i, (0, f))
                        if n + 1 < c.gone_looks:
                            self.gone[i] = (n + 1, f0)
                        else:
                            self.gone.pop(i, None)
                            if self.looks.get(i, 0) >= c.min_seen:
                                out.append({**self._event(i, cur, cur + 1, f0, f, 0.5), "iou_old": round(float(m[cur].max()), 3), "front_gone": True})
                            self.state[i], self.seen[i], self.looks[i] = cur + 1, f, 0
                    else:
                        self.gone.pop(i, None)
                    continue
                self.gone.pop(i, None)
                # a new front position: a good fit, and no other detection still at the old one
                if free[k, d] < c.iou_new or np.delete(m[cur], d).max(initial=0.0) >= c.old_max:
                    self.pending.pop(i, None)
                    continue
                pk, n, f0, box = self.pending.get(i, (k, 0, f, det[d]))
                n, f0 = (n + 1, f0) if pk == k and self._same(box, det[d]) else (1, f)
                if n < c.confirm:
                    self.pending[i] = (k, n, f0, det[d])
                    continue
                self.pending.pop(i, None)
                if self.looks.get(i, 0) >= c.min_seen:
                    out.append({**self._event(i, cur, k, f0, f, float(free[k, d])), "iou_old": round(float(m[cur].max()), 3)})
                self.state[i], self.seen[i], self.looks[i] = k, f, n
        return out

    def _in_view(self, box: np.ndarray, boxes: np.ndarray) -> bool:
        """Is the place of this box in plain view: some detected item covers a good part of it (nothing is in front)."""
        x0, y0 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
        x1, y1 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
        cover = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None) / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
        return bool((cover >= self.cfg.gone_cover).any())

    def _same(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.abs(a - b).max() <= self.cfg.jitter * max(a[2] - a[0], a[3] - a[1]) + 2)

    def _learn(self, f: int, det: np.ndarray, fits: dict[int, np.ndarray], taken: np.ndarray) -> None:
        """First sight of slots of one SKU: every detection goes to at most one slot (neighbouring slots of one SKU
        look like each other's back rows), as many slots as possible, front positions first. The predicted boxes of
        a learned slot are then moved onto what the detector draws for it."""
        from scipy.optimize import linear_sum_assignment
        c, idx = self.cfg, [i for i in fits]
        if not idx or not len(det):
            return
        prior = c.front_prior * np.arange(self.cfg.max_depth)
        best_k = {i: (fits[i] - prior[:len(fits[i]), None]).argmax(axis=0) for i in idx}
        val = np.array([[0.0 if taken[d] or fits[i][best_k[i][d], d] < c.learn_iou else 1.0 + fits[i][best_k[i][d], d] - prior[best_k[i][d]]
                         for d in range(len(det))] for i in idx])
        for r, d in zip(*linear_sum_assignment(-val)):
            i = idx[r]
            if val[r, d] <= 0:
                self.pending.pop(i, None)
                continue
            k = int(best_k[i][d])
            pk, n, f0, box = self.pending.get(i, (k, 0, f, det[d]))
            n = n + 1 if pk == k and self._same(box, det[d]) else 1
            if n < c.learn:
                self.pending[i] = (k, n, f, det[d])
                continue
            self.pending.pop(i, None)
            rows = self.rows[i]
            wh = np.c_[rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1], rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1]]
            self.rows[i] = rows + (det[d] - rows[k]) / wh[k] * wh
            self.state[i], self.seen[i], self.looks[i] = k, f, n

    def is_stock(self, sku: str, box, iou: float = 0.45) -> bool:
        """Is this detection an item standing in a slot of its SKU (at any row position), not one in a hand?"""
        return any(_iou(self.rows[i], np.asarray(box, float)[None, :4]).max() >= iou for i in self.by_sku.get(sku, []))

    def _event(self, i: int, old: int, new: int, f0: int, f: int, conf: float) -> dict:
        d, s, fps = self.diff, self.diff.slots[i], self.diff.fps
        kind = "take" if new > old else "put"
        t_end = f0 / fps
        t_start = min(self.seen.get(i, f0) / fps, t_end)
        t = max(t_end - self.cfg.take_lead_s, t_start)
        if kind == "take":      # the moment the slot stopped looking like the shelf picture, when that is in between
            t_px = float(np.median(d.last_same[d.ids == i])) / fps if d.last_same is not None and (d.ids == i).any() else -1.0
            t = t_px if t_start <= t_px <= t_end else t
        b = self.rows[i][min(old, new)]
        return {"camera_id": d.id, "t": round(t, 3), "t_start": round(t_start, 3), "t_end": round(t_end, 3), "kind": kind, "slot_id": s["id"],
                "sku_id": s.get("skuId"), "sku_conf": round(conf, 3), "count": abs(new - old), "source": "shelf_diff", "cue": "slot_state",
                "hand_px": [round(float(b[0] + b[2]) / 2, 1), round(float(b[1] + b[3]) / 2, 1)],
                "point_3d": [round(float(v), 3) for v in s["face"]], "point_sigma_m": slot_sigma(s),
                "evidence": {"before": None, "after": None, "frames": [int(self.seen.get(i, f0)), int(f0), int(f)]},
                "slots": [(s["id"], round(conf, 3))], "row": [int(old), int(new)]}
