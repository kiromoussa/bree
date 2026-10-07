"""Concealment cue from the item cameras: after a take, is the item still seen on that person?

Input (nothing from ground truth): the looks of the held-item detector per item camera (bree.concealment.scan), the
calibration of those cameras, the floor tracks (bree.track.floor), and the fused shelf acts with who did each.

  1. A detection that is not shelf stock, differs from the shelf picture and does not lie on a slot of its own product
     is a held item. Its line of sight passes a tracked person's body axis at hand height: that person holds it. Hand
     boxes are given to people the same way.
  2. Each take of a person opens a window that ends at their next take, when they reach the register zone, or when
     their track ends. Inside it: when was an item last seen on them (`t_last`), and in how many moments after that
     did an item camera see a hand of theirs with no item on them (`empty_bins`). A put act of theirs in the window
     means the item went back to the shelf. The last sightings also say how far the item came down (`y_drop`): a
     pocket is lower than a carrying hand.
  3. `score_take` turns the features into a probability with a small logistic model fitted on TRAIN-seed clips
     (scripts/conceal/fit.py writes model.json next to this file); without a model file a fixed rule is used.
  4. Per shopper the score accumulates over the visit: one minus the product of (1 - what each take adds), raised by
     a shelf sweep (several takes close in time and place) and by leaving without passing the register zone. What a
     take adds is its probability over that of a take with no evidence. Staff are left out first. A cue is written
     for the ledger against a take at or above the operating point, and against the strongest take of a shopper whose
     score is at or above `shopper_bar`. Count reconciliation stays the theft signal: the cue only says "this item
     was hidden". The tier is then the ledger's (its own rule and caps) or, when switched on, bree.concealment.tier
     (unpaid AND concealed with a high shopper score AND identity not uncertain).

`conceal_cues(...)` is what the pipeline calls. Each cue is the dict bree.events.shelf.store_events already takes:
{"t", "person_id", "sku_id", "conf", "source": "item_camera", "why": {...}}.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURES = ("cusum", "empty_bins", "seen_after", "seen_before", "empty_cams", "tail_s", "y_drop", "y_last", "back_m", "hand_back_m", "shelf_after", "item_after")
MODEL = Path(__file__).with_name("model.json")


@dataclass
class ConcealConfig:
    conf: float = 0.4            # held-item detection confidence
    fg: float = 0.5              # share of its box that differs from the shelf picture
    hand_conf: float = 0.4       # hand box confidence
    slot_k: float = 0.6          # a detection within this many item sizes of a slot of its own product is stock, not a held item
    reach_m: float = 0.6         # a held item or hand belongs to the person whose body axis its line of sight passes this close
    margin_m: float = 0.2        # ... with nobody else this much closer or less
    y_range: tuple[float, float] = (0.35, 1.75)   # height of that closest point, metres
    stale_s: float = 1.0         # a floor position older than this places nobody
    lead_s: float = 0.3          # the window of a take starts this long after it
    before_next_s: float = 1.5   # ... and ends this long before their next take (the reach, an opening cooler door)
    register_m: float = 1.3      # ... or this close to where customers stand to pay
    put_before_s: float = 0.0    # a put act of that person from this long before the item was last seen (0: an item seen in the hand after a put was not put back by it; 1.0 lost one true cue and no false one on the six tuning clips)
    put_grace_s: float = 1.0     # ... to this long after the window is the item going back to the shelf
    put_after_take_s: float = 2.0   # so is a put of the same product at least this long after the take, whenever the item was last seen
    two_hand_w: float = 2.2      # weight of a "no item" moment in which one camera saw two hands of theirs (1.0: as any other). An item is sighted in 0.60 of such moments of a carrying shopper against 0.35 with one hand seen, six tuning clips)
    seen_cost: float = 6.0       # one moment with an item seen on them cancels this many with a hand seen and no item (measured on TRAIN-seed clips: an item is sighted in 0.47 of the moments of a carrying shopper and 0.01 of an empty-handed one, scripts/conceal/rates.py)
    min_item_bins: int = 2       # the item has to be seen in the hand this many moments after the take before its absence means anything
    hand_border_px: float = 60.0 # a hand this close to the edge of the picture is not a moment (what it holds may be outside)
    back_s: float = 2.5          # the hand is looked for at that slot for this long after the item was last seen
    back_m: float = 0.25         # the item was last seen this close to the line of sight of the slot it came from: it went back
    border_px: float = 4.0       # a detection cut off by the edge of the picture is not used
    min_cusum: float = 8.0       # fixed rule (no model file): hand-seen-and-no-item moments, less seen_cost for each item sighting among them
    shopper_bar: float = 0.5     # the per-shopper score at or above which bree.concealment.tier may raise a record to alert, and the shopper's strongest take becomes a cue
    p_cue: float = 0.6           # a take at or above this becomes a cue for the ledger: the operating point, set on the six tuning clips (7 true, 0 false there)
    sweep_n: int = 3             # shelf sweep: this many takes by one person
    sweep_s: float = 8.0         # ... within this time
    sweep_m: float = 1.5         # ... and distance
    sweep_score: float = 0.3     # what a sweep adds to the shopper's score
    bypass_score: float = 0.15   # what leaving without entering the register zone adds, for a shopper with a take


@dataclass
class Sighting:
    t: float
    cam: str
    u: float
    v: float
    diag: float
    conf: float
    sku: str | None
    y: float          # height of the point of the line of sight nearest the person's body axis
    off: float        # how far that point is from the axis
    at_shelf: float | None = None     # hands only, from bree.concealment.where: the hand is at a shelf or counter
    item: float | None = None         # ... and an item is visible in it


@dataclass
class Take:
    person_id: int
    t: float
    sku: str | None
    stop: float                  # end of the window
    stop_why: str                # "next_take", "register", "track_end"
    feats: dict = field(default_factory=dict)
    where: str = ""              # "shelf" (a put of theirs) or "not_seen" (no item seen in the hand): never scored
    p: float = 0.0


def load_looks(folder: Path) -> dict[str, list[dict]]:
    return {f.stem[len("looks_"):]: [json.loads(x) for x in f.read_text().splitlines() if x] for f in sorted(Path(folder).glob("looks_*.jsonl"))}


def _axis(cam, u: float, v: float, xz: np.ndarray) -> tuple[float, float]:
    """(horizontal distance, height) of the point of the line of sight through (u, v) nearest the vertical line at xz."""
    d, c = cam.ray(u, v), cam.C
    h = d[[0, 2]]
    n = float(h @ h)
    s = float((xz - c[[0, 2]]) @ h) / n if n > 1e-9 else 0.0
    if s <= 0:
        return float("inf"), 0.0
    return float(np.linalg.norm(c[[0, 2]] + s * h - xz)), float(c[1] + s * d[1])


def _owner(cam, u: float, v: float, t: float, people: list, cfg: ConcealConfig) -> tuple[int, float, float] | None:
    best = []
    for p in people:
        got = p.at(t)
        if got is None or got[1] > cfg.stale_s:
            continue
        off, y = _axis(cam, u, v, got[0])
        if off <= cfg.reach_m and cfg.y_range[0] <= y <= cfg.y_range[1]:
            best.append((off, p.id, y))
    best.sort()
    if not best or (len(best) > 1 and best[1][0] - best[0][0] < cfg.margin_m):
        return None
    return best[0][1], best[0][2], best[0][0]


def _slot_lines(cam, layout: dict) -> dict[str, np.ndarray]:
    """sku -> (N, 5): each slot of that product in this camera's picture, as the line from its front to its back
    (u0, v0, u1, v1) and the size of one item there in pixels."""
    out: dict[str, list] = {}
    sizes = {s["id"]: s.get("size") for s in layout.get("skus", [])}
    for s in layout.get("slots", []):
        face, pos = np.asarray(s["face"], float), np.asarray(s["position"], float)
        px, z = cam.project(np.array([face, 2 * pos - face]))
        if z.min() <= 0.05:
            continue
        size = sizes.get(s.get("skuId")) or s["size"]
        out.setdefault(s.get("skuId"), []).append([*px[0], *px[1], cam.f * float(np.hypot(size[0], size[1])) / float(z[0])])
    return {k: np.array(v, float) for k, v in out.items()}


def _at_slot(lines: np.ndarray | None, u: float, v: float, diag: float, k: float) -> bool:
    if lines is None:
        return False
    a, b, p = lines[:, :2], lines[:, 2:4], np.array([u, v])
    ab = b - a
    w = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    d = np.linalg.norm(a + w[:, None] * ab - p, axis=1)
    return bool((d <= k * np.maximum(lines[:, 4], diag)).any())


def sightings(looks: dict[str, list[dict]], cams: dict, people: list, fps: float, cfg: ConcealConfig, layout: dict | None = None) -> tuple[dict[int, list[Sighting]], dict[int, list[Sighting]]]:
    """-> (held items per person, hands per person), each by time. With a layout, a detection lying on a slot of its own
    product (stock seen past a person or through an open door) is not a held item."""
    held: dict[int, list[Sighting]] = {}
    hands: dict[int, list[Sighting]] = {}
    for cam_id, rows in looks.items():
        cam = cams.get(cam_id)
        if cam is None:
            continue
        lines = _slot_lines(cam, layout) if layout else {}
        for r in rows:
            t = r["f"] / fps
            for x0, y0, x1, y1, cf, sku, fg, _moved, stock in r["items"]:
                if stock or cf < cfg.conf or fg < cfg.fg:
                    continue
                u, v = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = cam.resolution
                if w and (x0 < cfg.border_px or y0 < cfg.border_px or x1 > w - cfg.border_px or y1 > h - cfg.border_px):
                    continue
                if _at_slot(lines.get(sku), u, v, math.hypot(x1 - x0, y1 - y0), cfg.slot_k):
                    continue
                if (own := _owner(cam, u, v, t, people, cfg)) is not None:
                    held.setdefault(own[0], []).append(Sighting(t, cam_id, u, v, math.hypot(x1 - x0, y1 - y0), cf, sku, own[1], own[2]))
            for x0, y0, x1, y1, cf, *more in r["hands"]:
                w, h = cam.resolution
                if cf < cfg.hand_conf or (w and (x0 < cfg.hand_border_px or y0 < cfg.hand_border_px or x1 > w - cfg.hand_border_px or y1 > h - cfg.hand_border_px)):
                    continue      # a hand at the edge of the picture: what it holds may be outside it
                u, v = (x0 + x1) / 2, (y0 + y1) / 2
                if (own := _owner(cam, u, v, t, people, cfg)) is not None:
                    hands.setdefault(own[0], []).append(Sighting(t, cam_id, u, v, math.hypot(x1 - x0, y1 - y0), cf, None, own[1], own[2], *more[:2]))
    for d in (held, hands):
        for v_ in d.values():
            v_.sort(key=lambda s: s.t)
    return held, hands


def _second(v: list) -> float:
    v = sorted(v, reverse=True)
    return float(v[1]) if len(v) > 1 else 0.0


def _to_line(cam, u: float, v: float, X) -> float:
    """Distance from the 3D point X to the line of sight through pixel (u, v)."""
    d, w = cam.ray(u, v), np.asarray(X, float) - cam.C
    return float(np.linalg.norm(w - (w @ d) * d))


def takes_of(people: list, held: dict, hands: dict, acts: list[dict], who: list, register_xz, cfg: ConcealConfig, cams: dict | None = None) -> list[Take]:
    out = []
    for p in people:
        mine = sorted(((float(a["t"]), a.get("sku_id"), a["kind"], a.get("point_3d")) for a, w in zip(acts, who) if w == p.id), key=lambda x: x[0])
        from_ = [x for _, _, k, x in mine if k == "take"]
        mine = [m[:3] for m in mine]
        ts = [(t, sku) for t, sku, k in mine if k == "take"]
        puts = [(t, sku) for t, sku, k in mine if k == "put"]
        for i, (tk, sku) in enumerate(ts):
            t_reg = next((t for t, x, z in p.path if t > tk and register_xz is not None and math.hypot(x - register_xz[0], z - register_xz[1]) <= cfg.register_m), None)
            ends = [(max(ts[i + 1][0] - cfg.before_next_s, tk), "next_take")] if i + 1 < len(ts) else []      # the reach for the next item is not in the window
            ends += [(t_reg, "register")] if t_reg is not None else []
            stop, why = min(ends + [(p.path[-1][0], "track_end")])
            # a sighting named as another product this person had already taken belongs to that take; any other one to this take
            H = [s for s in held.get(p.id, []) if tk + cfg.lead_s < s.t < stop and not any(x == s.sku != sku and t < s.t for j, (t, x) in enumerate(ts) if j != i)]
            # moments (detector looks) in which a camera saw a hand of theirs or an item on them, and whether an item was seen.
            # One false sighting must not hide a concealment and one missed sighting must not make one, so the split is
            # the moment after which "hand seen, no item" outweighs "item seen" the most (a CUSUM change point).
            G = [s for s in hands.get(p.id, []) if tk + cfg.lead_s < s.t < stop]
            seen_at = {s.t for s in H}
            M = sorted(seen_at | {s.t for s in G})
            n_in_cam: dict = {}
            for s in G:
                n_in_cam[s.t, s.cam] = n_in_cam.get((s.t, s.cam), 0) + 1
            both = {t for (t, _), k in n_in_cam.items() if k >= 2}
            gain = [(sum((cfg.two_hand_w if t in both else 1.0) if t not in seen_at else -cfg.seen_cost for t in M[j:]), j) for j in range(len(M) + 1)]
            cus, j = max(gain, key=lambda g: (g[0], g[1]))
            before = [s for s in H if s.t < M[j]] if j < len(M) else H
            t_last = before[-1].t if before else tk
            after = M[j:]
            tail = before[-3:]
            f = {"cusum": float(cus), "seen_before": len({s.t for s in before}), "seen_after": sum(t in seen_at for t in after), "empty_bins": sum(t not in seen_at for t in after),
                 "empty_cams": len({s.cam for s in G if s.t > t_last and s.t not in seen_at}), "t_last": t_last, "tail_s": stop - t_last,
                 "y_last": float(np.mean([s.y for s in tail])) if tail else 1.0,
                 "y_drop": (max(s.y for s in before[-8:]) - float(np.mean([s.y for s in tail]))) if tail else 0.0, "last_cam": before[-1].cam if before else None,
                 # how close the last sightings came to the slot the item was taken from (a put back ends there, a pocket does not)
                 "back_m": min((_to_line(cams[s.cam], s.u, s.v, from_[i]) for s in tail), default=9.0) if cams and from_[i] is not None else 9.0,
                 # ... and how close a hand of theirs came to that slot in the seconds after the item was last seen
                 # the hand crop classifier (when the looks carry it): was a hand of theirs at a shelf right after the item was last
                 # seen (second highest answer, so one odd crop does not decide), and in what share of the "no item" moments
                 # does it see an item in a hand after all
                 "shelf_after": _second([s.at_shelf for s in G if t_last - 0.5 <= s.t <= t_last + cfg.back_s and s.at_shelf is not None]) if before else 0.0,
                 "item_after": float(np.mean([max((s.item or 0.0) for s in G if s.t == t) >= 0.5 for t in after if t not in seen_at and any(s.t == t for s in G)] or [0.0])),
                 "hand_back_m": min((_to_line(cams[s.cam], s.u, s.v, from_[i]) for s in G if t_last - 0.5 <= s.t <= t_last + cfg.back_s), default=9.0) if cams and from_[i] is not None and before else 9.0}
            tk_ = Take(p.id, tk, sku, stop, why, f)
            # the item went back to the shelf: a put of theirs once it was last seen, or a put of this product once the hand
            # had time to come away from the shelf and go back. Or it was never seen in the hand at all.
            tk_.where = ("shelf" if f["back_m"] <= cfg.back_m or any((max(tk, t_last - cfg.put_before_s) < t or (x == sku and t >= tk + cfg.put_after_take_s)) and t <= stop + cfg.put_grace_s for t, x in puts)
                         else "not_seen" if f["seen_before"] < cfg.min_item_bins else "")
            out.append(tk_)
    return out


def load_model(path: Path | None = None) -> dict | None:
    path = Path(path) if path else MODEL
    return json.loads(path.read_text()) if path.exists() else None


def score_take(f: dict, model: dict | None, cfg: ConcealConfig) -> float:
    """Probability that the item of this take went to the body. A take with a `where` is never scored (the caller)."""
    if model is None:      # fixed rule: hands seen again and again with no item on the person
        return 0.9 if f["cusum"] >= cfg.min_cusum else 0.0
    x = np.array([math.log1p(float(f[k])) if k in model.get("log", []) else float(f[k]) for k in model["features"]])
    z = float(((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"]) + model["intercept"])
    return 1.0 / (1.0 + math.exp(-z))


def sweeps(acts: list[dict], who: list, cfg: ConcealConfig) -> dict[int, float]:
    """person -> time of a shelf sweep: sweep_n takes within sweep_s and sweep_m."""
    out: dict[int, float] = {}
    by: dict[int, list[dict]] = {}
    for a, w in zip(acts, who):
        if w is not None and a["kind"] == "take" and a.get("point_3d"):
            by.setdefault(w, []).append(a)
    for pid, ts in by.items():
        ts.sort(key=lambda a: a["t"])
        for i in range(len(ts) - cfg.sweep_n + 1):
            grp = ts[i:i + cfg.sweep_n]
            pts = np.array([a["point_3d"] for a in grp], float)
            if grp[-1]["t"] - grp[0]["t"] <= cfg.sweep_s and np.linalg.norm(pts - pts[0], axis=1).max() <= cfg.sweep_m:
                out[pid] = float(grp[-1]["t"])
                break
    return out


def analyse(looks: dict[str, list[dict]], cams: dict, people: list, acts: list[dict], who: list, layout: dict, fps: float,
            cfg: ConcealConfig | None = None, model: dict | None = None) -> list[Take]:
    """Every take of every customer (staff left out), with its features, `where` and probability."""
    cfg = cfg or ConcealConfig()
    people = [p for p in people if not getattr(p, "staff", False) and p.path]
    held, hands = sightings(looks, cams, people, fps, cfg, layout)
    reg = (layout.get("poi") or {}).get("register")
    out = takes_of(people, held, hands, acts, who, (reg[0], reg[2]) if reg else None, cfg, cams)
    for t in out:
        t.p = 0.0 if t.where else score_take(t.feats, model, cfg)
    return out


def conceal_cues(looks: dict[str, list[dict]], cams: dict, people: list, acts: list[dict], who: list, layout: dict, fps: float,
                 cfg: ConcealConfig | None = None, model: dict | None | str = "default") -> tuple[list[dict], dict[int, dict]]:
    """-> (cues for bree.events.shelf.store_events(conceal=...), per-person concealment score).

    looks: bree.concealment.cue.load_looks(folder). cams: {camera id: bree.calib.camera.Camera} of the item cameras.
    people: FloorTracker.people(). acts, who: the fused shelf acts and the person each was given (associate)."""
    cfg = cfg or ConcealConfig()
    model = load_model() if model == "default" else model
    takes = analyse(looks, cams, people, acts, who, layout, fps, cfg, model)
    sw = sweeps(acts, who, cfg)
    by = {p.id: p for p in people}
    score: dict[int, dict] = {}
    cues = []
    # what one take adds to the shopper's score is its probability over that of a take with no evidence at all (a cusum
    # of 0), so that many plain takes do not add up to a score
    p0 = score_take({k: 0.0 for k in FEATURES}, model, cfg)
    for pid in {t.person_id for t in takes}:
        mine = [t for t in takes if t.person_id == pid]
        bypass = all(t.stop_why != "register" for t in mine)
        s = 1.0 - math.prod(1.0 - max(0.0, (t.p - p0) / (1.0 - p0)) for t in mine) * (1.0 - (cfg.sweep_score if pid in sw else 0.0)) * (1.0 - (cfg.bypass_score if bypass else 0.0))
        top = max(mine, key=lambda t: t.p)
        cued = [t for t in mine if t.p >= cfg.p_cue or (s >= cfg.shopper_bar and t is top and t.p > p0)]
        score[pid] = {"score": round(s, 3), "takes": len(mine), "concealed": len(cued), "sweep_t": sw.get(pid), "bypassed_register": bypass}
        for t in cued:
            got = by[pid].at(t.feats["t_last"])
            cues.append({"t": round(t.feats["t_last"] + 0.3, 2), "person_id": pid, "sku_id": t.sku, "conf": round(max(t.p, cfg.p_cue), 3), "source": "item_camera",
                         "at": [round(float(v), 2) for v in got[0]] if got is not None else None,
                         "why": {"take_t": t.t, "window": t.stop_why, "p_take": round(t.p, 3), "shopper_score": round(s, 3), **{a: (round(b, 3) if isinstance(b, float) else b) for a, b in t.feats.items()}}})
    return sorted(cues, key=lambda c: c["t"]), score
