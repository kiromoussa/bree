"""Who took it? Attach shelf events (the shared contract, made per camera without a person box) to store-wide people.

    result = associate(shelf_events, people, layout, cams=cameras)        # one Assoc per shelf event, same order

`people` are the identities of bree.track.floor.FloorTracker (anything with `.id`, `.at(t) -> ((x, z), seconds since
seen) | None`, and optionally `.kpts`, `.staff`, `.uncertain`, `.t_uncertain`). `layout` is the store layout with the
planogram (slots: id, face, normal). Nothing else: no ground truth, no appearance.

For each event and each person in the store at that time, a cost (smaller fits better, about a chi square):
  where they stand   in front of the shelf face (`stand_m` out along the slot's normal), level with the slot along the
                     shelf; someone on the other side of the gondola, or further than an arm from the point, is out;
  wrist              when an overhead camera has the person's wrist keypoints: how close the line of sight through
                     the wrist passes to the event's point.
Events close in time that share candidates are solved together: one person cannot take from two places further apart
than their arms and the walk between, and two takes in the same moment by one person cost a little more than one
each. An event is marked uncertain when the next best explanation is within `margin`, or when the person's identity
was already uncertain at that time. An event nobody fits stays unassigned.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

L_WRI, R_WRI = 9, 10


@dataclass
class AssocConfig:
    stand_m: float = 0.5              # how far in front of the shelf face a person stands when taking
    sigma_stand_m: float = 0.25
    sigma_along_m: float = 0.3        # along the shelf: a person takes from in front of themselves
    sigma_track_m: float = 0.15       # floor tracker error while the person is seen
    drift_mps: float = 0.7            # and per second since they were last seen
    max_unseen_s: float = 4.0         # a person not seen within this long of the event is not a candidate
    behind_m: float = 0.15            # further behind the shelf face than this: the other side, not them
    shoulder_m: float = 1.4
    reach_m: float = 1.5              # shoulder to the point: arm plus lean
    sigma_wrist_m: float = 0.25       # wrist line of sight to the point
    wrist_cap_m: float = 0.75
    wrist_conf: float = 0.4
    window_s: float = 0.5             # look at the person from this long before to this long after the event
    gate: float = 20.0                # a cost above this explains nothing
    margin: float = 4.0               # runner-up within this cost: uncertain (about 7 to 1 in likelihood)
    same_moment_s: float = 1.0
    same_moment_cost: float = 2.0     # one person, two different slots, same moment
    arm_span_m: float = 1.8
    walk_mps: float = 2.2
    group_gap_s: float = 1.5          # events closer than this are solved together
    max_group: int = 6
    staff: bool = False               # staff as candidates


@dataclass
class Assoc:
    person_id: int | None = None
    uncertain: bool = False
    candidates: list = field(default_factory=list)      # [(person id, cost)] best first, everyone under the gate
    margin: float = float("inf")                        # cost of the next best explanation minus this one
    why: str = ""
    cost: float | None = None


def slot_index(layout: dict) -> dict:
    return {s["id"]: s for s in layout.get("slots", [])}


def event_geometry(ev: dict, slots: dict) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    """(point xyz, shelf normal on the floor (x, z) or None, sigma of the point) of a shelf event."""
    s = slots.get(ev.get("slot_id")) if ev.get("slot_id") else None
    p = ev.get("point_3d")
    if p is None and s is None:
        return None
    pt = np.asarray(p if p is not None else s["face"], float)
    n = None
    if s is not None and s.get("normal") is not None:
        n = np.asarray(s["normal"], float)[[0, 2]]
        n = n / max(np.linalg.norm(n), 1e-9)
    return pt, n, float(ev.get("point_sigma_m") or 0.05)


def _wrist_m(person, cams: dict | None, pt: np.ndarray, t0: float, t1: float, cfg: AssocConfig) -> float | None:
    """Closest the line of sight through a wrist keypoint passes to the point, over the window. None: no wrist seen."""
    kp = getattr(person, "kpts", None)
    if not kp or not cams:
        return None
    best = None
    for t, cam, k in kp:
        if t < t0 or t > t1 or cam not in cams:
            continue
        c = cams[cam]
        for i in (L_WRI, R_WRI):
            if k[i, 2] < cfg.wrist_conf:
                continue
            d = c.ray(k[i, 0], k[i, 1])
            v = pt - c.C
            m = float(np.linalg.norm(v - (v @ d) * d))
            best = m if best is None else min(best, m)
    return best


def person_cost(ev: dict, geo, person, cams: dict | None, cfg: AssocConfig) -> tuple[float, str] | None:
    """Cost of `person` having made this shelf event, and a few words on why. None: cannot have."""
    pt, n, sig_pt = geo
    t = float(ev["t"])
    t0, t1 = max(float(ev.get("t_start", t)), t - 1.0) - cfg.window_s, min(float(ev.get("t_end", t)), t + 1.0) + cfg.window_s
    costs, last = [], ""
    for tau in np.unique(np.concatenate([np.linspace(t0, t1, 5), [t]])):
        got = person.at(float(tau))
        if got is None:
            continue
        xz, unseen = got
        if unseen > cfg.max_unseen_s:
            continue
        sp2 = cfg.sigma_track_m ** 2 + (cfg.drift_mps * unseen) ** 2 + sig_pt ** 2
        rel = np.asarray(xz, float) - pt[[0, 2]]
        reach = float(np.linalg.norm([rel[0], cfg.shoulder_m - pt[1], rel[1]]))
        if reach > cfg.reach_m + 2 * np.sqrt(sp2):
            continue
        if n is not None:
            dn, dl = float(rel @ n), float(rel @ np.array([-n[1], n[0]]))
            if dn < -cfg.behind_m - 2 * np.sqrt(sp2):
                continue
            c = (dn - cfg.stand_m) ** 2 / (cfg.sigma_stand_m ** 2 + sp2) + dl ** 2 / (cfg.sigma_along_m ** 2 + sp2)
            last = f"{dn:.2f} m in front, {abs(dl):.2f} m along"
        else:
            d = float(np.linalg.norm(rel))
            c = max(d - cfg.stand_m, 0.0) ** 2 / (cfg.sigma_along_m ** 2 + sp2)
            last = f"{d:.2f} m away"
        costs.append((c, last))
    if not costs:
        return None
    costs.sort(key=lambda x: x[0])
    c, why = costs[len(costs) // 2]                       # the median moment: someone walking past is near only briefly
    w = _wrist_m(person, cams, pt, t0, t1, cfg)
    if cams:
        wm = min(w, cfg.wrist_cap_m) if w is not None else 0.4      # no wrist seen: neither for nor against
        c += (wm / cfg.sigma_wrist_m) ** 2
        if w is not None:
            why += f", wrist line {w:.2f} m off"
    return (c, why) if c <= cfg.gate else None


def _conflict(e1: dict, g1, e2: dict, g2, cfg: AssocConfig) -> float:
    """Extra cost of ONE person having made both events."""
    dt, d = abs(float(e1["t"]) - float(e2["t"])), float(np.linalg.norm(g1[0] - g2[0]))
    if d > cfg.arm_span_m + cfg.walk_mps * dt:
        return 1e3
    same_place = e1.get("slot_id") is not None and e1.get("slot_id") == e2.get("slot_id")
    return cfg.same_moment_cost if dt <= cfg.same_moment_s and not same_place else 0.0


def associate(shelf: list[dict], people: list, layout: dict, cams: dict | None = None, cfg: AssocConfig | None = None) -> list[Assoc]:
    cfg = cfg or AssocConfig()
    slots = slot_index(layout)
    by_id = {p.id: p for p in people if cfg.staff or not getattr(p, "staff", False)}
    geo = [event_geometry(e, slots) for e in shelf]
    cand: list[dict[int, tuple[float, str]]] = []
    for e, g in zip(shelf, geo):
        cand.append({} if g is None else {pid: c for pid, p in by_id.items() if (c := person_cost(e, g, p, cams, cfg)) is not None})
    out = [Assoc() for _ in shelf]
    # groups: events close in time that share a candidate
    order = sorted(range(len(shelf)), key=lambda i: float(shelf[i]["t"]))
    groups: list[list[int]] = []
    for i in order:
        g = next((g for g in reversed(groups) if len(g) < cfg.max_group and any(
            abs(float(shelf[i]["t"]) - float(shelf[j]["t"])) <= cfg.group_gap_s and set(cand[i]) & set(cand[j]) for j in g)), None)
        (g.append(i) if g is not None else groups.append([i]))
    for g in groups:
        opts = [[None] + sorted(cand[i], key=lambda pid: cand[i][pid][0])[:4] for i in g]

        def total(choice) -> float:
            s = sum(cfg.gate if pid is None else cand[i][pid][0] for i, pid in zip(g, choice))
            for (a, pa), (b, pb) in itertools.combinations(zip(g, choice), 2):
                if pa is not None and pa == pb:
                    s += _conflict(shelf[a], geo[a], shelf[b], geo[b], cfg)
            return s
        scored = sorted(((total(ch), ch) for ch in itertools.product(*opts)), key=lambda x: x[0])
        best_cost, best = scored[0]
        for k, i in enumerate(g):
            a, pid = out[i], best[k]
            a.candidates = sorted(((p, round(c[0], 2)) for p, c in cand[i].items()), key=lambda x: x[1])
            other = next((s for s, ch in scored if ch[k] != pid and ch[k] is not None), None)   # someone else took this one
            a.margin = round(other - best_cost, 2) if other is not None else float("inf")
            if pid is None:
                a.why = "nobody in reach of the shelf at that time" if not cand[i] else "no explanation under the gate"
                continue
            a.person_id, a.cost, a.why = pid, round(cand[i][pid][0], 2), cand[i][pid][1]
            if a.margin < cfg.margin:
                rival = next(ch[k] for s, ch in scored if ch[k] != pid and ch[k] is not None)
                a.uncertain, a.why = True, a.why + f"; person {rival} fits almost as well (cost +{a.margin:.1f})"
            p = by_id[pid]
            tu = getattr(p, "t_uncertain", None)
            if getattr(p, "uncertain", None) and (tu is None or float(shelf[i]["t"]) >= tu):
                a.uncertain, a.why = True, a.why + f"; identity uncertain: {p.uncertain}"
    return out
