"""Scripted multi-shopper scenes for the association and the floor tracker. SYNTHETIC: geometry only, no images.

    python -m bree.track.scenes                 # 200 scenes: right-shopper rate of the association, tracker identity
    python -m bree.track.scenes --scenes 50 --seed 7 --write results/assoc_scripted.json

A scene is a small store (one gondola with shelves on both faces, a door, a counter) and 2 to 5 shoppers who walk
in, stand at a shelf, take an item and walk out. Around each take one of these is staged:
  alone         nobody else near
  same_shelf    a second shopper stands 0.7 to 1.3 m along the same shelf face (and may take something too)
  other_side    a second shopper stands right across the gondola, at the opposite face
  passer_by     a second shopper walks past behind the one taking
  shoulder      a second shopper stands 0.15 to 0.35 m from the one taking: nobody could tell from position, the
                right answer is "uncertain"
Shelf events are written in the shared contract (slot, time with jitter, 3D point with noise): the association never
sees who took what. Two ways to give it people:
  scripted tracks  the true paths with noise and gaps (tests the association alone)
  tracker          the true people are projected into two overhead cameras as boxes with keypoints, with missed
                   detections and a few second boxes, and bree.track.floor makes the identities (tests both)
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from bree.calib.camera import from_layout
from bree.track.associate import AssocConfig, associate
from bree.track.floor import FloorConfig, FloorTracker

FPS = 10.0
DOOR = (0.0, 6.5)
CASES = ("alone", "same_shelf", "other_side", "passer_by", "shoulder")


def layout() -> dict:
    slots = []
    for side, nx in (("L", -1.0), ("R", 1.0)):
        for i, z in enumerate(np.arange(-2.4, 2.41, 0.3)):
            for lv, y in enumerate((0.3, 0.8, 1.3)):
                slots.append({"id": f"G1{side}-S{lv}-{i}", "skuId": f"sku_{side}{lv}_{i}", "fixtureId": "G1", "zone": "gondola",
                              "face": [0.6 * nx, y, round(float(z), 2)], "normal": [nx, 0, 0]})
    cams = [{"id": "TRACK-A", "position": [-3.5, 3.0, 4.5], "yaw": math.atan2(-3.5, 4.5), "pitch": -0.85, "hfov": 110, "resolution": [1920, 1080], "kind": "overhead"},
            {"id": "TRACK-B", "position": [3.5, 3.0, -4.5], "yaw": math.atan2(3.5, -4.5), "pitch": -0.85, "hfov": 110, "resolution": [1920, 1080], "kind": "overhead"}]
    return {"store": {"width": 12, "depth": 13, "height": 3},
            "fixtures": [{"id": "ENTRANCE", "type": "door", "position": [DOOR[0], 1.1, DOOR[1]], "size": [2, 2.2, 0.05]},
                         {"id": "G1", "type": "gondola", "position": [0, 0.7, 0], "size": [1.2, 1.4, 5.4]},
                         {"id": "COUNTER", "type": "counter", "position": [4.5, 0.5, 4.0], "rotationY": -1.570796, "size": [2.4, 0.95, 0.8]}],
            "skus": [{"id": s["skuId"]} for s in slots], "slots": slots, "cameras": cams}


class Path2D:
    """Piecewise straight walk: [(t, x, z)] way points."""
    def __init__(self, pts):
        self.pts = sorted(pts)

    def at(self, t: float) -> np.ndarray | None:
        p = self.pts
        if t < p[0][0] - 1e-6 or t > p[-1][0] + 1e-6:
            return None
        i = min(max(bisect.bisect_right(p, (t, 1e9, 1e9)), 1), len(p) - 1)
        a, b = p[i - 1], p[i]
        w = 0.0 if b[0] == a[0] else min(max((t - a[0]) / (b[0] - a[0]), 0.0), 1.0)
        return np.array([a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])])


def _walk(t: float, a, b, speed: float = 1.2) -> float:
    return t + max(float(np.hypot(b[0] - a[0], b[1] - a[1])) / speed, 0.1)


def _visit(t_in: float, stand, dwell: float, side: float, rng) -> tuple[Path2D, float]:
    """Door -> end of the aisle on `side` -> stand point (dwell) -> back out, each on its own line (people do not walk
    through each other). Returns the path and when the dwell starts."""
    lane = lambda: (side * rng.uniform(1.5, 1.95), 3.6 + rng.uniform(-0.4, 0.4))      # noqa: E731
    behind = lambda: (side * rng.uniform(1.65, 1.9), stand[1])      # noqa: E731  walk down the aisle behind whoever stands at the shelf
    door_in, door_out = (rng.uniform(-0.7, 0.7), DOOR[1]), (rng.uniform(-0.7, 0.7), DOOR[1] + 0.5)
    pts, t, at = [(t_in, *door_in)], t_in, door_in
    for b in (lane(), behind(), stand):
        t = _walk(t, at, b)
        pts.append((t, *b))
        at = b
    t0 = t
    pts.append((t0 + dwell, *stand))
    t = t0 + dwell
    for b in (behind(), lane(), door_out):
        t = _walk(t, at, b)
        pts.append((t, *b))
        at = b
    return Path2D(pts), t0


def make_scene(seed: int, lay: dict | None = None) -> dict:
    """-> {"layout", "paths": {shopper: Path2D}, "shelf": [events], "truth": [{"shopper", "case", "expect_uncertain"}], "t_end"}."""
    rng = np.random.default_rng(seed)
    lay = lay or layout()
    slots = lay["slots"]
    paths: dict[int, Path2D] = {}
    shelf, truth = [], []
    t_in, n = 0.5, 0

    def take(shopper: int, slot: dict, t: float, case: str, expect_uncertain: bool = False) -> None:
        tt = t + rng.normal(0, 0.2)
        shelf.append({"camera_id": f"{slot['id'][:3]}-rail", "t": round(tt, 2), "t_start": round(tt - 0.4, 2), "t_end": round(tt + 0.4, 2), "kind": "take",
                      "slot_id": slot["id"], "sku_id": slot["skuId"], "sku_conf": 0.9, "count": 1, "source": "shelf_diff", "hand_px": None,
                      "point_3d": [round(float(v + rng.normal(0, 0.03)), 3) for v in slot["face"]], "point_sigma_m": 0.05,
                      "evidence": {"before": None, "after": None, "frames": []}})
        truth.append({"shopper": shopper, "case": case, "expect_uncertain": expect_uncertain, "t": tt})

    def stand_at(slot: dict, along: float = 0.0, out: float | None = None):
        nx = slot["normal"][0]
        return (slot["face"][0] + nx * (out if out is not None else rng.uniform(0.35, 0.65)), slot["face"][2] + along + rng.uniform(-0.1, 0.1))

    for _ in range(int(rng.integers(1, 4))):          # 1 to 3 staged takes, each with its own cast
        case = CASES[int(rng.integers(len(CASES)))]
        slot = slots[int(rng.integers(len(slots)))]
        side, dwell = slot["normal"][0], rng.uniform(2.5, 4.0)
        n += 1
        a = n
        paths[a], t0 = _visit(t_in, stand_at(slot), dwell, side, rng)
        t_take = t0 + dwell / 2
        take(a, slot, t_take, case, expect_uncertain=case == "shoulder")
        if case != "alone":
            n += 1
            b = n
            if case == "same_shelf":
                off = rng.choice([-1, 1]) * rng.uniform(0.7, 1.3)
                other = min((s for s in slots if s["normal"] == slot["normal"]), key=lambda s: abs(s["face"][2] - (slot["face"][2] + off)) + 0.01 * abs(s["face"][1] - 0.8))
                paths[b], tb = _visit(t_in + 1.5, stand_at(other), dwell, side, rng)
                if rng.random() < 0.6:
                    take(b, other, tb + dwell / 2 + rng.uniform(-0.5, 0.5), "same_shelf")
            elif case == "other_side":
                other = next(s for s in slots if s["normal"][0] == -side and abs(s["face"][2] - slot["face"][2]) < 0.01 and s["face"][1] == slot["face"][1])
                paths[b], _ = _visit(t_in + 1.2, stand_at(other), dwell + 0.5, -side, rng)
            elif case == "passer_by":
                x = slot["face"][0] + side * rng.uniform(1.05, 1.3)
                z0, tp = slot["face"][2], t_take - 0.2
                paths[b] = Path2D([(tp - (3.6 - z0 + 3) / 1.2 - 3, *DOOR), (tp - (3.6 - z0) / 1.2, x, 3.6), (tp + (z0 + 3.0) / 1.2, x, -3.0), (tp + (z0 + 3.0) / 1.2 + 6, DOOR[0], DOOR[1] + 0.5)])
            else:       # shoulder to shoulder
                st = paths[a].at(t_take)
                off = rng.choice([-1, 1]) * rng.uniform(0.15, 0.35)
                paths[b], _ = _visit(t_in + 1.4, (st[0], st[1] + off), dwell, side, rng)
        t_in += rng.uniform(1.5, 14.0)
    return {"layout": lay, "paths": paths, "shelf": shelf, "truth": truth, "t_end": max(p.pts[-1][0] for p in paths.values()) + 3.0}


class ScriptedTrack:
    """A person the way the association wants one: the true path with noise, and stretches where nobody saw them."""
    def __init__(self, pid: int, path: Path2D, rng, sigma_m: float = 0.08, gap_p: float = 0.02):
        self.id, self.path, self.kpts, self.staff, self.uncertain, self.t_uncertain = pid, [], [], False, None, None
        skip = 0
        for t in np.arange(path.pts[0][0], path.pts[-1][0], 1 / FPS):
            if skip > 0:
                skip -= 1
                continue
            if rng.random() < gap_p:
                skip = int(rng.integers(3, 15))
                continue
            x, z = path.at(float(t)) + rng.normal(0, sigma_m, 2)
            self.path.append((round(float(t), 2), float(x), float(z)))
        self.state, self.t_exit, self.born = "exited", self.path[-1][0], "door"

    at = None       # set below: the same interpolation as the tracker's identities


from bree.track.floor import Track as _Track       # noqa: E402
ScriptedTrack.at = _Track.at


def detections(scene: dict, cams: dict, rng, miss_p: float = 0.12, second_box_p: float = 0.02, px_sigma: float = 2.0) -> dict[str, list[list[dict]]]:
    """The true people as each overhead camera's detector would give them: box and 17 keypoints (shoulders, hips,
    wrists and ankles set; the rest zero confidence), some missed, a few detected twice. A person behind the gondola
    from a camera (the line of sight to the hips passes through it) is not detected by that camera."""
    def hidden(cam, hip) -> bool:
        c = cam.C
        for s in np.linspace(0.05, 0.95, 19):
            p = c + s * (hip - c)
            if abs(p[0]) < 0.6 and abs(p[2]) < 2.7 and p[1] < 1.4:
                return True
        return False
    n = int(scene["t_end"] * FPS)
    out = {c: [[] for _ in range(n)] for c in cams}
    for f in range(n):
        t = f / FPS
        for pid, path in scene["paths"].items():
            xz = path.at(t)
            if xz is None:
                continue
            body = {"head": 1.7, "sho": 1.42, "hip": 0.93, "foot": 0.0}
            for cid, cam in cams.items():
                if rng.random() < miss_p or hidden(cam, np.array([xz[0], 0.93, xz[1]])):
                    continue
                px, depth = cam.project(np.array([[xz[0], h, xz[1]] for h in body.values()]))
                if (depth <= 0.1).any() or not cam.in_frame(px[2:3], depth[2:3]).all():
                    continue
                px = px + rng.normal(0, px_sigma, px.shape)
                k = np.zeros((17, 3))
                for idx, row, dx in ((5, 1, -8), (6, 1, 8), (11, 2, -6), (12, 2, 6), (9, 2, -12), (10, 2, 12), (15, 3, -5), (16, 3, 5)):
                    k[idx] = [px[row, 0] + dx, px[row, 1], 0.9]
                x0, x1 = px[:, 0].min() - 15, px[:, 0].max() + 15
                det = {"bbox": [float(x0), float(px[:, 1].min()), float(x1), float(px[:, 1].max())], "conf": 0.8, "kpts": k.tolist()}
                out[cid][f].append(det)
                if rng.random() < second_box_p:
                    out[cid][f].append({**det, "bbox": [v + 6 for v in det["bbox"]], "kpts": (k + [5, 5, 0]).tolist()})
    return out


def run_scene(seed: int, tracker: bool = False, acfg: AssocConfig | None = None, fcfg: FloorConfig | None = None) -> dict:
    scene = make_scene(seed)
    rng = np.random.default_rng(10_000 + seed)
    lay = scene["layout"]
    ident = None
    if tracker:
        cams = {c["id"]: from_layout(c) for c in lay["cameras"]}
        dets = detections(scene, cams, rng)
        tr = FloorTracker(cams, layout=lay, cfg=fcfg)
        for f in range(len(next(iter(dets.values())))):
            tr.update(f / FPS, {c: dets[c][f] for c in dets})
        tr.finish(scene["t_end"])
        people = tr.people()

        def who(pid, t):        # the true shopper an identity is nearest to at time t (scoring only)
            k = next(p for p in people if p.id == pid)
            got = k.at(t)
            near = [(float(np.linalg.norm(got[0] - xz)), s) for s, path in scene["paths"].items() if got is not None and (xz := path.at(t)) is not None]
            return min(near)[1] if near and min(near)[0] < 0.6 else None
        owners = {p.id: Counter(w for t, _, _ in p.path[::5] if (w := who(p.id, t)) is not None) for p in people}
        ident = {"shoppers": len(scene["paths"]), "ids": len(people), "exits": sum(p.state == "exited" for p in people),
                 "mixed": sum(1 for c in owners.values() if c and c.most_common(1)[0][1] < 0.9 * sum(c.values())),
                 "mixed_not_marked": sum(1 for p in people if (c := owners[p.id]) and c.most_common(1)[0][1] < 0.9 * sum(c.values()) and not p.uncertain)}
    else:
        cams = None
        people = [ScriptedTrack(pid, path, rng) for pid, path in scene["paths"].items()]
        who = lambda pid, t: pid      # noqa: E731
    res = associate(scene["shelf"], people, lay, cams=None, cfg=acfg)
    rows = []
    for g, a in zip(scene["truth"], res):
        got = who(a.person_id, g["t"]) if a.person_id is not None else None
        rows.append({"case": g["case"], "expect_uncertain": g["expect_uncertain"], "assigned": a.person_id is not None, "right": got == g["shopper"], "uncertain": a.uncertain})
    return {"seed": seed, "rows": rows, "identity": ident}


def summarise(runs: list[dict]) -> dict:
    rows = [r for run in runs for r in run["rows"]]

    def stats(rs) -> dict:
        n = len(rs)
        sure = [r for r in rs if r["assigned"] and not r["uncertain"]]
        return {"events": n, "right_shopper": sum(r["right"] for r in rs), "right_shopper_rate": round(sum(r["right"] for r in rs) / n, 3) if n else None,
                "right_and_sure": sum(r["right"] for r in sure), "wrong_and_sure": sum(not r["right"] for r in sure),
                "marked_uncertain": sum(r["uncertain"] for r in rs), "wrong_and_uncertain": sum(1 for r in rs if r["assigned"] and r["uncertain"] and not r["right"]),
                "unassigned": sum(not r["assigned"] for r in rs)}
    clear = [r for r in rows if not r["expect_uncertain"]]
    out = {"scenes": len(runs), "all": stats(rows), "clear_cases": stats(clear), "by_case": {c: stats([r for r in rows if r["case"] == c]) for c in CASES}}
    ids = [run["identity"] for run in runs if run["identity"]]
    if ids:
        out["identity"] = {"shoppers": sum(i["shoppers"] for i in ids), "ids": sum(i["ids"] for i in ids), "ids_per_shopper": round(sum(i["ids"] for i in ids) / sum(i["shoppers"] for i in ids), 3),
                           "exits_seen": sum(i["exits"] for i in ids), "ids_covering_two_shoppers": sum(i["mixed"] for i in ids),
                           "of_those_not_marked_uncertain": sum(i["mixed_not_marked"] for i in ids)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write")
    args = ap.parse_args(argv)
    res = {"label": "SYNTHETIC scripted scenes (geometry only, no images)",
           "scripted_tracks": summarise([run_scene(args.seed + i) for i in range(args.scenes)]),
           "floor_tracker": summarise([run_scene(args.seed + i, tracker=True) for i in range(args.scenes)])}
    print(json.dumps(res, indent=1))
    if args.write:
        Path(args.write).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
