"""Tiny synthetic stand-in for an Isaac Sim run, so the sim-eval chain can be tested without a GPU.

TOY DATA. It writes the folder shape the Isaac kit documents (batch_00/ira_output/<camera prim>/
rgb_NNNN.png + camera_params_NNNN.json, batch_00/config.yaml, events.jsonl) plus layout.json, but the
frames are flat OpenCV drawings in the toy palette (bree.detect.toy), so the toy colour detector
can read them. Real Isaac output may differ in folder nesting, file names, frame numbering and
matrix convention; this only proves adapter -> pipeline -> scorer agree with each other.

An 8 x 6 m store, three cameras (overhead, cooler, shelf), three shoppers:
  Character     candy bar from the gondola, pays
  Character_01  energy drink from the cooler, conceals it, walks out   (the theft)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from bree.detect.toy import HAND_COLOR, PERSON_COLORS, PRODUCT_COLORS
from bree.sim.isaac.convert import hull, project
from bree.sim.isaac_adapter import camera_params_from_layout, local_to_world, prim_name, to_usd
from bree.sim.toy_render import BG, COOLER, COUNTER, DOORC, SHELF

SIM_FPS, WRITE_EVERY = 30, 3            # as the kit's scenario.yaml: 10 fps written
SPEED = 1.3                             # m / s
REACH_S, CONCEAL_S, PAY_S, PAID_AT_S = 1.6, 1.2, 6.0, 4.5
FILL = {"gondola": SHELF, "cooler_door": COOLER, "counter": COUNTER, "door": DOORC}

LAYOUT = {
    "version": 1, "units": "meters", "up": "y", "store": {"width": 8.0, "depth": 6.0, "height": 3.0},
    # The door is on the back wall so every camera sees a shopper whole on their first frame (the pipeline
    # fixes a person's cross-camera id from the first box it gets; a box cut by the frame edge lands
    # the feet in the wrong place).
    "fixtures": [
        {"id": "cooler-1", "type": "cooler_door", "position": [-2.0, 1.0, -2.65], "rotationY": 0.0, "size": [2.0, 2.0, 0.7]},
        {"id": "gondola-1", "type": "gondola", "position": [0.3, 0.9, -1.0], "rotationY": 0.0, "size": [2.4, 1.8, 0.8]},
        {"id": "counter", "type": "counter", "position": [2.4, 0.5, 0.4], "rotationY": 0.0, "size": [1.6, 1.0, 0.6]},
        {"id": "door-back", "type": "door", "position": [3.0, 1.05, -3.0], "rotationY": 0.0, "size": [1.2, 2.1, 0.1]},
    ],
    # SKU ids are the toy detector's class names, so its detections map straight onto the catalog
    "skus": [
        {"id": "soda_bottle", "name": "Soda 20 oz", "kind": "bottle", "size": [0.07, 0.22, 0.07], "price": 2.49},
        {"id": "energy_drink", "name": "Energy drink", "kind": "can", "size": [0.066, 0.16, 0.066], "price": 3.99},
        {"id": "candy_bar", "name": "Candy bar", "kind": "box", "size": [0.12, 0.04, 0.05], "price": 1.79},
        {"id": "chips_bag", "name": "Chips", "kind": "bag", "size": [0.2, 0.28, 0.08], "price": 2.29},
    ],
    # top-shelf slots: seen from behind and above, the product is then clear of the shopper's head
    "slots": [
        {"id": "C1-S1", "skuId": "soda_bottle", "fixtureId": "cooler-1", "position": [-2.5, 1.65, -2.4], "size": [0.6, 0.3, 0.2], "facings": 3, "depthCount": 2},
        {"id": "C1-S2", "skuId": "energy_drink", "fixtureId": "cooler-1", "position": [-1.5, 1.65, -2.4], "size": [0.6, 0.3, 0.2], "facings": 3, "depthCount": 2},
        {"id": "G1-S1", "skuId": "candy_bar", "fixtureId": "gondola-1", "position": [-0.3, 1.65, -0.7], "size": [0.6, 0.3, 0.2], "facings": 3, "depthCount": 2},
        {"id": "G1-S2", "skuId": "chips_bag", "fixtureId": "gondola-1", "position": [0.9, 1.65, -0.7], "size": [0.6, 0.3, 0.2], "facings": 3, "depthCount": 2},
    ],
    # far enough back that people are about toy-clip sized (the toy detector's thresholds are in pixels)
    "cameras": [
        {"id": "cam-overhead-01", "position": [3.8, 2.9, 2.9], "yaw": math.radians(25), "pitch": math.radians(-38), "hfov": 100, "resolution": [960, 540], "kind": "overhead"},
        {"id": "cam-cooler-01", "position": [-2.0, 2.8, 1.5], "yaw": math.radians(15), "pitch": math.radians(-20), "hfov": 80, "resolution": [960, 540], "kind": "cooler"},
        {"id": "cam-shelf-01", "position": [0.3, 2.9, 2.9], "yaw": math.radians(-8), "pitch": math.radians(-25), "hfov": 90, "resolution": [960, 540], "kind": "shelf"},
    ],
}
DOOR_IN, DOOR_OUT = (3.0, -2.3), (3.0, -2.95)
REGISTER = (2.4, 1.3)
# (name, colour index, enter time, slot, outcome, waypoints before the slot, waypoints before the register)
# One at a time: the event engine continues a lost track when a new one shows up near the same spot within
# 10 s, and these blobs carry no appearance to tell two shoppers apart.
SHOPPERS = [("Character", 0, 0.5, "G1-S1", "paid", [(1.9, -0.3)], []),
            ("Character_01", 1, 21.0, "C1-S2", "concealed", [], [])]


def _unit_pos(slot, fixture, facing: int = 0):
    """Front unit of a facing, three.js world."""
    off = ((facing + 0.5) / slot["facings"] - 0.5) * slot["size"][0]
    local = [slot["position"][i] - fixture["position"][i] for i in range(3)]      # fixtures here have rotation 0
    return local_to_world(fixture, (local[0] + off, local[1], local[2] + slot["size"][2] / 2))


class Shopper:
    """Scripted path: (t0, t1, kind, floor a, floor b). Hands and the carried item follow from the segment."""

    def __init__(self, name, color, t0, slot, fixture, outcome, via_slot, via):
        self.name, self.color, self.outcome, self.slot = name, color, outcome, slot
        self.item = _unit_pos(slot, fixture)
        self.segs, self.t, self.at = [], t0, DOOR_OUT
        self.t_pick = self.t_resolved = None
        stand = (self.item[0] - 0.25, self.item[2] + 0.45)   # a step right of centre so the right hand reaches straight
        for p in [DOOR_IN] + via_slot + [stand]:
            self._walk(p)
        self._hold("reach", REACH_S)
        self.t_pick = self.t - REACH_S / 2
        if outcome == "concealed":
            self._hold("wait", 0.5)
            self._hold("conceal", CONCEAL_S)
            self.t_resolved = self.t - CONCEAL_S * 0.45
        else:
            for p in via + [REGISTER]:
                self._walk(p)
            self._hold("wait", PAY_S)
            self.t_resolved = self.t - PAY_S + PAID_AT_S
        self._walk(DOOR_IN)
        self._walk(DOOR_OUT)
        self.t_start, self.t_exit = t0, self.t

    def _walk(self, to):
        dur = math.dist(self.at, to) / SPEED
        self.segs.append((self.t, self.t + dur, "walk", self.at, to))
        self.t, self.at = self.t + dur, to

    def _hold(self, kind, dur):
        self.segs.append((self.t, self.t + dur, kind, self.at, self.at))
        self.t += dur

    def state(self, t):
        """None when outside, else (floor xz, facing xz, {hand: xyz}, item xyz | None)."""
        if not self.t_start <= t <= self.t_exit:
            return None
        t0, t1, kind, a, b = next(s for s in self.segs if s[0] <= t <= s[1] + 1e-9)
        u = (t - t0) / max(t1 - t0, 1e-6)
        pos = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        f = (0.0, -1.0)                                  # holds face the fixture (all fixtures face +z)
        if kind == "walk":
            d = math.dist(a, b) or 1.0
            f = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
        r = (-f[1], f[0])                                # right-hand side
        hands = {s: (pos[0] + k * 0.3 * r[0] + 0.1 * f[0], 0.95, pos[1] + k * 0.3 * r[1] + 0.1 * f[1])
                 for s, k in (("left", -1), ("right", 1))}
        k = 1 - abs(2 * u - 1)                           # out and back
        if kind == "reach":
            hands["right"] = tuple(hands["right"][i] + (self.item[i] - hands["right"][i]) * k for i in range(3))
        elif kind == "conceal":
            goal = (pos[0] + 0.12 * f[0], 1.0, pos[1] + 0.12 * f[1])
            hands["right"] = tuple(hands["right"][i] + (goal[i] - hands["right"][i]) * k for i in range(3))
        gone = self.t_resolved if self.outcome == "concealed" else self.t_resolved - PAID_AT_S + 1.0   # hidden / on the counter
        item = hands["right"] if self.t_pick <= t < gone else None
        return pos, f, hands, item


def _px(cam, pts):
    px, front = project([to_usd(p) for p in pts], cam["view"], cam["proj"], cam["w"], cam["h"])
    return px, front


def _fill(img, cam, pts, color):
    px, front = _px(cam, pts)
    if front.all():
        cv2.fillConvexPoly(img, np.array(hull(px), np.int32), color)


def _background(cam, fixtures):
    img = np.full((cam["h"], cam["w"], 3), BG, np.uint8)
    eye = cam["position"]
    for f in sorted(fixtures, key=lambda f: -math.dist(f["position"], eye)):     # far to near
        w, h, d = f["size"]
        _fill(img, cam, [local_to_world(f, (sx * w / 2, sy * h / 2, sz * d / 2)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
              FILL[f["type"]])
    return img


def make_fixture(out_dir, noise: float = 2.0, seed: int = 0) -> dict:
    """Write the fake Isaac run into out_dir. Returns {"layout": path, "events": [...], "duration_s": s}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layout = LAYOUT
    (out / "layout.json").write_text(json.dumps(layout, indent=1))
    fixtures = {f["id"]: f for f in layout["fixtures"]}
    slots = {s["id"]: s for s in layout["slots"]}
    skus = {s["id"]: s for s in layout["skus"]}
    shoppers = [Shopper(n, c, t0, slots[sid], fixtures[slots[sid]["fixtureId"]], oc, v1, v2) for n, c, t0, sid, oc, v1, v2 in SHOPPERS]
    duration = math.ceil(max(s.t_exit for s in shoppers) + 10.0)       # tail: exit confirm + ledger grace
    cams = []
    for c in layout["cameras"]:
        p = camera_params_from_layout(c)
        cams.append({**c, "params": p, "view": np.array(p["cameraViewTransform"]).reshape(4, 4),
                     "proj": np.array(p["cameraProjection"]).reshape(4, 4), "w": c["resolution"][0], "h": c["resolution"][1],
                     "fy": (c["resolution"][0] / 2) / math.tan(math.radians(c["hfov"]) / 2)})

    batch = out / "batch_00"
    (batch / "ira_output").mkdir(parents=True, exist_ok=True)
    (batch / "config.yaml").write_text(yaml.safe_dump({"isaacsim.replicator.agent": {
        "version": "0.7.0", "global": {"seed": seed, "simulation_length": duration * SIM_FPS}}}))
    rng = np.random.default_rng(seed)
    stock = [(s, _unit_pos(s, fixtures[s["fixtureId"]], k)) for s in layout["slots"] for k in range(s["facings"])]
    for cam in cams:
        cdir = batch / "ira_output" / prim_name(cam["id"])
        cdir.mkdir(parents=True, exist_ok=True)
        bg = _background(cam, layout["fixtures"])
        bank = [rng.normal(0, noise, bg.shape).astype(np.int16) for _ in range(4)] if noise else []
        for n in range(duration * SIM_FPS // WRITE_EVERY):
            t = n * WRITE_EVERY / SIM_FPS
            img = bg.copy()
            states = [(s, s.state(t)) for s in shoppers]
            live = sorted(((s, st) for s, st in states if st), key=lambda r: -math.dist((r[1][0][0], 1.0, r[1][0][1]), cam["position"]))
            for s, (pos, f, _, _) in live:                                   # bodies, far to near
                r = (-f[1], f[0])
                body = [(pos[0] + a * 0.25 * r[0] + b * 0.15 * f[0], y, pos[1] + a * 0.25 * r[1] + b * 0.15 * f[1])
                        for a in (-1, 1) for b in (-1, 1) for y in (0.0, 1.5)]
                _fill(img, cam, body, PERSON_COLORS[s.color])
                (hp,), (ok,) = _px(cam, [(pos[0], 1.62, pos[1])])
                (hw,), _ = _px(cam, [(pos[0] + 0.12 * r[0], 1.62, pos[1] + 0.12 * r[1])])
                if ok:
                    cv2.circle(img, (int(hp[0]), int(hp[1])), max(4, int(math.dist(hp, hw))), PERSON_COLORS[s.color], -1)
            taken = {(s.slot["id"], 0) for s in shoppers if t >= s.t_pick}
            items = [(sl["skuId"], p) for sl, p in stock if (sl["id"], stock_index(stock, sl, p)) not in taken]
            items += [(s.slot["skuId"], st[3]) for s, st in live if st[3] is not None]
            for sku, p in items:
                (q,), (ok,) = _px(cam, [p])
                if ok:
                    held = any(p is st[3] for _, st in live)
                    cy = q[1] + (12 if held else 0)                       # carried items hang just below the hand, as in toy_render
                    cv2.rectangle(img, (int(q[0] - 7), int(cy - 9)), (int(q[0] + 7), int(cy + 9)), PRODUCT_COLORS[sku], -1)
            for s, (_, _, hands, _) in live:
                px, front = _px(cam, list(hands.values()))
                for q, ok in zip(px, front):
                    if ok:
                        cv2.circle(img, (int(q[0]), int(q[1])), 7, HAND_COLOR, -1)
            if noise:
                img = np.clip(img.astype(np.int16) + bank[n % len(bank)], 0, 255).astype(np.uint8)
            cv2.putText(img, f"TOY DATA sim fixture {cam['id']} t={t:5.1f}s", (cam["w"] - 400, cam["h"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            cv2.imwrite(str(cdir / f"rgb_{n:04d}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if n == 0:   # BasicWriter writes one per frame; the cameras are static, so one is enough here
                (cdir / f"camera_params_{n:04d}.json").write_text(json.dumps(cam["params"]))

    events = []
    for s in shoppers:
        seen = []
        for cam in cams:
            (q,), (ok,) = _px(cam, [s.item])
            if ok and 0 <= q[0] < cam["w"] and 0 <= q[1] < cam["h"]:
                depth = math.dist(s.item, cam["position"])
                seen.append({"id": cam["id"], "px": round(skus[s.slot["skuId"]]["size"][0] * cam["fy"] / depth, 1)})
        events.append({"t": round(s.t_pick, 3), "frame": round(s.t_pick * SIM_FPS), "shopper": s.name,
                       "skuId": s.slot["skuId"], "slotId": s.slot["id"],
                       "unit": f"/World/Products/{prim_name(s.slot['id'])}/{prim_name(s.slot['id'])}_f0_d0",
                       "price": skus[s.slot["skuId"]]["price"], "outcome": s.outcome, "tResolved": round(s.t_resolved, 3),
                       "cameras": seen, "camerasUpperBound": seen, "camerasRaycast": None,
                       "itemPosUsd": [round(v, 4) for v in to_usd(s.item)]})
    events.sort(key=lambda e: e["t"])
    (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return {"layout": str(out / "layout.json"), "events": events, "duration_s": duration}


def stock_index(stock, slot, p) -> int:
    """Facing index of a stock unit within its slot (0 = the one a shopper takes)."""
    return [q for sl, q in stock if sl is slot].index(p)


if __name__ == "__main__":
    import sys
    r = make_fixture(sys.argv[1] if len(sys.argv) > 1 else "out/sim_fixture")
    print(json.dumps({k: v for k, v in r.items() if k != "events"}), f"{len(r['events'])} ground-truth events")
