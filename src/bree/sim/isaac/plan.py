"""Episode plans for the Isaac Sim 4.5 gas-station c-store generator. Pure Python, no Isaac Sim.

A plan is everything about one episode that can be decided before simulating: the
store (layout, product placement), the camera rig, lighting, who comes in and what
each person does (omni.anim.people command lines plus the product action tied to
each command), and the intended outcome (who steals what). `gen_cstore.py` executes
plans inside Isaac Sim and records what actually happened (event times, boxes,
skeletons); `convert.py` turns the recordings into clips in the
`bree.sim.toy_render` truth schema.

Scenes vs episodes: a *scene* (scene_seed) fixes the layout variant, product
placement, colours and the camera rig; each scene is recorded `takes` times with
different people, lighting and small camera jitter. Train/val/test are split by
scene seed, so test scenes are never seen in training.

World frame: Z up, metres. Origin at the store centre, +x toward the counter (east),
+y toward the cooler wall (north). The front wall with the entrance is at y = -4.5.
Character facing: angle 0 faces -y (derived from the 4.5 character assets' bind
pose: eyes and toes sit at -y), angles are degrees counter-clockwise about +z,
which is the convention omni.anim.people uses for `GoTo x y z angle`.

    python -m bree.sim.isaac.plan --mode pilot --out uploads/bree_plans/pilot
    python -m bree.sim.isaac.plan --mode full  --out uploads/bree_plans/full
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STORE_CONFIG = ROOT / "configs" / "store_gas_station_small.yaml"

# ------------------------------------------------------------------ store geometry
HALF_X, HALF_Y, CEILING = 6.0, 4.5, 3.2
WALL_T = 0.15
DOOR_X = (-5.4, -4.0)                    # entrance opening in the front wall (y = -HALF_Y)
DOOR_IN = (-4.7, -3.6)                   # just inside the door
DOOR_OUT = (-4.7, -5.4)                  # just outside the door
AWAY = (-7.5, -7.5)                      # where people walk to after leaving (out of every camera)
WAIT_Y = -6.2                            # people wait here (behind the opaque front wall) before entering

COOLER_DOORS, COOLER_DOOR_W = 5, 0.75
COOLER_X0 = -COOLER_DOORS * COOLER_DOOR_W / 2
COOLER_FRONT_Y, COOLER_BACK_Y, COOLER_H = 3.7, 4.5, 2.1
COOLER_SHELF_Z = (0.3, 0.65, 1.0, 1.35, 1.7)
COOLER_PRODUCT_Y = 3.95
COOLER_DOOR_OPEN_DEG = 95.0              # hinge on the door's +x edge; at ~90 deg the open panel clears a right-hand reach

GONDOLA_X = (-2.6, 0.0, 2.6)
GONDOLA_Y = (-0.6, 2.4)
GONDOLA_W, GONDOLA_H = 0.9, 1.4
GONDOLA_SHELF_Z = (0.25, 0.6, 0.95, 1.3)

COUNTER = {"x": (4.3, 5.0), "y": (-3.5, -0.5), "h": 1.0}     # 0.8 m staff aisle behind it
PAY_SPOT = (3.9, -2.0)
QUEUE_SPOTS = [(3.9, -2.0), (3.1, -2.0), (2.3, -2.0), (1.5, -2.0)]
CASHIER_SPOT = (5.4, -2.0)
STAFF_BACK_SPOT = (5.4, -2.8)
IMPULSE = {"x": 4.4, "y": (-3.3, -2.5), "z": 1.05}
TERMINAL = (4.5, -2.0, 1.05)

# Reach geometry of the 4.5 `push_button.skelanim.usd` clip (forward kinematics on its 101 frames):
# the RIGHT wrist peaks ~0.5 m in front of the root, ~0.1 m to the right, ~1.4 m high, at ~1.2 s.
REACH_FWD, REACH_RIGHT = 0.5, 0.1
REACH_Z = (1.0, 1.6)                     # product centres a standing reach can plausibly take from
REACH_S = 2.0                            # PushButton duration we command
CONCEAL_S = 1.2                          # Idle duration used for a concealment
WALK_SPEED = 1.1                         # m/s, only for duration estimates

# category -> (shape, size, base colours). cyl size = (radius, height); box size = (x, y, z).
PRODUCT_SHAPES = {
    "soda_bottle":     ("cyl", (0.035, 0.23), [(0.75, 0.05, 0.05), (0.05, 0.15, 0.6), (0.1, 0.5, 0.1)]),
    "water_bottle":    ("cyl", (0.04, 0.26), [(0.55, 0.75, 0.95), (0.85, 0.9, 0.95)]),
    "energy_drink":    ("cyl", (0.033, 0.16), [(0.15, 0.2, 0.55), (0.1, 0.1, 0.1), (0.85, 0.8, 0.1)]),
    "beer_pack":       ("box", (0.2, 0.14, 0.13), [(0.9, 0.9, 0.85), (0.1, 0.2, 0.6)]),
    "candy_bar":       ("box", (0.14, 0.03, 0.05), [(0.35, 0.2, 0.1), (0.9, 0.5, 0.05), (0.8, 0.1, 0.1)]),
    "chips_bag":       ("box", (0.22, 0.07, 0.3), [(0.95, 0.8, 0.1), (0.9, 0.2, 0.1), (0.2, 0.5, 0.9)]),
    "jerky":           ("box", (0.13, 0.03, 0.2), [(0.15, 0.1, 0.05), (0.6, 0.3, 0.1)]),
    "phone_accessory": ("box", (0.1, 0.03, 0.18), [(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)]),
    "lighter":         ("box", (0.025, 0.012, 0.08), [(0.9, 0.1, 0.1), (0.1, 0.6, 0.9), (0.95, 0.8, 0.1)]),
}
COOLER_GROUPS = [["soda_bottle"], ["soda_bottle"], ["energy_drink"], ["water_bottle"], ["beer_pack"]]
GONDOLA_GROUPS = [["candy_bar"], ["chips_bag", "jerky"], ["phone_accessory", "lighter"]]
IMPULSE_CATS = ["candy_bar", "lighter"]

# Isaac/People/Characters/ folders in the 4.5 asset pack (from its filter.json; biped_demo excluded).
SHOPPER_ASSETS = [
    "M_Medical_01", "male_adult_construction_03", "male_adult_police_04", "original_male_adult_construction_02",
    "original_male_adult_medical_01", "original_male_adult_police_04", "F_Business_02", "F_Medical_01",
    "female_adult_police_01_new", "female_adult_police_02", "female_adult_police_03_new",
    "original_female_adult_business_02", "original_female_adult_medical_01", "original_female_adult_police_01",
    "original_female_adult_police_02", "original_female_adult_police_03",
]
STAFF_ASSETS = ["male_adult_construction_01_new", "male_adult_construction_05_new",
                "original_male_adult_construction_01", "original_male_adult_construction_05"]

ARCHETYPES = {                           # shopper behaviour mix (weights)
    "normal_buy": 0.36, "put_back": 0.12, "browse": 0.10, "conceal_partial_pay": 0.08,
    "conceal_walkout": 0.07, "pocket_then_pay": 0.05, "walkout_in_hand": 0.07,
}
CONCEAL_ANCHORS = ["pocket", "waistband", "under_clothing", "bag"]
LIGHT_MODES = {"day": 0.45, "night": 0.35, "flicker": 0.2}

# Camera rig as a real store would mount it. Focal lengths in mm on a 20.955 mm aperture.
CAMERA_RIG = [
    {"name": "cam_main", "eye": (5.6, -4.2, 3.0), "target": (-1.5, 2.8, 0.9), "focal_mm": 11.0},     # corner above register
    {"name": "cam_cooler", "eye": (0.0, 1.0, 2.9), "target": (0.0, 3.9, 1.1), "focal_mm": 10.0},     # facing the coolers
    {"name": "cam_door", "eye": (-2.4, -2.0, 2.9), "target": (-4.7, -4.4, 1.0), "focal_mm": 12.0},   # facing the door
    {"name": "cam_register", "eye": (5.7, 0.6, 2.9), "target": (3.9, -2.2, 1.0), "focal_mm": 12.0},  # above the register
]
H_APERTURE_MM = 20.955
RESOLUTION = (1280, 720)
FPS_OUT = 15
SPLITS = ["train"] * 8 + ["val", "test"]


# ------------------------------------------------------------------ small geometry helpers
def facing_angle(dx: float, dy: float) -> float:
    """GoTo angle (degrees) that makes a character face direction (dx, dy). Angle 0 faces -y."""
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def forward(angle: float) -> tuple[float, float]:
    a = math.radians(angle)
    return math.sin(a), -math.cos(a)


def right_of(angle: float) -> tuple[float, float]:
    fx, fy = forward(angle)
    return fy, -fx                       # forward x up(+z)


def reach_stand(target: tuple[float, float], face: tuple[float, float]) -> tuple[float, float, float]:
    """Where to stand (x, y, angle) so a right-hand PushButton reach ends at `target` (x, y)."""
    ang = facing_angle(*face)
    fx, fy = forward(ang)
    rx, ry = right_of(ang)
    return (target[0] - REACH_FWD * fx - REACH_RIGHT * rx, target[1] - REACH_FWD * fy - REACH_RIGHT * ry, ang)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> list[list[float]]:
    """Camera-to-world 4x4 (row-vector convention, as USD/Gf uses) for a USD camera at `eye`
    looking at `target`. USD cameras look down their local -Z with +Y up."""
    def sub(a, b): return [a[i] - b[i] for i in range(3)]
    def norm(v):
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        return [c / n for c in v]
    def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    z = norm(sub(eye, target))
    x = norm(cross(up, z))
    y = cross(z, x)
    return [x + [0.0], y + [0.0], z + [0.0], list(eye) + [1.0]]


def split_for_scene(scene_seed: int) -> str:
    return SPLITS[scene_seed % len(SPLITS)]


def _r(v, n=3):
    return [round(float(c), n) for c in v] if isinstance(v, (list, tuple)) else round(float(v), n)


# ------------------------------------------------------------------ catalog
def load_catalog(path: str | Path = STORE_CONFIG) -> dict[str, list[str]]:
    """category -> SKUs from the store config."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    out: dict[str, list[str]] = {}
    for sku, info in raw["catalog"].items():
        out.setdefault(info["category"], []).append(sku)
    return out


# ------------------------------------------------------------------ scene (layout + products)
def build_scene(scene_seed: int, catalog: dict[str, list[str]]) -> dict:
    rng = random.Random(f"scene-{scene_seed}")
    fixtures, products, doors = [], [], []

    def box(name, lo, hi, color, material="matte"):
        c = [(lo[i] + hi[i]) / 2 for i in range(3)]
        s = [hi[i] - lo[i] for i in range(3)]
        fixtures.append({"name": name, "center": _r(c), "size": _r(s), "color": _r(color), "material": material})

    wall = [rng.uniform(0.75, 0.92)] * 2 + [rng.uniform(0.7, 0.9)]
    floor = [rng.uniform(0.45, 0.7)] * 3
    box("floor", (-HALF_X - 4, -HALF_Y - 5, -0.02), (HALF_X + 2, HALF_Y + 1, 0.0), floor)
    box("ceiling", (-HALF_X, -HALF_Y, CEILING), (HALF_X, HALF_Y, CEILING + 0.05), (0.9, 0.9, 0.9))
    box("wall_back", (-HALF_X, HALF_Y, 0), (HALF_X, HALF_Y + WALL_T, CEILING), wall)
    box("wall_west", (-HALF_X - WALL_T, -HALF_Y, 0), (-HALF_X, HALF_Y, CEILING), wall)
    box("wall_east", (HALF_X, -HALF_Y, 0), (HALF_X + WALL_T, HALF_Y, CEILING), wall)
    box("wall_front_l", (-HALF_X, -HALF_Y - WALL_T, 0), (DOOR_X[0], -HALF_Y, CEILING), wall)
    box("wall_front_r", (DOOR_X[1], -HALF_Y - WALL_T, 0), (HALF_X, -HALF_Y, CEILING), wall)
    box("wall_front_top", (DOOR_X[0], -HALF_Y - WALL_T, 2.2), (DOOR_X[1], -HALF_Y, CEILING), wall)
    # entrance door panel stands open against the inside of the front wall (glass)
    box("entrance_door", (DOOR_X[0] - 0.9, -HALF_Y + 0.02, 0.0), (DOOR_X[0] - 0.05, -HALF_Y + 0.07, 2.15),
        (0.8, 0.9, 0.95), "glass")
    box("door_mat", (DOOR_X[0], -HALF_Y, 0.0), (DOOR_X[1], -HALF_Y + 1.5, 0.01), (0.15, 0.15, 0.15))

    # cooler bank: open-front cabinet (back, top, base, mullions), shelves, glass doors
    dark = (0.2, 0.2, 0.22)
    box("cooler_back", (COOLER_X0, COOLER_BACK_Y - 0.08, 0), (-COOLER_X0, COOLER_BACK_Y, COOLER_H), dark, "metal")
    box("cooler_top", (COOLER_X0, COOLER_FRONT_Y, COOLER_H - 0.1), (-COOLER_X0, COOLER_BACK_Y, COOLER_H), dark, "metal")
    box("cooler_base", (COOLER_X0, COOLER_FRONT_Y, 0), (-COOLER_X0, COOLER_BACK_Y, 0.12), dark, "metal")
    for d in range(COOLER_DOORS + 1):
        mx = COOLER_X0 + d * COOLER_DOOR_W
        box(f"cooler_mullion_{d}", (mx - 0.025, COOLER_FRONT_Y, 0), (mx + 0.025, COOLER_BACK_Y, COOLER_H), dark, "metal")
    for z in COOLER_SHELF_Z:
        box(f"cooler_shelf_{z:.2f}", (COOLER_X0, COOLER_FRONT_Y + 0.05, z - 0.02), (-COOLER_X0, COOLER_BACK_Y - 0.08, z),
            (0.7, 0.7, 0.72), "metal")
    groups = COOLER_GROUPS[:]
    rng.shuffle(groups)
    for d in range(COOLER_DOORS):
        x0 = COOLER_X0 + d * COOLER_DOOR_W
        door = {"name": f"cooler_door_{d}", "hinge": _r([x0 + COOLER_DOOR_W, COOLER_FRONT_Y - 0.02]),
                "width": COOLER_DOOR_W - 0.02, "z": _r([0.1, COOLER_H - 0.05]), "open_deg": COOLER_DOOR_OPEN_DEG,
                "products": []}
        doors.append(door)
        for si, z in enumerate(COOLER_SHELF_Z):
            for ci, dx in enumerate((-0.22, 0.0, 0.22)):
                cat = rng.choice(groups[d])
                p = _product(rng, catalog, f"cooler_{d}_{si}_{ci}", cat, "cooler_bank",
                             (x0 + COOLER_DOOR_W / 2 + dx + rng.uniform(-0.02, 0.02), COOLER_PRODUCT_Y, z),
                             face=(0.0, 1.0))
                p["door"] = door["name"]
                door["products"].append(p["id"])
                products.append(p)

    # gondolas: two-sided shelf runs perpendicular to the cooler wall
    ggroups = GONDOLA_GROUPS[:]
    rng.shuffle(ggroups)
    zone_names = ["shelf_A", "shelf_B", "shelf_C"]
    for gi, gx in enumerate(GONDOLA_X):
        grey = (rng.uniform(0.6, 0.85),) * 3
        box(f"gondola_{gi}_spine", (gx - 0.05, GONDOLA_Y[0], 0), (gx + 0.05, GONDOLA_Y[1], GONDOLA_H), grey, "metal")
        box(f"gondola_{gi}_base", (gx - GONDOLA_W / 2, GONDOLA_Y[0], 0), (gx + GONDOLA_W / 2, GONDOLA_Y[1], 0.12),
            grey, "metal")
        for z in GONDOLA_SHELF_Z[1:]:
            box(f"gondola_{gi}_shelf_{z:.2f}", (gx - GONDOLA_W / 2, GONDOLA_Y[0], z - 0.02),
                (gx + GONDOLA_W / 2, GONDOLA_Y[1], z), grey, "metal")
        for side in (-1, 1):
            face_x = gx + side * (GONDOLA_W / 2 + 0.01)
            for si, z in enumerate(GONDOLA_SHELF_Z):
                y = GONDOLA_Y[0] + 0.15
                ci = 0
                while y < GONDOLA_Y[1] - 0.15:
                    cat = rng.choice(ggroups[gi])
                    shape, size, _ = PRODUCT_SHAPES[cat]
                    depth = size[0] * 2 if shape == "cyl" else size[1]
                    width = size[0] * 2 if shape == "cyl" else size[0]
                    if rng.random() > 0.12:      # ~12% empty facings
                        products.append(_product(rng, catalog, f"g{gi}{'we'[side > 0]}_{si}_{ci}", cat, zone_names[gi],
                                                 (face_x - side * (depth / 2 + 0.03), y, z + 0.02),
                                                 face=(-side, 0.0), yaw=90.0))
                    y += width + rng.uniform(0.05, 0.12)
                    ci += 1

    # counter, POS terminal, impulse rack, cigarette wall (decor, unlabeled)
    box("counter", (COUNTER["x"][0], COUNTER["y"][0], 0), (COUNTER["x"][1], COUNTER["y"][1], COUNTER["h"]),
        (rng.uniform(0.3, 0.6), rng.uniform(0.2, 0.4), 0.15))
    box("pos_terminal", (TERMINAL[0] - 0.12, TERMINAL[1] - 0.15, TERMINAL[2] - 0.05),
        (TERMINAL[0] + 0.12, TERMINAL[1] + 0.15, TERMINAL[2] + 0.18), (0.1, 0.1, 0.1))
    box("impulse_rack", (IMPULSE["x"] - 0.1, IMPULSE["y"][0], COUNTER["h"]), (IMPULSE["x"] + 0.1, IMPULSE["y"][1], 1.03),
        (0.5, 0.5, 0.5), "metal")
    y, ci = IMPULSE["y"][0] + 0.06, 0
    while y < IMPULSE["y"][1] - 0.04:
        products.append(_product(rng, catalog, f"impulse_{ci}", rng.choice(IMPULSE_CATS), "impulse_rack",
                                 (IMPULSE["x"], y, IMPULSE["z"]), face=(1.0, 0.0), yaw=90.0))
        y += rng.uniform(0.09, 0.13)
        ci += 1
    box("cigarette_wall", (HALF_X - 0.2, COUNTER["y"][0], 1.0), (HALF_X, COUNTER["y"][1], 2.2), (0.35, 0.3, 0.25))
    box("stock_box", (5.45, -3.45, 0), (5.85, -3.2, 0.4), (0.6, 0.45, 0.3))

    zones3d = [
        {"name": "cooler_bank", "kind": "cooler", "lo": _r([COOLER_X0, COOLER_FRONT_Y - 0.15, 0.2]),
         "hi": _r([-COOLER_X0, COOLER_BACK_Y, 2.0])},
        *[{"name": zone_names[gi], "kind": "shelf", "lo": _r([gx - GONDOLA_W / 2 - 0.15, GONDOLA_Y[0], 0.0]),
           "hi": _r([gx + GONDOLA_W / 2 + 0.15, GONDOLA_Y[1], GONDOLA_H + 0.1])} for gi, gx in enumerate(GONDOLA_X)],
        {"name": "impulse_rack", "kind": "shelf", "lo": _r([IMPULSE["x"] - 0.2, IMPULSE["y"][0] - 0.05, 0.95]),
         "hi": _r([IMPULSE["x"] + 0.15, IMPULSE["y"][1] + 0.05, 1.3])},
        {"name": "register", "kind": "register", "lo": _r([3.3, COUNTER["y"][0], 0.0]),
         "hi": _r([COUNTER["x"][0], COUNTER["y"][1], 0.0])},
        {"name": "door", "kind": "exit", "lo": _r([DOOR_X[0] - 0.2, -HALF_Y - 0.1, 0.0]),
         "hi": _r([DOOR_X[1] + 0.2, -HALF_Y + 0.9, 0.0])},
    ]
    return {"scene_seed": scene_seed, "fixtures": fixtures, "products": products, "doors": doors, "zones3d": zones3d,
            "nav_volume": {"lo": [-HALF_X - 3, -HALF_Y - 4, -0.5], "hi": [HALF_X + 1, HALF_Y + 1, 2.5]},
            "door_line_y": -HALF_Y, "door_x": list(DOOR_X)}


def _product(rng, catalog, pid, cat, zone, pos, face, yaw=0.0) -> dict:
    shape, size, colors = PRODUCT_SHAPES[cat]
    base = rng.choice(colors)
    col = [min(1.0, max(0.0, c + rng.uniform(-0.08, 0.08))) for c in base]
    half_h = size[1] / 2 if shape == "cyl" else size[2] / 2
    center = (pos[0], pos[1], pos[2] + half_h)
    pickable = REACH_Z[0] <= center[2] <= REACH_Z[1]
    p = {"id": pid, "category": cat, "sku": rng.choice(catalog.get(cat, [cat.upper()])), "shape": shape,
         "size": _r(size), "color": _r(col), "pos": _r(center), "yaw": yaw, "zone": zone, "pickable": pickable,
         "initial": "shelf"}
    if pickable:
        sx, sy, ang = reach_stand((center[0], center[1]), face)
        p["stand"] = _r([sx, sy, ang], 2)
    return p


# ------------------------------------------------------------------ cameras + lighting
def camera_rig(scene_seed: int, take: int) -> list[dict]:
    rs, re = random.Random(f"cams-{scene_seed}"), random.Random(f"cams-{scene_seed}-{take}")
    cams = []
    for c in CAMERA_RIG:
        eye = [c["eye"][i] + rs.uniform(-0.2, 0.2) + re.uniform(-0.04, 0.04) for i in range(3)]
        eye[2] = min(3.0, max(2.5, eye[2]))
        tgt = [c["target"][i] + rs.uniform(-0.35, 0.35) + re.uniform(-0.08, 0.08) for i in range(3)]
        cams.append({"name": c["name"], "eye": _r(eye), "target": _r(tgt),
                     "focal_mm": round(c["focal_mm"] * rs.uniform(0.92, 1.08), 2), "h_aperture_mm": H_APERTURE_MM,
                     "resolution": list(RESOLUTION), "transform": [_r(row, 5) for row in look_at(eye, tgt)]})
    return cams


def lighting(rng: random.Random) -> dict:
    mode = rng.choices(list(LIGHT_MODES), weights=list(LIGHT_MODES.values()))[0]
    panels = [(x, y) for x in (-4.0, -1.3, 1.3, 4.0) for y in (-2.8, 0.0, 2.8)]
    spec = {"mode": mode, "panels": [_r([x, y, CEILING - 0.03]) for x, y in panels],
            "panel_scale": round(rng.uniform(0.8, 1.2) * (0.75 if mode == "night" else 1.0), 3),
            "color_temp_k": round(rng.uniform(3800, 6500) if mode != "night" else rng.uniform(3200, 4500)),
            "dome_scale": round(rng.uniform(0.8, 1.2) if mode == "day" else rng.uniform(0.0, 0.05), 3),
            "cooler_light_scale": round(rng.uniform(0.8, 1.3), 3), "flicker_panels": [], "flicker_seed": rng.randrange(1 << 30)}
    if mode == "flicker":
        spec["flicker_panels"] = rng.sample(range(len(panels)), rng.randint(1, 3))
    return spec


# ------------------------------------------------------------------ people + behaviours
class _Person:
    def __init__(self, name, role, archetype, asset, start):
        self.name, self.role, self.archetype, self.asset, self.start = name, role, archetype, asset, start
        self.commands: list[str] = []
        self.actions: list[dict | None] = []
        self.picked, self.put_back, self.concealed, self.paid = [], [], [], []
        self.has_bag = False
        self.queue = "register_q"
        self.path_len = 0.0
        self.busy_s = 0.0
        self.pos = (start[0], start[1])

    def cmd(self, c: str, action: dict | None = None, busy: float = 0.0):
        self.commands.append(c)
        self.actions.append(action)
        self.busy_s += busy

    def goto(self, x, y, angle=None):
        self.path_len += math.hypot(x - self.pos[0], y - self.pos[1])
        self.pos = (x, y)
        self.cmd(f"GoTo {x:.2f} {y:.2f} 0 {'_' if angle is None else f'{angle:.1f}'}")


def _enter(p: _Person, wait_s: float):
    if wait_s > 0:
        p.cmd(f"Idle {wait_s:.1f}", busy=wait_s)
    p.goto(*DOOR_OUT)
    p.goto(*DOOR_IN)


def _leave(p: _Person, from_queue: bool = False):
    if from_queue:   # Dequeue walks to the given point, here just inside the door
        p.path_len += math.hypot(DOOR_IN[0] - p.pos[0], DOOR_IN[1] - p.pos[1])
        p.pos = DOOR_IN
        p.cmd(f"Dequeue {p.queue} {DOOR_IN[0]:.2f} {DOOR_IN[1]:.2f} 0 _")
    else:
        p.goto(*DOOR_IN)
    p.goto(*DOOR_OUT)
    p.goto(*AWAY)


def _reach(p: _Person, prod: dict, verb: str, hand: str = "right"):
    sx, sy, ang = prod["stand"]
    p.goto(sx, sy, ang)
    p.cmd(f"PushButton {REACH_S:.1f}", {"verb": verb, "product": prod["id"], "category": prod["category"],
                                        "sku": prod["sku"], "hand": hand}, busy=REACH_S)


def _pay(p: _Person, txn: str, items: list[dict]):
    p.cmd(f"Queue {p.queue}")
    p.cmd(f"PushButton {REACH_S:.1f}", {"verb": "pay", "txn": txn, "terminal": "pos_1",
                                        "skus": [i["sku"] for i in items], "products": [i["id"] for i in items]},
          busy=REACH_S + 3.0)
    p.paid += items


def _shopper_script(p: _Person, rng: random.Random, take_product, txn: str, cooler_door: int | None = None):
    """Fill p.commands/actions for its archetype. `take_product(zone_hint)` reserves a pickable product."""
    a = p.archetype
    if a == "browse":
        for _ in range(rng.randint(1, 2)):
            prod = take_product(None, reserve=False)
            sx, sy, ang = prod["stand"]
            p.goto(sx, sy, ang)
            p.cmd(f"LookAround {rng.uniform(2.0, 4.0):.1f}", busy=3.0)
            if rng.random() < 0.4:   # touch the shelf, take nothing
                p.cmd(f"PushButton {REACH_S:.1f}", {"verb": "touch", "product": prod["id"],
                                                    "category": prod["category"], "sku": prod["sku"]}, busy=REACH_S)
        _leave(p)
        return

    first = take_product("cooler" if cooler_door is not None else None, door=cooler_door)
    _reach(p, first, "take")
    p.picked.append(first)
    if cooler_door is not None:
        p.cmd(f"Idle {rng.uniform(0.0, 0.6):.1f}")

    if a == "put_back":
        p.cmd(f"LookAround {rng.uniform(1.5, 3.0):.1f}", busy=2.0)
        _reach(p, first, "put")
        p.put_back.append(first)
        if rng.random() < 0.5:
            second = take_product(None)
            _reach(p, second, "take")
            p.picked.append(second)
            _pay(p, txn, [second])
            _leave(p, from_queue=True)
        else:
            _leave(p)
        return

    if a in ("normal_buy",) and rng.random() < 0.4:
        second = take_product(None)
        _reach(p, second, "take")
        p.picked.append(second)

    if a in ("conceal_partial_pay", "conceal_walkout", "pocket_then_pay"):
        anchor = "pocket" if a == "pocket_then_pay" else rng.choice(CONCEAL_ANCHORS)
        p.has_bag = anchor == "bag"
        if a == "conceal_partial_pay":     # the visible item is paid, a second one is concealed
            hidden = take_product(None)
            _reach(p, hidden, "take")
            p.picked.append(hidden)
        else:
            hidden = first
        if rng.random() < 0.5:
            p.cmd(f"LookAround {rng.uniform(1.0, 2.5):.1f}", busy=1.5)
        p.cmd(f"Idle {CONCEAL_S:.1f}", {"verb": "conceal", "product": hidden["id"], "category": hidden["category"],
                                         "sku": hidden["sku"], "anchor": anchor}, busy=CONCEAL_S)
        p.concealed.append(hidden)
        if a == "conceal_walkout":
            _leave(p)
            return
        paid = [i for i in p.picked if i is not hidden] if a == "conceal_partial_pay" else list(p.picked)
        _pay(p, txn, paid)
        _leave(p, from_queue=True)
        return

    if a == "walkout_in_hand":
        _leave(p)
        return

    _pay(p, txn, list(p.picked))       # normal_buy
    _leave(p, from_queue=True)


def plan_episode(scene_seed: int, take: int, catalog: dict[str, list[str]] | None = None) -> dict:
    catalog = catalog or load_catalog()
    scene = build_scene(scene_seed, catalog)
    rng = random.Random(f"episode-{scene_seed}-{take}")
    products = scene["products"]
    by_id = {p["id"]: p for p in products}
    used: set[str] = set()

    def take_product(kind: str | None, door: int | None = None, reserve: bool = True) -> dict:
        pool = [p for p in products if p["pickable"] and p["initial"] == "shelf" and p["id"] not in used]
        if door is not None:   # middle column, so two people at neighbouring doors stand >= 0.75 m apart
            pool = [p for p in pool if p.get("door") == f"cooler_door_{door}" and p["id"].endswith("_1")] or pool
        elif kind == "cooler":
            pool = [p for p in pool if p["zone"] == "cooler_bank"] or pool
        prod = rng.choice(pool)
        if reserve:
            used.add(prod["id"])
        return prod

    people: list[_Person] = []
    qname = f"register_q_{scene_seed}_{take}"     # unique per episode: omni.anim.people queues are process-global
    flags = {"two_at_cooler": rng.random() < 0.15, "crowded_register": rng.random() < 0.15,
             "restock": rng.random() < 0.2, "cashier": rng.random() < 0.85}
    n = rng.choice([1, 2, 2, 3, 3, 4, 5])
    if flags["crowded_register"] or flags["two_at_cooler"]:
        n = min(n, 2)                    # keep total shoppers <= 8
    assets = SHOPPER_ASSETS[:]
    rng.shuffle(assets)
    waits = sorted(rng.uniform(1.0, 22.0) for _ in range(n))
    specs = [(rng.choices(list(ARCHETYPES), weights=list(ARCHETYPES.values()))[0], w, None) for w in waits]
    if flags["two_at_cooler"]:
        d = rng.randrange(COOLER_DOORS - 1)
        w = rng.uniform(2.0, 12.0)
        specs += [("normal_buy", w, d), (rng.choice(["normal_buy", "walkout_in_hand"]), w, d + 1)]
    if flags["crowded_register"]:
        w0 = rng.uniform(5.0, 15.0)
        specs += [("normal_buy", w0 + k * rng.uniform(1.5, 4.0), None) for k in range(rng.randint(3, 4))]
    for i, (arch, wait, door) in enumerate(specs):
        start = (round(-3.6 + 0.9 * (i % 6), 2), round(WAIT_Y - 0.9 * (i // 6), 2), 0.0)
        p = _Person(f"Shopper_{i + 1:02d}", "shopper", arch, assets[i % len(assets)], start)
        p.queue = qname
        _enter(p, wait)
        _shopper_script(p, rng, take_product, f"T{scene_seed:04d}{take}{i + 1:02d}", door)
        people.append(p)

    staff = STAFF_ASSETS[:]
    rng.shuffle(staff)
    horizon = max(p.busy_s + p.path_len / WALK_SPEED for p in people)
    if flags["cashier"]:
        c = _Person("Staff_01", "employee", "cashier", staff[0], (*CASHIER_SPOT, 270.0))
        t = 0.0
        while t < horizon * 1.6 + 30:
            d = rng.uniform(4.0, 10.0)
            c.cmd(f"{rng.choice(['Idle', 'Idle', 'LookAround'])} {d:.1f}", busy=d)
            t += d
        people.append(c)
    if flags["restock"]:
        r = _Person("Staff_02", "employee", "restock", staff[1], (*STAFF_BACK_SPOT, 270.0))
        r.cmd(f"Idle {rng.uniform(2.0, 15.0):.1f}")
        free = [p for p in products if p["pickable"] and p["id"] not in used and p["zone"] != "impulse_rack"]
        for k, slot in enumerate(rng.sample(free, min(len(free), rng.randint(1, 3)))):
            slot["initial"] = f"hand:{r.name}"        # starts in the employee's hand, placed on the shelf
            used.add(slot["id"])
            _reach(r, slot, "restock")
        r.goto(*STAFF_BACK_SPOT, 270.0)
        r.cmd("Idle 300")
        people.append(r)

    max_s = max(p.busy_s + p.path_len / WALK_SPEED for p in people if p.role == "shopper") * 1.6 + 25.0
    split = split_for_scene(scene_seed)
    out_people = []
    for p in people:
        stolen = [i for i in p.picked if i not in p.paid and i not in p.put_back]
        out_people.append({
            "name": p.name, "role": p.role, "archetype": p.archetype, "asset": p.asset, "start": _r(p.start, 2),
            "has_bag": p.has_bag, "commands": p.commands, "actions": p.actions,
            "intent": {"picked": [i["id"] for i in p.picked], "put_back": [i["id"] for i in p.put_back],
                       "concealed": [i["id"] for i in p.concealed], "paid": [i["id"] for i in p.paid],
                       "stolen": [i["id"] for i in stolen], "thief": bool(stolen)},
        })
    return {
        "version": 1, "episode": f"s{scene_seed:05d}_t{take}", "scene_seed": scene_seed, "take": take, "split": split,
        "fps_out": FPS_OUT, "max_duration_s": round(min(max_s, 240.0), 1), "flags": flags,
        "scene": scene, "cameras": camera_rig(scene_seed, take), "lighting": lighting(rng),
        "queue": {"name": qname, "spots": [_r([x, y, 0.0, facing_angle(1.0, 0.0)], 2) for x, y in QUEUE_SPOTS]},
        "people": out_people,
        "categories": {pid: by_id[pid]["category"] for pid in by_id},
    }


def command_file(plan: dict) -> str:
    """The omni.anim.people command file for a plan: queue definition + one line per command."""
    q = plan["queue"]
    lines = [f"Queue {q['name']}"]
    lines += [f"Queue_Spot {q['name']} {i} {x:.2f} {y:.2f} {z:.2f} {a:.1f}" for i, (x, y, z, a) in enumerate(q["spots"])]
    for p in plan["people"]:
        lines += [f"{p['name']} {c}" for c in p["commands"]]
    return "\n".join(lines) + "\n"


def with_facing_offset(commands: str, offset: float) -> str:
    """Add `offset` degrees to every explicit angle in command lines (GoTo, Queue_Spot, Dequeue).
    Calibration knob for when the characters' rest facing turns out not to be -y."""
    if not offset:
        return commands
    out = []
    for line in commands.split("\n"):
        w = line.split(" ")
        cmd = w[0] if w[0] in ("Queue", "Queue_Spot") else (w[1] if len(w) > 1 else "")
        if cmd in ("GoTo", "Queue_Spot", "Dequeue") and w[-1] not in ("_", ""):
            w[-1] = f"{(float(w[-1]) + offset) % 360.0:.1f}"
        out.append(" ".join(w))
    return "\n".join(out)


MODES = {"pilot": {"scenes": 13, "takes": 1, "first_seed": 90000}, "full": {"scenes": 130, "takes": 4, "first_seed": 1}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(MODES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", type=int, help="override number of scenes")
    ap.add_argument("--takes", type=int, help="override takes per scene")
    a = ap.parse_args(argv)
    m = dict(MODES[a.mode])
    m["scenes"] = a.scenes or m["scenes"]
    m["takes"] = a.takes or m["takes"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    rows = []
    for s in range(m["first_seed"], m["first_seed"] + m["scenes"]):
        for t in range(m["takes"]):
            plan = plan_episode(s, t, catalog)
            (out / f"{plan['episode']}.json").write_text(json.dumps(plan))
            rows.append({"episode": plan["episode"], "split": plan["split"], "people": len(plan["people"]),
                         "thieves": sum(p["intent"]["thief"] for p in plan["people"]), "max_s": plan["max_duration_s"]})
    (out / "manifest.json").write_text(json.dumps({"mode": a.mode, **m, "episodes": rows}, indent=1))
    n_cams = len(CAMERA_RIG)
    print(f"{len(rows)} episodes x {n_cams} cameras = {len(rows) * n_cams} clips -> {out}")
    for sp in ("train", "val", "test"):
        print(f"  {sp}: {sum(r['split'] == sp for r in rows)} episodes")


if __name__ == "__main__":
    main()
