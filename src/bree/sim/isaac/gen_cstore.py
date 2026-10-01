"""Render planned gas-station c-store episodes in Isaac Sim 4.5 (standalone, headless).

Run with Isaac Sim's own Python inside the 4.5 container (see README.md runbook):

    /isaac-sim/python.sh /uploads/bree-vision/src/bree/sim/isaac/gen_cstore.py \
        --plans /uploads/bree_plans/pilot --out /results/raw_pilot

Plans come from `bree.sim.isaac.plan` (pure Python). This script only executes them:
builds the store from primitives, places the cameras, spawns Isaac/People characters
driven by omni.anim.people command files (set up with the same calls Isaac Sim
Replicator Agent 0.5.14 uses), moves products for pick / put-back / conceal / pay,
and records per camera RGB, tight 2D boxes, skeleton keypoints and camera matrices.

Hand-object interaction is not part of IRA. How it is done here:
  * Reach = the `PushButton` custom command (Isaac/People/Animations/push_button.skelanim.usd,
    registered through omni.anim.people's CustomCommandManager): a right-arm forward reach to
    ~1.4 m that peaks ~1.2 s in. Product actions are tied to commands through
    omni.anim.people's CommandStartEvent (payload = the command line).
  * Grasp = from `--grasp-at` seconds after the reach starts, the product prim follows the hand
    joint every frame (omni.anim.graph.core Character.get_joint_transform). Put-back / restock =
    the product is set down at its shelf slot. Conceal = during an Idle the product slides from
    the hand to a body anchor (pocket, waistband, under clothing, bag) and is made invisible.
    There is no concealment arm animation in the 4.5 asset pack, so the pose does not show it.

Output per episode, <out>/<episode>/: plan.json, commands.txt, events.json, world.jsonl,
<cam>/camera.json, <cam>/rgb/00000.jpg..., <cam>/ann.jsonl, meta.json, done.json (written last).
Episodes with done.json are skipped, so a stopped run resumes where it left off.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--plans", required=True, help="directory of episode plan JSONs (bree.sim.isaac.plan)")
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*", help="episode names to run (default: all)")
ap.add_argument("--shard", default="0/1", help="i/n: run every n-th episode starting at i (parallel containers)")
ap.add_argument("--sim-fps", type=int, default=30, help="simulation rate; IRA itself runs at 30")
ap.add_argument("--grasp-at", type=float, default=1.2, help="s after a PushButton starts when the hand is at the shelf")
ap.add_argument("--facing-offset", type=float, default=0.0, help="deg added to every GoTo angle if characters face wrong")
ap.add_argument("--light-scale", type=float, default=1.0, help="multiplies every light; tune on the pilot")
ap.add_argument("--panel-intensity", type=float, default=12000.0, help="ceiling RectLight inputs:intensity at scale 1")
ap.add_argument("--rt-subframes", type=int, default=-1, help="rep.orchestrator.step rt_subframes (more = less ghosting)")
ap.add_argument("--assets-root", default=None, help="override isaacsim.storage.native.get_assets_root_path()")
ap.add_argument("--character-folder", default="/Isaac/People/Characters", help="relative to the assets root")
ap.add_argument("--jpeg-quality", type=int, default=92)
ap.add_argument("--max-episodes", type=int, default=None)
ap.add_argument("--gui", action="store_true", help="run with a window (debugging on a desktop)")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must come before any omni import)

simulation_app = SimulationApp({"headless": not args.gui})

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

for _ext in ("omni.kit.scripting", "omni.anim.graph.core", "omni.anim.navigation.core", "omni.anim.people",
             "isaacsim.replicator.agent.core"):
    enable_extension(_ext)
for _ in range(10):
    simulation_app.update()

import carb  # noqa: E402
import carb.events  # noqa: E402
import numpy as np  # noqa: E402
import omni.anim.graph.core as ag  # noqa: E402
import omni.anim.navigation.core as nav  # noqa: E402
import omni.client  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import NavSchema  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402
from isaacsim.core.utils.semantics import add_update_semantics  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from isaacsim.replicator.agent.core.settings import AssetPaths  # noqa: E402
from isaacsim.replicator.agent.core.simulation import SimulationManager  # noqa: E402
from isaacsim.replicator.agent.core.stage_util import CharacterUtil  # noqa: E402
from omni.anim.people.scripts.custom_command.command_manager import CustomCommandManager  # noqa: E402

from bree.sim.isaac.convert import bbox_rows, split_skeletons  # noqa: E402
from bree.sim.isaac.plan import REACH_S, command_file, forward, right_of, with_facing_offset  # noqa: E402

COMMAND_FILE_SETTING = "/exts/omni.anim.people/command_settings/command_file_path"   # omni.anim.people 0.6.7 PeopleSettings
CHAR_ROOT = "/World/Characters"
START_EVT = carb.events.type_from_string("omni.anim.people/CommandStartEvent")
END_EVT = carb.events.type_from_string("omni.anim.people/CommandEndEvent")
HAND_JOINTS = {"right": ["R_Hand", "R_Wrist", "RightHand"], "left": ["L_Hand", "L_Wrist", "LeftHand"]}
ANCHORS = {  # (forward, right, up) metres from the character root, in the character's frame
    "pocket": (0.05, 0.17, 0.85), "waistband": (0.13, 0.0, 0.98), "under_clothing": (0.13, 0.0, 1.2),
    "bag": (0.0, -0.3, 0.85),
}
HAND_FALLBACK = {"right": (0.3, 0.2, 1.0), "left": (0.3, -0.2, 1.0)}
CONCEAL_SLIDE = (0.2, 0.8)          # s into the conceal Idle: product slides to the anchor, then disappears
DOOR_SWING_S = 0.4

settings = carb.settings.get_settings()
ASSETS = (args.assets_root or get_assets_root_path() or "").rstrip("/")
if not ASSETS:
    raise SystemExit("No Isaac Sim assets root (get_assets_root_path() returned nothing); pass --assets-root")
settings.set(AssetPaths.DEFAULT_BIPED_ASSET_PATH, f"{ASSETS}/Isaac/People/Characters/Biped_Setup.usd")
settings.set("/persistent/exts/omni.anim.people/character_prim_path", CHAR_ROOT)
settings.set("/omni/replicator/captureOnPlay", False)
settings.set("/app/player/useFixedTimeStepping", True)

_ccm = CustomCommandManager.get_instance()
if not _ccm.is_custom_command_name_exist("PushButton"):
    try:
        _ccm.add_custom_command(f"{ASSETS}/Isaac/People/Animations/push_button.skelanim.usd")
    except Exception:
        traceback.print_exc()
if not _ccm.is_custom_command_name_exist("PushButton"):
    raise SystemExit("could not register the PushButton custom command (push_button.skelanim.usd)")

SIM = SimulationManager()   # holds the recast interface, exactly like IRA does during a run
_char_usd_cache: dict[str, str] = {}


# ------------------------------------------------------------------ helpers
def log(*a):
    print("[bree-gen]", *a, flush=True)


def character_usd(folder: str) -> str:
    if folder not in _char_usd_cache:
        url = f"{ASSETS}{args.character_folder}/{folder}"
        result, entries = omni.client.list(url)
        usds = sorted(e.relative_path for e in entries if e.relative_path.endswith((".usd", ".usda", ".usdc"))) \
            if result == omni.client.Result.OK else []
        _char_usd_cache[folder] = f"{url}/{usds[0] if usds else folder + '.usd'}"
    return _char_usd_cache[folder]


def _payload(e, key):
    try:
        return e.payload[key]
    except Exception:
        return None


def yaw_of(rot) -> float:
    """carb.Float4 (x, y, z, w) -> angle in the GoTo convention (same math as omni.anim.people Utils)."""
    r = Gf.Rotation(Gf.Quatd(rot.w, rot.x, rot.y, rot.z))
    a = r.GetAngle()
    return (-a if r.GetAxis()[2] < 0 else a) - args.facing_offset


def body_point(root, yaw, off):
    fx, fy = forward(yaw)
    rx, ry = right_of(yaw)
    return (root[0] + off[0] * fx + off[1] * rx, root[1] + off[0] * fy + off[1] * ry, root[2] + off[2])


class Materials:
    def __init__(self, stage):
        self.stage, self.cache = stage, {}

    def get(self, color, kind):
        key = (tuple(round(c, 2) for c in color), kind)
        if key not in self.cache:
            path = f"/World/Looks/m{len(self.cache):03d}"
            mdl, name = ("OmniGlass.mdl", "OmniGlass") if kind == "glass" else ("OmniPBR.mdl", "OmniPBR")
            omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mdl, mtl_name=name, mtl_path=path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():   # OmniGlass missing: plain OmniPBR
                omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR",
                                          mtl_path=path)
                prim, kind = self.stage.GetPrimAtPath(path), "matte"
            shader = UsdShade.Shader(omni.usd.get_shader_from_material(prim, get_prim=True))
            if kind == "glass":
                shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key[0]))
            else:
                shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key[0]))
                shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(
                    0.35 if kind == "metal" else 0.75)
                shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.5 if kind == "metal" else 0.0)
            self.cache[key] = UsdShade.Material(prim)
        return self.cache[key]

    def bind(self, prim, color, kind="matte"):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.get(color, kind), UsdShade.Tokens.strongerThanDescendants)


def cube(stage, path, center, size, mats, color, kind="matte"):
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    xf = UsdGeom.XformCommonAPI(c)
    xf.SetTranslate(Gf.Vec3d(*center))
    xf.SetScale(Gf.Vec3f(*size))
    mats.bind(c.GetPrim(), color, kind)
    return c.GetPrim()


def nav_exclude(path):
    omni.kit.commands.execute("ApplyNavMeshAPICommand", prim_path=path, api=NavSchema.NavMeshExcludeAPI)


# ------------------------------------------------------------------ scene construction
def build_store(stage, plan, mats):
    sc = plan["scene"]
    UsdGeom.Xform.Define(stage, "/World/Store")
    for f in sc["fixtures"]:
        cube(stage, f"/World/Store/{f['name']}", f["center"], f["size"], mats, f["color"], f["material"])
    doors = {}
    for d in sc["doors"]:
        x = UsdGeom.Xform.Define(stage, f"/World/Store/{d['name']}")
        UsdGeom.XformCommonAPI(x).SetTranslate(Gf.Vec3d(d["hinge"][0], d["hinge"][1], 0.0))
        z0, z1 = d["z"]
        cube(stage, f"/World/Store/{d['name']}/glass", (-d["width"] / 2, 0, (z0 + z1) / 2), (d["width"], 0.02, z1 - z0),
             mats, (0.85, 0.92, 0.95), "glass")
        cube(stage, f"/World/Store/{d['name']}/handle", (-d["width"] + 0.06, -0.03, 1.1), (0.03, 0.03, 0.5),
             mats, (0.7, 0.7, 0.7), "metal")
        nav_exclude(str(x.GetPath()))
        doors[d["name"]] = {"xf": UsdGeom.XformCommonAPI(x), "open": d["open_deg"], "intervals": []}
    products = {}
    UsdGeom.Xform.Define(stage, "/World/Store/Products")
    for p in sc["products"]:
        path = f"/World/Store/Products/{p['id']}"
        x = UsdGeom.Xform.Define(stage, path)
        if p["shape"] == "cyl":
            g = UsdGeom.Cylinder.Define(stage, f"{path}/geom")
            g.GetRadiusAttr().Set(p["size"][0])
            g.GetHeightAttr().Set(p["size"][1])
            g.GetAxisAttr().Set("Z")
        else:
            g = UsdGeom.Cube.Define(stage, f"{path}/geom")
            g.GetSizeAttr().Set(1.0)
            UsdGeom.XformCommonAPI(g).SetScale(Gf.Vec3f(*p["size"]))
        mats.bind(g.GetPrim(), p["color"], "matte")
        add_update_semantics(x.GetPrim(), p["category"])
        xf = UsdGeom.XformCommonAPI(x)
        xf.SetTranslate(Gf.Vec3d(*p["pos"]))
        xf.SetRotate(Gf.Vec3f(0, 0, p["yaw"]))
        nav_exclude(path)
        products[p["id"]] = {"spec": p, "prim": x.GetPrim(), "xf": xf, "state": "shelf", "holder": None, "hand": None,
                             "pos": tuple(p["pos"])}
    return doors, products


def build_lights(stage, plan):
    L = plan["lighting"]
    s = args.light_scale
    panels = []
    for i, (x, y, z) in enumerate(L["panels"]):
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/panel_{i:02d}")
        lt.CreateWidthAttr(0.6)
        lt.CreateHeightAttr(1.2)
        lt.CreateEnableColorTemperatureAttr(True)
        lt.CreateColorTemperatureAttr(float(L["color_temp_k"]))
        base = args.panel_intensity * L["panel_scale"] * s
        lt.CreateIntensityAttr(base)
        UsdGeom.XformCommonAPI(lt).SetTranslate(Gf.Vec3d(x, y, z))   # RectLight emits along its -Z: down
        panels.append((lt.GetIntensityAttr(), base, i in L["flicker_panels"]))
    cl = UsdLux.RectLight.Define(stage, "/World/Lights/cooler")
    cl.CreateWidthAttr(3.7)
    cl.CreateHeightAttr(1.6)
    cl.CreateIntensityAttr(args.panel_intensity * 0.6 * L["cooler_light_scale"] * s)
    cxf = UsdGeom.XformCommonAPI(cl)
    cxf.SetTranslate(Gf.Vec3d(0.0, 4.35, 1.05))
    cxf.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))      # -Z emission turned toward -y (out of the cooler)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/sky")
    dome.CreateIntensityAttr(1000.0 * L["dome_scale"] * s)
    return panels, random.Random(L["flicker_seed"])


def build_cameras(stage, plan):
    cams = []
    for c in plan["cameras"]:
        path = f"/World/Cameras/{c['name']}"
        cam = UsdGeom.Camera.Define(stage, path)
        w, h = c["resolution"]
        cam.CreateFocalLengthAttr(float(c["focal_mm"]))
        cam.CreateHorizontalApertureAttr(float(c["h_aperture_mm"]))
        cam.CreateVerticalApertureAttr(float(c["h_aperture_mm"]) * h / w)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
        UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*[v for row in c["transform"] for v in row]))
        rp = rep.create.render_product(path, (w, h), name=c["name"])
        annots = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
                  "bbox": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight",
                                                              init_params={"semanticTypes": ["class"]}),
                  "skel": rep.AnnotatorRegistry.get_annotator("skeleton_data"),
                  "cam": rep.AnnotatorRegistry.get_annotator("camera_params")}
        for a in annots.values():
            a.attach(rp)
        rp.hydra_texture.set_updates_enabled(False)
        cams.append({"name": c["name"], "rp": rp, "annots": annots})
    return cams


def bake_navmesh(stage, plan, timeout_s=300):
    v = plan["scene"]["nav_volume"]
    vol = stage.DefinePrim("/World/NavMeshVolume", "NavMeshVolume")
    vol.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xf = UsdGeom.XformCommonAPI(vol)
    xf.SetTranslate(Gf.Vec3d(*[(v["lo"][i] + v["hi"][i]) / 2 for i in range(3)]))
    xf.SetScale(Gf.Vec3f(*[v["hi"][i] - v["lo"][i] for i in range(3)]))
    inav = nav.acquire_interface()
    done = {"ok": False}

    def on_nav(e):
        if e.type == nav.EVENT_TYPE_NAVMESH_UPDATED:
            done["ok"] = True
    sub = inav.get_navmesh_event_stream().create_subscription_to_pop(on_nav)
    if not inav.start_navmesh_baking():
        raise RuntimeError("NavMesh baking could not start (is the NavMeshVolume valid?)")
    t0 = time.time()
    while not (done["ok"] and inav.get_navmesh() is not None):
        simulation_app.update()
        if time.time() - t0 > timeout_s:
            raise RuntimeError("NavMesh baking timed out")
    del sub                     # release the navmesh event subscription
    log(f"navmesh baked in {time.time() - t0:.1f}s")


def spawn_people(stage, plan, mats):
    SIM.load_default_skeleton_and_animations()      # /World/Characters + Biped_Setup + anim graph (IRA 0.5.14)
    people = {}
    for p in plan["people"]:
        x, y, ang = p["start"]
        prim = CharacterUtil.load_character_usd_to_stage(character_usd(p["asset"]), (x, y, 0.0),
                                                         ang + args.facing_offset, p["name"])
        nav_exclude(str(prim.GetPath()))
        people[p["name"]] = {"spec": p, "root_path": str(prim.GetPath()), "skel": None, "char": None, "hand": {},
                             "inside": False, "done": False, "ptr": 0, "pos": (x, y, 0.0), "yaw": ang, "bag": None}
        if p["has_bag"]:
            people[p["name"]]["bag"] = UsdGeom.XformCommonAPI(
                cube(stage, f"/World/Props/bag_{p['name']}", (x, y, 0.85), (0.3, 0.12, 0.32), mats, (0.12, 0.1, 0.1)))
    skelroots = [s for s in CharacterUtil.get_characters_in_stage() if str(s.GetPath()).startswith(CHAR_ROOT)]
    SIM.setup_animation_graph_to_character(skelroots)
    SIM.setup_python_scripts_to_character(skelroots)
    for s in skelroots:
        add_update_semantics(s, "person")
        name = str(s.GetPath()).split("/")[3]
        if name in people:
            people[name]["skel"] = str(s.GetPath())
    missing = [n for n, p in people.items() if p["skel"] is None]
    if missing:
        raise RuntimeError(f"no SkelRoot found for {missing}")
    return people


# ------------------------------------------------------------------ the episode
class Episode:
    def __init__(self, plan: dict, out: Path):
        self.plan, self.out = plan, out
        self.events, self.warn, self.diag = [], [], {"reach_miss_m": [], "facing_err_deg": [], "reach_peak_s": []}
        self.pending = []           # scheduled product actions
        self.reaches = []           # PushButton reaches being measured
        self.t = 0.0

    def setup(self):
        ctx = omni.usd.get_context()
        ctx.new_stage()
        for _ in range(5):
            simulation_app.update()
        stage = self.stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        for p in ("/World", "/World/Looks", "/World/Lights", "/World/Cameras", "/World/Props"):
            UsdGeom.Xform.Define(stage, p)
        self.mats = Materials(stage)
        self.doors, self.products = build_store(stage, self.plan, self.mats)
        self.panels, self.flicker_rng = build_lights(stage, self.plan)
        bake_navmesh(stage, self.plan)
        self.people = spawn_people(stage, self.plan, self.mats)
        self.cams = build_cameras(stage, self.plan)
        cmd_path = self.out / "commands.txt"
        cmd_path.write_text(with_facing_offset(command_file(self.plan), args.facing_offset))
        settings.set(COMMAND_FILE_SETTING, str(cmd_path))
        # restock items start in the employee's hand (one visible at a time)
        self.stock = {}
        for pid, pr in self.products.items():
            init = pr["spec"]["initial"]
            if init.startswith("hand:"):
                self.stock.setdefault(init[5:], []).append(pid)
                pr["state"] = "stock"
                UsdGeom.Imageable(pr["prim"]).MakeInvisible()
        bus = omni.kit.app.get_app().get_message_bus_event_stream()
        self.subs = [bus.create_subscription_to_pop_by_type(START_EVT, self.on_start),
                     bus.create_subscription_to_pop_by_type(END_EVT, self.on_end)]
        for _ in range(10):
            simulation_app.update()

    # ---- omni.anim.people command events -> product actions
    def _match(self, e):
        name = _payload(e, "agent_name")
        person = self.people.get(name)
        if person is None:
            return None, None
        cmd = " ".join(str(_payload(e, "command"))[len(name):].split())
        if "cmds" not in person:
            person["cmds"] = [" ".join(c.split()) for c in
                              with_facing_offset("\n".join(person["spec"]["commands"]), args.facing_offset).split("\n")]
        cmds = person["cmds"]
        for i in range(person["ptr"], len(cmds)):
            if cmds[i] == cmd:
                return person, i
        for i in range(person["ptr"], len(cmds)):   # same command name, arguments reformatted by omni.anim.people
            if cmds[i].split(" ")[0] == cmd.split(" ")[0]:
                return person, i
        return person, None

    def on_start(self, e):
        person, i = self._match(e)
        if person is None:
            return
        if i is None:
            self.warn.append(f"t={self.t:.1f} {person['spec']['name']}: unplanned command '{_payload(e, 'command')}'")
            return
        person["ptr"] = i
        act = person["spec"]["actions"][i]
        if act:
            self.pending.append({**act, "person": person["spec"]["name"], "t0": self.t, "stage": 0})
            if act["verb"] != "conceal":
                self.reaches.append({"person": person["spec"]["name"], "t0": self.t, "dur": REACH_S, "best": (-9.0, 0.0)})
            if act["verb"] in ("take", "put", "restock", "touch"):
                door = self.products[act["product"]]["spec"].get("door")
                if door:
                    self.doors[door]["intervals"].append((self.t, self.t + REACH_S + 0.6))

    def on_end(self, e):
        person, i = self._match(e)
        if person is None or i is None:
            return
        if _payload(e, "status") == "failed":
            self.warn.append(f"t={self.t:.1f} {person['spec']['name']}: command failed '{_payload(e, 'command')}'")
        person["ptr"] = i + 1
        if i + 1 >= len(person["spec"]["commands"]):
            person["done"] = True

    # ---- per-frame state
    def _char(self, person):
        if person["char"] is None:
            person["char"] = ag.get_character(person["skel"])
        return person["char"]

    def update_people(self):
        door_y = self.plan["scene"]["door_line_y"]
        for name, person in self.people.items():
            ch = self._char(person)
            if ch is None:
                continue
            pos, rot = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 1)
            ch.get_world_transform(pos, rot)
            person["pos"], person["yaw"] = (pos.x, pos.y, pos.z), yaw_of(rot)
            inside = pos.y > door_y
            if person["spec"]["role"] == "shopper" and inside != person["inside"]:
                self.event("enter" if inside else "exit", name)
            person["inside"] = inside
            if person["bag"] is not None:
                person["bag"].SetTranslate(Gf.Vec3d(*body_point(person["pos"], person["yaw"], ANCHORS["bag"])))
                person["bag"].SetRotate(Gf.Vec3f(0, 0, person["yaw"] + args.facing_offset))

    def hand_pos(self, person, hand):
        """World position of a hand joint (cached joint name), or a body offset if no joint answers."""
        ch = self._char(person)
        if ch is None:
            return body_point(person["pos"], person["yaw"], HAND_FALLBACK[hand])
        if hand not in person["hand"]:
            person["hand"][hand] = None
            for j in HAND_JOINTS[hand]:
                p, r = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 1)
                try:
                    ch.get_joint_transform(j, p, r)
                except Exception:
                    continue
                root = person["pos"]
                if 0.3 < p.z < 2.3 and math.hypot(p.x - root[0], p.y - root[1]) < 1.2:
                    person["hand"][hand] = j
                    break
            if person["hand"][hand] is None:
                self.warn.append(f"{person['spec']['name']}: no {hand} hand joint answered; using a body offset")
        j = person["hand"][hand]
        if j is None:
            return body_point(person["pos"], person["yaw"], HAND_FALLBACK[hand])
        p, r = carb.Float3(0, 0, 0), carb.Float4(0, 0, 0, 1)
        ch.get_joint_transform(j, p, r)
        return (p.x, p.y, p.z)

    def held_by(self, name):
        return {pr["hand"]: pid for pid, pr in self.products.items() if pr["state"] == "hand" and pr["holder"] == name}

    def attach(self, pid, name, hand):
        pr = self.products[pid]
        pr.update(state="hand", holder=name, hand=hand)
        UsdGeom.Imageable(pr["prim"]).MakeVisible()

    def place(self, pid):
        pr = self.products[pid]
        pr.update(state="shelf", holder=None, hand=None, pos=tuple(pr["spec"]["pos"]))
        pr["xf"].SetTranslate(Gf.Vec3d(*pr["spec"]["pos"]))
        pr["xf"].SetRotate(Gf.Vec3f(0, 0, pr["spec"]["yaw"]))
        UsdGeom.Imageable(pr["prim"]).MakeVisible()

    def event(self, action, person, **kw):
        fps_out = self.plan["fps_out"]
        self.events.append({"t": round(self.t, 3), "frame": int(round(self.t * fps_out)), "person": person,
                            "action": action, **kw})

    def track_reaches(self):
        """Measure when the right hand is furthest forward during each PushButton (calibrates --grasp-at)."""
        for tr in list(self.reaches):
            person, dt = self.people[tr["person"]], self.t - tr["t0"]
            if dt > tr["dur"]:
                self.diag["reach_peak_s"].append(round(tr["best"][1], 2))
                self.reaches.remove(tr)
                continue
            hp = self.hand_pos(person, "right")
            fx, fy = forward(person["yaw"])
            ext = (hp[0] - person["pos"][0]) * fx + (hp[1] - person["pos"][1]) * fy
            if ext > tr["best"][0]:
                tr["best"] = (ext, dt)

    def run_actions(self):
        ga = args.grasp_at
        self.track_reaches()
        for a in list(self.pending):
            person = self.people[a["person"]]
            name, dt = a["person"], self.t - a["t0"]
            v = a["verb"]
            if dt < (CONCEAL_SLIDE[0] if v == "conceal" else ga):
                continue
            pid = a.get("product")
            if v in ("take", "touch"):
                self._reach_diag(person, pid)
            if v == "take":
                held = self.held_by(name)
                if "right" in held:
                    if "left" in held:
                        self.warn.append(f"{name}: both hands full at a take; dropping {held['left']} into the bag")
                        self.products[held["left"]]["state"] = "hidden"
                        UsdGeom.Imageable(self.products[held["left"]]["prim"]).MakeInvisible()
                    self.products[held["right"]]["hand"] = "left"
                self.attach(pid, name, "right")
                self.event("pick", name, product=pid, category=a["category"], sku=a["sku"], hand="right")
            elif v in ("put", "restock"):
                self.place(pid)
                self.event("put_back" if v == "put" else "restock", name, product=pid, category=a["category"],
                           sku=a["sku"])
                if v == "restock":
                    queue = self.stock.get(name, [])
                    if pid in queue:
                        queue.remove(pid)
                    if queue:
                        self.attach(queue[0], name, "right")
            elif v == "touch":
                self.event("touch", name, product=pid, category=a["category"])
            elif v == "pay":
                self.event("pay", name, txn=a["txn"], terminal=a["terminal"], skus=a["skus"], products=a["products"])
            elif v == "conceal":
                pr = self.products[pid]
                if a["stage"] == 0:
                    a["stage"], a["from_hand"] = 1, pr["hand"] or "right"
                    pr.update(state="concealing")
                if dt < CONCEAL_SLIDE[1]:
                    u = (dt - CONCEAL_SLIDE[0]) / (CONCEAL_SLIDE[1] - CONCEAL_SLIDE[0])
                    h = self.hand_pos(person, a["from_hand"])
                    b = body_point(person["pos"], person["yaw"], ANCHORS[a["anchor"]])
                    p = tuple(h[i] + (b[i] - h[i]) * u for i in range(3))
                    pr["pos"] = p
                    pr["xf"].SetTranslate(Gf.Vec3d(*p))
                    continue
                pr.update(state="hidden", holder=name, hand=None)
                UsdGeom.Imageable(pr["prim"]).MakeInvisible()
                self.event("conceal", name, product=pid, category=a["category"], sku=a["sku"], anchor=a["anchor"])
            self.pending.remove(a)
        # products follow hands; restock items wait in the employee's hand
        for name, queue in self.stock.items():
            if queue and self.products[queue[0]]["state"] == "stock":
                self.attach(queue[0], name, "right")
        for pid, pr in self.products.items():
            if pr["state"] == "hand":
                person = self.people[pr["holder"]]
                h = self.hand_pos(person, pr["hand"])
                pr["pos"] = (h[0], h[1], h[2] - 0.06)
                pr["xf"].SetTranslate(Gf.Vec3d(*pr["pos"]))
                pr["xf"].SetRotate(Gf.Vec3f(0, 0, person["yaw"] + args.facing_offset))

    def _reach_diag(self, person, pid):
        h = self.hand_pos(person, "right")
        p = self.products[pid]["pos"]
        self.diag["reach_miss_m"].append(round(math.dist(h, p), 3))
        fx, fy = forward(person["yaw"])
        dx, dy = p[0] - person["pos"][0], p[1] - person["pos"][1]
        ang = math.degrees(math.atan2(fx * dy - fy * dx, fx * dx + fy * dy))
        self.diag["facing_err_deg"].append(round(ang, 1))

    def update_scene(self, frame):
        for d in self.doors.values():
            k = 0.0
            for a, b in d["intervals"]:
                if a <= self.t <= b:
                    k = max(k, min(1.0, (self.t - a) / DOOR_SWING_S, (b - self.t) / DOOR_SWING_S))
            d["xf"].SetRotate(Gf.Vec3f(0, 0, d["open"] * k))
        for attr, base, flick in self.panels:
            if flick:
                r = self.flicker_rng.random()
                attr.Set(base * (self.flicker_rng.uniform(0.1, 0.5) if r < 0.12 else self.flicker_rng.uniform(0.9, 1.05)))

    # ---- capture
    def capture(self, k, files):
        for c in self.cams:
            c["rp"].hydra_texture.set_updates_enabled(True)
        rep.orchestrator.step(delta_time=0.0, rt_subframes=args.rt_subframes)
        for c in self.cams:
            d = self.out / c["name"]
            rgb = np.asarray(c["annots"]["rgb"].get_data())
            if rgb.ndim != 3:                       # renderer not warm yet: keep frame numbering, write black
                w, h = next(cc["resolution"] for cc in self.plan["cameras"] if cc["name"] == c["name"])
                rgb = np.zeros((h, w, 3), np.uint8)
                self.warn.append(f"{c['name']}: empty rgb at frame {k}")
            Image.fromarray(rgb[..., :3]).save(d / "rgb" / f"{k:05d}.jpg", quality=args.jpeg_quality)
            if k == 0:
                self.lum[c["name"]] = float(np.asarray(rgb)[..., :3].mean())
                cp = c["annots"]["cam"].get_data()
                (d / "camera.json").write_text(json.dumps(
                    {key: v if isinstance(v, str) else np.asarray(v).tolist() for key, v in cp.items()}))
            row = {"f": k, "t": round(self.t, 3), "boxes": bbox_rows(c["annots"]["bbox"].get_data()),
                   "skeletons": split_skeletons(c["annots"]["skel"].get_data(), CHAR_ROOT)}
            files[c["name"]].write(json.dumps(row) + "\n")
            c["rp"].hydra_texture.set_updates_enabled(False)
        world = {"f": k, "t": round(self.t, 3),
                 "people": {n: {"pos": [round(v, 3) for v in p["pos"]], "yaw": round(p["yaw"], 1), "inside": p["inside"]}
                            for n, p in self.people.items()},
                 "products": {pid: {"state": pr["state"], "holder": pr["holder"], "hand": pr["hand"],
                                    "pos": [round(v, 3) for v in pr["pos"]]}
                              for pid, pr in self.products.items() if pr["state"] != "shelf"}}
        files["world"].write(json.dumps(world) + "\n")

    def run(self):
        self.lum = {}
        fps_out, sim_fps = self.plan["fps_out"], args.sim_fps
        every = max(1, round(sim_fps / fps_out))
        max_frames = int(self.plan["max_duration_s"] * sim_fps)
        for c in self.cams:
            (self.out / c["name"] / "rgb").mkdir(parents=True, exist_ok=True)
        files = {c["name"]: open(self.out / c["name"] / "ann.jsonl", "w") for c in self.cams}
        files["world"] = open(self.out / "world.jsonl", "w")
        tl = omni.timeline.get_timeline_interface()
        tl.set_looping(False)
        tl.set_time_codes_per_second(sim_fps)
        tl.set_current_time(0.0)
        tl.set_end_time(self.plan["max_duration_s"] + 30.0)
        tl.play()
        tl.commit()
        shoppers = [p for p in self.people.values() if p["spec"]["role"] == "shopper"]
        k, t_all_out, wall0 = 0, None, time.time()
        try:
            for f in range(max_frames):
                self.t = tl.get_current_time()
                self.update_people()
                self.run_actions()
                self.update_scene(f)
                if f % every == 0:
                    self.capture(k, files)
                    k += 1
                if not tl.is_playing():
                    tl.play()
                simulation_app.update()
                if all(p["done"] and not p["inside"] for p in shoppers):
                    t_all_out = t_all_out if t_all_out is not None else self.t
                    if self.t - t_all_out > 3.0:
                        break
            else:
                self.warn.append(f"hit max_duration_s={self.plan['max_duration_s']} before every shopper left")
        finally:
            for fh in files.values():
                fh.close()
            tl.stop()
            tl.commit()
            self.subs = []
            for c in self.cams:
                for a in c["annots"].values():
                    try:
                        a.detach()
                    except Exception:
                        pass
                c["rp"].destroy()
        wall = time.time() - wall0
        return k, wall

    def finish(self, frames_out, wall):
        d = self.diag
        med = {key: statistics.median([abs(x) for x in v]) if v else None for key, v in d.items()}
        if med["reach_miss_m"] is not None and med["reach_miss_m"] > 0.35:
            self.warn.append(f"median hand-to-product distance at grasp {med['reach_miss_m']:.2f} m: check --grasp-at "
                             f"(median reach peak {med['reach_peak_s']} s) and --facing-offset "
                             f"(median facing error {med['facing_err_deg']} deg)")
        for cam, v in self.lum.items():
            if not 25 <= v <= 230:
                self.warn.append(f"{cam}: first frame mean brightness {v:.0f}/255, tune --light-scale")
        meta = {"episode": self.plan["episode"], "frames_out": frames_out, "fps_out": self.plan["fps_out"],
                "sim_fps": args.sim_fps, "duration_s": round(self.t, 2), "wall_s": round(wall, 1),
                "wall_s_per_video_s": round(wall / max(self.t, 1e-6), 2),
                "kit_version": omni.kit.app.get_app().get_kit_version(),
                "args": {k: v for k, v in vars(args).items() if k not in ("only",)},
                "hand_joints": {n: p["hand"] for n, p in self.people.items()},
                "people_done": {n: p["done"] for n, p in self.people.items()},
                "diagnostics": d, "diagnostics_median": med, "brightness": self.lum, "warnings": self.warn}
        (self.out / "events.json").write_text(json.dumps(self.events, indent=1))
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1))
        (self.out / "done.json").write_text(json.dumps({"frames_out": frames_out, "wall_s": round(wall, 1)}))
        return meta


def main():
    plans = sorted(Path(args.plans).glob("s*_t*.json"))
    if args.only:
        plans = [p for p in plans if p.stem in set(args.only)]
    i, n = (int(v) for v in args.shard.split("/"))
    plans = plans[i::n][: args.max_episodes]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    fails = 0
    for path in plans:
        out = out_root / path.stem
        if (out / "done.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        plan = json.loads(path.read_text())
        (out / "plan.json").write_text(json.dumps(plan))
        log(f"episode {plan['episode']} ({plan['split']}): {len(plan['people'])} people, max {plan['max_duration_s']}s")
        try:
            ep = Episode(plan, out)
            ep.setup()
            frames, wall = ep.run()
            meta = ep.finish(frames, wall)
            log(f"  {frames} frames x {len(plan['cameras'])} cams, {meta['duration_s']}s video in {wall:.0f}s wall")
            for w in meta["warnings"]:
                log("  WARNING", w)
            fails = 0
        except Exception:
            (out / "error.txt").write_text(traceback.format_exc())
            log(f"  FAILED, see {out / 'error.txt'}")
            fails += 1
            if fails >= 3:
                log("3 failures in a row, stopping")
                break
    simulation_app.close()


if __name__ == "__main__":
    main()
