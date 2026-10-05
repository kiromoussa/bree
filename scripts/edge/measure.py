"""Measure the node's first pass on recorded clips: trigger recall, false triggers per hour, bytes
sent against streaming full resolution all the time, and CPU cost on this machine.

    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py merl --split test     # real lab video, labelled reaches
    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py merl --split train --every 6 --sweep
    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py toy                   # TOY clips, true pick times
    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py cpu                   # per-frame cost + Zero 2 W estimate
    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py synthetic             # SYNTHETIC empty scene: shake, flicker
    PYTHONPATH=src .venv/bin/python scripts/edge/measure.py table                 # markdown tables from the saved results

Truth. MERL labels hand actions per frame: 1 reach to shelf, 2 retract, 3 hand in shelf. It has no
"pick" label; every pick starts with a reach, so recall is counted on reaches. Toy clips: the renderer's
own script gives the moment each item is taken or put back.

Definitions (PAD = 0.5 s):
  trigger recall   reaches with the trigger active at some frame in [start - PAD, end + PAD]
  captured recall  reaches whose whole [start, end] lies inside full-res bursts (pre-roll included);
                   toy: the moment the item is taken or put back lies inside a burst
  false trigger    a run of active frames that touches no labelled reach / retract / hand-in-shelf (+- PAD)
  baseline bytes   every captured frame as a full-res JPEG at the node's own frame rate and quality
  always-on control  the same recall and false-trigger counts for a trigger that is active on every frame.
                   It scores full recall and no false trigger, so those two rows alone cannot tell a
                   good trigger from one stuck on. The frame-level rows and the bytes can:
  active outside labels  active frames farther than PAD from any labelled reach / retract / hand-in-shelf
  reach frames active    frames inside a labelled reach (no PAD) with the trigger active
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from bree.edge.capture import FileSource
from bree.edge.node import Node, NodeConfig
from bree.edge.trigger import TriggerConfig, ZoneTrigger
from bree.events.zones import load_store_config

PAD = 0.5
ROOT = Path(__file__).resolve().parents[2]


class CountingUplink:
    """Stands in for the network: packs each message like the real uplink and keeps only its size."""
    status = None

    def __init__(self):
        self.windows: list[tuple[float, float]] = []

    def submit(self, header, blobs):
        from bree.edge.uplink import pack
        if header["kind"] == "burst":
            self.windows.append((header["frames"][0]["t"], header["frames"][-1]["t"]))
        return len(pack({**header, "camera_id": "x", "boot_id": "0" * 12, "seq": 0, "clock": "media"}, blobs))

    def pressure(self):
        return 0.0


class CountingSource(FileSource):
    """FileSource that also adds up what streaming every full-res frame as JPEG would cost."""
    baseline_bytes = 0
    quality = 85

    def frames(self):
        for fr in super().frames():
            self.baseline_bytes += len(self.encode(fr.main, self.quality))
            yield fr


def runs(log: list[tuple[float, bool]], dt: float) -> list[tuple[float, float]]:
    out, start, prev = [], None, None
    for t, on in log + [(float("inf"), False)]:
        if on and start is None:
            start = t
        elif not on and start is not None:
            out.append((start, prev))
            start = None
        prev = t
    return out


def _hit_false(t, on, picks, shelf, dt):
    trig = runs(list(zip(t.tolist(), on.tolist())), dt)
    hit = [bool(on[(t >= s - PAD) & (t <= e + PAD)].any()) for s, e in picks]
    false = [r for r in trig if not any(r[0] <= e + PAD and r[1] >= s - PAD for s, e in shelf)]
    return trig, hit, false


def _mask(t, spans, pad):
    m = np.zeros(len(t), bool)
    for s, e in spans:
        m |= (t >= s - pad) & (t <= e + pad)
    return m


def score(active: list[tuple[float, bool]], windows, picks, shelf, dt: float, reach=None) -> dict:
    """picks / shelf / reach: lists of (start_s, end_s). reach defaults to picks (MERL: picks are reaches)."""
    t = np.array([a[0] for a in active])
    on = np.array([a[1] for a in active], bool)
    trig, hit, false = _hit_false(t, on, picks, shelf, dt)
    _, hit1, false1 = _hit_false(t, np.ones(len(t), bool), picks, shelf, dt)      # always-on control
    lab, rch = _mask(t, shelf, PAD), _mask(t, picks if reach is None else reach, 0.0)
    inb = np.zeros(len(t), bool)
    for a, b in windows:
        inb |= (t >= a) & (t <= b)
    merged = runs(list(zip(t.tolist(), inb.tolist())), dt)       # back-to-back burst parts become one window
    cap = [any(a <= s and e <= b for a, b in merged) for s, e in picks]
    return {"picks": len(picks), "hit": int(sum(hit)), "captured": int(sum(cap)), "triggers": len(trig),
            "false_triggers": len(false), "seconds": float(len(t) * dt), "burst_frames": int(inb.sum()),
            "frames": int(len(t)), "active_frames": int(on.sum()),
            "always_on_hit": int(sum(hit1)), "always_on_false": len(false1),
            "label_frames": int(lab.sum()), "active_outside_label_frames": int((on & ~lab).sum()),
            "reach_frames": int(rch.sum()), "reach_active_frames": int((on & rch).sum())}


def pool(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float)) and k != "fps"}
    hours = s["seconds"] / 3600
    out = {"clips": len(rows), "hours": round(hours, 4), "picks": s["picks"],
           "trigger_recall": round(s["hit"] / max(s["picks"], 1), 4),
           "captured_recall": round(s["captured"] / max(s["picks"], 1), 4),
           "missed": s["picks"] - s["hit"], "triggers": s["triggers"], "false_triggers": s["false_triggers"],
           "false_triggers_per_hour": round(s["false_triggers"] / hours, 1),
           "trigger_active_share": round(s["active_frames"] / s["frames"], 4),
           "burst_share_of_frames": round(s["burst_frames"] / s["frames"], 4),
           "always_on_recall": round(s["always_on_hit"] / max(s["picks"], 1), 4),
           "always_on_false_triggers": s["always_on_false"],
           "label_share_of_frames": round(s["label_frames"] / s["frames"], 4),
           "active_outside_label_share": round(s["active_outside_label_frames"] / max(s["active_frames"], 1), 4),
           "reach_frames": s["reach_frames"],
           "reach_frames_active_share": round(s["reach_active_frames"] / max(s["reach_frames"], 1), 4)}
    if "burst_bytes" in s:
        out.update(
            baseline_jpeg_bytes=s["baseline_bytes"], burst_bytes=s["burst_bytes"], lowres_bytes=s["lowres_bytes"],
            saved_bursts_only=round(1 - s["burst_bytes"] / s["baseline_bytes"], 4),
            saved_bursts_plus_lowres=round(1 - (s["burst_bytes"] + s["lowres_bytes"]) / s["baseline_bytes"], 4),
            baseline_jpeg_mbit_s=round(s["baseline_bytes"] * 8 / s["seconds"] / 1e6, 3),
            bursts_mbit_s=round(s["burst_bytes"] * 8 / s["seconds"] / 1e6, 3),
            lowres_mbit_s=round(s["lowres_bytes"] * 8 / s["seconds"] / 1e6, 3),
            trigger_ms_per_frame=round(1000 * s["trigger_s"] / s["frames"], 3))
        if s.get("h264_bytes"):
            out.update(baseline_h264_bytes=s["h264_bytes"],
                       baseline_h264_mbit_s=round(s["h264_bytes"] * 8 / s["seconds"] / 1e6, 3),
                       saved_vs_h264_bursts_only=round(1 - s["burst_bytes"] / s["h264_bytes"], 4))
    return out


def h264_bytes(video: str, fps: float) -> int:
    """The same clip as continuous H.264 at the node frame rate (libx264, crf 23, veryfast)."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.mp4"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-r", str(fps), "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "23", str(out)], capture_output=True)
        return out.stat().st_size if r.returncode == 0 else 0


def run_clip(video: str, zones: dict, resolution, picks, shelf, a, tcfg: TriggerConfig, h264: bool, reach=None) -> dict:
    src = CountingSource(video, a.fps, tcfg.width)
    src.quality = a.quality
    # src.fps is the rate the clip really runs at (a 15 fps clip asked for 10 runs at 7.5): use it everywhere
    cfg = NodeConfig("cam", resolution, zones, source=video, fps=src.fps, pre_roll_s=a.pre_roll, post_roll_s=a.post_roll,
                     max_burst_s=a.max_burst, jpeg_quality=a.quality, lowres_uplink_fps=src.fps, trigger=tcfg)
    up = CountingUplink()
    node = Node(cfg, src, up)
    node.keep_log = True
    node.run()
    r = score([(t, bool(z)) for t, z in node.active_log], up.windows, picks, shelf, 1 / src.fps, reach)
    r.update(clip=Path(video).stem, fps=src.fps, baseline_bytes=src.baseline_bytes, burst_bytes=node.burst_bytes,
             lowres_bytes=node.lowres_bytes, trigger_s=node.trigger_s, bursts=node.bursts,
             h264_bytes=h264_bytes(video, src.fps) if h264 else 0)
    return r


def real_fps(rows: list[dict], a) -> float:
    """The frame rate the clips really ran at (FileSource thins by a whole step), not the one asked for."""
    got = sorted({round(r["fps"], 3) for r in rows})
    if len(got) != 1:
        sys.exit(f"clips ran at different frame rates: {got}")
    if abs(got[0] - a.fps) > 0.01:
        print(f"note: asked for {a.fps} fps, the clips ran at {got[0]} fps (whole-frame thinning)", file=sys.stderr)
    return got[0]


# ------------------------------------------------------------------ MERL (real lab video)
def merl_videos(split: str, every: int) -> list[str]:
    vids = sorted(glob.glob(str(ROOT / "data/merl/Videos_MERL_Shopping_Dataset/*_crop.mp4")),
                  key=lambda f: tuple(map(int, re.findall(r"(\d+)_(\d+)_crop", f)[0])))
    subj = lambda v: int(Path(v).name.split("_")[0])
    keep = {"test": lambda s: s >= 27, "train": lambda s: s <= 20, "val": lambda s: 21 <= s <= 26}[split]
    return [v for v in vids if keep(subj(v))][::every]


def merl_truth(video: str):
    from scipy.io import loadmat
    stem = Path(video).name.replace("_crop.mp4", "")
    tl = loadmat(ROOT / f"data/merl/Labels_MERL_Shopping_Dataset/{stem}_label.mat")["tlabs"]
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
    gt = {k + 1: np.asarray(tl[k][0]).reshape(-1, 2) / fps for k in range(5)}
    return [tuple(x) for x in gt[1]], [tuple(x) for k in (1, 2, 3) for x in gt[k]]


SWEEP = [dict(margin_px=m, min_motion_px=mm, release_s=r, min_blob_px=b)
         for m in (0, 6, 12, 18) for mm, r, b in ((6, 0.5, 30), (3, 1.0, 15), (0, 0.5, 30), (3, 1.0, 30))]
# the two disturbance options, at the default settings otherwise (gain_norm False + shake_px 0 = the first version)
SWEEP += [dict(gain_norm=g, shake_px=s) for g in (False, True) for s in (0, 1)]


def cmd_merl(a) -> dict:
    store = load_store_config(ROOT / "configs/merl_overhead.yaml")
    zones = {z.name: z.polygon for z in store.zones if z.is_merch}
    vids = merl_videos(a.split, a.every)
    if a.sweep:   # trigger only (no JPEG work), every setting on the same decoded frames
        rows = {json.dumps(s): [] for s in SWEEP}
        for v in vids:
            picks, shelf = merl_truth(v)
            src = FileSource(v, a.fps, 320)
            trigs = {k: ZoneTrigger(zones, store.resolution, TriggerConfig(**json.loads(k))) for k in rows}
            logs = {k: [] for k in rows}
            for fr in src.frames():
                for k, tr in trigs.items():
                    w = tr.cfg.width
                    lo = fr.lores if w == 320 else cv2.resize(fr.main, tr.lowres_size(*src.size), interpolation=cv2.INTER_AREA)
                    logs[k].append((fr.t, bool(tr.update(lo, fr.t))))
            for k in rows:
                rows[k].append(score(logs[k], [], picks, shelf, 1 / src.fps))
        out = {"split": a.split, "fps": src.fps, "sweep": [{"settings": json.loads(k), **pool(r)} for k, r in rows.items()]}
        for s in out["sweep"]:
            print(f"{json.dumps(s['settings']):52s} recall {s['trigger_recall']:.3f} ({s['missed']} missed of {s['picks']})  "
                  f"false/h {s['false_triggers_per_hour']:7.1f}  active {s['trigger_active_share']:.2f}")
        return out
    tcfg = TriggerConfig.from_dict(json.loads(a.trigger))
    rows = []
    for v in vids:
        picks, shelf = merl_truth(v)
        rows.append(run_clip(v, zones, store.resolution, picks, shelf, a, tcfg, a.h264))
        print(rows[-1]["clip"], rows[-1]["hit"], "/", rows[-1]["picks"], "false", rows[-1]["false_triggers"], flush=True)
    return {"data": f"MERL Shopping, {a.split} split, real overhead lab video 920x680", "fps": real_fps(rows, a),
            "pre_roll_s": a.pre_roll, "post_roll_s": a.post_roll, "jpeg_quality": a.quality,
            "trigger": vars(tcfg), "pooled": pool(rows), "clips": rows}


# ------------------------------------------------------------------ toy clips (synthetic)
def cmd_toy(a) -> dict:
    from bree.sim.toy_render import scenarios
    store = load_store_config(ROOT / "configs/store_gas_station_small.yaml")
    zones = {z.name: z.polygon for z in store.zones if z.is_merch}
    tcfg = TriggerConfig.from_dict(json.loads(a.trigger))
    rows = []
    for sc in scenarios():
        video = ROOT / f"data/toy/toy_{sc.name}.mp4"
        if not video.exists():
            sys.exit(f"{video} missing: run `make demo` once to render the toy clips")
        reach = [(s.t0, s.t1) for ac in sc.actors for s in ac.segs if s.kind == "reach"]
        # a take or put-back happens at the middle of its reach
        picks = [(t, t) for ac in sc.actors for t, _, act, _ in ac.item_events if act in ("attach", "place")]
        rows.append(run_clip(str(video), zones, store.resolution, picks, reach, a, tcfg, a.h264, reach))
    return {"data": "TOY clips (synthetic, flat-colour drawings) 1280x720", "fps": real_fps(rows, a), "pre_roll_s": a.pre_roll,
            "post_roll_s": a.post_roll, "jpeg_quality": a.quality, "trigger": vars(tcfg), "pooled": pool(rows), "clips": rows}


# ------------------------------------------------------------------ CPU
# ASSUMPTION, not a measurement: one Cortex-A53 core at 1 GHz (Pi Zero 2 W) is taken to be 8 to 20 times
# slower than one core of the machine this runs on for this kind of OpenCV work. Replace it with a real
# number from the Pi: the node reports trigger_ms in every heartbeat.
PI_SLOWDOWN = (8, 20)


def cmd_cpu(a) -> dict:
    import platform
    cv2.setNumThreads(1)
    store = load_store_config(ROOT / "configs/merl_overhead.yaml")
    zones = {z.name: z.polygon for z in store.zones if z.is_merch}
    v = merl_videos("test", 1)[0]
    src = FileSource(v, a.fps, 320, max_frames=1800)
    frames = list(src.frames())
    tcfg = TriggerConfig.from_dict(json.loads(a.trigger))
    best = []
    for _ in range(5):
        tr = ZoneTrigger(zones, store.resolution, tcfg)
        t0 = time.process_time()
        for fr in frames:
            tr.update(fr.lores, fr.t)
        best.append((time.process_time() - t0) / len(frames))
    trig_ms = 1000 * min(best)
    t0 = time.process_time()
    sizes = [len(src.encode(fr.main, a.quality)) for fr in frames[:200]]
    jpeg_ms = 1000 * (time.process_time() - t0) / 200
    # one clip frame blown up to the Camera Module 3 sizes: encode time and bytes of a frame that large.
    # Upscaled video is smoother than a real 12 MP frame, so real frames will be larger.
    big = {}
    for name, size in (("2304x1296", (2304, 1296)), ("4608x2592", (4608, 2592))):
        img = cv2.resize(frames[100].main, size, interpolation=cv2.INTER_CUBIC)
        t0 = time.process_time()
        n = [len(src.encode(img, a.quality)) for _ in range(5)]
        big[name] = {"jpeg_ms_here": round(1000 * (time.process_time() - t0) / 5, 1), "jpeg_bytes_upscaled": n[0]}
    lo, hi = PI_SLOWDOWN
    out = {"machine": platform.processor() or platform.machine(), "cpu": subprocess.run(
               ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "opencv_threads": 1, "lowres": list(frames[0].lores.shape[1::-1]), "frames_timed": len(frames),
           "trigger_ms_per_frame_here": round(trig_ms, 3),
           "trigger_core_share_here_at_fps": round(trig_ms * a.fps / 1000, 4), "fps": a.fps,
           "jpeg_ms_per_frame_here_920x680": round(jpeg_ms, 2), "jpeg_bytes_per_frame_920x680": int(np.mean(sizes)),
           "upscaled_frame_encode": big,
           "ESTIMATE_pi_zero_2w": {
               "assumption": f"one Zero 2 W core is {lo}x to {hi}x slower than one core here (assumed, not measured)",
               "trigger_ms_per_frame": [round(trig_ms * lo, 1), round(trig_ms * hi, 1)],
               "trigger_share_of_one_core_at_fps": [round(trig_ms * lo * a.fps / 1000, 3), round(trig_ms * hi * a.fps / 1000, 3)],
               "jpeg_ms_per_2304x1296_frame": [round(big["2304x1296"]["jpeg_ms_here"] * lo), round(big["2304x1296"]["jpeg_ms_here"] * hi)],
               "jpeg_ms_per_4608x2592_frame": [round(big["4608x2592"]["jpeg_ms_here"] * lo), round(big["4608x2592"]["jpeg_ms_here"] * hi)]}}
    return out


# ------------------------------------------------------------------ SYNTHETIC empty scene
def cmd_synthetic(a) -> dict:
    """Nobody in view: a textured shelf drawing with sensor noise, 3000 frames at 10 fps, one shelf zone.
    Three disturbances, each with gain_norm off and on: none, camera shake (the whole image shifted by a
    random -2..+2 full-res pixels each frame = up to 1 low-res pixel), brightness flicker (every frame
    times a random gain in 0.9..1.1). Any active frame here is a false trigger. SYNTHETIC, not a camera."""
    w, h, n, fps = 640, 480, 3000, 10.0
    rng = np.random.default_rng(0)
    base = np.full((h + 8, w + 8, 3), 110, np.float32)
    base[44:204, 204:604] = 150
    for _ in range(400):                                     # products: coloured boxes with hard edges
        x, y = int(rng.integers(204, 590)), int(rng.integers(44, 190))
        base[y:y + int(rng.integers(6, 20)), x:x + int(rng.integers(6, 20))] = rng.integers(30, 230, 3)
    zone = {"shelf": np.array([[200, 40], [600, 40], [600, 200], [200, 200]])}
    cases = {"still": (0, 0.0), "shake_2px": (2, 0.0), "flicker_10pct": (0, 0.1)}
    tcfg = TriggerConfig.from_dict(json.loads(a.trigger))
    out = {"data": "SYNTHETIC empty textured scene 640x480, nobody in view", "frames": n, "fps": fps, "cases": {}}
    variants = {"gain_norm_off": dict(gain_norm=False, shake_px=0), "gain_norm_on": dict(gain_norm=True, shake_px=0),
                "gain_norm_on_shake_px_1": dict(gain_norm=True, shake_px=1)}
    for name, (shake, flick) in cases.items():
        res = {}
        for vname, v in variants.items():
            r = np.random.default_rng(1)
            tr = ZoneTrigger(zone, (w, h), TriggerConfig(**{**vars(tcfg), **v}))
            log = []
            for i in range(n):
                dx, dy = (int(v) for v in r.integers(-shake, shake + 1, 2)) if shake else (0, 0)
                img = base[4 + dy:4 + dy + h, 4 + dx:4 + dx + w] + 2 * r.standard_normal((h, w, 3), dtype=np.float32)
                img = np.clip(img * (1 + r.uniform(-flick, flick)), 0, 255).astype(np.uint8)
                log.append((i / fps, bool(tr.update(cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA), i / fps))))
            res[vname] = {
                "active_frames": sum(on for _, on in log), "runs": len(runs(log, 1 / fps))}
        out["cases"][name] = res
        print(name, res, flush=True)
    return out


def cmd_table(a) -> dict:
    """Markdown tables from the three result files (so the report never holds a hand-typed number)."""
    d = ROOT / "results"
    merl, toy, cpu = (json.loads((d / f"edge_measure_{n}.json").read_text()) for n in ("merl_test", "toy", "cpu"))
    pct = lambda x: f"{100 * x:.1f} %"
    mb = lambda x: f"{x / 1e6:.1f} MB"
    rows = [("clips, length", lambda p: f"{p['clips']}, {p['hours'] * 60:.1f} min"),
            ("reaches (MERL) / takes and put-backs (toy)", lambda p: str(p["picks"])),
            ("trigger recall", lambda p: f"{pct(p['trigger_recall'])} ({p['picks'] - p['missed']} of {p['picks']})"),
            ("fully inside a full-res burst", lambda p: pct(p["captured_recall"])),
            ("triggers", lambda p: str(p["triggers"])),
            ("false triggers", lambda p: f"{p['false_triggers']} = {p['false_triggers_per_hour']:.0f} per hour"),
            ("always-on control (a trigger active on every frame)", lambda p: f"{pct(p['always_on_recall'])} recall, {p['always_on_false_triggers']} false triggers, 0 % saved"),
            ("frames within 0.5 s of labelled hand activity", lambda p: pct(p["label_share_of_frames"])),
            ("active frames outside labelled hand activity", lambda p: pct(p["active_outside_label_share"])),
            ("reach frames with the trigger active", lambda p: pct(p["reach_frames_active_share"])),
            ("share of time the trigger is active", lambda p: pct(p["trigger_active_share"])),
            ("share of frames sent at full resolution", lambda p: pct(p["burst_share_of_frames"])),
            ("full-res JPEG stream, every frame (baseline)", lambda p: f"{mb(p['baseline_jpeg_bytes'])} = {p['baseline_jpeg_mbit_s']:.2f} Mbit/s"),
            ("bursts only", lambda p: f"{mb(p['burst_bytes'])} = {p['bursts_mbit_s']:.2f} Mbit/s"),
            ("bandwidth saved, bursts only", lambda p: pct(p["saved_bursts_only"])),
            ("low-res stream as well (every frame)", lambda p: f"{mb(p['lowres_bytes'])} = {p['lowres_mbit_s']:.2f} Mbit/s"),
            ("bandwidth saved, bursts + low-res stream", lambda p: pct(p["saved_bursts_plus_lowres"])),
            ("same clips as continuous H.264 (libx264 crf 23)", lambda p: f"{mb(p['baseline_h264_bytes'])} = {p['baseline_h264_mbit_s']:.2f} Mbit/s"),
            ("JPEG bursts against that H.264 stream", lambda p: f"{p['burst_bytes'] / p['baseline_h264_bytes']:.1f} times larger")]
    out = [f"| | MERL test split (real lab video, {merl['fps']:g} fps) | toy clips (SYNTHETIC, {toy['fps']:g} fps) |", "|---|---|---|"]
    out += [f"| {name} | {f(merl['pooled'])} | {f(toy['pooled'])} |" for name, f in rows]
    p = merl["pooled"]
    out += ["", "Projection, not a measurement: MERL rates scaled by the share of time someone is at the shelf "
            "(MERL itself is 100 %: a shopper is in front of the shelf for the whole of every clip).", "",
            "| someone at the shelf | bursts, Mbit/s | saved against the full-res JPEG stream |", "|---|---|---|"]
    out += [f"| {int(100 * q)} % of the time | {p['bursts_mbit_s'] * q:.2f} | {pct(1 - q * p['burst_bytes'] / p['baseline_jpeg_bytes'])} |"
            for q in (1.0, 0.5, 0.2, 0.05)]
    e = cpu["ESTIMATE_pi_zero_2w"]
    big = cpu["upscaled_frame_encode"]
    out += ["", f"| CPU, one core, OpenCV on 1 thread | measured here ({cpu['cpu']}) | ESTIMATE for a Pi Zero 2 W ({e['assumption']}) |", "|---|---|---|",
            f"| trigger, per {cpu['lowres'][0]}x{cpu['lowres'][1]} frame | {cpu['trigger_ms_per_frame_here']:.2f} ms | {e['trigger_ms_per_frame'][0]} to {e['trigger_ms_per_frame'][1]} ms |",
            f"| trigger at {cpu['fps']:.0f} fps, share of one core | {pct(cpu['trigger_core_share_here_at_fps'])} | {pct(e['trigger_share_of_one_core_at_fps'][0])} to {pct(e['trigger_share_of_one_core_at_fps'][1])} |",
            f"| JPEG encode, 920x680 clip frame | {cpu['jpeg_ms_per_frame_here_920x680']:.1f} ms | not estimated |",
            f"| JPEG encode, clip frame upscaled to 2304x1296 | {big['2304x1296']['jpeg_ms_here']} ms | {e['jpeg_ms_per_2304x1296_frame'][0]} to {e['jpeg_ms_per_2304x1296_frame'][1]} ms |",
            f"| JPEG encode, clip frame upscaled to 4608x2592 | {big['4608x2592']['jpeg_ms_here']} ms | {e['jpeg_ms_per_4608x2592_frame'][0]} to {e['jpeg_ms_per_4608x2592_frame'][1]} ms |"]
    syn = d / "edge_measure_synthetic.json"
    if syn.exists():
        s = json.loads(syn.read_text())
        out += ["", f"| SYNTHETIC empty scene, nobody in view, {s['frames']} frames at {s['fps']:g} fps: frames with the trigger active | gain_norm off (first version) | gain_norm on (default) | gain_norm on, shake_px 1 (option) |", "|---|---|---|---|"]
        out += ["| " + " | ".join([k] + [f"{v[c]['active_frames']} ({v[c]['runs']} runs)" for c in ("gain_norm_off", "gain_norm_on", "gain_norm_on_shake_px_1")]) + " |"
                for k, v in s["cases"].items()]
    print("\n".join(out))
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["merl", "toy", "cpu", "synthetic", "table"])
    ap.add_argument("--split", default="test", choices=["test", "train", "val"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--pre-roll", type=float, default=2.0)
    ap.add_argument("--post-roll", type=float, default=1.5)
    ap.add_argument("--max-burst", type=float, default=6.0)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--trigger", default="{}", help="TriggerConfig overrides as JSON")
    ap.add_argument("--no-h264", dest="h264", action="store_false")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args()
    res = {"merl": cmd_merl, "toy": cmd_toy, "cpu": cmd_cpu, "synthetic": cmd_synthetic, "table": cmd_table}[a.cmd](a)
    if a.cmd == "table":
        return
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("clips", "sweep")}, indent=2, default=float))


if __name__ == "__main__":
    main()
