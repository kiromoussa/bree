"""Hand-only takes on dev2 with what each one knows about itself, to design a gate (truth for scoring only).
Held-item tracks are rebuilt from the stored detector looks as bree.shelf.hand.HandItemCue builds them. Each take
candidate is written with: sightings, cameras, how close a detected hand came to the slot in the 1.5 s before the item
was first seen (in slot reaches), how far from the slot the item was first seen, sightings of the same product in any
hand in the 8 s before (already carried), and whether the pixel comparison read a take or a put on that fixture then.
usage: r7_handgate.py [run folder under out/bench, default dev2]  -> out/bench/r7/handgate_<run>.json"""
import json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from bree.shelf.diff import ShelfDiff
from bree.shelf.events import skus_of
from bree.shelf.hand import HandConfig, HandItemCue
from bree.sim.bench import load_calibration
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def one(d: Path) -> list[dict]:
    clip = ROOT / "data/synth/bench/dev2" / d.name
    layout = json.loads((clip / "layout.json").read_text())
    slots = {s["id"]: s for s in layout["slots"]}
    cams = load_calibration(clip)
    fps = float(json.loads((clip / "clip.json").read_text())["fps"])
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    raw = J(d / "pipeline/shelf_events.jsonl")
    cand, held = [], []          # held: (t, sku, camera) of every item seen in a hand
    for p in sorted((d / "pipeline/conceal").glob("looks_*.jsonl")):
        cam = p.stem[6:]
        sd = ShelfDiff(cam, cams[cam], layout["slots"], fps, None, skus_of(layout), layout.get("fixtures"))
        cue = HandItemCue(sd, None, None, HandConfig())
        hands = {}
        for look in J(p):
            hands[look["f"]] = [((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in look["hands"]]
            for x0, y0, x1, y1, cf, sku, share, moved, stock in look["items"]:
                if share >= cue.cfg.fg_frac and moved and not stock:
                    cue._add(look["f"], sku, (x0 + x1) / 2, (y0 + y1) / 2, cf)
                    held.append((look["f"] / fps, sku, cam))
        for tr in cue.tracks:
            got = cue._slot(tr)
            if got is None or got[0] != "take":
                continue
            ev = cue.as_event(tr)
            b = sd.boxes[got[1]] / sd.cfg.scale
            c, reach = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), max(cue.cfg.near_slot * math.hypot(b[2] - b[0], b[3] - b[1]), cue.cfg.near_slot_min_px)
            f0 = tr["obs"][0][0]
            hd = [math.hypot(u - c[0], v - c[1]) / reach for f in range(f0 - int(1.5 * fps), f0 + 1) for u, v in hands.get(f, [])]
            hin = [1 for f in range(f0 - int(1.5 * fps), f0 + 1) for u, v in hands.get(f, []) if b[0] <= u <= b[2] and b[1] <= v <= b[3]]
            pts = np.array([o[1:3] for o in tr["obs"]])
            votes = {}          # slots of this product a hand was in just before
            for j in cue._by_sku[tr["sku"]]:
                bj = sd.boxes[j] / sd.cfg.scale
                n = sum(1 for f in range(f0 - int(1.5 * fps), f0 + 1) for u, v in hands.get(f, []) if bj[0] <= u <= bj[2] and bj[1] <= v <= bj[3])
                if n:
                    votes[sd.slots[j]["id"]] = n
            ev["hand_slots"] = votes
            ev.update(hand_d=round(min(hd), 3) if hd else None, hand_in=len(hin), first_d=round(math.hypot(pts[0][0] - c[0], pts[0][1] - c[1]) / reach, 3),
                      travel=round(float(np.linalg.norm(pts[-1] - pts[0]) / reach), 3), span=round((tr["obs"][-1][0] - f0) / fps, 2))
            cand.append(ev)
    out = []
    for e in sorted(cand, key=lambda e: e["t"]):
        g = next((g for g in out if g["sku_id"] == e["sku_id"] and abs(g["t"] - e["t"]) <= 3), None)
        if g is None:
            out.append({**e, "cams": [e["camera_id"]]})
        else:
            g["cams"] = sorted({*g["cams"], e["camera_id"]})
            for k, f in (("detector_frames", max), ("hand_in", max), ("travel", max), ("span", max), ("sku_conf", max)):
                g[k] = f(g[k], e[k])
            g["hand_slots"] = {k: g["hand_slots"].get(k, 0) + e["hand_slots"].get(k, 0) for k in {*g["hand_slots"], *e["hand_slots"]}}
            for k in ("hand_d", "first_d"):
                g[k] = min((x for x in (g[k], e[k]) if x is not None), default=None)
    takes = [(g["t"], g["skuId"], g["slotId"]) for g in truth] + [(a["t"], a["skuId"], a["slotId"]) for a in acts if a["kind"] == "staff_take"]
    for g in out:
        fx = slots[g["slot_id"]]["fixtureId"]
        g["truth"] = any(r[1] == g["sku_id"] and abs(r[0] - g["t"]) <= 3 and slots[r[2]]["fixtureId"] == fx for r in takes)
        g["right_slot"] = any(r[1] == g["sku_id"] and abs(r[0] - g["t"]) <= 3 and r[2] == g["slot_id"] for r in takes)
        g["true_slots"] = [r[2] for r in takes if r[1] == g["sku_id"] and abs(r[0] - g["t"]) <= 3 and slots[r[2]]["fixtureId"] == fx]
        g["other_take"] = [(r[1], r[2]) for r in takes if abs(r[0] - g["t"]) <= 3 and r[2].split("-")[0] == g["slot_id"].split("-")[0]]
        g["carried"] = sum(1 for t, s, _ in held if s == g["sku_id"] and g["t"] - 8 <= t <= g["t"] - 1.5)
        near = [r for r in raw if r.get("point_3d") and slots[r["slot_id"]]["fixtureId"] == fx and r["t_start"] - 3 <= g["t"] <= r["t_end"] + 3
                and math.dist(r["point_3d"][::2], g["point_3d"][::2]) <= 1.0]
        g["px_take"] = sum(r["kind"] == "take" for r in near)
        g["px_take_sku"] = sum(r["kind"] == "take" and r["sku_id"] == g["sku_id"] for r in near)
        g["px_put"] = sum(r["kind"] == "put" for r in near)
        g["clip"], g["fixture"] = d.name, fx
    return out


if __name__ == "__main__":
    name = next((a for a in sys.argv[1:] if not a.startswith("-")), "dev2")
    dirs = sorted(p for p in (ROOT / "out/bench" / name).glob("clip_*") if (p / "run.json").exists())
    with ProcessPoolExecutor(3) as ex:
        rows = [g for r in ex.map(one, dirs) for g in r]
    (ROOT / "out/bench/r7").mkdir(exist_ok=True)
    (ROOT / f"out/bench/r7/handgate_{name}.json").write_text(json.dumps(rows))
    print(len(rows), "take candidates,", sum(g["truth"] for g in rows), "at a true take")
