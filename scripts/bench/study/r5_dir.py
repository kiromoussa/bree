"""Take readings of the item cameras on dev2: was the product seen in a hand before the shelf change, after it, and
did it come towards the slot or go away (from the stored detector looks). Labelled with truth for scoring only:
a true take or a true put (put-back, staff put) at that place. usage: r5_dir.py [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from bree.sim.bench import load_calibration  # noqa: E402
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
rows = []
for d in sorted((ROOT / "out/bench/dev2").glob("clip_?????")):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    lay = json.loads((clip / "layout.json").read_text())
    slots = {s["id"]: s for s in lay["slots"]}
    fps = float(json.loads((clip / "clip.json").read_text())["fps"])
    cams = load_calibration(clip)
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    things = ([("take", g["t"], g["slotId"]) for g in truth] + [("take", a["t"], a["slotId"]) for a in acts if a["kind"] == "staff_take"]
              + [("put", g["tPutBack"], g.get("putBackSlot") or g["slotId"]) for g in truth if g.get("tPutBack") is not None] + [("put", a["t"], a["slotId"]) for a in acts if a["kind"] == "staff_put"])
    looks = {}
    for r in J(d / "pipeline/shelf_events.jsonl"):
        if r["kind"] != "take" or r.get("slot_id") not in slots:
            continue
        cam = r["camera_id"]
        if cam not in looks:
            p = d / "pipeline/conceal" / f"looks_{cam}.jsonl"
            looks[cam] = J(p) if p.exists() else []
        face = np.asarray(slots[r["slot_id"]]["face"], float)
        px = cams[cam].project(face[None])[0][0]
        near = sorted((abs(np.clip(x[1], r["t_start"], r["t_end"]) - x[1]), np.linalg.norm(np.asarray(slots[x[2]]["face"], float) - face), x[0], x[1]) for x in things
                      if r["t_start"] - 3 <= x[1] <= r["t_end"] + 3 and np.linalg.norm(np.asarray(slots[x[2]]["face"], float) - face) <= 0.35)
        label = near[0][2] if near else "nothing"
        if len({n[2] for n in near}) > 1:
            label = "both"
        obs = [(L["f"] / fps, float(np.hypot((i[0] + i[2]) / 2 - px[0], (i[1] + i[3]) / 2 - px[1]))) for L in looks[cam] if r["t_start"] - 3 <= L["f"] / fps <= r["t_end"] + 3
               for i in L["items"] if i[5] == r["sku_id"] and i[6] >= 0.4 and i[7] and not i[8]]
        obs = [o for o in obs if o[1] <= 600]
        pre, mid, post = (sum(a <= t < b for t, _ in obs) for a, b in ((-1e9, r["t_start"]), (r["t_start"], r["t_end"] + 1e-6), (r["t_end"] + 1e-6, 1e9)))
        way = "none" if len(obs) < 2 else "towards" if obs[-1][1] < 0.5 * obs[0][1] else "away" if obs[0][1] < 0.5 * obs[-1][1] else "neither"
        rows.append((d.name, cam, r["t_start"], r["t"], r["t_end"], r["slot_id"], r["sku_id"], r.get("source"), label, pre, mid, post, way))
tab = Counter()
for r in rows:
    order = "none seen" if r[9] + r[10] + r[11] == 0 else "before only" if r[11] == 0 and r[9] > 0 else "after (too)" if r[11] > 0 else "during only"
    tab[(r[8], order, r[12])] += 1
for k, n in sorted(tab.items()):
    print(n, *k)
if "-v" in sys.argv:
    for r in rows:
        if r[8] in ("put", "both"):
            print(*r)
