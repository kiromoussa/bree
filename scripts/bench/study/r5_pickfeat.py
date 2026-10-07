"""PICK events on dev2, true against false (truth for scoring only), by what the shelf pass recorded about each:
which cues, how many cameras, frames with the item in a hand, what the picture shows afterwards.
usage: r5_pickfeat.py [whatif name] [-v]"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
run = ROOT / "out/bench/whatif" / (args[0] if args else "r5_base") / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
TOL = 3.0
rows = []
for d in sorted(run.glob("clip_*")):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    fx = {s["id"]: s["fixtureId"] for s in json.loads((clip / "layout.json").read_text())["slots"]}
    P = [e for e in J(d / "pipeline/events.jsonl") if e["type"] == "pick"]
    S = {(round(g["t"], 2), g.get("person_id")): g for g in J(d / "pipeline/store_shelf_events.jsonl") if g["kind"] == "take"}
    slot = lambda e: (e.get("meta") or {}).get("slot") or {}
    T = [(g["t"], g["fixtureId"], g["slotId"], g["outcome"]) for g in truth] + [(a["t"], a["fixtureId"], a["slotId"], "staff take") for a in acts if a["kind"] == "staff_take"]
    other = ([("put-back", g["tPutBack"], fx.get(g.get("putBackSlot") or g["slotId"])) for g in truth if g.get("tPutBack") is not None]
             + [(a["kind"], a["t"], a["fixtureId"]) for a in acts if a["kind"] in ("staff_put", "touch", "shift")])
    used, got = {}, set()
    for _, _, i, j in sorted((slot(p).get("fixture") != g[1], abs(p["t"] - g[0]), i, j) for i, p in enumerate(P) for j, g in enumerate(T) if abs(p["t"] - g[0]) <= TOL):
        if i not in used and j not in got:
            used[i] = j; got.add(j)
    for i, p in enumerate(P):
        g = S.get((round(p["t"], 2), p["person_id"])) or {}
        if i in used:
            why = "true"
        else:
            near = sorted((abs(p["t"] - x[1]), x[0]) for x in other if abs(p["t"] - x[1]) <= TOL and slot(p).get("fixture") == x[2])
            rep = any(abs(p["t"] - x[0]) <= TOL and slot(p).get("fixture") == x[1] for x in T)
            why = "false: repeat of a true take" if rep else "false: at a " + near[0][1] if near else "false: nothing near"
        after = g.get("after") or []
        own = any(a[0] == g.get("sku_id") for a in after)
        rows.append({"clip": d.name, "t": p["t"], "why": why, "true": why == "true", "source": g.get("source"), "cue": g.get("cue"), "cams": len(g.get("cameras") or [1]),
                     "det": min(g.get("detector_frames") or 0, 3), "after": "nothing" if not after else "own product still there" if own else "another product",
                     "repeats": min(g.get("repeats", 0), 2), "conf": round(g.get("sku_conf") or 0, 1), "mount": (g.get("camera_id") or "?").split("-", 1)[-1].rsplit("-", 1)[0],
                     "readings": min(len(g.get("eids") or []), 3), "area": g.get("area_px"), "slot_px": g.get("slot_px"), "sku": p.get("sku"), "pid": p["person_id"], "cam": g.get("camera_id"), "amb": bool(p.get("candidates"))})
print(len(rows), "PICK events,", sum(r["true"] for r in rows), "true")
print(Counter(r["why"] for r in rows).most_common())
for k in ("source", "cue", "cams", "det", "after", "repeats", "readings", "conf", "mount", "amb"):
    tab = defaultdict(lambda: [0, 0])
    for r in rows:
        tab[r[k]][not r["true"]] += 1
    print(k, "(true, false):", {str(a): tuple(b) for a, b in sorted(tab.items(), key=lambda x: str(x[0]))})
if "-v" in sys.argv:
    for r in rows:
        if not r["true"]:
            print(r)
combo = defaultdict(lambda: [0, 0])
for r in rows:
    combo[(r["source"], "slot cue" if r["cue"] else "no slot cue", "2+ cams" if r["cams"] >= 2 else "1 cam", "item 2+ frames" if r["det"] >= 2 else "item 0-1 frames", r["after"])][not r["true"]] += 1
print("combinations (true, false):")
for k, v in sorted(combo.items(), key=lambda x: -(x[1][1] / (sum(x[1])))):
    print("  ", tuple(v), *k)
