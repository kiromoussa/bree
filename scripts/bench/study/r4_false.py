"""False PICK and PUT_BACK events on dev2 by what really happened near them (truth for scoring only).
usage: r4_false.py [whatif name] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
run = ROOT / "out/bench/whatif" / (args[0] if args else "r4_base") / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
TOL = 3.0
tab = {"pick": Counter(), "put_back": Counter()}
rows = []
for d in sorted(run.glob("clip_*")):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    fx = {s["id"]: s["fixtureId"] for s in json.loads((clip / "layout.json").read_text())["slots"]}
    ev = J(d / "pipeline/events.jsonl")
    slot = lambda e: (e.get("meta") or {}).get("slot") or {}
    at = lambda p, f: p.get("zone") == f or slot(p).get("fixture") == f
    things = ([("take", g["t"], g["fixtureId"], g["shopper"], g["slotId"]) for g in truth] + [("staff take", a["t"], a["fixtureId"], a["shopper"], a["slotId"]) for a in acts if a["kind"] == "staff_take"]
              + [("put-back" + (" into another slot" if g.get("putBackSlot") else ""), g["tPutBack"], fx.get(g.get("putBackSlot") or g["slotId"]), g["shopper"], g.get("putBackSlot") or g["slotId"]) for g in truth if g.get("tPutBack") is not None]
              + [("staff put", a["t"], a["fixtureId"], a["shopper"], a["slotId"]) for a in acts if a["kind"] == "staff_put"] + [("touch", a["t"], a["fixtureId"], a["shopper"], a["slotId"]) for a in acts if a["kind"] == "touch"])
    for typ, mine in (("pick", ("take", "staff take")), ("put_back", ("put-back", "put-back into another slot", "staff put"))):
        P = [e for e in ev if e["type"] == typ]
        T = [x for x in things if x[0] in mine]
        used, got = set(), set()
        for _, _, i, j in sorted((not at(p, g[2]), abs(p["t"] - g[1]), i, j) for i, p in enumerate(P) for j, g in enumerate(T) if abs(p["t"] - g[1]) <= TOL):
            if i not in used and j not in got:
                used.add(i); got.add(j)
        for i, p in enumerate(P):
            if i in used:
                continue
            near = sorted((abs(p["t"] - x[1]), x) for x in things if abs(p["t"] - x[1]) <= TOL and at(p, x[2]))
            why = "nothing happened at that fixture within 3 s" if not near else ("a second event at a true " if near[0][1][0] in mine else "at a ") + near[0][1][0]
            tab[typ][why] += 1
            rows.append((d.name, typ, p["t"], p["person_id"], p.get("sku"), slot(p).get("id"), ((p.get("meta") or {}).get("shelf") or {}).get("camera_id"), ((p.get("meta") or {}).get("shelf") or {}).get("source"), why, [(round(a, 1), x[0], x[3], x[4]) for a, x in near[:3]]))
for typ, c in tab.items():
    print(typ, sum(c.values()))
    for k, n in c.most_common():
        print("  ", n, k)
if "-v" in sys.argv:
    for r in rows:
        print(*r)
