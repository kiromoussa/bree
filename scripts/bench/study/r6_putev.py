"""Every PUT_BACK event the pipeline made on dev2, true or not (a true put-back or staff put at that fixture within
3 s; truth for scoring only), against what the put knows about itself. usage: r6_putev.py [whatif name] [-v]"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
run = ROOT / "out/bench/whatif" / (args[0] if args else "r6base") / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
tab = defaultdict(Counter)
for d in sorted(p for p in run.glob("clip_*") if p.is_dir()):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    slots = {s["id"]: s for s in json.loads((clip / "layout.json").read_text())["slots"]}
    fx = lambda sid: (slots.get(sid) or {}).get("fixtureId")
    true = [(g["tPutBack"], fx(g.get("putBackSlot") or g["slotId"])) for g in J(clip / "truth/events.jsonl") if g.get("tPutBack") is not None]
    true += [(a["t"], a["fixtureId"]) for a in J(clip / "truth/acts.jsonl") if a["kind"] == "staff_put"]
    acts = J(d / "pipeline/store_shelf_events.jsonl")
    for e in J(d / "pipeline/events.jsonl"):
        if e["type"] != "put_back":
            continue
        sid = (e["meta"].get("slot") or {}).get("id")
        ok = "true" if any(abs(t - e["t"]) <= 3 and f == (fx(sid) or e.get("zone")) for t, f in true) else "false"
        a = next((a for a in acts if a["kind"] == "put" and abs(a["t"] - e["t"]) < 0.06 and a["slot_id"] == sid), {})
        m = re.search(r"undoes the take at ([0-9.]+)s", a.get("why") or "")
        gap = e["t"] - float(m.group(1)) if m else None
        take = next((k for k in acts if m and k["kind"] == "take" and abs(k["t"] - float(m.group(1))) < 0.06), {})
        feats = {
            "how": "undoes a take given to somebody else, or a second take" if "undoes" in e["meta"] else "arrived" if a.get("arrived") else "undoes a take" if m else "other",
            "gap to the take": "n/a" if gap is None else "under 2 s" if gap < 2 else "2 to 3 s" if gap < 3 else "3 to 5 s" if gap < 5 else "over 5 s",
            "item seen going in": str(bool(a.get("item_in"))),
            "put source": str(a.get("source")),
            "take: item frames": "n/a" if not take else "0" if not take.get("detector_frames") else "1" if take["detector_frames"] == 1 else "2+",
            "take: cameras": "n/a" if not take else str(min(len(take.get("cameras") or []), 2)),
        }
        for k, v in feats.items():
            tab[k][(v, ok)] += 1
        tab["all"][("", ok)] += 1
        if "-v" in sys.argv:
            print(d.name, e["t"], sid, e["person_id"], ok, feats)
for k, c in tab.items():
    print(k)
    for v in sorted({v for v, _ in c}):
        print(f"   {v:18s}", {w: c[(v, w)] for w in ("true", "false")})
