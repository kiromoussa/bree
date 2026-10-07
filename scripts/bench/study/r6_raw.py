"""Every raw take reading of the item cameras on dev2 (shelf_events.jsonl, before any fusing), by what truly happened
at that place and time (truth for scoring only), against what the reading itself knows: was it taken back by a put of
the same camera, how many other takes that camera read in the same second, was a hand seen on the patch when it was
read. usage: r6_raw.py [run folder under out/bench, default dev2] [-v]"""
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
run = ROOT / "out/bench" / (args[0] if args else "dev2")
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
tab = defaultdict(Counter)
for d in sorted(p for p in run.glob("clip_*") if (p / "run.json").exists()):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    slots = {s["id"]: s for s in json.loads((clip / "layout.json").read_text())["slots"]}
    raw = J(d / "pipeline/shelf_events.jsonl")
    truth = J(clip / "truth/events.jsonl")
    acts = J(clip / "truth/acts.jsonl")
    undone = {}
    for r in raw:
        if r["kind"] == "put":
            for x in r.get("undoes") or []:
                undone.setdefault(x, r["t_end"])
    looks = {}
    def hands(cam, f0, f1, u, v, px=150):
        if cam not in looks:
            p = d / "pipeline/conceal" / f"looks_{cam}.jsonl"
            looks[cam] = {r["f"]: r["hands"] for r in J(p)} if p.exists() else {}
        return sum(1 for f in range(f0, f1 + 1) for h in looks[cam].get(f, []) if math.hypot((h[0] + h[2]) / 2 - u, (h[1] + h[3]) / 2 - v) <= px)
    for r in raw:
        if r["kind"] != "take" or r.get("cue") == "slot_state":
            continue
        p = r["point_3d"]
        near = lambda sid: sid in slots and math.dist(slots[sid]["face"], p) <= 0.25
        what = "nothing"
        if any(near(g["slotId"]) and r["t_start"] - 1.5 <= g["t"] <= r["t_end"] + 1.5 for g in truth) or any(a["kind"] == "staff_take" and near(a["slotId"]) and r["t_start"] - 1.5 <= a["t"] <= r["t_end"] + 1.5 for a in acts):
            what = "take"
        elif any(g.get("tPutBack") is not None and near(g.get("putBackSlot") or g["slotId"]) and r["t_start"] - 1.5 <= g["tPutBack"] <= r["t_end"] + 1.5 for g in truth) or any(
                (a["kind"] == "staff_put" and near(a["slotId"]) and r["t_start"] - 1.5 <= a["t"] <= r["t_end"] + 1.5) or (a["kind"] == "staff_take" and a.get("putSlot") and near(a["putSlot"]) and r["t_start"] - 1.5 <= a["tPut"] <= r["t_end"] + 1.5) for a in acts):
            what = "put"
        elif any(a["kind"] == "touch" and near(a["slotId"]) and r["t_start"] - 1.5 <= a["t"] <= r["t_end"] + 1.5 for a in acts):
            what = "touch"
        back = [undone[x] - r["t_end"] for x in r.get("eids") or [] if x in undone]
        f = r["evidence"]["frames"][-1]
        burst = sum(1 for o in raw if o is not r and o["kind"] == "take" and o["camera_id"] == r["camera_id"] and abs(o["t_end"] - r["t_end"]) <= 1.0 and o.get("cue") != "slot_state")
        h = hands(r["camera_id"], f - 6, f, *r["hand_px"]) if r.get("source") == "shelf_diff" else -1
        feats = {
            "taken back": "never" if not back else "within 3 s" if min(back) <= 3 else "later",
            "others in that second": "0" if burst == 0 else "1" if burst == 1 else "2+",
            "source": r["source"] + ("/" + r["cue"] if r.get("cue") else ""),
            "item frames": "0" if not r.get("detector_frames") else "1" if r["detector_frames"] == 1 else "2-3" if r["detector_frames"] <= 3 else "4+",
            "span": "under 1 s" if r["t_end"] - r["t_start"] < 1 else "1 to 3 s" if r["t_end"] - r["t_start"] < 3 else "over 3 s",
            "hand on patch when read (pixel-only)": "n/a" if h < 0 else "yes" if h else "no",
            "kept and alone": ("never back" if not back else "back") + (", alone" if burst == 0 else ", not alone"),
        }
        for k, v in feats.items():
            tab[k][(v, what)] += 1
        tab["all"][("", what)] += 1
        if "-v" in sys.argv:
            print(d.name, r["camera_id"], r["t_start"], r["t_end"], r["slot_id"], what, feats)
for k, c in tab.items():
    print(k)
    for v in sorted({v for v, _ in c}):
        print(f"   {v:22s}", {w: c[(v, w)] for w in ("take", "put", "touch", "nothing")})
