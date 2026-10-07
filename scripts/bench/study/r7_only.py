"""The hand-only takes of a dev2 what-if run (truth for scoring only): what truly happened at each one, who it was
given to, and whether that was the person who took it. usage: r7_only.py [whatif name, default r7on] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
name = next((a for a in sys.argv[1:] if not a.startswith("-")), "r7on")
run, tab = ROOT / "out/bench/whatif" / name / "dev2", Counter()
for c in json.loads((run / "bench.json").read_text())["clips"]:
    clip = ROOT / f"data/synth/bench/dev2/clip_{c['clip']}"
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    who = {str(p.get("pipeline_id")): p.get("shopper") for p in c.get("people", [])} if c.get("people") and isinstance(c["people"], list) else {}
    door = lambda s: s.split("-")[0]
    for e in J(run / f"clip_{c['clip']}/pipeline/store_shelf_events.jsonl"):
        if e.get("source") != "hand_item":
            continue
        near = [g for g in truth if abs(g["t"] - e["t"]) <= 3 and g["slotId"][0] == e["slot_id"][0]]
        same = [g for g in near if g["skuId"] == e["sku_id"]]
        put = [g for g in truth if g.get("tPutBack") is not None and abs(g["tPutBack"] - e["t"]) <= 3 and (g.get("putBackSlot") or g["slotId"])[0] == e["slot_id"][0]]
        staff = [a for a in acts if a["kind"].startswith("staff") and abs(a["t"] - e["t"]) <= 3]
        what = ("true take, right product" if same else "true take, another product" if near else "a put-back" if put else "staff" if staff else "nothing")
        tab[(what, "given to somebody" if e.get("person_id") is not None else "nobody fits")] += 1
        if "-v" in sys.argv:
            print(c["clip"], e["t"], e["sku_id"], e["slot_id"], e["detector_frames"], "person", e.get("person_id"), what, [(g["shopper"], g["skuId"], g["slotId"], g["outcome"]) for g in same or near])
for k in sorted(tab):
    print(tab[k], *k)
