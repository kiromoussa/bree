"""Each true put-back on dev2 (truth for scoring only): the PICK and PUT_BACK events the pipeline made around the take
and around the put, and every camera reading near the put. usage: r5_puts.py [whatif name] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
args = [a for a in sys.argv[1:] if not a.startswith("-")]
name = args[0] if args else "r5_base"
run = ROOT / "out/bench/whatif" / name / "dev2"
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
tab = Counter()
for d in sorted(run.glob("clip_*")):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    slots = {s["id"]: s for s in json.loads((clip / "layout.json").read_text())["slots"]}
    ev = [e for e in J(d / "pipeline/events.jsonl") if e["type"] in ("pick", "put_back")]
    raw = J(d / "pipeline/shelf_events.jsonl")
    acts = J(d / "pipeline/store_shelf_events.jsonl")
    fx = lambda e: ((e.get("meta") or {}).get("slot") or {}).get("fixture") or e.get("zone")
    for g in J(clip / "truth/events.jsonl"):
        if g.get("tPutBack") is None:
            continue
        to = g.get("putBackSlot") or g["slotId"]
        tf = slots[to]["fixtureId"]
        take = [e for e in ev if e["type"] == "pick" and abs(e["t"] - g["t"]) <= 3 and fx(e) == g["fixtureId"]]
        put = [e for e in ev if e["type"] == "put_back" and abs(e["t"] - g["tPutBack"]) <= 3 and fx(e) == tf]
        falsepick = [e for e in ev if e["type"] == "pick" and abs(e["t"] - g["tPutBack"]) <= 3 and fx(e) == tf and e not in take]
        near = [r for r in raw if r["t_start"] - 3 <= g["tPutBack"] <= r["t_end"] + 3 and (slots.get(r.get("slot_id")) or {}).get("fixtureId") == tf]
        kind = ("take found" if take else "take not found") + ", " + ("put found" if put else "put not found") + (", a PICK at the put" if falsepick else "")
        tab[(kind, "another slot" if g.get("putBackSlot") else "same slot")] += 1
        if "-v" in sys.argv and (not put or falsepick):
            print(f"\n{d.name} {g['shopper']} {g['skuId']} take {g['t']} {g['slotId']} put {g['tPutBack']} {to}: {kind}")
            for e in take + put + falsepick:
                print("   event", e["type"], e["t"], "person", e["person_id"], e["sku"], (e["meta"].get("slot") or {}).get("id"), (e["meta"].get("shelf") or {}).get("source"))
            for r in near:
                print("   reading", r["camera_id"], r["kind"], r["t_start"], r["t"], r["t_end"], r["slot_id"], r["sku_id"], r.get("source"), r.get("cue"), "det", r.get("detector_frames"), "item_in", r.get("item_in"), "undoes", r.get("undoes"), "after", r.get("after"))
            for a in acts:
                if abs(a["t"] - g["tPutBack"]) <= 3 and (slots.get(a.get("slot_id")) or {}).get("fixtureId") == tf:
                    print("   act", a["kind"], a["t"], a["slot_id"], a["sku_id"], "person", a.get("person_id"), "arrived", a.get("arrived"), (a.get("why") or "")[-160:])
for k, n in sorted(tab.items()):
    print(n, *k)
