"""Does the product seen in a hand name the taken product where the slot does not? (dev2, truth for scoring only)
For every true pick with a shelf event: the products the item cameras saw in a hand (stored detector looks: changed,
moving, not shelf stock) from 1 s before to 4 s after the event, on cameras of the event's fixture, against the true
product and the product of the event's slot. usage: r7_heldsku.py [run folder under out/bench, default dev2] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
name = next((a for a in sys.argv[1:] if not a.startswith("-")), "dev2")
run, tab = ROOT / "out/bench" / name, Counter()
for c in json.loads((run / "bench.json").read_text())["clips"]:
    d = run / f"clip_{c['clip']}"
    fps = float(json.loads((ROOT / "data/synth/bench/dev2" / d.name / "clip.json").read_text())["fps"])
    held = []
    for p in sorted((d / "pipeline/conceal").glob("looks_*.jsonl")):
        cam = p.stem[6:]
        for look in J(p):
            held += [(look["f"] / fps, it[5], cam, it[4]) for it in look["items"] if it[6] >= 0.4 and it[7] and not it[8]]
    for r in c["picks"]:
        if not r["stages"]["shelf_event_emitted"]:
            continue
        det = r["detail"]
        fx = (det.get("event_slot") or "").split("-")[0]
        n = Counter(s for t, s, cam, cf in held if det["event_t"] - 1 <= t <= det["event_t"] + 4 and cam.split("-")[0] == fx)
        top = n.most_common(2)
        lead = "nothing seen" if not top else "tie" if len(top) > 1 and top[0][1] == top[1][1] else "true product leads" if top[0][0] == r["sku"] else "slot's product leads" if top[0][0] == det["event_sku"] else "a third product leads"
        strong = "" if not top or top[0][1] < 3 or (len(top) > 1 and top[0][1] < 2 * top[1][1]) else ", 3+ and twice the next"
        tab[("event names the true product" if r["stages"]["right_sku"] else "event names another product", lead + strong)] += 1
        if "-v" in sys.argv and not r["stages"]["right_sku"]:
            print(c["clip"], r["shopper"], r["t"], r["sku"], "event", det["event_sku"], det["event_slot"], "true slot", r["slot"], top, n.get(r["sku"], 0))
for k in sorted(tab):
    print(tab[k], *k)
