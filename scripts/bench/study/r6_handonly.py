"""What hand-only events would add on dev2 (truth for scoring only): held-item tracks are rebuilt from the stored
detector looks exactly as bree.shelf.hand.HandItemCue builds them, and each track that starts at a slot of its product
and moves away (a take) or ends at one (a put) is compared with the true takes and puts.
usage: r6_handonly.py [min sightings, default 3] [-v]"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from bree.shelf.diff import ShelfDiff
from bree.shelf.events import skus_of
from bree.shelf.hand import HandConfig, HandItemCue
from bree.sim.bench import load_calibration
J = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
nmin = int(next((a for a in sys.argv[1:] if a.isdigit()), 3))
run, tab = ROOT / "out/bench/dev2", Counter()
bench = {c["clip"]: c for c in json.loads((ROOT / "out/bench/whatif/r6base/dev2/bench.json").read_text())["clips"]}
for d in sorted(p for p in run.glob("clip_*") if (p / "run.json").exists()):
    clip = ROOT / "data/synth/bench/dev2" / d.name
    layout = json.loads((clip / "layout.json").read_text())
    slots = {s["id"]: s for s in layout["slots"]}
    cams = load_calibration(clip)
    truth, acts = J(clip / "truth/events.jsonl"), J(clip / "truth/acts.jsonl")
    cand = []
    for p in sorted((d / "pipeline/conceal").glob("looks_*.jsonl")):
        cam = p.stem[6:]
        sd = ShelfDiff(cam, cams[cam], layout["slots"], 10.0, None, skus_of(layout), layout.get("fixtures"))
        cue = HandItemCue(sd, None, None, HandConfig(min_obs=nmin))
        for look in J(p):
            for x0, y0, x1, y1, cf, sku, share, moved, stock in look["items"]:
                if share >= cue.cfg.fg_frac and moved and not stock:
                    cue._add(look["f"], sku, (x0 + x1) / 2, (y0 + y1) / 2, cf)
        for tr in cue.tracks:
            ev = cue.as_event(tr)
            if ev:
                cand.append(ev)
    # one act seen by several cameras: same kind and product within 3 s
    acts1 = []
    for e in sorted(cand, key=lambda e: e["t"]):
        g = next((g for g in acts1 if g["kind"] == e["kind"] and g["sku_id"] == e["sku_id"] and abs(g["t"] - e["t"]) <= 3), None)
        if g is None:
            acts1.append({**e, "cams": 1})
        else:
            g["cams"] += 1; g["detector_frames"] = max(g["detector_frames"], e["detector_frames"])
    takes = [(g["t"], g["skuId"], g["slotId"]) for g in truth] + [(a["t"], a["skuId"], a["slotId"]) for a in acts if a["kind"] == "staff_take"]
    puts = [(g["tPutBack"], g["skuId"], g.get("putBackSlot") or g["slotId"]) for g in truth if g.get("tPutBack") is not None] + [(a["t"], a["skuId"], a["slotId"]) for a in acts if a["kind"] == "staff_put"]
    lost = {(p["t"], p["sku"]): p["lost_at"] for p in bench[int(d.name[5:])]["picks"]}
    for g in acts1:
        fx = slots[g["slot_id"]]["fixtureId"]
        hit = lambda rows: next((r for r in rows if r[1] == g["sku_id"] and abs(r[0] - g["t"]) <= 3 and slots[r[2]]["fixtureId"] == fx), None)
        t, p = hit(takes), hit(puts)
        what = ("true take" if t else "at a true put" if p else "nothing") if g["kind"] == "take" else ("true put" if p else "at a true take" if t else "nothing")
        n = "3+ sightings" if g["detector_frames"] < 6 else "6+ sightings"
        tab[(g["kind"], n, what)] += 1
        if g["kind"] == "take" and t:
            tab[("true takes reached, by where the pipeline lost them now", str(lost.get((t[0], t[1]), "staff take")), "")] += 1
        if "-v" in sys.argv:
            print(d.name, g["kind"], g["t"], g["sku_id"], g["slot_id"], g["detector_frames"], g["cams"], what)
for k in sorted(tab, key=str):
    print(tab[k], *k)
