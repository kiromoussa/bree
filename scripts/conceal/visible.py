"""With truth (SIMULATED, TRAIN-range clips): could a rendered item camera see each concealment at all?
Per concealment: the rendered item cameras among the simulator's `concealSeenBy`, and in how many frames between the
pick and the hide a rendered item camera showed the item in the hand at MINPX visible pixels or more.
  python scripts/conceal/visible.py data/synth/bench/train 4900 4901 ..."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bree.shelf.events import ITEM_KINDS
MINPX, HIDE_BEFORE_S = 400, 0.49
n = seen_by_any = held_any = held_5 = 0
for seed in sys.argv[2:]:
    clip = Path(sys.argv[1]) / f"clip_{seed}"
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    item_cams = {c for c in json.loads((clip / "clip.json").read_text())["cameras"] if kind[c] in ITEM_KINDS}
    frames = [json.loads(x) for x in (clip / "truth/frames.jsonl").read_text().splitlines()]
    for e in (json.loads(x) for x in (clip / "truth/events.jsonl").read_text().splitlines()):
        if e.get("tConceal") is None:
            continue
        hide = e["tConceal"] - HIDE_BEFORE_S
        saw = sorted(set(e.get("concealSeenBy") or []) & item_cams)
        held = {(r["camera"], r["frame"]) for r in frames if r["camera"] in item_cams and e["t"] < r["t"] <= hide
                for it in r["items"] if it["kind"] == "hand" and it.get("shopper") == e["shopper"] and it["vis_px"] >= MINPX}
        n += 1; seen_by_any += bool(saw); held_any += bool(held); held_5 += len({f for _, f in held}) >= 5
        print(seed, e["shopper"], e["skuId"], f"pick {e['t']} hide {hide:.2f}; item cameras that saw the hide: {saw or 'none'}; frames with the item in the hand on an item camera: {len({f for _, f in held})}")
print(f"{n} concealments; hide seen by a rendered item camera: {seen_by_any}; item in the hand on an item camera in at least 1 frame: {held_any}, in at least 5 frames: {held_5}")
