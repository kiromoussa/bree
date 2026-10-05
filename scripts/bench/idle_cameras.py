"""How many shelf events and PICK events come only from rendered item cameras that see no true pick. Scoring only.

A clip renders the item cameras that see a pick plus 2 that see none (clip.json camera_choice); about 25 more
item cameras of the layout are not rendered at all. This counts what the 2 idle ones produce, as a measured
rate per idle camera minute, so the unrendered ones can be allowed for.

  python scripts/bench/idle_cameras.py out/bench/dev out/bench/test out/bench/train
"""
import json, sys
from collections import Counter
from pathlib import Path

rows = lambda f: [json.loads(l) for l in open(f) if l.strip()] if Path(f).exists() else []
for run in sys.argv[1:]:
    n, minutes = Counter(), 0.0
    for d in sorted(Path(run).glob("clip_*")):
        if not (d / "clip" / "clip.json").exists():
            continue
        c = json.loads((d / "clip" / "clip.json").read_text())
        idle = set(c["camera_choice"]["item_cameras_that_see_no_pick"])
        n["idle_cameras"] += len(idle)
        n["item_cameras_rendered"] += len(idle) + len(c["camera_choice"]["item_cameras_that_see_a_pick"])
        minutes += len(idle) * c["sim_seconds"] / 60
        for e in rows(d / "pipeline" / "shelf_events.jsonl"):
            if e["camera_id"] in idle:
                n["camera_events_" + e["kind"]] += 1
        for e in rows(d / "pipeline" / "store_shelf_events.jsonl"):
            if set(e.get("cameras") or [e["camera_id"]]) <= idle:
                n["store_events_" + e["kind"]] += 1
        for e in rows(d / "pipeline" / "events.jsonl"):
            if e["type"] == "pick" and ((e.get("meta") or {}).get("shelf") or {}).get("camera_id") in idle:
                n["pick_events"] += 1
    ev = n["camera_events_take"] + n["camera_events_put"]
    print(run, dict(n), "idle camera minutes", round(minutes, 1), "camera events per idle camera minute", round(ev / minutes, 3),
          "PICK events per idle camera minute", round(n["pick_events"] / minutes, 3))
