"""Shelf-state events (bree.shelfstate `events.jsonl`) -> `shelf` signals.

The shelf track writes one change per line ({"camera_id","t","kind","region_id","conf","wall","detail",...}).
Each becomes one `shelf` signal: zone = region_id, meta.state from KIND_STATE, value = 1 for a good state and
0.5 / 0 for low / empty (a rough fullness, not a measured one). Evidence image paths are NOT copied (the log
holds no images). Nothing here reads video.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from bree.product.signals import Signal

# shelfstate event kind -> (signals.SHELF_STATES entry, fullness 0..1 or None when unknown)
KIND_STATE = {
    "stock_down": ("low", 0.5), "low_stock": ("low", 0.4), "empty_facing": ("empty", 0.0),
    "restock": ("ok", 1.0), "floor_cleared": ("ok", 1.0),
    "door_ajar": ("door_open", None), "door_change": ("door_open", None),
    "floor_spill": ("dirty", None), "floor_object": ("misplaced", None), "layout_change": ("misplaced", None),
}


def adapt_shelfstate(events_jsonl: str | Path, store: str, camera: str | None = None, source: str = "real_unlabelled",
                     tz=None, clock_offset_s: float = 0.0, min_conf: float = 0.0) -> tuple[list[Signal], dict]:
    """`wall` (naive ISO local time, written by the shelf runner) is read in `tz` (default: this machine's zone)."""
    tz = tz or dt.datetime.now().astimezone().tzinfo
    out, skipped = [], {"no_wall": 0, "unknown_kind": 0, "low_conf": 0, "bad_line": 0}
    for line in Path(events_jsonl).read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            skipped["bad_line"] += 1
            continue
        if e.get("kind") not in KIND_STATE:
            skipped["unknown_kind"] += 1
            continue
        if not e.get("wall"):
            skipped["no_wall"] += 1
            continue
        if float(e.get("conf", 1.0)) < min_conf:
            skipped["low_conf"] += 1
            continue
        state, full = KIND_STATE[e["kind"]]
        ts = dt.datetime.fromisoformat(e["wall"]).replace(tzinfo=tz).timestamp() + clock_offset_s
        # door/spill/object have no fullness; 0.0 would read as "empty", so use the neutral 1.0 and let meta.state speak
        out.append(Signal(ts=ts, kind="shelf", value=1.0 if full is None else full, store=store,
                          camera=camera or e.get("camera_id", "store"), zone=e.get("region_id") or "whole",
                          source=source, meta={"state": state, "event": e["kind"], "conf": e.get("conf")}).validate())
    return out, {"signals": len(out), "skipped": skipped}
