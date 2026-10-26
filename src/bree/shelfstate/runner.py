"""Run the shelf-state monitor over recorded files (or a cache of sampled frames) and write its output.

Output folder: events.jsonl (one event per line, evidence file names inside), state.json (per-region fill snapshots
against the golden picture), summary.json (counts, events per hour, hours covered), evidence/ (people pixelated).
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from bree.shelfstate.frames import PersonDetector, sample_frames
from bree.shelfstate.model import Event, ShelfStateConfig, StateMonitor
from bree.shelfstate.regions import CameraRegions, load_regions, whole_picture


def file_start(path: str | Path, clock_offset_h: float = 0.0) -> datetime | None:
    """Wall time of a recorder file from its name (YYYYmmdd_HHMMSS, the recording box's clock) plus an offset in hours.
    The DVR overlay clock of the Truman Rd store read 2 h behind the box clock: pass clock_offset_h=-2 to match the overlay."""
    m = re.search(r"(\d{8})_(\d{6})", Path(path).stem)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") + timedelta(hours=clock_offset_h) if m else None


def run_files(videos: list[str | Path], camera_id: str, regions: CameraRegions | None, out_dir: str | Path,
              cfg: ShelfStateConfig | None = None, fps: float = 1.0, detector: PersonDetector | None = None,
              cache_dir: str | Path | None = None, clock_offset_h: float = 0.0, verbose: bool = False) -> dict:
    """Videos are played in order as one continuous timeline (a recorder's consecutive segments). Pass segments that are
    not contiguous to separate calls: a gap of minutes would make the old reference stale."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = cfg or ShelfStateConfig()
    cfg.evidence_dir = str(out / "evidence")
    mon = StateMonitor(camera_id, regions or whole_picture(camera_id), cfg)
    events: list[Event] = []
    t_off, t_wall0, t_run = 0.0, None, time.time()
    for v in videos:
        v = Path(v)
        t_wall0 = t_wall0 or file_start(v, clock_offset_h)
        if cache_dir and (Path(cache_dir) / f"{camera_id}_{v.stem}.npz").exists():
            z = np.load(Path(cache_dir) / f"{camera_id}_{v.stem}.npz")
            bx = json.loads((Path(cache_dir) / f"{camera_id}_{v.stem}.boxes.json").read_text())
            stream = ((float(t), f, np.array(b).reshape(-1, 5)) for t, f, b in zip(z["t"], z["frames"], bx))
        else:
            def gen(v=v):
                for t, small, big in sample_frames(v, fps, (1280, 720) if detector else None):
                    yield t, small, (detector(big) if detector else None)
            stream = gen()
        last = 0.0
        for t, frame, persons in stream:
            events += mon.update(t_off + t, frame, persons)
            last = t
        t_off += last + 1.0 / fps
        if verbose:
            print(f"{v.name}: {len(events)} events so far", flush=True)
    hours = t_off / 3600.0
    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    rows = []
    for e in events:
        d = e.to_dict()
        if t_wall0:
            d["wall"] = (t_wall0 + timedelta(seconds=e.t)).isoformat(timespec="seconds")
        rows.append(d)
    (out / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "state.json").write_text(json.dumps(mon.snapshots, indent=1))
    summary = {"camera_id": camera_id, "hours": round(hours, 3), "events": len(events), "events_per_hour": round(len(events) / max(hours, 1e-9), 2),
               "by_kind": by_kind, "wall_start": t_wall0.isoformat() if t_wall0 else None, "clock_offset_h": clock_offset_h,
               "process_s": round(time.time() - t_run, 1), "frames": mon.n,
               "dynamic_cells": int((mon.dyn > cfg.dynamic_thr).sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
