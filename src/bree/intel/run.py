"""Run the flow analytics over recorded segments: detections (cached) -> FlowEngine -> JSONL records + heat grids.

Each segment is analysed on its own (the recorder's segments are 5 minutes with short gaps); the segment's start time is
read from the file name (YYYYmmdd_HHMMSS, the recorder's local clock) and shifted by the layout's clock offset.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from bree.intel.detections import detect_segment, read_detections
from bree.intel.flow import FlowEngine, IntelConfig, run_detections
from bree.intel.layout import load_layout

NAME = re.compile(r"(\d{8})_(\d{6})")


def segment_start(video: str | Path, offset_s: float = 0.0) -> datetime | None:
    m = NAME.search(Path(video).name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") + timedelta(seconds=offset_s)


def run_segments(videos: list[str], layout_path: str, out_dir: str, dets_dir: str | None = None, stride: int = 5,
                 imgsz: int | None = 960, cfg: IntelConfig | None = None, detector=None, learn_mask: bool = False) -> Path:
    """Analyse `videos` in name order. Writes <out_dir>/flow.jsonl and <out_dir>/heat.npz. Returns the flow.jsonl path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dets_dir_p = Path(dets_dir) if dets_dir else out / "dets"
    heat = {"customer": None, "staff": None}
    flow = out / "flow.jsonl"
    with flow.open("w") as fh:
        for v in sorted(videos):
            d = dets_dir_p / (Path(v).stem + ".jsonl")
            if not d.exists():
                detect_segment(v, d, stride=stride, imgsz=imgsz, detector=detector)
            meta, frames = read_detections(d)
            layout = load_layout(layout_path, (meta["width"], meta["height"]))
            if learn_mask:     # static ghosts: boxes that sit on one spot all segment at low confidence
                from bree.intel.staticmask import learn_static_mask
                layout.ignore = list(layout.ignore) + [np.array(p) for p in learn_static_mask(frames, (meta["width"], meta["height"]))]
            start = segment_start(v, layout.clock_offset_s)
            recs, eng = run_detections(frames, layout, cfg, start)
            t_end = frames[-1]["t"] if frames else 0.0
            fh.write(json.dumps({"type": "segment", "video": Path(v).name, "t0": 0.0, "t1": round(t_end, 2),
                                 "ts": start.isoformat(timespec="seconds") if start else None, "entries": eng.entries,
                                 "exits": eng.exits, "initial": eng.initial, "camera": layout.camera}) + "\n")
            for r in recs:
                r["camera"] = layout.camera
                fh.write(json.dumps(r) + "\n")
            for k in heat:
                heat[k] = eng.heat[k] if heat[k] is None else heat[k] + eng.heat[k]
    np.savez_compressed(out / "heat.npz", **{k: v for k, v in heat.items() if v is not None})
    return flow
