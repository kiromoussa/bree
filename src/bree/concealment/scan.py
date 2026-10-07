"""Every look of the hand and held-item detector on the item cameras of one clip, kept for the concealment cue.

  .venv/bin/python -m bree.concealment.scan <clip folder> <out folder> [--jobs 3]

Writes <out>/shelf_<camera>.json (that camera's shelf events and health records, the same as the shelf stage makes) and
<out>/looks_<camera>.jsonl, one line per frame the detector looked at (bree.shelf.hand.HandItemCue.log):
{"f", "items": [[x0, y0, x1, y1, conf, sku, changed share, moved, stock]], "hands": [[x0, y0, x1, y1, conf, at_shelf, item]]}.
at_shelf and item are the two answers of the hand crop classifier (bree.concealment.where), there when its weights
exist (models/conceal_where.pt); `--where` adds them to looks written without them.
It runs the same ShelfCamera as the shelf stage (no ground truth is read), so the shelf stage could write these
files itself in its one pass: set `ShelfCamera.cue.log = []` and dump it.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def patient_nms(seconds_per_image: float = 30.0) -> None:
    """ultralytics stops its NMS after 2 s + 0.05 s per image and silently drops the boxes of the images it did not reach
    ("NMS time limit exceeded"). On a busy machine that happens (19 times in the logs of the recorded TEST run), so the
    result depends on what else was running. Give it time instead. Call once per process, before the first detection."""
    import functools

    import ultralytics.utils.nms as N
    if not getattr(N.non_max_suppression, "bree_patient", False):
        f = functools.partial(N.non_max_suppression, max_time_img=seconds_per_image)
        f.bree_patient = True
        N.non_max_suppression = f


class _NmsCut:
    """Counts ultralytics' "NMS time limit exceeded" warnings: on an overloaded machine it stops early and drops boxes."""

    def __init__(self):
        import logging
        self.n = 0
        self.h = logging.Handler()
        self.h.emit = lambda r: setattr(self, "n", self.n + ("NMS time limit" in r.getMessage()))
        logging.getLogger("ultralytics").addHandler(self.h)


def _one(args) -> tuple[str, int]:
    clip, cam_id, out = args
    import cv2
    patient_nms()
    cut = _NmsCut()
    from bree.shelf.events import ShelfCamera, frames
    from bree.shelf.hand import sku_detector
    from bree.sim.bench import load_calibration
    cv2.setNumThreads(2)
    clip = Path(clip)
    layout = json.loads((clip / "layout.json").read_text())
    fps = float(json.loads((clip / "clip.json").read_text())["fps"])
    sc = ShelfCamera(cam_id, load_calibration(clip)[cam_id], layout, fps, None, sku_detector(), None)
    sc.cue.log = []
    where, events = _where(), []
    for im in frames(clip / f"{cam_id}.mp4"):
        events += sc.update(im)
        if where is not None and sc.cue.log and sc.cue.log[-1]["f"] == sc.f:
            _add_where(where, im, sc.cue.log[-1])
    events += sc.finish()
    if cut.n:          # detections were dropped: do not keep a file that looks complete
        return cam_id, -cut.n
    # the same camera's shelf events and health records, as bree.shelf.events.camera_events gives them: one pass can serve both
    Path(out, f"shelf_{cam_id}.json").write_text(json.dumps({"events": events, "status": sc.status}))
    Path(out, f"looks_{cam_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc.cue.log))
    return cam_id, len(sc.cue.log)


def _where():
    from bree.concealment.where import WEIGHTS, Where
    return Where() if WEIGHTS.exists() else None


def _add_where(where, image, look: dict) -> None:
    pr = where(image, [h[:4] for h in look["hands"]])
    look["hands"] = [[*h[:5], round(float(q[0] + q[1]), 3), round(float(q[1] + q[3]), 3)] for h, q in zip(look["hands"], pr)]


def _annotate(args) -> str:
    """Add the hand crop classifier's answers to looks that were written without them (reads the video again, no detector)."""
    clip, cam_id, out = args
    from bree.shelf.events import frames
    f = Path(out, f"looks_{cam_id}.jsonl")
    rows = [json.loads(x) for x in f.read_text().splitlines() if x]
    todo = {r["f"]: r for r in rows if r["hands"] and len(r["hands"][0]) < 7}
    if todo:
        where = _where()
        for i, im in enumerate(frames(Path(clip) / f"{cam_id}.mp4", max(todo) + 1)):
            if i in todo:
                _add_where(where, im, todo[i])
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return cam_id


def annotate(clip: Path, out: Path, jobs: int = 3) -> None:
    with ProcessPoolExecutor(jobs) as ex:
        list(ex.map(_annotate, [(str(clip), f.stem[len("looks_"):], str(out)) for f in sorted(Path(out).glob("looks_*.jsonl"))]))


def scan(clip: Path, out: Path, jobs: int = 3, verbose: bool = False) -> dict[str, int]:
    """looks_<camera>.jsonl for every item camera that has none yet. -> {camera: looks}"""
    import os
    from bree.shelf.events import ITEM_KINDS
    from bree.shelf.store import SKU_WEIGHTS
    os.environ.setdefault("BREE_SKU_WEIGHTS", SKU_WEIGHTS)
    clip, out = Path(clip), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    cams = [c for c in json.loads((clip / "clip.json").read_text())["cameras"] if kind[c] in ITEM_KINDS]
    todo = [c for c in cams if not (out / f"looks_{c}.jsonl").exists()]
    got = {}
    if todo:
        with ProcessPoolExecutor(max(1, min(jobs, len(todo)))) as ex:
            for cam, n in ex.map(_one, [(str(clip), c, str(out)) for c in todo]):
                got[cam] = n
                if verbose:
                    print(f"  {cam}: {n} looks" if n >= 0 else f"  {cam}: NOT KEPT, the detector's NMS hit its time limit {-n} times (machine overloaded): run again", flush=True)
    if any(n < 0 for n in got.values()):
        raise RuntimeError(f"scan of {clip.name} is incomplete: {sorted(c for c, n in got.items() if n < 0)} lost detections to the NMS time limit; run again on a quieter machine")
    return got


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--where", action="store_true", help="only add the hand crop classifier's answers to existing looks")
    a = ap.parse_args()
    annotate(Path(a.clip), Path(a.out), a.jobs) if a.where else scan(Path(a.clip), Path(a.out), a.jobs, verbose=True)
