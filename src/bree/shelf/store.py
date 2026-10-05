"""One store, one pipeline path for multi-camera stores (what `make bench-dev` runs):

    item cameras (shelf, cooler, counter)   bree.shelf.events    shelf events, no person box needed
    people cameras (overhead, entrance)     bree.track.people    person boxes with pose -> bree.track.floor: one track per shopper
    who took it                             bree.track.associate via bree.events.shelf.store_events -> PICK / PUT_BACK
    theft at exit                           bree.ledger with the register feed -> alerts -> review store (bree.review)

    run(clip, out)    clip folder: <camera>.mp4, calibration.json, layout.json (planogram), register.jsonl, clip.json

Everything read here is pixels, calibration, the planogram and the register feed. The two slow steps keep their output
in <out>/pipeline (shelf_events.jsonl, people_<camera>.jsonl); `rejoin(clip, out)` repeats the rest from those files.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bree.events.shelf import load_payments, run_ledger, store_events, track_people, write_run
from bree.shelf.events import fuse_views, run_clip, unreliable_windows
from bree.track.associate import associate
from bree.track.people import PEOPLE_KINDS, appearance, frames, person_detector

SKU_WEIGHTS = "sim_sku_hands_v3"      # items + the hand class; better than sim_sku on the held-out seeds (docs: hand detector)
# The register feed of the simulated clips stamps a receipt 1.5 to 4.5 s after the payment (scripts/bench/render_clip.mjs:
# the sale closes, the receipt prints). A real store measures this once for its POS and sets it in the ledger settings.
POS_LAG_S = (1.5, 4.5)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _dump(p: Path, rows) -> None:
    p.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def one_act_per_reach(acts: list[dict], who: list, within_s: float = 4.0, within_m: float = 1.0) -> list[dict]:
    """Fused shelf events and the shopper each was attached to -> one event per reach. One reach into a shelf is often
    read more than once: by one camera at two facings, or by two cameras that put it at slots too far apart to be fused
    (45 of 60 false takes on DEV were such repeats of a true take). Events of one kind, by one shopper, this close in
    time and place are one act: the best-evidenced one is kept (two cues, then an item seen in the hand for more frames,
    then more cameras) with the earliest time. Two shoppers at neighbouring cooler doors stay two acts. The distance
    is along the floor: one reach read at two shelf heights of one bay (two cameras, or a cooler door) is one act.
    ponytail: one shopper taking two different products from the same metre of shelf within 4 s becomes one take; split
    by the hand track or the count once a clip has that."""
    rank = lambda e: (e.get("cue") == "both_cues", e["source"] == "both", e.get("detector_frames") or 0, len(e.get("cameras") or []), e.get("sku_conf") or 0.0)      # noqa: E731
    out: list[dict] = []
    for e, pid in sorted(zip(acts, who), key=lambda x: x[0]["t"]):
        g = next((g for g in reversed(out) if pid is not None and g["by"] == pid and g["kind"] == e["kind"] and e["t"] - g["t"] <= within_s
                  and e.get("point_3d") is not None and g.get("point_3d") is not None and math.dist(g["point_3d"][::2], e["point_3d"][::2]) <= within_m), None)
        if g is None:
            out.append({**e, "by": pid, "repeats": e.get("repeats", 0)})
        else:
            best = e if rank(e) > rank(g) else g
            g.update({**best, "t": g["t"], "t_start": min(g["t_start"], e["t_start"]), "by": pid, "repeats": g["repeats"] + e.get("repeats", 0) + 1,
                      "cameras": sorted(set(g.get("cameras") or []) | set(e.get("cameras") or []))})
    return out


def people_boxes(clip: Path, pipe: Path, cams: list[str], max_frames: int | None, verbose: bool) -> dict[str, list[list[dict]]]:
    """boxes[camera][frame] = [{"bbox", "conf", "kpts", "app"}], kept in people_<camera>.jsonl."""
    out, detect = {}, None
    for cam in cams:
        keep = pipe / f"people_{cam}.jsonl"
        if not keep.exists():
            detect = detect or person_detector()
            rows = []
            for f, im in enumerate(frames(clip, cam)):
                if max_frames and f >= max_frames:
                    break
                rows.append([{**p, "app": appearance(im, p)} for p in detect(im)])
            _dump(keep, rows)
            if verbose:
                print(f"  {cam}: {len(rows)} frames, {sum(len(r) for r in rows)} person boxes", flush=True)
        out[cam] = _jsonl(keep) if keep.stat().st_size else []
    return out


def rejoin(clip: Path, out: Path, review: bool = True, floor=None, assoc=None, join=None, ledger: dict | None = None, reach: dict | None = None) -> dict:
    """Stored shelf events and person boxes -> floor tracks, PICK / PUT_BACK, ledger, alerts, review store."""
    from bree.sim.bench import load_calibration
    clip, pipe = Path(clip), Path(out) / "pipeline"
    layout, meta = json.loads((clip / "layout.json").read_text()), json.loads((clip / "clip.json").read_text())
    fps = float(meta["fps"])
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    cams = {c: v for c, v in load_calibration(clip).items() if c in meta["cameras"] and kind[c] in PEOPLE_KINDS}
    boxes = {c: _jsonl(pipe / f"people_{c}.jsonl") for c in cams}
    shelf = _jsonl(pipe / "shelf_events.jsonl")
    tracker, ids = track_people(cams, layout, boxes, fps, floor)
    acts = fuse_views(shelf, layout)
    for _ in range(3):         # merging repeats changes who fits what is left (two reads of one reach can go to two people), so look again
        n = len(acts)
        acts = one_act_per_reach(acts, [a.person_id for a in associate(acts, tracker.people(), layout, cams=cams, cfg=assoc)], **(reach or {}))
        if len(acts) == n:
            break
    events, assocs = store_events(acts, tracker.people(), layout, cams=cams, assoc_cfg=assoc, **(join or {}))
    alerts, book = run_ledger(events, load_payments(clip / "register.jsonl"), layout, **{"pos_lag_s": POS_LAG_S, **(ledger or {})})
    write_run(out, events, alerts, boxes, ids, fps)
    _dump(pipe / "store_shelf_events.jsonl", [{**{k: v for k, v in g.items() if k != "views"}, "person_id": a.person_id, "uncertain": a.uncertain, "why": a.why}
                                              for g, a in zip(acts, assocs)])
    (pipe / "ledger_log.txt").write_text("\n\n".join(f"person {pid}:\n" + "\n".join(rec.log) for pid, rec in sorted(book.people.items())))
    (pipe / "engine_log.txt").write_text("\n".join(tracker.log))
    # frame logs of the item cameras (the scorer's "hand or item detected" stage): the reach point of each shelf event
    hands: dict[tuple[str, int], list] = {}
    for e in shelf:
        if e.get("hand_px"):
            for f in range(int(e["t_start"] * fps) - 5, int(e["t_end"] * fps) + 12):
                hands.setdefault((e["camera_id"], f), []).append(e["hand_px"])
    n = max((len(b) for b in boxes.values()), default=0)
    for cam in (c for c in meta["cameras"] if c not in cams):
        _dump(pipe / f"frames_{cam}.jsonl", [{"frame": f, "t": round(f / fps, 3), "persons": [], "products": [], "hands": hands.get((cam, f), [])} for f in range(n)])
    got = None
    if review and alerts:          # alerts go to the human review store, as in scripts/e2e_sim_chain.py
        import shutil
        from bree.review.__main__ import ingest
        from bree.review.store import ReviewStore
        shutil.rmtree(Path(out) / "review", ignore_errors=True)
        got = ingest(ReviewStore(Path(out) / "review"), str(pipe / "alerts.jsonl"), camera="store")
    people = [p for p in tracker.people() if not getattr(p, "staff", False)]
    return {"shelf_events": len(shelf), "store_shelf_events": len(acts), "unassigned_shelf_events": sum(a.person_id is None for a in assocs),
            "people": len(people), "uncertain_people": sum(bool(p.uncertain) for p in people), "alerts": len(alerts),
            "review_store": {k: got[k] for k in ("records", "new") if k in got} if got else None,
            "pipeline_events": {t: sum(1 for e in events if e.type.value == t) for t in sorted({e.type.value for e in events})}}


def run(clip: Path, out: Path, max_frames: int | None = None, verbose: bool = False, jobs: int = 3, **_) -> dict:
    """The pipeline on one clip folder (no ground truth in it). Writes what bree.sim.bench scores."""
    os.environ.setdefault("BREE_SKU_WEIGHTS", SKU_WEIGHTS)       # read by the shelf cameras' worker processes
    clip, pipe = Path(clip), Path(out) / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    meta = json.loads((clip / "clip.json").read_text())
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    t0, wall = time.time(), {}

    def shelf_stage() -> None:
        if not (pipe / "shelf_events.jsonl").exists():
            status: list[dict] = []
            evs = run_clip(clip, jobs=jobs, max_frames=max_frames, verbose=verbose, status=status)
            _dump(pipe / "shelf_status.jsonl", status)
            _dump(pipe / "shelf_events.jsonl", evs)
        wall["shelf"] = round(time.time() - t0, 1)
    with ThreadPoolExecutor(1) as ex:          # shelf cameras in worker processes while this process reads the people cameras
        fut = ex.submit(shelf_stage)
        people_boxes(clip, pipe, [c for c in meta["cameras"] if kind[c] in PEOPLE_KINDS], max_frames, verbose)
        wall["people"] = round(time.time() - t0, 1)
        fut.result()
    info = rejoin(clip, out)
    status = _jsonl(pipe / "shelf_status.jsonl") if (pipe / "shelf_status.jsonl").exists() else []
    info = {"runner": "bree.shelf.store:run", "sku_weights": os.environ["BREE_SKU_WEIGHTS"], "max_frames": max_frames, "fps": float(meta["fps"]),
            "wall_s": {**wall, "pipeline": round(time.time() - t0, 1)}, "unreliable_camera_spans": sum(len(v) for v in unreliable_windows(status).values()), **info}
    (Path(out) / "run.json").write_text(json.dumps(info, indent=1))
    if verbose:
        print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":      # .venv/bin/python -m bree.shelf.store <clip folder> <out folder> [--rejoin]
    import argparse
    ap = argparse.ArgumentParser(description="Shelf events + floor tracks + ledger on one clip folder.")
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--rejoin", action="store_true", help="reuse the stored shelf events and person boxes of <out>")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    print(json.dumps(rejoin(Path(a.clip), Path(a.out)) if a.rejoin else run(Path(a.clip), Path(a.out), a.max_frames, verbose=True), indent=1))
