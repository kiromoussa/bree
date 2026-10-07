"""The pipeline with the concealment cue switched on, as a benchmark runner.

  .venv/bin/python -m bree.sim.bench train --clips 4960 --runner bree.concealment.run:run --keep --out out/conceal/runs/train --name conceal

`run` does what bree.shelf.store.run does, with the item cameras read by bree.concealment.scan (the same ShelfCamera, so
the same shelf events, plus the detector's looks; skipped when already there), and hands the cues to store_events. It wraps store_events for the length of one call instead of editing
bree.shelf.store; the fragment in docs/fragments/concealment says what the four-line edit there looks like.
"""
from __future__ import annotations

import json
from pathlib import Path

from bree.concealment.cue import ConcealConfig, conceal_cues, load_looks
from bree.concealment.scan import scan


def context(clip: Path, out: Path, floor=None, assoc=None, reach: dict | None = None) -> dict:
    """Stored shelf events and person boxes -> tracks, fused acts and who did each: the first steps of
    bree.shelf.store.rejoin, for the scripts that study the cue."""
    from bree.shelf.events import fuse_views
    from bree.shelf.store import PEOPLE_KINDS, _jsonl, one_act_per_reach, track_people
    from bree.sim.bench import load_calibration
    from bree.track.associate import associate
    clip, pipe = Path(clip), Path(out) / "pipeline"
    layout, meta = json.loads((clip / "layout.json").read_text()), json.loads((clip / "clip.json").read_text())
    fps = float(meta["fps"])
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    calib = load_calibration(clip)
    cams = {c: v for c, v in calib.items() if c in meta["cameras"] and kind[c] in PEOPLE_KINDS}
    tracker, _ = track_people(cams, layout, {c: _jsonl(pipe / f"people_{c}.jsonl") for c in cams}, fps, floor)
    from bree.shelf import store
    from bree.shelf.events import arrivals
    shelf = _jsonl(pipe / "shelf_events.jsonl")
    acts = fuse_views(arrivals(shelf, layout) if store.ARRIVALS else shelf, layout)      # as rejoin does
    for _ in range(3):
        n = len(acts)
        acts = one_act_per_reach(acts, [a.person_id for a in associate(acts, tracker.people(), layout, cams=cams, cfg=assoc)], **(reach or {}))
        if len(acts) == n:
            break
    who = [a.person_id for a in associate(acts, tracker.people(), layout, cams=cams, cfg=assoc)]
    return {"layout": layout, "fps": fps, "calib": calib, "people": tracker.people(), "acts": acts, "who": who}


def run(clip: Path, out: Path, max_frames: int | None = None, verbose: bool = False, jobs: int = 3, conceal: ConcealConfig | None = None, cues=None, tier: bool = False, **_) -> dict:
    """cues: callable(acts, people, who) -> cue dicts, in place of the item-camera cue (scripts/conceal/eval.py uses it for
    the run with the cue off and for the run fed the true concealment times).
    tier: also apply bree.concealment.tier.retier to the ledger's records (the ledger's own are kept in alerts_ledger.jsonl)."""
    import bree.shelf.store as S
    from bree.sim.bench import load_calibration
    from bree.track.associate import associate
    clip, out = Path(clip), Path(out)
    pipe = out / "pipeline"
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    order = json.loads((clip / "clip.json").read_text())["cameras"]
    if not (pipe / "shelf_events.jsonl").exists():
        # one pass over the item cameras gives the shelf events and the looks; a pass that lost detections to an
        # overloaded machine raises instead of leaving files that look complete
        import os
        os.environ.setdefault("BREE_SKU_WEIGHTS", S.SKU_WEIGHTS)
        (pipe / "conceal").mkdir(parents=True, exist_ok=True)
        scan(clip, pipe / "conceal", jobs=jobs, verbose=verbose)
        per = [json.loads(f.read_text()) for c in order if (f := pipe / "conceal" / f"shelf_{c}.json").exists()]
        S._dump(pipe / "shelf_status.jsonl", sorted((s for d in per for s in d["status"]), key=lambda s: s["t"]))
        S._dump(pipe / "shelf_events.jsonl", sorted((e for d in per for e in d["events"]), key=lambda e: e["t"]))
    people_cams = [c for c in order if kind[c] in S.PEOPLE_KINDS]
    if any(not (pipe / f"people_{c}.jsonl").exists() for c in people_cams):
        from bree.concealment.scan import _NmsCut, patient_nms
        patient_nms()
        have, cut = {c for c in people_cams if (pipe / f"people_{c}.jsonl").exists()}, _NmsCut()
        S.people_boxes(clip, pipe, people_cams, max_frames, verbose)
        if cut.n:
            for c in people_cams:
                if c not in have:
                    (pipe / f"people_{c}.jsonl").unlink(missing_ok=True)
            raise RuntimeError(f"person boxes of {clip.name}: the detector's NMS hit its time limit {cut.n} times (machine overloaded); run again")
    if cues is None:
        scan(clip, pipe / "conceal", jobs=jobs, verbose=verbose)
    looks, calib = load_looks(pipe / "conceal") if cues is None else {}, load_calibration(clip)
    fps = float(json.loads((clip / "clip.json").read_text())["fps"])
    orig, kept = S.store_events, {}

    def with_cues(acts, people, layout, cams=None, **kw):
        kw.pop("conceal", None)          # bree.shelf.store.rejoin passes its own (None while its switch is off)
        who = [a.person_id for a in associate(acts, people, layout, cams=cams, cfg=kw.get("assoc_cfg"))]
        kept["cues"], kept["score"] = conceal_cues(looks, calib, people, acts, who, layout, fps, conceal) if cues is None else (cues(acts, people, who), {})
        return orig(acts, people, layout, cams=cams, conceal=kept["cues"], **kw)
    S.store_events = with_cues
    try:
        info = S.rejoin(clip, out)
    finally:
        S.store_events = orig
    (pipe / "conceal_cues.jsonl").write_text("".join(json.dumps(c) + "\n" for c in kept.get("cues", [])))
    (pipe / "conceal_scores.json").write_text(json.dumps(kept.get("score", {}), indent=1))
    if tier:
        from bree.concealment.tier import retier
        rows = S._jsonl(pipe / "alerts.jsonl")
        new = retier(rows, kept.get("score") if cues is None else None, (conceal or ConcealConfig()).shopper_bar)
        S._dump(pipe / "alerts_ledger.jsonl", rows)
        S._dump(pipe / "alerts.jsonl", new)
        info["raised_to_alert"] = sum(a["tier"] != b["tier"] for a, b in zip(rows, new))
        if new and (out / "review").exists():       # the review store was filled from the ledger's records: fill it again
            import shutil
            from bree.review.__main__ import ingest
            from bree.review.store import ReviewStore
            shutil.rmtree(out / "review", ignore_errors=True)
            ingest(ReviewStore(out / "review"), str(pipe / "alerts.jsonl"), camera="store")
    info = {"runner": "bree.concealment.run:run" + ("_tier" if tier else ""), "conceal_cues": len(kept.get("cues", [])), **info}
    (out / "run.json").write_text(json.dumps(info, indent=1))
    return info


def run_tier(clip: Path, out: Path, **kw) -> dict:
    """The cue and the tier rule of bree.concealment.tier, as a benchmark runner: --runner bree.concealment.run:run_tier"""
    return run(clip, out, **{**kw, "tier": True})
