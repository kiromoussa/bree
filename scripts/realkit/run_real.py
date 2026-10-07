#!/usr/bin/env python
"""Run the pipeline on REAL clips of one shelf and score it against the act log, in the shape of the simulator benchmark.

    P=.venv/bin/python; S=~/bree-footage/session1/session.yaml
    $P scripts/realkit/run_real.py prepare $S                 # cut every camera's file at its sync moment, turn it upright, 10 fps
    $P scripts/realkit/run_real.py mark $S                    # click the shelf corners in each camera (once per mounting)
    $P scripts/realkit/run_real.py check $S                   # calibration error + overlay pictures of the slot boxes. Look at them
    $P scripts/realkit/run_real.py run $S --log acts.csv      # the pipeline, the scorecard, a contact sheet per missed act
    $P scripts/realkit/run_real.py run $S --log acts.csv --score-only      # score again after fixing the log
    $P scripts/realkit/run_real.py run $S --log acts.csv --max-frames 600  # the first minute only, to see that it runs

The session file is described in scripts/realkit/session.py, the act log in scripts/realkit/make_truth.py.
Everything is written under <folder of the session file>/work (or --work):

    clip/<camera>.mp4, clip.json, calibration.json, layout.json, register.jsonl, truth/     the clip folder (the simulator's format)
    marks.json                 the clicked points
    overlay_<camera>.jpg       the slot boxes drawn on a frame: if they do not sit on the products, fix the marks or the shelf measures
    run/pipeline/              what the pipeline wrote (events.jsonl, alerts.jsonl, shelf_events.jsonl, ...)
    scorecard.md, scorecard.json
    missed/act_<n>_<what>.jpg  frames of every camera around each act the pipeline got wrong

This file only calls existing pipeline functions (bree.shelf.store.run, bree.sim.bench.score_clip / aggregate,
bree.calib.camera.calibrate, bree.shelf.diff.item_corners). What it cannot score on real footage, and why, is printed
in the scorecard: nobody drew person boxes, so "right person" is replaced by two weaker checks. A take the shelf
cameras read but no tracked person was attached to still counts as found (score_view); it cannot be flagged as a theft.
The pipeline run is heavy (minutes per minute of footage on a laptop with three cameras): close other jobs first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from make_truth import make_truth, read_log, write_truth      # noqa: E402
from session import build_layout, load_session, mark_points      # noqa: E402

NOTE = "REAL footage of one shelf, labelled from a hand-kept act log. Times are good to a few seconds; no person boxes were drawn."
RAW = (".h264", ".264", ".mjpeg", ".mjpg")      # streams without timestamps: the frame rate has to be told (fps_in)
ITEM_KINDS = ("shelf", "cooler", "checkout")


# ------------------------------------------------------------------ prepare: files of the cameras -> the clip folder
def ffmpeg_cmd(src, dst, sync_s: float = 0.0, fps: float = 10.0, rotate: int = 0, fps_in: float | None = None,
               max_s: float | None = None, max_side: int = 2688) -> list[str]:
    """Cut `src` at its sync moment, turn it upright, resample to `fps`, cap the long side. rotate: degrees clockwise."""
    turn = {0: [], 90: ["transpose=1"], 180: ["hflip", "vflip"], 270: ["transpose=2"]}[int(rotate) % 360]
    scale = f"scale='if(gt(iw,ih),min(iw,{max_side}),-2)':'if(gt(iw,ih),-2,min(ih,{max_side}))'"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
    if Path(src).suffix.lower() in RAW:
        cmd += ["-r", str(fps_in or fps)]
    cmd += ["-i", str(src), "-ss", f"{sync_s:.3f}"]      # after -i: exact, and it works on raw streams
    if max_s:
        cmd += ["-t", f"{max_s:.3f}"]
    return cmd + ["-vf", ",".join(turn + [f"fps={fps:g}", scale]), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]


def pts_drift(pts_file, fps_in: float) -> float | None:
    """rpicam-vid --save-pts file (one time in ms per frame) -> seconds the recording is longer than frames / fps_in.
    Over half a second means frames were dropped: that camera's clip runs ahead of the others by this much at the end."""
    ms = [float(x) for x in Path(pts_file).read_text().split("\n") if x.strip() and not x.startswith("#")]
    return None if len(ms) < 2 else round((ms[-1] - ms[0]) / 1000 - (len(ms) - 1) / fps_in, 2)


def prepare(s: dict, work: Path, a) -> None:
    import cv2
    clip = work / "clip"
    clip.mkdir(parents=True, exist_ok=True)
    for cid, c in s["cameras"].items():
        src = Path(s["_dir"]) / c["file"]
        if not src.exists():
            raise SystemExit(f"camera {cid}: {src} not found")
        pts = next((p for p in (src.with_suffix(".pts"), Path(str(src) + ".pts")) if p.exists()), None)      # rpicam-vid --save-pts, copied along with the video
        if pts or src.suffix.lower() in RAW:
            d = pts_drift(pts, c.get("fps_in", s["fps"])) if pts else None
            print(f"{cid}: " + ("no .pts file beside it, the frame rate is taken on trust" if d is None else f"timestamps say {d:+.2f} s against frames / fps_in"
                                + (" (frames were dropped: expect this camera to run early by that much at the end)" if abs(d) > 0.5 else " (fine)")))
        print(f"{cid}: {src.name} from {c.get('sync_s', 0)} s -> {clip / (cid + '.mp4')}", flush=True)
        subprocess.run(ffmpeg_cmd(src, clip / f"{cid}.mp4", float(c.get("sync_s", 0)), s["fps"], c.get("rotate", 0), c.get("fps_in"), a.max_s, a.max_side), check=True)
    n = {}
    for cid in s["cameras"]:
        cap = cv2.VideoCapture(str(clip / f"{cid}.mp4"))
        n[cid] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    print("frames per camera: " + ", ".join(f"{k} {v}" for k, v in n.items()) + f". The run uses the first {min(n.values())} ({min(n.values()) / s['fps']:.0f} s).")


# ------------------------------------------------------------------ mark and calibrate
def mark(s: dict, work: Path, a) -> None:
    from px_per_mm import click, first_frame
    path = work / "marks.json"
    marks = json.loads(path.read_text()) if path.exists() else {}
    names = mark_points(s)
    for cid in ([a.camera] if a.camera else list(s["cameras"])):
        img = first_frame(str(work / "clip" / f"{cid}.mp4"), a.at)
        print(f"{cid}: click each spot the window asks for; s = cannot see it, u = undo. Floor marks make the fit much safer.")
        got = click(img, [f"{cid}: {p}" for _, p, _ in names])
        marks[cid] = {"resolution": [img.shape[1], img.shape[0]], "marks": {n: [round(p[0], 1), round(p[1], 1)] for (n, _, _), p in zip(names, got) if p}}
        path.write_text(json.dumps(marks, indent=1))
        print(f"{cid}: {len(marks[cid]['marks'])} marks saved to {path}")


def calibrate_camera(s: dict, cid: str, entry: dict):
    """Clicked marks -> (bree.calib.camera.Camera, report). The lens: `hfov_deg` of the camera in the session file
    (the horizontal angle of the upright picture), else fitted from 6 or more marks."""
    from bree.calib.camera import calibrate
    where = {n: xyz for n, _, xyz in mark_points(s)}
    rows = [[u, v, *where[n]] for n, (u, v) in entry["marks"].items() if n in where]
    w, h = entry["resolution"]
    hfov = s["cameras"][cid].get("hfov_deg")
    cam, rep = calibrate(rows, (w, h), f=(w / 2) / math.tan(math.radians(hfov) / 2) if hfov else None)
    if cam is None:
        raise SystemExit(f"camera {cid}: no pose from {len(rows)} marks. Click 4 or more (6 or more without hfov_deg), not all on one line: run_real.py mark --camera {cid}")
    return cam, rep


def build_clip(s: dict, work: Path, log: str | None = None, max_frames: int | None = None) -> tuple[Path, dict, dict]:
    """Write calibration.json, layout.json, clip.json (and truth/ + register.jsonl from the act log) into work/clip."""
    import cv2
    clip = work / "clip"
    if not (work / "marks.json").exists():
        raise SystemExit(f"{work / 'marks.json'} not found: run `run_real.py prepare` and then `run_real.py mark` first")
    marks, cams, calib, n = json.loads((work / "marks.json").read_text()), {}, [], []
    for cid, c in s["cameras"].items():
        if cid not in marks:
            raise SystemExit(f"camera {cid} has no marks: run_real.py mark <session> --camera {cid}")
        cam, rep = calibrate_camera(s, cid, marks[cid])
        cams[cid] = cam
        bad = " <- high: click the marks again" if rep["rms_px"] > 0.006 * max(cam.resolution) else ""
        behind = " <- the fit puts the camera behind the shelf front: add floor marks and click again" if cam.C[2] <= 0 else ""
        print(f"{cid}: {rep['method']}, {rep['n']} marks, error {rep['rms_px']:.1f} px RMS{bad}{behind}; lens {cam.hfov_deg:.1f} deg across; camera "
              f"{cam.C[1] / 0.0254:.0f} in above the floor, {cam.C[2] / 0.0254:.0f} in out from the shelf front, {(cam.C[0] + s['shelf']['width_in'] * 0.0127) / 0.0254:.0f} in along from its left end")
        calib.append({"id": cid, "kind": c["kind"], "resolution": list(cam.resolution), "fx": cam.f, "fy": cam.f, "cx": cam.cx, "cy": cam.cy,
                      "position": [float(v) for v in cam.C], "R": [[float(v) for v in r] for r in cam.R], "k_div": 0.0, "fit": {k: rep[k] for k in ("method", "n", "rms_px", "max_px")}})
        cap = cv2.VideoCapture(str(clip / f"{cid}.mp4"))
        n.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    layout = build_layout(s)
    frames = min(n + ([max_frames] if max_frames else []))
    (clip / "calibration.json").write_text(json.dumps({"frame": "metres, y up, origin on the floor under the middle of the shelf front, z out into the aisle",
                                                        "model": "pinhole, no lens term", "note": "REAL cameras, fitted from clicked marks (run_real.py mark)", "cameras": calib}, indent=1))
    (clip / "layout.json").write_text(json.dumps(layout))
    (clip / "clip.json").write_text(json.dumps({"source": "REAL footage (scripts/realkit)", "generator": 2, "seed": s["name"], "fps": s["fps"], "frames": frames,
                                                "sim_seconds": round(frames / s["fps"], 1), "cameras": list(s["cameras"]), "planogram": "the shelf as listed in the session file"}, indent=1))
    if log:
        try:
            t = make_truth(read_log(log), s["sync_clock_s"], s["lead_s"], s["fps"], {x["id"]: x["skuId"] for x in layout["slots"]})
        except ValueError as e:
            raise SystemExit(f"{log}: {e}")
        late = [e for e in t["events"] + t["acts"] if e["t"] > frames / s["fps"]]
        for w in t["warnings"] + ([f"{len(late)} acts are after the end of the clips ({frames / s['fps']:.0f} s): check sync.clock and the clap times"] if late else []):
            print("warning:", w)
        write_truth(t, clip)
    elif not (clip / "register.jsonl").exists():
        (clip / "register.jsonl").write_text("")
    return clip, layout, cams


def slot_box(cam, slot: dict, skus: dict):
    """x0, y0, x1, y1 of the front item of a slot in the picture, or None when it is behind the camera."""
    from bree.shelf.diff import item_corners
    px, z = cam.project(item_corners(slot, skus.get(slot.get("skuId"))))
    return None if (z <= 0.05).any() else (px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max())


def overlays(clip: Path, layout: dict, cams: dict, work: Path, at_s: float = 1.0) -> list[Path]:
    """The slot boxes drawn on a frame of each item camera, with the width of each box in pixels."""
    import cv2
    from px_per_mm import first_frame
    skus, out = {x["id"]: x["size"] for x in layout["skus"]}, []
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    for cid, cam in cams.items():
        img = first_frame(str(clip / f"{cid}.mp4"), at_s)
        widths = []
        for sl in layout["slots"]:
            b = slot_box(cam, sl, skus)
            if b is None:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in b)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(img, sl["id"], (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if 0 <= x0 and x1 < img.shape[1] and 0 <= y0 and y1 < img.shape[0]:
                widths.append(x1 - x0)
        p = work / f"overlay_{cid}.jpg"
        cv2.imwrite(str(p), img)
        out.append(p)
        if kind[cid] in ITEM_KINDS:
            print(f"{cid}: {len(widths)} of {len(layout['slots'])} slots fully in view" + (f", item boxes {min(widths)} to {max(widths)} px wide" if widths else "") + f" -> {p}")
    return out


# ------------------------------------------------------------------ score
def headline(c: dict, visits: set | None = None) -> dict:
    """The six numbers of the simulator scorecard, from one score_clip result. visits: only these shopper visits."""
    keep = lambda r: visits is None or r["shopper"] in visits      # noqa: E731
    picks = [p for p in c["picks"] if keep(p)]
    found = [p for p in picks if p["stages"]["shelf_event_emitted"]]
    thefts = [t for t in c["thefts"] if keep(t)]
    honest = [q for q in c["people"] if keep(q) and q["role"] == "shopper" and not q["thief"]]
    puts = [g for g in c["put_backs"] if keep(g) and not g["staff"]]
    by_act: dict[str, list] = {}
    for p in found:
        by_act.setdefault(p["shopper"].split("-")[0], []).append(p)
    two = [v for v in by_act.values() if len({p["shopper"] for p in v}) > 1]
    return {"takes": len(picks), "takes_found": len(found),
            "right_slot": sum(p["stages"]["right_slot"] for p in found), "right_product": sum(p["stages"]["right_sku"] for p in found),
            "takes_tied_to_a_tracked_person": sum(p["stages"]["associated_to_a_shopper"] for p in found),
            "two_person_acts_found": len(two), "two_person_acts_given_to_two_identities": sum(len({p["detail"]["event_person"] for p in v} - {None}) > 1 for v in two),
            "thefts": len(thefts), "thefts_flagged": sum(t["alert_or_review"] for t in thefts), "thefts_flagged_alert_tier": sum(t["alert"] for t in thefts),
            "honest_visits": len(honest), "honest_visits_flagged": sum(q["alerted"] or q["reviewed"] for q in honest),
            "put_backs": len(puts), "put_backs_found": sum(g["found"] for g in puts), "put_backs_right_slot": sum(g["right_slot"] for g in puts)}


def headline_lines(h: dict) -> list[str]:
    of = lambda a, b: f"{a} of {b}" + (f" ({a / b:.0%})" if b else "")      # noqa: E731
    return [f"- Takes found by the shelf cameras (with or without a person attached): {of(h['takes_found'], h['takes'])}",
            f"- Right slot, of the takes found: {of(h['right_slot'], h['takes_found'])}",
            f"- Right product, of the takes found: {of(h['right_product'], h['takes_found'])}",
            f"- Right person: not scored (no person boxes were drawn). Takes tied to a tracked person: {of(h['takes_tied_to_a_tracked_person'], h['takes_found'])}. "
            f"Two-person acts whose takes went to two different identities: {of(h['two_person_acts_given_to_two_identities'], h['two_person_acts_found'])}",
            f"- Thefts flagged (alert or review): {of(h['thefts_flagged'], h['thefts'])}, of them alert tier: {h['thefts_flagged_alert_tier']}",
            f"- Honest visits flagged: {of(h['honest_visits_flagged'], h['honest_visits'])}",
            f"- Put-backs found: {of(h['put_backs_found'], h['put_backs'])}, into the right slot: {h['put_backs_right_slot']}"]


def missed_acts(c: dict) -> list[dict]:
    """Every act the pipeline got wrong: {"act", "what", "t", "slot", "why"}. One contact sheet each."""
    act = lambda name: (name or "A?").split("-")[0][1:]      # noqa: E731
    out = []
    for p in c["picks"]:
        st, d = p["stages"], p["detail"]
        if not st["shelf_event_emitted"]:
            out.append({"act": act(p["shopper"]), "what": "take_missed", "t": p["t"], "slot": p["slot"], "why": f"no take event within the tolerance ({p['outcome']}, {p['sku']})"})
        elif not st["right_slot"]:
            out.append({"act": act(p["shopper"]), "what": "wrong_slot", "t": p["t"], "slot": p["slot"], "why": f"the take was put at {d['event_slot']} ({d['event_sku']}), it was {p['slot']} ({p['sku']})"})
    for t in c["thefts"]:
        if not t["alert_or_review"]:
            out.append({"act": act(t["shopper"]), "what": "theft_not_flagged", "t": t["t_conceal"], "slot": t["slot"], "why": f"{t['sku']} taken at {t['t_pick']} s, hidden at {t['t_conceal']} s, no alert or review"})
    for g in c["put_backs"]:
        if not g["found"]:
            out.append({"act": act(g["shopper"]), "what": "restock_missed" if g["staff"] else "put_back_missed", "t": g["t"], "slot": g["slot"], "why": "no put-back event within the tolerance"})
    for x in c["touches"]:
        if x["counted_as_pick"]:
            out.append({"act": act(x["shopper"]), "what": "shift_counted_as_take", "t": x["t"], "slot": x["slot"], "why": "the item was only shifted"})
    for i, x in enumerate(c["false_picks"]):
        if not x["at_a_touch"]:
            out.append({"act": f"none{i + 1}", "what": "take_with_no_act", "t": x["t"], "slot": None, "why": f"a take event from {x['camera']} where the log has no act"})
    return out


def contact_sheet(clip: Path, layout: dict, cams: dict, t: float, slot_id: str | None, title: str, out: Path,
                  fps: float, offsets=(-2, -1, 0, 1, 2, 4), cell: int = 300) -> Path:
    """One row per camera, one column per moment around t. An item camera is cropped around the slot (green box)."""
    import cv2
    import numpy as np
    skus = {x["id"]: x["size"] for x in layout["skus"]}
    slot = next((x for x in layout["slots"] if x["id"] == slot_id), None)
    rows = []
    for cid, cam in cams.items():
        cap, cells = cv2.VideoCapture(str(clip / f"{cid}.mp4")), []
        box = slot_box(cam, slot, skus) if slot else None
        for dt in offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, round((t + dt) * fps)))
            ok, im = cap.read()
            tile = np.zeros((cell, cell, 3), np.uint8)
            if ok:
                if box is not None:
                    x0, y0, x1, y1 = (int(round(v)) for v in box)
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    r = int(max(240, 3 * max(x1 - x0, y1 - y0)))          # half side of the crop: room for the arm
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    im = im[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                if im.size:
                    k = cell / max(im.shape[:2])
                    im = cv2.resize(im, (max(1, int(im.shape[1] * k)), max(1, int(im.shape[0] * k))), interpolation=cv2.INTER_AREA)
                    tile[:im.shape[0], :im.shape[1]] = im
            cv2.putText(tile, f"{cid} {dt:+d}s", (4, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cells.append(tile)
        cap.release()
        rows.append(np.hstack(cells))
    head = np.zeros((28, rows[0].shape[1], 3), np.uint8)
    cv2.putText(head, title[:140], (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head] + rows))
    return out


def score_view(run: Path, layout: dict) -> Path:
    """A copy of the run for the scorer in which shelf events that no tracked person was attached to also count.
    bree.events.shelf.store_events writes a PICK or PUT_BACK only for a shelf event with a person, so a test with no
    people camera (or a person the tracker lost) would score "0 takes found" although the shelf cameras read the take.
    Those shelf events are in <run>/pipeline/store_shelf_events.jsonl; they are added here with person_id None."""
    src, dst = run / "pipeline", run / "score" / "pipeline"
    shutil.rmtree(run / "score", ignore_errors=True)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "events.jsonl":
            (dst / f.name).symlink_to(f.resolve())
    if (run / "run.json").exists():
        (run / "score" / "run.json").symlink_to((run / "run.json").resolve())
    slots = {x["id"]: x for x in layout["slots"]}
    rows = [json.loads(x) for x in (src / "events.jsonl").read_text().splitlines() if x.strip()]
    loose = [json.loads(x) for x in (src / "store_shelf_events.jsonl").read_text().splitlines() if x.strip()] if (src / "store_shelf_events.jsonl").exists() else []
    for e in loose:
        if e.get("person_id") is None and e.get("kind") in ("take", "put"):
            sl = slots.get(e.get("slot_id")) or {}
            sku = e.get("sku_id") or sl.get("skuId")
            rows.append({"type": "pick" if e["kind"] == "take" else "put_back", "t": e["t"], "person_id": None, "item": sku or "unknown", "sku": sku, "zone": sl.get("fixtureId"),
                         "confidence": 0.9, "candidates": [], "meta": {"slot": {"id": e.get("slot_id"), "fixture": sl.get("fixtureId"), "sku": sl.get("skuId")},
                                                                      "shelf": {"camera_id": e.get("camera_id"), "source": e.get("source")}, "no_person": e.get("why") or True}})
    (dst / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["t"])))
    return run / "score"


def scorecard(s: dict, clip: Path, run: Path, work: Path, layout: dict, cams: dict, tol_s: float, sheets: bool = True) -> dict:
    from bree.sim.bench import STAGE_HELP, _score_rows, aggregate, score_clip
    c = score_clip(clip, score_view(run, layout), pick_tol_s=tol_s)
    people = json.loads((clip / "truth" / "shoppers.json").read_text())
    light = {r["shopper"]: r.get("light", "bright") for r in (json.loads(x) for f in ("events.jsonl", "acts.jsonl") for x in (clip / "truth" / f).read_text().splitlines())}
    res = {"split": f"real:{s['name']}", "data": NOTE, "runner": "bree.shelf.store:run", "options": {"pick_tol_s": tol_s, "sku_weights": os.environ.get("BREE_SKU_WEIGHTS")},
           "scored_at": time.strftime("%Y-%m-%d %H:%M"), "headline": headline(c), **aggregate([c]), "stage_help": STAGE_HELP, "clips": [c]}
    L = [f"# Real footage scorecard, {s['name']}", "", NOTE, "",
         f"{c['sim_seconds']:.0f} s of footage, cameras {', '.join(c['cameras'])}. {len(people)} visits in the log, {len(c['picks'])} takes, {len(c['thefts'])} stolen items. "
         f"An event counts for an act when it is within {tol_s:g} s of the logged time. Scored {res['scored_at']}.", "", "## Headline", ""] + headline_lines(res["headline"])
    for name in sorted(set(light.values()) - {None}) if len(set(light.values())) > 1 else []:
        vis = {q["shopper"] for q in people if light.get(q["shopper"], "bright") == name}
        res[f"headline_{name}"] = headline(c, vis)
        L += ["", f"### Only the acts filmed in {name} light", ""] + headline_lines(res[f"headline_{name}"])
    sc = res["scorecard"]
    L += ["", f"Also: {sc['touches_counted_as_picks']} of {sc['touches']} shifted items counted as a take; {sc['staff_puts_with_a_put_event']} of {sc['staff_puts']} restocks read as a put; "
          f"{sc['staff_takes_listed_unpaid']} of {sc['staff_takes']} staff takes listed as unpaid; {c['pipeline_picks'] - c['pipeline_picks_paired']} of {c['pipeline_picks']} take events match no act.",
          "", "## The simulator benchmark's table, same rows", "",
          "Rows that need person boxes (right shopper, identities, never tracked) read 0 or n/a here: that is the missing labels, not the pipeline. "
          "With one clip the bracketed intervals come from drawing the visits again.", "", "| Metric | Value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in _score_rows(sc, res.get("intervals"))]
    L += ["", "## Every take in the log", "", "| act | t (s) | who | slot | product | outcome | found | slot read | product read | person id |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in c["picks"]:
        d = p["detail"]
        L.append(f"| {p['shopper'].split('-')[0][1:]} | {p['t']} | {p['shopper'].split('-', 1)[-1]} | {p['slot']} | {p['sku']} | {p['outcome']} | {'yes' if p['stages']['shelf_event_emitted'] else 'NO'} | "
                 f"{d['event_slot'] or ''} | {d['event_sku'] or ''} | {'' if d['event_person'] is None else d['event_person']} |")
    flagged = [q for q in c["people"] if (q["alerted"] or q["reviewed"]) and not q["thief"]]
    if flagged:
        L += ["", "Honest visits that were flagged: " + ", ".join(f"{q['shopper']} ({'alert' if q['alerted'] else 'review'})" for q in flagged) + "."]
    miss = missed_acts(c)
    res["missed"] = miss
    L += ["", f"## What went wrong: {len(miss)} acts", "", "One picture per row in `missed/`: every camera, from 2 s before to 4 s after, the logged slot boxed in green.", "",
          "| act | what | t (s) | slot | why | picture |", "|---|---|---|---|---|---|"]
    shutil.rmtree(work / "missed", ignore_errors=True)
    for m in miss:
        name = f"act_{m['act']}_{m['what']}_{m['t']:.0f}s.jpg"
        if sheets:
            contact_sheet(clip, layout, cams, m["t"], m["slot"], f"act {m['act']}: {m['what'].replace('_', ' ')} at {m['t']} s, {m['slot'] or ''}. {m['why']}", work / "missed" / name, c["run"].get("fps", s["fps"]))
        L.append(f"| {m['act']} | {m['what'].replace('_', ' ')} | {m['t']} | {m['slot'] or ''} | {m['why']} | missed/{name} |")
    (work / "scorecard.json").write_text(json.dumps(res, indent=1, default=str))
    (work / "scorecard.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:L.index("## The simulator benchmark's table, same rows")]))
    print(f"written: {work / 'scorecard.md'}, {work / 'scorecard.json'}, {len(miss) if sheets else 0} contact sheets in {work / 'missed'}")
    return res


def run(s: dict, work: Path, a) -> None:
    if not a.log:
        raise SystemExit("run needs the act log: --log acts.csv")
    clip, layout, cams = build_clip(s, work, a.log, a.max_frames)
    overlays(clip, layout, cams, work)
    out = work / "run"
    if a.weights:
        os.environ["BREE_SKU_WEIGHTS"] = a.weights
    if not a.score_only:
        from bree.shelf.store import run as store_run
        from bree.sim.bench import public_view
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        store_run(public_view(clip, out), out, max_frames=a.max_frames, verbose=True, jobs=a.jobs)      # the pipeline sees the folder without truth/
    scorecard(s, clip, out, work, layout, cams, a.tol, sheets=not a.no_sheets)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("prepare", "mark", "check", "run"))
    ap.add_argument("session", help="session.yaml")
    ap.add_argument("--work", help="output folder (default: work/ beside the session file)")
    ap.add_argument("--log", help="run: the act log CSV")
    ap.add_argument("--max-s", type=float, help="prepare: keep only this many seconds after the sync moment")
    ap.add_argument("--max-side", type=int, default=2688, help="prepare: shrink a picture whose long side is over this (the simulator's 4MP cameras are 2688 px)")
    ap.add_argument("--camera", help="mark: only this camera")
    ap.add_argument("--at", type=float, default=1.0, help="mark: use the frame this many seconds in")
    ap.add_argument("--max-frames", type=int, help="run: only the first frames of every camera")
    ap.add_argument("--tol", type=float, default=5.0, help="run: seconds between a logged act and a pipeline event (the simulator uses 3; a hand-kept log needs more)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--weights", help="run: detector weights (BREE_SKU_WEIGHTS), for example real_sku after scripts/train/finetune_real.py")
    ap.add_argument("--score-only", action="store_true", help="run: score the pipeline output that is already there")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    s = load_session(a.session)
    work = Path(a.work) if a.work else Path(s["_dir"]) / "work"
    work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prepare":
        prepare(s, work, a)
    elif a.cmd == "mark":
        mark(s, work, a)
    elif a.cmd == "check":
        clip, layout, cams = build_clip(s, work, a.log)
        overlays(clip, layout, cams, work)
        print("Open the overlay pictures. Each green box should sit on the front item of its slot. If not: click the marks again, or fix the shelf measures.")
    else:
        run(s, work, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
