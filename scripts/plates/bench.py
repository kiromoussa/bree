"""Plate reader and drive-off bench on SYNTHETIC rendered plates and SCRIPTED scenes.

    .venv/bin/python scripts/plates/bench.py                    # full: about 16 minutes measured on a busy laptop
    .venv/bin/python scripts/plates/bench.py --quick            # 2 widths, 2 conditions, few samples, about 3 minutes
    .venv/bin/python scripts/plates/bench.py --scenarios-only   # keep the reader table of the last full run,
                                                                # rerun the vote calibration and the timelines

Writes results/plates_bench.md (the tables) and out/plates/bench.json (every number).
Nothing here is real footage: plates are drawn with system fonts on crude car shapes.

Columns of the reader table:
  ocr char / ocr plate   the text reader alone on a plate crop (as a perfect plate detector would give it)
  vote5 plate            the same plate read in 5 frames (new angle and noise each), one voted answer
  det open / e2e open    open plate detector inside the vehicle box: plate found (IoU >= 0.4), whole plate right
  det cv / e2e cv        the OpenCV-only localiser instead of the open detector
  e2e both               candidates from both detectors, keeping the one the text reader is most sure about
Character accuracy is 1 - (sum of edit distances / sum of true lengths). Whole plate means every character right.

The vote calibration (its own seed, never the reader table's) is what the plate storage threshold
`DriveOffConfig.min_plate_conf` rests on: how often a voted plate is right, by vote confidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from bree.plates import synth
from bree.plates.reader import BothDetectors, ClassicalDetector, OpenDetector, OpenOcr, PlatePipeline, _iou, edit_distance, vote
from bree.plates.retention import PlateStore
from bree.plates.scenarios import SCENARIOS, run

ROOT = Path(__file__).resolve().parents[2]
WIDTHS = (32, 48, 64, 96, 160)
CONDS = ("day", "day_motion", "night", "night_noisy")


def reader_cell(rng, ocr_pipe, pipes, w: int, cond: str, n_crop: int, n_vote: int, n_scene: int) -> dict:
    ed = ln = whole = 0
    for _ in range(n_crop):
        s = synth.plate_crop(rng, w, cond)
        r = ocr_pipe.read_crop(s.image)
        ed += min(len(s.text), edit_distance(s.text, r.text))
        ln += len(s.text)
        whole += r.text == s.text
    v_ok = 0
    for _ in range(n_vote):
        text = synth.random_text(rng)
        reads = [ocr_pipe.read_crop(synth.plate_crop(rng, w, cond, text=text).image) for _ in range(5)]
        v_ok += vote(reads, min_conf=0.0)[0] == text.replace(" ", "")
    out = {"width_px": w, "condition": cond, "n_crop": n_crop, "n_vote": n_vote, "n_scene": n_scene,
           "ocr_char": 1 - ed / ln, "ocr_plate": whole / n_crop, "vote5_plate": v_ok / max(1, n_vote)}
    size = (1920, 1080)
    det = {k: 0 for k in (*pipes, "both")}
    e2e = {k: 0 for k in (*pipes, "both")}
    for _ in range(n_scene):
        s, car = synth.scene(rng, w, cond, size=size)
        union = []
        for k, p in pipes.items():
            reads = p.read(s.image, roi=car)
            union += reads
            det[k] += any(_iou(r.box, s.box) >= 0.4 for r in reads)
            e2e[k] += bool(reads) and reads[0].text == s.text
        union.sort(key=lambda r: r.conf * r.det_conf * (len(r.text) >= 4), reverse=True)   # as BothDetectors does
        det["both"] += any(_iou(r.box, s.box) >= 0.4 for r in union)
        e2e["both"] += bool(union) and union[0].text == s.text
    for k in det:
        out[f"det_{k}"] = det[k] / max(1, n_scene)
        out[f"e2e_{k}"] = e2e[k] / max(1, n_scene)
    return out


def vote_calibration(ocr_pipe, seed: int, n: int) -> dict:
    """How often is a 5-frame vote right, by its confidence? 48 to 96 px, day, motion, night, `n` plates per cell."""
    rng = np.random.default_rng(seed)
    bins = {k: {"n": 0, "right": 0} for k in ("0.9 and up", "0.8 to 0.9", "under 0.8")}
    for w in (48, 64, 96):
        for cond in ("day", "day_motion", "night"):
            for _ in range(n):
                text = synth.random_text(rng)
                reads = [ocr_pipe.read_crop(synth.plate_crop(rng, w, cond, text=text).image) for _ in range(5)]
                got, conf = vote(reads, min_conf=0.0)
                b = bins["0.9 and up" if conf >= 0.9 else "0.8 to 0.9" if conf >= 0.8 else "under 0.8"]
                b["n"] += 1
                b["right"] += got == text.replace(" ", "")
    return {"seed": seed, "plates": 9 * n, "bins": bins}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-crop", type=int, default=150)
    ap.add_argument("--n-vote", type=int, default=30)
    ap.add_argument("--n-scene", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--scenarios-only", action="store_true",
                    help="reuse the reader table in out/plates/bench.json, rerun only the scripted timelines")
    ap.add_argument("--out", default=str(ROOT / "results/plates_bench.md"))
    a = ap.parse_args()
    logging.disable(logging.INFO)
    widths, conds = (WIDTHS, CONDS) if not a.quick else ((48, 128), ("day", "night"))
    if a.quick:
        a.n_crop, a.n_vote, a.n_scene = 30, 10, 8
    rng = np.random.default_rng(a.seed)
    ocr = OpenOcr()
    pipes = {"open": PlatePipeline(OpenDetector(), ocr), "cv": PlatePipeline(ClassicalDetector(), ocr)}
    t0, rows = time.time(), []
    old = json.loads((ROOT / "out/plates/bench.json").read_text()) if a.scenarios_only else None
    if old:
        rows, widths = old["reader"], ()
    for w in widths:
        for c in conds:
            rows.append(reader_cell(rng, pipes["open"], pipes, w, c, a.n_crop, a.n_vote, a.n_scene))
            r = rows[-1]
            print(f"[{time.time() - t0:5.0f}s] w={w:3d} {c:12s} ocr char {r['ocr_char']:.3f} plate {r['ocr_plate']:.3f} "
                  f"vote5 {r['vote5_plate']:.3f} | open det {r['det_open']:.2f} e2e {r['e2e_open']:.2f} | "
                  f"cv det {r['det_cv']:.2f} e2e {r['e2e_cv']:.2f} | both e2e {r['e2e_both']:.2f}", flush=True)
    reader_s = round(time.time() - t0)
    both = PlatePipeline(BothDetectors(*(p.detector for p in pipes.values())), ocr)
    # OCR speed, single plate crop, CPU
    crop = synth.plate_crop(rng, 128).image
    t = time.time()
    for _ in range(50):
        ocr.recognise(crop)
    ocr_ms = (time.time() - t) / 50 * 1000

    cal = vote_calibration(pipes["open"], a.seed + 1, 5 if a.quick else 40)
    print(f"[{time.time() - t0:5.0f}s] vote calibration (seed {cal['seed']}): {cal['bins']}", flush=True)

    scen = []
    for sc in SCENARIOS if not a.quick else SCENARIOS[:5]:
        t, row = time.time(), {}
        for k in ("open", "both"):                       # the rule is the same; only the plate read can differ
            d = tempfile.mkdtemp(prefix="plates_bench_")
            ps = PlateStore(d, now=1_800_000_000.0)
            r = run(sc, reader=both if k == "both" else pipes["open"], render=True, plate_store=ps, out_dir=d, seed=a.seed)
            row[k] = {"ok": r["ok"], "alerts": len(r["alerts"]), "retractions": len(r["retractions"]),
                      "plates": [{"truth": tr, "read": rd, "edit_distance": e} for tr, rd, e in r["plates"]],
                      "plates_left_in_store": ps.count()}
            ps.close()
        scen.append({"name": sc.name, "what": sc.what, "condition": sc.condition, "expected_alerts": len(sc.expect),
                     "expected_retractions": sc.retractions, **row["open"], "plates_both": row["both"]["plates"],
                     "ok": row["open"]["ok"] and row["both"]["ok"], "frames": r["frames"], "seconds": round(time.time() - t, 1)})
        print(f"[{time.time() - t0:5.0f}s] scenario {sc.name}: ok={scen[-1]['ok']} alerts={scen[-1]['alerts']} "
              f"retractions={scen[-1]['retractions']} open={row['open']['plates']} both={row['both']['plates']}", flush=True)

    res = {"synthetic": True, "seed": a.seed, "quick": a.quick, "reader": rows,
           "ocr_ms_per_crop": old["ocr_ms_per_crop"] if old else ocr_ms, "scenarios": scen, "vote_calibration": cal,
           "scenarios_only": a.scenarios_only, "rerun_seconds": round(time.time() - t0),
           "reader_seconds": old["reader_seconds"] if old else reader_s}
    res["seconds"] = round(time.time() - t0) + (old["reader_seconds"] if old else 0)
    (ROOT / "out/plates").mkdir(parents=True, exist_ok=True)
    name = "bench_quick.json" if a.quick else "bench.json"            # a quick run never replaces the full numbers
    (ROOT / "out/plates" / name).write_text(json.dumps(res, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(markdown(res))
    print(f"wrote {a.out} and out/plates/{name} in {res['seconds']} s")


def markdown(res: dict) -> str:
    r0 = res["reader"][0]
    L = ["# Plate reader and drive-off rule: measured on SYNTHETIC data", "",
         "Every plate is drawn with system fonts on a crude car shape and every scene is scripted "
         "(`bree.plates.synth`, `bree.plates.scenarios`). None of it is real footage, so these numbers show how "
         "the pipeline reacts to plate size, light and noise. They are not a forecast for a real forecourt.", "",
         f"Command: `.venv/bin/python scripts/plates/bench.py{' --quick' if res['quick'] else ''}` "
         f"(seed {res['seed']}, {res['seconds']} s)."
         + (f" The reader table below is from that full run ({res['reader_seconds']} s). The vote calibration and the "
            f"scripted timelines were rerun afterwards with `.venv/bin/python scripts/plates/bench.py --scenarios-only` "
            f"({res['rerun_seconds']} s), which keeps the reader table." if res.get("scenarios_only") else ""), "",
         "## Plate read accuracy by plate width and condition", "",
         f"Per cell: {r0['n_crop']} plate crops for the OCR columns, {r0['n_vote']} plates x 5 frames for the vote, "
         f"{r0['n_scene']} forecourt frames for the detector columns. Text reader: fast-plate-ocr `cct-s-v2-global-model`. "
         "`open` = open-image-models YOLOv9-t plate detector run inside the vehicle box, `cv` = the OpenCV-only localiser, `both` = candidates from both, keeping the one the text reader is most sure about. "
         "Plate found means a box with IoU of 0.4 or more with the true plate; a looser box can still be read, so whole plate can be above plate found.", "",
         "| plate width px | condition | OCR char | OCR whole plate | vote of 5 whole plate | open: plate found | open: whole plate | cv: plate found | cv: whole plate | both: whole plate |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["reader"]:
        L.append(f"| {r['width_px']} | {r['condition']} | {r['ocr_char']:.1%} | {r['ocr_plate']:.1%} | {r['vote5_plate']:.1%} | "
                 f"{r['det_open']:.1%} | {r['e2e_open']:.1%} | {r['det_cv']:.1%} | {r['e2e_cv']:.1%} | {r['e2e_both']:.1%} |")
    L += ["", "Conditions (brightness gain, sensor noise sigma in grey levels, motion blur as a fraction of plate width): "
          + "; ".join(f"{k}: {v['gain']}, {v['sigma']}, {v['blur']}" for k, v in synth.CONDITIONS.items()) + ".",
          f"Text reader speed: {res['ocr_ms_per_crop']:.1f} ms per plate crop on this laptop CPU while other jobs were running.", "",
          *calibration_md(res), "## Drive-off rule on scripted timelines", "",
          "Each timeline is rendered at 2 frames per second and played through `DriveOffMonitor` with plates read from "
          "pixels, once with the open plate detector and once with both detectors. Vehicle boxes are the scripted ones. Grace period 20 s here "
          "(default 120 s) so a timeline stays short. A plate is stored only when the vote over frames reaches confidence 0.9.", "",
          "| scenario | what happens | light | alerts expected | alerts raised | retractions expected | retractions | correct | plate truth -> read (open) | plate truth -> read (both) | plates left in store (open run) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["scenarios"]:
        pl, pl2 = (", ".join(f"{p['truth']} -> {'deleted' if s['retractions'] else p['read'] or 'not read'}"
                             for p in s[k]) or "n/a" for k in ("plates", "plates_both"))
        L.append(f"| {s['name']} | {s['what']} | {s['condition']} | {s['expected_alerts']} | {s['alerts']} | "
                 f"{s['expected_retractions']} | {s['retractions']} | {'yes' if s['ok'] else 'NO'} | {pl} | {pl2} | {s['plates_left_in_store']} |")
    n, ok = len(res["scenarios"]), sum(s["ok"] for s in res["scenarios"])
    L += ["", f"Timelines with the right alerts and retractions: {ok} of {n}."]
    for k, name in (("plates", "open detector"), ("plates_both", "both detectors")):
        want = [p for s in res["scenarios"] if not s["retractions"] for p in s[k]]
        L.append(f"Drive-off plates, {name}: {sum(p['edit_distance'] == 0 for p in want)} of {len(want)} read exactly right, "
                 f"{sum(p['read'] is None for p in want)} not read (nothing stored), "
                 f"{sum(p['read'] is not None and p['edit_distance'] > 0 for p in want)} stored wrong. "
                 "Retracted alerts have their plate deleted and are not counted.")
    L.append("")
    return "\n".join(L)


def calibration_md(res: dict) -> list[str]:
    cal = res.get("vote_calibration")
    if not cal:
        return []
    L = ["## How often a voted plate is right, by vote confidence", "",
         f"{cal['plates']} drawn plates with seed {cal['seed']} (not the seed of the table above): widths 48, 64 and 96 px, "
         "day, day with motion blur and night, 5 frames per plate, one vote per plate. The drive-off monitor stores a "
         "plate only at vote confidence 0.9 or more (`min_plate_conf`); this table is what that threshold rests on. "
         "The monitor votes over 3 to 12 frames, so 5 is a stand-in.", "",
         "| vote confidence | plates | voted plate right | share right |", "|---|---|---|---|"]
    for k, b in cal["bins"].items():
        L.append(f"| {k} | {b['n']} | {b['right']} | {b['right'] / b['n']:.0%} |" if b["n"] else f"| {k} | 0 | 0 | n/a |")
    return [*L, ""]


if __name__ == "__main__":
    main()
