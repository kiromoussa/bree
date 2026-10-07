"""Every take the concealment cue looked at on rendered clips, with the truth beside it (SIMULATED clips).

  .venv/bin/python scripts/conceal/rows.py data/synth/bench/train 4900 4901 ... --runs out/bench/train --looks out/conceal/train --out out/conceal/train/rows.jsonl

Truth is read here only, to label rows for fitting (TRAIN-seed clips) and for scoring. A row is positive when the
shopper under that floor track hid an item inside the take's window (the item mesh disappears 0.49 s before tConceal).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bree.concealment.cue import ConcealConfig, analyse, load_looks, load_model  # noqa: E402
from bree.concealment.run import context  # noqa: E402

HIDE_BEFORE_S = 0.49        # agents.js: the item mesh is hidden at 0.65 of the 1.4 s gesture, tConceal is its end


def truth_of(clip: Path):
    tracks = {}
    for line in (clip / "truth/tracks.jsonl").read_text().splitlines():
        r = json.loads(line)
        tracks[round(r["t"], 1)] = {s["shopper"]: (s["x"], s["z"], s.get("state")) for s in r["shoppers"]}
    events = [json.loads(x) for x in (clip / "truth/events.jsonl").read_text().splitlines()]
    shoppers = {s["shopper"]: s for s in json.loads((clip / "truth/shoppers.json").read_text())}
    return tracks, events, shoppers


def shopper_at(tracks, xz, t: float, within: float = 1.0):
    row = tracks.get(round(t, 1)) or {}
    d = sorted((math.hypot(x - xz[0], z - xz[1]), n, st) for n, (x, z, st) in row.items())
    return (d[0][1], d[0][2]) if d and d[0][0] <= within else (None, None)


def rows_of(clip: Path, run: Path, looks_dir: Path, cfg=None, model=None) -> list[dict]:
    ctx = context(clip, run)
    tracks, events, shoppers = truth_of(clip)
    by = {p.id: p for p in ctx["people"]}
    out = []
    for k in analyse(load_looks(looks_dir), ctx["calib"], ctx["people"], ctx["acts"], ctx["who"], ctx["layout"], ctx["fps"], cfg or ConcealConfig(), model):
        got = by[k.person_id].at(k.t)
        name, _ = shopper_at(tracks, got[0], k.t) if got is not None else (None, None)
        pick = min((e for e in events if e["shopper"] == name), key=lambda e: abs(e["t"] - k.t), default=None)
        pick = pick if pick is not None and abs(pick["t"] - k.t) <= 3.0 else None
        hid = [e for e in events if e["shopper"] == name and e.get("tConceal") is not None and k.t < e["tConceal"] - HIDE_BEFORE_S <= k.stop + 1.0]
        out.append({"clip": clip.name, "person_id": k.person_id, "shopper": name, "thief": bool(name and shoppers[name]["thief"]), "t": round(k.t, 2), "sku": k.sku,
                    "stop": round(k.stop, 2), "stop_why": k.stop_why, "where": k.where, "label": int(bool(hid)), "t_hide": round(hid[0]["tConceal"] - HIDE_BEFORE_S, 2) if hid else None,
                    "truth_pick": pick["outcome"] if pick else None, "p": round(k.p, 4), **{a: (round(float(v), 4) if isinstance(v, (int, float)) else v) for a, v in k.feats.items()}})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", help="folder with clip_<seed> folders")
    ap.add_argument("seeds", nargs="+")
    ap.add_argument("--runs", required=True, help="folder with clip_<seed>/pipeline (stored shelf events and person boxes)")
    ap.add_argument("--looks", help="folder with clip_<seed>/looks_*.jsonl (default: <runs>/clip_<seed>/pipeline/conceal, where bree.concealment.run leaves them)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-model", action="store_true")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        rows += rows_of(Path(a.clips) / f"clip_{s}", Path(a.runs) / f"clip_{s}", (Path(a.looks) / f"clip_{s}") if a.looks else Path(a.runs) / f"clip_{s}/pipeline/conceal", model=None if a.no_model else load_model())
        print(f"clip {s}: {len(rows)} rows so far", flush=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
