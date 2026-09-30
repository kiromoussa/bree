"""Subcommands beyond `hw` / `run`. Each one imports its heavy modules lazily."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "configs" / "store_gas_station_small.yaml"
TOY_DIR = ROOT / "data" / "toy"


def cmd_render_toy(args) -> None:
    from bree.events.zones import load_store_config
    from bree.sim.toy_render import render_all
    store = load_store_config(args.store)
    for r in render_all(store, args.out):
        t = r["truth"]
        print(f"  {r['video']}: {t['frames']} frames, {t['description']}")


def ensure_toy(store_path=STORE, toy_dir=TOY_DIR) -> None:
    if not list(Path(toy_dir).glob("toy_*.mp4")):
        print(f"rendering toy clips into {toy_dir} ...")
        cmd_render_toy(type("A", (), {"store": str(store_path), "out": str(toy_dir)}))


def print_toy_table(rows: list[dict]) -> None:
    print(f"\n{'clip':22s} {'who':4s} {'truth':18s} {'system':8s} {'conf':>5s}  flagged")
    for r in rows:
        truth = "THEFT " + ",".join(r["stolen"]) if r["thief"] else "honest"
        print(f"{r['clip']:22s} {r['person']:4s} {truth:18s} {r['tier'] or '-':8s} "
              f"{'' if r['confidence'] is None else format(r['confidence'], '.2f'):>5s}  {','.join(r['flagged_items'])}")


def cmd_demo(args) -> None:
    from bree.eval.toy_eval import run_toy_suite, score_rows
    ensure_toy()
    print("TOY DATA demo: rendered 2D clips -> toy colour detector -> ByteTrack -> event engine -> ledger -> alerts")
    rows, summaries = run_toy_suite(TOY_DIR, args.out, STORE, verbose=False)
    print_toy_table(rows)
    sc = score_rows(rows)
    fps = sum(s.frames for s in summaries) / sum(s.wall_s for s in summaries)
    print(f"\nscore: {json.dumps(sc)}")
    print(f"pipeline FPS on toy clips (toy detector, incl. rendering annotated video): {fps:.1f}")
    print(f"outputs (annotated videos, alerts/*.json + evidence clips, ledger logs): {args.out}/<clip>/")


def register(sub) -> None:
    r = sub.add_parser("render-toy", help="render labelled toy clips")
    r.add_argument("--store", default=str(STORE))
    r.add_argument("--out", default=str(TOY_DIR))
    r.set_defaults(func=cmd_render_toy)

    d = sub.add_parser("demo", help="toy clips end to end with a scorecard")
    d.add_argument("--out", default=str(ROOT / "out" / "demo"))
    d.set_defaults(func=cmd_demo)
