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


def cmd_bench(args) -> None:
    from bree.eval.bench import run_bench
    parts = tuple(args.only.split(",")) if args.only else ("event", "toy", "real")
    run_bench(ROOT, quick=args.quick, parts=parts)


def cmd_sim(args) -> None:
    from bree.eval.metrics import run_event_bench
    from bree.events.zones import load_store_config
    from bree.sim.events_sim import SimConfig, VisionNoise
    store = load_store_config(STORE)
    print(f"event-level simulator: {args.hours:.0f} store hours, seed {args.seed}, payment mode {args.mode}")
    print(f"{'vision noise':14s} {'thieves':>7s} {'precision':>9s} {'recall':>7s} {'FA/hour':>8s} {'basket ok':>9s}")
    for name, k in (("perfect", 0.0), ("baseline", 1.0), ("2x errors", 2.0)):
        r = run_event_bench(store, SimConfig(hours=args.hours, payment_mode=args.mode), VisionNoise().scaled(k), args.seed)
        print(f"{name:14s} {r['thieves']:7d} {r['precision']:9.3f} {r['recall']:7.3f} "
              f"{r['false_alerts_per_hour']:8.2f} {r['basket_exact_match']:9.3f}")


def cmd_dashboard(args) -> None:
    import threading
    import time
    from bree.cli import make_backend
    from bree.dashboard.server import DashboardState, serve
    from bree.events.zones import load_store_config
    from bree.ledger.payments import open_payments
    from bree.pipeline import run_pipeline
    store = load_store_config(args.store)
    source, payments, backend = args.source, args.payments, args.backend
    if source is None:   # default: a toy clip with its POS feed, replayed at camera speed
        ensure_toy()
        source = str(TOY_DIR / "toy_conceal_partial_pay.mp4")
        payments = payments or str(TOY_DIR / "toy_conceal_partial_pay.payments.jsonl")
        backend = "toy"
    state = DashboardState(source)
    httpd = serve(state, args.host, args.port)
    print(f"dashboard: http://{args.host}:{httpd.server_address[1]}  (source: {source})")

    def work():
        run_pipeline(source, store, make_backend(backend, store), args.out,
                     payments=open_payments(payments, stream_start_wall=time.time()),
                     on_alert=state.on_alert, on_frame=state.on_frame, verbose=False, realtime=True)
        state.done = True
    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        httpd.shutdown()


def cmd_export(args) -> None:
    from bree.edge.export import export_and_benchmark
    print(json.dumps(export_and_benchmark(ROOT, args.imgsz), indent=2))


def register(sub) -> None:
    db = sub.add_parser("dashboard", help="local web dashboard: live alerts + baskets")
    db.add_argument("--source", default=None, help="video/webcam/rtsp; default: a toy clip")
    db.add_argument("--backend", choices=["yolo", "toy"], default="yolo")
    db.add_argument("--payments", default=None)
    db.add_argument("--store", default=str(STORE))
    db.add_argument("--out", default=str(ROOT / "out" / "dashboard"))
    db.add_argument("--host", default="127.0.0.1")
    db.add_argument("--port", type=int, default=8080)
    db.set_defaults(func=cmd_dashboard)

    ex = sub.add_parser("export", help="export YOLO models to ONNX and compare CPU latency")
    ex.add_argument("--imgsz", type=int, default=640)
    ex.set_defaults(func=cmd_export)

    b = sub.add_parser("bench", help="benchmark -> results/bench.json + results/bench.md")
    b.add_argument("--quick", action="store_true", help="50 sim hours, 200 real frames")
    b.add_argument("--only", default=None, help="comma list of: event,toy,real")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("sim", help="event-level simulator: ledger accuracy under vision noise")
    s.add_argument("--hours", type=float, default=100)
    s.add_argument("--seed", type=int, default=2)
    s.add_argument("--mode", choices=["pos", "dwell"], default="pos")
    s.set_defaults(func=cmd_sim)

    r = sub.add_parser("render-toy", help="render labelled toy clips")
    r.add_argument("--store", default=str(STORE))
    r.add_argument("--out", default=str(TOY_DIR))
    r.set_defaults(func=cmd_render_toy)

    d = sub.add_parser("demo", help="toy clips end to end with a scorecard")
    d.add_argument("--out", default=str(ROOT / "out" / "demo"))
    d.set_defaults(func=cmd_demo)
