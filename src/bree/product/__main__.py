"""python -m bree.product <command>

  demo      write SIMULATED signals, review items and a station file into a folder (for screenshots and trials)
  serve     the owner dashboard over a data folder
  ingest    turn one `bree run` output (frames.jsonl) into signals in a data folder
  digest    write a daily or weekly digest as an HTML file
  notify    replay the log through the notification rules and print what would fire (nothing is sent)

Data folder layout:  <data>/station.yaml  <data>/signals/signals-YYYYMMDD.jsonl  <data>/review/ (ReviewStore)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from bree.product import app as appmod
from bree.product import metrics
from bree.product.signals import SignalLog, day_bounds, local_date
from bree.product.station import load_station


def _open(args):
    data = Path(args.data)
    st = load_station(args.station or (data / "station.yaml" if (data / "station.yaml").is_file() else None))
    log = SignalLog(data / "signals", st.tz)
    review = None
    if (data / "review" / "review.sqlite").is_file() or args.cmd == "serve" and getattr(args, "with_review", False):
        from bree.review.store import ReviewStore
        review = ReviewStore(data / "review")
    return data, st, log, review


def cmd_demo(args):
    from bree.product.sim import populate_log
    from bree.review.synthetic import BASE, populate
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "station.yaml").write_text(
        "# SIMULATED demo station. Nothing here describes a real store.\n"
        "name: Demo Convenience Store\nstore_id: demo-store\ntz: America/Chicago\ntz_assumed: true\n"
        "notes: [\"All numbers in this folder are SIMULATED.\"]\n"
        "cameras:\n  ch07: {label: Front door, role: door}\n  ch15: {label: Coolers, role: aisle}\n"
        "  ch14: {label: Snack aisle, role: aisle}\n  ch10: {label: Coffee and fountain, role: aisle}\n"
        "  ch29: {label: Register, role: counter}\n")
    start = dt.datetime.fromtimestamp(BASE, dt.timezone.utc).date()
    populate_log(out / "signals", start, args.days, args.seed, "America/Chicago", "demo-store")
    populate(out / "review", n=args.alerts, seed=args.seed)
    print(f"wrote simulated data to {out}\nview it:  python -m bree.product serve --data {out} --replay")


def cmd_serve(args):
    data, st, log, review = _open(args)
    if review is None and (data / "review").is_dir():
        from bree.review.store import ReviewStore
        review = ReviewStore(data / "review")
    app = appmod.ProductApp(log, st, review, replay=args.replay)
    srv = appmod.serve(app, args.host, args.port, args.token)
    print(f"BREE store view on http://{args.host}:{args.port}/" + (f"?token={args.token}" if args.token else "")
          + ("  (replay mode: 'now' is the newest saved data)" if args.replay else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ingest(args):
    from bree.events.zones import load_store_config
    from bree.product.adapt import AdaptConfig, adapt_run, segment_start
    data = Path(args.data)
    st = load_station(args.station or (data / "station.yaml" if (data / "station.yaml").is_file() else None))
    log = SignalLog(data / "signals", st.tz)
    start = segment_start(args.segment) + st.clock_offset_s if args.start_ts is None else args.start_ts
    cfg = AdaptConfig(store=st.store_id, camera=args.camera, role=args.role, source=args.source,
                      queue_zone=args.queue_zone, staff_zone=args.staff_zone)
    sc = load_store_config(args.store_yaml) if args.store_yaml else None
    sigs, rep = adapt_run(args.frames, start, cfg, sc)
    log.append(sigs)
    print(rep)


def cmd_digest(args):
    data, st, log, review = _open(args)
    app = appmod.ProductApp(log, st, review, replay=True)
    html = app.weekly_html(args.date) if args.weekly else app.daily_html(args.date)
    Path(args.out).write_text(html)
    print("wrote", args.out)


def cmd_notify(args):
    from bree.product.notify import Notifier
    data, st, log, _ = _open(args)
    n = Notifier(st.rules, [])
    sigs = log.read()
    fired = n.feed(sigs)
    tz = log.tz
    for f in fired:
        print(f"{dt.datetime.fromtimestamp(f.ts, tz):%Y-%m-%d %H:%M} [{f.severity}] {f.rule}: {f.message}")
    print(f"{len(fired)} notification(s) would fire over {len(sigs)} signals. Nothing was sent.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bree.product", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="write SIMULATED demo data")
    d.add_argument("--out", required=True); d.add_argument("--days", type=int, default=21)
    d.add_argument("--alerts", type=int, default=40); d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("serve", help="run the dashboard")
    s.add_argument("--data", required=True); s.add_argument("--station")
    s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8090)
    s.add_argument("--token", help="required for any non-local host")
    s.add_argument("--replay", action="store_true", help="treat the newest saved signal as 'now' (for saved data)")
    s.add_argument("--with-review", action="store_true", help="create an empty review store if none exists")
    s.set_defaults(fn=cmd_serve)
    i = sub.add_parser("ingest", help="signals from one pipeline run (frames.jsonl)")
    i.add_argument("--data", required=True); i.add_argument("--station")
    i.add_argument("--frames", required=True, help="frames.jsonl from `bree run`")
    i.add_argument("--segment", required=True, help="the video file name (YYYYmmdd_HHMMSS in it gives the start time)")
    i.add_argument("--start-ts", type=float, help="epoch start, overrides --segment")
    i.add_argument("--camera", required=True); i.add_argument("--role", default="aisle", choices=["door", "counter", "aisle"])
    i.add_argument("--store-yaml"); i.add_argument("--queue-zone"); i.add_argument("--staff-zone")
    i.add_argument("--source", default="real_unlabelled", choices=["simulated", "real_unlabelled", "real_model_labelled", "real_ground_truth", "live"])
    i.set_defaults(fn=cmd_ingest)
    g = sub.add_parser("digest", help="write a digest HTML file")
    g.add_argument("--data", required=True); g.add_argument("--station")
    g.add_argument("--date", help="day (daily) or last day (weekly), YYYY-MM-DD; default the newest data")
    g.add_argument("--weekly", action="store_true"); g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_digest)
    n = sub.add_parser("notify", help="dry-run the notification rules over the log")
    n.add_argument("--data", required=True); n.add_argument("--station")
    n.set_defaults(fn=cmd_notify)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
