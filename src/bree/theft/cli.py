"""`bree theft-*` commands. Registered from bree.commands.register.

  bree theft-cache   --station trumanrd [--channels 7,29]       detector + tracker once per segment -> perception cache
  bree theft-shadow  --station trumanrd --out out/theft_shadow  all detectors over the cached segments, one review queue
  bree theft-rates   --station trumanrd                         incidents per camera-hour with honest bounds
  bree theft-sim                                                POS-less bands in the simulator (SIMULATED)

Everything reads recorded files (`~/Projects/bree-data/real/<station>/chNN/*.mp4`, which stay out of git). Numbers printed are
labelled: REAL-UNLABELLED (ran on real footage, nobody has said which items are real) or SIMULATED.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path.home() / "Projects" / "bree-data"
ROOT = Path(__file__).resolve().parents[3]


def _segments(station: str, channels: list[str] | None, cache_root: Path):
    base = DATA / "real" / station
    for ch in sorted(base.glob("ch*")):
        if channels and ch.name not in channels:
            continue
        for mp4 in sorted(ch.glob("*.mp4")):
            cdir = cache_root / station / ch.name / mp4.stem
            if mp4.stat().st_size >= 1_000_000 and (cdir / "info.json").exists():
                yield ch.name, mp4, cdir


def _chans(arg):
    return [f"ch{int(c):02d}" for c in arg.split(",")] if arg else None


def _runs(args):
    from bree.theft.hours import StoreClock
    from bree.theft.shadow_run import run_segment
    from bree.theft.station import load_station
    root = Path(args.cache)
    stations, runs, videos, caches = {}, [], {}, {}
    for ch, mp4, cdir in _segments(args.station, _chans(args.channels), root):
        cfg = ROOT / "configs" / "stations" / args.station / f"{ch}.yaml"
        if not cfg.exists():
            continue
        st = stations.setdefault(ch, load_station(cfg, ch))
        clock = StoreClock.from_segment_name(mp4.name, args.clock_offset_s)
        r = run_segment(cdir, st, clock, camera=ch)
        runs.append(r)
        videos[(ch, r.segment)], caches[(ch, r.segment)] = mp4, cdir
        print(f"{ch}/{r.segment}: {len(r.incidents)} incident(s), persons raw {r.stats['persons_raw']} -> engine "
              f"{r.stats['persons_engine']}, events {r.stats['events']}", flush=True)
    return runs, stations, videos, caches


def cmd_theft_shadow(args) -> None:
    from bree.theft.shadow_run import rates, write_review
    runs, stations, videos, caches = _runs(args)
    log = write_review(runs, args.out, videos, caches, stations, top_clips=args.clips)
    r = rates(runs)
    Path(args.out, "rates.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r["all"], indent=2))
    print(f"{len(log.alerts())} item(s) in {args.out}; review: bree shadow-labels --dir {args.out} --serve")
    print("REAL-UNLABELLED: flagged items, not confirmed. Footage: " + str(round(r["camera_hours"], 2)) + " camera-hours.")


def cmd_theft_rates(args) -> None:
    from bree.theft.shadow_run import rates
    runs, *_ = _runs(args)
    print(json.dumps(rates(runs), indent=2))


def cmd_theft_sim(args) -> None:
    from bree.events.zones import load_store_config
    from bree.theft.eval_sim import posless_bench
    store = load_store_config(ROOT / "configs" / "store_gas_station_small.yaml")
    for k in (0.0, 1.0, 2.0):
        r = posless_bench(store, args.hours, args.seed, k)
        print(f"SIMULATED, vision noise x{k:g}: {r['thieves']} thieves among {r['thieves'] + r['honest']} visits "
              f"(simulator prevalence, far above a real store)")
        for b, v in r["bands"].items():
            print(f"  {b}: {v['per_hour']:.2f} items/h, precision {v['precision']}, cumulative thief recall {v['cum_thief_recall']}")


def register(sub) -> None:
    def common(p):
        p.add_argument("--station", default="trumanrd")
        p.add_argument("--channels", default=None, help="comma list of channel numbers (default all cached)")
        p.add_argument("--cache", default=str(DATA / "theft" / "cache"))
        p.add_argument("--clock-offset-s", type=float, default=None,
                       help="store local time minus the recorder clock of the segment names, in seconds. Unset = "
                            "after-hours rules stay off (the DVR clock offset of the pilot footage is unresolved)")
    s = sub.add_parser("theft-shadow", help="theft shadow mode over recorded real footage: all detectors, ranked review queue")
    common(s)
    s.add_argument("--out", default=str(ROOT / "out" / "theft_shadow"))
    s.add_argument("--clips", type=int, default=40, help="head-pixelated clips for the top N items")
    s.set_defaults(func=cmd_theft_shadow)
    r = sub.add_parser("theft-rates", help="incidents per camera-hour (REAL-UNLABELLED) with honest upper bounds")
    common(r)
    r.set_defaults(func=cmd_theft_rates)
    t = sub.add_parser("theft-sim", help="POS-less evidence bands in the simulator (SIMULATED)")
    t.add_argument("--hours", type=float, default=100)
    t.add_argument("--seed", type=int, default=2)
    t.set_defaults(func=cmd_theft_sim)
