"""Subcommands for stations: onboard, fleet, clock-check, record. Registered by bree.commands.register."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_DATA = Path.home() / "Projects" / "bree-data"


def _fleet(args):
    from bree.fleet.registry import Fleet
    return Fleet(args.fleet)


def _read_secret(env_var: str, file: str | None) -> str | None:
    if file:
        return Path(file).expanduser().read_text().strip()
    return os.environ.get(env_var) or None


def cmd_onboard(args) -> None:
    from bree.onboard.run import onboard
    from bree.onboard.stats import YoloScene
    fleet = _fleet(args)
    chans = [int(c) for c in args.channels.split(",")] if args.channels else None
    zres = tuple(int(v) for v in args.zone_res.lower().split("x")) if args.zone_res else None
    offset = args.clock_offset
    if args.from_dir:
        from bree.onboard.discover import FolderDiscovery
        disc = FolderDiscovery(args.from_dir)
        if offset is None and not args.no_clock:
            offset = _footage_clock(args.from_dir)
    else:
        from bree.ingest.imou import ImouClient, load_credentials
        from bree.onboard.discover import ImouDiscovery
        creds = load_credentials()
        client = ImouClient(creds, allow_bind=args.bind)
        if args.bind:
            pw = _read_secret("BREE_DEVICE_PASSWORD", args.password_file)
            if not pw:
                raise SystemExit("--bind needs the device password in $BREE_DEVICE_PASSWORD or --password-file (never on the command line)")
            client.bind_device(args.serial, pw)
            print("device bound to the Imou app")
        disc = ImouDiscovery(client, args.serial, seconds=args.seconds)
    rep = onboard(disc, YoloScene(), fleet, args.station, args.serial, chans, zone_res=zres, clock_offset_s=offset)
    d = fleet.dir(args.station)
    n_review = sum(1 for c in rep["cameras"] if c["suggestion"]["needs_review"])
    print(f"station folder: {d}\n  contact sheet: {d / 'contact_sheet.jpg'} (heads pixelated; local only)\n"
          f"  {len(rep['cameras'])} cameras, {n_review} need a human look; drafts in {d / 'cameras'}; correct roles with `bree fleet set-role`")


def _footage_clock(root) -> float | None:
    """DVR overlay clock minus file-name (edge box) clock from one segment of the first channel that reads."""
    from bree.ingest.overlay import check_file_clock, have_tesseract
    if not have_tesseract():
        return None
    for ch in sorted(Path(root).glob("ch*")):
        segs = sorted(ch.glob("*.mp4"))
        if segs:
            est = check_file_clock(segs[0], n=5)["estimate"]
            if est["ok"]:
                return round(est["offset_s"])
    return None


def cmd_clock_check(args) -> None:
    from bree.ingest.overlay import check_file_clock
    for f in args.file:
        r = check_file_clock(f, n=args.n)
        e = r["estimate"]
        line = f"{Path(f).parent.name}/{Path(f).name}: "
        line += (f"overlay minus file-name clock = {e['offset_s']:+.0f} s (+-{e['spread_s']:.1f}, {e['n_read']}/{e['n_tried']} reads), "
                 f"overlay runs {r['overlay_seconds_per_file_second']:.3f} s per file second" if e["offset_s"] is not None
                 else "overlay clock not readable (no overlay in this view, or tesseract missing)")
        print(line)


def cmd_fleet(args) -> None:
    fleet = _fleet(args)
    if args.action == "list":
        for st in fleet.stations():
            print(f"{st.id}  serial {st.serial}  {len(st.cameras)} cameras  clock offset {st.clock_offset_s}")
        return
    if args.action == "set-role":
        fleet.set_role(args.station, int(args.channel), args.role)
        print(f"{args.station} ch{int(args.channel):02d} -> {args.role} (human)")
        return
    sts = [fleet.load(args.station)] if args.station else fleet.stations()
    if not sts:
        raise SystemExit(f"no stations in {fleet.root}")
    bad = 0
    for st in sts:
        if args.action == "status":
            h = fleet.station_health(st)
            age = h["last_frame_age_s"]
            print(f"{st.id}: {h['overall']}  last frame {'never' if age is None else f'{age:.0f} s ago'}")
            for c in h["cameras"]:
                print(f"  ch{c['channel']:02d} {c['role']:9s} {c['state']:12s}" + (f" age {c['last_frame_age_s']}s" if c.get("last_frame_age_s") is not None else "")
                      + (f"  {c['last_error']}" if c.get("last_error") else ""))
            bad += h["overall"] not in ("ok", "no_cameras")
        else:  # check
            for r in fleet.check(st, args.data):
                print(f"  [{'ok' if r['ok'] else 'FAIL'}] {st.id}: {r['name']}  {r['detail']}")
                bad += not r["ok"]
            if args.live:
                from bree.ingest.imou import ImouClient, ImouError, load_credentials
                try:
                    ds = ImouClient(load_credentials()).device_online(st.serial)
                    off = [c for c, ok in ds.channels.items() if not ok]
                    print(f"  [{'ok' if ds.online else 'FAIL'}] {st.id}: DVR {'online' if ds.online else 'OFFLINE'}; offline channels: {off or 'none'}")
                    bad += not ds.online
                except ImouError as e:
                    print(f"  [FAIL] {st.id}: platform check failed: {e}")
                    bad += 1
    sys.exit(1 if bad else 0)


def cmd_imou_mock(args) -> None:
    import time
    from bree.ingest.imou_mock import APP_ID, APP_SECRET, MockImou
    with MockImou(channels={int(c): "ok" for c in args.channels.split(",")}, video=args.video) as m:
        print(f"mock Imou platform up (local, no network, no real device). In another shell:\n"
              f"  export IMOU_APP_ID={APP_ID} IMOU_APP_SECRET={APP_SECRET} IMOU_BASE_URL={m.base_url}\n"
              f"  bree run --source imou://{m.serial}/{args.channels.split(',')[0]} --backend toy --no-video --max-frames 50\n"
              "Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def cmd_record(args) -> None:
    from bree.ingest.record import main as record_main
    record_main(args)


def register(sub) -> None:
    o = sub.add_parser("onboard", help="new station: list channels, stills + contact sheet, role suggestions, draft configs")
    o.add_argument("--serial", required=True, help="DVR serial number")
    o.add_argument("--station", required=True, help="station id (lowercase letters, digits, - _)")
    o.add_argument("--from-dir", default=None, help="onboard from recorded footage <dir>/chNN/*.mp4 instead of the live DVR "
                   "(live sampling uses the Imou platform's metered flow: about 20 s of sub stream per channel)")
    o.add_argument("--fleet", default=None, help="fleet root (default $BREE_FLEET or ~/.local/share/bree/fleet)")
    o.add_argument("--channels", default=None, help="comma list; default all online channels")
    o.add_argument("--zone-res", default=None, help="WxH of the stream the zones are for (default: the sampled stream)")
    o.add_argument("--clock-offset", type=float, default=None, help="DVR clock minus edge-box clock in seconds (default: read from the overlay when footage is given)")
    o.add_argument("--no-clock", action="store_true", help="skip the overlay clock reading")
    o.add_argument("--seconds", type=float, default=20.0, help="live sample length per channel")
    o.add_argument("--bind", action="store_true", help="first attach the DVR to the Imou app (state-changing; password from $BREE_DEVICE_PASSWORD or --password-file)")
    o.add_argument("--password-file", default=None, help="file holding the device password (never pass it as an argument)")
    o.set_defaults(func=cmd_onboard)

    c = sub.add_parser("clock-check", help="read the DVR overlay clock in recorded segments and compare it with the file-name clock")
    c.add_argument("file", nargs="+")
    c.add_argument("--n", type=int, default=6)
    c.set_defaults(func=cmd_clock_check)

    f = sub.add_parser("fleet", help="stations: list | status | check [--live] | set-role")
    f.add_argument("action", choices=["list", "status", "check", "set-role"])
    f.add_argument("--fleet", default=None)
    f.add_argument("--station", default=None)
    f.add_argument("--channel", default=None)
    f.add_argument("--role", default=None)
    f.add_argument("--data", default=str(DEFAULT_DATA), help="footage root for the privacy check")
    f.add_argument("--live", action="store_true", help="also ask the platform whether the DVR is online (deviceOnline, no stream)")
    f.set_defaults(func=cmd_fleet)

    m = sub.add_parser("imou-mock", help="run a fake Imou platform locally (tests and offline demos of onboard/run/record)")
    m.add_argument("--channels", default="0,1")
    m.add_argument("--video", default=None, help="mp4 to serve on every channel (default: a generated test pattern)")
    m.set_defaults(func=cmd_imou_mock)

    r = sub.add_parser("record", help="record rolling segments from a station's DVR (read-only Imou APIs, stops on a stub stream)")
    r.add_argument("--station", required=True)
    r.add_argument("--serial", default=None, help="default: from the fleet registry")
    r.add_argument("--fleet", default=None)
    r.add_argument("--data", default=str(DEFAULT_DATA))
    r.add_argument("--channels", default=None, help="comma list (at most 8); default: enabled cameras of the station")
    r.add_argument("--stream", choices=["main", "sub"], default="main")
    r.add_argument("--seg-seconds", type=int, default=300)
    r.add_argument("--hours", type=float, default=12)
    r.add_argument("--cap-gb", type=float, default=80)
    r.set_defaults(func=cmd_record)
