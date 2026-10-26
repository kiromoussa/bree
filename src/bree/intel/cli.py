"""`bree intel ...`: people-flow analytics from recorded segments (door counts, occupancy, dwell, queue, staff, outcomes).

All numbers are of the footage given. Door counts were checked against REAL-MODEL-LABELLED door events on one store camera
(docs/ultra/intel.md); every other metric is REAL-UNLABELLED or SIMULATED (tests) and says so in its docs.
"""
from __future__ import annotations

import json
from pathlib import Path


def cmd_run(args) -> None:
    from bree.intel.run import run_segments
    flow = run_segments(args.video, args.layout, args.out, dets_dir=args.dets_dir, stride=args.stride, imgsz=args.imgsz, learn_mask=args.learn_mask)
    n = sum(1 for _ in flow.open())
    print(f"{n} records -> {flow}")


def cmd_report(args) -> None:
    from bree.intel import aggregate as ag
    recs = ag.read_records(args.flow)
    cams = sorted({r.get("camera", "") for r in recs})
    if len(cams) > 1:       # several cameras: one report folder per camera
        for c in cams:
            sub = [r for r in recs if r.get("camera", "") == c]
            tmp = Path(args.out) / (c or "camera")
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "flow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub))
            args2 = type(args)(**{**vars(args), "flow": [str(tmp / "flow.jsonl")], "out": str(tmp)})
            print(f"--- camera {c}")
            cmd_report(args2)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hours = ag.hourly(recs)
    days = ag.daily(hours)
    ag.to_json(hours, out / "hourly.json")
    ag.to_json(days, out / "daily.json")
    from bree.intel.anomaly import rule_anomalies
    anomalies = rule_anomalies(recs)
    (out / "anomalies.json").write_text(json.dumps(anomalies, indent=1))
    for day, d in days.items():
        text = ag.digest(d)
        if anomalies:
            text += "\n- Flags (rules, advisory): " + "; ".join(f"{a['kind']} {a['detail']}" for a in anomalies[:5])
        (out / f"digest_{day}.txt").write_text(text + "\n")
        print(text)
    if args.background:
        import cv2
        import numpy as np
        from bree.intel.layout import load_layout
        from bree.intel.render import draw_layout, heatmap_overlay, median_background
        heat_path = Path(args.flow[0]).parent / "heat.npz"
        if heat_path.exists():
            heat = np.load(heat_path)
            bg = median_background(args.background)
            L = load_layout(args.layout, (1280, 720)) if args.layout else None
            for role in heat.files:
                img = heatmap_overlay(bg, heat[role])
                cv2.imwrite(str(out / f"heat_{role}.png"), draw_layout(img, L) if L else img)
                if L is not None:
                    from bree.intel.render import plan_view
                    pv = plan_view(heat[role], load_layout(args.layout, (2560, 1440)))
                    if pv is not None:
                        cv2.imwrite(str(out / f"plan_{role}.png"), pv)
            print(f"heatmaps in {out} (background is a per-segment median; look before sharing)")


def cmd_anomalies(args) -> None:
    from bree.intel import aggregate as ag
    from bree.intel.anomaly import Baseline, rule_anomalies
    recs = ag.read_records(args.flow)
    found = rule_anomalies(recs)
    if args.history:
        hist = []
        for p in args.history:
            hist += list(json.loads(Path(p).read_text()).values())
        base = Baseline.fit(hist, args.min_history)
        for h in ag.hourly(recs).values():
            found += base.score(h)
    print(json.dumps(found, indent=1))


def cmd_eval(args) -> None:
    from bree.intel import aggregate as ag, evaluate as ev
    recs = ag.read_records(args.flow)
    res = ev.score(ev.by_segment(recs), ev.load_labels(args.labels), args.tol)
    print(json.dumps({k: v for k, v in res.items() if k != "detail"}, indent=1))
    if args.detail:
        print(json.dumps(res["detail"], indent=1))


def cmd_activity(args) -> None:
    from bree.intel.labeltools import door_activity
    for v in args.video:
        sp = door_activity(v, tuple(float(x) for x in args.roi.split(",")), thresh=args.thresh)
        print(Path(v).name, [(round(a), round(b)) for a, b in sp])


def cmd_counts(args) -> None:
    from bree.intel.detections import read_detections
    from bree.intel.flow import IntelConfig, run_detections
    from bree.intel.labelcounts import load_counts, score_counts
    from bree.intel.layout import load_layout
    from bree.intel.run import segment_start
    from bree.intel.staticmask import learn_static_mask
    import numpy as np
    rows = load_counts(args.labels)
    logs, size = {}, None
    for v in args.video:
        stem = Path(v).stem
        d = Path(args.dets_dir) / (stem + ".jsonl")
        if not d.exists():
            continue
        meta, frames = read_detections(d)
        size = (meta["width"], meta["height"])
        layout = load_layout(args.layout, size)
        if args.learn_mask:
            layout.ignore = list(layout.ignore) + [np.array(p) for p in learn_static_mask(frames, size)]
        _, eng = run_detections(frames, layout, IntelConfig(), segment_start(v, layout.clock_offset_s), keep_frames=True)
        logs[stem] = eng.frame_log
    print(json.dumps(score_counts(logs, rows, size, include_staff=True), indent=1))


def register(sub) -> None:
    p = sub.add_parser("intel", help="people-flow analytics from recorded segments (counts, dwell, queue, staff, daily digest)")
    s = p.add_subparsers(dest="intel_cmd", required=True)

    r = s.add_parser("run", help="detect people in segments, track, and write flow.jsonl + heat.npz")
    r.add_argument("video", nargs="+")
    r.add_argument("--layout", required=True, help="flow layout YAML (see configs/intel_trumanrd_ch07.yaml)")
    r.add_argument("--out", required=True)
    r.add_argument("--dets-dir", help="cache for detections (default <out>/dets)")
    r.add_argument("--stride", type=int, default=5, help="analyse every Nth frame (default 5: 6 fps from a 30 fps source)")
    r.add_argument("--imgsz", type=int, default=960)
    r.add_argument("--learn-mask", action="store_true", help="ignore spots where low-confidence boxes sit still all segment (posters, displays, reflections)")
    r.set_defaults(func=cmd_run)

    g = s.add_parser("report", help="hourly / daily aggregates, digest text and heatmaps from flow.jsonl")
    g.add_argument("flow", nargs="+")
    g.add_argument("--out", required=True)
    g.add_argument("--background", help="video to take a person-free background from, for heatmap images")
    g.add_argument("--layout", help="layout YAML, to outline the zones on the heatmaps")
    g.set_defaults(func=cmd_report)

    e = s.add_parser("eval", help="score door events in flow.jsonl against labelled door events")
    e.add_argument("flow", nargs="+")
    e.add_argument("--labels", required=True)
    e.add_argument("--tol", type=float, default=4.0, help="match tolerance in seconds")
    e.add_argument("--detail", action="store_true")
    e.set_defaults(func=cmd_eval)

    n = s.add_parser("anomalies", help="rule flags (queue, unattended counter, loitering, door imbalance) and, with --history, hour-of-week pattern flags")
    n.add_argument("flow", nargs="+")
    n.add_argument("--history", nargs="*", help="hourly.json files from earlier days/weeks")
    n.add_argument("--min-history", type=int, default=3)
    n.set_defaults(func=cmd_anomalies)

    c = s.add_parser("counts", help="score people-in-view counts against agent-made count labels (REAL-MODEL-LABELLED, not ground truth)")
    c.add_argument("video", nargs="+")
    c.add_argument("--layout", required=True)
    c.add_argument("--labels", required=True)
    c.add_argument("--dets-dir", required=True)
    c.add_argument("--learn-mask", action="store_true")
    c.set_defaults(func=cmd_counts)

    a = s.add_parser("activity", help="labelling helper: intervals where the door area changes (detector independent)")
    a.add_argument("video", nargs="+")
    a.add_argument("--roi", default="0.12,0.18,0.45,0.64", help="normalised x0,y0,x1,y1")
    a.add_argument("--thresh", type=float, default=2.5)
    a.set_defaults(func=cmd_activity)
