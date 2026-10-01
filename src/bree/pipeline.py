"""End-to-end runner: video -> detect/pose -> track -> events -> ledger -> alerts.

Outputs in `out_dir`:
  frames.jsonl     per-frame log: person boxes, track IDs, keypoints, products, events
                   (no images, no faces)
  events.jsonl     store events
  alerts.jsonl     alerts (+ alerts/<id>.json and alerts/<id>.mp4 evidence clips);
                   a late receipt can add a retraction record (`retracts` = earlier alert id)
  annotated.mp4    debug video with zones, boxes, IDs, skeletons (heads pixelated)
  summary.json     counts, FPS, per-stage timings, latencies

Several cameras of one store (`run_store`): frames are processed in camera-time order,
each camera has its own tracker and zones, floor-plane handoff gives one person id across
cameras, and ONE ledger reconciles the store. `frames_<camera>.jsonl` and
`annotated_<camera>.mp4` are per camera; everything else is per store.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.annotate import draw_frame
from bree.alerts.writer import AlertSink, EvidenceBuffer, MultiEvidence
from bree.detect.base import PerceptionBackend
from bree.events.observations import FrameObs
from bree.events.types import Event, EventType
from bree.events.zones import StoreConfig, merge_stores
from bree.ingest.source import VideoSource
from bree.ledger import build_ledger
from bree.ledger.payments import PaymentSource, MockPayments
from bree.track.multicam import StoreEvents


@dataclass
class RunSummary:
    source: str
    backend: str
    frames: int = 0
    video_fps: float = 0.0
    wall_s: float = 0.0
    pipeline_fps: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)      # mean ms per frame per stage
    persons_tracked: int = 0
    events: dict[str, int] = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    decision_latency_s: list[float] = field(default_factory=list) # camera time: exit -> alert
    processing_latency_ms: list[float] = field(default_factory=list)  # wall: frame read -> alert written
    handoffs: int = 0                                             # multi-camera: tracks linked across cameras


@dataclass
class CameraInput:
    name: str             # unique per store; used in per-camera file names and event meta
    source: str           # video file, webcam index, or rtsp:// URL
    store: StoreConfig    # this camera's zones (its pixels), terminals, catalog, floor_points


class _Cam:
    """One camera's runtime state: video, tracker, evidence frames, per-camera logs."""

    def __init__(self, cam: CameraInput, out: Path, suffix: str, max_frames: int | None, tracker_cls):
        self.name, self.store = cam.name, cam.store
        self.src = VideoSource(cam.source, max_frames=max_frames)
        self.frames = iter(self.src)
        self.tracker = tracker_cls(self.src.fps)
        self.evidence = EvidenceBuffer(self.src.fps)
        self.frames_log = (out / f"frames{suffix}.jsonl").open("w")
        self.video_path = out / f"annotated{suffix}.mp4"
        self.writer = None
        self.recent: list[Event] = []


def _obs_to_json(obs: FrameObs, events: list[Event]) -> dict:
    return {
        "frame": obs.frame, "t": round(obs.t, 3),
        "persons": [{"id": p.track_id, "bbox": [round(v, 1) for v in p.bbox], "conf": round(p.conf, 3),
                     "kpts": None if p.keypoints is None else np.round(p.keypoints, 1).tolist()}
                    for p in obs.persons],
        "products": [{"id": q.track_id, "bbox": [round(v, 1) for v in q.bbox], "cat": q.category,
                      "conf": round(q.conf, 3)} for q in obs.products],
        "events": [e.to_dict() for e in events],
    }


def run_pipeline(source: str, store: StoreConfig, backend: PerceptionBackend, out_dir: str | Path,
                 payments: PaymentSource | None = None, save_video: bool = True,
                 max_frames: int | None = None, ledger_overrides: dict | None = None,
                 on_alert=None, on_frame=None, verbose: bool = True, realtime: bool = False,
                 log_frames: bool = True) -> RunSummary:
    """One camera."""
    return run_store([CameraInput(store.camera_id, source, store)], backend, out_dir, payments=payments,
                     save_video=save_video, max_frames=max_frames, ledger_overrides=ledger_overrides,
                     on_alert=on_alert, on_frame=on_frame, verbose=verbose, realtime=realtime,
                     log_frames=log_frames)


def run_store(cameras: list[CameraInput], backend: PerceptionBackend, out_dir: str | Path,
              payments: PaymentSource | None = None, save_video: bool = True,
              max_frames: int | None = None, ledger_overrides: dict | None = None,
              on_alert=None, on_frame=None, verbose: bool = True, realtime: bool = False,
              log_frames: bool = True, handoff: dict | None = None) -> RunSummary:
    """One store, one or more cameras, one ledger. `handoff`: MultiCamIdentity settings
    (max_dist_m, max_gap_s). Several cameras need `camera.floor_points` in every store YAML."""
    from bree.track.bytetrack import Tracker   # ultralytics: only imported when video actually runs
    if len({c.name for c in cameras}) != len(cameras):
        raise ValueError("camera names must be unique")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    multi = len(cameras) > 1
    fusion = StoreEvents({c.name: c.store for c in cameras}, handoff)
    ledger = build_ledger(merge_stores([c.store for c in cameras]), **(ledger_overrides or {}))
    cams = [_Cam(c, out, f"_{c.name}" if multi else "", max_frames, Tracker) for c in cameras]
    by_name = {c.name: c for c in cams}
    payments = payments or MockPayments([])
    sink = AlertSink(out, MultiEvidence([c.evidence for c in cams]) if multi else cams[0].evidence)
    summary = RunSummary(source=",".join(str(c.source) for c in cameras), backend=backend.name,
                         video_fps=cams[0].src.fps)

    events_log = (out / "events.jsonl").open("w")
    timings = {"detect": 0.0, "track": 0.0, "events": 0.0, "ledger": 0.0, "render": 0.0}
    persons_seen: set[int] = set()
    ev_counts: dict[str, int] = {}
    t_start = time.perf_counter()
    t_wall0 = time.time()
    fps_ema = None
    last_frame_wall = time.time()

    def handle_alerts(alerts, frame_wall):
        for a in alerts:
            sink.emit(a)
            if a.retracts:
                if verbose:
                    print(f"  RETRACT {a.retracts} -> {a.tier}: {a.reasons[0]}")
            else:
                summary.decision_latency_s.append(round(a.latency_s, 3))
                summary.processing_latency_ms.append(round((time.time() - frame_wall) * 1000, 2))
                if verbose:
                    print(f"  ALERT {a.alert_id} [{a.tier}] person {a.person_id} conf={a.confidence:.2f} "
                          f"unpaid={[i.category for i in a.unpaid_items]}")
            if on_alert:
                on_alert(a)

    def feed(ev: Event, cam: _Cam) -> list:
        ev_counts[ev.type.value] = ev_counts.get(ev.type.value, 0) + 1
        events_log.write(json.dumps(ev.to_dict()) + "\n")
        cam.evidence.on_event(ev)
        return ledger.on_event(ev)

    heads = {c.name: next(c.frames, None) for c in cams}
    while True:
        # Next frame in camera time across all cameras (one camera: just the next frame).
        waiting = [c for c in cams if heads[c.name] is not None]
        if not waiting:
            break
        c = min(waiting, key=lambda k: heads[k.name].t)
        fr = heads[c.name]
        heads[c.name] = next(c.frames, None)
        if multi:   # live cameras share one clock (each source starts its own at first read)
            fr = replace(fr, camera=c.name, t=fr.wall - t_wall0 if c.src.live else fr.t)

        f0 = time.perf_counter()
        persons_det, products_det = backend(fr.image)
        f1 = time.perf_counter()
        persons, products = c.tracker.update(persons_det, products_det, fr.image.shape[:2])
        f2 = time.perf_counter()
        obs = FrameObs(fr.index, fr.t, persons, products)
        events = fusion.update(c.name, obs)
        f3 = time.perf_counter()
        alerts = []
        for pay in payments.poll(fr.t):
            alerts += ledger.on_payment(pay)
        for ev in events:
            alerts += feed(ev, c)
        alerts += ledger.tick(fr.t)
        f4 = time.perf_counter()

        c.recent = (c.recent + [e for e in events if e.type != EventType.ENTER])[-6:]
        open_baskets = {pid: [it.category for it in rec.basket] for pid, rec in ledger.people.items()
                        if not rec.reconciled}
        baskets = {p.track_id: open_baskets.get(fusion.person_id(c.name, p.track_id), []) for p in persons}
        dt = f4 - f0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
        vis = draw_frame(fr.image, c.store, persons, products, c.recent, baskets, fps_ema, blur=True)
        c.evidence.add_frame(fr.t, vis)
        if save_video:
            if c.writer is None:
                h, w = vis.shape[:2]
                c.writer = cv2.VideoWriter(str(c.video_path), cv2.VideoWriter_fourcc(*"mp4v"), c.src.fps, (w, h))
            c.writer.write(vis)
        handle_alerts(alerts, fr.wall)
        # Reconciled with no flag -> evidence is deleted right away.
        for pid, rec in ledger.people.items():
            if rec.reconciled:
                for k in cams:
                    if pid in k.evidence.snippets:
                        k.evidence.discard(pid)
        f5 = time.perf_counter()

        for k, v in zip(timings, (f1 - f0, f2 - f1, f3 - f2, f4 - f3, f5 - f4)):
            timings[k] += v
        persons_seen.update(fusion.person_id(c.name, p.track_id) for p in persons)
        if log_frames:   # off for long shadow runs: one line per frame, forever
            c.frames_log.write(json.dumps(_obs_to_json(obs, events)) + "\n")
        summary.frames += 1
        last_frame_wall = fr.wall
        if on_frame:
            on_frame(fr, obs, events, ledger, vis)
        if realtime and not c.src.live:   # replay a file at camera speed (dashboard demos)
            lag = (fr.t - (time.perf_counter() - t_start))
            if lag > 0:
                time.sleep(lag)
        if verbose and summary.frames % 100 == 0:
            el = time.perf_counter() - t_start
            print(f"  frame {summary.frames}: {summary.frames / el:.1f} FPS, {len(persons_seen)} people, events {ev_counts}")

    # End of stream: close out everyone still in view, then let the ledger decide.
    alerts = []
    for ev in fusion.flush():
        alerts += feed(ev, by_name.get(ev.meta.get("camera"), cams[0]))
    alerts += ledger.finalize()
    handle_alerts(alerts, last_frame_wall)

    summary.wall_s = round(time.perf_counter() - t_start, 3)
    summary.pipeline_fps = round(summary.frames / summary.wall_s, 2) if summary.wall_s else 0.0
    summary.stage_ms = {k: round(1000 * v / max(summary.frames, 1), 2) for k, v in timings.items()}
    summary.persons_tracked = len(persons_seen - {None})
    summary.events = ev_counts
    summary.alerts = [a.to_dict() for a in sink.written]
    summary.handoffs = len(fusion.identity.handoffs) if fusion.identity else 0
    events_log.close(); sink.close()
    for c in cams:
        c.frames_log.close()
        if c.writer is not None:
            c.writer.release()
    (out / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    (out / "ledger_log.txt").write_text("\n\n".join(
        f"person {pid}:\n" + "\n".join(rec.log) for pid, rec in sorted(ledger.people.items())))
    return summary
