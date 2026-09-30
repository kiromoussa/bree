"""End-to-end runner: video -> detect/pose -> track -> events -> ledger -> alerts.

Outputs in `out_dir`:
  frames.jsonl     per-frame log: person boxes, track IDs, keypoints, products, events
                   (no images, no faces)
  events.jsonl     store events
  alerts.jsonl     alerts (+ alerts/<id>.json and alerts/<id>.mp4 evidence clips)
  annotated.mp4    debug video with zones, boxes, IDs, skeletons (heads pixelated)
  summary.json     counts, FPS, per-stage timings, latencies
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.annotate import draw_frame
from bree.alerts.writer import AlertSink, EvidenceBuffer
from bree.detect.base import PerceptionBackend
from bree.events.engine import EventEngine
from bree.events.observations import FrameObs
from bree.events.types import Event, EventType
from bree.events.zones import StoreConfig
from bree.ingest.source import VideoSource
from bree.ledger import build_ledger
from bree.ledger.payments import PaymentSource, MockPayments
from bree.track.bytetrack import Tracker


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
                 on_alert=None, on_frame=None, verbose: bool = True, realtime: bool = False) -> RunSummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = VideoSource(source, max_frames=max_frames)
    tracker = Tracker(src.fps)
    engine = EventEngine(store)
    ledger = build_ledger(store, **(ledger_overrides or {}))
    payments = payments or MockPayments([])
    evidence = EvidenceBuffer(src.fps)
    sink = AlertSink(out, evidence)
    summary = RunSummary(source=str(source), backend=backend.name, video_fps=src.fps)

    frames_log = (out / "frames.jsonl").open("w")
    events_log = (out / "events.jsonl").open("w")
    writer = None
    recent: list[Event] = []
    timings = {"detect": 0.0, "track": 0.0, "events": 0.0, "ledger": 0.0, "render": 0.0}
    persons_seen: set[int] = set()
    ev_counts: dict[str, int] = {}
    t_start = time.perf_counter()
    fps_ema = None
    last_frame_wall = time.time()

    def handle_alerts(alerts, frame_wall):
        for a in alerts:
            sink.emit(a)
            summary.decision_latency_s.append(round(a.latency_s, 3))
            summary.processing_latency_ms.append(round((time.time() - frame_wall) * 1000, 2))
            if verbose:
                print(f"  ALERT {a.alert_id} [{a.tier}] person {a.person_id} conf={a.confidence:.2f} "
                      f"unpaid={[i.category for i in a.unpaid_items]}")
            if on_alert:
                on_alert(a)

    for fr in src:
        f0 = time.perf_counter()
        persons_det, products_det = backend(fr.image)
        f1 = time.perf_counter()
        persons, products = tracker.update(persons_det, products_det, fr.image.shape[:2])
        f2 = time.perf_counter()
        obs = FrameObs(fr.index, fr.t, persons, products)
        events = engine.update(obs)
        f3 = time.perf_counter()
        alerts = []
        for pay in payments.poll(fr.t):
            alerts += ledger.on_payment(pay)
        for ev in events:
            alerts += ledger.on_event(ev)
            evidence.on_event(ev)
            ev_counts[ev.type.value] = ev_counts.get(ev.type.value, 0) + 1
            events_log.write(json.dumps(ev.to_dict()) + "\n")
        alerts += ledger.tick(fr.t)
        f4 = time.perf_counter()

        recent = (recent + [e for e in events if e.type != EventType.ENTER])[-6:]
        baskets = {pid: [it.category for it in rec.basket] for pid, rec in ledger.people.items()
                   if not rec.reconciled}
        dt = f4 - f0
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
        vis = draw_frame(fr.image, store, persons, products, recent, baskets, fps_ema, blur=True)
        evidence.add_frame(fr.t, vis)
        if save_video:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
            writer.write(vis)
        handle_alerts(alerts, fr.wall)
        # Reconciled with no flag -> evidence is deleted right away.
        for pid, rec in ledger.people.items():
            if rec.reconciled and pid in evidence.snippets:
                evidence.discard(pid)
        f5 = time.perf_counter()

        for k, v in zip(timings, (f1 - f0, f2 - f1, f3 - f2, f4 - f3, f5 - f4)):
            timings[k] += v
        persons_seen.update(p.track_id for p in persons)
        frames_log.write(json.dumps(_obs_to_json(obs, events)) + "\n")
        summary.frames += 1
        last_frame_wall = fr.wall
        if on_frame:
            on_frame(fr, obs, events, ledger, vis)
        if realtime and not src.live:   # replay a file at camera speed (dashboard demos)
            lag = (fr.t - (time.perf_counter() - t_start))
            if lag > 0:
                time.sleep(lag)
        if verbose and summary.frames % 100 == 0:
            el = time.perf_counter() - t_start
            print(f"  frame {summary.frames}: {summary.frames / el:.1f} FPS, {len(persons_seen)} people, events {ev_counts}")

    # End of stream: close out everyone still in view, then let the ledger decide.
    tail = engine.flush()
    alerts = []
    for ev in tail:
        alerts += ledger.on_event(ev)
        evidence.on_event(ev)
        ev_counts[ev.type.value] = ev_counts.get(ev.type.value, 0) + 1
        events_log.write(json.dumps(ev.to_dict()) + "\n")
    alerts += ledger.finalize()
    handle_alerts(alerts, last_frame_wall)

    summary.wall_s = round(time.perf_counter() - t_start, 3)
    summary.pipeline_fps = round(summary.frames / summary.wall_s, 2) if summary.wall_s else 0.0
    summary.stage_ms = {k: round(1000 * v / max(summary.frames, 1), 2) for k, v in timings.items()}
    summary.persons_tracked = len(persons_seen)
    summary.events = ev_counts
    summary.alerts = [a.to_dict() for a in sink.written]
    frames_log.close(); events_log.close(); sink.close()
    if writer is not None:
        writer.release()
    (out / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    (out / "ledger_log.txt").write_text("\n\n".join(
        f"person {pid}:\n" + "\n".join(rec.log) for pid, rec in sorted(ledger.people.items())))
    return summary
