"""Alert output: JSON records + a short evidence clip per alert.

Evidence handling (privacy by default):
- Frames are kept only in memory: a ~2 s ring buffer, plus short snippets
  around each person's pick / conceal / exit events.
- Frames are already head-pixelated and downscaled before they are buffered.
- A person's snippets are written to disk only if they get an alert or review
  flag; otherwise they are deleted as soon as the ledger reconciles them.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.types import Alert
from bree.events.types import Event, EventType

CLIP_EVENTS = (EventType.PICK, EventType.CONCEAL, EventType.PUT_BACK, EventType.EXIT)


@dataclass
class _Snippet:
    event: str
    t_event: float
    t_until: float
    frames: list[tuple[float, bytes]] = field(default_factory=list)   # (t, jpeg bytes)


class EvidenceBuffer:
    def __init__(self, fps: float, pre_s: float = 2.0, post_s: float = 2.0, width: int = 640,
                 max_snippets_per_person: int = 8):
        self.pre_s, self.post_s, self.width = pre_s, post_s, width
        self.ring: deque[tuple[float, bytes]] = deque(maxlen=max(1, int(pre_s * fps)))
        self.snippets: dict[int, list[_Snippet]] = {}
        self.max_snippets = max_snippets_per_person
        self.fps = fps

    def _encode(self, img: np.ndarray) -> bytes:
        h, w = img.shape[:2]
        if w > self.width:
            img = cv2.resize(img, (self.width, int(h * self.width / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def add_frame(self, t: float, annotated_blurred: np.ndarray) -> None:
        jpg = self._encode(annotated_blurred)
        self.ring.append((t, jpg))
        for snips in self.snippets.values():
            for s in snips:
                if t <= s.t_until and (not s.frames or t > s.frames[-1][0]):
                    s.frames.append((t, jpg))

    def on_event(self, ev: Event) -> None:
        if ev.type not in CLIP_EVENTS:
            return
        snips = self.snippets.setdefault(ev.person_id, [])
        if len(snips) >= self.max_snippets:
            snips.pop(0)
        s = _Snippet(ev.type.value, ev.t, ev.t + self.post_s)
        s.frames = [(t, j) for t, j in self.ring if t >= ev.t - self.pre_s]
        snips.append(s)

    def discard(self, person_id: int) -> None:
        self.snippets.pop(person_id, None)

    def write_clip(self, person_ids: list[int], path: Path) -> str | None:
        frames = []
        for pid in person_ids:
            for s in self.snippets.get(pid, []):
                frames += s.frames
        frames.sort(key=lambda f: f[0])
        dedup, last_t = [], None
        for t, j in frames:
            if t != last_t:
                dedup.append(j)
                last_t = t
        if not dedup:
            return None
        imgs = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in dedup]
        h, w = imgs[0].shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, self.fps), (w, h))
        for im in imgs:
            vw.write(im)
        vw.release()
        return str(path)


class AlertSink:
    """Writes alerts to <out>/alerts/<id>.json, <out>/alerts.jsonl, and optional clips."""

    def __init__(self, out_dir: str | Path, evidence: EvidenceBuffer | None = None):
        self.out = Path(out_dir)
        (self.out / "alerts").mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        self.jsonl = (self.out / "alerts.jsonl").open("w")
        self.written: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        if self.evidence is not None:
            pids = alert.group or [alert.person_id]
            alert.clip_path = self.evidence.write_clip(pids, self.out / "alerts" / f"{alert.alert_id}.mp4")
            for pid in pids:
                self.evidence.discard(pid)
        d = alert.to_dict()
        (self.out / "alerts" / f"{alert.alert_id}.json").write_text(json.dumps(d, indent=2))
        self.jsonl.write(json.dumps(d) + "\n")
        self.jsonl.flush()
        self.written.append(alert)

    def close(self) -> None:
        self.jsonl.close()
