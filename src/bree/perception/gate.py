"""Track confirmation: keep unconfirmed tracks away from the event engine.

Real baseline runs gave 40-80 track ids for 3-8 people per 5 minutes, and every id fires an 'enter'. Most were
flickering low-confidence boxes that lived a few frames. The gate hides a track until it has been seen for
`min_track_age_s` of video time (default 0 = off), after which it is passed through for good, so the engine only ever
sees confirmed people. Cost: the first `min_track_age_s` of each person is not observed (an ENTER fires that much later).
"""
from __future__ import annotations

from bree.detect.base import Detections
from bree.events.observations import PersonObs, ProductObs
from bree.perception.profile import PerceptionProfile
from bree.track.bytetrack import Tracker


class TrackGate:
    def __init__(self, fps: float, min_age_s: float):
        self.min_frames = max(1, int(round(min_age_s * fps))) if min_age_s > 0 else 1
        self.seen: dict[int, int] = {}
        self.confirmed: set[int] = set()

    def filter(self, persons: list[PersonObs]) -> list[PersonObs]:
        out = []
        for p in persons:
            if p.track_id not in self.confirmed:
                self.seen[p.track_id] = self.seen.get(p.track_id, 0) + 1
                if self.seen[p.track_id] < self.min_frames:
                    continue
                self.confirmed.add(p.track_id)
                self.seen.pop(p.track_id, None)
            out.append(p)
        return out


class GatedTracker(Tracker):
    def __init__(self, fps: float, profile: PerceptionProfile):
        super().__init__(fps, person_buffer_s=profile.track_buffer_s, bytetrack=profile.tracker_overrides())
        self.gate = TrackGate(fps, profile.min_track_age_s)

    def update(self, persons: Detections, products: Detections, shape) -> tuple[list[PersonObs], list[ProductObs]]:
        p, q = super().update(persons, products, shape)
        return self.gate.filter(p), q
