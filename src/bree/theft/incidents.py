"""One shape for everything a human reviewer might be asked to look at, and the priority score that orders the queue.

Sources: ledger review items (bree.theft.tiers bands), tamper alarms, after-hours / loitering incidents and bulk-grab /
sweep events. All become an `Incident`; `priority()` puts a number from 0 to 100 on it so the review page can show the
few that matter first. The score is a transparent sum (every term is listed in `Incident.priority_why`), not a model:
there is no labelled real data to fit one. When reviewers have labelled items, `PriorityModel.fit` replaces the base
weights of each kind (and band) with the observed share of confirmed items, smoothed towards the prior.
Nothing here is a verdict. Wording is "review", never "theft".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bree.alerts.types import Alert
from bree.theft.tiers import BAND_TEXT, band_of

SEVERITY_W = {"low": 0.3, "medium": 0.6, "high": 0.9}


@dataclass
class Incident:
    kind: str                         # theft_review | tamper | after_hours | loiter | bulk_grab
    t: float                          # stream seconds of the key moment
    t_start: float
    severity: str                     # low | medium | high
    headline: str
    details: dict = field(default_factory=dict)
    camera: str = ""
    band: str | None = None           # R1..R3 for theft_review
    wall: float | None = None         # epoch of the key moment on the recorder clock, when known
    clip: str | None = None
    priority: float = 0.0
    priority_why: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "camera": self.camera, "t": round(self.t, 2), "t_start": round(self.t_start, 2),
                "severity": self.severity, "headline": self.headline, "details": self.details, "band": self.band,
                "wall": self.wall, "clip": self.clip, "priority": round(self.priority, 1), "priority_why": self.priority_why}


def from_alert(a: Alert, camera: str = "") -> Incident:
    b = band_of(a)
    cats = ", ".join(sorted({i.category for i in a.unpaid_items})) or "item"
    return Incident("theft_review", a.t_exit, min((i.t_pick for i in a.unpaid_items), default=a.t_exit),
                    "high" if b == "R1" else "medium" if b == "R2" else "low",
                    f"{b}: {BAND_TEXT.get(b, '')} ({cats})", {
                        "alert_id": a.alert_id, "confidence": a.confidence, "visited_register": a.visited_register,
                        "concealment_seen": any(i.concealed for i in a.unpaid_items),
                        "items": [i.category for i in a.unpaid_items], "reasons": a.reasons[:6]},
                    camera=camera, band=b)


def from_tamper(ev, camera: str = "") -> Incident:
    nice = {"covered": "lens looks covered", "dark": "picture went dark", "blurred": "picture is blurred / out of focus",
            "moved": "view no longer matches the camera's normal scene (moved, rotated or blocked)",
            "frozen": "picture is frozen (stalled or looped stream)"}[ev.kind]
    return Incident("tamper", ev.t, ev.t_start, ev.severity, f"camera {nice}", dict(ev.evidence, tamper=ev.kind), camera=camera)


def from_sweep(ev, camera: str = "") -> Incident:
    what = {"sweep": "fast sweep of the hand along", "repeat": "repeated reach-ins to", "both": "sweep and repeated reaches at"}[ev.kind]
    return Incident("bulk_grab", ev.t, ev.t_start, "medium" if ev.score >= 0.7 else "low",
                    f"{what} {ev.zone}", dict(ev.detail, zone=ev.zone, track=ev.person_id, score=round(ev.score, 2)), camera=camera)


# ------------------------------------------------------------------ priority

BASE = {                              # prior probability-ish weight that a human finds this worth their time
    ("theft_review", "R1"): 0.80, ("theft_review", "R2"): 0.50, ("theft_review", "R3"): 0.15,
    ("tamper", None): 0.70, ("after_hours", None): 0.60, ("bulk_grab", None): 0.30, ("loiter", None): 0.15,
}


@dataclass
class PriorityModel:
    base: dict = field(default_factory=lambda: dict(BASE))
    camera_noise: dict = field(default_factory=dict)       # camera -> share of its past items labelled false (0..1)
    prior_n: float = 4.0                                    # pseudo-labels behind the prior when fitting

    def key(self, i: Incident):
        return (i.kind, i.band if i.kind == "theft_review" else None)

    def fit(self, labelled: list[tuple[Incident, bool]]) -> "PriorityModel":
        """labelled: (incident, True if a reviewer confirmed it was worth acting on). Smooths each key's confirmed share
        towards its prior with `prior_n` pseudo-labels, and records each camera's false share."""
        by: dict = {}
        cam: dict = {}
        for inc, ok in labelled:
            k = self.key(inc)
            n, y = by.get(k, (0, 0))
            by[k] = (n + 1, y + bool(ok))
            n, y = cam.get(inc.camera, (0, 0))
            cam[inc.camera] = (n + 1, y + (not ok))
        base = dict(self.base)
        for k, (n, y) in by.items():
            p0 = base.get(k, 0.3)
            base[k] = (p0 * self.prior_n + y) / (self.prior_n + n)
        noise = {c: (y + 0.0) / n for c, (n, y) in cam.items() if n >= 5}
        return PriorityModel(base, noise, self.prior_n)

    def score(self, i: Incident, camera_tampered: bool = False, after_hours: bool = False, repeats_nearby: int = 0) -> Incident:
        """Fill i.priority (0-100) and i.priority_why."""
        why = []
        s = self.base.get(self.key(i), 0.3)
        why.append(f"base {s:.2f} for {i.kind}{'/' + i.band if i.band else ''}")
        sev = SEVERITY_W.get(i.severity, 0.5)
        s *= 0.6 + 0.4 * sev
        why.append(f"severity {i.severity}")
        if i.kind == "theft_review":
            conf = float(i.details.get("confidence", 0.5))
            s *= 0.7 + 0.3 * conf
            why.append(f"ledger confidence {conf:.2f}")
            if i.details.get("concealment_seen"):
                s = min(1.0, s + 0.1)
                why.append("concealment seen (+0.10)")
            if after_hours:
                s = min(1.0, s + 0.15)
                why.append("store closed (+0.15)")
        if i.kind == "bulk_grab":
            s *= 0.7 + 0.3 * float(i.details.get("score", 0.5))
        if repeats_nearby:
            s = min(1.0, s + 0.05 * min(repeats_nearby, 3))
            why.append(f"{repeats_nearby} other item(s) from the same camera close in time (+{0.05 * min(repeats_nearby, 3):.2f})")
        if camera_tampered and i.kind != "tamper":
            s *= 0.6
            why.append("camera is flagged as tampered/unhealthy: evidence from it is less reliable (x0.6)")
        noise = self.camera_noise.get(i.camera)
        if noise is not None and i.kind != "tamper":
            s *= 1.0 - 0.5 * noise
            why.append(f"this camera's past items were {noise:.0%} false (x{1.0 - 0.5 * noise:.2f})")
        i.priority, i.priority_why = round(100 * min(s, 1.0), 1), why
        return i


def rank(incidents: list[Incident], model: PriorityModel | None = None, tampered: dict | None = None,
         closed: bool | dict = False) -> list[Incident]:
    """Score every incident and sort the queue: highest priority first, newest first among equals. `tampered`: camera ->
    list of (t0, t1) windows where it raised a tamper alarm."""
    m = model or PriorityModel()
    out = []
    for i in incidents:
        bad = any(a <= i.t <= b for a, b in (tampered or {}).get(i.camera, []))
        near = sum(1 for j in incidents if j is not i and j.camera == i.camera and abs(j.t - i.t) < 120 and j.kind == i.kind)
        closed_now = closed.get(i.camera, False) if isinstance(closed, dict) else closed
        out.append(m.score(i, camera_tampered=bad, after_hours=closed_now, repeats_nearby=near))
    return sorted(out, key=lambda i: (-i.priority, -i.t))
