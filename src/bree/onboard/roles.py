"""Suggest what each camera is for (door, register, cooler, aisle, backroom, outdoor, obscured, unknown).

A transparent additive rule list over the statistics in bree.onboard.stats: every rule that fires adds a weight to
one role and writes the sentence that explains it. The suggestion carries the scores, the evidence lines and a
`needs_review` flag; a human corrects it with `bree fleet set-role` and the correction is never overwritten.

LIMITS, read before trusting it (numbers in docs/ultra/platform.md):
  * Thresholds were written from the first 7 channels of ONE store and then adjusted after looking at their
    statistics; those 7 channels are therefore a tuning set, not a test set, and no accuracy is claimed.
  * Door versus aisle-toward-a-window is the weakest call (both see glass, daylight and parked cars).
  * Cooler versus aisle depends on COCO "refrigerator" boxes; an aisle that ends at a cooler looks like a cooler.
  * backroom has no rule that fires on its own (no staff-only footage was available): the best that can happen
    is `unknown`. `obscured` is tested on synthetic frames only.
  * Every suggestion is a draft for a person to confirm while looking at the contact sheet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ROLES = ("door", "register", "cooler", "aisle", "backroom", "outdoor", "obscured", "unknown")


@dataclass
class Rule:
    role: str
    weight: float
    text: str                                   # evidence sentence when the rule fires
    test: Callable[[dict], bool]


def _g(f: dict, k: str, default: float = 0.0) -> float:
    v = f.get(k)
    return default if v is None else v


RULES = [
    # obscured / bad picture
    Rule("obscured", 5, "very blurry or featureless picture (sharpness < 600)", lambda f: 0 < _g(f, "sharpness", 1e9) < 600),
    Rule("obscured", 5, "almost no contrast (std < 0.06): covered or dark lens", lambda f: 0 < _g(f, "contrast", 1) < 0.06),
    Rule("obscured", 5, "mostly black or white frame", lambda f: f.get("brightness") is not None and (f["brightness"] < 0.06 or f["brightness"] > 0.94)),
    # outdoor
    Rule("outdoor", 3, "large vehicles in view (mean box > 3% of the frame)", lambda f: _g(f, "vehicle_mean_area") > 0.03),
    Rule("outdoor", 2, "few edges (edge density < 0.14): open ground, not shelving", lambda f: 0 < _g(f, "edge_density", 1) < 0.14),
    Rule("outdoor", 1, "vehicles in most frames", lambda f: _g(f, "vehicle_frac") >= 0.5),
    # door
    Rule("door", 2, "large bright blue/white region at the top (glass doors or windows)", lambda f: _g(f, "sky_frac") >= 0.12),
    Rule("door", 1.5, "small vehicles seen through glass while people are indoors",
         lambda f: _g(f, "vehicle_frac") >= 0.5 and 0 < _g(f, "vehicle_mean_area") <= 0.03 and _g(f, "person_frac") > 0),
    Rule("door", 1, "people are large in the frame and present in most frames",
         lambda f: _g(f, "person_frac") >= 0.4 and _g(f, "person_mean_height") >= 0.25),
    # register
    Rule("register", 4, "many screens/keyboards/terminals, fixed in place (counter equipment)",
         lambda f: _g(f, "posgear_mean_count") >= 3 and _g(f, "posgear_static") >= 0.8),
    Rule("register", 1.5, "one fixed screen/terminal in nearly every frame",
         lambda f: _g(f, "posgear_frac") >= 0.8 and _g(f, "posgear_static") >= 0.8),
    # cooler
    Rule("cooler", 3, "several refrigerator doors per frame (>= 1.5)", lambda f: _g(f, "fridge_mean_count") >= 1.5),
    Rule("cooler", 1.5, "drinks fixed in place in most frames", lambda f: _g(f, "drinks_frac") >= 0.8 and _g(f, "drinks_static") >= 0.6),
    # aisle
    Rule("aisle", 1, "indoor clutter (edge density >= 0.14) with people passing", lambda f: _g(f, "edge_density") >= 0.14 and _g(f, "person_frac") > 0),
    Rule("aisle", 2, "shelved goods in view (>= 4 detections per frame)", lambda f: _g(f, "goods_mean_count") >= 4),
    Rule("aisle", 1, "people walk in the picture most of the time", lambda f: _g(f, "person_frac") >= 0.8 and _g(f, "posgear_frac") == 0),
]


@dataclass
class RoleSuggestion:
    channel: int
    role: str
    confidence: float
    scores: dict[str, float]
    evidence: list[str] = field(default_factory=list)
    needs_review: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {"channel": self.channel, "role": self.role, "confidence": round(self.confidence, 2),
                "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
                "evidence": self.evidence, "needs_review": self.needs_review, "note": self.note}


def suggest_role(channel: int, feats: dict, error: str | None = None, online: bool = True) -> RoleSuggestion:
    if not online:
        return RoleSuggestion(channel, "unknown", 0.0, {}, ["channel is offline"], True, "no picture to judge")
    if error or not feats:
        return RoleSuggestion(channel, "unknown", 0.0, {}, [error or "no frames"], True, "could not get frames")
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for r in RULES:
        if r.test(feats):
            scores[r.role] = scores.get(r.role, 0) + r.weight
            evidence.setdefault(r.role, []).append(f"{r.role} +{r.weight:g}: {r.text}")
    if not scores:
        return RoleSuggestion(channel, "unknown", 0.0, {}, ["no rule fired"], True, "nothing in the statistics points anywhere")
    order = sorted(scores, key=lambda k: -scores[k])
    top = order[0]
    second = scores[order[1]] if len(order) > 1 else 0.0
    conf = scores[top] / (scores[top] + second + 1.0)
    review = conf < 0.55 or scores[top] < 2.5 or (second > 0 and scores[top] - second < 1.5)
    note = ""
    if review and second > 0:
        note = f"close call with {order[1]} ({scores[order[1]]:g} vs {scores[top]:g})"
    ev = evidence[top] + [e for r in order[1:3] for e in evidence[r][:1]]
    return RoleSuggestion(channel, top, conf, scores, ev, review, note)
