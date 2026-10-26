"""Camera tamper / obstruction detection from image health features (bree.theft.health). No detector needed.

A shoplifter (or a bored employee) can cover a lens, spray it, twist the camera away, unplug it, or just put a display
in front of it. A cheap monitor over per-frame statistics catches all of those without any model:

  covered   the picture went (nearly) uniform: a hand, tape, paint, a cloth. Contrast collapses against what this camera
            normally shows.
  dark      the picture is black or nearly black while it normally is not (lens covered, power to the lights cut,
            night-vision failing). A dark store at night is expected: pass `expect_dark` hours to the monitor.
  blurred   sharpness (variance of the Laplacian) fell to a fraction of the camera's own normal: defocus, smear, fog,
            a film on the dome.
  moved     the scene is no longer the scene: the 32x18 thumbnail, with people masked out and brightness normalised, does
            not match the camera's own reference for a sustained time. Rotated, tilted, knocked, or something large and
            still in front of the lens. People walking through do not trigger it (their boxes are masked, and it has to persist).
  frozen    consecutive frames are bit-identical: the stream is stalled or looped, so the cameras you "watch" are a picture.

Each alarm has a start, an end ("cleared") and a severity, and says which numbers moved. The camera's own normal (reference
thumbnail, sharpness, contrast, brightness) is learned online from quiet frames and is never updated while an alarm is on.
Thresholds are in `TamperConfig`; the numbers they were checked against are in docs/ultra/theft.md (false alarms on
real footage: REAL-UNLABELLED; detection of tampering: SIMULATED on real frames, we have no real tamper clip).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("covered", "dark", "blurred", "moved", "frozen")
SEVERITY = {"covered": "high", "dark": "medium", "blurred": "medium", "moved": "high", "frozen": "high"}


@dataclass
class TamperConfig:
    warmup_s: float = 20.0           # no alarms until this long after the first frame (the reference is being learned)
    hold_s: dict = field(default_factory=lambda: {"covered": 8.0, "dark": 8.0, "blurred": 15.0, "moved": 30.0, "frozen": 10.0})
    clear_s: float = 6.0             # an alarm ends after this long back to normal
    uniform_std: float = 7.0         # covered: frame contrast (grey levels) below this...
    uniform_ref_ratio: float = 0.35  # ...and below this share of the normal contrast
    dark_mean: float = 25.0          # dark: mean below this and below dark_ref_ratio of the normal mean
    dark_ref_ratio: float = 0.4
    blur_ratio: float = 0.3          # blurred: sharpness below this share of normal (and not covered/dark)
    moved_ncc: float = 0.45          # moved: normalised correlation with the reference below this...
    moved_changed: float = 0.25      # ...and this share of unmasked cells changed by more than 1 sd
    max_masked: float = 0.6          # more than this share of cells hidden behind people: no verdict (not "moved")
    frozen_diff: float = 0.02        # mean abs thumbnail difference between consecutive samples below this = identical
    adopt_s: float = 1800.0          # a "moved" alarm that lasts this long is the new normal (furniture moved): relearn the reference
    ref_every_s: float = 10.0        # reference update step (exponential average on quiet cells)
    ref_alpha: float = 0.15
    expect_dark: tuple = ()          # [(start_s_of_day, end_s_of_day)] where darkness is normal (needs `clock`)


@dataclass
class TamperEvent:
    kind: str
    phase: str                       # "start" (hold time reached; t_start is when it began) or "end"
    t: float
    t_start: float
    severity: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "phase": self.phase, "t": round(self.t, 2), "t_start": round(self.t_start, 2),
                "severity": self.severity, "evidence": self.evidence}


def _z(a: np.ndarray) -> np.ndarray | None:
    s = float(a.std())
    return None if s < 1e-3 else (a - a.mean()) / s


class TamperMonitor:
    """Feed samples in time order: `update(t, features, thumb, person_boxes)`; returns the events that happened at t.

    features: bree.theft.health.frame_features dict. thumb: uint8 (18, 32) grey thumbnail.
    person_boxes: normalised (x1, y1, x2, y2) of every detected person (all of them, not only gated ones): their cells
    are left out of the scene comparison.
    clock: optional callable t -> seconds of the local day, for `expect_dark`.
    """

    def __init__(self, cfg: TamperConfig | None = None, clock=None):
        self.cfg = cfg or TamperConfig()
        self.clock = clock
        self.t0: float | None = None
        self.ref: np.ndarray | None = None          # (18, 32) float, nan where never seen without a person
        self.ref_std = self.ref_mean = self.ref_lap = None
        self._t_ref = -1e9
        self.bad_since: dict[str, float | None] = {k: None for k in KINDS}
        self.ok_since: dict[str, float | None] = {k: None for k in KINDS}
        self.active: dict[str, TamperEvent] = {}
        self.last: dict = {}                        # latest measurements (debug / plots)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _mask(boxes, shape) -> np.ndarray:
        m = np.zeros(shape, bool)
        h, w = shape
        for x1, y1, x2, y2 in boxes or []:
            m[max(0, int(np.floor(y1 * h)) - 1):min(h, int(np.ceil(y2 * h)) + 1),
              max(0, int(np.floor(x1 * w)) - 1):min(w, int(np.ceil(x2 * w)) + 1)] = True
        return m

    def _dark_expected(self, t: float) -> bool:
        if not self.cfg.expect_dark or self.clock is None:
            return False
        s = self.clock(t) % 86400
        return any((a <= s < b) if a <= b else (s >= a or s < b) for a, b in self.cfg.expect_dark)

    def _scene(self, thumb: np.ndarray, masked: np.ndarray) -> tuple[float | None, float | None, float]:
        """(NCC, share of cells changed, share of cells left out) of thumb against the reference."""
        valid = ~masked & ~np.isnan(self.ref)
        left_out = 1.0 - float(valid.mean())
        if valid.sum() < 0.3 * valid.size:
            return None, None, left_out
        a, b = thumb[valid].astype(np.float32), self.ref[valid]
        za, zb = _z(a), _z(b)
        if za is None or zb is None:
            return None, None, left_out
        return float((za * zb).mean()), float((np.abs(za - zb) > 1.0).mean()), left_out

    # ------------------------------------------------------------------ main

    def update(self, t: float, f: dict, thumb: np.ndarray, person_boxes=None) -> list[TamperEvent]:
        c = self.cfg
        if self.t0 is None:
            self.t0 = t
        masked = self._mask(person_boxes, thumb.shape)
        warm = t - self.t0 < c.warmup_s
        th = thumb.astype(np.float32)

        if self.ref is None:
            self.ref = np.full(thumb.shape, np.nan, np.float32)
        ncc0, ch0, lo0 = self._scene(th, masked)
        # The reference only learns from frames that still look like the scene (else a slow tamper is "learned" before it is seen).
        looks_normal = warm or (ncc0 is not None and ncc0 > 0.55 and ch0 < 0.3) or lo0 > c.max_masked
        quiet = not self.active and looks_normal
        if quiet and (warm or t - self._t_ref >= c.ref_every_s):
            fresh = ~masked
            upd = fresh & np.isnan(self.ref)
            self.ref[upd] = th[upd]
            both = fresh & ~np.isnan(self.ref) & ~upd
            self.ref[both] = (1 - c.ref_alpha) * self.ref[both] + c.ref_alpha * th[both]
            a = c.ref_alpha if self.ref_std is not None else 1.0
            # contrast / brightness / sharpness learn only while they are near normal: a slow cover or defocus must not become "normal"
            sane = self.ref_std is None or (f["std"] > 0.6 * self.ref_std and f["mean"] > 0.6 * self.ref_mean
                                            and f["lap"] > 0.6 * self.ref_lap)
            for name, v in (("ref_std", f["std"]), ("ref_mean", f["mean"]), ("ref_lap", f["lap"])):
                old = getattr(self, name)
                if sane or warm:
                    setattr(self, name, v if old is None else (1 - a) * old + a * v)
            self._t_ref = t

        ncc, changed, left_out = ncc0, ch0, lo0
        std_ref = max(self.ref_std or 0.0, 1e-6)
        bad = {k: False for k in KINDS}
        covered = f["std"] < c.uniform_std and f["std"] < c.uniform_ref_ratio * std_ref
        dark = (f["mean"] < c.dark_mean and f["mean"] < c.dark_ref_ratio * (self.ref_mean or 1e9)
                and not self._dark_expected(t))
        bad["covered"] = covered and not (f["mean"] < c.dark_mean)      # black = "dark", flat but lit = "covered"
        bad["dark"] = dark
        bad["blurred"] = (not covered and not dark and self.ref_lap is not None
                          and f["lap"] < c.blur_ratio * self.ref_lap and self.ref_lap > 5)
        bad["moved"] = (not covered and not dark and ncc is not None and left_out <= c.max_masked
                        and ncc < c.moved_ncc and changed > c.moved_changed)
        bad["frozen"] = f["diff"] < c.frozen_diff and t - self.t0 > 0.5
        self.last = {"t": t, "ncc": ncc, "changed": changed, "left_out": left_out, "lap_ratio":
                     None if not self.ref_lap else f["lap"] / self.ref_lap, "std": f["std"], "mean": f["mean"]}

        out = []
        mv = self.active.get("moved")
        if mv is not None and t - mv.t_start >= c.adopt_s:      # months of a different view is not tampering any more
            self.ref = None
            self.ref_std = self.ref_mean = self.ref_lap = None
            self.t0 = t                                           # relearn: warm-up again
            self.active.pop("moved")
            self.bad_since["moved"] = self.ok_since["moved"] = None
            return [TamperEvent("moved", "end", t, mv.t_start, mv.severity, dict(mv.evidence, adopted_new_normal=True))]
        for k in KINDS:
            if bad[k] and not warm:
                self.ok_since[k] = None
                if self.bad_since[k] is None:
                    self.bad_since[k] = t
                if k not in self.active and t - self.bad_since[k] >= c.hold_s[k]:
                    ev = TamperEvent(k, "start", t, self.bad_since[k], SEVERITY[k], self._evidence(k, f))
                    self.active[k] = ev
                    out.append(ev)
            else:
                self.bad_since[k] = None
                if k in self.active:
                    if self.ok_since[k] is None:
                        self.ok_since[k] = t
                    if t - self.ok_since[k] >= c.clear_s:
                        st = self.active.pop(k)
                        out.append(TamperEvent(k, "end", t, st.t_start, st.severity, st.evidence))
        return out

    def _evidence(self, kind: str, f: dict) -> dict:
        L = self.last
        ev = {"std": round(f["std"], 1), "mean": round(f["mean"], 1), "lap": round(f["lap"], 1),
              "ref_std": round(self.ref_std or 0, 1), "ref_mean": round(self.ref_mean or 0, 1),
              "ref_lap": round(self.ref_lap or 0, 1)}
        if kind == "moved":
            ev.update(ncc=round(L["ncc"], 2), changed=round(L["changed"], 2))
        return ev

    def flush(self, t: float) -> list[TamperEvent]:
        """End of stream: close every open alarm."""
        out = [TamperEvent(k, "end", t, e.t_start, e.severity, e.evidence) for k, e in self.active.items()]
        self.active.clear()
        return out
