"""Anomalies in the flow stream: things that break the store's own repeating pattern, plus a few event rules.

Two kinds:
  rules     from one record stream, no history: long queue, unattended counter (customers at the till, nobody behind it),
            loitering (standing still a long time away from the till), door imbalance (entries and exits far apart over
            the observed time, which usually means missed crossings or a second door).
  baseline  a store's own hour-of-week profile built from past hourly aggregates (bree.intel.aggregate.hourly): robust z
            (median and MAD) of entries per hour, queue and till time against the same weekday and hour. Needs history:
            nothing is flagged until `min_history` earlier weeks of that slot exist, and an hour with little video
            (`min_coverage_s`) is not judged.

No identity: "repeat" means the same time slot repeating week after week, not the same person coming back. Every flag
carries the numbers behind it. Rules are tested on SIMULATED records only; no real day with a known anomaly exists yet.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RuleConfig:
    queue_len: int = 3
    queue_s: float = 30.0
    unattended_s: float | None = None   # OFF by default: the door camera detects counter staff in under half of labelled frames, so on
    #                                     the real footage this rule fired while the counter was staffed (docs/ultra/intel.md, negative result)
    loiter_still_s: float = 300.0
    imbalance_min: int = 6          # do not judge door balance below this many crossings
    imbalance_frac: float = 0.5


def rule_anomalies(records: list[dict], cfg: RuleConfig | None = None) -> list[dict]:
    cfg = cfg or RuleConfig()
    out: list[dict] = []
    samples = [r for r in records if r["type"] == "sample"]

    def run(pred, min_s):
        start = last = None
        for s in samples:
            if pred(s):
                start = s if start is None else start
                last = s
            else:
                if start is not None and last["t"] - start["t"] >= min_s:
                    yield start, last
                start = None
        if start is not None and last["t"] - start["t"] >= min_s:
            yield start, last

    for a, b in run(lambda s: s["queue"] >= cfg.queue_len, cfg.queue_s):
        out.append({"type": "anomaly", "kind": "long_queue", "t": a["t"], "ts": a.get("ts"), "duration_s": round(b["t"] - a["t"], 1),
                    "detail": f"{cfg.queue_len}+ people at the till"})
    for a, b in (run(lambda s: s["queue"] >= 1 and s["staff"] == 0, cfg.unattended_s) if cfg.unattended_s else []):
        out.append({"type": "anomaly", "kind": "unattended_counter", "t": a["t"], "ts": a.get("ts"), "duration_s": round(b["t"] - a["t"], 1),
                    "detail": "customer at the till, no staff seen behind the counter (advisory: staff are often not detected)"})
    for r in records:
        if r["type"] == "zone" and r["kind"] == "browse" and r["still_s"] >= cfg.loiter_still_s:
            out.append({"type": "anomaly", "kind": "loitering", "t": r["t_in"], "ts": r.get("ts"), "duration_s": r["still_s"],
                        "detail": f"standing still {r['still_s']:.0f} s at {r['zone']}"})
    ins = sum(1 for r in records if r["type"] == "door" and r["dir"] == "in")
    outs = sum(1 for r in records if r["type"] == "door" and r["dir"] == "out")
    if ins + outs >= cfg.imbalance_min and abs(ins - outs) / max(ins + outs, 1) >= cfg.imbalance_frac:
        out.append({"type": "anomaly", "kind": "door_imbalance", "t": 0.0, "ts": None, "duration_s": None,
                    "detail": f"{ins} entries vs {outs} exits: check for missed crossings or another door"})
    return out


METRICS = ("entries_per_hour", "queue_mean", "register_time_median_s")


def _slot(hour: str) -> tuple[int, int]:
    d = datetime.fromisoformat(hour)
    return d.weekday(), d.hour


@dataclass
class Baseline:
    """Hour-of-week robust profile. `fit` takes the list of hourly dicts (aggregate.hourly(...).values())."""
    stats: dict
    min_history: int = 3
    min_coverage_s: float = 600.0

    @staticmethod
    def fit(hours: list[dict], min_history: int = 3, min_coverage_s: float = 600.0) -> "Baseline":
        by: dict = {}
        for h in hours:
            if h["coverage_s"] < min_coverage_s:
                continue
            for m in METRICS:
                if h.get(m) is not None:
                    by.setdefault((_slot(h["hour"]), m), []).append(float(h[m]))
        stats = {}
        for key, v in by.items():
            med = statistics.median(v)
            mad = statistics.median([abs(x - med) for x in v]) or 0.0
            stats[key] = {"n": len(v), "median": med, "mad": mad}
        return Baseline(stats, min_history, min_coverage_s)

    def score(self, hour: dict, z_min: float = 3.5) -> list[dict]:
        """Anomalies of one hourly dict against the profile. Scale = 1.4826 MAD, floored at 10% of the median (and 0.5) so a
        very regular history does not make every wobble an alarm."""
        out = []
        if hour["coverage_s"] < self.min_coverage_s:
            return out
        for m in METRICS:
            v = hour.get(m)
            st = self.stats.get((_slot(hour["hour"]), m))
            if v is None or st is None or st["n"] < self.min_history:
                continue
            scale = max(1.4826 * st["mad"], 0.1 * abs(st["median"]), 0.5)
            z = (v - st["median"]) / scale
            if abs(z) >= z_min:
                out.append({"type": "anomaly", "kind": "pattern_" + ("high" if z > 0 else "low"), "metric": m, "hour": hour["hour"],
                            "value": v, "typical": round(st["median"], 2), "z": round(z, 1), "history_n": st["n"]})
        return out
