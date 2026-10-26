"""Hourly and daily aggregates of the flow record stream (bree.intel.flow), and the daily digest text.

Rates are per observed hour: the recorder writes segments with gaps, so every hour carries `coverage_s` (seconds of video
actually analysed) and the day carries the same. Counts are raw counts of what the camera saw during that coverage; nothing
is extrapolated to the missing time. Station time comes from the records' `ts` (file name time plus the layout's clock offset).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REGISTER_MIN_STILL_S = 2.0


def read_records(paths) -> list[dict]:
    out: list[dict] = []
    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def _hour(ts: str | None) -> str:
    return ts[:13] + ":00" if ts else "unknown"


def _q(v: list[float], p: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    return round(v[min(int(p * len(v)), len(v) - 1)], 1)


def hourly(records: list[dict]) -> dict[str, dict]:
    """{hour: metrics}. Needs "segment" records (coverage) plus door / zone / sample / track records."""
    H: dict[str, dict] = defaultdict(lambda: {"coverage_s": 0.0, "entries": 0, "exits": 0, "door_via": defaultdict(int),
                                              "samples": 0, "customers_sum": 0, "customers_max": 0, "staff_samples": 0,
                                              "queue_sum": 0, "queue_max": 0, "queue_ge2_samples": 0, "occupancy_max": None,
                                              "visits": defaultdict(int), "dwell": defaultdict(list), "register_visits": [],
                                              "outcomes": defaultdict(int), "customer_tracks": 0, "walk_s": []})
    for r in records:
        h = H[_hour(r.get("ts"))]
        k = r["type"]
        if k == "segment":
            # a segment may straddle an hour: split its coverage by the hour of its start only (5 min segments; the error is minutes)
            h["coverage_s"] += r["t1"] - r["t0"]
        elif k == "door":
            h["entries" if r["dir"] == "in" else "exits"] += 1
            h["door_via"][r["via"]] += 1
        elif k == "sample":
            h["samples"] += 1
            h["customers_sum"] += r["customers"]
            h["customers_max"] = max(h["customers_max"], r["customers"])
            h["staff_samples"] += r["staff"] > 0
            h["queue_sum"] += r["queue"]
            h["queue_max"] = max(h["queue_max"], r["queue"])
            h["queue_ge2_samples"] += r["queue"] >= 2
            if r.get("occupancy") is not None:
                h["occupancy_max"] = max(h["occupancy_max"] or 0, r["occupancy"])
        elif k == "zone":
            h["visits"][r["zone"]] += 1
            h["dwell"][r["zone"]].append(r["dwell_s"])
            if r["kind"] == "register" and r["still_s"] >= REGISTER_MIN_STILL_S:   # standing at the till, not walking past it
                h["register_visits"].append(r["dwell_s"])
        elif k == "track" and r["role"] == "customer":
            if r["outcome"] != "glimpse":
                h["customer_tracks"] += 1
                h["outcomes"][r["outcome"]] += 1
            h["walk_s"].append(r["t1"] - r["t0"])
    out = {}
    for hour, h in sorted(H.items()):
        n = max(h["samples"], 1)
        cov_h = h["coverage_s"] / 3600 if h["coverage_s"] else None
        out[hour] = {
            "hour": hour, "coverage_s": round(h["coverage_s"], 1),
            "entries": h["entries"], "exits": h["exits"], "door_via": dict(h["door_via"]),
            "entries_per_hour": round(h["entries"] / cov_h, 1) if cov_h else None,
            "customers_in_view_mean": round(h["customers_sum"] / n, 2), "customers_in_view_max": h["customers_max"],
            "staff_present_share": round(h["staff_samples"] / n, 2),
            "occupancy_max": h["occupancy_max"],
            "queue_mean": round(h["queue_sum"] / n, 2), "queue_max": h["queue_max"],
            "queue_ge2_share": round(h["queue_ge2_samples"] / n, 2),
            "register_visits": len(h["register_visits"]),
            "register_time_median_s": _q(h["register_visits"], 0.5), "register_time_p90_s": _q(h["register_visits"], 0.9),
            "zone_visits": dict(h["visits"]),
            "zone_dwell_median_s": {z: _q(v, 0.5) for z, v in h["dwell"].items()},
            "zone_dwell_p90_s": {z: _q(v, 0.9) for z, v in h["dwell"].items()},
            "customer_tracks": h["customer_tracks"], "outcomes": dict(h["outcomes"]),
        }
    return out


def daily(hours: dict[str, dict]) -> dict[str, dict]:
    """{date: {...totals, busiest_hour, hours: [...]}} from `hourly`."""
    byday: dict[str, list[dict]] = defaultdict(list)
    for hk, h in hours.items():
        byday[hk[:10]].append(h)
    out = {}
    for day, hs in sorted(byday.items()):
        cov = sum(h["coverage_s"] for h in hs)
        entries = sum(h["entries"] for h in hs)
        outc: dict[str, int] = defaultdict(int)
        for h in hs:
            for k, v in h["outcomes"].items():
                outc[k] += v
        rated = [h for h in hs if h["entries_per_hour"] is not None]
        busiest = max(rated, key=lambda h: h["entries_per_hour"])["hour"] if rated and any(h["entries"] for h in rated) else None
        reg = [h["register_time_median_s"] for h in hs if h["register_time_median_s"] is not None]
        known = sum(v for k, v in outc.items() if k in ("registered", "left_without_register", "passed_through"))
        out[day] = {"date": day, "coverage_s": round(cov, 1), "entries": entries, "exits": sum(h["exits"] for h in hs),
                    "busiest_hour": busiest, "queue_max": max(h["queue_max"] for h in hs),
                    "register_time_median_s": round(statistics.median(reg), 1) if reg else None,
                    "outcomes": dict(outc),
                    "register_visit_rate_of_known_exits": round(outc.get("registered", 0) / known, 2) if known else None,
                    "hours": hs}
    return out


def digest(day: dict) -> str:
    """Plain-text daily digest. Says what was observed and how much video it rests on."""
    cov = day["coverage_s"] / 60
    lines = [f"Store digest {day['date']}  (analysed {cov:.0f} min of video: counts below are of that footage only)",
             f"- Door: {day['entries']} entries, {day['exits']} exits seen at the door camera."]
    if day["busiest_hour"]:
        lines.append(f"- Busiest hour by entries per observed hour: {day['busiest_hour'][11:13]}:00.")
    if day["register_time_median_s"] is not None:
        lines.append(f"- Time at the register (per visit): median {day['register_time_median_s']:.0f} s; most people waiting at once in view: {day['queue_max']}.")
    o = day["outcomes"]
    if o:
        lines.append("- Customer walks seen: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(o.items(), key=lambda kv: -kv[1])) + ".")
    lines.append("- These are proxies from one camera; see docs/ultra/intel.md for what each can and cannot say.")
    return "\n".join(lines)


def to_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: dict(o) if isinstance(o, defaultdict) else str(o)))


_ = datetime  # (ts strings are ISO local time; parsed only by callers that need arithmetic)
