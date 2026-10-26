"""Pure aggregation over signals: traffic by hour, occupancy, dwell, queue episodes, shelf issues,
camera uptime. No I/O. Every function takes a list of Signal and returns plain dicts/lists, so the
dashboard, the digest and the notifier all see the same numbers.

Rules that keep the numbers honest:
- Samples are weighted by the time they cover (capped by MAX_GAP_S), not counted.
- Store occupancy from several cameras is the MAX over cameras per bucket: it can under-count
  (people seen by no camera) but never counts one person twice.
- Anything with no data says None, not 0.
"""
from __future__ import annotations

import datetime as dt
import statistics
from collections import defaultdict

from bree.product.signals import Signal, SOURCES

MAX_GAP_S = 60.0          # a sample covers at most this long (a silent camera is not "empty")
BUCKET_S = 60.0


def _of(sigs, kind):
    return [s for s in sigs if s.kind == kind]


def source_mix(sigs: list[Signal]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in sigs:
        out[s.source] = out.get(s.source, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: SOURCES.index(kv[0])))


def weakest_source(sigs: list[Signal]) -> str | None:
    mix = source_mix(sigs)
    return next(iter(mix), None)


def _weights(samples: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """(ts, value) sorted -> (ts, value, seconds covered)."""
    out = []
    for i, (t, v) in enumerate(samples):
        nxt = samples[i + 1][0] if i + 1 < len(samples) else t + 5.0
        out.append((t, v, max(0.0, min(nxt - t, MAX_GAP_S))))
    return out


def occupancy_series(sigs: list[Signal], bucket_s: float = BUCKET_S) -> list[dict]:
    """[{ts, n, cameras}] one point per bucket that has data. `n` = max over cameras of the bucket mean."""
    per: dict[tuple[float, str], list[float]] = defaultdict(list)
    for s in _of(sigs, "occupancy"):
        per[(s.ts // bucket_s * bucket_s, s.camera)].append(s.value)
    store_cam = {k for k in per if k[1] == "store"}
    by_bucket: dict[float, dict[str, float]] = defaultdict(dict)
    for (b, cam), vals in per.items():
        by_bucket[b][cam] = sum(vals) / len(vals)
    out = []
    for b in sorted(by_bucket):
        cams = by_bucket[b]
        n = cams["store"] if (b, "store") in store_cam else max(cams.values())
        out.append({"ts": b, "n": round(n, 2), "cameras": len(cams)})
    return out


def traffic_by_hour(sigs: list[Signal], tz) -> list[dict]:
    """People through the door per store-local clock hour: [{hour_ts, hour, in, out}] for hours with data."""
    hrs: dict[float, dict] = {}
    for s in sigs:
        if s.kind not in ("door_in", "door_out"):
            continue
        d = dt.datetime.fromtimestamp(s.ts, tz).replace(minute=0, second=0, microsecond=0)
        h = hrs.setdefault(d.timestamp(), {"hour_ts": d.timestamp(), "hour": d.hour, "in": 0, "out": 0})
        h["in" if s.kind == "door_in" else "out"] += int(round(s.value))
    return [hrs[k] for k in sorted(hrs)]


def door_totals(sigs: list[Signal]) -> dict:
    n_in = sum(int(round(s.value)) for s in _of(sigs, "door_in"))
    n_out = sum(int(round(s.value)) for s in _of(sigs, "door_out"))
    return {"in": n_in, "out": n_out}


def busiest_hour(by_hour: list[dict]) -> dict | None:
    return max(by_hour, key=lambda h: h["in"], default=None) if any(h["in"] for h in by_hour) else None


def occupancy_summary(series: list[dict]) -> dict:
    if not series:
        return {"peak": None, "peak_ts": None, "mean": None}
    pk = max(series, key=lambda p: p["n"])
    return {"peak": pk["n"], "peak_ts": pk["ts"], "mean": round(sum(p["n"] for p in series) / len(series), 2)}


def dwell_stats(sigs: list[Signal]) -> dict[str, dict]:
    """Per zone: visits, median and 90th percentile seconds. Needs >= 1 visit; p90 needs >= 5."""
    per: dict[str, list[float]] = defaultdict(list)
    for s in _of(sigs, "dwell"):
        per[s.zone].append(s.value)
    out = {}
    for z, v in sorted(per.items()):
        v.sort()
        out[z] = {"visits": len(v), "median_s": round(statistics.median(v), 1),
                  "p90_s": round(v[min(len(v) - 1, int(0.9 * len(v)))], 1) if len(v) >= 5 else None,
                  "total_s": round(sum(v), 1)}
    return out


def queue_episodes(sigs: list[Signal], min_len: int = 2, merge_gap_s: float = 60.0) -> list[dict]:
    """Stretches where >= min_len people waited. [{start, end, peak, seconds}] sorted by start."""
    samples = sorted((s.ts, s.value) for s in _of(sigs, "queue"))
    eps: list[dict] = []
    for t, v, cover in _weights(samples):
        if v < min_len:
            continue
        if eps and t - eps[-1]["end"] <= merge_gap_s:
            e = eps[-1]
            e["end"] = t + cover
            e["peak"] = max(e["peak"], v)
        else:
            eps.append({"start": t, "end": t + cover, "peak": v})
    for e in eps:
        e["seconds"] = round(e["end"] - e["start"], 1)
        e["peak"] = int(e["peak"])
    return eps


def queue_summary(sigs: list[Signal]) -> dict:
    samples = sorted((s.ts, s.value) for s in _of(sigs, "queue"))
    if not samples:
        return {"peak": None, "mean": None, "episodes": [], "covered_s": 0.0, "share_2plus": None}
    w = _weights(samples)
    cov = sum(c for _, _, c in w)
    mean = sum(v * c for _, v, c in w) / cov if cov else None
    over = sum(c for _, v, c in w if v >= 2)
    eps = queue_episodes(sigs)
    return {"peak": int(max(v for _, v in samples)), "mean": None if mean is None else round(mean, 2),
            "episodes": sorted(eps, key=lambda e: (-e["peak"], -e["seconds"]))[:5],
            "covered_s": round(cov, 1), "share_2plus": round(over / cov, 3) if cov else None}


def shelf_latest(sigs: list[Signal]) -> list[dict]:
    """Latest reading per (camera, zone): [{camera, zone, fullness, state, ts, source}], worst first."""
    last: dict[tuple, Signal] = {}
    for s in _of(sigs, "shelf"):
        last[(s.camera, s.zone)] = s
    rows = [{"camera": c, "zone": z, "fullness": round(s.value, 2), "state": s.meta.get("state", "ok"),
             "ts": s.ts, "source": s.source} for (c, z), s in last.items()]
    return sorted(rows, key=lambda r: (r["state"] == "ok", r["fullness"]))


def shelf_issues(sigs: list[Signal]) -> list[dict]:
    """Episodes where a section was not ok: [{camera, zone, state, start, end, seconds, min_fullness}]."""
    per: dict[tuple, list[Signal]] = defaultdict(list)
    for s in _of(sigs, "shelf"):
        per[(s.camera, s.zone)].append(s)
    eps = []
    for (cam, zone), ss in per.items():
        cur = None
        for s in ss:
            state = s.meta.get("state", "ok")
            if state != "ok":
                if cur and cur["state"] == state:
                    cur["end"] = s.ts
                    cur["min_fullness"] = min(cur["min_fullness"], s.value)
                else:
                    if cur:
                        eps.append(cur)
                    cur = {"camera": cam, "zone": zone, "state": state, "start": s.ts, "end": s.ts,
                           "min_fullness": s.value, "source": s.source}
            elif cur:
                cur["end"] = s.ts
                eps.append(cur)
                cur = None
        if cur:
            eps.append(cur)
    for e in eps:
        e["seconds"] = round(e["end"] - e["start"], 1)
        e["min_fullness"] = round(e["min_fullness"], 2)
    return sorted(eps, key=lambda e: e["start"])


def camera_uptime(sigs: list[Signal], start: float, end: float) -> dict[str, dict]:
    """Per camera, from health heartbeats (and any other signal as proof of life): share of the window
    covered, and the last time heard. Coverage counts a gap longer than MAX_GAP_S as down."""
    per: dict[str, list[float]] = defaultdict(list)
    for s in sigs:
        if s.camera != "store" and (s.kind != "health" or s.value > 0):
            per[s.camera].append(s.ts)
    out = {}
    span = max(end - start, 1e-9)
    for cam, ts in sorted(per.items()):
        ts.sort()
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
        cap = max(MAX_GAP_S, 2.5 * gaps[len(gaps) // 2]) if gaps else MAX_GAP_S   # a camera that reports every 5 min is not down between reports
        cov = sum(min(b - a, cap) for a, b in zip(ts, ts[1:]))
        out[cam] = {"covered_s": round(cov, 1), "share": round(min(1.0, cov / span), 3), "last_ts": ts[-1],
                    "gap_s": round(gaps[len(gaps) // 2], 1) if gaps else None}
    return out


def sales(sigs: list[Signal]) -> dict | None:
    s = _of(sigs, "pos_sale")
    return {"count": len(s), "total_usd": round(sum(x.value for x in s), 2)} if s else None


def shrink_summary(alerts: list[dict]) -> dict:
    """From ReviewStore.alerts() rows: how many would-be/real alerts, what reviewers said, items at risk.
    Never an estimate of dollars lost: only what reviewers confirmed."""
    live = [a for a in alerts if not a.get("purged_ts") and a.get("retracted_tier") != "retracted"]
    dec = defaultdict(int)
    for a in live:
        dec[a.get("decision") or "unreviewed"] += 1
    items: dict[str, int] = defaultdict(int)
    for a in live:
        if a.get("decision") == "confirmed_theft":
            for it in a.get("items") or []:
                items[it.get("category") or it.get("sku") or "item"] += 1
    reviewed = sum(v for k, v in dec.items() if k in ("confirmed_theft", "not_theft", "wrong_item"))
    return {"alerts": len(live), "by_decision": dict(dec), "confirmed_items": dict(items),
            "false_alert_share": round(dec.get("not_theft", 0) / reviewed, 3) if reviewed >= 5 else None,
            "by_tier": {t: sum(1 for a in live if a.get("tier") == t) for t in ("alert", "review")}}


def day_report(sigs: list[Signal], start: float, end: float, tz, alerts: list[dict] | None = None) -> dict:
    """Everything the digest and the dashboard history need for [start, end)."""
    win = [s for s in sigs if start <= s.ts < end]
    series = occupancy_series(win)
    by_hour = traffic_by_hour(win, tz)
    return {"start": start, "end": end, "signals": len(win), "sources": source_mix(win),
            "door": door_totals(win), "by_hour": by_hour, "busiest": busiest_hour(by_hour),
            "occupancy": occupancy_summary(series), "occupancy_series": series,
            "dwell": dwell_stats(win), "queue": queue_summary(win), "shelf_issues": shelf_issues(win),
            "shelf_now": shelf_latest(win), "cameras": camera_uptime(win, start, end), "sales": sales(win),
            "shrink": shrink_summary(alerts or [])}
