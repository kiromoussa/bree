"""SIMULATED store signals for demos, screenshots and tests. Nothing here measures a real store.

Customers arrive by a made-up hourly curve, shop in zones, then queue at one register with a single
server; shelves drain and get restocked, a cooler door is occasionally left open, and one camera has
an outage. Every signal is written with source='simulated'.
"""
from __future__ import annotations

import datetime as dt
import heapq

import numpy as np

from bree.product.signals import Signal, SignalLog, day_bounds, tzinfo

# arrivals per hour, weekday; gas station c-store: commute peaks, lunch, evening, quiet night
WEEKDAY = [3, 2, 2, 2, 4, 9, 18, 26, 22, 14, 13, 17, 21, 15, 12, 14, 20, 25, 18, 12, 9, 7, 5, 4]
WEEKEND = [4, 3, 2, 2, 3, 6, 10, 15, 19, 21, 20, 20, 21, 19, 17, 17, 18, 19, 17, 14, 12, 10, 7, 5]
ZONES = ("cooler_drinks", "snack_aisle", "register")
SHELVES = (("ch15", "cooler_door_1"), ("ch15", "cooler_door_2"), ("ch14", "snack_rack"), ("ch10", "coffee_station"))


def simulate_day(day: dt.date, tz, rng, store="demo-store", cameras=("ch07", "ch15", "ch29"),
                 outage: tuple[float, float] | None = None) -> list[Signal]:
    t0, t1 = day_bounds(day, tz)
    rate = WEEKEND if day.weekday() >= 5 else WEEKDAY
    sigs: list[Signal] = []

    def add(ts, kind, value=1.0, camera="ch07", zone=None, **meta):
        if outage and camera in cameras[:1] and outage[0] <= ts < outage[1]:
            return
        sigs.append(Signal(ts, kind, value, store, camera, zone, "simulated", meta))

    # customers
    arrivals = []
    for h in range(24):
        n = rng.poisson(rate[h] * rng.uniform(0.85, 1.15))
        arrivals += list(t0 + h * 3600 + rng.uniform(0, 3600, n))
        if h in (7, 12, 17) and rng.random() < 0.6:          # a burst: a few cars pull in together
            c = t0 + h * 3600 + rng.uniform(0, 3300)
            arrivals += list(c + rng.uniform(0, 90, rng.integers(3, 7)))
    arrivals.sort()
    ready = []        # (time ready for register, id, shop_start)
    visits = []       # per customer: (arrive, [(zone, start, end)], done)
    for i, a in enumerate(arrivals):
        t = a + rng.uniform(5, 15)
        stops = []
        for z in rng.permutation(ZONES[:2])[: rng.integers(0, 3)]:
            d = float(np.clip(rng.lognormal(3.3, 0.6), 6, 600))
            stops.append((z, t, t + d))
            t += d + rng.uniform(3, 12)
        visits.append([a, stops, t])
        heapq.heappush(ready, (t, i))
    # register: single server FIFO
    free = t0
    q_events = []         # (ts, +1/-1)
    done = {}
    while ready:
        tr, i = heapq.heappop(ready)
        start = max(tr, free)
        svc = float(np.clip(rng.lognormal(4.0, 0.4), 20, 300))
        free = start + svc
        q_events += [(tr, +1), (start, -1)]
        visits[i][1].append(("register", start, free))
        done[i] = free + rng.uniform(4, 12)
    for i, (a, stops, _) in enumerate(visits):
        add(a, "door_in", 1, "ch07")
        add(done[i], "door_out", 1, "ch07")
        for z, s, e in stops:
            cam = {"cooler_drinks": "ch15", "snack_aisle": "ch14", "register": "ch29"}[z]
            if e - s >= 3:
                add(e, "dwell", round(e - s, 1), cam, z)
    # sampled series every 30 s
    arr = np.array([(visits[i][0], done[i]) for i in range(len(visits))]) if visits else np.zeros((0, 2))
    qe = sorted(q_events)
    qts = np.array([t for t, _ in qe]) if qe else np.zeros(0)
    qcum = np.cumsum([d for _, d in qe]) if qe else np.zeros(0)
    t = t0
    while t < t1:
        n = int(((arr[:, 0] <= t) & (arr[:, 1] > t)).sum()) if len(arr) else 0
        k = np.searchsorted(qts, t, side="right")
        q = int(qcum[k - 1]) if k else 0
        add(t, "occupancy", n, "ch07")
        add(t, "queue", max(q, 0), "ch29")
        for cam in cameras:
            add(t, "health", 1, cam, fps=round(float(rng.normal(9, 0.4)), 1))
        t += 30
    # shelves: fullness drains with traffic; staff restock at ~06:00 and ~15:00, and sometimes forget
    for cam, zone in SHELVES:
        level = float(rng.uniform(0.7, 1.0))
        restocks = [t0 + (6 + rng.uniform(-0.5, 0.5)) * 3600, t0 + (15 + rng.uniform(-1, 1)) * 3600]
        restocks = [r for r in restocks if rng.random() > 0.08]
        door_open_until = 0.0
        t = t0
        while t < t1:
            h = int((t - t0) // 3600)
            level -= rate[h] / 12 * 0.003 * rng.uniform(0.5, 1.5)
            while restocks and t >= restocks[0]:
                level = float(rng.uniform(0.9, 1.0))
                restocks.pop(0)
            level = max(level, 0.02)
            state = "ok" if level > 0.3 else "low" if level > 0.12 else "empty"
            if zone.startswith("cooler") and t > door_open_until and rng.random() < 0.004:
                door_open_until = t + rng.uniform(240, 900)
            if t < door_open_until:
                state = "door_open"
            add(t, "shelf", round(level, 2), cam, zone, state=state)
            t += 300
    return sigs


def populate_log(folder, start_day: dt.date, days: int = 21, seed: int = 0, tz: str | None = "America/Chicago",
                 store: str = "demo-store") -> SignalLog:
    log = SignalLog(folder, tz)
    rng = np.random.default_rng(seed)
    z = tzinfo(tz)
    for i in range(days):
        d = start_day + dt.timedelta(days=i)
        outage = None
        if i == 5:
            a = day_bounds(d, z)[0] + 14 * 3600
            outage = (a, a + 50 * 60)
        log.append(simulate_day(d, z, rng, store, outage=outage))
    return log
