"""The owner dashboard: a small stdlib HTTP server over the signal log and the review store.

  GET  /                      the page (web/index.html; plain JS, no CDN)
  GET  /api/state             live tiles: people inside, register line, today's door count, cameras, shelves, attention list
  GET  /api/day?date=         one day's report (charts and tables)
  GET  /api/days              days that have data
  GET  /api/review            alerts waiting for a decision (+ recent decisions)
  POST /api/review/decide     {"id", "decision", "reviewer", "note"} -> ReviewStore.decide
  GET  /clips/<id>            the alert's head-pixelated clip (range requests supported)
  GET  /digest/daily?date=    printable daily digest
  GET  /digest/weekly?end=    printable weekly digest (7 days ending at `end`)
  GET  /api/notifications     what the notifier fired lately

Binds to 127.0.0.1. Any other host needs a token (cookie, bearer header, or ?token=), so a phone on the
store wifi can be given access deliberately. POSTs must be JSON (a browser form from another site cannot send that).
Nothing here sends anything off the machine.
"""
from __future__ import annotations

import datetime as dt
import hmac
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bree.product import clips, digest, metrics
from bree.product.notify import FileSink, LogSink, Notification, Notifier
from bree.product.signals import SignalLog, day_bounds, local_date
from bree.product.station import Station

WEB = Path(__file__).parent / "web"
STALE_S = 120.0
OFFLINE_S = 600.0
DECISION_LABELS = {"confirmed_theft": "Theft", "not_theft": "Not theft", "unclear": "Can't tell"}


class Monitor:
    """Feeds new signals and new alerts to the notifier. `tick(now)` is called by a thread and by tests."""
    def __init__(self, log: SignalLog, station: Station, notifier: Notifier, review=None, replay: bool = False):
        self.log, self.station, self.notifier, self.review, self.replay = log, station, notifier, review, replay
        self.cursor: float | None = None
        self.notified_alerts: set[str] = set()
        self.last_heard: dict[str, float] = {}
        self._primed = False

    def tick(self, now: float | None = None) -> list[Notification]:
        now = time.time() if now is None else now
        fired: list[Notification] = []
        if self.cursor is None and self.replay:     # replaying old data: only news from now on, not the whole history
            self.cursor = self.log.latest_ts() or 0.0
        new = self.log.read(since=self.cursor if self.cursor is not None else now - 3600)
        new = [s for s in new if self.cursor is None or s.ts > self.cursor]
        if new:
            self.cursor = new[-1].ts
            for s in new:
                if s.camera != "store":
                    self.last_heard[s.camera] = max(self.last_heard.get(s.camera, 0.0), s.ts)
            fired += self.notifier.feed(new)
        if not self.replay:
            fired += self.notifier.check_silence(self.last_heard, now)
        if self.review is not None:
            for a in self.review.alerts():
                if a["alert_id"] in self.notified_alerts:
                    continue
                self.notified_alerts.add(a["alert_id"])
                if self._primed and a["tier"] == "alert" and not a["decision"] and not a["purged_ts"]:
                    n = Notification(a["event_ts"], "basket_flagged", "urgent",
                                     f"A basket was flagged at {self.station.label(a['camera'] or '')}. Please take a look in Review.",
                                     a["camera"], a["zone"], a["confidence"], "live")
                    for sink in self.notifier.sinks:
                        sink.send(n)
                    fired.append(n)
            self._primed = True
        return fired


class ProductApp:
    def __init__(self, log: SignalLog, station: Station, review=None, notif_file: str | Path | None = None,
                 replay: bool = False, now_fn=time.time):
        self.log, self.station, self.review, self.replay = log, station, review, replay
        self.tz = log.tz
        self.now_fn = now_fn
        self.notif = FileSink(notif_file or Path(log.dir) / "notifications.jsonl")
        self.notifier = Notifier(station.rules, [LogSink(), self.notif])
        self.monitor = Monitor(log, station, self.notifier, review, replay)
        self._cache: dict = {}

    # ---- time
    def now(self) -> float:
        if self.replay:
            return self.log.latest_ts() or self.now_fn()
        return self.now_fn()

    def _day(self, d: dt.date, alerts=None):
        a, b = day_bounds(d, self.tz)
        sigs = self.log.read(a, b)
        key = (d, len(sigs), sigs[-1].ts if sigs else 0, len(alerts or []))
        if self._cache.get("day") and self._cache["day"][0] == key:
            return self._cache["day"][1], sigs
        rep = metrics.day_report(sigs, a, b, self.tz, alerts)
        self._cache["day"] = (key, rep)
        return rep, sigs

    def _alerts_between(self, a, b):
        return self.review.alerts(a, b) if self.review is not None else []

    # ---- api
    def api_state(self) -> dict:
        now = self.now()
        today = local_date(now, self.tz)
        d0, _ = day_bounds(today, self.tz)
        sigs = self.log.read(d0, now + 1)
        recent = [s for s in sigs if s.ts >= now - STALE_S]
        occ_now = [s for s in recent if s.kind == "occupancy"]
        store_occ = [s for s in occ_now if s.camera == "store"]
        occ = None
        if store_occ:
            occ = store_occ[-1].value
        elif occ_now:
            last: dict[str, float] = {}
            for s in occ_now:
                last[s.camera] = s.value
            occ = max(last.values())
        q_now = [s for s in recent if s.kind == "queue"]
        q = None
        if q_now:
            last = {}
            for s in q_now:
                last[s.camera] = s.value
            q = max(last.values())
        door = metrics.door_totals(sigs)
        up = metrics.camera_uptime(sigs, d0, now)
        cams = []
        for cid in sorted(set(up) | set(self.station.cameras)):
            heard = up.get(cid, {}).get("last_ts")
            age = None if heard is None else now - heard
            gap = up.get(cid, {}).get("gap_s") or 0.0         # a camera that reports every 5 min is not stale after 2
            stale, off = max(STALE_S, 3 * gap), max(OFFLINE_S, 6 * gap)
            status = "unknown" if age is None else "ok" if age <= stale else "stale" if age <= off else "offline"
            cams.append({"id": cid, "label": self.station.label(cid), "status": status, "age_s": age,
                         "role": self.station.cameras[cid].role if cid in self.station.cameras else None})
        shelves = metrics.shelf_latest([s for s in sigs if s.ts >= now - 1800])
        for r in shelves:
            r["label"] = self.station.label(r["camera"])
        pending = self._pending()
        by_hour = metrics.traffic_by_hour(sigs, self.tz)
        series = [p for p in metrics.occupancy_series(sigs) if p["ts"] >= now - 3 * 3600]
        attention = []
        for r in shelves:
            if r["state"] != "ok":
                attention.append({"kind": "shelf", "text": f"{r['zone']}: {r['state'].replace('_', ' ')}", "camera": r["camera"]})
        for c in cams:
            if c["status"] in ("stale", "offline") and not self.replay:     # saved clips have gaps by construction
                attention.append({"kind": "camera", "text": f"{c['label']} has not reported for {int((c['age_s'] or 0) // 60)} min", "camera": c["id"]})
        if pending:
            attention.append({"kind": "review", "text": f"{len(pending)} item(s) waiting for your review"})
        return {"now": now, "replay": self.replay, "store": self.station.name, "tz": str(self.tz),
                "tz_assumed": self.station.tz_assumed, "date": today.isoformat(),
                "occupancy": occ, "queue": q, "door_today": door, "by_hour": by_hour, "occupancy_recent": series,
                "cameras": cams, "shelves": shelves, "attention": attention, "pending_review": len(pending),
                "sources": metrics.source_mix(sigs), "notes": self.station.notes,
                "data_age_s": (now - sigs[-1].ts) if sigs else None}

    def api_day(self, date: str | None) -> dict:
        d = dt.date.fromisoformat(date) if date else local_date(self.now(), self.tz)
        a, b = day_bounds(d, self.tz)
        rep, _ = self._day(d, self._alerts_between(a, b))
        rep = {k: v for k, v in rep.items() if k != "shelf_now"}
        rep["date"] = d.isoformat()
        rep["tz"] = str(self.tz)
        return rep

    def api_days(self) -> list[str]:
        return [d.isoformat() for d in self.log.days()]

    def _pending(self) -> list[dict]:
        if self.review is None:
            return []
        return [a for a in self.review.alerts() if not a["decision"] and not a["purged_ts"]
                and a["retracted_tier"] != "retracted"]

    def api_review(self) -> dict:
        if self.review is None:
            return {"pending": [], "recent": []}

        def view(a):
            return {"id": a["alert_id"], "time": a["event_ts"], "camera": a["camera"], "label": self.station.label(a["camera"] or ""),
                    "tier": a["tier"], "confidence": a["confidence"], "items": [i.get("category") or i.get("sku") or "item" for i in a["items"]],
                    "reasons": a["reasons"], "clip": bool(self.review.clip_file(a["alert_id"])), "decision": a["decision"],
                    "retracted": a["retracted_tier"], "visited_register": (a.get("register") or {}).get("visited_register") if isinstance(a.get("register"), dict) else None}
        allr = [a for a in self.review.alerts() if not a["purged_ts"]]
        pend = [view(a) for a in allr if not a["decision"] and a["retracted_tier"] != "retracted"]
        rec = [view(a) for a in allr if a["decision"]][-20:]
        return {"pending": pend[::-1], "recent": rec[::-1], "decisions": DECISION_LABELS}

    def decide(self, body: dict) -> dict:
        if self.review is None:
            raise ValueError("no review store configured")
        dec = body.get("decision")
        if dec not in DECISION_LABELS:
            raise ValueError(f"decision must be one of {list(DECISION_LABELS)}")
        return self.review.decide(str(body.get("id", "")), str(body.get("reviewer", "")), dec, note=str(body.get("note", ""))[:500])

    def api_notifications(self) -> list[dict]:
        return self.notif.read(50)[::-1]

    # ---- digests
    def daily_html(self, date: str | None) -> str:
        d = dt.date.fromisoformat(date) if date else local_date(self.now(), self.tz)
        a, b = day_bounds(d, self.tz)
        al = self._alerts_between(a, b)
        rep, _ = self._day(d, al)
        pend = [x for x in al if not x["decision"] and not x["purged_ts"]]
        return digest.daily_digest(rep, store_name=self.station.name, day=d, tz=self.tz, review_pending=pend,
                                   tz_assumed=self.station.tz_assumed, notes=self.station.notes)

    def weekly_html(self, end: str | None) -> str:
        e = dt.date.fromisoformat(end) if end else local_date(self.now(), self.tz)
        days = []
        for i in range(6, -1, -1):
            d = e - dt.timedelta(days=i)
            a, b = day_bounds(d, self.tz)
            days.append((d, self._day(d, self._alerts_between(a, b))[0]))
        prev = 0
        for i in range(13, 6, -1):
            d = e - dt.timedelta(days=i)
            prev += metrics.door_totals(self.log.read(*day_bounds(d, self.tz)))["in"]
        has_prev = prev > 0
        return digest.weekly_digest(days, store_name=self.station.name, tz=self.tz, prev_total_in=prev if has_prev else None,
                                    tz_assumed=self.station.tz_assumed, notes=self.station.notes)


def serve(app: ProductApp, host: str = "127.0.0.1", port: int = 8090, token: str | None = None,
          monitor_every_s: float | None = 10.0) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        raise ValueError("binding to a non-local address needs a token")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'")
            for k, v in extra:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200, extra=()):
            self._send(code, json.dumps(obj).encode(), "application/json", extra)

        def _auth(self, q) -> tuple[bool, tuple]:
            if not token:
                return True, ()
            cands = [q.get("token", [""])[0], self.headers.get("Authorization", "").removeprefix("Bearer ").strip()]
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "bree_token":
                    cands.append(v)
            ok = any(c and hmac.compare_digest(c, token) for c in cands)
            extra = ()
            if ok and q.get("token"):
                extra = (("Set-Cookie", f"bree_token={token}; Path=/; HttpOnly; SameSite=Strict"),)
            return ok, extra

        def do_GET(self):  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            ok, extra = self._auth(q)
            if not ok:
                return self._send(401, b"This page needs the access link you were given.", "text/plain; charset=utf-8")
            p = u.path
            one = lambda k: q.get(k, [None])[0]  # noqa: E731
            try:
                if p == "/":
                    return self._send(200, (WEB / "index.html").read_bytes(), extra=extra)
                if p in ("/app.js", "/app.css"):
                    return self._send(200, (WEB / p[1:]).read_bytes(), "text/javascript; charset=utf-8" if p.endswith("js") else "text/css; charset=utf-8", extra)
                if p == "/api/state":
                    return self._json(app.api_state(), extra=extra)
                if p == "/api/day":
                    return self._json(app.api_day(one("date")), extra=extra)
                if p == "/api/days":
                    return self._json(app.api_days(), extra=extra)
                if p == "/api/review":
                    return self._json(app.api_review(), extra=extra)
                if p == "/api/notifications":
                    return self._json(app.api_notifications(), extra=extra)
                if p == "/digest/daily":
                    return self._send(200, app.daily_html(one("date")).encode(), extra=extra)
                if p == "/digest/weekly":
                    return self._send(200, app.weekly_html(one("end")).encode(), extra=extra)
                if p.startswith("/clips/") and app.review is not None:
                    clip = app.review.clip_file(urllib.parse.unquote(p[7:]))
                    if clip:
                        return self._clip(clips.playable(clip, Path(app.log.dir).parent / "cache" / "clips"))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def _clip(self, clip: Path):
            data = clip.read_bytes()
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                try:
                    start = int(a) if a else max(0, len(data) - int(b))
                    end = int(b) if (a and b) else len(data) - 1
                except ValueError:
                    return self._send(416, b"", "text/plain")
                end = min(end, len(data) - 1)
                if start > end:
                    return self._send(416, b"", "text/plain")
                return self._send(206, data[start:end + 1], "video/mp4", (("Content-Range", f"bytes {start}-{end}/{len(data)}"), ("Accept-Ranges", "bytes")))
            self._send(200, data, "video/mp4", (("Accept-Ranges", "bytes"),))

        def do_POST(self):  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            ok, _ = self._auth(urllib.parse.parse_qs(u.query))
            if not ok:
                return self._send(401, b"unauthorized", "text/plain")
            if u.path != "/api/review/decide":
                return self._send(404, b"not found", "text/plain")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, b"JSON only", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 20000:
                    return self._send(413, b"too large", "text/plain")
                self._json(app.decide(json.loads(self.rfile.read(n) or b"{}")))
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, 400)

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    if monitor_every_s:
        stop = threading.Event()

        def loop():
            while not stop.wait(monitor_every_s):
                try:
                    app.monitor.tick(app.now_fn())
                except Exception:
                    import logging
                    logging.getLogger("bree.product").exception("monitor tick failed")
        threading.Thread(target=loop, daemon=True).start()
        srv.stop_monitor = stop.set
    return srv
