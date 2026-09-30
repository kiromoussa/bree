"""Tiny local dashboard: live alerts, per-person baskets, latest annotated frame.

Runs the pipeline in a background thread and serves (stdlib only):
  /            the page (polls /state every second)
  /state       JSON: people in store + baskets, alerts, recent events, FPS
  /frame.jpg   latest annotated frame (heads pixelated)
Binds to 127.0.0.1 by default: nothing leaves the machine.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BREE live</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e3de;--alert:#c62828;--review:#b26a00;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#1f1f1e;--line:#33332f}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0} .muted{color:var(--muted)} main{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;padding:16px}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
img{width:100%;border-radius:6px;display:block} h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
.alert{border-left:4px solid var(--alert);padding:6px 10px;margin:6px 0}.review{border-left:4px solid var(--review);padding:6px 10px;margin:6px 0}
table{width:100%;border-collapse:collapse} td{padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
.tag{font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid var(--line)}
</style></head><body>
<header><h1>BREE live</h1><span class="muted" id="src"></span><span class="muted" id="fps"></span></header>
<main><section class="card"><h2>Camera</h2><img id="frame" alt="latest annotated frame (heads pixelated)"></section>
<section><div class="card"><h2>Alerts</h2><div id="alerts" class="muted">none yet</div></div>
<div class="card" style="margin-top:16px"><h2>In store now</h2><table id="people"></table></div>
<div class="card" style="margin-top:16px"><h2>Recent events</h2><table id="events"></table></div></section></main>
<script>
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function tick(){try{const s=await (await fetch('/state')).json();
document.getElementById('src').textContent=s.source;document.getElementById('fps').textContent=s.fps.toFixed(1)+' FPS · t='+s.t.toFixed(1)+'s'+(s.done?' · finished':'');
document.getElementById('alerts').innerHTML=s.alerts.length?s.alerts.slice().reverse().map(a=>`<div class="${a.tier}"><b>${a.tier.toUpperCase()}</b> person ${a.person_id} · conf ${a.confidence.toFixed(2)} · ${esc(a.unpaid_items.map(i=>i.category).join(', '))}<br><span class="muted">${esc(a.reasons.join('; '))}</span></div>`).join(''):'none yet';
document.getElementById('people').innerHTML=s.people.map(p=>`<tr><td>ID ${p.id}</td><td>${esc(p.basket.join(', ')||'—')}</td><td>${p.paid.length?'<span class="tag">paid: '+esc(p.paid.join(', '))+'</span>':''}</td></tr>`).join('')||'<tr><td class="muted">nobody</td></tr>';
document.getElementById('events').innerHTML=s.events.slice().reverse().map(e=>`<tr><td>${e.t.toFixed(1)}s</td><td>P${e.person_id}</td><td>${e.type}</td><td>${esc(e.item||'')} ${esc(e.zone||'')}</td></tr>`).join('');
document.getElementById('frame').src='/frame.jpg?'+Date.now();}catch(e){}}
setInterval(tick,1000);tick();
</script></body></html>"""


class DashboardState:
    def __init__(self, source: str):
        self.lock = threading.Lock()
        self.source = source
        self.jpeg: bytes | None = None
        self.alerts: list[dict] = []
        self.events: list[dict] = []
        self.people: list[dict] = []
        self.t = 0.0
        self.fps = 0.0
        self.done = False
        self._last = time.time()

    def on_frame(self, fr, obs, events, ledger, annotated=None):
        now = time.time()
        with self.lock:
            self.t = fr.t
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last, 1e-6)
            self._last = now
            self.events = (self.events + [e.to_dict() for e in events if e.type.value != "enter"])[-15:]
            self.people = [{"id": pid, "basket": [it.category for it in r.basket],
                            "paid": [li.category or li.sku for li in r.paid]}
                           for pid, r in ledger.people.items() if r.t_exit is None and pid in ledger.active]
            if annotated is not None:
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg = buf.tobytes()

    def on_alert(self, alert):
        with self.lock:
            self.alerts.append(alert.to_dict())

    def snapshot(self) -> dict:
        with self.lock:
            return {"source": self.source, "t": self.t, "fps": self.fps, "done": self.done,
                    "alerts": self.alerts, "events": self.events, "people": self.people}


def serve(state: DashboardState, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/state":
                body, ctype = json.dumps(state.snapshot()).encode(), "application/json"
            elif path == "/frame.jpg" and state.jpeg:
                body, ctype = state.jpeg, "image/jpeg"
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
