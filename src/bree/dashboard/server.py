"""Tiny local dashboard: live alerts, per-person baskets, latest annotated frame.

Runs the pipeline in a background thread and serves (stdlib only):
  /            the page (polls /state every second)
  /state       JSON: people in store + baskets, alerts, recent events, FPS
  /frame.jpg   latest annotated frame (heads pixelated)
With a shadow-mode log (`review=ShadowLog`), also:
  /review                 label would-be alerts: real theft / false alert / unsure
  /api/review             JSON: would-be alerts with their latest label + summary
  /api/review/label       POST {"id", "label", "reviewer", "note"} -> appended to labels.jsonl
  /clips/<id>             that alert's head-pixelated clip
Binds to 127.0.0.1 by default: nothing leaves the machine.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
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


REVIEW_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BREE shadow review</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e3de;--alert:#c62828;--review:#b26a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#1f1f1e;--line:#33332f}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0} .muted{color:var(--muted)} main{padding:16px;display:grid;gap:12px;max-width:1100px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;display:grid;grid-template-columns:minmax(0,2fr) minmax(0,3fr);gap:12px}
@media (max-width:760px){.card{grid-template-columns:1fr}}
.card.alert{border-left:4px solid var(--alert)}.card.review{border-left:4px solid var(--review)}
video{width:100%;border-radius:6px;background:#000} a{color:inherit}
input,button{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
button{cursor:pointer} button.on{background:var(--fg);color:var(--bg)} .tag{font-size:11px;padding:1px 6px;border-radius:10px;border:1px solid var(--line)}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:6px} pre{white-space:pre-wrap;font-size:12px;margin:4px 0}
</style></head><body>
<header><h1>BREE shadow review</h1><span class="muted">Would-be alerts. Staff never saw these.</span>
<label>Reviewer <input id="who" placeholder="your name"></label>
<label><input type="checkbox" id="todo"> unlabelled only</label><span class="muted" id="sum"></span></header>
<main id="list" class="muted">loading</main>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
try{$('who').value=localStorage.getItem('who')||''}catch(e){}
$('who').onchange=()=>{try{localStorage.setItem('who',$('who').value)}catch(e){}};
const NAMES={real_theft:'Real theft',false_alert:'False alert',unsure:'Unsure'};
function pct(p){return p==null?'n/a':(100*p).toFixed(0)+'%'}
async function load(){const d=await (await fetch('/api/review')).json();const s=d.summary;
$('sum').textContent=`${s.labelled}/${s.would_be_alerts} labelled · precision: alert tier ${pct(s.alert.precision)}, review tier ${pct(s.review.precision)}`;
const rows=d.alerts.slice().reverse().filter(a=>!$('todo').checked||!a.label);
$('list').innerHTML=rows.map(a=>{const clip='/clips/'+encodeURIComponent(a.id);return `<div class="card ${a.tier}" data-id="${esc(a.id)}">
<div>${a.clip?`<video src="${clip}" controls muted loop playsinline preload="metadata"></video><div class="muted"><a href="${clip}" download>download clip</a> (heads pixelated)</div>`:'<span class="muted">no clip</span>'}</div>
<div><b>${a.tier.toUpperCase()}</b> · conf ${a.confidence.toFixed(2)} · ${esc(a.camera)} · person ${a.person_id} · ${esc(a.time)}
${a.concealment_seen?' <span class="tag">concealment seen</span>':''}
<div>Basket: ${esc(a.basket.join(', ')||'none')}</div><div>Unpaid: ${esc(a.unpaid.map(i=>i.category).join(', '))}</div>
<div class="muted">${esc(a.reasons.join('; '))}</div>
<details><summary class="muted">audit log</summary><pre>${esc(a.audit_log.join('\\n'))}</pre></details>
<div class="row">${Object.keys(NAMES).map(k=>`<button data-l="${k}" class="${a.label&&a.label.label===k?'on':''}">${NAMES[k]}</button>`).join('')}
<input class="note" placeholder="note (optional)" value="${esc(a.label?a.label.note:'')}"></div>
<div class="muted">${a.label?`labelled ${esc(NAMES[a.label.label])} by ${esc(a.label.reviewer)} at ${esc(a.label.ts)}`:'not labelled yet'}</div></div></div>`}).join('')||'nothing to review';}
$('list').onclick=async e=>{const b=e.target.closest('button[data-l]');if(!b)return;const card=b.closest('.card');
if(!$('who').value.trim()){alert('Enter your name first');$('who').focus();return}
const r=await fetch('/api/review/label',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({id:card.dataset.id,label:b.dataset.l,reviewer:$('who').value,note:card.querySelector('.note').value})});
if(!r.ok){alert(await r.text());return}load();};
$('todo').onchange=load;load();
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


def serve(state: DashboardState | None, host: str = "127.0.0.1", port: int = 8080,
          review=None) -> ThreadingHTTPServer:
    """`state`: the live pipeline (None = review only). `review`: a bree.shadow.ShadowLog."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_clip(self, data: bytes):
            # Safari only plays <video> from servers that answer byte ranges.
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes=") and "-" in rng:
                a, b = rng[6:].split(",")[0].split("-")
                start = int(a) if a else max(0, len(data) - int(b))
                end = min(int(b), len(data) - 1) if a and b else len(data) - 1
                return self._send(206, data[start:end + 1], "video/mp4",
                                  [("Content-Range", f"bytes {start}-{end}/{len(data)}"), ("Accept-Ranges", "bytes")])
            self._send(200, data, "video/mp4", [("Accept-Ranges", "bytes")])

        def do_GET(self):  # noqa: N802
            path = urllib.parse.unquote(self.path.split("?")[0])
            if path == "/" and state is not None:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/state" and state is not None:
                body, ctype = json.dumps(state.snapshot()).encode(), "application/json"
            elif path == "/frame.jpg" and state is not None and state.jpeg:
                body, ctype = state.jpeg, "image/jpeg"
            elif path in ("/", "/review") and review is not None:
                body, ctype = REVIEW_PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api/review" and review is not None:
                body = json.dumps({"alerts": review.labelled(), "summary": review.summary()}).encode()
                ctype = "application/json"
            elif path.startswith("/clips/") and review is not None and (clip := review.clip_file(path[7:])):
                return self._send_clip(clip.read_bytes())
            else:
                self.send_response(404); self.end_headers(); return
            self._send(200, body, ctype)

        def do_POST(self):  # noqa: N802
            if self.path != "/api/review/label" or review is None:
                self.send_response(404); self.end_headers(); return
            # JSON only: a cross-site form can't send this type without a CORS preflight, which we never answer.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, b"Content-Type must be application/json")
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                rec = review.add_label(str(d["id"]), str(d["label"]), str(d.get("reviewer", "")),
                                       str(d.get("note", "")))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, str(e).encode())
            self._send(200, json.dumps(rec).encode(), "application/json")

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
