"""Local reviewer page: one alert at a time, clip + item + register record, decide with one key.

Stdlib http.server, same pattern as bree.dashboard.server. Binds to 127.0.0.1: nothing leaves
the machine, and the page loads no outside script, font or image.
  /                   the page
  /api/next?reviewer= the oldest alert this reviewer has not decided, plus how many are left.
                      With &alert=<id>: that alert again (the page's "back" button).
  /api/decide         POST JSON {alert_id, reviewer, decision, corrected_item, pick_seen,
                      conceal_seen, note, view_s}
  /clips/<alert_id>   that alert's head-pixelated clip

Requests whose Host header is not 127.0.0.1 or localhost on this port get 403, so a web page that
points its own name at this machine (DNS rebinding) cannot read alerts or post decisions.

Guards against decisions nobody looked at: a held key counts once, a decision is ignored while the
last one is still being saved and for MIN_VIEW_MS after an alert appears, and "Back" (key U) reopens
the alert just decided so a slip can be corrected (a new row; the latest per reviewer counts).
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bree.review.store import ReviewStore

MAX_BODY = 64 * 1024

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BREE review</title><style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e3de;--warn:#b26a00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--card:#1f1f1e;--line:#33332f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}.muted{color:var(--muted)}
main{padding:16px;display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;max-width:1200px}
@media (max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
video{width:100%;border-radius:6px;background:#000}
dl{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;margin:0}dt{color:var(--muted)}dd{margin:0}
.item{font-size:22px;font-weight:600}.tag{font-size:12px;padding:1px 8px;border-radius:10px;border:1px solid var(--warn);color:var(--warn)}
input,select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{cursor:pointer}.keys{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.keys button{padding:12px;text-align:left}kbd{border:1px solid var(--line);border-radius:4px;padding:0 6px;margin-right:6px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
</style></head><body>
<header><h1>BREE review</h1><label>Reviewer <input id="who" placeholder="your id"></label>
<span class="muted" id="left"></span><button id="back" disabled><kbd>U</kbd>Back to the last alert</button><span class="muted">Heads are pixelated. Clips stay on this machine.</span></header>
<main id="main"><p class="muted">Enter your reviewer id to start.</p></main>
<script>
const $=id=>document.getElementById(id);const MIN_VIEW_MS=400;
let cur=null,t0=0,busy=false,last=null,held=null,items=[],unlisted='';
function esc(s){return String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
const norm=s=>s.trim().toLowerCase().replace(/ +/g,'_');
try{$('who').value=localStorage.getItem('who')||''}catch(e){}
$('who').onchange=()=>{try{localStorage.setItem('who',$('who').value)}catch(e){};$('who').blur();last=null;load()};
const REG={no_register_visit:'Never went to the register',visited_no_receipt:'Went to the register, no receipt matched',
receipt_missing_item:'Paid, but this item is not on the receipt',receipt_matches:'Receipt matches'};
const WORDS={confirmed_theft:'confirmed theft',not_theft:'not theft',wrong_item:'theft, wrong item',unclear:'unclear'};
async function load(id){const who=$('who').value.trim();if(!who)return;
const d=await (await fetch('/api/next?reviewer='+encodeURIComponent(who)+(id?'&alert='+encodeURIComponent(id):''))).json();
cur=d.alert;items=d.items;unlisted='';$('left').textContent=d.left+' left';$('back').disabled=!last;
if(!cur){$('main').innerHTML='<p class="muted">Nothing left to review.</p>';return}
const a=cur,paid=a.register.paid_items||[],rt=a.retracted_tier;
$('main').innerHTML=`<section class="card">${a.clip_path?`<video src="/clips/${encodeURIComponent(a.alert_id)}" controls autoplay muted loop playsinline></video>`:'<p class="muted">No clip for this alert.</p>'}</section>
<section class="card"><div class="item">${esc(a.predicted_item||'unknown item')}</div>
${a.items.length>1?`<div class="muted">All unpaid items: ${esc(a.items.map(i=>i.category).join(', '))}</div>`:''}
${a.my_decision?`<p><span class="tag">You decided: ${esc(WORDS[a.my_decision]||a.my_decision)}. Decide again to change it.</span></p>`:''}
${rt?`<p><span class="tag">${rt==='retracted'?'Later withdrawn by a late receipt':'Later lowered to '+esc(rt)+' by a late receipt'}: the register lines below are from before it</span></p>`:''}
${a.identity_uncertain?'<p><span class="tag">Identity uncertain: the system may have mixed up two people</span></p>':''}
<dl style="margin-top:10px"><dt>When</dt><dd>${new Date(a.event_ts*1000).toLocaleString()}</dd>
<dt>Camera</dt><dd>${esc(a.camera)}</dd><dt>Zone</dt><dd>${esc(a.zone)}</dd>
<dt>Register</dt><dd>${esc(REG[a.register_match]||a.register_match)}</dd>
<dt>Receipt</dt><dd>${esc(paid.map(p=>p.category||p.sku).join(', ')||'none')}</dd>
<dt>Why</dt><dd class="muted">${esc(a.reasons.join('; '))}</dd></dl>
<div class="keys"><button data-d="confirmed_theft"><kbd>1</kbd>Confirmed theft</button><button data-d="not_theft"><kbd>2</kbd>Not theft</button>
<button data-d="wrong_item"><kbd>3</kbd>Theft, wrong item</button><button data-d="unclear"><kbd>4</kbd>Unclear</button></div>
<div class="row"><label>Correct item <input id="item" list="items" placeholder="if wrong, then Enter"></label>
<datalist id="items">${d.items.map(i=>`<option value="${esc(i)}">`).join('')}</datalist></div>
<div class="row muted" id="hint" role="status"></div>
<div class="row"><label>Pick seen <select id="pick"><option value="">not answered</option><option value="1">yes</option><option value="0">no</option></select></label>
<label>Concealment seen <select id="conceal"><option value="">not answered</option><option value="1">yes</option><option value="0">no</option></select></label></div>
<div class="row"><input id="note" placeholder="note (optional)" style="flex:1"></div></section>`;t0=performance.now()}
const tri=v=>v===''?null:v==='1';
// busy: one decision at a time. MIN_VIEW_MS: nothing can be decided before the alert has been on screen that long.
async function decide(dec){if(!cur||busy||performance.now()-t0<MIN_VIEW_MS)return;const item=dec==='wrong_item'?norm($('item').value):'';
if(dec==='wrong_item'&&!item){$('item').focus();return}
// An item name the system has never used is probably a typo: ask for Enter a second time.
if(item&&!items.includes(item)&&unlisted!==item){unlisted=item;$('hint').textContent=`"${item}" is not in the list. Press Enter again to add it as a new item.`;$('item').focus();return}
busy=true;try{
const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alert_id:cur.alert_id,
reviewer:$('who').value,decision:dec,corrected_item:item,pick_seen:tri($('pick').value),conceal_seen:tri($('conceal').value),
note:$('note').value,view_s:(performance.now()-t0)/1000})});
if(!r.ok){alert(await r.text());return}last=cur.alert_id;await load()}finally{busy=false}}
// Back: show the alert just decided again. Deciding it again adds a new row and the latest one counts.
async function back(){if(busy||!last)return;busy=true;try{const id=last;last=null;await load(id)}finally{busy=false}}
$('back').onclick=back;
$('main').onclick=e=>{const b=e.target.closest('button[data-d]');if(b)decide(b.dataset.d)};
const KEYS={1:'confirmed_theft',2:'not_theft',3:'wrong_item',4:'unclear'};
// A held key repeats about 30 times a second. Only the first press counts; the repeats are swallowed
// (also after key 3 has moved focus to the item field, so they are not typed there).
// preventDefault: key 3 moves focus to the item field, and without it the "3" would be typed there.
document.onkeydown=e=>{if(e.repeat){if(e.key===held)e.preventDefault();return}
if(e.target.matches('input,select')||e.metaKey||e.ctrlKey||e.altKey)return;
if(KEYS[e.key]){e.preventDefault();held=e.key;decide(KEYS[e.key])}else if(e.key==='u'||e.key==='U')back()};
document.onkeyup=e=>{if(e.key===held)held=null};
// Enter in the item field sends "wrong item". keyup, so a suggestion picked with Enter is filled in first.
$('main').onkeyup=e=>{if(e.key==='Enter'&&e.target.id==='item')decide('wrong_item')};
load();
</script></body></html>"""


def serve(store: ReviewStore, host: str = "127.0.0.1", port: int = 8090, purge_every_s: float = 3600) -> ThreadingHTTPServer:
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

        def _local(self) -> bool:
            """Only answer when the browser asked for this machine by its loopback name."""
            port = self.server.server_address[1]
            if self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return True
            self._send(403, b"this page only answers on 127.0.0.1")
            return False

        def _send_clip(self, data: bytes):
            # Safari only plays <video> from servers that answer byte ranges.
            rng = self.headers.get("Range", "")
            try:
                a, b = rng[6:].split(",")[0].split("-") if rng.startswith("bytes=") else ("", "")
                start = int(a) if a else max(0, len(data) - int(b)) if b else None
                end = min(int(b), len(data) - 1) if a and b else len(data) - 1
            except ValueError:                  # malformed header: answer with the whole file
                start = None
            if start is not None:
                if start >= len(data) or start > end:
                    return self._send(416, b"", "video/mp4", [("Content-Range", f"bytes */{len(data)}")])
                return self._send(206, data[start:end + 1], "video/mp4",
                                  [("Content-Range", f"bytes {start}-{end}/{len(data)}"), ("Accept-Ranges", "bytes")])
            self._send(200, data, "video/mp4", [("Accept-Ranges", "bytes")])

        def do_GET(self):  # noqa: N802
            if not self._local():
                return
            url = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(url.path)
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/next":
                q = urllib.parse.parse_qs(url.query)
                who = q.get("reviewer", [""])[0].strip()
                alert, left = store.next_for(who, q.get("alert", [None])[0]) if who else (None, 0)
                if alert:   # the page needs no pose data and no file paths beyond "is there a clip"
                    mine = next((r["decision"] for r in alert["reviews"] if r["reviewer_id"] == who), None)
                    alert = {k: v for k, v in alert.items() if k not in ("frames", "reviews")}
                    alert["my_decision"] = mine
                    alert["clip_path"] = bool(store.clip_file(alert["alert_id"]))
                body = json.dumps({"alert": alert, "left": left, "items": store.known_items()})
                return self._send(200, body.encode(), "application/json")
            if path.startswith("/clips/") and (clip := store.clip_file(path[7:])):
                return self._send_clip(clip.read_bytes())
            self._send(404, b"not found")

        def do_POST(self):  # noqa: N802
            if not self._local():
                return
            if self.path != "/api/decide":
                return self._send(404, b"not found")
            # JSON only: a cross-site form can't send this type without a CORS preflight, which we never answer.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, b"Content-Type must be application/json")
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 <= size <= MAX_BODY:   # a negative length would make the read wait forever
                    raise ValueError(f"Content-Length must be 0 to {MAX_BODY}")
                d = json.loads(self.rfile.read(size))
                if not isinstance(d, dict):
                    raise ValueError("body must be a JSON object")
                view = d.get("view_s")
                rec = store.decide(str(d["alert_id"]), str(d.get("reviewer", "")), str(d["decision"]),
                                   d.get("corrected_item") and str(d["corrected_item"]),
                                   d.get("pick_seen"), d.get("conceal_seen"), str(d.get("note") or ""),
                                   None if view is None else float(view))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, str(e).encode())
            self._send(200, json.dumps(rec).encode(), "application/json")

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def purge_loop():
        stop = threading.Event()
        while not stop.wait(purge_every_s):
            try:
                store.purge()
            except Exception:      # the store was closed: stop quietly
                return
    threading.Thread(target=purge_loop, daemon=True).start()
    return httpd
