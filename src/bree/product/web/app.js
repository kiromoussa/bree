"use strict";
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmtT = (ts, tz) => new Date(ts * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", timeZone: tz});
const dur = s => { s = Math.round(s); return s >= 60 ? `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`; };
const ago = s => s == null ? "never" : s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
const get = async u => { const r = await fetch(u); if (!r.ok) throw new Error(r.status); return r.json(); };
const CW = () => Math.max(280, Math.min(900, document.querySelector("main").clientWidth - 30));
let S = null, tab = "now", day = null, days = [];

const SOURCE = {
  simulated: ["SIMULATED DATA", "Made-up numbers from a simulator. They show how the screens look, not how your store is doing."],
  real_unlabelled: ["ESTIMATES", "Counted automatically from real video. Nobody has checked these against the truth yet."],
  real_model_labelled: ["CHECKED BY AI", "Compared against frames labelled by an AI model; useful but not verified by a person."],
  real_ground_truth: ["VERIFIED", "Checked against labels made by people."],
  live: ["LIVE", "Live counts."]
};

function bars(vals, labels, title, opts = {}) {
  const W = CW(), H = opts.h || 180, L = 30, B = 22, T = 14, pw = W - L - 4, ph = H - B - T, n = vals.length;
  const vmax = Math.max(1, ...vals.filter(v => v != null));
  const step = vmax <= 5 ? 1 : vmax <= 12 ? 2 : vmax <= 40 ? 5 : vmax <= 100 ? 10 : 50;
  const ymax = (Math.floor(vmax / step) + 1) * step, bw = pw / n;
  let o = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
  for (let g = 0; g <= ymax; g += step) { const y = T + ph - ph * g / ymax; o += `<line x1="${L}" x2="${W - 4}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${L - 4}" y="${y + 4}" text-anchor="end">${g}</text>`; }
  vals.forEach((v, i) => {
    const x = L + i * bw;
    if (v == null) o += `<circle cx="${x + bw / 2}" cy="${T + ph - 2}" r="1.6" fill="var(--muted)"/>`;
    else if (v > 0) { const h = ph * v / ymax; o += `<rect x="${x + 1.5}" y="${T + ph - h}" width="${Math.max(bw - 3, 1)}" height="${h}" rx="2" fill="${(opts.faded || []).includes(i) ? "var(--bar2)" : "var(--bar)"}"><title>${esc(labels[i])}: ${v}</title></rect>`; if (bw > 24) o += `<text class="v" x="${x + bw / 2}" y="${T + ph - h - 3}" text-anchor="middle">${v}</text>`; }
    if (i % Math.ceil(n / Math.max(1, Math.floor(pw / 30))) === 0) o += `<text x="${x + bw / 2}" y="${H - 7}" text-anchor="middle">${esc(labels[i])}</text>`;
  });
  return o + "</svg>";
}

function line(points, tz, title) {
  if (points.length < 2) return `<div class="empty small">Not enough data yet.</div>`;
  const W = CW(), H = 150, L = 30, B = 22, T = 10, pw = W - L - 6, ph = H - B - T;
  const t0 = points[0].ts, t1 = points[points.length - 1].ts, vmax = Math.max(2, ...points.map(p => p.n));
  const X = t => L + pw * (t - t0) / Math.max(t1 - t0, 1), Y = v => T + ph - ph * v / vmax;
  let d = points.map((p, i) => `${i ? "L" : "M"}${X(p.ts).toFixed(1)},${Y(p.n).toFixed(1)}`).join(" ");
  let o = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
  for (let g = 0; g <= vmax; g += Math.max(1, Math.ceil(vmax / 4))) o += `<line x1="${L}" x2="${W - 6}" y1="${Y(g)}" y2="${Y(g)}" stroke="var(--line)"/><text x="${L - 4}" y="${Y(g) + 4}" text-anchor="end">${g}</text>`;
  for (let i = 0; i <= (W < 450 ? 2 : 4); i++) { const t = t0 + (t1 - t0) * i / (W < 450 ? 2 : 4); o += `<text x="${X(t)}" y="${H - 6}" text-anchor="middle">${fmtT(t, tz)}</text>`; }
  return o + `<path d="${d}" fill="none" stroke="var(--bar)" stroke-width="2"/></svg>`;
}

function trust(sources, assumed) {
  const k = Object.keys(sources || {}), el = $("#trust");
  if (!k.length) { el.hidden = true; return; }
  const [name, text] = SOURCE[k[0]] || [k[0], ""];
  el.hidden = false; el.className = "trust " + k[0];
  el.innerHTML = `<b>${esc(name)}</b>${esc(text)}${assumed ? " Clock and time zone are not confirmed yet." : ""}`;
}

async function renderNow() {
  S = await get("/api/state");
  $("#store").textContent = S.store;
  $("#clock").textContent = (S.replay ? "Replay of saved data, " : "") + new Date(S.now * 1000).toLocaleString([], {timeZone: S.tz.startsWith("UTC") || S.tz.includes("/") ? S.tz : undefined, weekday: "short", hour: "2-digit", minute: "2-digit"});
  trust(S.sources, S.tz_assumed);
  const rb = $("#rv-badge"); rb.hidden = !S.pending_review; rb.textContent = S.pending_review;
  const t = (big, lab, cls = "") => `<div class="tile"><div class="big ${cls}">${big}</div><div class="lab">${lab}</div></div>`;
  const q = S.queue;
  let h = `<div class="tiles">` +
    t(S.occupancy == null ? "–" : Math.round(S.occupancy), "people inside now") +
    t(q == null ? "–" : q, q >= 3 ? "waiting at the register" : "in the register line", q >= 3 ? "warn" : "") +
    t(S.door_today.in || S.door_today.out ? S.door_today.in : "–", "came in today") +
    t(S.pending_review, "items to review", S.pending_review ? "warn" : "") + `</div>`;
  h += `<div class="card"><h2>Needs your attention</h2>` + (S.attention.length ? S.attention.map(a => `<div class="row"><span>${esc(a.text)}</span><span class="tag">${esc(a.kind)}</span></div>`).join("") : `<div class="row ok">Nothing right now.</div>`) + `</div>`;
  h += `<div class="card"><h2>People inside, last 3 hours</h2>${line(S.occupancy_recent, S.tz, "People inside")}</div>`;
  h += `<div class="card"><h2>Came in today, by hour</h2>${hourChart(S.by_hour)}</div>`;
  h += `<div class="card"><h2>Cameras</h2>` + S.cameras.map(c => `<div class="row"><span><span class="dot ${c.status}"></span>${esc(c.label)}</span><span class="muted small">${c.status === "ok" ? "working" : c.status === "unknown" ? "no data" : "last heard " + ago(c.age_s)}</span></div>`).join("") + `</div>`;
  if (S.shelves.length) h += `<div class="card"><h2>Shelves and coolers</h2>` + S.shelves.map(s => `<div class="row"><span>${esc(s.zone)} <span class="muted small">${esc(s.label)}</span></span><span class="${s.state === "ok" ? "ok" : "warn"}">${s.state === "ok" ? Math.round(s.fullness * 100) + "% full" : esc(s.state.replace("_", " "))}</span></div>`).join("") + `</div>`;
  $("#now").innerHTML = h;
}

function hourChart(byHour) {
  const v = Array(24).fill(null), cov = {};
  byHour.forEach(x => v[x.hour] = x.in);
  if (!byHour.length) return `<div class="empty small">No door counts yet today.</div>`;
  return bars(v.map(x => x), [...Array(24).keys()].map(h => String(h).padStart(2, "0")), "People in by hour");
}

async function renderDay() {
  if (!days.length) days = await get("/api/days");
  if (!day) day = days.length ? days[days.length - 1] : null;
  const r = await get("/api/day" + (day ? `?date=${day}` : ""));
  const tz = r.tz;
  const opts = days.slice(-30).reverse().map(d => `<option ${d === r.date ? "selected" : ""}>${d}</option>`).join("");
  let h = `<div class="bar"><select id="daysel" aria-label="Day">${opts}</select><a class="btn" href="/digest/daily?date=${r.date}" target="_blank">Printable digest</a></div>`;
  if (!r.signals) return $("#today").innerHTML = h + `<div class="empty">No camera data for this day.</div>`;
  const v = Array(24).fill(null);
  r.occupancy_series.forEach(p => { const hr = new Date(p.ts * 1000).toLocaleString([], {hour: "2-digit", hour12: false, timeZone: tz}); if (v[+hr % 24] == null) v[+hr % 24] = 0; });
  r.by_hour.forEach(x => v[x.hour] = x.in);
  h += `<div class="tiles">` + [[r.door.in || "–", "came in"], [r.busiest ? String(r.busiest.hour).padStart(2, "0") + ":00" : "–", "busiest hour"], [r.occupancy.peak == null ? "–" : Math.round(r.occupancy.peak), "most inside at once"], [r.queue.peak == null ? "–" : r.queue.peak, "longest register line"]].map(([a, b]) => `<div class="tile"><div class="big">${a}</div><div class="lab">${b}</div></div>`).join("") + `</div>`;
  h += `<div class="card"><h2>People through the door, by hour</h2>${bars(v, [...Array(24).keys()].map(x => String(x).padStart(2, "0")), "People in by hour")}<div class="muted small">A dot means the cameras had no data that hour; it is not zero customers.</div></div>`;
  h += `<div class="card"><h2>People inside during the day</h2>${line(r.occupancy_series, tz, "People inside")}</div>`;
  const q = r.queue;
  h += `<div class="card"><h2>Waiting at the register</h2>` + (q.peak == null ? `<div class="muted">No register-line data.</div>` : `<div class="small muted">Average line ${q.mean} people.${q.share_2plus != null ? ` Two or more waiting ${Math.round(q.share_2plus * 100)}% of the time.` : ""}</div>` + (q.episodes.length ? `<table><tr><th>When</th><th class="n">Peak</th><th class="n">Lasted</th></tr>${q.episodes.sort((a, b) => a.start - b.start).map(e => `<tr><td>${fmtT(e.start, tz)}–${fmtT(e.end, tz)}</td><td class="n">${e.peak}</td><td class="n">${dur(e.seconds)}</td></tr>`).join("")}</table>` : `<div class="ok">Nobody waited in a line.</div>`)) + `</div>`;
  const dw = Object.entries(r.dwell);
  h += `<div class="card"><h2>Where people spend time</h2>` + (dw.length ? `<table><tr><th>Area</th><th class="n">Visits</th><th class="n">Typical stay</th></tr>${dw.sort((a, b) => b[1].visits - a[1].visits).map(([z, d]) => `<tr><td>${esc(z)}</td><td class="n">${d.visits}</td><td class="n">${dur(d.median_s)}</td></tr>`).join("")}</table>` : `<div class="muted">No data.</div>`) + `</div>`;
  h += `<div class="card"><h2>Shelf and cooler problems</h2>` + (r.shelf_issues.length ? `<table><tr><th>Where</th><th>Problem</th><th>From</th><th class="n">For</th></tr>${r.shelf_issues.map(e => `<tr><td>${esc(e.zone)}</td><td class="warn">${esc(e.state.replace("_", " "))}</td><td>${fmtT(e.start, tz)}</td><td class="n">${dur(e.seconds)}</td></tr>`).join("")}</table>` : `<div class="muted">None recorded.</div>`) + `</div>`;
  $("#today").innerHTML = h;
  $("#daysel").onchange = e => { day = e.target.value; renderDay(); };
}

let reviewer = ""; try { reviewer = localStorage.getItem("reviewer") || ""; } catch (e) {}
async function renderReview() {
  const r = await get("/api/review"), tz = S ? S.tz : undefined;
  let h = `<div class="bar"><label>Your name <input id="who" value="${esc(reviewer)}" placeholder="e.g. Sam"></label></div>`;
  h += `<p class="muted small">These are baskets the system is not sure were paid for. It can be wrong. Watch the short clip (faces are blurred) and say what you see.</p>`;
  const card = (a, done) => `<div class="card alert" data-id="${esc(a.id)}"><div>${a.clip ? `<video src="/clips/${encodeURIComponent(a.id)}" controls muted loop playsinline preload="metadata"></video>` : `<div class="empty">No clip saved</div>`}</div>
    <div><div><b>${esc(a.items.join(", ") || "items")}</b> <span class="tag">${a.tier === "alert" ? "high confidence" : "please check"}</span>${a.retracted ? `<span class="tag">a later receipt lowered this</span>` : ""}</div>
    <div class="muted small">${new Date(a.time * 1000).toLocaleString([], {timeZone: tz, weekday: "short", hour: "2-digit", minute: "2-digit"})} · ${esc(a.label)} · confidence ${(a.confidence || 0).toFixed(2)}${a.visited_register === false ? " · did not go to the register" : ""}</div>
    <ul class="small">${a.reasons.slice(0, 3).map(x => `<li>${esc(x)}</li>`).join("")}</ul>
    ${done ? `<div class="muted">You decided: ${esc((r.decisions || {})[a.decision] || a.decision)}</div>` : `<div class="btns"><button class="theft" data-d="confirmed_theft">Theft</button><button class="fine" data-d="not_theft">Not theft</button><button data-d="unclear">Can't tell</button></div>`}</div></div>`;
  h += r.pending.length ? r.pending.map(a => card(a, false)).join("") : `<div class="empty">Nothing waiting for review.</div>`;
  if (r.recent.length) h += `<h3 class="muted small">Recently decided</h3>` + r.recent.slice(0, 5).map(a => card(a, true)).join("");
  $("#review").innerHTML = h;
  $("#who").onchange = e => { reviewer = e.target.value; try { localStorage.setItem("reviewer", reviewer); } catch (x) {} };
}
$("#review").onclick = async e => {
  const b = e.target.closest("button[data-d]"); if (!b) return;
  const who = $("#who").value.trim(); if (!who) { alert("Please type your name first."); $("#who").focus(); return; }
  const res = await fetch("/api/review/decide", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: b.closest(".card").dataset.id, decision: b.dataset.d, reviewer: who})});
  if (!res.ok) { alert((await res.json()).error || "Could not save"); return; }
  refresh();
};

async function renderReports() {
  if (!days.length) days = await get("/api/days");
  const last = days.length ? days[days.length - 1] : "";
  $("#reports").innerHTML = `<div class="card"><h2>Daily digest</h2><p class="muted small">One printable page: traffic by hour, register line, shelves, review items.</p>
    <div class="bar"><input type="date" id="dd" value="${last}"><a class="btn" id="dl" target="_blank" href="/digest/daily?date=${last}">Open</a></div></div>
    <div class="card"><h2>Weekly digest</h2><p class="muted small">The 7 days ending on the date you pick, compared with the week before.</p>
    <div class="bar"><input type="date" id="wd" value="${last}"><a class="btn" id="wl" target="_blank" href="/digest/weekly?end=${last}">Open</a></div></div>`;
  $("#dd").onchange = e => $("#dl").href = "/digest/daily?date=" + e.target.value;
  $("#wd").onchange = e => $("#wl").href = "/digest/weekly?end=" + e.target.value;
}

const R = {now: renderNow, today: renderDay, review: renderReview, reports: renderReports};
async function refresh() { try { await renderNow(); if (tab !== "now") await R[tab](); } catch (e) { console.error(e); } }
$("#tabs").onclick = e => {
  const b = e.target.closest("button[data-tab]"); if (!b) return;
  tab = b.dataset.tab;
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("on", x === b));
  ["now", "today", "review", "reports"].forEach(t => $("#" + t).hidden = t !== tab);
  refresh();
};
refresh(); setInterval(() => { if (tab === "now") refresh(); }, 5000);
