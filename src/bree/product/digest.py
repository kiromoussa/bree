"""Daily and weekly store digests: one self-contained, printable HTML page (inline CSS and SVG, no scripts,
no network). Written for a store owner: plain sentences first, numbers second, and every page says how
far its data can be trusted.
"""
from __future__ import annotations

import datetime as dt
import html
from collections import defaultdict

from bree.product import metrics
from bree.product.signals import Signal, day_bounds

E = html.escape

SOURCE_TEXT = {
    "simulated": ("SIMULATED DATA", "These numbers come from a simulator. They show what the report looks like; they say nothing about your store."),
    "real_unlabelled": ("ESTIMATES FROM REAL VIDEO", "Counted automatically from real camera video. Nobody has checked these counts against the truth, so treat them as estimates."),
    "real_model_labelled": ("CHECKED BY AN AI REVIEWER", "Counts were compared against frames labelled by an AI model. Useful, but not ground truth."),
    "real_ground_truth": ("VERIFIED", "Counts were checked against labels made by people."),
    "live": ("LIVE", "Live counts from the cameras."),
}

CSS = """
:root{--bg:#fff;--fg:#1c1c1a;--muted:#66665f;--line:#e2e2dc;--card:#f7f7f4;--bar:#2f6f8f;--bar2:#a9c9d8;--warn:#b26a00;--bad:#b3261e;--ok:#2e7d32}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#9a9a94;--line:#33332f;--card:#1f1f1e;--bar:#6fb1d1;--bar2:#35566a}}
@media print{:root{--bg:#fff;--fg:#000;--muted:#444;--line:#bbb;--card:#fff;--bar:#2f6f8f;--bar2:#a9c9d8}.noprint{display:none}section{break-inside:avoid}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:24px;margin:0 0 2px}h2{font-size:16px;margin:28px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
.sub{color:var(--muted);margin:0}.badge{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;padding:2px 8px;border-radius:4px;border:1px solid var(--line);background:var(--card);margin-right:6px}
.badge.simulated{border-color:var(--warn);color:var(--warn)}
.trust{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:14px 0;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:26px;line-height:1.15}.tile span{color:var(--muted);font-size:13px}
.lead{font-size:17px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}
svg text{fill:var(--muted);font:11px system-ui,sans-serif}svg .v{fill:var(--fg)}.small{font-size:13px}
footer{margin-top:32px;color:var(--muted);font-size:12px}
.noprint{margin:10px 0}.noprint button{font:inherit;padding:4px 10px}
"""


def _t(ts, tz, fmt="%H:%M"):
    return dt.datetime.fromtimestamp(ts, tz).strftime(fmt)


def _dur(s):
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def bar_chart(values, labels, *, title, unit="", width=760, height=190, faded=(), color_var="--bar") -> str:
    """Vertical bars. `faded`: indexes drawn light (partial coverage). Values None = no data (drawn as a dot)."""
    n = len(values)
    left, bottom, top = 34, 24, 14
    pw, ph = width - left - 6, height - bottom - top
    vmax = max([v for v in values if v] + [1])
    step = 1 if vmax <= 5 else 2 if vmax <= 12 else 5 if vmax <= 40 else 10 if vmax <= 100 else 50
    ymax = ((int(vmax) // step) + 1) * step
    bw = pw / n
    o = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{E(title)}" width="100%">']
    for g in range(0, ymax + 1, step):
        y = top + ph - ph * g / ymax
        o.append(f'<line x1="{left}" x2="{width - 6}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--line)"/><text x="{left - 4}" y="{y + 4:.1f}" text-anchor="end">{g}</text>')
    for i, (v, lab) in enumerate(zip(values, labels)):
        x = left + i * bw
        if v is None:
            o.append(f'<circle cx="{x + bw / 2:.1f}" cy="{top + ph - 2}" r="1.5" fill="var(--muted)"/>')
        elif v > 0:
            h = ph * v / ymax
            fill = "var(--bar2)" if i in faded else f"var({color_var})"
            o.append(f'<rect x="{x + 1.5:.1f}" y="{top + ph - h:.1f}" width="{max(bw - 3, 1):.1f}" height="{h:.1f}" fill="{fill}" rx="2"><title>{E(str(lab))}: {v:g}{unit}</title></rect>')
            if bw > 16:
                o.append(f'<text class="v" x="{x + bw / 2:.1f}" y="{top + ph - h - 3:.1f}" text-anchor="middle">{v:g}</text>')
        if lab != "" and (n <= 12 or i % 2 == 0 or n <= 24):
            o.append(f'<text x="{x + bw / 2:.1f}" y="{height - 8}" text-anchor="middle">{E(str(lab))}</text>')
    o.append("</svg>")
    return "".join(o)


def _trust_block(sources: dict, tz_assumed: bool, notes: list[str]) -> str:
    keys = list(sources) or []
    weakest = keys[0] if keys else None
    out = []
    if weakest:
        name, text = SOURCE_TEXT.get(weakest, (weakest, ""))
        badges = "".join(f'<span class="badge {k}">{E(SOURCE_TEXT.get(k, (k,))[0])}</span>' for k in keys)
        out.append(f'<div class="trust">{badges}<br>{E(text)}</div>')
    if tz_assumed:
        out.append('<div class="trust small">Clock note: hours are shown in the time zone set for this store. '
                   'The camera recorder clock has not been checked against a receipt time yet, so an hour may be off.</div>')
    for n in notes:
        out.append(f'<div class="trust small">{E(n)}</div>')
    return "".join(out)


def _hour_arrays(rep, tz):
    """24 values for traffic (None = no data that hour), plus indexes of partially covered hours."""
    covered = defaultdict(set)
    for p in rep["occupancy_series"]:
        covered[dt.datetime.fromtimestamp(p["ts"], tz).hour].add(int(p["ts"] // 60))
    ins = {h["hour"]: h["in"] for h in rep["by_hour"]}
    vals, faded = [], []
    for h in range(24):
        c = len(covered.get(h, ()))
        vals.append(ins.get(h, 0) if c or h in ins else None)
        if 0 < c < 50:
            faded.append(h)
    return vals, faded


def daily_digest(rep: dict, *, store_name: str, day: dt.date, tz, review_pending: list[dict] | None = None,
                 tz_assumed: bool = False, notes: list[str] | None = None, generated: float | None = None) -> str:
    door, occ, q = rep["door"], rep["occupancy"], rep["queue"]
    busiest = rep["busiest"]
    has_data = rep["signals"] > 0
    review_pending = review_pending or []
    sh = rep["shrink"]
    vals, faded = _hour_arrays(rep, tz)
    lead = []
    if not has_data:
        lead.append("No camera data for this day.")
    else:
        if door["in"] or door["out"]:
            lead.append(f"About {door['in']} people came in.")
            if busiest:
                lead.append(f"The busiest hour was {busiest['hour']:02d}:00 with {busiest['in']}.")
        if occ["peak"] is not None:
            lead.append(f"Most people in the store at once: about {occ['peak']:.0f} (around {_t(occ['peak_ts'], tz)}).")
        if q["peak"] is not None and q["peak"] >= 2:
            lead.append(f"Longest wait at the register: {q['peak']} people; {_dur(q['episodes'][0]['seconds'])} that time.")
        elif q["peak"] is not None:
            lead.append("Nobody was waiting in line at the register.")
        n_issue = len(rep["shelf_issues"])
        if n_issue:
            lead.append(f"{n_issue} shelf or cooler problem{'s' if n_issue != 1 else ''} to look at.")
        if sh["alerts"]:
            lead.append(f"{len(review_pending)} item{'s' if len(review_pending) != 1 else ''} waiting for your review.")
    o = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>{E(store_name)} daily digest {day:%Y-%m-%d}</title><style>{CSS}</style></head><body><main>',
         f'<h1>{E(store_name)}: {day:%A, %B %-d, %Y}</h1><p class="sub">Daily digest</p>',
         '<p class="noprint"><button onclick="window.print()">Print</button></p>',
         _trust_block(rep["sources"], tz_assumed, notes or []),
         f'<p class="lead">{E(" ".join(lead))}</p>']
    if has_data:
        o.append('<div class="tiles">')
        for big, small in (
            (door["in"] if door["in"] or door["out"] else "n/a", "people in the door"),
            (f"{busiest['hour']:02d}:00" if busiest else "n/a", "busiest hour"),
            (f"{occ['peak']:.0f}" if occ["peak"] is not None else "n/a", "most inside at once"),
            (q["peak"] if q["peak"] is not None else "n/a", "longest register line"),
            (len(rep["shelf_issues"]), "shelf or cooler issues"),
            (len(review_pending), "items to review"),
        ):
            o.append(f'<div class="tile"><b>{E(str(big))}</b><span>{small}</span></div>')
        o.append("</div>")
        o.append('<h2>People through the door, by hour</h2>')
        o.append(bar_chart(vals, [f"{h:02d}" for h in range(24)], title="People through the door by hour", faded=faded))
        notes_h = []
        if faded:
            notes_h.append("Light bars are hours with less than 50 minutes of camera data.")
        if any(v is None for v in vals):
            notes_h.append("A dot means no camera data that hour (not zero customers).")
        if notes_h:
            o.append(f'<p class="sub small">{E(" ".join(notes_h))}</p>')
        # queue
        o.append('<h2>Waiting at the register</h2>')
        if q["peak"] is None:
            o.append('<p class="muted">No register-line data for this day.</p>')
        else:
            share = f"{q['share_2plus'] * 100:.0f}%" if q["share_2plus"] is not None else "n/a"
            o.append(f'<p>Average line: {q["mean"]} people. Time with 2 or more waiting: {share} of the time we could see.</p>')
            if q["episodes"]:
                o.append('<table><tr><th>When</th><th class="n">Peak</th><th class="n">Lasted</th></tr>')
                for e in sorted(q["episodes"], key=lambda e: e["start"]):
                    o.append(f'<tr><td>{_t(e["start"], tz)} to {_t(e["end"], tz)}</td><td class="n">{e["peak"]}</td><td class="n">{_dur(e["seconds"])}</td></tr>')
                o.append("</table>")
        # dwell
        o.append('<h2>Where people spend time</h2>')
        if rep["dwell"]:
            o.append('<table><tr><th>Area</th><th class="n">Visits</th><th class="n">Typical stay</th><th class="n">Long stays (90th pct)</th></tr>')
            for z, d in sorted(rep["dwell"].items(), key=lambda kv: -kv[1]["visits"]):
                o.append(f'<tr><td>{E(z)}</td><td class="n">{d["visits"]}</td><td class="n">{_dur(d["median_s"])}</td><td class="n">{_dur(d["p90_s"]) if d["p90_s"] else "too few visits"}</td></tr>')
            o.append("</table>")
        else:
            o.append('<p class="muted">No visit-length data for this day.</p>')
        # shelves
        o.append('<h2>Shelves and coolers</h2>')
        if rep["shelf_issues"]:
            o.append('<table><tr><th>Where</th><th>Problem</th><th>From</th><th class="n">For</th></tr>')
            names = {"low": "running low", "empty": "empty", "door_open": "door open", "misplaced": "items out of place", "dirty": "needs cleaning", "unknown": "unclear"}
            for e in rep["shelf_issues"]:
                o.append(f'<tr><td>{E(e["zone"] or "")} <span class="muted">({E(e["camera"])})</span></td><td class="warn">{names.get(e["state"], e["state"])}</td><td>{_t(e["start"], tz)}</td><td class="n">{_dur(e["seconds"])}</td></tr>')
            o.append("</table>")
        elif rep["shelf_now"]:
            o.append('<p>All watched shelves and coolers looked fine.</p>')
        else:
            o.append('<p class="muted">Shelf and cooler checks are not running for this store yet.</p>')
        # shrink
        o.append('<h2>Theft risk and review</h2>')
        if sh["alerts"] == 0:
            o.append('<p>No suspicious baskets were flagged.</p>')
        else:
            bd = sh["by_decision"]
            o.append(f'<p>{sh["alerts"]} baskets were flagged: {bd.get("confirmed_theft", 0)} confirmed as theft, '
                     f'{bd.get("not_theft", 0)} judged not theft, {bd.get("unreviewed", 0)} not reviewed yet.</p>')
            if sh["confirmed_items"]:
                o.append("<p>Confirmed items: " + E(", ".join(f"{k} x{v}" for k, v in sh["confirmed_items"].items())) + ".</p>")
            if review_pending:
                o.append('<table><tr><th>Time</th><th>Camera</th><th>What</th><th class="n">Confidence</th></tr>')
                for a in review_pending[:10]:
                    what = ", ".join(i.get("category") or "item" for i in a.get("items") or []) or "items"
                    o.append(f'<tr><td>{_t(a["event_ts"], tz)}</td><td>{E(str(a.get("camera") or ""))}</td><td>{E(what)}</td><td class="n">{(a.get("confidence") or 0):.2f}</td></tr>')
                o.append("</table>")
        # health
        o.append('<h2>Are the cameras working?</h2>')
        if rep["cameras"]:
            span = rep["end"] - rep["start"]
            o.append('<table><tr><th>Camera</th><th class="n">Seen working</th><th class="n">Last heard</th></tr>')
            for c, u in rep["cameras"].items():
                o.append(f'<tr><td>{E(c)}</td><td class="n">{_dur(u["covered_s"])}</td><td class="n">{_t(u["last_ts"], tz)}</td></tr>')
            o.append("</table>")
        else:
            o.append('<p class="muted">No camera heartbeats recorded.</p>')
    gen = dt.datetime.fromtimestamp(generated, tz) if generated else dt.datetime.now(tz)
    o.append(f'<footer>Generated {gen:%Y-%m-%d %H:%M %Z}. {rep["signals"]} data points. '
             'Counts come from cameras and are never an accusation: check a flagged item yourself before acting.</footer></main></body></html>')
    return "".join(o)


def weekly_digest(days: list[tuple[dt.date, dict]], *, store_name: str, tz, prev_total_in: int | None = None,
                  tz_assumed: bool = False, notes: list[str] | None = None, generated: float | None = None) -> str:
    """`days`: [(date, day_report)] in order (usually 7). Adds a weekday x hour heat strip and a week-on-week line."""
    srcs: dict[str, int] = {}
    for _, r in days:
        for k, v in r["sources"].items():
            srcs[k] = srcs.get(k, 0) + v
    srcs = dict(sorted(srcs.items(), key=lambda kv: ("simulated", "real_unlabelled", "real_model_labelled", "real_ground_truth", "live").index(kv[0])))
    totals = [r["door"]["in"] for _, r in days]
    with_data = [(d, r) for d, r in days if r["signals"]]
    total = sum(totals)
    lead = []
    if not with_data:
        lead.append("No camera data this week.")
    else:
        best = max(with_data, key=lambda dr: dr[1]["door"]["in"])
        lead.append(f"About {total} people came in over {len(with_data)} day{'s' if len(with_data) != 1 else ''} with data.")
        if best[1]["door"]["in"]:
            lead.append(f"Busiest day: {best[0]:%A} ({best[1]['door']['in']}).")
        if prev_total_in:
            ch = (total - prev_total_in) / prev_total_in * 100
            lead.append(f"That is {abs(ch):.0f}% {'more' if ch >= 0 else 'less'} than the week before ({prev_total_in}).")
        if len(with_data) < len(days):
            lead.append(f"Only {len(with_data)} of {len(days)} days had camera data, so compare weeks with care.")
    hours_sum = defaultdict(int)
    for _, r in days:
        for h in r["by_hour"]:
            hours_sum[h["hour"]] += h["in"]
    peak_hours = sorted(hours_sum.items(), key=lambda kv: -kv[1])[:3]
    q_peaks = [(d, r["queue"]["peak"]) for d, r in days if r["queue"]["peak"] is not None]
    issues = defaultdict(lambda: [0, 0.0])
    for _, r in days:
        for e in r["shelf_issues"]:
            k = (e["zone"], e["state"])
            issues[k][0] += 1
            issues[k][1] += e["seconds"]
    shr = defaultdict(int)
    for _, r in days:
        for k, v in r["shrink"]["by_decision"].items():
            shr[k] += v
    start, end = days[0][0], days[-1][0]
    o = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>{E(store_name)} weekly digest {start:%Y-%m-%d}</title><style>{CSS}</style></head><body><main>',
         f'<h1>{E(store_name)}: week of {start:%B %-d} to {end:%B %-d, %Y}</h1><p class="sub">Weekly digest</p>',
         '<p class="noprint"><button onclick="window.print()">Print</button></p>',
         _trust_block(srcs, tz_assumed, notes or []), f'<p class="lead">{E(" ".join(lead))}</p>']
    if with_data:
        o.append('<h2>People through the door, by day</h2>')
        o.append(bar_chart([r["door"]["in"] if r["signals"] else None for _, r in days], [f"{d:%a}" for d, _ in days],
                           title="People through the door by day"))
        o.append('<h2>Busy times, by weekday and hour</h2>')
        o.append(_heat(days, tz))
        if peak_hours and peak_hours[0][1]:
            o.append('<p>Busiest hours of the week: ' + E(", ".join(f"{h:02d}:00 ({n})" for h, n in peak_hours if n)) + ".</p>")
        o.append('<h2>Register line</h2>')
        if q_peaks:
            o.append('<table><tr><th>Day</th><th class="n">Longest line</th><th class="n">Time with 2+ waiting</th></tr>')
            for d, r in days:
                qq = r["queue"]
                if qq["peak"] is not None:
                    o.append(f'<tr><td>{d:%a %b %-d}</td><td class="n">{qq["peak"]}</td><td class="n">{(qq["share_2plus"] or 0) * 100:.0f}%</td></tr>')
            o.append("</table>")
        else:
            o.append('<p class="muted">No register-line data this week.</p>')
        o.append('<h2>Shelf and cooler problems</h2>')
        if issues:
            o.append('<table><tr><th>Where</th><th>Problem</th><th class="n">Times</th><th class="n">Total time</th></tr>')
            for (z, st), (n, s) in sorted(issues.items(), key=lambda kv: -kv[1][1])[:10]:
                o.append(f'<tr><td>{E(z or "")}</td><td>{E(st.replace("_", " "))}</td><td class="n">{n}</td><td class="n">{_dur(s)}</td></tr>')
            o.append("</table>")
        else:
            o.append('<p class="muted">No shelf or cooler problems recorded.</p>')
        o.append('<h2>Theft risk and review</h2>')
        n = sum(shr.values())
        if n:
            o.append(f'<p>{n} baskets flagged this week: {shr.get("confirmed_theft", 0)} confirmed theft, {shr.get("not_theft", 0)} not theft, '
                     f'{shr.get("unreviewed", 0)} not reviewed. Items confirmed taken: '
                     + E(", ".join(f"{k} x{v}" for k, v in _merge(r["shrink"]["confirmed_items"] for _, r in days).items()) or "none") + ".</p>")
        else:
            o.append('<p>No suspicious baskets were flagged.</p>')
    gen = dt.datetime.fromtimestamp(generated, tz) if generated else dt.datetime.now(tz)
    o.append(f'<footer>Generated {gen:%Y-%m-%d %H:%M %Z}. Flags are prompts to look, not accusations.</footer></main></body></html>')
    return "".join(o)


def _merge(dicts):
    out = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return dict(out)


def _heat(days, tz) -> str:
    """Rows = days, columns = hours; colour intensity = people in. Hours with no data are blank."""
    cw, ch, left, top = 28, 22, 44, 16
    width = left + 24 * cw
    vmax = max([h["in"] for _, r in days for h in r["by_hour"]] + [1])
    o = [f'<svg viewBox="0 0 {width} {top + ch * len(days) + 4}" width="100%" role="img" aria-label="People per hour by weekday">']
    for h in range(0, 24, 2):
        o.append(f'<text x="{left + h * cw + cw / 2}" y="11" text-anchor="middle">{h:02d}</text>')
    for r_i, (d, r) in enumerate(days):
        y = top + r_i * ch
        o.append(f'<text x="{left - 6}" y="{y + 15}" text-anchor="end">{d:%a}</text>')
        got = {h["hour"]: h["in"] for h in r["by_hour"]}
        for h in range(24):
            v = got.get(h)
            if v is None:
                o.append(f'<rect x="{left + h * cw}" y="{y}" width="{cw - 1}" height="{ch - 1}" fill="none" stroke="var(--line)"/>')
            else:
                op = 0.12 + 0.88 * v / vmax
                o.append(f'<rect x="{left + h * cw}" y="{y}" width="{cw - 1}" height="{ch - 1}" fill="var(--bar)" fill-opacity="{op:.2f}"><title>{d:%a} {h:02d}:00: {v}</title></rect>')
    o.append("</svg>")
    return "".join(o)
