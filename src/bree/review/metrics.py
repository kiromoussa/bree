"""Metrics from the review store, and the weekly owner report.

Definitions (used everywhere in this module):
  sent        alerts recorded in the store in the period (by event time), minus two kinds that
              are counted on their own and are in no other figure:
              - alerts caused by staged tests (`staged_alerts`): an alert that names a test id, or
                one matched to a logged test by time and camera. Tests cannot move precision.
              - alerts a late receipt withdrew completely (`retracted_alerts`). Staff were told to
                stand down, and they are not queued for review. An alert a late receipt only
                lowered to the review tier stays in.
              `metrics()` and the owner report use the same rule, so their numbers agree.
              The same event logged under two alert ids counts twice here (the label export
              dedups by content, the metrics do not).
  reviewed    alerts with at least one reviewer decision
  review_rate reviewed / sent
  precision   (confirmed_theft + wrong_item) / (confirmed_theft + wrong_item + not_theft).
              A wrong_item alert still caught a real theft. `unclear` and `disputed` are left out.
  item_precision  confirmed_theft / (confirmed_theft + wrong_item + not_theft): theft AND item right
  median_event_to_decision_s  median of (first reviewer decision time minus event time)
  median_view_s               median time a reviewer spent on one alert (open to decision)
  agreement   over alerts seen by two or more reviewers, using the first two: share with the same
              decision, and Cohen's kappa (None when there is nothing to compare or no variation)
Weeks are ISO weeks in UTC. Event time is the wall-clock time of the alert when the record has
one, else the time it was ingested (see bree.review.store).
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path
from statistics import median

from bree.review.store import DECISIONS, DISPUTED, RETRACTED

UTC = dt.timezone.utc
STAGED_BEFORE_S, STAGED_AFTER_S = 30.0, 300.0     # an alert this close to a staged test belongs to it


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _week(ts: float) -> str:
    y, w, _ = dt.datetime.fromtimestamp(ts, UTC).isocalendar()
    return f"{y}-W{w:02d}"


def _block(alerts: list[dict]) -> dict:
    c = Counter(a["decision"] for a in alerts if a["decision"])
    theft = c["confirmed_theft"] + c["wrong_item"]
    lag = [a["first_decided_ts"] - a["event_ts"] for a in alerts if a["first_decided_ts"] is not None]
    view = [r["time_to_decision_s"] for a in alerts for r in a["reviews"] if r["time_to_decision_s"] is not None]
    reviewed = sum(c.values())
    return {"sent": len(alerts), "reviewed": reviewed, "review_rate": _ratio(reviewed, len(alerts)),
            **{d: c[d] for d in (*DECISIONS, DISPUTED)}, "not_reviewed": len(alerts) - reviewed,
            "precision": _ratio(theft, theft + c["not_theft"]),
            "item_precision": _ratio(c["confirmed_theft"], theft + c["not_theft"]),
            "median_event_to_decision_s": round(median(lag), 1) if lag else None,
            "median_view_s": round(median(view), 1) if view else None}


def _by(alerts: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = {}
    for a in alerts:
        groups.setdefault(str(key(a)), []).append(a)
    return {k: _block(v) for k, v in sorted(groups.items())}


def agreement(alerts: list[dict]) -> dict:
    pairs = [(a["reviews"][0]["decision"], a["reviews"][1]["decision"]) for a in alerts if len(a["reviews"]) >= 2]
    n = len(pairs)
    same = sum(1 for x, y in pairs if x == y)
    kappa = None
    if n:
        po = same / n
        pe = sum((sum(1 for x, _ in pairs if x == d) / n) * (sum(1 for _, y in pairs if y == d) / n) for d in DECISIONS)
        kappa = round((po - pe) / (1 - pe), 3) if pe < 1 else None
    return {"alerts_with_two_reviewers": n, "same_decision": same, "percent_agreement": _ratio(same, n),
            "cohens_kappa": kappa}


def _customer(store, since: float, until: float) -> tuple[list[dict], dict]:
    """Alerts in [since, until) split by the `sent` rule above: (customer alerts, the rest).
    Staged matching looks across the period's edges, so an alert just inside the period that
    belongs to a test just outside it is still a staged one."""
    alerts = store.alerts(since, until)
    nearby = store.alerts(since - STAGED_AFTER_S - STAGED_BEFORE_S, until + STAGED_AFTER_S)
    matched = match_staged(store, nearby, since - STAGED_AFTER_S, until)
    logged = {t["test_id"] for t in store.staged_tests()}
    staged_ids = {s["alert_id"] for s in matched if s["caught"]} | {a["alert_id"] for a in alerts if a["staged_test_id"]}
    staged = [a for a in alerts if a["alert_id"] in staged_ids]
    retracted = [a for a in alerts if a["alert_id"] not in staged_ids and a["retracted_tier"] == RETRACTED]
    out = {a["alert_id"] for a in staged + retracted}
    return [a for a in alerts if a["alert_id"] not in out], {
        "recorded": len(alerts), "staged": staged, "retracted": retracted,
        "tests": [s for s in matched if since <= s["ts"] < until],
        "unlogged": sorted({a["staged_test_id"] for a in staged if a["staged_test_id"] and a["staged_test_id"] not in logged})}


def metrics(store, since: float | None = None, until: float | None = None) -> dict:
    alerts, rest = _customer(store, -1e18 if since is None else since, 1e18 if until is None else until)
    return {"alerts_recorded": rest["recorded"], "staged_alerts": len(rest["staged"]),
            "retracted_alerts": len(rest["retracted"]),
            "overall": _block(alerts), "by_week": _by(alerts, lambda a: _week(a["event_ts"])),
            "by_zone": _by(alerts, lambda a: a["zone"]), "by_camera": _by(alerts, lambda a: a["camera"]),
            "agreement": agreement(alerts)}


def match_staged(store, alerts: list[dict], since: float, until: float,
                 before_s: float = STAGED_BEFORE_S, after_s: float = STAGED_AFTER_S) -> list[dict]:
    """Staged test thefts in the period, each caught (an alert was sent for it) or missed. An
    alert matches when it names the test id, or its event time is within [-before_s, +after_s]
    of the staged time on the same camera. A test logged with no camera matches an alert on ANY
    camera in that window, so log the camera. One alert per test."""
    used: set[str] = set()
    out = []
    for t in store.staged_tests():
        if not since <= t["ts"] < until:
            continue
        hit = next((a for a in alerts if a["alert_id"] not in used and a["staged_test_id"] == t["test_id"]), None)
        if hit is None:
            near = [a for a in alerts if a["alert_id"] not in used and not a["staged_test_id"]
                    and -before_s <= a["event_ts"] - t["ts"] <= after_s
                    and (not t["camera"] or a["camera"] == t["camera"])]
            hit = min(near, key=lambda a: abs(a["event_ts"] - t["ts"]), default=None)
        if hit:
            used.add(hit["alert_id"])
        out.append({**t, "caught": hit is not None, "alert_id": hit and hit["alert_id"],
                    "predicted_item": hit and hit["predicted_item"], "decision": hit and hit["decision"]})
    return out


def owner_report(store, week_start: str | dt.date, out_dir: str | Path | None = None) -> dict:
    """Seven days from `week_start` (YYYY-MM-DD, UTC). Returns the report dict; with `out_dir`
    also writes owner_report_<week_start>.json and .md. Alerts caused by staged tests are counted
    under "staged tests" and left out of the precision figure, so tests cannot inflate it. That
    covers every alert that names a test id, logged or not: ids with no logged test are listed
    under `staged_alerts_with_no_logged_test`. Alerts a late receipt withdrew are counted on
    their own too. Matching looks across the week boundary: a test staged just before midnight is
    caught by an alert a few minutes into the next week, and that alert is a staged one in the
    next report."""
    d0 = dt.date.fromisoformat(week_start) if isinstance(week_start, str) else week_start
    since = dt.datetime(d0.year, d0.month, d0.day, tzinfo=UTC).timestamp()
    until = since + 7 * 86400
    customer, rest = _customer(store, since, until)
    staged = rest["tests"]                                             # this week's tests
    real = _block(customer)
    rep = {
        "week_start": d0.isoformat(), "week_end": (d0 + dt.timedelta(days=6)).isoformat(),
        "alerts_sent": rest["recorded"], "alerts_from_staged_tests": len(rest["staged"]),
        "alerts_withdrawn_by_late_receipt": len(rest["retracted"]),
        "staged_alerts_with_no_logged_test": rest["unlogged"],
        "customer_alerts": {"sent": real["sent"], "confirmed": real["confirmed_theft"] + real["wrong_item"],
                            "confirmed_wrong_item": real["wrong_item"], "rejected": real["not_theft"],
                            "unclear": real["unclear"], "reviewers_disagree": real[DISPUTED],
                            "not_reviewed": real["not_reviewed"], "precision": real["precision"],
                            "review_rate": real["review_rate"],
                            "median_event_to_decision_s": real["median_event_to_decision_s"]},
        "staged_tests": {"staged": len(staged), "caught": sum(s["caught"] for s in staged),
                         "missed": sum(not s["caught"] for s in staged), "tests": staged},
        "by_zone": _by(customer, lambda a: a["zone"]),
        "by_camera": _by(customer, lambda a: a["camera"]),
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"owner_report_{d0.isoformat()}.json").write_text(json.dumps(rep, indent=2))
        (out / f"owner_report_{d0.isoformat()}.md").write_text(owner_markdown(rep))
    return rep


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def _dur(s) -> str:
    return "n/a" if s is None else f"{s:.0f} s" if s < 120 else f"{s / 60:.0f} min" if s < 7200 else f"{s / 3600:.1f} h"


def owner_markdown(rep: dict) -> str:
    c, s = rep["customer_alerts"], rep["staged_tests"]
    lines = [
        f"# BREE weekly report: {rep['week_start']} to {rep['week_end']}", "",
        f"- Alerts sent: **{rep['alerts_sent']}** ({rep['alerts_from_staged_tests']} from staged tests, "
        f"{rep['alerts_withdrawn_by_late_receipt']} withdrawn after a late receipt, {c['sent']} from customers)",
        f"- Confirmed theft: **{c['confirmed']}**" + (f" ({c['confirmed_wrong_item']} with the wrong item named)" if c["confirmed_wrong_item"] else ""),
        f"- Rejected (not theft): **{c['rejected']}**",
        f"- Unclear: {c['unclear']}. Reviewers disagreed: {c['reviewers_disagree']}. Not reviewed yet: {c['not_reviewed']}.",
        (f"- Of the customer alerts with a clear decision, {_pct(c['precision'])} were real theft." if c["precision"] is not None
         else "- No customer alert has a clear decision yet, so there is no real theft share to report."),
        (f"- Reviewed: {_pct(c['review_rate'])} of customer alerts. Typical time from the event to a decision: {_dur(c['median_event_to_decision_s'])}."
         if c["sent"] else "- No customer alerts this week."),
        "", "## Staged test thefts", "",
        f"Staged: {s['staged']}. Caught: **{s['caught']}**. Missed: **{s['missed']}**.", "",
    ]
    if rep["staged_alerts_with_no_logged_test"]:
        lines += ["Alerts that name a staged test nobody logged (counted as staged, not as customer alerts): "
                  + ", ".join(rep["staged_alerts_with_no_logged_test"]) + ".", ""]
    if s["tests"]:
        lines += ["| Test | Camera | Item staged | Result | Item the system named |", "|---|---|---|---|---|"]
        lines += [f"| {t['test_id']} | {t['camera'] or ''} | {t['item'] or ''} | {'caught' if t['caught'] else 'MISSED'} | {t['predicted_item'] or ''} |"
                  for t in s["tests"]]
        lines.append("")
    for title, key in (("By zone", "by_zone"), ("By camera", "by_camera")):
        lines += [f"## {title} (customer alerts)", "", "| | Sent | Confirmed | Rejected | Real theft share |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {b['sent']} | {b['confirmed_theft'] + b['wrong_item']} | {b['not_theft']} | {_pct(b['precision'])} |"
                  for k, b in rep[key].items()]
        lines.append("")
    return "\n".join(lines)
