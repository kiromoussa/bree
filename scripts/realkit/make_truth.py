#!/usr/bin/env python
"""The act log of a filming session (a CSV filled in by hand) -> the truth files the benchmark scorer reads.

    .venv/bin/python scripts/realkit/make_truth.py acts.csv --session session.yaml --out work/clip

The log has one row per item handled. Columns (the first six are needed):

    act          number from the shot list. Two rows with one act number: two people at once, or one person, two items
    clap_time    what the phone stopwatch showed at the clap that starts the act (m:ss, h:mm:ss or seconds)
    actor        who (a first name; `staff` rows are the restocker)
    product      the product id from session.yaml
    slot         S<shelf>-<position>, shelf 1 at the bottom, position 1 at the left
    outcome      paid, put_back, put_back_wrong, concealed_pocket, concealed_waistband, concealed_bag,
                 concealed_jacket, walk_out, walk_past, shift, restock, staff_take
    to_slot      put_back_wrong only: the slot the item went into
    second_time  optional: stopwatch time of the second moment (the hiding, the put-back, the pretend payment)
    light        optional: bright (default) or dim
    notes        optional, kept

Times: every clip of the session starts at the sync clap (run_real.py prepare cuts them there, using each camera's
own offset), so an act's time in the clips is clap_time minus the stopwatch reading at the sync clap, plus `lead_s`
(the hand reaches the shelf about 2 s after the clap). The scorer allows a few seconds either way.

Each person in each act is scored as one shopper visit (`A07-maya`), so "honest shoppers flagged" counts visits.
Written into <out>/truth: events.jsonl, acts.jsonl, shoppers.json, faults.json, planogram.json, and empty
frames.jsonl and tracks.jsonl (nobody drew person boxes: the scorer's "right shopper" stays empty on real footage).
Written into <out>: register.jsonl, one pretend receipt per paying visit. Swap in the real register export when
there is one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session import FIXTURE, build_layout, load_session, parse_clock      # noqa: E402

CONCEAL = {"concealed_pocket": "pocket", "concealed_waistband": "waistband", "concealed_bag": "bag", "concealed_jacket": "jacket",
           "concealed": "unspecified", "walk_out": "in_hand"}        # walk_out: carried past the register in plain sight, never paid
OUTCOMES = ("paid", "put_back", "put_back_wrong", *CONCEAL, "walk_past", "shift", "restock", "staff_take")
AFTER_S = {"concealed": 4.0, "put_back": 6.0, "paid": 15.0}       # second moment after the take when the log gives none
RECEIPT_LAG_S = 3.0      # bree.shelf.store.POS_LAG_S expects a receipt 1.5 to 4.5 s after the payment


def read_log(path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("act") and not r["act"].startswith("#")]


def make_truth(rows: list[dict], sync_clock_s: float = 0.0, lead_s: float = 2.0, fps: float = 10.0, planogram: dict | None = None) -> dict:
    """rows of the log -> {"events", "acts", "shoppers", "faults", "planogram", "register", "warnings"}.
    planogram: {slot id: product id} of the session; a slot that is not in it is an error, a product that differs a warning."""
    events, acts, register, warnings, people = [], [], [], [], {}
    actors_of: dict[str, set] = {}
    for r in rows:
        actors_of.setdefault(r["act"], set()).add(r["actor"])
    for n, r in enumerate(rows, start=2):          # line 1 of the file is the header
        where = f"line {n} (act {r.get('act')})"
        for k in ("act", "clap_time", "actor", "outcome"):
            if not r.get(k):
                raise ValueError(f"{where}: `{k}` is empty")
        out = r["outcome"].lower()
        if out not in OUTCOMES:
            raise ValueError(f"{where}: outcome {r['outcome']!r} is not one of {', '.join(OUTCOMES)}")
        act = f"{int(r['act']):02d}" if r["act"].isdigit() else r["act"]
        t = round(parse_clock(r["clap_time"]) - sync_clock_s + lead_s, 2)
        if t < 0:
            raise ValueError(f"{where}: clap_time {r['clap_time']} is before the sync clap")
        staff = out in ("restock", "staff_take") or r["actor"].lower() == "staff"
        name = f"A{act}-{r['actor']}"
        p = people.setdefault(name, {"shopper": name, "thief": False, "role": "staff" if staff else "shopper",
                                     "group": f"A{act}" if len(actors_of[r["act"]]) > 1 else None, "tEnter": max(0.0, t - lead_s - 3), "tExit": None, "act": r["act"], "actor": r["actor"]})
        if out == "walk_past":
            continue
        slot, sku = r.get("slot"), r.get("product")
        if not slot or not sku:
            raise ValueError(f"{where}: `slot` and `product` are needed for {out}")
        for s in filter(None, (slot, r.get("to_slot"))):
            if planogram is not None and s not in planogram:
                raise ValueError(f"{where}: slot {s!r} is not on the shelf of the session file")
        if planogram is not None and planogram[slot] != sku and out != "restock":
            warnings.append(f"{where}: the session file says {slot} holds {planogram[slot]}, the log says {sku}")
        t2 = round(parse_clock(r["second_time"]) - sync_clock_s, 2) if r.get("second_time") else None
        base = {"t": t, "frame": round(t * fps), "shopper": name, "skuId": sku, "slotId": slot, "fixtureId": FIXTURE, "zone": "gondola", "zoneId": FIXTURE,
                "cameras": [], "act": r["act"], "actor": r["actor"], "light": (r.get("light") or "bright").lower(), "notes": r.get("notes") or ""}
        if out == "shift":
            acts.append({"kind": "touch", **base})
        elif out == "restock":
            acts.append({"kind": "staff_put", **base, "from": None})
        elif out == "staff_take":
            acts.append({"kind": "staff_take", **base, "outcome": "carried_out", "tPut": None, "putSlot": None})
        else:
            kind = "concealed" if out in CONCEAL else "put_back" if out.startswith("put_back") else "paid"
            t2 = t2 if t2 is not None else round(t + AFTER_S[kind], 2)
            if out == "put_back_wrong" and not r.get("to_slot"):
                raise ValueError(f"{where}: put_back_wrong needs `to_slot`")
            p["thief"] |= kind == "concealed"
            events.append({**base, "thief": kind == "concealed", "outcome": kind, "tResolved": t2, "tExit": None,
                           "tConceal": t2 if kind == "concealed" else None, "tPay": t2 if kind == "paid" else None, "tPutBack": t2 if kind == "put_back" else None,
                           "where": CONCEAL.get(out), "group": p["group"], "paidBy": None, "tHand": None, "scanDropped": False,
                           "putBackSlot": r.get("to_slot") if out == "put_back_wrong" else None})
            if kind == "paid":
                rec = next((x for x in register if x["_who"] == name), None)
                if rec is None:
                    register.append(rec := {"t": round(t2 + RECEIPT_LAG_S, 2), "terminal": "pos_1", "txn_id": f"REAL{len(register) + 1:04d}", "items": [], "_who": name})
                rec["items"].append({"sku": sku, "qty": 1})
    events.sort(key=lambda e: e["t"])
    acts.sort(key=lambda e: e["t"])
    lights = {(r.get("light") or "bright").lower() for r in rows}
    night = lights == {"dim"}
    return {"events": events, "acts": acts, "shoppers": list(people.values()),
            "faults": {"features": {"night": night}, "light": "night" if night else "day", "camera_faults": {}, "note": "REAL footage, labelled from the act log"},
            "planogram": {"note": "the shelf as the session file gives it", "slots_where_the_item_differs": []},
            "register": [{k: v for k, v in x.items() if k != "_who"} for x in sorted(register, key=lambda x: x["t"])], "warnings": warnings}


def write_truth(truth: dict, out) -> Path:
    """<out>/truth/* and <out>/register.jsonl. Returns the truth folder."""
    out = Path(out)
    d = out / "truth"
    d.mkdir(parents=True, exist_ok=True)
    jl = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)      # noqa: E731
    (d / "events.jsonl").write_text(jl(truth["events"]))
    (d / "acts.jsonl").write_text(jl(truth["acts"]))
    (d / "frames.jsonl").write_text("")
    (d / "tracks.jsonl").write_text("")
    for k in ("shoppers", "faults", "planogram"):
        (d / f"{k}.json").write_text(json.dumps(truth[k], indent=1))
    (out / "register.jsonl").write_text(jl(truth["register"]))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="the act log CSV")
    ap.add_argument("--session", help="session.yaml: gives the sync clock, lead_s and the shelf to check slots against")
    ap.add_argument("--sync-clock", default="0", help="without --session: the stopwatch reading at the sync clap")
    ap.add_argument("--lead-s", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="folder to write truth/ and register.jsonl into")
    a = ap.parse_args(argv)
    s = load_session(a.session) if a.session else None
    plan = {x["id"]: x["skuId"] for x in build_layout(s)["slots"]} if s else None
    try:
        t = make_truth(read_log(a.log), s["sync_clock_s"] if s else parse_clock(a.sync_clock), s["lead_s"] if s else a.lead_s, s["fps"] if s else 10.0, plan)
    except ValueError as e:
        raise SystemExit(f"{a.log}: {e}")
    d = write_truth(t, a.out)
    for w in t["warnings"]:
        print("warning:", w)
    print(f"{len(t['events'])} takes ({sum(e['outcome'] == 'concealed' for e in t['events'])} stolen), {len(t['acts'])} staff or shift acts, "
          f"{len(t['shoppers'])} visits, {len(t['register'])} receipts -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
