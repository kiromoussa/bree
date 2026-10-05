"""Shelf events + store-wide people -> the events the ledger already understands.

This replaces the single-view pick rule for the item cameras (a rail camera sees the shelf and the item, rarely a
whole person): a shelf event in the shared contract says WHAT left WHICH slot WHEN; bree.track.associate says WHO;
this module turns the pair into PICK / PUT_BACK, and makes ENTER / PAY (register visit) / EXIT from the floor
tracks, so the unchanged ledger reconciles against the register feed.

    people, ids = track_people(cams, layout, boxes, fps)                 # bree.track.floor on the people cameras
    events, assocs = store_events(shelf_events, people, layout, cams)    # this module
    alerts, ledger = run_ledger(events, payments, layout)

Theft rule (the ledger's, unchanged): an item taken, not put back and not on a receipt when the person has left is
unpaid. Unpaid alone reaches "review"; a conceal cue on that item raises it to "alert". Conceal cues are an input
here ({"t", "sku_id"?, "conf", and "person_id" or "point_3d"}): on the simulated clips the overhead pose does not
show concealment (measured, see REPORT.md), so the cue has to come from an item camera.

Doubt is carried two ways, both already understood by the ledger:
  who took it   Event.candidates lists the other people who fit (the ledger halves the item's score and waits for
                their receipts); meta["uncertain"] is True
  who is who    meta["identity_uncertain"] = reason, from the floor tracker (the ledger caps the person at review)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bree.events.types import Catalog, Event, EventType, LineItem, Payment
from bree.track.associate import Assoc, AssocConfig, associate, slot_index
from bree.track.floor import FloorConfig, FloorTracker, counter_zones, in_zone

REGISTER = "register"


def track_people(cams: dict, layout: dict, boxes: dict[str, list[list]], fps: float, cfg: FloorConfig | None = None):
    """boxes[camera][frame] = [{"bbox", "kpts"?}] for the people cameras (overhead, entrance)
    -> (tracker, ids[camera][frame] = store-wide id of each detection or None)."""
    tracker = FloorTracker(cams, layout=layout, cfg=cfg)
    n = max((len(v) for v in boxes.values()), default=0)
    ids: dict[str, list[list]] = {c: [] for c in boxes}
    for f in range(n):
        got = tracker.update(f / fps, {c: (boxes[c][f] if f < len(boxes[c]) else []) for c in boxes})
        for c in boxes:
            ids[c].append(got[c])
    tracker.finish(n / fps)
    for t0, t1, a, b in tracker.swaps:        # ids given out while two tracks followed each other's person
        for f in range(int(t0 * fps), min(int(t1 * fps) + 1, n)):
            for c in ids:
                ids[c][f] = [b if i == a else a if i == b else i for i in ids[c][f]]
    return tracker, ids


def register_visits(person, zone, dwell_s: float = 1.0, gap_s: float = 1.0) -> list[tuple[float, float]]:
    """[(t_start, t_end)] the person stood in front of the counter for at least dwell_s."""
    out, cur = [], None
    for t, x, z in person.path:
        if in_zone(zone, x, z, 0.1):
            cur = [t, t] if cur is None or t - cur[1] > gap_s else [cur[0], t]
            if out and out[-1][0] == cur[0]:
                out[-1] = cur
            else:
                out.append(cur)
    return [(a, b) for a, b in out if b - a >= dwell_s]


def parties(people: list, enter_within_s: float = 4.0, together_m: float = 1.5, share: float = 0.5, min_s: float = 5.0) -> dict[int, list[int]]:
    """{id: [ids of the people they shop with]}. Coming in within a few seconds of each other is not enough (strangers
    do that all day): a party also stays together, within `together_m` for at least `share` of the time both are on
    the floor tracks, over at least `min_s`."""
    paths = {p.id: {round(t, 1): (x, z) for t, x, z in p.path} for p in people if p.path and not getattr(p, "staff", False)}
    out: dict[int, list[int]] = {i: [] for i in paths}
    ids = sorted(paths)
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            common = sorted(set(paths[a]) & set(paths[b]))
            if abs(min(paths[a]) - min(paths[b])) > enter_within_s or len(common) < 2 or common[-1] - common[0] < min_s:
                continue
            near = sum(float(np.hypot(paths[a][t][0] - paths[b][t][0], paths[a][t][1] - paths[b][t][1])) <= together_m for t in common)
            if near >= share * len(common):
                out[a].append(b)
                out[b].append(a)
    return out


def _doubt(person, t: float) -> dict:
    tu = getattr(person, "t_uncertain", None)
    if getattr(person, "uncertain", None) and (tu is None or t >= tu):
        return {"identity_uncertain": person.uncertain, "uncertain": True}
    return {}


def confirm_puts(shelf: list[dict], assocs: list[Assoc], slots: dict, margin: float, same_place_m: float = 0.25, put_returns: str = "in") -> tuple[list[bool], dict[int, list[int]]]:
    """Which shelf events may reach the ledger, and which takes each put returns: (keep[i], {put index: [take index]}).
    A put of the pixel comparison names the reading it undoes ("undoes": that camera saw that place go back to the
    picture from before its own take, because the item is back or because an arm that had covered the slot went
    away). It takes that reading out of the act it was merged into. An act with no reading left is returned, by
    whoever was given the take. An act other cameras still read stands, under the name one of them gave it, unless
    the item was seen going into the slot (`put_returns` "in": the put's "item_in"; "both": any item in a hand near
    it; "none": never), which returns the act whatever the other cameras say. Puts without that link:
    Takes: all but those of the slot watch alone. Puts: about 6 put events in 10 match no act (4 of 23 with one cue on
    the TRAIN-seed clips), so a put counts when the item was seen in the hand going in (source "both") or when it lands
    where this person took something that is still out (same or neighbouring facing). Who puts it back is the one who
    took it: a put that fits two people about equally goes to the one with an item out from that place. Changes
    `assocs` in place for those."""
    out_by: dict[int, list[tuple[np.ndarray, int]]] = {}          # person -> (where from, take index) of what they still hold
    keep, pair = [True] * len(shelf), {}
    left = {i: set(ev.get("eids") or []) for i, ev in enumerate(shelf) if ev["kind"] == "take"}      # readings of each take not undone yet
    of = {x: i for i, names in left.items() for x in names}
    back: set[int] = set()         # takes returned through such a link

    def held(pid, at) -> int | None:
        return next((k for k, (q, _) in enumerate(out_by.get(pid, [])) if at is not None and float(np.linalg.norm(q - at)) <= same_place_m), None)
    for i in sorted(range(len(shelf)), key=lambda i: float(shelf[i]["t"])):
        ev, a = shelf[i], assocs[i]
        face = (slots.get(ev.get("slot_id")) or {}).get("face")
        at = np.asarray(face, float) if face is not None else None
        hit = sorted({of[x] for x in ev.get("undoes") or [] if x in of}) if ev["kind"] == "put" else []
        if hit:
            for x in ev["undoes"]:
                if x in of:
                    left[of[x]].discard(x)
            strong = bool(ev.get("item_in")) if put_returns == "in" else ev.get("source") == "both" if put_returns == "both" else False
            for k in hit:
                rest = [shelf[k]["read_as"][x] for x in sorted(left[k]) if x in (shelf[k].get("read_as") or {})]
                if rest and not strong and shelf[k].get("slot_id") not in {r[0] for r in rest}:
                    sid, sku = max(rest, key=lambda r: (r[2], r[3]))[:2]       # the act keeps a name a remaining reading gave it
                    shelf[k].update(slot_id=sid, sku_id=sku, **({"point_3d": list(slots[sid]["face"])} if sid in slots else {}))
            done = [k for k in hit if (strong or not left[k]) and k not in back and assocs[k].person_id is not None
                    and not (shelf[k].get("cue") == "slot_state" and shelf[k].get("source") != "both")]
            back.update(done)
            for k in done:
                held_k = out_by.get(assocs[k].person_id, [])
                held_k[:] = [h for h in held_k if h[1] != k]
            if done:
                pair[i] = done
                a.why += "; undoes the take at " + ", ".join(f"{float(shelf[k]['t']):.1f}s" for k in done) + " (the camera that read it saw the place as before)"
            else:
                keep[i] = False
                a.why += "; undoes one camera's reading of a take that is still read elsewhere, or that nobody was given: not passed to the ledger"
            continue
        if a.person_id is None:
            continue
        if ev["kind"] == "take":
            if ev.get("cue") == "slot_state" and ev.get("source") != "both":
                keep[i] = False         # 0 of 6 such events matched an act on the TRAIN-seed clips (make shelf-eval)
                a.why += "; slot watch alone (no pixel change, no item in a hand): not passed to the ledger"
            elif at is not None and i not in back:
                out_by.setdefault(a.person_id, []).append((at, i))
            continue
        if held(a.person_id, at) is None and a.margin < margin:
            rivals = [pid for pid, c in a.candidates if pid != a.person_id and c <= (a.cost or 0.0) + margin and held(pid, at) is not None]
            if len(rivals) == 1:
                a.why += f"; given to person {rivals[0]}, who took from this place (person {a.person_id} did not)"
                a.person_id, a.cost = rivals[0], dict(a.candidates)[rivals[0]]
        k = held(a.person_id, at)
        if k is not None:
            pair[i] = [out_by[a.person_id].pop(k)[1]]
        elif ev.get("source") != "both":
            keep[i] = False
            a.why += "; put not confirmed (no item seen in the hand, nothing of theirs out from this place): not passed to the ledger"
    return keep, pair


def store_events(shelf: list[dict], people: list, layout: dict, cams: dict | None = None, conceal: list[dict] | None = None,
                 assoc_cfg: AssocConfig | None = None, register_dwell_s: float = 1.0, put_returns: str = "in") -> tuple[list[Event], list[Assoc]]:
    """-> (events in time order, the association of each shelf event). Staff identities make no events."""
    acfg = assoc_cfg or AssocConfig()
    slots = slot_index(layout)
    by_id = {p.id: p for p in people}
    assocs = associate(shelf, people, layout, cams=cams, cfg=acfg)
    keep, pair = confirm_puts(shelf, assocs, slots, acfg.margin, put_returns=put_returns)
    name = lambda e: e.get("sku_id") or (slots.get(e.get("slot_id")) or {}).get("skuId")      # noqa: E731  the planogram names the product when the camera could not
    events: list[Event] = []
    _, reg = counter_zones(layout)
    party = parties(people)
    for p in people:
        if getattr(p, "staff", False) or not p.path:
            continue
        t0 = p.path[0][0]
        born = getattr(p, "born", "")
        events.append(Event(EventType.ENTER, t0, p.id, meta={"born": born, "party": party.get(p.id, []), **({"missed_entry": True} if born == "inside" else {}), **_doubt(p, t0)}))
        for a, b in register_visits(p, reg, register_dwell_s):
            events.append(Event(EventType.PAY, a + register_dwell_s, p.id, zone=REGISTER, meta={"phase": "start", "t_start": a, **_doubt(p, a)}))
            events.append(Event(EventType.PAY, b, p.id, zone=REGISTER, meta={"phase": "end", **_doubt(p, b)}))
        if getattr(p, "state", "") == "exited":
            te = p.t_exit if p.t_exit is not None else p.path[-1][0]
            events.append(Event(EventType.EXIT, te, p.id, zone="exit", meta=_doubt(p, te)))
    for i, (ev, a, ok) in enumerate(zip(shelf, assocs, keep)):
        if not ok:
            continue
        for k in pair.get(i, [])[1:] + ([pair[i][0]] if i in pair and assocs[pair[i][0]].person_id != a.person_id else []):
            # a put that undoes a take given to somebody else (or more than one take): that item goes back for them
            events.append(Event(EventType.PUT_BACK, max(float(ev["t"]), float(shelf[k]["t"]) + 0.01), assocs[k].person_id, item=name(shelf[k]) or "unknown",
                                sku=name(shelf[k]), zone=(slots.get(shelf[k].get("slot_id")) or {}).get("fixtureId"), confidence=0.9,
                                meta={"shelf": {x: ev.get(x) for x in ("camera_id", "source", "point_3d")}, "undoes": float(shelf[k]["t"])}))
        if a.person_id is None or (i in pair and assocs[pair[i][0]].person_id != a.person_id):
            continue
        s = slots.get(ev.get("slot_id")) or {}
        sku = name(shelf[pair[i][0]]) if i in pair else name(ev)      # a put back where they took it returns that item, whatever name this view gave it
        rivals = [pid for pid, c in a.candidates if pid != a.person_id and c <= (a.cost or 0.0) + acfg.margin]
        doubt = _doubt(by_id[a.person_id], float(ev["t"]))
        meta = {"slot": {"id": ev.get("slot_id"), "fixture": s.get("fixtureId"), "sku": s.get("skuId")},
                "shelf": {k: ev.get(k) for k in ("camera_id", "source", "sku_conf", "point_3d", "evidence")},
                "assoc": {"cost": a.cost, "margin": a.margin if np.isfinite(a.margin) else None, "why": a.why},
                "uncertain": bool(a.uncertain), **doubt}
        kind = EventType.PICK if ev["kind"] == "take" else EventType.PUT_BACK
        # one event per act: a count above 1 (the slot watch's row positions) was never right on the simulated clips
        # ponytail: two units taken in one reach are read as one; use ev["count"] once a clip set confirms it
        t = max(float(ev["t"]), float(shelf[pair[i][0]]["t"]) + 0.01) if i in pair else float(ev["t"])
        events.append(Event(kind, t, a.person_id, item=sku or "unknown", sku=sku, zone=s.get("fixtureId"),
                            confidence=1.0 if ev.get("source") == "both" else 0.9,
                            candidates=[a.person_id] + rivals if rivals else [], meta=meta))
    for c in conceal or []:
        pid = c.get("person_id")
        if pid is None and c.get("point_3d") is not None:
            near = [(float(np.linalg.norm(got[0] - np.asarray(c["point_3d"], float)[[0, 2]])), p.id) for p in people
                    if not getattr(p, "staff", False) and (got := p.at(float(c["t"]))) is not None]
            near.sort()
            if near and near[0][0] <= 1.0 and (len(near) == 1 or near[1][0] - near[0][0] >= 0.3):
                pid = near[0][1]
        if pid is not None and pid in by_id:
            events.append(Event(EventType.CONCEAL, float(c["t"]), pid, item=c.get("sku_id") or "unknown", sku=c.get("sku_id"),
                                confidence=float(c.get("conf", 0.6)), meta={"cue": c.get("source", "item_camera"), **_doubt(by_id[pid], float(c["t"]))}))
    events.sort(key=lambda e: (e.t, e.type != EventType.ENTER))
    return events, assocs


def load_payments(path) -> list[Payment]:
    """register.jsonl: {t, terminal, txn_id, items: [{sku, qty}]} per line."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append(Payment(t=float(r["t"]), terminal=r.get("terminal", REGISTER), txn_id=r.get("txn_id", ""),
                               items=[LineItem(sku=i.get("sku"), category=i.get("sku"), qty=int(i.get("qty", 1))) for i in r.get("items", [])]))
    return out


def run_ledger(events: list[Event], payments: list[Payment], layout: dict, **ledger_overrides):
    """Events + receipts through the existing ledger. Products are matched by exact SKU. -> (alerts, ledger)."""
    from bree.ledger.ledger import Ledger, LedgerConfig
    skus = {s["id"]: s["id"] for s in layout.get("skus", [])} | {s["skuId"]: s["skuId"] for s in layout.get("slots", []) if s.get("skuId")}
    ledger = Ledger(Catalog(skus), LedgerConfig(**ledger_overrides), terminal_zones={p.terminal: REGISTER for p in payments},
                    zone_kinds={REGISTER: "register", "exit": "exit"})
    return ledger.replay(events, payments), ledger


def write_run(out, events: list[Event], alerts, boxes: dict | None = None, ids: dict | None = None, fps: float = 10.0) -> None:
    """Write <out>/pipeline in the layout bree.sim.bench scores: events.jsonl, alerts.jsonl, frames_<camera>.jsonl
    (each person box with its store-wide id as "gid")."""
    from dataclasses import asdict
    pipe = Path(out) / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    (pipe / "events.jsonl").write_text("".join(json.dumps(e.to_dict(), default=str) + "\n" for e in events))
    (pipe / "alerts.jsonl").write_text("".join(json.dumps(asdict(a), default=str) + "\n" for a in alerts))
    for cam, frames in (boxes or {}).items():
        with (pipe / f"frames_{cam}.jsonl").open("w") as fh:
            for f, dets in enumerate(frames):
                persons = [{"id": i, "gid": (ids[cam][f][i] if ids else None), "bbox": d["bbox"], **({"kpts": d["kpts"]} if d.get("kpts") is not None else {})}
                           for i, d in enumerate(dets)]
                fh.write(json.dumps({"frame": f, "t": round(f / fps, 3), "persons": persons, "products": []}) + "\n")
