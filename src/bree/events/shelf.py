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


def _doubt(person, t: float) -> dict:
    tu = getattr(person, "t_uncertain", None)
    if getattr(person, "uncertain", None) and (tu is None or t >= tu):
        return {"identity_uncertain": person.uncertain, "uncertain": True}
    return {}


def store_events(shelf: list[dict], people: list, layout: dict, cams: dict | None = None, conceal: list[dict] | None = None,
                 assoc_cfg: AssocConfig | None = None, register_dwell_s: float = 1.0) -> tuple[list[Event], list[Assoc]]:
    """-> (events in time order, the association of each shelf event). Staff identities make no events."""
    acfg = assoc_cfg or AssocConfig()
    slots = slot_index(layout)
    by_id = {p.id: p for p in people}
    assocs = associate(shelf, people, layout, cams=cams, cfg=acfg)
    events: list[Event] = []
    _, reg = counter_zones(layout)
    for p in people:
        if getattr(p, "staff", False) or not p.path:
            continue
        t0 = p.path[0][0]
        born = getattr(p, "born", "")
        events.append(Event(EventType.ENTER, t0, p.id, meta={"born": born, **({"missed_entry": True} if born == "inside" else {}), **_doubt(p, t0)}))
        for a, b in register_visits(p, reg, register_dwell_s):
            events.append(Event(EventType.PAY, a + register_dwell_s, p.id, zone=REGISTER, meta={"phase": "start", "t_start": a, **_doubt(p, a)}))
            events.append(Event(EventType.PAY, b, p.id, zone=REGISTER, meta={"phase": "end", **_doubt(p, b)}))
        if getattr(p, "state", "") == "exited":
            te = p.t_exit if p.t_exit is not None else p.path[-1][0]
            events.append(Event(EventType.EXIT, te, p.id, zone="exit", meta=_doubt(p, te)))
    for ev, a in zip(shelf, assocs):
        if a.person_id is None:
            continue
        s = slots.get(ev.get("slot_id")) or {}
        sku = ev.get("sku_id") or s.get("skuId")          # the planogram names the product when the camera could not
        rivals = [pid for pid, c in a.candidates if pid != a.person_id and c <= (a.cost or 0.0) + acfg.margin]
        doubt = _doubt(by_id[a.person_id], float(ev["t"]))
        meta = {"slot": {"id": ev.get("slot_id"), "fixture": s.get("fixtureId"), "sku": s.get("skuId")},
                "shelf": {k: ev.get(k) for k in ("camera_id", "source", "sku_conf", "point_3d", "evidence")},
                "assoc": {"cost": a.cost, "margin": a.margin if np.isfinite(a.margin) else None, "why": a.why},
                "uncertain": bool(a.uncertain), **doubt}
        kind = EventType.PICK if ev["kind"] == "take" else EventType.PUT_BACK
        for _ in range(max(int(ev.get("count") or 1), 1)):
            events.append(Event(kind, float(ev["t"]), a.person_id, item=sku or "unknown", sku=sku, zone=s.get("fixtureId"),
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
