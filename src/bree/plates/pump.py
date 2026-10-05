"""Pump-zone tracker and the fuel drive-off rule.

    mon = DriveOffMonitor(zones, reader=make_reader(), plate_store=PlateStore(dir), out_dir=out)
    per frame:   alerts = mon.update(t, vehicles, frame)       # vehicles: [(track_id, (x1, y1, x2, y2))]
    dispenser:   mon.on_sale(FuelSale(pump_id, t_start, t_end, amount, txn_id=...))
    payment:     alerts = mon.on_payment(t, txn_id=...)         # or pump_id=...
    each alert:  sink.emit(a); review_store.add_alert(a.to_dict(), camera=...)   # the existing path

A "pump" here is one fuelling position (one side of a dispenser): that is the unit a forecourt
controller reports sales for. A zone is a polygon in one camera's image; a vehicle is in the zone
when the bottom centre of its box is inside.

The rule: a vehicle was in the pump zone while fuel was dispensed, the sale is not prepaid and
not paid, the vehicle has left the zone, and no payment arrived within `grace_s` after it left.
That gives a "review" alert, never an "alert": a person looks at it before anything happens.
A payment that arrives later retracts it (and deletes the stored plate).

A visit is a list of spans (unbroken stretches in the zone). A track id change with no gap carries
on the same span. A box back in the same place after a gap, while the visit has an unpaid sale,
adds a span to the same visit (someone stood in front of the camera). The visit is split again at
that gap when a new sale starts at the pump after it, or when the plate read after the gap
disagrees with the plate read before it: then it was the next car. The stored plate and the
evidence clip come only from spans that overlap the fuelling time of the sale.

Vehicle boxes come from the caller. `yolo_vehicles(model, frame)` gets them from the COCO YOLO
weights the repo already ships; `IouTracker` gives plain boxes track ids. Fuel sales and payments
come from the forecourt controller or POS: this module does not guess them from video.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from bree.alerts.types import Alert, UnpaidItem
from bree.alerts.writer import write_frames
from bree.plates.reader import Box, PlateRead, _iou, edit_distance, vote

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")


@dataclass
class PumpZone:
    pump_id: str
    polygon: list[tuple[float, float]]     # image pixels of `camera_id`
    camera_id: str = ""

    def contains(self, box: Box) -> bool:
        pt = ((box[0] + box[2]) / 2.0, float(box[3]))
        return cv2.pointPolygonTest(np.asarray(self.polygon, np.float32), pt, False) >= 0


@dataclass
class FuelSale:
    pump_id: str
    t_start: float                 # fuel started flowing (stream clock, seconds)
    t_end: float | None            # fuel stopped; None while still dispensing
    amount: float = 0.0            # sale value in dollars
    gallons: float = 0.0
    txn_id: str = ""
    grade: str | None = None
    prepaid: bool = False          # pay at pump or prepay inside: can not be a drive-off
    t_paid: float | None = None


@dataclass
class DriveOffConfig:
    grace_s: float = 120.0         # time to pay after the vehicle leaves the zone (moved the car, then went in)
    leave_s: float = 3.0           # not seen in the zone for this long = left
    rejoin_s: float = 60.0         # a box back in the same place within this long, sale unpaid = same vehicle
                                   # (no limit while that sale is still dispensing: the hose is in the car)
    same_place_iou: float = 0.5    # "the same place" for the two rules above
    plate_differs_edits: int = 2   # plates before and after a gap further apart than this = a different car
    min_dwell_s: float = 5.0       # shorter stays are cars driving through, not fuelling
    read_every_s: float = 0.5      # at most one plate read attempt per vehicle per this long
    max_reads: int = 12            # plate read attempts kept per span of a visit
    enough_conf: float = 0.9       # stop reading once the voted plate is this sure (3+ reads)
    min_plate_conf: float = 0.9    # voted plate below this: the alert says "plate not read", no text is stored
                                   # (synthetic check: the vote calibration table in results.md)
    late_payment_window_s: float = 86400.0   # a payment this long after the alert still retracts it
    no_vehicle_after_s: float = 300.0   # unpaid sale with no vehicle seen at all: flag after this long
    evidence_frames: int = 24
    purge_every_s: float = 3600.0  # the monitor purges expired plates from the plate store this often
    evidence_width: int = 640
    base_confidence: float = 0.6


@dataclass
class _Span:
    t0: float
    t1: float
    reads: list[PlateRead] = field(default_factory=list)
    best_crop: tuple[float, bytes] | None = None       # (read confidence, plate crop jpeg)
    frames: list[tuple[float, bytes]] = field(default_factory=list)


@dataclass
class _Visit:
    track_id: int
    pump_id: str
    camera_id: str
    spans: list[_Span]
    t_read: float = -1e9
    left: bool = False
    done: bool = False                                 # its alert was raised: nothing more to collect
    box: Box = (0, 0, 0, 0)                            # last box seen

    @property
    def t_enter(self) -> float:
        return self.spans[0].t0

    @property
    def t_last(self) -> float:
        return self.spans[-1].t1

    def during(self, a: float, b: float) -> list[_Span]:
        return [sp for sp in self.spans if min(sp.t1, b) > max(sp.t0, a)]

    def overlap(self, a: float, b: float) -> float:
        return sum(max(0.0, min(sp.t1, b) - max(sp.t0, a)) for sp in self.spans)


@dataclass
class _Pending:
    sale: FuelSale
    visit: _Visit | None
    rivals: int = 0
    alert: Alert | None = None


def _jpg(img: np.ndarray, width: int | None = None) -> bytes:
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


class DriveOffMonitor:
    def __init__(self, zones: list[PumpZone], reader=None, config: DriveOffConfig | None = None,
                 plate_store=None, out_dir: str | Path | None = None, fps: float = 5.0,
                 wall0: float | None = None):
        self.zones, self.reader, self.cfg = zones, reader, config or DriveOffConfig()
        self.plate_store, self.fps = plate_store, fps
        self.out = Path(out_dir) if out_dir else None
        self.wall0 = time.time() if wall0 is None else wall0     # epoch seconds at stream t = 0
        self.open: dict[tuple[str, int], _Visit] = {}
        self.closed: list[_Visit] = []
        self.sales: list[FuelSale] = []
        self.pending: list[_Pending] = []
        self.now = 0.0
        self.settled: dict[str, float] = {}       # txn ids already paid -> when (forgotten after a day)
        self._purged = 0.0

    # ------------------------------------------------------------------ inputs
    def update(self, t: float, vehicles: list[tuple[int, Box]], frame: np.ndarray | None = None,
               camera_id: str = "", evidence: np.ndarray | None = None) -> list[Alert]:
        """One frame of one camera. `evidence` is the frame to keep for the alert clip when it differs
        from `frame` (pass the head-pixelated copy when people are in view); plates are read on `frame`."""
        live = {tid for tid, _ in vehicles}
        for zone in (z for z in self.zones if z.camera_id == camera_id or not z.camera_id or not camera_id):
            for tid, box in vehicles:
                if not zone.contains(box):
                    continue
                key = (zone.pump_id, tid)
                v = self.open.get(key)
                if v is None:
                    v = self._rejoin(zone, tid, box, t, live) or \
                        _Visit(tid, zone.pump_id, zone.camera_id or camera_id, [_Span(t, t)])
                    self.open[key] = v
                v.spans[-1].t1, v.box = t, box
                if frame is None:
                    continue
                if self.reader is not None and t - v.t_read >= self.cfg.read_every_s and not self._sure(v):
                    v.t_read = t
                    self._read(v, frame, box)
                    v = self.open[key]                 # a plate that disagrees splits the visit
                fr = v.spans[-1].frames
                if len(fr) < self.cfg.evidence_frames and (not fr or t - fr[-1][0] >= 0.5):
                    fr.append((t, _jpg(frame if evidence is None else evidence, self.cfg.evidence_width)))
        return self.tick(t)

    def _rejoin(self, zone: PumpZone, tid: int, box: Box, t: float, live: set[int]) -> _Visit | None:
        """The visit a new track id at this pump carries on, if any."""
        cfg = self.cfg
        # The tracker gave the car a new id: an open visit in the same place whose track is not in this frame.
        for key, c in self.open.items():
            if key[0] == zone.pump_id and key[1] not in live and c.camera_id in ("", zone.camera_id) \
                    and _iou(c.box, box) >= cfg.same_place_iou:
                del self.open[key]
                c.track_id = tid
                return c
        # A box back in the same place after a gap (someone stood in front of the camera) while the visit
        # has an unpaid sale: the same visit, not a departure. It gets a new span, so the gap stays on record.
        # ponytail: place only. A different car that takes the spot is split off again when it starts its own
        # sale or its plate reads differently (_split). One that does neither (parks, shops, plate unreadable)
        # delays the first car's alert until it leaves; the alert names the gap. Compare car appearance if it matters.
        for c in reversed(self.closed):
            if c.pump_id == zone.pump_id and not c.done and _iou(c.box, box) >= cfg.same_place_iou and any(
                    s.pump_id == c.pump_id and not s.prepaid and s.t_paid is None
                    and c.overlap(s.t_start, t if s.t_end is None else s.t_end) > 0
                    and (s.t_end is None or t - c.t_last <= cfg.rejoin_s) for s in self.sales):
                self.closed.remove(c)
                c.left, c.track_id = False, tid
                c.spans.append(_Span(t, t))
                return c
        return None

    def _split(self, v: _Visit, k: int) -> None:
        """Spans k and later of `v` were a different vehicle: `v` keeps the earlier ones and has left."""
        new = _Visit(v.track_id, v.pump_id, v.camera_id, v.spans[k:], t_read=v.t_read, left=v.left, box=v.box)
        v.spans = v.spans[:k]
        if v.left:
            self.closed.append(new)
        else:
            self.open[(v.pump_id, v.track_id)] = new
            v.left = True
            self.closed.append(v)

    def _sure(self, v: _Visit) -> bool:
        reads = v.spans[-1].reads
        if len(reads) >= self.cfg.max_reads:
            return True
        return len(reads) >= 3 and vote(reads)[1] >= self.cfg.enough_conf

    def _read(self, v: _Visit, frame: np.ndarray, box: Box) -> None:
        reads = self.reader.read(frame, roi=box)
        if not reads or not reads[0].text:
            return
        r, sp = reads[0], v.spans[-1]
        sp.reads.append(r)
        if sp.best_crop is None or r.conf > sp.best_crop[0]:
            x1, y1, x2, y2 = r.box
            sp.best_crop = (r.conf, _jpg(frame[max(0, y1):y2, max(0, x1):x2]))
        if len(v.spans) > 1 and len(sp.reads) >= 3:    # the car back after a gap: is it the same plate?
            a, ca = vote([x for q in v.spans[:-1] for x in q.reads], min_conf=0.0)
            b, cb = vote(sp.reads, min_conf=0.0)
            if min(ca, cb) >= self.cfg.min_plate_conf and edit_distance(a, b) > self.cfg.plate_differs_edits:
                self._split(v, len(v.spans) - 1)

    def on_sale(self, sale: FuelSale) -> None:
        """A dispenser sale (new, or an update of one already known: fuel stopped, amount final).
        Updates match by txn_id. A controller that sends no txn_id: the update matches the sale at the same
        pump with the same start time, or the one still dispensing (a pump dispenses one sale at a time)."""
        for s in self.sales:
            if (s.txn_id == sale.txn_id if sale.txn_id else not s.txn_id and s.pump_id == sale.pump_id
                    and (s.t_start == sale.t_start or s.t_end is None)):
                paid = s.t_paid
                s.__dict__.update(sale.__dict__)          # in place: pending alerts point at this object
                s.t_paid = sale.t_paid if sale.t_paid is not None else paid
                return
        if sale.txn_id and sale.txn_id in self.settled:    # an update that arrives after the payment
            return
        self.sales.append(sale)
        # A new sale that starts after a gap in a visit: the box after the gap is the next car, not the one
        # that was there before. Split the visit at that gap.
        for v in [*self.open.values(), *self.closed]:
            if v.pump_id != sale.pump_id:
                continue
            k = next((i for i, sp in enumerate(v.spans)
                      if sp.t1 >= sale.t_start or (not v.left and i == len(v.spans) - 1)), 0)
            if k:
                self._split(v, k)

    def on_payment(self, t: float, txn_id: str | None = None, pump_id: str | None = None) -> list[Alert]:
        """A fuel sale was paid. Match by txn_id, else the oldest unpaid sale at `pump_id`."""
        unpaid = [s for s in self.sales if s.t_paid is None and not s.prepaid]
        hit = next((s for s in unpaid if txn_id and s.txn_id == txn_id), None) or \
            next((s for s in sorted(unpaid, key=lambda s: s.t_start) if pump_id and s.pump_id == pump_id), None)
        if hit is None:
            return []
        hit.t_paid = t
        if hit.txn_id:
            self.settled[hit.txn_id] = t
        return self.tick(max(t, self.now))

    # ------------------------------------------------------------------ rule
    def tick(self, t: float) -> list[Alert]:
        self.now = max(self.now, t)
        cfg, out = self.cfg, []
        for key, v in list(self.open.items()):
            if self.now - v.t_last >= cfg.leave_s:
                v.left = True
                del self.open[key]
                if v.t_last - v.t_enter >= cfg.min_dwell_s:
                    self.closed.append(v)             # short stays are dropped here, plate reads with them
        taken = {id(p.sale) for p in self.pending}
        for s in self.sales:
            if id(s) in taken or s.prepaid or s.t_paid is not None or s.t_end is None or (s.amount <= 0 and s.gallons <= 0):
                continue
            cands = [(v.overlap(s.t_start, s.t_end), v) for v in [*self.open.values(), *self.closed] if v.pump_id == s.pump_id]
            cands = sorted(((o, v) for o, v in cands if o > 0), key=lambda c: -c[0])
            if cands:
                best_o, best = cands[0]
                if best.left:
                    rivals = sum(1 for o, _ in cands[1:] if o >= 0.3 * best_o)
                    self.pending.append(_Pending(s, best, rivals))
            elif self.now - s.t_end >= cfg.no_vehicle_after_s:
                self.pending.append(_Pending(s, None))
        for p in self.pending:
            if p.alert is None and p.sale.t_paid is None and self.now >= self._deadline(p):
                p.alert = self._alert(p)
                out.append(p.alert)
            elif p.alert is not None and p.sale.t_paid is not None:
                out.append(self._retract(p))
        # Forget what is settled: paid sales, alerts too old to retract, visits nothing points at.
        self.pending = [p for p in self.pending if p.sale.t_paid is None
                        and self.now - self._deadline(p) <= cfg.late_payment_window_s]
        live = {id(p.sale) for p in self.pending}
        self.sales = [s for s in self.sales if not s.prepaid and s.t_paid is None and (id(s) in live or id(s) not in taken)]
        horizon = self.now - cfg.grace_s - cfg.no_vehicle_after_s
        held = {id(p.visit) for p in self.pending if p.alert is None}
        held |= {id(v) for v in self.closed if not v.done and any(      # its sale is still dispensing
            s.pump_id == v.pump_id and s.t_end is None and v.overlap(s.t_start, self.now) > 0 for s in self.sales)}
        self.closed = [v for v in self.closed if v.t_last >= horizon or id(v) in held]
        self.settled = {k: v for k, v in self.settled.items() if self.now - v <= cfg.late_payment_window_s}
        if self.plate_store is not None and self.now - self._purged >= cfg.purge_every_s:
            self._purged = self.now                    # an idle store still loses its expired plates
            self.plate_store.purge(now=self.wall0 + self.now)
        return out

    def _deadline(self, p: _Pending) -> float:
        if p.visit is None:
            return p.sale.t_end + self.cfg.no_vehicle_after_s
        return p.visit.t_last + self.cfg.grace_s if p.visit.left else float("inf")   # inf: the car came back

    def _alert(self, p: _Pending) -> Alert:
        s, v, cfg = p.sale, p.visit, self.cfg
        aid = "DO" + uuid.uuid4().hex[:10]
        conf = cfg.base_confidence
        what = f"{s.gallons:.1f} gal" if s.gallons else "fuel"
        reasons = [f"drive-off: {what} (${s.amount:.2f}) dispensed at {s.pump_id}, sale {s.txn_id or 'n/a'} unpaid"]
        log = [f"{s.t_start:7.1f}s fuel started at {s.pump_id}", f"{s.t_end:7.1f}s fuel stopped, ${s.amount:.2f}"]
        clip = None
        if v is None:
            conf -= 0.2
            reasons.append("no vehicle was tracked in the pump zone during the sale (camera blocked or zone wrong)")
            t_exit = s.t_end
        else:
            t_exit = v.t_last
            reasons.append(f"vehicle left the pump zone at {v.t_last:.1f}s; no payment within {cfg.grace_s:.0f}s")
            log += [f"{v.t_enter:7.1f}s vehicle {v.track_id} entered zone {v.pump_id}",
                    f"{v.t_last:7.1f}s vehicle {v.track_id} left the zone"]
            if p.rivals:
                conf -= 0.2
                reasons.append(f"identity uncertain: {p.rivals + 1} vehicles were in the zone during the sale")
            for a, b in zip(v.spans, v.spans[1:]):
                reasons.append(f"not seen in the zone from {a.t1:.1f}s to {b.t0:.1f}s, then a vehicle in the same "
                               "place: treated as the same vehicle")
            spans = v.during(s.t_start, s.t_end)       # plate and clip only from the time fuel was flowing
            reads = [r for sp in spans for r in sp.reads]
            crop = max((sp.best_crop for sp in spans if sp.best_crop), key=lambda c: c[0], default=None)
            frames = [f for sp in spans for f in sp.frames][:cfg.evidence_frames]
            text, pc = vote(reads, min_conf=0.0)
            if text and pc >= cfg.min_plate_conf:
                conf += 0.2
                if self.plate_store is not None:
                    rid = self.plate_store.add(aid, text, pc, self.wall0 + v.t_last, s.pump_id, v.camera_id,
                                               crop[1] if crop else None, now=self.wall0 + self.now)
                    reasons.append(f"plate read from {len(reads)} frames, confidence {pc:.2f}, stored as plate record {rid}")
                else:
                    reasons.append(f"plate read from {len(reads)} frames, confidence {pc:.2f}, not stored (no plate store)")
            else:
                reasons.append("plate not read" + (f" with enough confidence ({pc:.2f})" if text else ""))
            if self.out is not None and frames:
                clip = write_frames(frames, self.out / "alerts" / f"{aid}.mp4", self.fps)
        log.append(f"{self.now:7.1f}s grace period over, still unpaid -> review")
        if v is not None:                              # the plate is in the store now (or was not readable):
            v.done = True                                  # nothing of it stays in memory
            for sp in v.spans:
                sp.reads, sp.frames, sp.best_crop = [], [], None
        item = UnpaidItem(category="fuel", sku=f"fuel:{s.grade}" if s.grade else "fuel", t_pick=s.t_start,
                          zone=s.pump_id, pick_confidence=1.0, concealed=False, held_at_exit=False,
                          ambiguous_with=[], score=round(conf, 3))
        return Alert(alert_id=aid, person_id=v.track_id if v else -1, tier="review", confidence=round(conf, 3),
                     t_exit=t_exit, t_emitted=self.now, unpaid_items=[item], reasons=reasons, clip_path=clip,
                     basket=["fuel"], audit_log=log)

    def _retract(self, p: _Pending) -> Alert:
        a = p.alert
        if self.plate_store is not None:
            self.plate_store.delete_alert(a.alert_id, "late payment", now=self.wall0 + self.now)
        r = Alert(alert_id=a.alert_id + "-r", person_id=a.person_id, tier="retracted", confidence=0.0,
                  t_exit=a.t_exit, t_emitted=self.now, unpaid_items=[],
                  reasons=[f"sale {p.sale.txn_id or 'n/a'} paid at {p.sale.t_paid:.1f}s, after the alert"],
                  retracts=a.alert_id, audit_log=a.audit_log + [f"{self.now:7.1f}s late payment -> retracted"])
        return r


class IouTracker:
    """Track ids for plain vehicle boxes: greedy IoU match to the last box of each track.
    Enough for cars that stand still at a pump."""
    # ponytail: no motion model; swap in bree.track.bytetrack if vehicles cross each other fast.

    def __init__(self, min_iou: float = 0.3, max_gap_s: float = 2.0):
        self.min_iou, self.max_gap_s, self.tracks, self.next_id = min_iou, max_gap_s, {}, 1

    def update(self, t: float, boxes: list[Box]) -> list[tuple[int, Box]]:
        self.tracks = {i: (tt, b) for i, (tt, b) in self.tracks.items() if t - tt <= self.max_gap_s}
        out, free = [], dict(self.tracks)
        for b in boxes:
            best = max(free, key=lambda i: _iou(free[i][1], b), default=None)
            if best is None or _iou(free[best][1], b) < self.min_iou:
                best, self.next_id = self.next_id, self.next_id + 1
            free.pop(best, None)
            self.tracks[best] = (t, b)
            out.append((best, b))
        return out


def yolo_vehicles(model, frame: np.ndarray, conf: float = 0.3) -> list[Box]:
    """Vehicle boxes from an Ultralytics COCO model (e.g. YOLO('models/yolo26n.pt'))."""
    ids = [i for i, n in model.names.items() if n in VEHICLE_CLASSES]
    r = model.predict(frame, conf=conf, classes=ids, verbose=False)[0]
    return [tuple(int(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]
