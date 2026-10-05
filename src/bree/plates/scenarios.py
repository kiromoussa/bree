"""SCRIPTED forecourt timelines for the drive-off rule. Everything here is synthetic.

A scenario lists cars (which pump, when they stand in the zone, an optional gap where the camera
loses them, an optional track id change, an optional unreadable plate), dispenser sales (when fuel flows, when and whether it is paid) and the drive-off
alerts a correct system raises. `run()` plays it through the real DriveOffMonitor at 2 frames
per second. With `render=True` every frame is drawn (bree.plates.synth) and plates are read from
pixels; the vehicle boxes are the scripted ones (the plate bench does not test vehicle detection).
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bree.plates import synth
from bree.plates.pump import DriveOffConfig, DriveOffMonitor, FuelSale, IouTracker, PumpZone
from bree.plates.reader import edit_distance

SIZE = (1280, 720)
PLATE_W = 90
FPS = 2.0
PUMP_X = {"P1": 320.0, "P2": 960.0}
ZONES = [PumpZone(p, [(x - 300, 520), (x + 300, 520), (x + 300, 715), (x - 300, 715)], "forecourt")
         for p, x in PUMP_X.items()]
Y_IN, Y_OUT, MOVE_S = 420.0, 195.0, 1.5      # parked at the pump / just outside the zone / time to roll between
CFG = dict(grace_s=20.0, no_vehicle_after_s=40.0)     # short so a scripted timeline is 1 to 2 minutes


@dataclass
class Car:
    plate: str
    pump: str
    t_in: float
    t_out: float
    hidden: tuple[float, float] | None = None      # the camera does not see the car in this interval
    new_id_at: float | None = None                 # the tracker gives the car a new id from this time on
    blank_plate: bool = False                      # the plate can not be read (covered, missing)


@dataclass
class Sale:
    pump: str
    t0: float
    t1: float
    amount: float = 41.37
    prepaid: bool = False
    paid_at: float | None = None
    reported_at: float | None = None               # when the controller tells us (default: t0, then t1)


@dataclass
class Scenario:
    name: str
    what: str
    cars: list[Car]
    sales: list[Sale]
    expect: list[tuple[str, str | None]] = field(default_factory=list)   # (pump, plate or None) per alert
    retractions: int = 0
    condition: str = "day"
    end: float = 0.0


SCENARIOS = [
    Scenario("pay_inside", "fuels, pays inside, then leaves", [Car("ABC 1234", "P1", 5, 60)],
             [Sale("P1", 10, 40, paid_at=52)]),
    Scenario("prepaid", "pays at the pump, fuels, leaves", [Car("KLM 482", "P1", 5, 50)],
             [Sale("P1", 12, 40, prepaid=True)]),
    Scenario("drive_off", "fuels and leaves without paying", [Car("7XYZ 123", "P1", 5, 45)],
             [Sale("P1", 10, 40)], [("P1", "7XYZ123")]),
    Scenario("moves_car_then_pays", "leaves the pump, pays 12 s later (inside the grace period)",
             [Car("GHT 9021", "P2", 5, 45)], [Sale("P2", 10, 40, paid_at=57)]),
    Scenario("late_payment", "leaves, pays after the grace period: alert, then retraction",
             [Car("BRE 5521", "P1", 5, 45)], [Sale("P1", 10, 40, paid_at=80)], [("P1", "BRE5521")], retractions=1),
    Scenario("drive_through", "stops at a pump for 12 s, no fuel", [Car("QQA 7710", "P2", 5, 17)], []),
    Scenario("two_pumps", "P1 pays, P2 drives off at the same time",
             [Car("PAY 1001", "P1", 5, 60), Car("RUN 2002", "P2", 8, 50)],
             [Sale("P1", 10, 40, paid_at=50), Sale("P2", 12, 44)], [("P2", "RUN2002")]),
    Scenario("same_pump_two_cars", "first car pays, the next car at the same pump drives off",
             [Car("AAA 1111", "P1", 5, 45), Car("ZZZ 9999", "P1", 60, 100)],
             [Sale("P1", 10, 35, paid_at=40), Sale("P1", 65, 95)], [("P1", "ZZZ9999")]),
    Scenario("sale_reported_late", "the controller reports the unpaid sale 10 s after the car left",
             [Car("LTE 4040", "P2", 5, 45)], [Sale("P2", 10, 40, reported_at=55)], [("P2", "LTE4040")]),
    Scenario("camera_blocked_then_pays", "the car is hidden for 15 s after fuelling, pays later, then leaves",
             [Car("HID 3030", "P1", 5, 90, hidden=(42, 57))], [Sale("P1", 10, 40, paid_at=85)]),
    Scenario("camera_blocked_past_grace", "hidden for longer than the grace period, then pays: alert, then retraction "
             "(the known limit: the rule can not tell a hidden car from a gone car)",
             [Car("HID 4040", "P1", 5, 90, hidden=(42, 72))], [Sale("P1", 10, 40, paid_at=85)], [("P1", "HID4040")],
             retractions=1),
    Scenario("blocked_mid_fuelling_then_pays", "hidden for 6 s while fuel is flowing, pays inside, then leaves",
             [Car("MID 5050", "P1", 5, 90, hidden=(32, 38))], [Sale("P1", 10, 40, paid_at=75)]),
    Scenario("blocked_mid_fuelling_drive_off", "hidden for 10 s while fuel is flowing, then leaves without paying",
             [Car("MDO 6161", "P2", 5, 45, hidden=(20, 30))], [Sale("P2", 10, 40)], [("P2", "MDO6161")]),
    Scenario("track_id_changes_then_pays", "the tracker gives the car a new id while fuel is flowing; it pays, then leaves",
             [Car("IDS 7272", "P1", 5, 80, new_id_at=30)], [Sale("P1", 10, 40, paid_at=70)]),
    Scenario("next_car_takes_the_spot", "drive-off with an unreadable plate; 10 s later the next car takes the same "
             "spot, fuels, pays and leaves: one alert, and the second car's plate must not be stored",
             [Car("THF 0000", "P1", 5, 40, blank_plate=True), Car("PAY 1001", "P1", 50, 110)],
             [Sale("P1", 10, 35), Sale("P1", 55, 85, paid_at=95)], [("P1", None)]),
    Scenario("no_vehicle_seen", "unpaid sale, the camera never saw a vehicle", [], [Sale("P1", 10, 40)], [("P1", None)]),
    Scenario("night_drive_off", "drive-off at night", [Car("NGT 6060", "P1", 5, 45)], [Sale("P1", 10, 40)],
             [("P1", "NGT6060")], condition="night"),
    Scenario("night_pays", "pays inside at night", [Car("NPY 7070", "P2", 5, 60)], [Sale("P2", 10, 40, paid_at=50)],
             condition="night"),
]


def _car_y(c: Car, t: float) -> float | None:
    """Vertical centre of the car at t, None when it is not in the frame."""
    if c.t_in - MOVE_S <= t < c.t_in:
        return Y_OUT + (Y_IN - Y_OUT) * (t - (c.t_in - MOVE_S)) / MOVE_S
    if c.t_in <= t <= c.t_out:
        return Y_IN
    if c.t_out < t <= c.t_out + MOVE_S:
        return Y_IN + (Y_OUT - Y_IN) * (t - c.t_out) / MOVE_S
    return None


def run(sc: Scenario, reader=None, render: bool = False, plate_store=None, out_dir: str | Path | None = None,
        seed: int = 0, config: DriveOffConfig | None = None) -> dict:
    """Play one scenario. Returns {"alerts", "retractions", "ok", "plates": [(truth, read)], "frames"}."""
    cfg = config or DriveOffConfig(**CFG)
    mon = DriveOffMonitor(ZONES, reader=reader if render else None, config=cfg, plate_store=plate_store,
                          out_dir=out_dir, fps=FPS, wall0=1_800_000_000.0)
    tracker = IouTracker()
    rng = np.random.default_rng(seed)
    end = sc.end or max([c.t_out for c in sc.cars] + [s.t1 for s in sc.sales]
                        + [s.paid_at or 0 for s in sc.sales]) + max(cfg.grace_s, cfg.no_vehicle_after_s) + 10
    msgs = []                                         # (t, kind, sale) controller messages in time order
    for s in sc.sales:
        fs = FuelSale(s.pump, s.t0, None, 0.0, 0.0, txn_id=f"T{len(msgs)}", grade="regular", prepaid=s.prepaid)
        if s.reported_at is None:
            msgs.append((s.t0, "sale", fs))
        done = FuelSale(s.pump, s.t0, s.t1, s.amount, s.amount / 3.49, fs.txn_id, "regular", s.prepaid)
        msgs.append((max(s.t1, s.reported_at or 0), "sale", done))
        if s.paid_at is not None:
            msgs.append((s.paid_at, "pay", fs))
    msgs.sort(key=lambda m: m[0])
    alerts, n_frames = [], 0
    for i in range(int(end * FPS) + 1):
        t = i / FPS
        while msgs and msgs[0][0] <= t:
            _, kind, fs = msgs.pop(0)
            if kind == "sale":
                mon.on_sale(fs)
            else:
                alerts += mon.on_payment(t, txn_id=fs.txn_id)
        seen = [(c, y) for c in sc.cars if (y := _car_y(c, t)) is not None
                and not (c.hidden and c.hidden[0] <= t < c.hidden[1])]
        frame, boxes = None, []
        if render and seen:
            frame = synth.forecourt(np.random.default_rng(seed), SIZE)
        for c, y in seen:
            car_rng = np.random.default_rng(zlib.crc32(c.plate.encode()))    # one look per car, every frame
            if frame is not None:
                box, pl = synth.draw_car(frame, car_rng, PUMP_X[c.pump], y, PLATE_W, c.plate,
                                         colour=tuple(int(v) for v in car_rng.integers(30, 200, 3)))
                if c.blank_plate:
                    frame[pl[1]:pl[3], pl[0]:pl[2]] = 90
            else:
                cw, ch = synth.car_box(PLATE_W)
                box = (int(PUMP_X[c.pump] - cw / 2), int(y - ch / 2), int(PUMP_X[c.pump] + cw / 2), int(y + ch / 2))
            boxes.append(box)
        if frame is not None:
            frame = synth.degrade(frame, rng, plate_w=PLATE_W, **synth.CONDITIONS[sc.condition])
            n_frames += 1
        tracks = [(tid + 1000 * (c.new_id_at is not None and t >= c.new_id_at), b)
                  for (tid, b), (c, _) in zip(tracker.update(t, boxes), seen)]
        alerts += mon.update(t, tracks, frame, camera_id="forecourt")
    raised = [a for a in alerts if not a.retracts]
    retracted = [a for a in alerts if a.retracts]
    got = sorted(a.unpaid_items[0].zone for a in raised)
    ok = got == sorted(p for p, _ in sc.expect) and len(retracted) == sc.retractions \
        and all(a.tier == "review" for a in raised)
    plates = []
    for a in raised:
        truth = next((pl for p, pl in sc.expect if p == a.unpaid_items[0].zone), None)
        read = None
        if plate_store is not None:
            for rid in plate_store.for_alert(a.alert_id):
                read = plate_store.get(rid, "scenario runner", "score the synthetic bench")["plate_text"]
        if truth is not None:
            plates.append((truth, read, None if read is None else edit_distance(truth, read)))
        elif read is not None:                         # a plate was stored where none could be read: wrong
            plates.append(("(none)", read, len(read)))
            ok = False
    return {"name": sc.name, "alerts": raised, "retractions": retracted, "ok": ok, "plates": plates, "frames": n_frames}
