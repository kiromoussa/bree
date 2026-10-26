"""After-hours intrusion, the store clock, and loitering."""
from __future__ import annotations

import numpy as np

from bree.events.observations import FrameObs, PersonObs
from bree.events.zones import Zone
from bree.theft.hours import AfterHoursMonitor, HoursConfig, LoiterMonitor, StoreClock


def person(i, x=500, y=600, conf=0.8):
    return PersonObs(i, (x - 40, y - 200, x + 40, y), conf)


def test_clock_from_segment_name_and_offset():
    c = StoreClock.from_segment_name("trumanrd/ch07/20261025_182851.mp4", clock_offset_s=-2 * 3600)
    assert c.known and abs(c.local(0) - (16 * 3600 + 28 * 60 + 51)) < 1
    assert abs(c.local(600) - (16 * 3600 + 38 * 60 + 51)) < 1
    assert not StoreClock.from_segment_name("20261025_182851.mp4").known


def test_unknown_clock_raises_nothing():
    m = AfterHoursMonitor(StoreClock.from_segment_name("20261025_030000.mp4"))
    assert m.update(FrameObs(0, 10.0, [person(1)])) == []


def test_after_hours_presence_is_one_incident_with_cooldown():
    clock = StoreClock.from_segment_name("20261025_030000.mp4", clock_offset_s=0)
    m = AfterHoursMonitor(clock, HoursConfig(open="06:00", close="23:00"))
    out = []
    for k in range(600):
        out += m.update(FrameObs(k, k / 2, [person(1)]))
    assert len(out) == 1 and out[0].kind == "after_hours" and out[0].severity == "high"
    assert out[0].details["state"] == "closed"


def test_open_hours_and_grace():
    m = AfterHoursMonitor(StoreClock.from_segment_name("20261025_120000.mp4", clock_offset_s=0))
    assert m.update(FrameObs(0, 0.0, [person(1)])) == [] and m.state(0.0) == "open"
    assert m.state(-12 * 3600 + 5.6 * 3600) == "grace"         # 05:36 local: inside the 30 min before opening
    g = AfterHoursMonitor(StoreClock.from_segment_name("20261025_230500.mp4", clock_offset_s=0))
    out = []
    for k in range(40):
        out += g.update(FrameObs(k, k, [person(1)]))
    assert out and out[0].severity == "low" and g.state(0) == "grace"     # 23:05: inside the 15 min after closing
    late = AfterHoursMonitor(StoreClock.from_segment_name("20261025_233000.mp4", clock_offset_s=0))
    assert late.state(0) == "closed" and late.update(FrameObs(0, 0.0, [person(1)])) == []
    assert late.update(FrameObs(1, 5.0, [person(1)]))[0].severity == "high"


def test_overnight_store_hours():
    m = AfterHoursMonitor(StoreClock.from_segment_name("20261025_010000.mp4", clock_offset_s=0), HoursConfig(open="05:00", close="02:00"))
    assert m.state(0) == "open"
    m2 = AfterHoursMonitor(StoreClock.from_segment_name("20261025_030000.mp4", clock_offset_s=0), HoursConfig(open="05:00", close="02:00"))
    assert m2.state(0) == "closed"


def test_loiter_by_role_and_exemption():
    m = LoiterMonitor("door")
    out = []
    for k in range(400):                      # 400 s standing near the door
        out += m.update(FrameObs(k, float(k), [person(1, 500 + (k % 3), 600)]))
    assert len(out) == 1 and out[0].kind == "loiter" and out[0].details["dwell_s"] >= 180
    assert LoiterMonitor("counter").update(FrameObs(0, 0.0, [person(1)])) == []
    ex = LoiterMonitor("door", exempt=[Zone("reg", "register", np.array([[0, 0], [2000, 0], [2000, 2000], [0, 2000]], float))])
    assert not any(ex.update(FrameObs(k, float(k), [person(1)])) for k in range(400))


def test_walking_person_does_not_loiter():
    m = LoiterMonitor("outdoor")
    assert not any(m.update(FrameObs(k, float(k), [person(1, 100 + 10 * k, 600)])) for k in range(400))
