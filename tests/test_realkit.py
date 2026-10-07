"""scripts/realkit: the real footage kit. Session file -> layout, clicked marks -> camera, act log -> truth,
and the scorer of the simulator benchmark run on a hand-made clip folder. No model is loaded here."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "realkit"))
import make_truth as mt      # noqa: E402
import px_per_mm as ppm      # noqa: E402
import run_real as rr      # noqa: E402
import session as ses      # noqa: E402

cv2 = pytest.importorskip("cv2")

SESSION = {
    "name": "t", "fps": 10, "lead_s": 2.0, "sync_clock_s": 5.0,
    "shelf": {"width_in": 36, "depth_in": 12, "boards_in": [4, 18, 32], "top_in": 48,
              "rows": [["coke", "coke", "coke_zero", "sprite"], {"from_in": 0, "to_in": 18, "slots": ["chips", None, "chips_bbq"]}]},
    "products": {"coke": {"size_in": [2.6, 4.8, 2.6]}, "coke_zero": {"size_in": [2.6, 4.8, 2.6]}, "sprite": {"size_in": [2.6, 4.8, 2.6]},
                 "chips": {"size_in": [5, 7, 2]}, "chips_bbq": {"size_in": [5, 7, 2]}},
    "door_spot_in": [-72, 36], "register_spot_in": [108, 36],
    "floor_marks_in": {"A": [0, 24], "B": [36, 24], "C": [36, 60], "D": [0, 60]},
    "cameras": {"railA": {"file": "a.mp4", "kind": "shelf", "hfov_deg": 50}},
}
LOG = [
    {"act": "1", "clap_time": "0:15", "actor": "maya", "product": "coke", "slot": "S1-1", "outcome": "paid"},
    {"act": "2", "clap_time": "0:45", "actor": "sam", "product": "coke_zero", "slot": "S1-3", "outcome": "concealed_pocket", "second_time": "0:52"},
    {"act": "3", "clap_time": "1:15", "actor": "maya", "product": "chips", "slot": "S2-1", "outcome": "put_back_wrong", "to_slot": "S2-3"},
    {"act": "4", "clap_time": "1:45", "actor": "sam", "product": "", "slot": "", "outcome": "walk_past"},
    {"act": "5", "clap_time": "2:15", "actor": "maya", "product": "sprite", "slot": "S1-4", "outcome": "shift"},
    {"act": "6", "clap_time": "2:45", "actor": "staff", "product": "coke", "slot": "S1-1", "outcome": "restock"},
    {"act": "7", "clap_time": "3:15", "actor": "maya", "product": "coke", "slot": "S1-2", "outcome": "paid", "light": "dim"},
    {"act": "7", "clap_time": "3:15", "actor": "sam", "product": "sprite", "slot": "S1-4", "outcome": "concealed_bag", "light": "dim"},
]


def camera(res=(960, 720), hfov=50.0):
    """A camera 1.5 m out from the shelf, 1 m to the left of its middle, 0.9 m up, looking at the shelf."""
    from bree.calib.camera import Camera
    C, target = np.array([-1.0, 0.9, 1.5]), np.array([0.0, 0.6, 0.0])
    fwd = (target - C) / np.linalg.norm(target - C)
    right = np.cross(fwd, [0, 1.0, 0])
    right /= np.linalg.norm(right)
    f = res[0] / 2 / np.tan(np.radians(hfov) / 2)
    return Camera(f, res[0] / 2, res[1] / 2, np.stack([right, np.cross(fwd, right), fwd]), C, res)


def test_px_per_mm():
    assert ppm.px_per_mm((0, 0), (30, 40), 100) == 0.5
    low, mid, high = ppm.verdict(0.2), ppm.verdict(0.6), ppm.verdict(1.6)
    assert "FAIL" in low[1] and "PASS" in mid[1] and "FAIL" in mid[2] and "PASS" in high[2]
    assert ppm.main(["x.jpg", "--mm", "66", "--points", "0,0,66,0"]) == 0
    with pytest.raises(ValueError):
        ppm.px_per_mm((0, 0), (1, 1), 0)


def test_clock_and_layout():
    assert ses.parse_clock("1:02:03") == 3723 and ses.parse_clock("12:34.5") == 754.5 and ses.parse_clock(7) == 7.0
    lay = ses.build_layout(SESSION)
    ids = [s["id"] for s in lay["slots"]]
    assert ids == ["S1-1", "S1-2", "S1-3", "S1-4", "S2-1", "S2-3"]          # the empty place makes no slot
    s11 = lay["slots"][0]
    assert s11["face"][2] == 0 and s11["normal"] == [0, 0, 1] and abs(s11["face"][0] - (-0.4572 + 0.1143)) < 1e-3
    assert abs(s11["position"][1] - (4 * 0.0254 + 4.8 * 0.0254 / 2)) < 1e-3
    assert {f["type"] for f in lay["fixtures"]} == {"gondola", "door", "counter"}
    from bree.calib.slots import SlotMap          # the pipeline's own reading of the layout: every slot faces the aisle
    sm = SlotMap(lay)
    assert len(sm) == 6 and (sm.normal[:, 2] == 1).all() and np.allclose(sm.front[:, 2], 0)
    from bree.track.floor import counter_zones          # the pay strip is between the counter and the shelf, around the register spot
    _, pay = counter_zones(lay)
    rx, _, rz = lay["poi"]["register"]
    assert pay[0] <= rx <= pay[1] and pay[2] <= rz <= pay[3]


def test_marks_give_the_camera_back():
    cam = camera()
    pts = ses.mark_points(SESSION)
    px, _ = cam.project(np.array([[x, h, z] for _, _, (x, z, h) in pts]))
    entry = {"resolution": [960, 720], "marks": {n: [float(u), float(v)] for (n, _, _), (u, v) in zip(pts, px)}}
    got, rep = rr.calibrate_camera(SESSION, "railA", entry)
    assert rep["rms_px"] < 0.5 and np.linalg.norm(got.C - cam.C) < 0.02
    no_lens = {**SESSION, "cameras": {"railA": {"file": "a.mp4", "kind": "shelf"}}}      # lens fitted from the marks
    got, rep = rr.calibrate_camera(no_lens, "railA", entry)
    assert rep["method"] == "pnp+focal" and abs(got.hfov_deg - 50) < 1.0


def test_make_truth():
    plan = {s["id"]: s["skuId"] for s in ses.build_layout(SESSION)["slots"]}
    t = mt.make_truth(LOG, sync_clock_s=5.0, lead_s=2.0, planogram=plan)
    ev = {e["shopper"]: e for e in t["events"]}
    assert ev["A01-maya"]["t"] == 12.0 and ev["A01-maya"]["outcome"] == "paid" and ev["A01-maya"]["tPay"] == 27.0
    assert ev["A02-sam"]["outcome"] == "concealed" and ev["A02-sam"]["tConceal"] == 47.0 and ev["A02-sam"]["where"] == "pocket"
    assert ev["A03-maya"]["outcome"] == "put_back" and ev["A03-maya"]["putBackSlot"] == "S2-3"
    assert [a["kind"] for a in t["acts"]] == ["touch", "staff_put"]
    who = {p["shopper"]: p for p in t["shoppers"]}
    assert len(who) == 8 and who["A04-sam"]["thief"] is False and who["A02-sam"]["thief"] and who["A06-staff"]["role"] == "staff"
    assert who["A07-maya"]["group"] == "A07" == who["A07-sam"]["group"] and who["A01-maya"]["group"] is None
    assert [r["items"] for r in t["register"]] == [[{"sku": "coke", "qty": 1}], [{"sku": "coke", "qty": 1}]] and t["register"][0]["t"] == 30.0
    assert not t["warnings"] and t["faults"]["light"] == "day"
    with pytest.raises(ValueError, match="not on the shelf"):
        mt.make_truth([{**LOG[0], "slot": "S9-9"}], planogram=plan)
    with pytest.raises(ValueError, match="outcome"):
        mt.make_truth([{**LOG[0], "outcome": "stolen"}])
    assert mt.make_truth([{**LOG[0], "product": "sprite"}], planogram=plan)["warnings"]


def test_log_file_round_trip(tmp_path):
    p = tmp_path / "acts.csv"
    p.write_text("act,clap_time,actor,product,slot,outcome,to_slot,second_time,light,notes\n# a comment row\n1,0:15,maya,coke,S1-1,paid,,,,\n2,0:45,sam,coke_zero,S1-3,concealed_pocket,,0:52,,\n")
    t = mt.make_truth(mt.read_log(p), 5.0)
    d = mt.write_truth(t, tmp_path)
    assert len(t["events"]) == 2 and (d / "frames.jsonl").read_text() == "" and len((tmp_path / "register.jsonl").read_text().splitlines()) == 1


def test_ffmpeg_command_and_pts(tmp_path):
    cmd = rr.ffmpeg_cmd("cam.h264", "out.mp4", sync_s=12.4, fps=10, rotate=90, fps_in=15)
    assert cmd[cmd.index("-i") - 2:cmd.index("-i")] == ["-r", "15"] and cmd[cmd.index("-ss") + 1] == "12.400" and cmd.index("-ss") > cmd.index("-i")
    assert cmd[cmd.index("-vf") + 1].startswith("transpose=1,fps=10,scale=")
    assert "-r" not in rr.ffmpeg_cmd("phone.mov", "out.mp4")
    p = tmp_path / "cam.pts"
    p.write_text("# timecode format v2\n" + "\n".join(str(i * 100.0 + (300 if i > 50 else 0)) for i in range(101)) + "\n")
    assert rr.pts_drift(p, 10) == 0.3


def _video(path: Path, cam, lay: dict, n: int = 50, gone: tuple[str, int] | None = None) -> None:
    """A grey shelf with a coloured block at every slot; the block of slot `gone[0]` vanishes at frame gone[1]."""
    skus = {x["id"]: x["size"] for x in lay["skus"]}
    w, h = cam.resolution
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    rng = np.random.default_rng(0)
    colours = {s["id"]: [int(v) for v in rng.integers(40, 255, 3)] for s in lay["slots"]}
    for f in range(n):
        im = np.full((h, w, 3), 120, np.uint8)
        for s in lay["slots"]:
            if gone and s["id"] == gone[0] and f >= gone[1]:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in rr.slot_box(cam, s, skus))
            cv2.rectangle(im, (x0, y0), (x1, y1), colours[s["id"]], -1)
        vw.write(im)
    vw.release()


def test_shelf_comparison_reads_a_take_at_the_right_slot(tmp_path):
    """The layout and camera this kit builds, through the pipeline's own shelf comparison (no detector)."""
    from bree.shelf.events import camera_events
    lay, cam = ses.build_layout(SESSION), camera()
    _video(tmp_path / "railA.mp4", cam, lay, gone=("S1-3", 20))
    ev = camera_events(tmp_path / "railA.mp4", "railA", cam, lay, 10.0, detector=None)
    assert [(e["kind"], e["slot_id"]) for e in ev] == [("take", "S1-3")] and abs(ev[0]["t"] - 2.0) < 1.5


def test_scorecard_on_a_hand_made_run(tmp_path, monkeypatch):
    """The benchmark scorer on a real-kit clip folder: one take found at the right slot, one theft missed, one flagged."""
    work, lay, cam = tmp_path / "work", ses.build_layout(SESSION), camera()
    clip, pipe = work / "clip", work / "run" / "pipeline"
    clip.mkdir(parents=True)
    pipe.mkdir(parents=True)
    _video(clip / "railA.mp4", cam, lay, n=30)
    px, _ = cam.project(np.array([[x, h, z] for _, _, (x, z, h) in ses.mark_points(SESSION)]))
    (work / "marks.json").write_text(json.dumps({"railA": {"resolution": [960, 720], "marks": {n: [float(u), float(v)] for (n, _, _), (u, v) in zip(ses.mark_points(SESSION), px)}}}))
    log = tmp_path / "acts.csv"
    cols = ["act", "clap_time", "actor", "product", "slot", "outcome", "to_slot", "second_time", "light"]
    log.write_text(",".join(cols) + "\n" + "".join(",".join(r.get(c, "") for c in cols) + "\n" for r in LOG))
    s = {**SESSION, "_dir": str(tmp_path)}
    clip, lay, cams = rr.build_clip(s, work, str(log))
    assert json.loads((clip / "clip.json").read_text())["frames"] == 30 and (clip / "truth" / "events.jsonl").exists()
    assert rr.overlays(clip, lay, cams, work)[0].exists()
    pick = lambda t, slot, sku, pid: {"type": "pick", "t": t, "person_id": pid, "sku": sku, "item": sku, "zone": "SHELF", "confidence": 0.9, "candidates": [],      # noqa: E731
                                      "meta": {"slot": {"id": slot, "fixture": "SHELF", "sku": sku}, "shelf": {"camera_id": "railA"}}}
    events = [pick(13.0, "S1-1", "coke", 1), pick(42.5, "S1-2", "coke", 2), pick(100.0, "S1-4", "sprite", None),      # act 1 right; act 2 found at the wrong slot; no act at 100 s
              pick(192.5, "S1-2", "coke", 5), pick(193.0, "S1-4", "sprite", 6)]                                       # act 7: two people, two identities
    alert = {"alert_id": "a1", "tier": "review", "person_id": 2, "t_emitted": 60.0, "confidence": 0.5, "group": [],
             "unpaid_items": [{"sku": "coke", "category": "coke", "t_pick": 42.5}]}
    (pipe / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (pipe / "alerts.jsonl").write_text(json.dumps(alert) + "\n")
    (pipe / "person_ids.jsonl").write_text("")
    (pipe / "store_shelf_events.jsonl").write_text(json.dumps({"kind": "take", "t": 73.0, "slot_id": "S2-1", "sku_id": None, "camera_id": "railA", "person_id": None, "why": "nobody in reach"}) + "\n"
                                                   + json.dumps({"kind": "take", "t": 13.0, "slot_id": "S1-1", "sku_id": "coke", "camera_id": "railA", "person_id": 1}) + "\n")      # act 3: read by the shelf camera, no person
    (pipe / "frames_railA.jsonl").write_text("")
    res = rr.scorecard(s, clip, work / "run", work, lay, cams, tol_s=5.0)
    h = res["headline"]
    assert (h["takes"], h["takes_found"], h["right_slot"], h["right_product"]) == (5, 5, 4, 4)          # act 3 counts through its shelf event without a person
    assert h["takes_tied_to_a_tracked_person"] == 0          # no person boxes in the frame logs of this hand-made run
    assert (h["thefts"], h["thefts_flagged"], h["honest_visits"], h["honest_visits_flagged"]) == (2, 1, 5, 0)
    assert (h["two_person_acts_found"], h["two_person_acts_given_to_two_identities"]) == (1, 1)
    assert res["headline_dim"]["takes"] == 2 and res["headline_bright"]["takes"] == 3
    what = sorted((m["act"], m["what"]) for m in res["missed"])
    assert ("03", "put_back_missed") in what and ("02", "wrong_slot") in what and ("07", "theft_not_flagged") in what and ("none1", "take_with_no_act") in what
    md = (work / "scorecard.md").read_text()
    assert "REAL footage" in md and "SIMULATED" not in md and "attached): 5 of 5" in md
    sheets = list((work / "missed").glob("*.jpg"))
    assert len(sheets) == len(res["missed"]) and cv2.imread(str(sheets[0])) is not None
