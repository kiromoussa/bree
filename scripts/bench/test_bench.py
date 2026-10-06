"""Scorer self-check on a hand-made clip (no video, no simulator): one stolen item, one paid item, one put back.
    .venv/bin/python -m pytest scripts/bench/test_bench.py -q
"""
import json
from pathlib import Path

from bree.sim.bench import STAGES, aggregate, manifest, markdown, public_view, score_clip


def _write(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) if isinstance(rows, list) else json.dumps(rows))


def _clip(tmp: Path) -> Path:
    clip = tmp / "clip"
    cam = {"id": "RAIL", "kind": "shelf", "resolution": [1000, 1000], "fx": 900, "fy": 900, "cx": 500, "cy": 500, "position": [0, 2, 0], "R": [[1, 0, 0], [0, -1, 0], [0, 0, -1]]}
    _write(clip / "clip.json", {"seed": 1, "fps": 10, "frames": 300, "sim_seconds": 30.0, "cameras": ["RAIL"]})
    _write(clip / "calibration.json", {"cameras": [cam]})
    _write(clip / "layout.json", {"cameras": [], "slots": [{"id": f"S{k}", "skuId": "abcde"[k - 1], "fixtureId": "G1"} for k in range(1, 6)]})
    _write(clip / "register.jsonl", [{"t": 16.0, "terminal": "pos_1", "txn_id": "T1", "items": [{"sku": "b", "qty": 1}]}])
    ev = lambda t, who, sku, slot, outcome, **k: {"t": t, "shopper": who, "skuId": sku, "slotId": slot, "fixtureId": "G1", "zone": "gondola", "outcome": outcome,  # noqa: E731
                                                  "tConceal": None, "tPay": None, "tPutBack": None, "cameras": [{"id": "RAIL", "px": 30}], **k}
    _write(clip / "truth" / "events.jsonl", [ev(5.0, "P001", "a", "S1", "concealed", tConceal=7.0), ev(6.0, "P002", "b", "S2", "paid", tPay=15.0),
                                             ev(8.0, "P002", "c", "S3", "put_back", tPutBack=10.0)])
    (clip / "truth" / "shoppers.json").write_text(json.dumps([{"shopper": "P001", "thief": True}, {"shopper": "P002", "thief": False}]))
    box = {"P001": [100, 100, 300, 900], "P002": [600, 100, 800, 900]}
    held = {"P001": [("a", "S1", 50, 60)], "P002": [("b", "S2", 60, 200), ("c", "S3", 80, 100)]}
    frames = []
    for f in range(300):
        persons = [{"shopper": n, "bbox": b, "vis_px": 9000, "hand": [b[2] - 10, 400]} for n, b in box.items()]
        items = [{"sku": s, "kind": "hand", "bbox": [box[n][2] - 30 - 60 * k, 380, box[n][2] - 60 * k, 430], "vis_px": 900, "shopper": n, "slot": slot}
                 for n, hs in held.items() for k, (s, slot, a, b) in enumerate(hs) if a <= f < b]
        frames.append({"frame": f, "t": f / 10, "camera": "RAIL", "persons": persons, "items": items})
    _write(clip / "truth" / "frames.jsonl", frames)
    return clip


def _run(tmp: Path, perfect: bool) -> Path:
    run = tmp / ("good" if perfect else "blind")
    box = {7: [100, 100, 300, 900], 9: [600, 100, 800, 900]}
    rows = [{"frame": f, "t": f / 10, "persons": [{"id": i, "gid": i, "bbox": b, "kpts": [[0, 0, 0]] * 9 + [[b[2] - 12, 402, 0.9], [0, 0, 0]]} for i, b in box.items()],
             "products": [{"bbox": [270, 380, 300, 430], "cat": "a"}] if perfect and 50 <= f < 60 else []} for f in range(300)]
    _write(run / "pipeline" / "frames_RAIL.jsonl", rows)
    E = lambda typ, t, pid, **k: {"type": typ, "t": t, "person_id": pid, "item": None, "sku": None, "zone": None, "meta": {}, **k}  # noqa: E731
    events = [E("pick", 5.2, 7, sku="a", zone="G1", meta={"slot": {"id": "S1", "sku": "a", "fixture": "G1"}}), E("conceal", 7.1, 7),
              E("pick", 6.1, 9, sku="b", zone="G1", meta={"slot": {"id": "S2", "sku": "b", "fixture": "G1"}}), E("pay", 14.0, 9),
              E("pick", 8.3, 9, sku="x", zone="G1"), E("put_back", 10.2, 9)] if perfect else []
    _write(run / "pipeline" / "events.jsonl", events)
    alert = {"alert_id": "A1", "person_id": 7, "tier": "alert", "confidence": 0.9, "t_exit": 20.0, "t_emitted": 20.5, "group": [],
             "unpaid_items": [{"category": "a", "sku": "a", "t_pick": 5.2, "zone": "G1"}]}
    _write(run / "pipeline" / "alerts.jsonl", [alert] if perfect else [])
    return run


def test_scorer(tmp_path):
    clip = _clip(tmp_path)
    good = score_clip(clip, _run(tmp_path, True))
    stolen, paid, back = good["picks"]
    assert stolen["lost_at"] is None and all(stolen["stages"].values())
    assert paid["lost_at"] is None
    assert back["lost_at"] == "right_slot" and not back["stages"]["right_sku"] and back["stages"]["conceal_or_pay_classified"]
    s = aggregate([good])["scorecard"]
    assert (s["theft_recall_alert"], s["alert_precision"], s["false_alerts_on_honest_shoppers"], s["pick_recall"], s["pick_precision"]) == (1.0, 1.0, 0, 1.0, 1.0)
    assert s["time_to_alert_s"]["median"] == 13.5 and s["store_wide_ids_per_shopper"] == 1.0 and s["right_sku_of_paired_picks"] == 0.667
    blind = score_clip(clip, _run(tmp_path, False))
    assert [p["lost_at"] for p in blind["picks"]] == ["shelf_event_emitted"] * 3      # people and wrists are seen, nothing is emitted
    f = aggregate([blind])
    assert f["scorecard"]["theft_recall_alert"] == 0.0 and f["funnel"]["through_every_stage"] == 0
    assert sum(r["lost_here"] for r in f["funnel"]["stages"]) == 3 and [r["stage"] for r in f["funnel"]["stages"]] == list(STAGES)


def test_public_view_hides_truth(tmp_path):
    clip = _clip(tmp_path)
    pub = public_view(clip, tmp_path / "work")
    assert sorted(p.name for p in pub.iterdir()) == ["calibration.json", "clip.json", "layout.json", "register.jsonl"]
    assert json.loads((pub / "layout.json").read_text()) == json.loads((clip / "layout.json").read_text())


def test_generator_2_truth(tmp_path):
    """A staff take that ends on a review, a shifted item read as a pick, a staff put: counted apart from the shoppers' picks."""
    clip, run = _clip(tmp_path), _run(tmp_path, True)
    _write(clip / "clip.json", json.loads((clip / "clip.json").read_text()) | {"generator": 2})
    act = lambda kind, t, slot, **k: {"kind": kind, "t": t, "shopper": "STAFF1", "skuId": "d", "slotId": slot, "fixtureId": "G1", "zone": "gondola", "cameras": [], **k}  # noqa: E731
    _write(clip / "truth" / "acts.jsonl", [act("staff_take", 12.0, "S4", outcome="carried_off"), act("touch", 15.0, "S5") | {"shopper": "P002"}, act("staff_put", 18.0, "S4")])
    _write(clip / "truth" / "faults.json", {"features": {"staff": True, "shift": True, "night": False}, "light": "day", "camera_faults": {"bump": [], "block": []}, "dropped_scan": None})
    _write(clip / "truth" / "planogram.json", {"exact": {}, "slots_where_the_item_differs": ["S2"]})
    (clip / "truth" / "shoppers.json").write_text(json.dumps([{"shopper": "P001", "thief": True}, {"shopper": "P002", "thief": False}, {"shopper": "STAFF1", "thief": False, "role": "staff"}]))
    ev = [json.loads(x) for x in (run / "pipeline" / "events.jsonl").read_text().splitlines()]
    E = lambda typ, t, pid, **k: {"type": typ, "t": t, "person_id": pid, "item": None, "sku": None, "zone": "G1", "meta": {}, **k}  # noqa: E731
    _write(run / "pipeline" / "events.jsonl", ev + [E("pick", 12.1, 11, sku="d"), E("pick", 15.2, 9, sku="e"), E("put_back", 18.1, 11)])
    al = [json.loads(x) for x in (run / "pipeline" / "alerts.jsonl").read_text().splitlines()]
    _write(run / "pipeline" / "alerts.jsonl", al + [{"alert_id": "A2", "person_id": 11, "tier": "review", "confidence": 0.5, "t_exit": 25.0, "t_emitted": 25.5, "group": [],
                                                      "unpaid_items": [{"category": "d", "sku": "d", "t_pick": 12.1, "zone": "G1"}]}])
    c = score_clip(clip, run)
    res = aggregate([c])
    s = res["scorecard"]
    assert (s["staff_takes"], s["staff_takes_with_a_pick_event"], s["staff_takes_listed_unpaid"]) == (1, 1, 1)
    assert (s["touches"], s["touches_counted_as_picks"]) == (1, 1) and s["pick_precision"] == 0.8      # 3 picks and the staff take of 5 PICK events
    assert (s["put_backs"], s["put_backs_found"], s["pipeline_put_backs"], s["put_back_precision"]) == (1, 1, 2, 1.0)
    assert s["honest_shoppers"] == 1 and s["staff_members"] == 1 and s["planogram_given"] == "nominal"
    assert c["picks"][1]["tags"] == ["slot_holds_another_product"] and res["intervals"]["pick_recall"] == [1.0, 1.0]
    assert "Staff takes listed as unpaid on a record | 1 of 1" in markdown({"split": "x", "runner": "r", "options": {}, "scored_at": "", "commit": "", **res, "clips": [c]})


def test_stress_view(tmp_path):
    clip = _clip(tmp_path)
    pub = public_view(clip, tmp_path / "work", {"drop_item_cameras": 1.0, "calib_noise_deg": 1.0, "register_delay_s": 5, "wrong_planogram": 1.0, "seed": 3})
    assert json.loads((pub / "clip.json").read_text())["cameras"] == [] and not (pub / "truth").exists()
    assert json.loads((pub / "register.jsonl").read_text())["t"] == 21.0
    assert all(a["skuId"] != b["skuId"] for a, b in zip(json.loads((pub / "layout.json").read_text())["slots"], json.loads((clip / "layout.json").read_text())["slots"]))
    assert json.loads((clip / "clip.json").read_text())["cameras"] == ["RAIL"]      # the clip itself is not touched
    again = public_view(clip, tmp_path / "work2", {"wrong_planogram": 1.0, "seed": 3})
    assert (again / "layout.json").read_text() != (clip / "layout.json").read_text() and json.loads((tmp_path / "work2" / "stress.json").read_text())["slots_renamed"] == 5


def test_splits_do_not_overlap():
    sp, sets = manifest()["splits"], {}
    for name, s in sp.items():
        seeds = s["seeds"]
        sets[name] = set(range(*map(int, seeds.split(":")))) if isinstance(seeds, str) else set(seeds)
    names = list(sets)
    assert all(not (sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:])
    assert len(sets["dev"]) == 6 and len(sets["test"]) == 6 and len(sets["train"]) >= 40
    assert len(sets["dev2"]) == 20 and len(sets["checkpoint"]) >= 24
    ck = sp["checkpoint"]
    assert set(ck["used"]) <= sets["checkpoint"] and sorted(x for b in ck["batches"] for x in b) == sorted(ck["used"])
