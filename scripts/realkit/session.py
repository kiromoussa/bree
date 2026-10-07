"""The session file of a real shelf test (session.yaml) and the store layout built from it.

One session is one recording of one shelf by a few cameras. The file is written in inches, the way the shelf is
measured with a tape; everything is turned into the pipeline's own frame here: metres, y up, floor at y = 0, the
origin on the floor under the middle of the shelf front, x to the right as you face the shelf, z out of the shelf
into the aisle.

    name: dorm_shelf_1
    fps: 10                          # the pipeline's frame rate; every clip is resampled to it
    sync: {clock: "0:05"}            # what the phone stopwatch showed at the sync clap
    lead_s: 2.0                      # the hand reaches the shelf this long after an act's clap
    shelf:
      width_in: 36
      depth_in: 12
      boards_in: [4, 18, 32, 46]     # top of each shelf board above the floor, bottom shelf first
      top_in: 60                     # top of the unit (default: last board + 14)
      rows:                          # bottom shelf first; one entry per facing, left to right as you face the shelf
        - [coke, coke, coke_zero, sprite]
        - {from_in: 2, to_in: 30, slots: [lays_classic, lays_classic, lays_bbq]}     # a row that does not span the shelf
    products:
      coke: {name: Coca-Cola 12 oz can, size_in: [2.6, 4.8, 2.6]}    # width, height, depth of one item
    door_spot_in: [-72, 36]          # where people step into view: inches along the shelf from its left end
                                     # (negative: left of the shelf), inches out from the shelf front
    register_spot_in: [108, 36]      # optional: where people stand to "pay"
    floor_marks_in: {A: [0, 24], B: [36, 24], C: [36, 60], D: [0, 60]}     # tape crosses on the floor (people camera)
    cameras:
      railA: {file: railA.h264, kind: shelf, sync_s: 12.4, rotate: 90, hfov_deg: 41, fps_in: 10}
      top:   {file: top.mp4, kind: overhead, sync_s: 9.0}

Slot ids are S<shelf>-<position>: S1-1 is the bottom shelf, first facing from the left.
"""
from __future__ import annotations

from pathlib import Path

IN = 0.0254
KINDS = ("shelf", "cooler", "checkout", "overhead", "entrance")
FIXTURE = "SHELF"


def parse_clock(v) -> float:
    """'1:02:03', '12:34.5', '75' or 75 -> seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.strip() for p in parts):
        raise ValueError(f"not a time: {v!r} (write 12:34 or 1:02:03 or seconds)")
    s = 0.0
    for p in parts:
        s = s * 60 + float(p)
    return s


def load_session(path) -> dict:
    import yaml
    s = yaml.safe_load(Path(path).read_text())
    for k in ("name", "shelf", "products", "cameras", "door_spot_in"):
        if not s.get(k):
            raise SystemExit(f"{path}: `{k}` is missing")
    for cid, c in s["cameras"].items():
        if c.get("kind") not in KINDS:
            raise SystemExit(f"{path}: camera {cid}: kind must be one of {', '.join(KINDS)}")
        if "file" not in c:
            raise SystemExit(f"{path}: camera {cid}: `file` is missing")
    s.setdefault("fps", 10)
    s.setdefault("lead_s", 2.0)
    s["sync_clock_s"] = parse_clock((s.get("sync") or {}).get("clock", 0))
    s["_dir"] = str(Path(path).resolve().parent)
    return s


def _xz(session: dict, spot) -> tuple[float, float]:
    """[inches along the shelf from its left end, inches out from the shelf front] -> layout (x, z) in metres."""
    return float(spot[0]) * IN - session["shelf"]["width_in"] * IN / 2, float(spot[1]) * IN


def build_layout(session: dict) -> dict:
    """session -> the layout.json the pipeline reads (fixtures, skus, slots with the planogram)."""
    sh = session["shelf"]
    W, D = sh["width_in"] * IN, sh.get("depth_in", 12) * IN
    boards = [b * IN for b in sh["boards_in"]]
    top = sh.get("top_in", sh["boards_in"][-1] + 14) * IN
    skus = [{"id": pid, "name": p.get("name", pid), "kind": p.get("kind", "item"), "size": [round(v * IN, 4) for v in p["size_in"]], "price": p.get("price", 0)}
            for pid, p in session["products"].items()]
    size = {s["id"]: s["size"] for s in skus}
    slots = []
    if len(sh["rows"]) > len(boards):
        raise SystemExit(f"shelf: {len(sh['rows'])} rows but only {len(boards)} boards_in heights")
    for r, row in enumerate(sh["rows"]):
        names = row["slots"] if isinstance(row, dict) else row
        x0 = (row.get("from_in", 0) if isinstance(row, dict) else 0) * IN - W / 2
        x1 = (row.get("to_in", sh["width_in"]) if isinstance(row, dict) else sh["width_in"]) * IN - W / 2
        w = (x1 - x0) / len(names)        # ponytail: facings of a row are equal width; give from_in / to_in per run of products if the overlay is off
        for c, pid in enumerate(names):
            if pid in (None, "-", ""):
                continue                   # an empty place: no slot
            if pid not in size:
                raise SystemExit(f"shelf row {r + 1}, position {c + 1}: product {pid!r} is not under `products`")
            h = size[pid][1]
            x, y = x0 + (c + 0.5) * w, boards[r] + h / 2
            slots.append({"id": f"S{r + 1}-{c + 1}", "skuId": pid, "fixtureId": FIXTURE, "position": [round(x, 4), round(y, 4), round(-D / 2, 4)],
                          "size": [round(w, 4), round(h, 4), round(D, 4)], "facings": 1, "depthCount": max(1, int(D / max(size[pid][2], 0.01))),
                          "zone": "gondola", "zoneId": FIXTURE, "face": [round(x, 4), round(y, 4), 0.0], "normal": [0, 0, 1]})
    dx, dz = _xz(session, session["door_spot_in"])
    fixtures = [{"id": FIXTURE, "type": "gondola", "position": [0, round(top / 2, 4), round(-D, 4)], "rotationY": 0, "size": [round(W, 4), round(top, 4), round(2 * D, 4)]},
                {"id": "DOOR", "type": "door", "position": [round(dx, 3), 1.0, round(dz, 3)], "rotationY": 0, "size": [1.0, 2.0, 0.1]}]
    poi = {}
    if session.get("register_spot_in"):
        rx, rz = _xz(session, session["register_spot_in"])
        # a counter just beyond the spot (further from the shelf): bree.track.floor.counter_zones puts the pay strip on its shelf side
        fixtures.append({"id": "COUNTER", "type": "counter", "position": [round(rx, 3), 0.45, round(rz + 0.55, 3)], "rotationY": 0, "size": [1.0, 0.9, 0.5]})
        poi["register"] = [round(rx, 3), 0, round(rz, 3)]
    return {"version": 1, "units": "metres", "up": "y", "store": {"width": 12, "depth": 12, "height": 3}, "fixtures": fixtures, "skus": skus, "slots": slots,
            "cameras": [], "poi": poi}


def mark_points(session: dict) -> list[tuple[str, str, tuple[float, float, float]]]:
    """(name, what to click, (x_m, z_m, height_m)): spots whose place in the layout is known from the tape measure.
    The shelf front: floor corners, both ends of every board's front edge, top corners. Then the floor marks."""
    sh = session["shelf"]
    W = sh["width_in"] * IN
    top = sh.get("top_in", sh["boards_in"][-1] + 14) * IN
    out = [("floor_left", "shelf front, LEFT end, at the floor", (-W / 2, 0.0, 0.0)), ("floor_right", "shelf front, RIGHT end, at the floor", (W / 2, 0.0, 0.0))]
    for i, b in enumerate(sh["boards_in"]):
        out += [(f"board{i + 1}_left", f"shelf board {i + 1} (from the bottom), front edge top, LEFT end", (-W / 2, 0.0, b * IN)),
                (f"board{i + 1}_right", f"shelf board {i + 1} (from the bottom), front edge top, RIGHT end", (W / 2, 0.0, b * IN))]
    out += [("top_left", "top of the shelf unit, front, LEFT end", (-W / 2, 0.0, top)), ("top_right", "top of the shelf unit, front, RIGHT end", (W / 2, 0.0, top))]
    for name, spot in (session.get("floor_marks_in") or {}).items():
        x, z = _xz(session, spot)
        out.append((f"floor_{name}", f"floor tape mark {name}", (x, z, 0.0)))
    return out
