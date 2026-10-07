"""Shelf events for a whole clip or one camera, in the shared contract, without a person box.

    ShelfCamera(camera_id, cam, layout, fps)      one item camera: .update(frame) -> shelf events, .finish() -> the rest
    camera_events(video, camera_id, cam, layout)  the same over a video file
    run_clip(clip)                                every shelf, cooler and counter camera of a clip folder
    fuse_views(events, layout)                    the same act seen by two cameras becomes one event (slot by vote)

Two cues per camera, fused here:
  shelf_diff  bree.shelf.diff: which slot's picture changed once the hand was gone (slot and SKU from the planogram)
  hand_item   bree.shelf.hand: an item in a hand (SKU detector on the moving regions) and the hand point
A shelf-diff event that has a held item of a candidate slot's SKU next to it in time and place becomes source "both":
the SKU is confirmed, the slot is the candidate with that SKU, and the time is the moment the item was first (take) or
last (put) seen in the hand. A held item that appears at a slot of its SKU with no readable shelf change is reported
as source "hand_item".

Beyond the shared contract, for the join (bree.events.shelf.confirm_puts): a take carries "eids" (names of the
readings it is made of), a put of the pixel comparison carries "undoes" (the names of the take whose place that camera
saw go back to the picture from before), and a put with the item seen in a hand carries "item_in" (the item's track
ends at the slot, coming from further away: it went in, it did not come out).

Reads pixels, calibration and layout.json (the planogram). Never truth/.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from bree.calib.camera import Camera
from bree.shelf.diff import DiffConfig, ShelfDiff
from bree.shelf.hand import HandConfig, HandItemCue
from bree.shelf.slots import SlotConfig, SlotWatch

ITEM_KINDS = ("shelf", "cooler", "checkout")


@dataclass
class ShelfConfig:
    diff: DiffConfig = field(default_factory=DiffConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    slots: SlotConfig = field(default_factory=SlotConfig)
    use_hand: bool = True
    use_slots: bool = True             # per-slot row position from the detector (bree.shelf.slots)
    confirm_window_s: tuple[float, float] = (-1.0, 3.0)   # held item seen this long before t_start .. after t_end of a shelf change
    confirm_px: float = 3.0            # ... within this many slot-box diagonals of the slot (and at least 80 px)
    transient_s: float = 2.0           # a take and a put of one slot this close together: nothing happened
    hand_only: bool = False            # report held items that start at a slot of their SKU with no shelf change. Off:
                                       # measured on TRAIN clip 4900, 14 of 15 such events were false
    same_act_s: float = 2.5            # fuse_views: two cameras' events of one act are within this
    same_act_m: float = 0.45
    same_slot_m: float = 0.2           # one camera, two cues: slots this close are the same act
    repeat_s: float = 4.0              # one camera, a second take (or put) of the same or the neighbouring facing this soon,
    repeat_m: float = 0.1              # with no opposite event in between, is the same act read twice: dropped
    put_in: tuple[float, float] = (0.25, 0.6)   # a put with the item seen going in: its track ends within this share of the
                                       # search radius of the slot, at no more than this share of where it began (TRAIN-seed
                                       # clips: 19 of 27 real put-backs with an item track pass, 9 of 38 other puts do)
    item_lead_s: float = 0.3           # a taken item is first seen in the hand about this long after it left the slot
                                       # (TRAIN-seed clips, 84 takes: median 0.17 s, 9 in 10 under 0.46 s)


def skus_of(layout: dict) -> dict:
    return {s["id"]: s["size"] for s in layout.get("skus", []) if "size" in s}


class ShelfCamera:
    """One item camera. detector: bree.train.backend.SkuDetector (or anything with the same call), or None for the
    shelf comparison alone. hand: callable(image, changed_mask, scale) -> [(u, v, conf)], see bree.shelf.hand."""

    def __init__(self, camera_id: str, cam: Camera, layout: dict, fps: float, cfg: ShelfConfig | None = None, detector=None, hand=None):
        self.cfg = cfg or ShelfConfig()
        self.id, self.cam, self.fps, self.slots = camera_id, cam, fps, layout["slots"]
        self.diff = ShelfDiff(camera_id, cam, self.slots, fps, self.cfg.diff, skus_of(layout), layout.get("fixtures"))
        self.cue = HandItemCue(self.diff, detector, hand, self.cfg.hand) if self.cfg.use_hand and (detector or hand) else None
        self.watch = SlotWatch(self.diff, skus_of(layout), self.cfg.slots) if detector is not None and self.cfg.use_slots and self.cue else None
        self.det = detector
        if self.watch is not None:
            self.cue.stock = self.watch.is_stock
        self.emitted: list[dict] = []
        self.pending: list[dict] = []      # shelf-diff events waiting for the confirm window to close
        self.dropped: list[tuple[dict, dict]] = []     # (take, put) pairs dropped as a passing change
        self.f = -1

    def update(self, image: np.ndarray) -> list[dict]:
        self.f += 1
        new = self.diff.update(image)
        if self.cue is not None:
            self.cue.update(image, self.f)
        if self.watch is not None:
            if self.f < self.cfg.slots.learn:          # the first frames: one look at the whole picture to learn every row
                boxes, _, cls = self.det(image, None)
                self.watch.update(self.f, boxes, [self.det.names[int(k)] for k in cls])
            elif self.cue.last is not None:
                new += self.watch.update(*self.cue.last)
        return self._step(new)

    def _step(self, new: list[dict]) -> list[dict]:
        """Take in this frame's raw events of both cues; give back the ones whose waiting time is over."""
        for e in new:
            # a slot that changes and is back as it was within moments (a cooler door swinging over it, a sleeve in
            # front of it): nothing left the shelf, drop both
            took = next((p for p in self.pending if e["kind"] == "put" and p["kind"] == "take" and p["slot_id"] == e["slot_id"]
                         and e["t_start"] - p["t_start"] <= self.cfg.transient_s), None)
            twin = next((p for p in self.pending + self.emitted[-20:] if self._same_act(p, e)), None)
            if took is not None:
                self.pending.remove(took)
                self.dropped.append((took, e))
            elif twin is None and (again := self._repeat(e)) is not None:
                self._absorb(again, e)
                self.dropped.append((again, e))
            elif twin is not None and any(twin is p for p in self.pending):
                cues = [*(twin.get("cues") or [twin.get("cue")]), e.get("cue")]
                self._merge(twin, e)
                twin["cues"] = cues
            elif twin is not None:
                self._absorb(twin, e)
                self.dropped.append((twin, e))     # already reported by the other cue
            else:
                self.pending.append(e)
        t, hold = self.f / self.fps, self.cfg.confirm_window_s[1]
        ready = [e for e in self.pending if t >= e["t_end"] + hold]
        self.pending = [e for e in self.pending if t < e["t_end"] + hold]
        return self._out(ready, t)

    def finish(self) -> list[dict]:
        out, self.pending = self.pending, []
        return self._out(out, float("inf"))

    def _out(self, ready: list[dict], t: float) -> list[dict]:
        if self.cue is not None:
            ready = [self._confirm(e) for e in ready] + self._hand_only(t)
        for e in ready:        # every take has a name, so that a put can say which reading it undoes
            if e["kind"] == "take" and not e.get("eids"):
                e["eids"] = [f"{self.id}:{e['t_end']}:{e['slot_id']}:{e.get('cue')}"]
        self.emitted += ready
        return ready

    @staticmethod
    def _absorb(kept: dict, e: dict) -> None:
        """`e` is dropped as a second reading of `kept`: its names stay with `kept`, so a later put that undoes `e`
        is still understood. ponytail: `kept` may already have been handed out; a camera node that streams events
        sends this as an amendment, the clip runner here writes its events at the end."""
        for k in ("eids", "undoes"):
            if e.get(k):
                kept[k] = sorted({*(kept.get(k) or []), *e[k]})

    def _same_act(self, a: dict, b: dict) -> bool:
        """Two cues reporting one act (never two events of the same cue)."""
        if a["kind"] != b["kind"] or a.get("cue") == b.get("cue") or abs(a["t"] - b["t"]) > self.cfg.same_act_s:
            return False
        if b.get("cue") in (a.get("cues") or []):
            return False
        ids = lambda e: {e["slot_id"], *(sid for sid, _ in e.get("slots") or [])}
        return bool(ids(a) & ids(b)) or float(np.linalg.norm(np.subtract(a["point_3d"], b["point_3d"]))) <= self.cfg.same_slot_m

    def _repeat(self, e: dict) -> dict | None:
        """The earlier event this one repeats, if any. ponytail: two real units taken from one facing within repeat_s
        read as one event of count 1; the slot watch's row count is the upgrade path for that."""
        c, seen = self.cfg, self.pending + self.emitted[-20:]
        for p in seen:
            if (p["kind"] == e["kind"] and abs(e["t"] - p["t"]) <= c.repeat_s
                    and (p["slot_id"] == e["slot_id"] or float(np.linalg.norm(np.subtract(p["point_3d"], e["point_3d"]))) <= c.repeat_m)
                    and not any(o["kind"] != e["kind"] and o["slot_id"] in (p["slot_id"], e["slot_id"]) and min(p["t"], e["t"]) <= o["t"] <= max(p["t"], e["t"]) for o in seen)):
                return p
        return None

    @property
    def status(self) -> list[dict]:
        """Camera health records of the pixel comparison (unreliable, reference_reset, reliable, needs_recalibration)."""
        return self.diff.status

    def _merge(self, kept: dict, e: dict) -> None:
        """Two cues saw one act. Slot and take time from the pixel comparison when it saw it (measured on TRAIN clips
        4900 to 4902: where the two disagreed on the slot, it was the slot watch that had the neighbour of the same
        SKU), then the slot watch, then the item in a hand. The unit count is the slot watch's (row positions)."""
        rank = {None: 0, "both_cues": 0, "slot_state": 1, "hand_item": 2}
        base, other = sorted((kept, e), key=lambda x: rank[x.get("cue")])
        pix = next((x for x in (kept, e) if x.get("cue") is None), None)
        t = pix["t"] if pix is not None and kept["kind"] == "take" else base["t"]
        row = next((x for x in (kept, e) if x.get("row")), None)
        if row is not None:
            base = {**base, "count": row["count"], "row": row["row"]}
        ev = {**base["evidence"], **{k: v for k, v in other["evidence"].items() if v and k != "frames"},
              "frames": sorted({*base["evidence"]["frames"], *other["evidence"]["frames"]})}
        for k in ("eids", "undoes"):
            if kept.get(k) or e.get(k):
                base = {**base, k: sorted({*(kept.get(k) or []), *(e.get(k) or [])})}
        kept.update({**base, "t_end": kept["t_end"], "t": t, "t_start": min(base["t_start"], other["t_start"], kept["t_end"]), "source": "both", "cue": "both_cues", "evidence": ev,
                     "slots": list(base.get("slots") or []) + [x for x in other.get("slots") or [] if x[0] != base["slot_id"]]})

    def _confirm(self, e: dict) -> dict:
        """Look for the item in a hand next to the changed slot; fix SKU, slot and time with it."""
        c, sidx = self.cfg, {s["id"]: i for i, s in enumerate(self.slots)}
        j = sidx[e["slot_id"]]
        b = self.diff.boxes[j] / self.cfg.diff.scale
        centre, reach = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), max(80.0, c.confirm_px * float(np.hypot(b[2] - b[0], b[3] - b[1])))
        cand_sku = {self.slots[sidx[sid]].get("skuId"): sid for sid, _ in reversed(e.get("slots") or [(e["slot_id"], 1.0)])}
        cand_sku[e["sku_id"]] = e["slot_id"]
        best = None
        for tr in self.cue.tracks:
            if tr["sku"] not in cand_sku:
                continue
            obs = [(f, u, v, cf) for f, u, v, cf in tr["obs"]
                   if e["t_start"] + c.confirm_window_s[0] <= f / self.fps <= e["t_end"] + c.confirm_window_s[1] and np.hypot(u - centre[0], v - centre[1]) <= reach]
            if obs and (best is None or len(obs) * (tr["sku"] == e["sku_id"]) + len(obs) > best[0]):
                best = (len(obs) * (tr["sku"] == e["sku_id"]) + len(obs), tr, obs)
        hands = [(f, u, v) for f, u, v, _ in self.cue.hands if e["t_start"] - 1.0 <= f / self.fps <= e["t_end"] + 0.5 and np.hypot(u - centre[0], v - centre[1]) <= reach]
        if best is None:
            if hands:       # no item seen, but a hand was at the slot: keep its point and time of closest approach
                f, u, v = min(hands, key=lambda h: np.hypot(h[1] - centre[0], h[2] - centre[1]))
                e.update(hand_px=[round(u, 1), round(v, 1)])
            return e
        _, tr, obs = best
        tr["used"] = True
        f, u, v, _ = obs[0] if e["kind"] == "take" else obs[-1]
        if tr["sku"] != e["sku_id"] and e.get("cue") is None:       # the detector saw the SKU of a neighbouring candidate slot: that slot it is
            s = self.slots[sidx[cand_sku[tr["sku"]]]]
            e.update(slot_id=s["id"], sku_id=s.get("skuId"), point_3d=[round(float(x), 3) for x in s["face"]], slot_changed_by_detector=True)
        if e["kind"] == "put":
            d0, d1 = (float(np.hypot(o[1] - centre[0], o[2] - centre[1])) / reach for o in (tr["obs"][0], tr["obs"][-1]))
            e["item_in"] = bool(d1 <= c.put_in[0] and d1 <= c.put_in[1] * d0)
        if e["kind"] == "take" and e["t_start"] <= f / self.fps <= e["t_end"]:
            # the slot was hidden from t_start (somebody stood in front of it) and the item is first seen in a hand
            # inside that span: it left the shelf just before that sighting, not when the slot was first hidden
            e["t"] = round(max(e["t"], f / self.fps - c.item_lead_s), 3)
        e.update(source="both", t_item=round(f / self.fps, 3), hand_px=[round(u, 1), round(v, 1)],
                 sku_conf=round(float(max(e["sku_conf"], np.mean([o[3] for o in obs]))), 3), detector_frames=len(obs))
        e["evidence"]["frames"] = sorted({*e["evidence"]["frames"], f})
        return e

    def _hand_only(self, t: float) -> list[dict]:
        if not self.cfg.hand_only or (self.pending and t != float("inf")):
            return []          # a shelf change is still waiting for its confirm window: it may claim one of these tracks
        out = []
        for tr in self.cue.closed(t):
            ev = self.cue.as_event(tr)
            if ev and not tr.get("used") and not any(self._same_act(p, ev) for p in self.emitted[-20:] + out):
                out.append(ev)
        return out


def frames(path: Path, max_frames: int | None = None):
    cap, i = cv2.VideoCapture(str(path)), 0
    while max_frames is None or i < max_frames:
        ok, im = cap.read()
        if not ok:
            break
        yield im
        i += 1
    cap.release()


def camera_events(video: Path, camera_id: str, cam: Camera, layout: dict, fps: float, cfg: ShelfConfig | None = None,
                  detector=None, hand=None, max_frames: int | None = None, status: list | None = None, looks: list | None = None) -> list[dict]:
    """Shelf events of one video. Pass a list as `status` to also get the camera health records, and a list as `looks`
    to get every look of the hand and held-item detector (bree.shelf.hand.HandItemCue.log; bree.concealment reads them)."""
    sc, out = ShelfCamera(camera_id, cam, layout, fps, cfg, detector, hand), []
    if looks is not None and sc.cue is not None:
        sc.cue.log = looks
    for im in frames(video, max_frames):
        out += sc.update(im)
    out += sc.finish()
    if status is not None:
        status += sc.status
    return out


def unreliable_windows(status: list[dict], t_end: float = float("inf")) -> dict[str, list[tuple[float, float]]]:
    """camera_id -> [(from, to)]: spans in which that camera reported nothing because its picture could not be compared
    with the reference. The association step should not read "no shelf event" from a camera inside its windows."""
    out: dict[str, list[tuple[float, float]]] = {}
    start: dict[str, float] = {}
    for s in sorted(status, key=lambda s: s["t"]):
        cam = s["camera_id"]
        if s["status"] in ("unreliable", "needs_recalibration"):
            start.setdefault(cam, s["t"])
        elif s["status"] in ("reliable", "reference_reset") and cam in start:
            out.setdefault(cam, []).append((start.pop(cam), s["t"]))
    for cam, t0 in start.items():
        out.setdefault(cam, []).append((t0, t_end))
    return out


def _one_camera(args) -> tuple[list[dict], list[dict]]:
    clip, cam_id, cfg, use_detector, max_frames, looks_dir = args
    from bree.concealment.scan import patient_nms
    from bree.sim.bench import load_calibration
    cv2.setNumThreads(2)
    patient_nms()          # the detector's NMS gives up after 2 s on a busy machine and drops boxes: the result must not depend on the load
    layout = json.loads((Path(clip) / "layout.json").read_text())
    fps = float(json.loads((Path(clip) / "clip.json").read_text())["fps"])
    det = None
    if use_detector and (cfg or ShelfConfig()).use_hand:
        from bree.shelf.hand import sku_detector
        det = sku_detector()
    status: list[dict] = []
    looks: list[dict] | None = [] if looks_dir and det is not None else None
    events = camera_events(Path(clip) / f"{cam_id}.mp4", cam_id, load_calibration(clip)[cam_id], layout, fps, cfg, det, None, max_frames, status, looks)
    if looks is not None:
        Path(looks_dir, f"looks_{cam_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in looks))
    return events, status


def run_clip(clip: Path, cfg: ShelfConfig | None = None, jobs: int = 4, detector: bool = True, max_frames: int | None = None,
             cameras: list[str] | None = None, verbose: bool = False, evidence_dir: str | None = None, status: list | None = None,
             looks_dir: str | None = None) -> list[dict]:
    """Shelf events of every item camera of a clip folder (video, calibration.json, layout.json, clip.json), by t.
    evidence_dir: write before and after crops of every pixel-comparison event there (evidence.before / .after).
    status: pass a list to also get the camera health records (see unreliable_windows).
    looks_dir: write looks_<camera>.jsonl there, every look of the held-item detector (for bree.concealment.cue)."""
    clip = Path(clip)
    if looks_dir:
        Path(looks_dir).mkdir(parents=True, exist_ok=True)
    if evidence_dir:
        cfg = replace(cfg or ShelfConfig(), diff=replace((cfg or ShelfConfig()).diff, evidence_dir=str(evidence_dir)))
    kind = {c["id"]: c["kind"] for c in json.loads((clip / "calibration.json").read_text())["cameras"]}
    cams = [c for c in json.loads((clip / "clip.json").read_text())["cameras"] if kind[c] in ITEM_KINDS and (not cameras or c in cameras)]
    with ProcessPoolExecutor(max(1, min(jobs, len(cams)))) as ex:
        runs = list(ex.map(_one_camera, [(str(clip), c, cfg, detector, max_frames, str(looks_dir) if looks_dir else None) for c in cams]))
    if verbose:
        for c, r in zip(cams, runs):
            print(f"  {c}: {len(r[0])} shelf events" + (f", status {[s['status'] for s in r[1]]}" if r[1] else ""), flush=True)
    if status is not None:
        status += sorted((s for r in runs for s in r[1]), key=lambda s: s["t"])
    return sorted((e for r in runs for e in r[0]), key=lambda e: e["t"])


def fuse_views(events: list[dict], layout: dict, cfg: ShelfConfig | None = None) -> list[dict]:
    """One event per act: events of different cameras with the same kind, close in time and place, are merged. The
    slot is a vote (each view's candidate slots weighted by overlap, a detector-confirmed view counts double)."""
    cfg, slots, out = cfg or ShelfConfig(), {s["id"]: s for s in layout["slots"]}, []
    for e in sorted(events, key=lambda e: e["t_start"]):
        if e.get("point_3d") is None:
            out.append({**e, "cameras": [e["camera_id"]], "views": [e]})
            continue
        p = np.asarray(e["point_3d"], float)
        for g in out:
            if (g.get("point_3d") is not None and g["kind"] == e["kind"] and e["camera_id"] not in g["cameras"]
                    and (abs(g["t_start"] - e["t_start"]) <= cfg.same_act_s or abs(g["t"] - e["t"]) <= cfg.same_act_s)
                    and np.linalg.norm(np.asarray(g["point_3d"], float) - p) <= cfg.same_act_m):
                g["cameras"].append(e["camera_id"])
                g["views"].append(e)
                break
        else:
            out.append({**e, "cameras": [e["camera_id"]], "views": [e]})
    for g in out:
        for k in ("eids", "undoes"):       # the readings this act is made of, and the readings it undoes
            if any(v.get(k) for v in g["views"]):
                g[k] = sorted({x for v in g["views"] for x in v.get(k) or []})
        if g.get("eids"):                  # what each reading said, for when a put undoes some of them
            g["read_as"] = {x: [v["slot_id"], v["sku_id"], v["source"] == "both", v.get("detector_frames") or 0] for v in g["views"] for x in v.get("eids") or []}
        if g["kind"] == "put":
            g["item_in"] = any(v.get("item_in") for v in g["views"])
        if len(g["views"]) < 2:
            continue
        votes: dict[str, float] = {}
        for v in g["views"]:
            k = 2.0 if v["source"] == "both" else 1.0
            if v["source"] == "both" or not v.get("slots"):
                votes[v["slot_id"]] = votes.get(v["slot_id"], 0.0) + k
            for sid, w in v.get("slots") or []:
                votes[sid] = votes.get(sid, 0.0) + k * w * (v["source"] != "both" or slots[sid].get("skuId") == v["sku_id"])
        best = max(votes, key=votes.get)
        sku = slots[best].get("skuId")
        timed = [v for v in g["views"] if v["source"] != "shelf_diff"] or g["views"]
        g.update(slot_id=best, sku_id=sku, point_3d=[round(float(x), 3) for x in slots[best]["face"]],
                 sku_conf=round(sum(w for s, w in votes.items() if slots[s].get("skuId") == sku) / sum(votes.values()), 3),
                 t=min(v["t"] for v in timed), t_start=min(v["t_start"] for v in g["views"]), t_end=min(v["t_end"] for v in g["views"]),
                 source="both" if {v["source"] for v in g["views"]} != {"shelf_diff"} and {v["source"] for v in g["views"]} != {"hand_item"} else g["views"][0]["source"])
    return out


if __name__ == "__main__":      # .venv/bin/python -m bree.shelf.events <clip folder> [--out shelf_events.jsonl] [--no-detector]
    import argparse
    ap = argparse.ArgumentParser(description="Shelf events of every item camera of a clip folder, one JSON record per line.")
    ap.add_argument("clip")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--fused", action="store_true", help="one record per act (views of several cameras merged)")
    ap.add_argument("--evidence", help="folder for before and after crops of every pixel-comparison event")
    ap.add_argument("--status", help="write the camera health records (unreliable, reference_reset, ...) to this .jsonl")
    a = ap.parse_args()
    health: list[dict] = []
    evs = run_clip(Path(a.clip), jobs=a.jobs, detector=not a.no_detector, verbose=True, evidence_dir=a.evidence, status=health)
    if a.status:
        Path(a.status).write_text("".join(json.dumps(s) + "\n" for s in health))
    if a.fused:
        evs = [{k: v for k, v in g.items() if k != "views"} for g in fuse_views(evs, json.loads((Path(a.clip) / "layout.json").read_text()))]
    text = "".join(json.dumps(e) + "\n" for e in evs)
    Path(a.out).write_text(text) if a.out else print(text, end="")
