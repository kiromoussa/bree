"""SYNTHETIC alerts, clips, reviewers and staged tests, for tests and the end to end demo.

Nothing here is a measurement of the vision pipeline. The pictures are drawn rectangles, the
"reviewers" are a seeded random number generator, and the theft / honest mix is made up. It proves
the feedback loop works: alert in, decision, label export, metrics, owner report.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from bree.alerts.types import Alert, UnpaidItem
from bree.review.store import ReviewStore, save_evidence_frame

BASE = dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc).timestamp()   # a Monday, 00:00 UTC
CAMERAS = {"synthetic_cam_cooler": "cooler_bank", "synthetic_cam_snacks": "snack_aisle"}
ITEMS = ("energy_drink", "soda_bottle", "candy_bar", "chips")
W, H = 320, 240


def _frame_set(folder: Path, aid: str, item: str, rng) -> tuple[list[dict], str]:
    """Drawn frames for one alert: 8 pick frames + 8 conceal frames with pose, one clean
    head-pixelated image with the item box, and a clip of the same pixelated frames."""
    import cv2
    from bree.alerts.writer import write_frames
    from bree.events.observations import PersonObs
    px = int(rng.integers(60, 200))
    products = [{"id": 100 + i, "bbox": [20.0 + 40 * i, 30.0, 40.0 + 40 * i, 60.0], "cat": ITEMS[i % 4], "conf": 0.8}
                for i in range(6)]
    item_box = [float(px + 30), 100.0, float(px + 44), 124.0]
    frames, jpgs = [], []
    for i in range(16):
        kp = np.zeros((17, 3), np.float32)
        kp[:, 0] = px + rng.normal(0, 3, 17)
        kp[:, 1] = np.linspace(70, 220, 17) + rng.normal(0, 2, 17)
        kp[:, 2] = 0.9
        kp[9] = [px + 30 - (2 * i if i >= 8 else 0), 110 + (4 * (i - 8) if i >= 8 else 0), 0.9]   # wrist: reach, then down to the hip
        img = np.full((H, W, 3), 225, np.uint8)
        for p in products:
            cv2.rectangle(img, (int(p["bbox"][0]), int(p["bbox"][1])), (int(p["bbox"][2]), int(p["bbox"][3])), (90, 140, 200), -1)
        cv2.rectangle(img, (px - 14, 80), (px + 14, 220), (70, 70, 70), -1)           # body
        cv2.circle(img, (px, 68), 12, (150, 170, 210), -1)                           # head (gets pixelated)
        cv2.rectangle(img, tuple(map(int, item_box[:2])), tuple(map(int, item_box[2:])), (40, 40, 200), -1)
        cv2.putText(img, f"SYNTHETIC {aid}", (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        person = PersonObs(1, (px - 14.0, 56.0, px + 14.0, 220.0))
        f = {"t": round(4.0 + i / 10, 2), "event": "pick" if i < 8 else "conceal", "item": item, "kpts": kp.round(1).tolist()}
        if i == 6:      # the pick frame: the one clean image kept for the detector
            f.update(image=save_evidence_frame(img, [person], folder / "frames" / f"{aid}.jpg"),
                     width=W, height=H, item_box=item_box, products=products + [{"id": 1, "bbox": item_box, "cat": item, "conf": 0.7}])
            pix = cv2.imread(f["image"])
        else:
            from bree.alerts.annotate import blur_heads
            pix = blur_heads(img, [person])
        jpgs.append((f["t"], cv2.imencode(".jpg", pix)[1].tobytes()))
        frames.append(f)
    clip = write_frames(jpgs, folder / "alerts" / f"{aid}.mp4", 10.0)
    return frames, clip


def populate(folder: str | Path, n: int = 60, seed: int = 0, retention_days: float = 30.0) -> tuple[ReviewStore, float]:
    """A store with `n` SYNTHETIC alerts over three weeks, two reviewers and three staged tests.
    Returns (store, now) where `now` is the pretend current time (three weeks after BASE)."""
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    now = BASE + 21 * 86400
    store = ReviewStore(folder, retention_days=retention_days, now=now)
    cams = list(CAMERAS)
    staged_alert = {2: "T1", 7: None}            # alert 2 names its test; alert 7 is matched to T2 by time
    for k in range(n):
        aid = f"SYN{k:04d}"
        cam = cams[k % 2]
        item = ITEMS[int(rng.integers(0, 4))]
        ts = BASE + (k + 0.5) * (21 * 86400 / n)
        truth = rng.choice(["theft", "theft_wrong_item", "honest"], p=[0.55, 0.15, 0.30])
        fixed = k == 0 or k in staged_alert       # alert 0 is logged twice below; staged tests are real thefts
        if fixed:
            truth = "theft"
        concealed = bool(truth != "honest" and rng.random() < 0.6)
        visited = bool(truth == "honest" or rng.random() < 0.3)
        frames, clip = _frame_set(folder, aid, item, rng)
        uncertain = bool(rng.random() < 0.1)
        alert = Alert(aid, int(rng.integers(1, 50)), "review" if uncertain else "alert", round(float(rng.uniform(0.5, 0.95)), 3),
                      10.0, 15.0, [UnpaidItem(item, None, 4.6, CAMERAS[cam], 0.85, concealed, not concealed, [], 0.8)],
                      [f"{item}: taken from {CAMERAS[cam]} and not paid for"]
                      + (["capped at review: identity uncertain (synthetic)"] if uncertain else []),
                      paid_items=[{"category": "chips", "sku": None}] if visited else [], visited_register=visited,
                      clip_path=clip, basket=[item])
        store.add_alert(alert.to_dict(), camera=cam, event_ts=ts, frames=frames,
                        staged_test_id=staged_alert.get(k), now=ts + 5)
        if k in staged_alert:
            store.add_staged_test(staged_alert[k] or "T2", ts - 20, cam, CAMERAS[cam], item, "SYNTHETIC staged test")

        def answer(noise: float) -> tuple[str, str | None]:
            if rng.random() < noise:
                return "unclear", None
            if truth == "theft_wrong_item":
                return "wrong_item", ITEMS[(ITEMS.index(item) + 1) % 4]
            return ("confirmed_theft" if truth == "theft" else "not_theft"), None

        if rng.random() < 0.9 or fixed:                  # reviewer one sees most alerts
            dec, fix = answer(0.0 if fixed else 0.05)
            lag = float(rng.uniform(600, 20 * 3600))
            store.decide(aid, "reviewer_one", dec, fix, pick_seen=truth != "honest",
                         conceal_seen=concealed if rng.random() < 0.7 else None,
                         time_to_decision_s=float(rng.uniform(3, 9)), now=ts + lag)
            if rng.random() < 0.4 and not fixed:         # reviewer two sees some of the same alerts
                dec2, fix2 = answer(0.15)
                store.decide(aid, "reviewer_two", dec2, fix2, time_to_decision_s=float(rng.uniform(3, 9)),
                             now=ts + lag + float(rng.uniform(600, 7200)))
    store.add_staged_test("T3", BASE + 3.3 * 86400, cams[0], CAMERAS[cams[0]], "candy_bar", "SYNTHETIC staged test, no alert")
    # The same alert logged a second time under another id (two ledgers saw it): export must dedup.
    first = store.alerts()[0]
    store.add_alert({"alert_id": "SYN_DUP", "tier": first["tier"], "confidence": first["confidence"],
                     "unpaid_items": first["items"], "reasons": first["reasons"], "clip_path": first["clip_path"]},
                    camera=first["camera"], event_ts=first["event_ts"] + 1, frames=first["frames"], now=first["event_ts"] + 6)
    for r in first["reviews"]:
        store.decide("SYN_DUP", r["reviewer_id"], r["decision"], r["corrected_item"], r["pick_seen"], r["conceal_seen"],
                     time_to_decision_s=r["time_to_decision_s"], now=r["decided_ts"])
    return store, now
