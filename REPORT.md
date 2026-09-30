# BREE vision: overnight build report

**Bottom line.** The full pipeline runs end to end: video in, people/products/pose/tracks, store events, per-person basket, reconciliation against POS receipts, theft alert with a short head-pixelated clip. It runs at **10.1 FPS on a 4-core CPU with no GPU** on real footage. The theft logic (ledger) is close to perfect **when vision is right**: on 200 simulated store hours it reached 97.9% precision and 0.04 false alerts/hour. Under the vision error rates I *assumed* for a decent camera, it gets **70.9% precision, 50.3% recall, and 0.51 false alerts per hour** (about 12 a day in a 24-hour store). That is too many to have staff confront anyone. **None of these accuracy numbers come from real store footage; we don't have any yet.** They are the logic's behaviour under stated assumptions. The pilot's first job is to replace those assumptions with measurements.

---

## 1. What works right now, and how to see it

Everything runs from the repo root after `make setup` (creates `.venv`, pinned deps, downloads YOLO26 weights).

| What | Command | What you'll see |
|---|---|---|
| Hardware check | `make hw` | CPU-only box detected -> YOLO26 **nano** models (small models on CUDA/Apple GPUs) |
| Full pipeline on any video / webcam / RTSP | `.venv/bin/python -m bree.cli run --source <file\|0\|rtsp://...> --out out/x [--payments pos.jsonl\|stdin\|http:8765]` | `annotated.mp4` (zones, boxes, track IDs, skeletons, heads pixelated), `frames.jsonl` (per-frame boxes/IDs/keypoints; no images), `events.jsonl`, `alerts/*.json` + `alerts/*.mp4` clips, `ledger_log.txt` (plain-English reasoning per person), `summary.json` (FPS, stage timings) |
| Toy end-to-end demo | `make demo` | 8 rendered **toy** clips go through the real tracker, event engine, ledger and alerting. Result: 10 of 10 people handled correctly (2 thieves alerted, 0 flags on 8 honest shoppers) |
| Unit tests | `make test` | 63 tests: ledger (31), event engine (12), simulator (7), payments (5), multi-camera (4), vision smoke tests (4) |
| Ledger accuracy under vision noise | `make sim` | precision / recall / false alerts per hour at perfect, baseline, and 2x vision error rates |
| Everything measurable | `make bench` (~7 min) | writes `results/bench.json` and `results/bench.md` |
| Live dashboard | `make dashboard` -> http://127.0.0.1:8080 | camera view, alerts with reasons, who is in the store and what's in their basket (replays a toy clip by default; `--source rtsp://...` for a camera) |
| Edge export | `make export` | ONNX versions of both models; identical detections to PyTorch on the test frame |

Main pieces:
- **Perception** (`detect/`, `pose/`, `track/`): one YOLO26n detector pass finds people and products. Pose (COCO-17 keypoints) then runs on each person *crop*, not the full frame. On CCTV-sized people the full-frame nano pose model found **0 of ~7** people in a real frame; the crop approach found all 7. People are tracked with ByteTrack. Products use a centroid tracker, because ByteTrack's box-overlap matching lost small items moving in a hand (this made the toy walkout go undetected until fixed).
- **Events** (`events/engine.py`): readable rules, one per event. `pick` = an item appears in a hand right after that hand was inside a shelf/cooler zone. `put_back` = an item leaves the hand inside a zone. `conceal` = an in-hand item vanishes inside the shoulders-to-hips box, away from the register. `pay` = dwell in the register zone. `exit` = the track ends at the door. Zones are polygons in `configs/*.yaml`; an example single-camera gas station layout is in `configs/store_gas_station_small.yaml` (3 shelf runs, cooler bank, register, door).
- **Ledger** (`ledger/ledger.py`): per-person basket; receipts from POS or cooler taps are credited to the right person; on exit, basket minus paid = unpaid, scored with an explicit evidence table. **Alert** (tell staff) at >= 0.7; **review** (manager looks later, nobody is confronted) at >= 0.4. One uncorroborated pick can never reach "alert" on its own. Every decision is explained in plain English.
- **Payments** (`ledger/payments.py`): JSON lines from a file, stdin, or `POST /payments` on a local port; a mock source for tests. A real POS or tap reader only has to emit `{"terminal", "ts", "items":[{"sku","qty"}]}`.

## 2. Benchmark results

Source: `results/bench.json` (from `make bench`). Machine: 4-core Intel Xeon x86_64 @ 2.8 GHz, no GPU.

### 2a. Theft logic on simulated shoppers (event-level simulator, no video)
200 simulated store hours per row. 10,404 visitors, of whom 495 stole something (4.8%, deliberately higher than a real store so recall is measurable). The seed is **held out**: all tuning used a different seed. The POS feed carries item lists.

| vision error rates | precision | recall (alert) | recall (alert or review) | false alerts / hour | false alerts / 1000 honest visitors | honest "review" flags / hour | baskets exactly right |
|---|---|---|---|---|---|---|---|
| perfect vision | 97.9% | 75.4% | 96.2% | 0.04 | 0.81 | 0.03 | 100.0% |
| **baseline (assumed)** | **70.9%** | **50.3%** | **83.2%** | **0.51** | **10.29** | **4.00** | **75.5%** |
| 2x the baseline error rates | 45.7% | 22.6% | 65.1% | 0.67 | — | 5.31 | 58.0% |

Without a POS feed ("dwell" mode: standing at the register counts as paying for everything picked so far), recall at baseline drops to **29.3%** (39.8% counting reviews), at 62.8% precision and 0.43 false alerts per hour. Partial payment can't be seen without a POS feed. **The POS feed roughly doubles what we can catch.**

What was measured, and how:
- **Precision and recall** are per person. A "thief" is anyone who left with an item they didn't pay for.
- **False alerts per hour** are counted over simulated store-open hours. At the default 0.7 threshold there were 102 false alerts in 200 hours. The alert-threshold sweep shows precision flattening around 70–72% up to a 0.85 threshold, so raising the threshold mostly costs recall: 0.51 false alerts/hr at 50.3% recall at 0.7, 0.21/hr at 21.8% recall at 0.85, 0.10/hr at 15.4% recall at 0.9.
- **Recall by theft type** (baseline, alert tier):

  | theft type | caught | out of |
  |---|---|---|
  | walkout | 143 | 215 |
  | conceal + partial pay | 101 | 176 |
  | pays for one item, carries another out openly | 5 | 104 |

  The last type lands in "review" (75 of 104), by design: without concealment it looks the same as a missed put-back.
- **Basket accuracy**: 100% with perfect vision; 75.5% exact per person at baseline (item-level precision 88.9%, recall 89.0%).
- **Decision latency** (true exit to alert, camera time): median **5.0 s**, which is the deliberate wait for late POS messages; p95 35.8 s. The long tail is decisions held while a group mate or a "crowded pick" candidate is still in the store.

Which vision errors matter? I ran perfect vision with one error type switched on at a time, at its baseline rate:

| error type | false alerts / hour |
|---|---|
| none | 0.03 |
| missed register visit | 0.17 |
| POS message dropped or clock off | 0.11 |
| two people's track IDs swapped | 0.08 |
| a person's track ID split in two | 0.06 |
| missed exit | 0.06 |
| each of the others | ~0.03 |

Recall is hurt most by **missed concealments** (77.4% -> 58.8%). Singly, each error type adds little; the 0.51/hour at baseline comes from errors **combining**. For example, a missed put-back plus a slightly-off POS match adds up to an "unpaid" item with an in-hand sighting at exit.

**What this proves:** the reconciliation logic, payment attribution (queues, groups paying together, cooler taps, late or missing POS messages), hold-and-wait rules, and scoring behave correctly on correct inputs. It also shows which vision errors to invest in.
**What it does not prove:** anything about real-world accuracy. The baseline error rates (missed pick 8%, missed put-back 15%, missed conceal 40%, missed register visit 3%, ID swap 1%, ...; full list in `bench.json -> noise_model_baseline`) are my guesses. The simulator also encodes my assumptions about how people shop and steal.

### 2b. Full pipeline on toy video clips
Data: 8 rendered 2D clips (`make demo`), clearly marked **TOY DATA**. People are coloured blobs, products are coloured squares. The toy detector reads the **pixels**, not the renderer's ground truth, so the real tracker, event engine, ledger and alerting code all run for real.

| clip | truth | result |
|---|---|---|
| walkout | thief | alert |
| conceal + partial pay | thief | alert (flags the candy bar, not the paid soda) |
| normal pay | honest | nothing |
| put-back | honest | nothing |
| crowded cooler | honest | nothing (both picks flagged ambiguous; decision waited for the other person) |
| lingerer | honest | nothing |
| pocket then pay | honest | nothing |
| group, one person pays | honest | nothing |

Pipeline ran at 12.1 FPS over 3,652 frames. Processing latency from the frame that triggered the alert to the alert JSON and clip being written: 447 ms and 262 ms, mostly clip encoding.

**Proves:** the pieces fit together; outputs, clips, and logs are produced. **Does not prove:** anything about YOLO on real store footage. The clips were written alongside the rules they test.

### 2c. Real footage throughput
- **Data:** OpenCV's `vtest.avi` (real pedestrians, 795 frames, 768x576). **It has no theft labels**; the store zones don't correspond to anything in that scene, so it measures speed and tracking only.
- **Speed:** **10.13 FPS end to end** on 4 CPU cores. Per frame: detect + pose 88.3 ms, tracking 1.2 ms, events 0.7 ms, ledger 0.02 ms, drawing 7.0 ms. 45 person tracks.
- **Enough?** Cameras run at 10–15 FPS, and the event rules were tuned at 15 FPS. On a Jetson-class GPU we should comfortably exceed camera rate. That is not measured; I had no GPU.
- **ONNX** (`results/export.json`), same CPU, nothing else running: detector 43.8 ms (PyTorch) vs 46.1 ms (ONNX Runtime); pose crop 13.9 ms vs 8.2 ms. Detections are identical.

## 3. What didn't work or got skipped, and why

- **No real shoplifting video was evaluated.**
  - The Kaggle (Simuletic, DCSASS), Mendeley, HuggingFace and Google Drive hosts are **blocked by this build environment's network policy** (HTTP 403).
  - Instructions for downloading them yourself are in `data/README.md`; `make data` picks up Kaggle automatically once an API token is present.
- **PoseLift (real shoplifting, pose-only) was not used.**
  - The official data is on Google Drive (blocked here).
  - A helper agent found and cloned a third-party GitHub copy into `data/poselift/`. The automated safety check then blocked pulling data from that unofficial mirror, so **no result in this report uses it**, and `scripts/download_data.sh` doesn't reference it.
  - You may want to delete `data/poselift*` and download the official copy.
- **Concealment classifier over pose sequences (Stage 5): skipped.** Without PoseLift there was no labelled real pose data. Training on my own synthetic poses would only learn my renderer.
- **Multi-camera:** handoff by floor position (homographies, no appearance features) is built and unit-tested (`track/multicam.py`). It is **not wired into `bree run`**: I had no multi-camera footage to run it on.
- **Isaac Sim:** prep only, as asked (`src/bree/sim/isaac/`). The IRA config is a draft that has not been loaded by a real IRA build; key names must be checked against the installed version. Hand-object behaviours (reach, conceal) need custom animations; the README explains how.
- **Products:** COCO (the detector's training set) only knows "bottle", "cup" and "cell phone" among relevant items. A store-specific product model is required before real product-level results mean anything.
- **Bugs found and fixed along the way** (details in DECISIONS.md):
  - ByteTrack losing small in-hand items
  - picks credited to the wrong hand's reach
  - items "released" when the person walked out of view
  - an infinite loop on a duplicate exit
  - a simulator queue bug. Combined with a naive "first at the counter pays" rule, it produced 7.5 false alerts/hour even with perfect vision.
  - non-reproducible simulation (Python set ordering)
  - an ONNX timing I first measured while the benchmark was running; discarded and re-measured.

## 4. What I need from you

1. **Footage from the 2 pilot stores**: a few hours per camera, including busy periods, plus a still frame from each camera so I can draw zones.
2. **The POS transaction export** for the same hours (see section 6). It's the single biggest lever: recall at baseline is 50.3% with a POS feed vs 29.3% without.
3. **A Kaggle API token** (`~/.kaggle/kaggle.json`) for Simuletic and DCSASS, or run `make data` somewhere with normal internet access.
4. **The official PoseLift download** (public Google Drive link in `data/README.md`).
5. **A GPU box**, even briefly: a Jetson Orin (the likely ship target) or any RTX machine, for real edge FPS and for Isaac Sim.
6. **Decisions:**
   - What may staff do on an alert? My recommendation: nothing customer-facing during the pilot.
   - Who reviews the "review" queue?
   - Clip retention period (my suggestion: 7 days for flagged clips; nothing kept for unflagged people, which is already the behaviour).

## 5. Honest take: top 3 things before this goes into the 2 gas stations

1. **Run in shadow mode first and measure the real error rates.**
   - Record the pilot cameras and POS for 1–2 weeks, with alerts shown to no one.
   - Hand-label a few hours: picks, put-backs, conceals, register visits, exits.
   - Put the measured rates into `VisionNoise` and re-run `make bench`.
   - Only then pick the alert threshold. At my assumed rates, 0.51 false alerts/hour is ~12 per day per 24-hour store, too many to act on in front of customers. Concealment-backed alerts are the precise ones: at baseline, alerts where concealment was seen were right 175/192 times (91.1%) vs 74/159 (46.5%) for alerts without it. So the first "live" mode could be limited to those.
2. **Get the POS integration and camera placement right, because that's where false alerts come from.**
   - Missed register visits, dropped or late POS messages, and track-ID mix-ups are the top false-alert sources; product recognition errors barely matter (category confusions add ~0 in the ablation).
   - Concretely:
     - a camera with a clean view of the counter
     - a POS feed with timestamps synced to camera time (NTP on both)
     - a camera covering the door
     - a second camera on the cooler bank for crowded reaches and concealment
3. **Train a store-specific product detector and validate pose on the real cameras.**
   - COCO can't tell an energy drink from a candy bar.
   - Collect a few hundred labelled frames per camera of the operator's top-stolen SKUs, plus Isaac Sim renders.
   - Check that the crop-pose approach still gets wrists reliably at the real camera height and resolution: every pick/put-back/conceal rule depends on wrists.

## 6. What to ask the operator for

- **Cameras:**
  - make and model
  - resolution, FPS and codec
  - whether we can pull RTSP locally (and credentials)
  - mounting height and where each camera points
  - a still frame from every camera in both pilot stores
  - whether he's open to adding 1–2 cameras (counter-facing, cooler-facing)
- **Existing CCTV footage:**
  - a few hours per store, including rush hours and at least a few known theft incidents (with rough timestamps) if he has them
  - how long his system keeps footage
- **POS:**
  - which POS system (e.g. Verifone Commander, Gilbarco Passport, NCR, Clover)
  - whether it exports transactions in real time or only in batches
  - which fields: timestamp, register/terminal ID, line items with SKU/UPC and quantity, voids and refunds
  - how fuel and lottery lines appear
  - whether the POS clock is NTP-synced
  - a sample export covering the same hours as the footage
- **Shrink:**
  - which items get stolen most (energy drinks, beer, candy, lighters, phone accessories, cigarettes?)
  - estimated monthly shrink in dollars per store
  - when theft happens (time of day)
  - common methods (walkouts, pocketing, "paid for one, took two", employee theft)
  - how he finds out today
- **Store layout:**
  - a floor plan or phone photos of each store
  - the cooler brand
  - whether coolers are already card-tap/locked
  - number of registers
  - typical staff count per shift
- **Policy:**
  - what staff are allowed to do when an alert fires
  - local rules on camera analytics and signage
  - who at his company signs off on privacy

---
Design choices and their reasons: `DECISIONS.md`. Build status: `PROGRESS.md`. Raw numbers: `results/bench.json`, `results/bench.md`, `results/export.json`.
