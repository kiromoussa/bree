# BREE vision: overnight build report

**Bottom line.** The full pipeline runs end to end, **at 9.7 FPS on a 4-core CPU with no GPU** on real footage:

```
video in → people / products / pose / tracks → store events → per-person basket
         → reconciliation against POS receipts → theft alert (+ short clip, heads pixelated)
```

**When vision is right**, the theft logic (the "ledger") works well. On 200 simulated store hours it scored:
- **97.9% precision** (almost every alert was a real thief)
- **0.04 false alerts per hour**
- **76.4% recall** at alert level: the share of thieves who triggered an alert. That rises to 96.8% if you also count "review" flags.

**Under the vision error rates I *assumed*** for a decent camera, it scores:
- **72.5% precision, 47.9% recall, 0.45 false alerts per hour**
- Across 5 simulator seeds: precision 67–76%, recall 45–53%, 0.40–0.53 false alerts per hour.
- That is roughly 11 false alerts a day in a 24-hour store: **too many to confront anyone on.**

Alerts backed by an observed concealment are the exception: 93% of those were right.

**None of the accuracy numbers come from real store footage; we have none yet.** They describe the logic's behaviour under stated assumptions. The pilot's first job is to replace those assumptions with measurements.

---

## 1. What works right now, and how to see it

Run everything from the repo root after `make setup`. That creates `.venv`, installs pinned dependencies, and downloads the YOLO26 weights.

| What | Command | What you'll see |
|---|---|---|
| Hardware check | `make hw` | This CPU-only box → YOLO26 **nano** models (small models on CUDA / Apple GPUs) |
| Full pipeline on a video, webcam or RTSP camera | `.venv/bin/python -m bree.cli run --source <file\|0\|rtsp://...> --out out/x [--payments pos.jsonl\|stdin\|http:8765] [--no-video]` | Output files listed below this table |
| Toy end-to-end demo | `make demo` | 8 rendered **toy** clips go through the real tracker, event engine, ledger and alerting: 10 of 10 people handled correctly (2 thieves alerted, 0 flags on 8 honest shoppers) |
| Unit tests | `make test` | **80 tests**: ledger 42, event engine 18, simulator 7, payments 5, multi-camera 4, vision smoke tests 4 |
| Ledger accuracy under vision noise | `make sim` | Precision, recall and false alerts per hour at perfect, baseline and 2x vision error rates |
| Everything measurable | `make bench` (~9–10 min) | Writes `results/bench.json` and `results/bench.md` |
| Live dashboard | `make dashboard`, then open http://127.0.0.1:8080 | Camera view; alerts with their reasons; who is in the store and what's in their basket. Replays a toy clip by default; use `--source rtsp://...` for a real camera |
| Edge export | `make export` | ONNX versions of both models |

What `bree run` writes to `--out`:
- `annotated.mp4`: a debug video of everyone, heads pixelated. Skip it with `--no-video`.
- `frames.jsonl`: every person's boxes and keypoints, per frame. No images.
- `events.jsonl`: the store events.
- `alerts/*.json` + `alerts/*.mp4`: one record and one evidence clip per flagged person.
- `ledger_log.txt`: plain-English reasoning for every person.
- `summary.json`: FPS and per-stage timings.

### Perception (`detect/`, `pose/`, `track/`)
- One YOLO26n detector pass finds people and products.
- Body keypoints (COCO-17) come from a pose model run on each **person crop**, not on the whole frame. On one real frame with ~7 small, distant people, the full-frame nano pose model found 0 of them; the crop approach found all 7.
- People are tracked with ByteTrack. Products use a centroid tracker, because ByteTrack's box-overlap matching lost small items moving in a hand.

### Events (`events/engine.py`)
Readable rules, one per event:

| Event | Rule |
|---|---|
| `pick` | A product that was seen in a shelf/cooler zone moves into a hand that was just inside that zone |
| `put_back` | An item leaves the hand inside a zone |
| `conceal` | An in-hand item vanishes inside the shoulders-to-hips box, away from the register, and **stays gone for 1.5 s**. If it reappears, the conceal is cancelled |
| `pay` | The person dwells in the register zone |
| `exit` | The track ends at the door and **stays gone for 3 s** |

Zones are polygons in `configs/*.yaml`. The example single-camera gas station layout has 3 shelf runs, a cooler bank, a register and a door.

### Ledger (`ledger/ledger.py`)
- Keeps a basket per person.
- Credits POS receipts and cooler taps to the right person, first by **what the receipt contains**, then by who was at the counter.
- On exit: basket minus paid = unpaid, scored with an explicit evidence table.
- Tiers:
  - **alert** (tell staff): score ≥ 0.7, **and** at least one unpaid item was either seen being concealed or seen in hand at the exit.
  - **review** (a manager looks later; nobody is confronted): score ≥ 0.4.
- Every decision is explained in plain English.

### Payments (`ledger/payments.py`)
- Accepts JSON lines from a file, from stdin, or via `POST /payments` on a local port; a mock source for tests.
- A POS or tap reader only needs to send `{"terminal", "ts", "items":[{"sku","qty"}]}`.

## 2. Benchmark results

Sources:
- `results/bench.json` and `results/bench.md`, from a full `make bench` at the final code version.
- The event-level section was then re-run with `bench --only event`, after one change to how the threshold sweep counts capped decisions. The toy and real-footage sections come from the full run.
- `results/export.json` for ONNX.

Machine: 4-core x86_64 Xeon @ 2.8 GHz (from `/proc/cpuinfo`), no GPU.

### 2a. Theft logic on simulated shoppers (event-level simulator, no video)

**Setup:**
- 200 simulated store hours per row: 10,404 visitors, 495 of whom stole something (4.8%). The theft rate is deliberately higher than a real store's so recall can be measured.
- The reported seed (2) is **held out**: tuning used seed 1. The alert threshold (0.7) was not changed after seeing seed 2.
- "POS feed" means the register sends item lists.

| vision error rates | precision | recall (alert) | recall (alert or review) | false alerts / hour | false alerts / 1000 honest visitors | honest "review" flags / hour | baskets exactly right |
|---|---|---|---|---|---|---|---|
| perfect vision | 97.9% | 76.4% | 96.8% | 0.04 | 0.81 | 0.03 | 100.0% |
| **baseline (assumed)** | **72.5%** | **47.9%** | **83.4%** | **0.45** | **9.08** | **4.07** | **75.5%** |
| 2x the baseline error rates | 45.4% | 20.0% | 65.1% | 0.59 | 12.01 | 5.44 | 58.0% |

**Seed-to-seed spread** (seeds 2–6, POS feed, mean [min–max]):

| error rates | precision | recall | recall (alert or review) | false alerts / hour |
|---|---|---|---|---|
| baseline | 72.1% [67.0–75.5] | 50.2% [45.3–53.2] | 85.8% [83.4–88.3] | 0.46 [0.40–0.53] |
| perfect vision | 97.6% [96.6–98.5] | 77.8% [76.0–79.5] | — | 0.04 [0.03–0.07] |

Treat any single-seed number as **± ~3–4 points**.

**Without a POS feed** ("dwell" mode: standing at the register counts as paying for everything picked so far, except concealed items), at baseline:
- Recall drops to **33.1%** at alert level (59.6% counting reviews).
- Precision is 65.1%, with 0.44 false alerts per hour.
- A POS feed therefore gives **~1.45x the recall** at alert level (47.9 vs 33.1) and ~1.4x counting reviews. It is also the only way to catch "paid for one, pocketed another" openly.

**How to read the pieces:**
- **Precision and recall** are counted per person. A "thief" is anyone who left with something unpaid.
- **Threshold sweep.** Raising the alert threshold buys few false alerts back and costs a lot of recall. Precision stays at 70–73% between thresholds 0.6 and 0.85:

  | threshold | false alerts / hour | recall |
  |---|---|---|
  | 0.7 | 0.45 | 47.9% |
  | 0.85 | 0.20 | 21.6% |
  | 0.9 | 0.09 | 16.0% |

- **Evidence matters more than threshold.** At baseline:
  - Alerts where a concealment was seen were right 163 of 175 times (**93.1%**).
  - Alerts without one were right 74 of 152 times (48.7%).
  - An alert tier restricted to concealment-backed alerts would have caught 163 of 495 thieves (**~33% recall**), with 12 false alerts in 200 hours (**~0.06 per hour**, about 1.4 a day).
- **Recall by theft type** at baseline, alert level:

  | theft type | alerted | total | alert or review |
  |---|---|---|---|
  | walkout | 143 | 215 | 191 |
  | conceal + partial pay | 89 | 176 | 147 |
  | paid for one item, carried another out openly | 5 | 104 | 75 |

  The last type mostly lands in review only (70 of 104). That is by design: without concealment it looks the same as a missed put-back or a misread receipt.
- **Basket accuracy:** 75.5% of baskets exactly right at baseline (item-level precision 88.9%, recall 89.0%). 100% with perfect vision.
- **Decision latency** (from the person walking out to the alert, in camera time):
  - The median is **5.0 s**, which is the deliberate wait for late POS messages.
  - The 95th percentile is 34.5 s. By design, long waits come from holding a decision while a group mate or a crowded-pick candidate is still inside; I did not measure how much of the tail each accounts for.

**Which vision errors cause false alerts?** Same seed, 200 hours, perfect vision with one error type switched on at its baseline rate:

| error type | false alerts / hour |
|---|---|
| none | 0.04 |
| **missed register visit** | **0.24** |
| track ID split in two | 0.12 |
| two people's track IDs swapped | 0.10 |
| POS message dropped or clock off | 0.10 |
| crowded-pick misattribution | 0.06 |
| missed exit | 0.06 |
| all others | 0.03–0.04 |

- The single-error increases add up to about the whole baseline excess (≈0.41 per hour). So false alerts are mainly driven by **register-visit detection, track stability and POS integration**, not by product recognition.
- Recall is hurt most by **missed concealments** (76.4% → 58.4%) and **missed picks** (→ 68.5%).

**What this proves:** the reconciliation logic, payment attribution, hold-and-wait rules and scoring behave correctly on correct inputs. Payment attribution here covers queues, groups paying together, cooler taps, and late or missing POS messages. It also shows which vision errors to invest in.

**What it does not prove:** real-world accuracy.
- The baseline error rates are my guesses: missed pick 8%, missed put-back 15%, missed conceal 40%, missed register visit 3%, track ID split 3%, ID swap 1%. The full list is in `bench.json → noise_model_baseline`.
- The simulator also encodes my assumptions about how people shop and steal.

### 2b. Full pipeline on toy video clips

**Data:** 8 rendered 2D clips (`make demo`), clearly marked **TOY DATA**. The toy detector reads the **pixels**, not the renderer's ground truth.

| clip | truth | result |
|---|---|---|
| walkout | thief | alert |
| conceal + partial pay | thief | alert (flags the candy bar, not the soda they paid for) |
| normal pay | honest | nothing |
| put-back | honest | nothing |
| crowded cooler | honest | nothing (both picks flagged ambiguous; decision waited for the other person) |
| lingerer | honest | nothing |
| pocket then pay | honest | nothing |
| group, one person pays | honest | nothing |

- Speed: 11.1 FPS over 3,652 frames.
- Time from reading the frame that triggered an alert to having the alert JSON and clip written: 444 ms and 316 ms. This total includes clip encoding; I did not time the steps separately.

**Proves:** the pieces fit together, and the outputs, clips and logs are produced. **Does not prove:** anything about YOLO on real store footage. I wrote these clips alongside the rules they test.

### 2c. Real footage throughput

- **Data:** OpenCV's `vtest.avi`: real pedestrians, 795 frames, 768x576. **No theft labels.** The store zones don't match that scene, so this measures speed and tracking only.
- **Speed:** **9.66 FPS end to end** on 4 CPU cores. Per frame:

  | stage | ms |
  |---|---|
  | detect + pose | 92.6 |
  | tracking | 1.2 |
  | events | 0.8 |
  | ledger | 0.02 |
  | drawing | 7.3 |

  45 person tracks.
- **ONNX** (`results/export.json`, same CPU, measured with nothing else running; a first attempt made while the benchmark was running was discarded):

  | model | PyTorch | ONNX Runtime |
  |---|---|---|
  | detector | 43.8 ms | 46.1 ms |
  | pose (160 px crop) | 13.9 ms | 8.2 ms |

  It found the **same number of detections** on the one test frame (9/9 and 2/2). I did not compare the boxes and keypoints themselves.
- **No GPU numbers:** I had no GPU. Jetson/GPU speed is unmeasured.

### 2d. Adversarial review (independent reviewers tried to break it)

Three independent reviewers ran in parallel. One re-checked every number in this report against the result files. Two built concrete event sequences and frame sequences to break the ledger and the event engine. They found real holes; **all of the following were fixed, each with a regression test that fails on the old code:**

**Event engine — honest shoppers turned into "concealments" or picks:**
- A customer's **own** drink became a "concealment" after a half-second detection dropout, a tracker ID change, or a hidden wrist.
- A set-down item that was **still visible** was flagged as concealed.
- A re-acquired item counted as a second pick.
- The customer's own phone, or an item merely brushed past near a shelf, became a "pick".
- A 1.6 s occlusion at the door became a false exit.
- A passer-by inherited someone else's drink.

**Ledger — false alerts from several weak signals:**
- Two or three missed put-backs added up to an alert. Alerts now need concealment or item-in-hand evidence.
- A vision/POS category mismatch alerted a paying customer.
- A very low-confidence concealment counted in full.

**Ledger — thieves slipping through:**
- A thief who picked two items, pocketed one and put one back lost the pocketed one from their basket.
- A walkout thief could claim a stranger's unattributed receipt.
- Dwell mode cleared concealed items.
- A thief's own concealment still left a crowded pick at "ambiguous".
- A reused track ID hid the second visit.

**Crash and hang bugs:**
- A malformed register event crashed the ledger.
- A floating-point rounding bug at the 120 s hold deadline made the replay loop forever. It appeared on 2 of 6 simulator seeds; a loop guard now prevents any repeat.

**Known and not fixed** (listed so nobody is surprised):
- **Counter tie-break.** At the counter, a tie between a thief who arrived first and the honest payer goes to the thief.
- **Strangers entering together.** A stranger who enters within 4 s of someone is treated as their group, and their extra payment can cover that person's concealed item.
- **No retraction.** Once an alert is out, a receipt that arrives late, or a group mate who pays after 120 s, does not retract it.
- **Unbounded history.** The ledger keeps every person record for the life of the process. Fine for a day; needs rotation for a months-long deployment.
- **Handoffs.** An item handed to someone outside the group stays in the picker's basket.

## 3. What didn't work or got skipped, and why

- **No real shoplifting video was evaluated.**
  - The Kaggle (Simuletic, DCSASS), Mendeley, HuggingFace and Google Drive hosts are **blocked by this build environment's network policy** (HTTP 403).
  - Instructions are in `data/README.md`. `make data` fetches the Kaggle sets once the `kaggle` CLI is installed (`pip install kaggle`) and an API token is present.
- **PoseLift (real shoplifting, pose-only) was not used.**
  - The official data is on Google Drive, which is blocked here.
  - A helper agent found and cloned a third-party GitHub copy into `data/poselift/`. The automated safety check then blocked pulling data from that unofficial mirror.
  - So **no result uses it**, and `scripts/download_data.sh` doesn't reference it. You may want to delete `data/poselift*` and download the official copy instead.
- **Concealment classifier over pose sequences (Stage 5): skipped.** Without PoseLift there was no labelled real pose data, and training on my own synthetic poses would only learn my renderer.
- **Multi-camera:** hand-off between cameras by floor position is built and unit-tested (`track/multicam.py`). It uses homographies (a per-camera mapping from image pixels to floor coordinates) and no appearance features. It is **not wired into `bree run`**: I had no multi-camera footage to run it on.
- **Isaac Sim:** prep only, as asked (`src/bree/sim/isaac/`).
  - The IRA config is a draft that has not been loaded by a real IRA build. Its keys must be checked against the installed version.
  - Hand-object behaviours (reach, conceal) need custom animations; the README explains how.
- **Products:** COCO (the detector's training set) knows only "bottle", "cup" and "cell phone" among relevant items. A store-specific product model is needed before product-level results mean anything.
- **Bugs found and fixed while building:** listed in DECISIONS.md, "Bugs found and fixed".
  - Notably, a simulator queue bug combined with a naive "whoever reached the counter first paid" rule produced 7.5 false alerts per hour even with perfect vision.
  - It also turned up a non-reproducible simulation (a Python hash-ordering issue); there is now a test that runs the simulation under 3 hash seeds.

## 4. What I need from you

1. **Footage from the 2 pilot stores:** a few hours per camera, including busy periods, plus a still frame from each camera so I can draw zones.
2. **The POS transaction export** for the same hours (details in section 6). The ledger relies on it: without it, recall drops from 47.9% to 33.1% and partial-pay theft is invisible.
3. **A Kaggle API token** (`~/.kaggle/kaggle.json`), plus `pip install kaggle`. Or run `make data` somewhere with normal internet access.
4. **The official PoseLift download** (public Google Drive link in `data/README.md`).
5. **A GPU box, even briefly:** a Jetson Orin (the likely ship target) or any RTX machine, for real edge FPS and for Isaac Sim.
6. **Decisions:**
   - What may staff do on an alert? My recommendation: nothing customer-facing during the pilot.
   - Who reviews the "review" queue?
   - Clip retention: my suggestion is 7 days for flagged clips.
   - Today, alert evidence clips are only kept for flagged people. But the debug outputs `annotated.mp4` (heads pixelated) and `frames.jsonl` (boxes and keypoints) cover **everyone** unless run with `--no-video`. In production, those debug outputs should be off.

## 5. Honest take: top 3 things before this goes into the 2 gas stations

1. **Run in shadow mode first, then measure the real error rates.**
   - Record the pilot cameras and the POS for 1–2 weeks, with alerts shown to no one.
   - Hand-label a few hours: picks, put-backs, concealments, register visits, exits.
   - Put the measured rates into `VisionNoise` and re-run `make bench`.
   - Only then choose the live policy. At my assumed rates, 0.45 false alerts per hour (~11 a day) is too many to act on in front of customers.
   - The evidence split points to a sane first live mode: **alert only when a concealment was seen**. That was ~93% precise and ~0.06 false alerts per hour at ~33% recall. Everything else goes to a manager's review queue.
2. **Get the POS integration and camera placement right: that's where false alerts come from.**
   - Missed register visits, track-ID splits and swaps, and POS drops or clock drift are the top false-alert sources. Product recognition errors barely matter.
   - Concretely:
     - a camera with a clean view of the counter
     - a POS feed with timestamps synced to camera time (NTP on both)
     - a camera covering the door
     - a second camera on the cooler bank for crowded reaches and concealment
3. **Train a store-specific product detector and validate pose on the real cameras.**
   - COCO can't tell an energy drink from a candy bar.
   - Collect a few hundred labelled frames per camera of the operator's most-stolen items, plus Isaac Sim renders.
   - Check that wrist keypoints from the crop-pose approach hold up at the real camera height and resolution. Every pick, put-back and concealment rule depends on them.

## 6. What to ask the operator for

- **Cameras:**
  - make and model; resolution, FPS and codec
  - whether we can pull RTSP locally (and credentials)
  - mounting height and where each camera points
  - a still frame from every camera in both pilot stores
  - whether he's open to adding 1–2 cameras (counter-facing, cooler-facing)
- **Existing CCTV footage:**
  - a few hours per store, including rush hours and any known theft incidents (with rough timestamps)
  - how long his system keeps footage
- **POS:**
  - which system (e.g. Verifone Commander, Gilbarco Passport, NCR, Clover)
  - real-time export or batch only
  - fields: timestamp, register/terminal ID, line items with SKU/UPC and quantity, voids and refunds
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
  - cooler brand, and whether coolers are already card-tap / locked
  - number of registers
  - typical staff per shift
- **Policy:**
  - what staff may do when an alert fires
  - local rules on camera analytics and signage
  - who at his company signs off on privacy

---

- Design choices and their reasons: `DECISIONS.md`
- Build status: `PROGRESS.md`
- Raw numbers: `results/bench.json`, `results/bench.md`, `results/export.json`
