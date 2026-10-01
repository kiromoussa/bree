# BREE vision: report

Three parts: **Overnight (2026-10-01)** first, then **Phase 2 (2026-09-30, real data)**, then the **Phase 1 report** unchanged. Every number names the file it came from; real-data results and simulated results are kept apart.

# Overnight (2026-10-01, ~00:20 to ~06:20 EDT)

**Bottom line.** No GPU quota yet (ticket #2610010040000169 still open, limits 0), so everything ran on the Mac. New capabilities are in and tested (148 tests pass, including vision smoke tests), perception tuning and long-gap stitching gave mixed results, and **concealment from pose is still not solved**.

**New capabilities (merged)**
- **Late POS receipts can retract an alert** (`late_receipt_window_s`, default 300 s). Simulator, seed 2, baseline noise, 200 h, with the POS exporting in 60 s batches (new `VisionNoise.pos_batch_s` option; default 0 keeps the headline bench unchanged): without retraction **2.54 false alerts/h at 28.1% precision**; with retraction **0.435/h at 69.0%**; with `exit_grace_s: 65` plus retraction 0.455/h at 72.1% and 47.5% recall, about the live-POS baseline (0.45/h, 72.5%, 47.9%). Source: `results/late_receipts.json` (`scripts/late_receipts.py`). If the operator's POS exports in batches, this matters a lot.
- **Multi-camera stores feed one ledger** (`bree run` with several `--source`/`--store`, `multicam:` in shadow mode); per-camera floor homographies, no appearance features. Tested on synthetic per-camera streams only; a handoff bug that dropped returning shoppers was found and fixed.
- **ONNX runtime** (`--runtime onnx`) with the best ONNX Runtime provider present (TensorRT > CUDA > CoreML > CPU). With the same letterboxed input it gives the same detections as PyTorch within 0.0003 px (`results/onnx_parity.json`); on the pipeline path, padding to 640x640 shifts boxes (median 11 px on MERL). CoreML parity holds on the default compute units used in the speed run too (`results/onnx_parity_coreml_all.json`: same-letterbox boxes within 0.0003 px). **On this Mac, CoreML runs the full pipeline at 77.0 FPS with the small models vs 45.9 FPS for PyTorch MPS** (`results/speed.json`), about 5 camera streams at 15 fps instead of 3.
- **Isaac Sim 4.5 gas-station generator, ready to run** (`src/bree/sim/isaac/`, runbook in its README): store, cameras, behaviours, randomisation, truth files in our schema, converter, overlays, 8/1/1 split by scene seed for the full run (2,080 clips; pilot 52 clips, 11/1/1). Not executed (no GPU). It also fixed `scripts/azure_gpu.sh sim`, which could not have worked (current Isaac Automator only deploys Isaac Sim 5.x; now pinned to v3.13.0 for the 4.5.0 container) and closes the Automator's open VNC/NoMachine ports while restricting SSH to this IP. Isaac Sim 4.5 has no concealment animation, so these renders help detection, tracking and event timing, not concealment poses.

**Biggest gain tonight: pick detection and visit continuity** (event engine; choices made on MERL train videos, measured once on the 28 test videos)
| MERL test split | Phase 2 | + track stitching | + hand point + duplicate-box removal (new defaults) |
|---|---|---|---|
| reach recall (pick detection upper bound) | 64.2% | 64.2% | **87.6%** |
| false reaches / min | 0.78 | 0.78 | 1.07 |
| visits per single-shopper video | 5.8 | 3.1 | **2.5** |
| shoppers split into more than one visit | 100% | 79% | **68%** |
Sources: `results/merl_offline_train.json` (selection), `results/merl_offline_test.json` (test, current engine incl. the stitching ambiguity guard below). The first stitching run, before the guard, gave 3.0 visits and 82% split (`results/merl_stitch.json`). Stitching continues a visit when a new tracker id appears away from the door within 10 s and one body height of someone just lost (position and time only). The hand point is the wrist pushed half a forearm further (fingertips reach deeper than the wrist). Duplicate removal drops a person box lying 85% inside a larger one. Stitching can mis-merge different people; MOT16 (below) showed that in crowds, so stitching now skips ambiguous cases (two lost people qualify, or someone visible stands where the new track appeared).

**A missed-theft bug found on the way.** The put-back rule counted any hand in a shelf zone; a resting hand next to a gondola turned a concealment by the other hand into a put-back. Now only the holding hand counts (regression test fails on the old code).

**Simulator with the new measured rates** (`results/bench.json`, POS feed, 200 h, seed 2; measured pick detection 87.6%, visit splits 68%):
| vision noise | precision | recall (alert) | recall (alert + review) | false alerts / hour |
|---|---|---|---|---|
| assumed baseline | 72.5% | 47.9% | 83.4% | 0.45 |
| measured, Phase 2 rates | 24.2% | 13.5% | 37.4% | 1.05 |
| **measured, rates after tonight** | **28.6%** | **18.0%** | **53.3%** | 1.11 |
| measured tonight, ID switch at the assumed 3% | 54.9% | 28.1% | 79.0% | 0.57 |
More thieves are caught (alert + review recall 37% to 53%) at slightly more false alerts (1.05 to 1.11 per hour). Visit splitting is still the biggest drag: at the assumed split rate, false alerts would be 0.57 per hour. The toy clips are unchanged (2 of 2 thieves alerted, no flags on honest shoppers).

**Tracking on real angled footage with ground truth (MOT16 train, 7 sequences, 517 people; evaluation only; `results/mot16.json`).** Settings fixed before running:
| setting | MOTA | IDF1 | ID switches | fragmentations | recall | precision |
|---|---|---|---|---|---|---|
| old default (YOLO26s, conf 0.3, 2 s buffer) | 0.325 | 0.435 | 487 | 1,369 | 38.5% | 87.4% |
| **current default** (+ duplicate-box removal) | **0.327** | **0.438** | 454 | 1,357 | 37.9% | 88.6% |
| MERL-tuned (YOLO26n, conf 0.15, 5 s buffer) | 0.300 | 0.395 | 430 | 978 | 34.3% | 89.8% |
| current default + engine stitching, no guard | 0.328 | 0.388 | 290 | 1,390 | 37.9% | 88.6% |
| same, stricter distance (0.5 heights; exploratory) | 0.328 | 0.411 | 308 | 1,383 | 37.9% | 88.6% |
| **current default + stitching with ambiguity guard** | **0.327** | **0.441** | 441 | 1,359 | 37.9% | 88.6% |
Duplicate removal helps slightly on real angled footage; the MERL-tuned detector loses too much recall, which confirms not making it the default. Unguarded stitching cut ID switches by a third but merged different people in crowds (IDF1 0.438 to 0.388); with the ambiguity guard it is slightly better than no stitching (IDF1 0.441, 441 switches) while keeping most of the single-shopper gain on MERL. MOT16 is street scenes with many small pedestrians (hence the low recall); stitching here runs with no door zone.

**Pilot day-one tools** (merged): `scripts/draw_zones.py` (click zone polygons and multi-camera floor points on a camera still, video or RTSP frame; writes the store YAML, validated by loading it back) and **POS CSV import** through a declarative column mapping (`configs/pos_mapping_example.yaml`: columns, time format and timezone, terminal names, POS item names to our SKUs or categories, clock offset). `bree pos-convert` turns an export into our receipt format; with `--video-start` it adds stream time so recorded footage can be replayed with its POS export (`bree run --payments`). Shadow mode reads CSV exports straight from the POS folder. Rehearsed end to end on a toy clip with the sample CSV. **Pilot runbook:** `docs/PILOT_RUNBOOK.md` (before the visit, install day, daily review, what to measure before going live, privacy).

**Tried and dropped (negative results, recorded in DECISIONS.md):**
- Open-vocabulary product detection (YOLOE, text prompts, no training) to see a product in the hand: on MERL train videos it fired near the hand about as often with empty hands as with a product (best: 62% vs 50%; `results/merl_product_in_hand_train.json`). A product detector still needs our own labelled products.
- Long-gap stitching (30 to 120 s when the new track is within 0.25 to 0.5 body heights of where the shopper was lost): on the train cache, visits per shopper 1.58 to 1.50 but split shoppers 33% to 50%; left off.
- Unsupervised concealment scoring (distance to normal shopping poses, fit on PoseLift normal only): held-out AUC-ROC RetailS staged 0.531, DCSASS 0.481, UCF-Crime 0.642 (`results/conceal_knn.json`). Like the classifiers, near chance. Conclusion: public pose-only data does not give a concealment signal that transfers; concealment evidence should come from the product leaving the hand near the torso (needs a product detector) and from shadow-mode labels.

**Perception tuning** (chosen on MERL's train split, measured once on the test split; `results/tuning/`, `results/merl_measure_tuned.json`)
| MERL test split (28 videos, 64.3 min) | current default (YOLO26s, conf 0.3, 2 s buffer) | tuned (YOLO26n, conf 0.15, 5 s buffer, new-track 0.15) |
|---|---|---|
| reach recall | 64.2% | 62.7% |
| false reaches / min | 0.78 | 0.25 |
| extra track ids per single-shopper video | 11.0 | 7.3 |
| detector + pose + tracker, ms / frame (MPS; different sessions, same-session train split: 25.0 vs 24.1) | 30.4 | 23.3 |
On the train split the tuned setting had 67.7% vs 57.1% reach recall; that gain did not carry over to the test split. Fewer track splits did carry over (5.3 to 3.8 on train, 11.0 to 7.3 on test). False reaches went up on train (0.00 to 0.22/min) and down on test (0.78 to 0.25/min), so that change is not consistent. Defaults are unchanged; the trade-off is a choice to make on our own camera footage.

**Classifier v2** (features: wrist position relative to hips and torso, speed; augmentation: mirroring, speed change, keypoint dropout, jitter). Selected on PoseLift leave-one-camera-out only: AUC-ROC 0.681 vs 0.585 for v1 (`results/conceal_select.json`, 2 seeds; P2.3's 0.592 for v1 is the same protocol from a different seed run). Held-out sets, scored once (`results/conceal_variant.json`):
| AUC-ROC | v1 (shipped) | v2 + augmentation |
|---|---|---|
| RetailS staged | 0.501 | 0.504 |
| DCSASS (clip level) | 0.517 | 0.541 |
| UCF-Crime shoplifting | 0.671 | 0.685 |
Slightly better everywhere, still near chance on RetailS and DCSASS. v1 stays the shipped model; v2 is kept as `models/conceal_poselift_v2_aug2.pt`.

**Not done overnight:** YouTube or other unlicensed footage (terms of service, and the people filmed never consented); GPU work (no quota).

---

# Phase 2 (2026-09-30): real data

> **Superseded overnight:** the measured pick-detection and ID-switch rates and the simulator "measured" rows below are the 2026-09-30 values. `results/measured_error_rates.json` and `results/bench.json` now hold the overnight values; see the Overnight section above.

**Bottom line.**
- **No GPUs yet.** Azure GPU quota is 0; all 9 automatic quota requests were rejected (`ContactSupport`). Kiro opened support ticket **#2610010040000169** (East US: NC A100 v4 -> 24 vCPUs, NVads A10 v5 -> 72), status Open. So Isaac Sim, the A100 training and TensorRT speed did not run; everything else ran on the Mac (M1 Max).
- **First real measurements of our own perception** (MERL Shopping, real video, overhead camera): the pose model puts a wrist in the shelf zone for **64.2%** of 589 labelled reach instances (assumed pick detection: 92%), and every single-shopper video got more than one track id (mean 12.0) (`results/merl_measure.json`).
- **Concealment from pose does not work yet.** A classifier trained on real shoplifting poses (PoseLift, Apache-2.0) beats a pose-only rule on held-out PoseLift incidents (AUC-ROC **0.649** vs 0.550), but is **at chance** on every dataset it was not trained on: RetailS staged 0.501, DCSASS 0.517 (`results/conceal_*.json`). UCF-Crime 0.671 is the one exception and is weak evidence (see P2.3).
- **False triggers on real normal footage** (RetailS, 36.4 h of real shoppers, evaluation only): the classifier alone would fire **15.4 times per hour** at threshold 0.5 (5.5 at 0.9); the pose-only rule **99 per hour**. Pose-based concealment can only ever corroborate the ledger, never alert on its own.
- **Simulator re-run with the measured rates:** precision falls from 72.5% to **24.2%**, alert recall from 47.9% to **13.5%**, false alerts rise from 0.45 to **1.05 per hour** (POS feed, 200 h, `results/bench.json`). With the assumed ID-switch rate instead of MERL's overhead worst case: 42.9% / 19.6% / 0.65 per hour.
- **Shadow mode is built** (`bree shadow`), so the pilot can collect labelled data from our own cameras without showing staff anything. That data, plus Isaac Sim once quota lands, is the path forward.
- An independent adversarial review found real flaws in the first version of these numbers (incident leakage, empty frames inflating AUC, loose event counting). All were fixed and every result was re-run; see DECISIONS.md, "Adversarial review of the classifier".

## P2.1 What changed since last night
- `bree shadow` + `/review` page + `bree shadow-labels` (Phase 5; README "Shadow mode").
- `src/bree/conceal.py`: PoseLift/RetailS loaders, incident chains, gap splitting, pose-only rule, temporal CNN classifier, metrics, triggers. Scripts: `conceal_experiment.py`, `eval_retails.py`, `eval_dcsass.py`, `eval_ucf.py`, `measure_merl.py`, `measured_error_rates.py`, `speed.py`, `drive_fetch.py`, `azure_gpu.sh`, `phase2_rerun.sh` (re-runs every real-data result in order).
- `make bench` prints a **REAL DATA** section (read from the result files) before the **SIMULATED** section, and adds "measured" noise rows to the simulator.
- `data/README.md` (counts, formats, closeness to a gas station) and `data/LICENSES.md`.
- 98 tests, all passing (the Phase 1 report had 80).

## P2.2 Datasets
| Dataset | License | Used for | What it gave us |
|---|---|---|---|
| PoseLift (official Drive) | Apache-2.0 | **train + eval** | 151 clips (47 labelled) from 6 cameras; 36 incident chains contain labelled clips. Trains the shipped classifier. |
| RetailS | none stated ("academic use only" on an unfinished page) | eval only | 624 staged clips; 36.4 h sampled normal footage. Its real-world test set **is** the PoseLift test set, so not used. |
| MERL Shopping | research only | eval only | 28 test videos, 64.3 min, 589 labelled reach instances. |
| UCF-Crime (authors' Dropbox) | research only | eval only | 21 shoplifting test videos + 15 of 150 normal test videos (0.38 h) through our pipeline. |
| DCSASS (Kaggle) | research only (from UCF-Crime) | eval only | 896 labelled shoplifting-category clips (155 shoplifting) through our pipeline. |
| Simuletic (Kaggle) | CC BY 4.0 | not used | The free release is only 8 clips. |
| SKU-110K | research | not used | Download paused; no gas-station product labels to pair it with. |
Details and counts: `data/README.md`, `data/LICENSES.md`.

## P2.3 Real-data results (no simulator)
**Pose/track layer, MERL Shopping test split** (`results/merl_measure.json`; overhead lab camera; YOLO26s + crop pose + ByteTrack on MPS at 15 fps):
- Reach recall **64.2%**: 378 of 589 labelled "Reach To Shelf" + "Hand In Shelf" instances had a wrist in the shelf zone within +-0.5 s. An **upper bound** on pick detection: a pick also needs the product detected, and MERL's products aren't COCO classes.
- False reaches: **0.78 per minute**.
- Tracking: all 28 single-shopper videos got more than one track id (3 to 22 ids, mean 12.0, i.e. 11.0 extra ids per shopper). Straight-overhead views are hard for a COCO person detector.

**Concealment, PoseLift** (`results/conceal_poselift.json`). Folds hold out whole incident chains (5 folds x 3 seeds); frame metrics on the 3,721 frames with a pose (1,489 shoplifting):
| scorer | AUC-ROC | AUC-PR | EER | per-fold AUC-ROC mean [min-max] | AUC-ROC incl. empty frames |
|---|---|---|---|---|---|
| pose-only rule | 0.550 | 0.435 | 0.475 | 0.560 [0.37-0.74] | 0.673 |
| classifier | **0.649 +- 0.010** | **0.602** | 0.390 | 0.651 [0.49-0.81] | 0.746 |
- Per unique clip, mean over seeds, at threshold 0.5: classifier caught 10.3 of 41 shoplifting clips and triggered on 1 of 6 clean clips; the rule caught 31 of 41 and triggered on all 6 clean clips.
- Leave one camera out, AUC-ROC rule / classifier: cam 1 0.46 / 0.61, cam 2 0.69 / 0.68, cam 3 0.47 / 0.74, cam 4 0.64 / 0.71, cam 5 0.45 / 0.57, cam 6 0.69 / 0.24 (2 clips); mean 0.567 / 0.592.

**Concealment, RetailS** (`results/conceal_retails.json`; evaluation only; model trained on all of PoseLift; leak checks in DECISIONS.md found no shared or near-identical poses):
| scorer | staged AUC-ROC | AUC-PR | EER | staged clips caught (th 0.5) | triggers / hour, 36.4 h normal footage | person tracks (>= 2 s) triggered |
|---|---|---|---|---|---|---|
| pose-only rule | 0.618 | 0.575 | 0.405 | 514 / 622 | 99.2 (th 0.5) | 90.1% of 3,131 |
| classifier | **0.501** | 0.507 | 0.507 | 73 / 622 (11.7%) | **15.4** (th 0.5), 10.6 (0.7), **5.5** (0.9) | 13.9% (0.5), 4.9% (0.9) |
The staged set has only 2 clean clips, so staged "caught" counts say nothing about false alarms; the normal footage does.

**Concealment, DCSASS through our own pipeline** (`results/conceal_dcsass.json`; 896 clips, 155 shoplifting, from 28 UCF-Crime videos; clip score = max frame score):
| scorer | clip AUC-ROC | AUC-PR | clips triggered at 0.5: shoplifting / normal |
|---|---|---|---|
| pose-only rule | 0.412 | 0.144 | 30% / 43% |
| classifier | 0.517 | 0.184 | 0.6% / 2.7% |

**Concealment, UCF-Crime through our own pipeline** (`results/conceal_ucf.json`; 21 shoplifting test videos with the official temporal annotation, 32,272 frames with a person, 3,578 in the incident window; 15 normal test videos, 0.38 h):
| scorer | AUC-ROC | AUC-PR | EER | triggers / hour on normal videos |
|---|---|---|---|---|
| pose-only rule | 0.548 | 0.126 | 0.460 | 789 (th 0.5) |
| classifier | 0.671 | 0.199 | 0.367 | 7.9 (0.5), 2.6 (0.7), 0 (0.9) |
UCF's labels mark the whole incident window, not the concealment itself, and only 0.38 h of normal video was scored, so this is weak evidence; DCSASS, cut from the same source videos with clip labels, shows no signal.

## P2.4 Measured vs assumed vision error rates
From `results/measured_error_rates.json`. Only four have a real measurement; the other 13 simulator parameters stay assumed and say why.
| parameter | assumed | measured | source |
|---|---|---|---|
| pick detected | 92% | <= 64.2% | MERL reach recall (upper bound) |
| concealment detected | 60% | 11.7% | RetailS staged, classifier th 0.5 (never trained on) |
| false concealment | 3% per carried item | 13.9% per person track (per-shopper stand-in) | RetailS normal footage, classifier th 0.5 |
| visit's track splits | 3% | 100% | MERL, overhead camera (worst case) |

**Event-level simulator** (held-out seed 2, 200 h per row, `results/bench.json`; vision error rates are inputs):
| vision noise | payment feed | precision | recall (alert) | recall (alert + review) | false alerts / hour |
|---|---|---|---|---|---|
| assumed baseline | POS | 72.5% | 47.9% | 83.4% | 0.45 |
| **measured** | POS | **24.2%** | **13.5%** | 37.4% | **1.05** |
| measured, ID switch assumed | POS | 42.9% | 19.6% | 64.6% | 0.65 |
| assumed baseline | dwell only | 65.1% | 33.1% | 59.6% | 0.44 |
| measured | dwell only | 23.1% | 12.7% | 29.1% | 1.05 |

## P2.5 False alerts per hour: the number the operator will care about
- **From the simulator with measured rates: 1.05 false alerts per hour** (POS feed), about 25 a day in a 24-hour store; 0.65 per hour if tracking is as good as assumed. Source: `results/bench.json`, "measured" rows.
- **From real footage, concealment signal alone: 15.4 triggers per hour** on 36.4 h of real normal shopping (RetailS, classifier, th 0.5). Not the pipeline's alert rate (an alert also needs an unpaid item from the ledger), but it means concealment-from-pose would flag many honest shoppers.
- A real end-to-end false-alert rate needs our cameras, zones, products and POS: that is what shadow mode is for.

## P2.6 Speed per hardware
Measured on this Mac only, re-run 2026-10-01 with nothing else running (`results/speed.json`; 300 frames of a real 920x680 MERL video; detector + crop pose + ByteTrack + event engine):
| runtime | nano models | small models |
|---|---|---|
| PyTorch CPU (M1 Max) | 25.1 FPS | 16.0 FPS |
| PyTorch MPS (M1 Max GPU) | 49.3 FPS | 45.9 FPS |
| ONNX Runtime CPU | 28.6 FPS | 11.2 FPS |
| **ONNX Runtime CoreML** (`--runtime onnx`) | **89.5 FPS** | **77.0 FPS** |
- A100, A10 and TensorRT: **not measured** (no GPU quota yet).
- Store cameras run about 15 fps, so this Mac handles about 5 streams with the small models on CoreML (77.0 / 15 = 5.1), or about 3 on PyTorch MPS (45.9 / 15 = 3.1); more people per frame means more pose crops, so budget lower.
- (The first run on 2026-09-30, without CoreML, measured 41.2 / 40.5 FPS on MPS while other jobs were running; superseded.)
- **Recommendation (not measured on the device itself):** for the pilot, one Apple-silicon mini PC per station (Mac mini class, roughly $600 list) running `--runtime onnx` (CoreML); measure a Jetson Orin (roughly $250 to $500 list for Orin Nano / NX kits) with TensorRT once we have one, because it is the cheaper ship target. Prices are list prices from memory, to confirm before buying.

## P2.7 Azure resources used
None created. GPU quota is 0 pending the support ticket; no VM, disk or resource group exists, so spend is **$0**. `scripts/azure_gpu.sh` creates `bree-rg` (tagged `project=bree`), locks SSH to this machine's IP, sets auto-shutdown, and `stop` deallocates everything.

## P2.8 Still missing before the pilot
1. **GPU quota** (ticket #2610010040000169) -> Isaac Sim gas-station renders with labelled behaviours and products, A100 training, TensorRT speed.
2. **Footage from our own cameras**: operator stills + a few hours + matching POS export, or the staged session. Every measured rate above comes from someone else's store, camera angle or lab.
3. **A store-specific product detector.** Nothing measures product picks yet; MERL only bounds the reach half.
4. **Tracking on the real camera angle.** MERL's overhead angle broke tracking; an angled ceiling camera must be checked before trusting multi-minute visits.
5. **Concealment that generalises.** Train on synthetic concealment (Isaac Sim) + pilot labels from shadow mode; until then treat it as weak corroboration only. The first live policy should not depend on it.
6. RetailS license answer from the authors (email not sent yet).

---

# Phase 1 report (last night, unchanged)


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
- **No retraction.** Once an alert is out, a receipt that arrives late, or a group mate who pays after 120 s, does not retract it. (Update: a late POS receipt now retracts or downgrades it within `late_receipt_window_s`; see DECISIONS.md, Phase 2. A group mate paying after 120 s still does not.)
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
- **Multi-camera:** hand-off between cameras by floor position is built and unit-tested (`track/multicam.py`). It uses homographies (a per-camera mapping from image pixels to floor coordinates) and no appearance features. It is **not wired into `bree run`**: I had no multi-camera footage to run it on. (Update: now wired into `bree run` and `bree shadow`, tested on synthetic per-camera streams only, still no real multi-camera footage; see DECISIONS.md, Phase 2.)
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
