# DECISIONS

Every judgment call made while building this unattended, with the reason. Newest at the bottom of each section.

## Environment / hardware
- **Build machine**: Linux x86_64, Intel Xeon @ 2.80GHz, 4 vCPUs, 15 GB RAM, **no GPU** (no `nvidia-smi`, `torch.cuda.is_available() == False`, not Apple Silicon). Python 3.11.15.
- **Model sizes**: CPU-only -> **YOLO26 nano** (`yolo26n.pt` detection, `yolo26n-pose.pt` pose), the newest Ultralytics family available (ultralytics 8.4.166). `bree.hw.detect_hardware()` switches to the small (`s`) variants on CUDA or Apple MPS. Recorded by `make hw`.
- **Network**: PyPI and GitHub (incl. Ultralytics release assets) reachable. `kaggle.com`, `data.mendeley.com`, `huggingface.co` are **blocked by the egress proxy (HTTP 403)** in this build environment. Consequence: the Simuletic (Kaggle) and DCSASS (Mendeley) datasets cannot be fetched tonight; instructions for doing it manually are in `data/README.md`.
- **Packaging**: `pyproject.toml` with exact pins; `uv` for installs. Core (events/ledger/sim) depends only on numpy + pyyaml so the theft logic can be tested/installed without torch. Vision deps are the `[vision]` extra.

## Event + ledger design (Stage 3)
- **One pipeline, two products.** Vision emits a small vocabulary of store events (`enter, pick, put_back, conceal, pay, exit`). The ledger never looks at pixels. That makes the theft logic testable with the event-level simulator and keeps it debuggable: every person record keeps a human-readable audit log.
- **Payments are attributed by place and time**, not by identity. A POS receipt names a terminal; the store YAML maps terminal -> zone; the ledger credits whoever was standing in that zone at that time. Queue at the counter: prefer (1) standing there at exactly that time over "within 3 s slack", (2) hasn't already paid this visit, (3) got to the counter first. Cooler card/RFID tap: credited to whoever picked from that cooler closest in time (within 20 s).
- **Two payment modes.** `pos` (we get a POS feed: partial payment is detectable) and `dwell` (no POS feed: a register visit is assumed to pay for everything picked before it — only walkouts and items picked after the register visit can be caught). Pilot should push hard for the POS feed.
- **Items are matched by SKU when known, else by category.** Vision can realistically tell "energy drink can" from "candy bar", not Red Bull from Monster. The catalog maps every SKU to one category.
- **Payments clear the least suspicious item first.** If someone had two sodas, pocketed one and paid for one, the pocketed one is the unpaid one.
- **Evidence scoring, not a black box.** Per unpaid item: `0.5 x pick_conf` + `0.35` if concealed + `0.2` if visibly carried out + `0.15` if never went to the register; x `0.5` if the pick was crowded/ambiguous. Items combined by noisy-OR in the first version; replaced by max + small bonus per extra item (see Stage 4 below). `>= 0.7` = **alert** (tell staff now), `>= 0.4` = **review** (manager looks at it later, nobody is confronted), else dropped (kept in the debug log). Consequence by design: **a single uncorroborated pick can never trigger an alert** (max 0.5 + 0.15 = 0.65). Two or more high-confidence uncorroborated picks by someone who never visits the register can (0.65 + 0.05 per extra item), because a missed put-back looks exactly like that.
- **Hold decisions instead of guessing.** After exit, wait `exit_grace_s = 5 s` for late POS messages. Hold longer (up to 120 s) while (a) someone who entered within 4 s of this person (likely same party — one person often pays for the group) is still inside, or (b) someone who reached into the same cooler at the same moment (crowded pick) is still inside. Parties are reconciled with pooled baskets and payments; crowded-pick items are handed to the other candidate if that person paid for one extra.
- **Conceal without a seen pick** adds the item to the basket (at the conceal confidence), since the pick was evidently missed.
- **Put-back of an item not in the basket is ignored** (e.g. their own drink brought from the car).

## Perception (Stage 2)
- **Top-down pose instead of full-frame pose.** Measured on frame 150 of OpenCV's `vtest.avi` (real pedestrians, ~70 px tall): `yolo26n-pose` full-frame at 640 px found **0 of ~7** people (3 at 960 px); `yolo11n-pose` also 0; the `yolo26n` detector found 8. Running `yolo26n-pose` on each detected person's crop (padded 25%/10%, resized to 160 px) gave keypoints for **7/7** with mean wrist confidence 0.56-0.86. CCTV people are small, so the pipeline is: one detector pass (people + product classes together) -> pose on person crops -> ByteTrack.
- **ByteTrack = Ultralytics' `BYTETracker`, called directly** on detection arrays (not `model.track`), so the YOLO backend and the toy backend share one tracker. `supervision.ByteTrack` was the alternative but is deprecated in the pinned supervision version (removed in 0.31). Persons and products have separate tracker instances; lost-track buffer 2 s for people, 0.5 s for products.
- **Products from COCO are a stand-in.** COCO only has `bottle`, `cup`, `cell phone` that matter here. A store-specific product detector (categories in the store YAML) is the real plan; the backend takes a class->category map so it drops in.
- **Timestamps**: file input uses frame_index / fps (camera time); live input uses wall clock since start. The ledger runs on camera time.

## Privacy
- Heads are **pixelated** (5x5 blocks) in every image we write (annotated video, alert clips), located from nose/eye/ear keypoints, falling back to the top of the person box.
- Per-frame logs contain boxes, track IDs, and keypoints only. No crops, no embeddings, no face features. There is no re-identification across visits. (2026-10-03: body appearance features for within-visit re-ID now exist in memory only; still none on disk, none of the face, none across visits. See "Re-identification without the face".)
- Evidence frames live in memory only (2 s ring buffer + short snippets around a person's pick/conceal/put-back/exit). They are written to disk **only** for people who get an alert or review flag, and dropped as soon as a person is reconciled clean.
- **Products are tracked by centre distance, not ByteTrack.** Found on the toy clips: a 14x18 px item carried in a hand moves ~12 px/frame, IoU between frames ~0.2, so ByteTrack never confirmed the track and the walkout was missed entirely. `CentroidTracker` matches within the same category, gated at 3x the object's diagonal (min 30 px). People stay on ByteTrack.
- **A pick is credited to the reach of the hand that holds the item** (not the most recent reach of either hand). Found in tests: a shopper standing next to a gondola has their resting hand inside its polygon.
- **An in-hand item is only "released" if we kept seeing the person** for 0.5 s after the item left the hand. If the person vanished too (walked out the door), the item is still theirs and is reported as `held_items` on EXIT. Found in tests.

## Toy clips (Stage 3/4 end-to-end check)
- No store footage yet, so `bree render-toy` renders 8 scripted 2D scenarios in the example store layout (normal pay, walkout, conceal + partial pay, put-back, crowded cooler, lingerer, pocket-then-pay, group where one pays). Every frame says "TOY DATA".
- The toy backend detects from **pixels** (colour lookup table + connected components) and synthesises COCO keypoints from blob geometry + detected hands. It never reads the renderer's ground truth, so tracker/engine/ledger/alert code paths are exercised for real. It says nothing about YOLO accuracy on real footage.

## Simulator + evaluation (Stage 4)
- **Ground truth and vision noise are separate layers.** `WorldBuilder` generates what really happened (shoppers, a FIFO register queue, POS receipts, cooler taps, crowded cooler reaches). `observe()` turns it into what the camera would report using `VisionNoise` error rates. **Those rates are assumptions** (missed pick 8%, missed put-back 15%, missed conceal 40%, missed register visit 3%, ID switch 3%, ID swap 1%, POS drop 1%, ...), so every result is reported at perfect / baseline / 2x error rates, plus a one-error-at-a-time ablation.
- **Held-out seed.** All development and tuning used simulator seed 1. `make bench` reports seed 2.
- **Traffic**: 40 visits/hour average, 20% of hours at 2x (rush); ~4-5% of visitors steal (higher than a real store so recall is measurable with a few hundred thieves; false alerts/hour are driven by honest traffic, which is realistic). First version used 3x rush, which overloaded the single register (5-minute queues); 2x is more realistic.
- **Bug found by the simulator (fixed)**: the first queue model served people in session-generation order instead of arrival order. That, plus the ledger's "first to arrive at the counter pays" rule, produced 7.5 false alerts/hour *with perfect vision*. Both fixed: FIFO by arrival in the sim, and content-based receipt attribution in the ledger (below).
- **Receipt attribution is content-first**: among people at the counter when the receipt printed, the one whose party's unpaid picks best match the receipt's items. If the receipt matches nobody and several people are there, it is *not* guessed; it stays unassigned and is claimed at reconciliation by basket match (>= 50% of receipt items), or as a last resort by "stood at the register while it printed". Cooler taps use the same content match, then time. This took perfect-vision false alerts from 7.55/h to 0.04/h and cut the "missed register visit" false alerts by ~3x.
- **Confidence combines items by max + small bonus, not noisy-OR.** Measured on the dev seed at baseline noise: noisy-OR 0.60 precision / 0.55 recall / 0.90 false alerts per hour; max+0.05/extra item (cap 0.1) 0.71 / 0.52 / 0.53. Reason: unpaid items in one basket share one failure mode (a payment-matching error leaves all of them unpaid), so they are not independent evidence.
- **"Went to the register but no receipt matched" is downweighted (x0.6)**: more likely a POS gap than theft.
- **Alert threshold stays 0.7** (sweep in results/bench.md). The sweep shows precision plateaus ~70% under baseline noise; raising the threshold mostly costs recall.

## Stage 5
- **Multi-camera**: floor-plane homographies + position/time handoff (`track/multicam.py`), no appearance features (privacy). Library + unit tests; not yet wired into `bree run` (no multi-camera footage to test on).
- **Concealment classifier over pose sequences: skipped.** The only labelled real pose data reachable tonight (PoseLift) came via an unofficial mirror, which the automated safety check blocked; training on our own synthetic poses would only learn our renderer. Needs the official PoseLift download (data/README.md) or pilot footage.
- **Edge export**: ONNX export of both models works and gives identical detections to PyTorch on the test frame (results/export.json). Measured on this x86 CPU with nothing else running: detector 43.8 ms PyTorch vs 46.1 ms ONNX Runtime; pose (160 px crop) 13.9 ms vs 8.2 ms. (An earlier measurement showing ONNX much slower was taken while the benchmark was running concurrently, and was discarded.) The pipeline still runs PyTorch; ONNX is the hand-off format for TensorRT (Jetson) / NPUs, where engines are built on the device.
- **Dashboard**: stdlib HTTP server on 127.0.0.1, polls JSON once a second; nothing leaves the machine.

## Bugs found and fixed (for the record)
- Products lost by ByteTrack when carried in a hand (IoU too low) -> centroid tracker.
- Pick credited to the wrong hand's reach -> reach of the holding hand.
- In-hand item "released" when the person walked out of view -> release requires the person still visible.
- **Infinite loop in replay on a second EXIT for an already reconciled track** (possible after an ID swap): it was re-added to `pending` forever. Fixed: second exit ignored and logged; reconciled records dropped from `pending`. Regression test `test_second_exit_after_reconciliation_is_ignored_and_replay_terminates`.
- Simulator register queue served in generation order -> FIFO by arrival.
- **Non-reproducible simulation**: a zone was drawn from `list(set(...))` of strings, whose order depends on PYTHONHASHSEED. Fixed with `sorted(...)`; regression test runs the sim under 3 hash seeds.
- ONNX timing first measured while the benchmark was running: discarded, re-measured in isolation.

## Changes from the adversarial review (end of night)
Three independent reviewers: one fact-checked REPORT.md against the result files, two built concrete event/frame sequences to break the ledger and the engine. Every fix below has a regression test that fails on the previous code.
- **Alert tier requires corroboration**: at least one unpaid item that was concealed or seen in hand at the exit, and more such items than paid-but-unmatched receipt lines (those suggest a vision/POS category mismatch). Otherwise the decision is capped at review. Reason: 2-3 missed put-backs, or a soda read as "water", were reaching alert. After all ledger review fixes, dev seed 1 at baseline noise, 200 h: 72.5% precision, 49.4% recall, 0.47 false alerts/hour (before the fixes, a 100 h dev-seed run gave 70.6% / 51.6% / 0.53; not the same run length, so only roughly comparable).
- **Concealment confidence is used**: detections under 0.3 are ignored; the conceal weight scales with confidence up to 0.6.
- **A put-back removes a non-concealed item first** (pick two, pocket one, put one back: the pocketed one stays).
- **Receipts claimed by basket match can't clear concealed items**, and basket-match claiming only looks at the party's open items (a walkout thief could otherwise claim a stranger's receipt). Receipt lines with SKUs missing from the catalog cover an open item of any category.
- **Dwell mode never counts a concealed item as paid.** If the person who was credited with a crowded pick then conceals it, the pick is no longer treated as ambiguous. A re-used track id starts a new visit.
- **Engine: concealment needs the product to be gone**: a released in-hand item becomes CONCEAL only if its track is not seen again for 1.5 s and no same-category item reappears in that person's hand (else it's cancelled). A released item that is still visible is a set-down, never a conceal. Items the person never picked here (their own drink, their phone) never produce CONCEAL.
- **Engine: a PICK needs the product itself to come from the zone**: seen in (or within 25 px of) the reached zone shortly before, and displaced >= 15 px. A track this person already picked coming back is a re-acquisition, not a second pick.
- **Engine: missing wrists mean "unknown", not "released"**; a handoff needs 5 frames near the other person's hand with the owner's hands visibly elsewhere; EXIT needs 3 s of absence after the door (brief door occlusions happen); a person lost inside the store and seen again resumes the same visit.
- **Replay hang fixed**: `t_exit + max_hold - t_exit` can round just below `max_hold`, so a decision was never "ready" while its deadline equalled `now` (seen on 2 of 6 seeds). Epsilon + a no-progress guard in the replay loops.
- **Reporting**: the ablation now runs at the same 200 h as the main table; a 5-seed spread is reported next to single-seed numbers; the threshold sweep applies the same corroboration cap as the ledger.
- **Known, not fixed**: counter tie-break can favour a loiterer who arrived first; co-entry within 4 s is weak evidence of a group; no alert retraction when a late receipt arrives; `Ledger.people` grows for the process lifetime; handoffs to non-group people aren't moved between baskets.

## Phase 2
- **Shadow mode reuses `run_pipeline` per camera** with no staff-facing output: no console alerts, no live dashboard, no annotated video, no per-frame log. The only images written are the ledger's existing head-pixelated alert clips (now H.264 when OpenCV has it, so browsers play them; mp4v otherwise).
- **Every alert and review decision is logged**, not just alerts: review-tier labels tell us whether the 0.4 to 0.7 band is worth anything.
- **Alerts now carry `basket` and `audit_log`** (the ledger's reasoning lines) so a reviewer sees why without opening `ledger_log.txt`.
- **POS export folder is polled, not watched** (stdlib only, every 2 s). A line is read once it ends in a newline or the file stopped growing between two scans. Files present at start are skipped (old receipts). Each camera only gets receipts for terminals in its own store config.
- **Labels are append-only JSONL** with reviewer name and time; the latest label per alert wins, so a changed mind is kept in history. Precision leaves "unsure" out.
- **Raw recording is off by default** and prints a warning when on: raw segments contain faces, are rolled every few minutes, and deleted past the retention window. The review page serves only alert clips inside the output folder.
- **Known limits (superseded overnight 2026-10-01: multi-camera and late-receipt retraction now exist, see below)**: cameras are reconciled independently (multi-camera handoff is still not wired); a receipt later than `exit_grace_s` does not retract a would-be alert (raise it for batched POS exports); people still inside when shadow mode is stopped are not reconciled.
- **No GPU tonight.** GPU quota was 0 in every region and the automatic requests came back `ContactSupport` (needs a human support ticket). Everything that doesn't need an NVIDIA GPU ran on the Mac (M1 Max, MPS). Isaac Sim (Phase 3) did not run; `scripts/azure_gpu.sh` is ready for when quota exists. A100/TensorRT speed not measured.
- **Drive downloads**: `gdown --folder` stalls on Drive's rate limit; `scripts/drive_fetch.py` lists folders via `embeddedfolderview` and pulls each file from `drive.usercontent.google.com`, which worked. UCF-Crime's official page (visionlab.uncc.edu) no longer resolves; per-file links from the authors' Dropbox folder were used, only Part 4 (Shoplifting) and the testing normal videos (bandwidth ~3 MB/s).
- **RetailS real-world test = PoseLift test.** Same 47 clips, identical keypoints (checked). A PoseLift-trained model is never evaluated on it. RetailS normal-footage files are fingerprinted against every PoseLift pose frame and dropped on any match.
- **Concealment classifier**: small temporal CNN (2 conv layers, 24-frame = 1.6 s causal window) over keypoints normalised by a keypoint box (RetailS ships no detector boxes, so features never use them). Trained on PoseLift only (Apache-2.0). PoseLift's own train split is all normal, so supervised training uses labelled test clips; results are 5-fold CV grouped by incident chain (x3 seeds; see the adversarial review entry below) plus leave-one-camera-out, never scoring a clip the model trained on. Windows subsampled every 3rd frame for training (neighbours are near-duplicates); trained on MPS (7x faster than CPU).
- **Baseline for comparison is a pose-only rule** (share of the last 1.6 s with a wrist in the engine's torso box). The engine's real CONCEAL rule needs product tracks, which pose-only datasets don't have, so it can't be run on them.
- **PoseLift has NaN keypoints** (38,233 values): treated as undetected (conf 0).
- **Thresholds fixed up front** (model 0.5/0.7/0.9, rule 0.25/0.5/0.75; trigger = 0.5 s over threshold on one track, merged within 10 s) and never tuned on an evaluation set.
- **MERL measures the pose layer only**: its products aren't COCO classes, so reach recall is an upper bound on pick detection. Shelf zone drawn on a training-split frame; only test subjects 27-41 scored.
- **Measured error rates replace assumed ones only where measured** (`scripts/measured_error_rates.py`); `make bench` prints a "measured" noise row next to the assumed ones, in a REAL DATA section separate from SIMULATED.
- **Product detector not trained.** The plan was synthetic gas-station renders + SKU-110K. Without Isaac Sim there are no gas-station product labels, and SKU-110K alone is one class ("object") under a research-only release, so a model trained on it could not ship and would not tell soda from candy. Partial SKU-110K download kept at `~/datasets/.SKU110K_fixed.tar.gz.*.part` (resume with `curl -C -` from `http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz`).
- **Simuletic (Kaggle, CC BY 4.0) is only a free sample**: 8 synthetic clips (4 shoplifting, 4 normal; 145-241 frames, 544x544 @ 24 fps), 456 frames with YOLO person boxes + 17 keypoints, VLM captions. Too small to train on (4 theft clips vs PoseLift's 41), so not used for training; kept for a qualitative check only. The full dataset is sold by Simuletic.
- **NMS time-limit warnings during extraction**: YOLO's NMS stops after 2 s per call. Checked in ultralytics/utils/nms.py: each image's boxes are saved before the time check, so the single-frame detector call loses nothing; the pose call batches person crops, so a timeout can drop keypoints for the remaining people in that one frame (same effect as a missed pose). Seen 1-4 times per run of hundreds of clips. Out of caution the DCSASS clips and UCF videos extracted while two perception jobs overlapped were deleted and re-extracted one job at a time; rule going forward: one perception job at a time.
- **Adversarial review of the classifier (2026-09-30) and what changed:**
  - *Incident leakage in PoseLift CV*: clips are consecutive cuts of the same recordings (clip N's last pose is 2-10 px from clip N+1's first pose, vs 339 px median for random pairs), so one incident could sit in train and test. Folds are now built from **incident chains** (`bree.conceal.incident_chains`: same camera, consecutive ids, boundary pose within 20 px) and a held-out chain's Train clips are dropped from training too. Leave-one-camera-out (no such leak) is reported next to it.
  - *Empty frames scored 0 for free*: 1,080 of 3,312 negative frames had no pose. Frame metrics are now on frames with at least one pose; all-frame AUC kept for reference.
  - *Loose event recall*: +-15 frame padding on ~32-frame clips made "hit" nearly "fired anywhere". Now a clip is caught only if some person's score stays over the threshold for 0.5 s on a labelled frame (no padding, no merging; the first version used merged triggers, which made the count non-monotone in the threshold, caught on the rerun), counts are per unique clip (not x3 seeds), and are always shown next to the clean-clip trigger rate.
  - *Simulator concealment rate came from in-distribution CV*: `p_conceal_detected` now comes from RetailS staged (same store, never trained on). `p_false_conceal` is described as a per-track stand-in (a shopper can be several tracks; the simulator draws it per carried item).
  - *Track gaps*: tracks are split where frames are missing (`split_gaps`), so windows, velocities and trigger runs never span a gap. Trigger length is 0.5 s at the video's real fps (UCF 25 fps sources run at 12.5 fps).
  - *RetailS vs PoseLift leak*: a tolerant nearest-neighbour check (34-d keypoints) found no RetailS normal-footage pose within 10 px of any PoseLift pose (closest 0.01%: 26.7 px) and no staged pose within 5 px; the exact fingerprint catches all 4,433 frames of the known duplicate (RetailS real-world set). No leak.
  - *Pre-registration*: window (24), architecture, 20 epochs, the 0.5 / 0.7 / 0.9 thresholds and using 0.5 for the simulator were set before any evaluation result was seen; the only change after seeing results was fixing bugs (NaN keypoints, CPU speed, the items above). This can't be proven from history (the result files were overwritten), so the 0.9 values are also reported.
- **All real-data results re-run in order** by `scripts/phase2_rerun.sh` after these fixes.
- **ONNX runtime path (`--runtime onnx`, default stays `pytorch`).** Ultralytics' ONNX backend picks CUDA (device cuda) or CoreML (device mps) and never TensorRT, and with an accelerator device it also runs pre/post-processing there. So Ultralytics keeps letterboxing and decoding on the CPU and `bree.edge.ort.load_yolo` swaps its ONNX Runtime session for one with our providers: TensorRT > CUDA > CoreML > CPU, whichever are present, the rest as fallback. A thin ORT wrapper would have meant re-implementing letterbox + decode for two tasks; the swap is ~5 lines but depends on Ultralytics internals (`predictor.model.backend.session`, checked on 8.4.166).
- **Static exports, batch 1** (detector 640x640, pose 160x160), what TensorRT, CoreML and NPUs run best. The pose model is called once per person crop on this path; the old `scripts/speed.py` ONNX row passed all crops in one call and would have crashed on any frame with two or more people.
- **CoreML uses MLProgram**: all 368 detector / 401 pose nodes run in CoreML; the older NeuralNetwork format splits the graph into 7 CoreML partitions plus CPU nodes.
- **Parity (results/onnx_parity.json, yolo26s, 4 frames each of vtest.avi and MERL 27_1)**: with the same input, ONNX Runtime matches PyTorch to 0.0003 px on boxes and 0.0001 px on keypoints, CPU and CoreML alike. On the pipeline path the detector differs more (vtest: 1 of 25 people unmatched, box max 7.8 px, median 0.2 px; MERL: box median 11 px, conf up to 0.17 on a half-visible person) because PyTorch letterboxes to the frame's aspect (640x480) and the static graph pads to 640x640; the same happens to pose on single-person frames. Exporting the detector at the camera's aspect would remove it; not done (the camera resolution isn't known at load time). CoreML was checked with `MLComputeUnits=CPUAndNeuralEngine` because the GPU was busy; GPU (`ALL`, the default) parity is not checked yet.
- **Concealment classifier ONNX** (`models/conceal_poselift.onnx`, dynamic batch, legacy TorchScript exporter because the dynamo one needs onnxscript): normalisation and sigmoid are inside the graph, so edge code needs only `track_features` + `windows` (numpy). Max score difference on 656 windows from 6 PoseLift test clips: 1.8e-7.
- **Late POS receipts retract or downgrade a decision** (was "Known, not fixed" and a shadow-mode known limit). A receipt that arrives after an alert/review decision goes through the same attribution as on time (place/time + content first, then the basket-match claim, which still can't clear a concealed item). The party is re-scored; if the tier drops, the ledger emits an `Alert` with `retracts` = the original id and tier `review` (downgrade) or `retracted`, and logs why. It **never raises** a tier after the fact (the person has left; a late escalation would only be noise). Open for `late_receipt_window_s` (default 300 s, 0 = off) after the exit; reconciled people are kept in the working set that long (was 204 s). Shadow mode appends the retraction to `would_be_alerts.jsonl`; the review page folds it into the original card (RETRACTED / DOWNGRADED, with the reason). Shadow precision still counts the tier as first decided, since that is what staff would have been told; `retracted` is counted separately. Regression tests in `tests/test_ledger.py` and `tests/test_shadow.py`.
- **Simulator option for late receipts: `VisionNoise.pos_batch_s`** (POS exports every N s; a receipt reaches the ledger at the next boundary via `Payment.t_received`). Default 0, so `make bench` / `make sim` are unchanged. The default sim never produces late receipts (a receipt prints 5 s or more before the exit, the decision waits `exit_grace_s` = 5 s more, POS jitter sd 1 s), so retraction has no effect there: seed 2, baseline noise, 200 h gives the same 327 alerts, 72.5% precision, 47.9% recall, 0.45 false alerts/hour with or without it; honest reviews 814 vs 815 (one group-mate lookup changed because reconciled people are now kept 360 s instead of 204 s). With batched exports, same seed / noise / hours (precision, recall, false alerts/hour, honest reviews/hour):
  - 60 s batches, no retraction: 28.1% / 40.0% / 2.54 / 14.1. With retraction: 69.0% / 39.2% / 0.435 / 3.71 (2,535 retractions).
  - 300 s batches, no retraction: 21.5% / 31.5% / 2.85 / 18.3. With retraction: 65.5% / 31.1% / 0.405 / 3.52 (3,482 retractions).
  - 60 s batches with `exit_grace_s: 65` alone: 63.4% / 47.9% / 0.685 / 4.29. Both together: 72.1% / 47.5% / 0.455 / 4.04, about the live-feed numbers.
  - Reading: retraction removes the false alerts that late receipts cause; it does not bring back recall, because "never raise" keeps a partial payer (whose receipt was late, so the decision took the no-receipt discount) at review. For batched exports set `exit_grace_s` to the batch delay and keep retraction for the stragglers.
- **Multi-camera handoff wired into `bree run` and `bree shadow`.** Config: each camera keeps its own store YAML (zones in its own pixels) plus `camera.floor_points`, 4+ `[x_px, y_px, x_m, y_m]` marks on one shared floor plan; `bree run` takes repeated `--source/--store` pairs, the shadow YAML takes `multicam: true` (or `{max_dist_m, max_gap_s}`). Single-camera configs need no change and produce byte-identical outputs on a toy clip, apart from the new `retracts` field. `StoreEvents` (in `track/multicam.py`) runs one EventEngine per camera and remaps events to global ids; `run_store` processes all cameras' frames in camera-time order (live cameras share one wall clock; recorded files are assumed to start together) into ONE ledger built from `merge_stores` (every camera's zone names, terminals and catalog; a zone name must have one kind everywhere). Evidence clips merge every camera's snippets in time order. Per-camera files: `frames_<cam>.jsonl`, `annotated_<cam>.mp4`. Still no appearance features.
- **Bug found while wiring (fixed)**: `MultiCamIdentity` excluded every global id a camera had ever held from matching in that camera, so a shopper who left the door camera's view and came back to it to leave became a new person, and the real visit never reconciled. Now only ids with a live local track in that camera (seen within `max_gap_s`) are excluded. Regression tests: `test_coming_back_into_a_camera_keeps_the_global_id`, plus a three-camera store (door, cooler, register) driven by synthetic per-camera FrameObs through the real engines and one ledger (paid visit is clean, walkout is one flagged visit), and `bree shadow` with `multicam:` end to end with a fake video source and tracker.
- **Multi-camera known limits**: tested only on synthetic streams (no multi-camera footage). Handoff is position + time only, so views must overlap or touch; someone who reappears more than `max_gap_s` later or `max_dist_m` from their last floor position becomes a new person (the visit splits: the picking id never exits and is never reconciled, the exiting id has an empty basket), and two people crossing in a blind spot can swap. One camera failing to open fails the whole store session (it reconnects together). Identity maps grow for the process lifetime, like `Ledger.people`.
- **Isaac Sim (Phase 3) is ready to run, not run** (no GPU quota). Code in `src/bree/sim/isaac/` (`plan.py`, `gen_cstore.py`, `convert.py`); runbook, first-run checklist and verified-vs-assumed table in its README.
- **Isaac Sim 4.5.0 container via Isaac Automator v3.13.0.** v3.13.0 is the last Automator that deploys the container (`--isaac-image nvcr.io/nvidia/isaac-sim:4.5.0`) and it installs the Azure GRID driver 535.161.08 (4.5 needs >= 535.129.03). Automator v4.x installs Isaac Sim from GitHub source tags, which start at v5.0.0, so the previous `--isaacsim 4.5.0` command could not have worked; `scripts/azure_gpu.sh sim` was rewritten. v3.13.0's terraform opens SSH/VNC/NoMachine to `*`, so the script replaces those rules with SSH from this IP, tags the resources and sets auto-shutdown. `--resource-group` takes the RG resource ID and imports it into terraform state: never `./destroy` bree-sim.
- **Sources for "verified"**: the 4.5.0 docs are gone from docs.isaacsim.omniverse.nvidia.com (404), so they are cited from the Wayback Machine; the 4.5.0 extension code (IRA core 0.5.14, omni.anim.people 0.6.7, omni.anim.graph.core 106.5.1, omni.replicator.core 1.11.35) was read from NVIDIA's `isaacsim-extscache-kit` 4.5.0.0 wheel, and the character/animation assets were inspected with usd-core.
- **IRA used as a library, not through its YAML + `sdg_scheduler.py`.** Product actions need a per-frame hook synchronised with each character's commands, which the scheduler does not offer. The script calls the same setup functions IRA does (Biped_Setup + anim graph, behaviour scripts, NavMesh exclude), drives people with omni.anim.people command files, and reads annotators directly (the documented 4.5 "custom FPS" pattern: render products enabled only on capture frames). The old draft `ira_gas_station.yaml` / `commands_mixed_episode.txt` were deleted.
- **Hand-object interaction**: reach = the `PushButton` custom command (4.5 sample animation; forward kinematics on it: right wrist 0.5 m forward, 1.4 m high, peak ~1.2 s); the product action is keyed to omni.anim.people's `CommandStartEvent`; the product follows the hand joint (`omni.anim.graph.core` `get_joint_transform`); conceal = slide to a body anchor and hide. No concealment/crouch/left-hand animation exists in 4.5, so pose does not show concealment: train the conceal classifier on PoseLift, use the renders for boxes, tracks, product state and event timing. Only products 1.0-1.6 m high are picked.
- **Primitives for products and fixtures**: the 4.5 asset pack and SimReady have no c-store merchandise. Semantic labels = item categories from `configs/store_gas_station_small.yaml`; SKUs on receipts come from its catalog.
- **Clip = one camera of one episode; split by scene seed % 10 (8/1/1)**, each scene recorded as 4 takes in the full run (130 scenes, 520 episodes, 2,080 clips); pilot 13 scenes = 52 clips. Truth labels come from the event log of what actually happened, with `matches_plan` per person, not from the plan.
- **Calibration knobs instead of guesses** where the docs are silent: `--facing-offset`, `--grasp-at`, `--light-scale`/`--panel-intensity`; each has a measured diagnostic in `meta.json` (facing error, reach peak time, hand-to-product distance, brightness) and `convert.py check` re-projects 3D joints to validate the camera matrices before zones are trusted.
- **`toy_eval.run_toy_suite` takes a clip prefix, a backend factory and per-clip store configs**; people who never exit (staff) are skipped in matching.
- **Overnight perception tuning (MERL TRAIN split only, 6 videos)**: medium models split tracks least (2.8 extra ids/video vs 5.3) but nano caught the most reaches (67.7% vs 57.1%) and is fastest. Chosen before touching the test split: nano detector + nano pose, person conf 0.15, 5 s track buffer, new-track threshold 0.15 (67.7% reach recall, 3.8 extra ids, 24.1 ms/frame on train). Reach recall ranked first because it caps pick detection. Run once on the test split: results/merl_measure_tuned.json.
- **Classifier variant selection (PoseLift leave-one-camera-out only, 2 seeds)**: mean AUC-ROC v1 0.585, v1+augmentation 0.646, v2 features 0.614, **v2 features + augmentation 0.681** (winner). Evaluated once on RetailS staged, DCSASS and UCF next to v1: results/conceal_variant.json. v1 stays the shipped model unless the held-out sets agree.
- **Track stitching in the event engine (overnight).** A new tracker id appearing outside the door zone within `stitch_s` (10 s) and `stitch_dist` (1 body height) of someone lost inside the store continues that person's visit (position and time only). Without it a visit splits: the first half never exits, so its basket is never reconciled. Defaults set before measuring. MERL test split (`results/merl_stitch.json`): visits per single-shopper video 5.8 -> 3.0; videos still split 100% -> 82%. One video got worse (4 -> 7 visits), so it can mis-merge. Never stitches onto someone visible in the same frame.
- **Hand point, duplicate-box removal (overnight).** Chosen on 12 MERL train videos from cached tracks (`scripts/merl_offline.py`, `results/merl_offline_train.json`) with a rule stated before looking (reach recall within 1 point of the best, then fewest visits, then fewest false reaches): reach point = wrist + 0.5 x (wrist - elbow) (fingertips go deeper into a shelf than the wrist), and drop person boxes >= 85% inside a larger one. Run once on the 28 test videos (`results/merl_offline_test.json`): reach recall 64.2% -> 87.6%, false reaches 0.78 -> 1.07 / min, visits per shopper 3.04 -> 2.36, videos split 82% -> 64%. Now the defaults (`EngineRules.hand_extend = 0.5`, `YoloBackend(dedupe_inside=0.85)`). The k grid stopped at 0.5, the best value; larger k was not tried.
- **Put-back uses only the holding hand (bug found by the hand-point change).** The rule "item released while a hand is in a shelf zone" used ANY hand: a resting hand inside the neighbouring gondola's polygon turned a concealment by the other hand into a put-back (a missed theft). It also existed with wrists (test reproduces it with hand_extend = 0). Fixed: only the hand nearest the item counts. Regression test fails on the old code.
- **Long-gap stitching tried, kept off.** The 12 splits left on the train cache reappear at almost the same spot (median 0.09 body heights) after 10-50 s (overhead detector drops a shopper standing at the shelf). A second tier (`stitch_long_s` 30 / 60 / 120 s, `stitch_near` 0.25 / 0.5 body heights) for long gaps at near-identical positions cut visits 1.58 -> 1.50 but raised the split share 33% -> 50% on train, so it is not a clear win and defaults to off. Not run on test.
- **Zone drawing tool (`scripts/draw_zones.py`)**: OpenCV highgui (already a dependency), names and kinds typed in the terminal, plus a `--from-json` path so it is testable headless. It rewrites only `zones`, `camera.id`/`resolution` (from the frame) and `camera.floor_points`; floor points are never inherited from a template, since they are in another camera's pixels. Comments in the template are lost on rewrite (PyYAML). The written file is loaded with `load_store_config` (and the floor homography computed) before it reports success. No frame is saved except an explicit `--preview`, which is not pixelated (no detector runs in the tool).
- **POS CSV through a declarative mapping, not per-vendor parsers.** One YAML says which column is time (one or several joined; strptime, ISO or epoch), receipt, terminal (or a constant), item, qty, price, and maps item names to a SKU, a category or `skip` (exact names first, then globs). Rows are grouped per (terminal, receipt); the receipt time is its latest row; quantities add up so void lines cancel; receipts left empty (fuel only, fully voided) are dropped. Output is the existing JSONL wire format with `ts`, so CSV and JSONL go through the same `parse_payment` (a test checks they give equal `Payment`s).
- **Unmapped POS items default to `keep`** (as an unknown SKU) so nothing paid for is lost, but the ledger lets an unknown SKU cover an open item of any category, so non-merchandise lines must be mapped to `skip`; `pos-convert` lists every unmapped name.
- **Clock alignment is explicit**: IANA timezone for the POS's local time (zoneinfo, DST handled; the repeated hour at DST end resolves to its first occurrence), then `offset_s` = edge box clock minus POS clock, measured with a test sale at install. The ledger stays on the edge box clock.
- **CSV in the POS folder**: a changed CSV is re-read whole (ponytail: fine for daily files of a few MB) and receipts not seen before in that file are taken. While the file is still growing its last receipt is held back one scan (2 s), since its rows may not all be written. Receipts in CSVs present at start are marked seen, so a daily file the POS appends to works. Known limit: a POS that pauses more than one scan in the middle of writing a receipt gets that receipt split, and the rest of it is not read (same receipt id already taken).
- **Open-vocabulary product-in-hand detection tried (YOLOE-26s, text prompts, no training): no useful signal.** MERL train videos (12, 40 frames per class per video; `results/merl_product_in_hand_train.json`): share of frames with a detection within one forearm of a wrist, while holding a product ("Inspect Product") vs with empty hands (no labelled action): "object held in hand" @0.05 62% vs 50%; generic package words @0.05 34% vs 22%; snack words @0.2 10.5% vs 5.8%. Small snack packages from straight overhead are not separated from empty hands, so no setting was worth spending the test split on. A product detector still needs our own labelled products (Isaac Sim renders or pilot footage). Optional dependency installed for this: ultralytics' CLIP (`git+https://github.com/ultralytics/CLIP.git`); weights `models/yoloe-26s-seg.pt`.
- **Unsupervised concealment scoring tried (k-nearest-neighbour distance to normal shopping): also near chance.** Normal bank = 40,000 windows from PoseLift Train (normal only, v2 features, standardised + PCA fit on the bank; no theft labels used to fit). k and PCA size chosen on PoseLift's labelled clips (best AUC-ROC 0.559, k=1, PCA 16); held out, scored once (`results/conceal_knn.json`): RetailS staged 0.531, DCSASS 0.481, UCF-Crime 0.642. With the supervised v1/v2 results this settles it for now: public pose-only data does not give a concealment signal that transfers. Concealment evidence should come from the product leaving the hand near the torso (the engine's existing CONCEAL rule, which needs a product detector) and from labels collected in shadow mode, not from body pose alone.
- **MOT16 tracking check (real footage with identity ground truth, research licence, evaluation only).** 7 train sequences, 517 people, every frame, simplified protocol (pedestrians marked for evaluation, IoU 0.5; `scripts/eval_mot.py`, `results/mot16.json`). Settings fixed before running: A old default (YOLO26s, conf 0.3, 2 s buffer): MOTA 0.325, IDF1 0.435, 487 ID switches, recall 38.5%, precision 87.4%. B = A + contained-box removal (current default): MOTA 0.327, IDF1 0.438, 454 switches, recall 37.9%, precision 88.6%. C = MERL-tuned (YOLO26n, conf 0.15, 5 s buffer, new-track 0.15) + removal: MOTA 0.300, IDF1 0.395, 430 switches, 978 vs 1,357 fragmentations, recall 34.3%. Confirms keeping the small detector as default and adding duplicate removal. Engine-level stitching was measured afterwards (see the next entry). motmetrics 1.4 needed a local IoU function (its iou_matrix uses np.asfarray, removed in NumPy 2).
- **Stitching ambiguity guard (after MOT16).** Unguarded stitching on MOT16 crowds: ID switches 454 -> 290 but IDF1 0.438 -> 0.388 (different people merged); a stricter 0.5-height distance (exploratory, chosen after seeing that) gave IDF1 0.411. Principled fix instead of tuning: skip stitching when two lost people qualify or when someone visible stands within the stitch distance of the new track. MOT16: IDF1 0.441, 441 switches (better than no stitching on both). MERL test (single shoppers): visits per shopper 2.54 (2.36 unguarded), split 68% (64%); unguarded rows in `results/merl_offline_test_extra.json`, long-gap rows in `results/merl_offline_train_extra.json`. Default on (`stitch_ambiguous_skip`). `results/merl_offline_test.json`, `results/measured_error_rates.json` and `results/bench.json` were regenerated with it (simulator, measured rates: 28.6% precision, 18.0% recall, 53.3% alert-or-review recall, 1.11 false alerts/h).

## Re-identification without the face (2026-10-03)

**What it is for.** Basket minus paid only works if the shopper at the shelf, at the register and at the door is one identity. Position and time alone cannot carry an identity through a long occlusion, between cameras that do not overlap, or when two people stand close. `bree/track/reid.py` adds body appearance for exactly those three joins. It does not recognise anyone: it answers "is this new track the same body as one we lost a moment ago in this store".

**Cues.** All computed from the shoulder line down; the head is cut off the crop and any head keypoint that falls inside it (bowed head) is greyed out first, so no face pixel reaches a feature (`tests/test_reid.py::test_head_pixels_never_reach_a_feature`).
- Appearance embedding: **DINOv2 ViT-S/14**, 384 numbers per 224x112 body crop, run from ONNX on CPU.
- Clothing colour per body part from the pose keypoints (upper body, lower body, shoes): hue x saturation histogram plus four brightness bins for grey/black/white.
- Body shape: shoulder/hip width ratio and torso/leg length ratio from pose. Weak and view dependent (the fitted weight is small).
- Carried object: a backpack / handbag / suitcase box of the detector (COCO classes) overlapping the person.
- Motion: walking speed. A brand-new track has no motion history at the moment it has to be matched, so the fitted weight is 0; speed acts only through the plausibility gate below.
- Space and time are gates, not part of the score: the existing stitch distance / gap, and for longer gaps "could they have walked there" (1.5 body heights/s in one camera; 2.0 m/s on the floor plan between cameras).
- **Not built: height in metres.** `floor_points` give a homography of the floor plane only; a head top is not on that plane, so metric height needs a full camera calibration (or a vertical reference per camera). Left out rather than faked.

**Model choice and licences.**
- Picked: DINOv2 ViT-S/14. Code and weights are Apache-2.0 (https://github.com/facebookresearch/dinov2/blob/main/LICENSE, https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md). It is a general image model, not trained for re-ID, which costs accuracy (see REPORT) but leaves no licence question.
- Not picked: OSNet (x0.25 / x1.0) from torchreid. The code is MIT (https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE), but the published re-ID weights are trained on Market-1501, MSMT17, DukeMTMC-reID or CUHK03 (https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), datasets released for research use (DukeMTMC was withdrawn by its authors in 2019). The weights carry no commercial grant of their own, so they are not safe to ship. If Kiro later gets consented footage of our own, fine-tuning a small re-ID head on it is the obvious upgrade.
- Ultralytics YOLO (detector, pose, ByteTrack) is AGPL-3.0 as before; nothing new.

**Fusion.** One logistic score over the seven pair features (`match_prob`). Weights fitted on 21,748 MOT16 stitch-candidate pairs (418 same person); every MOT16 number in REPORT for a sequence uses weights fitted on the other six. The probability is on the scale of "anyone lost in the last minute who could have walked here" (about 2% of those are the same person), so 0.2 is already a strong match.

**How it is used (extends the ambiguity guard, does not replace it).**
- Stitching: a position/time candidate that looks different (p < `reid_reject` 0.2) is dropped. That removes wrong merges and also resolves "two lost people qualify" when only one of them looks right. If two still qualify, the guard skips as before. The "someone visible stands there" guard is unchanged.
- Longer gaps: someone lost up to `reid_long_s` (60 s) ago is relinked at p >= `reid_long_p` (0.3) if the distance is walkable. This is the shelf to register link: a shopper lost at the cooler and picked up as a new track at the counter keeps their basket, so the receipt clears it (`test_register_payment_lands_on_the_shopper_who_picked`).
- Cameras: `MultiCamIdentity(reid=True)` applies the same veto, prefers the best-looking candidate, and links non-overlapping views at a walkable distance.
- `reid_reject` 0.2 was fixed before the tracking run. `reid_long_p` was first 0.9, which a 2% base rate never reaches; 0.3 was set after seeing the score distribution, so treat the long-gap tier as tuned on MOT16.
- Flag: `rules: {reid: true|false}` in the store YAML (`EngineRules.reid`). **Default off.** The rule was "on only if it beats the current numbers": on MOT16 (angled, crowded) it does (IDF1 0.441 to 0.443, ID switches 441 to 434, false merges 13 to 7), on MERL test (overhead, one shopper) it does not (visits per shopper 2.43 to 4.64: the veto rejects the same person seen from above). No veto threshold fixed both (`results/reid_sweep.json`). So it is opt-in per camera until the pilot's own footage says otherwise; for overhead cameras use `reid_reject: 0` (relink only) or leave it off.
- Regression guard: `tests/test_reid.py::test_reid_does_not_regress_on_mot16` runs two MOT16 sequences in `make test` against `tests/fixtures/reid_guard.json`.

**Privacy rules (enforced in code and tests).**
- No facial recognition, no face features, heads still pixelated in everything written.
- Features exist only in process memory: `PersonObs.reid`, a per-track cache (dropped 10 s after the track ends) and a per-person gallery of the last 8 (cleared at exit, or once the person has been gone longer than the longest stitch window, or after `reid_max_age_s`, default 7200 s = 2 h, whichever is first).
- Never written to disk, logs, alerts or the network: `frames.jsonl`, events and alert records have no field for them (`test_features_are_not_serialised`). The benchmark also keeps them in memory only.
- No identity across visits: a person who exits is forgotten; a restart forgets everyone. There is no enrolment, no watchlist, no name or account attached.

**Law: what Kiro should take to a lawyer before a pilot. This is not legal advice and none of it is a conclusion.**
The design avoids faces, but several laws define biometric data broadly enough that body or gait measurements used to tell people apart might be covered, and "we delete it within two hours" is not by itself an exemption anywhere below.
- **Illinois BIPA**, 740 ILCS 14/10: "biometric identifier" is a closed list (retina or iris scan, fingerprint, voiceprint, scan of hand or face geometry) and excludes photographs and physical descriptions such as height. Body appearance is not on the list, which helps, but the statute has a private right of action with statutory damages (14/20), requires written consent and a public retention policy when it does apply (14/15), and courts have not, to our knowledge, ruled on body-embedding re-ID. https://www.ilga.gov/Legislation/ILCS/Articles?ActID=3004&ChapterID=57
- **Texas CUBI**, Tex. Bus. & Com. Code 503.001: same closed list ("record of hand or face geometry"); enforced by the Attorney General, no private action. https://statutes.capitol.texas.gov/Docs/BC/htm/BC.503.htm
- **Washington**, RCW 19.375.010: open definition, "data generated by automatic measurements of an individual's biological characteristics ... or other unique biological patterns or characteristics that is used to identify a specific individual". A body embedding used to re-identify could plausibly be argued to fit. https://app.leg.wa.gov/RCW/default.aspx?cite=19.375
- **California CCPA/CPRA**, Cal. Civ. Code 1798.140(c): "biometric information" expressly includes "gait patterns or rhythms" and imagery from which an identifier template can be extracted. Applies above size thresholds. https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=CIV&sectionNum=1798.140
- **New York City**, Admin. Code 22-1201 to 22-1205: commercial establishments that collect "biometric identifier information" (includes "any other identifying characteristic") must post a sign at entrances. (Cited from memory: the city code site blocked automated access from here, so the wording was not re-checked. The Illinois, Texas, Washington and California wording above was checked against the linked pages on 2026-10-03.)
- **GDPR**: Art. 4(14) defines biometric data as data from specific technical processing of "physical, physiological or behavioural characteristics" that allow or confirm unique identification; Art. 9 restricts it when processed to uniquely identify a person. Whether short-lived in-store re-ID without any link to a known person is "unique identification" is debated; either way the video itself is personal data (Art. 4(1)) and needs a lawful basis, notice and likely a DPIA (Art. 35). See EDPB Guidelines 3/2019 on processing of personal data through video devices. https://eur-lex.europa.eu/eli/reg/2016/679/oj
- **EU AI Act**, Regulation (EU) 2024/1689, Art. 3(34) to 3(43) and Annex III: defines biometric data and biometric identification / categorisation systems; worth checking whether in-store re-ID falls in a regulated class. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Questions for counsel: (1) in the pilot's state, is a body appearance vector or a gait/speed measure a biometric identifier; (2) does notice at the door suffice or is consent needed; (3) does in-memory, per-visit retention change the answer; (4) signage wording; (5) what the operator's own policies and insurer require. The pilot stores are gas stations: the state they are in decides which of the above matters first.
- Fallback if counsel says no: `reid: false` returns to position/time only (no appearance features computed at all), with the tracking numbers in REPORT's "before" column.

## Closed-world identity (2026-10-03)

**Why.** Re-ID made MERL worse (visits per shopper 2.43 to 4.64) for one reason: when the appearance veto rejected a stitch, the engine created a NEW person, so one shopper became several and the basket was stranded on an identity that never exits. Kiro's idea removes that outcome instead of tuning the veto: a store is a closed room with a door. A shopper who has been seen stays the same shopper until they leave or have not been seen for a long time. Nobody is created in the middle of the store.

**Rules (`rules.closed_world`, `EventEngine._closed_world`).**
- **Births only at the door.** A track id not seen before becomes a new person only if its foot point is in an `exit` zone (the door: in and out) or an `entrance` zone (new zone kind: one-way way in, never counted as an exit), or its box is within `entry_border_px` of the frame edge (default 0 = off; for a camera whose frame edge is the way in). `closed_world_warmup_s` (5 s) after start, people already inside may appear anywhere (a restart in the middle of the day).
- **Config without any door** (no exit/entrance zone, no border): births in the warm-up, and the first appearance when nobody is unaccounted for (`born: no_door`, not flagged). Everything else is assigned.
- **Any other new track is one of the lost people.** Lost pool = seen before, not exited, not timed out, not visible in this frame. All new tracks and all lost people in a frame are solved as ONE assignment (Hungarian, `scipy.optimize.linear_sum_assignment`), not greedily: `test_assignment_is_solved_jointly_not_greedily` is the case greedy gets wrong. Score per pair = 0.5 x appearance + 0.35 x walkability + 0.15 x recency (`CW_WEIGHTS`, fixed before any measurement, not fitted):
  - appearance: the re-ID match probability over `reid_long_p` (capped at 1); 0.5 for every candidate when `reid` is off or a side has no features, so it then drops out;
  - walkability: 1 / (1 + (d / reach)^2), d from where they were last seen, reach = `stitch_dist` + `reid_max_speed` x gap, in floor-plan metres over 1.7 when the camera has `floor_points`, else in body heights of the lost box;
  - recency: exp(-gap / 60 s).
  The open-world gates (`stitch_s`, `stitch_dist`, `reid_reject` as a veto, the ambiguity skip) are not applied: with the flag on they could only create a person, which is the thing ruled out.
- **Deaths only at the exit zone or the timeout.** `closed_world_timeout_s` = 3600 s (60 minutes), configurable. A lost identity keeps its engine state and its ledger record (basket, register visits, payments). On timeout the identity is dropped from the pool with a log line and no decision: no exit was seen, so nobody is accused. The ledger drops a track that has been silent for 1 h on its own (`Ledger._prune`, fixed at 3600 s): keep the engine timeout at or below that, or payments stop being matched to the person in between.
- **Too close to call.** If the best score beats the runner-up (another lost person for this track, or another new track for this person) by less than `closed_world_margin` (0.1), the assignment is still made, one identity per track and nobody new, and both identities are marked `identity_uncertain` for `closed_world_uncertain_s`. Also marked: a match that looks different (p < `reid_reject`) but was the only choice, which is what a missed entry looks like. Every event of a marked person carries `meta.identity_uncertain` (+ `_until`); the ledger caps any decision whose party includes such a person at **review** ("capped at review: identity uncertain (...)" in the alert reasons, "alert downgraded to review: identity uncertain" in `ledger_log.txt`). The period defaults to 3600 s, the whole visit: waiting does not undo a swap, so a shorter period would turn back into a silent guess. It is a knob because a store may decide that picks made after the uncertain moment are safe to alert on; that finer rule (uncertainty only for items picked before the mark) is not built.
- **Occupancy sanity.** `EventEngine.occupancy()` = people created minus exits and timeouts. A track that appears inside when nobody is unaccounted for becomes a person flagged `missed_entry` (ENTER `meta.missed_entry`, "entry not seen" in the ledger log, `births_missed_entry` in `summary.json` `identity`). It never raises. Two details found on MERL: a detector flicker that was never at the door and lasted under `min_store_time_s` is not kept as someone to come back to; and a box that starts on top of someone visible when nobody is lost is far more often a second detection of that person than a missed entry, so it is left unplaced (no person, no events) for up to `min_store_time_s`, takes over the identity if the first track ends, and becomes a missed entry if it stays or steps away. That second rule was added after seeing the first MERL numbers (train and test, cached tracks); its thresholds are the existing `stitch_dist` and `min_store_time_s`, not tuned.

**Privacy (unchanged rules, one longer lifetime).** With `reid` on, a lost shopper's gallery (last 8 feature sets) now has to outlive the old 60 s stitch window: it stays in process memory until that shopper exits or times out (60 minutes by default), then it is cleared (`test_identity_survives_a_long_absence_until_the_timeout`, `test_exit_ends_the_identity_and_clears_features`). Still memory only, never on disk, logs, alerts or the network; still nothing across visits; a restart forgets everyone. With `reid` off closed world uses position and time only and no appearance feature is computed at all. The longer in-memory lifetime is one more fact for counsel (question 3 in the re-ID section).

**Scope and limits.**
- **Single camera.** With several cameras of one store each camera's engine would take a person walking in from another camera's area for someone lost in its own view. `StoreEvents` switches the flag off per camera (logged) and the floor-plan handoff stays in charge. The right fix is one lost pool per store inside `MultiCamIdentity`; not built.
- **MOT16 does not apply.** It is street footage: people walk in and out of every frame edge, nobody exits through a door, and most people who leave the frame never come back. The lost pool fills with people who are gone and every new pedestrian is forced onto one of them. It stays in `make reid-bench` as a sanity check only and is excluded from the store default. The MOT16 regression guard still runs with the flags off and now also asserts that row is unchanged.
- No real store footage with several shoppers and identity labels exists here. MERL is real but has one shopper per video, so it measures splits and cannot show a false merge. The multi-shopper numbers come from a scripted store (`synthetic_store_rows`: synthetic tracks, an occlusion every 4 to 10 s, half of them hiding two shoppers at once), which tests the logic, not perception.

**Default: off (2026-10-04).** Rule set beforehand: make it the default for store configs only if it clearly beats the current default on store data without raising false merges. MERL test (real, one shopper per video): visits per shopper 2.43 to 1.36, with or without re-ID. Toy clips: scorecard unchanged. Scripted multi-shopper store: splits 3,209 to 1,214 and silent false merges 554 to 105, but raw false merges 554 to 1,238 without re-ID, and 62% of visits review-only. That fails the rule as written, so `EngineRules.closed_world` and the example store YAML stay off. With re-ID on as well it passes (false merges 314, splits 329), but re-ID is opt-in pending counsel. To switch a single-camera store on: `rules: {closed_world: true}`. Numbers: REPORT "Closed-world identity", `results/reid_bench.json`.

## Calibration, 3D slots and store-wide identity (2026-10-04)

- **One frame for everything: the layout file's.** Metres, y up, floor at y = 0; a store YAML floor point (x_m, y_m) is the layout point (x, 0, z). The simulator, the sim-eval adapter, the calibration and the slot map all use it, and a test checks that a layout camera projects to the same pixels through `bree.calib.camera` and through the adapter's matrices.
- **Pinhole, no lens distortion.** The planned item lenses are about 25 degrees wide, where distortion is small. Add a k1 term (and a checkerboard) if the reprojection error of real marks grows toward the frame edge. The tracking cameras are about 120 degrees wide and will need it; they are only used for the floor mapping today.
- **What `fit` does depends on what it is given.** 4 or more floor marks: a floor homography, which is all the handoff needs. A known lens (`--hfov`, `camera.hfov_deg`, or the layout entry): the full pose with `cv2.solvePnP`. 6 or more marks and no lens: pose and focal length together, which the bench shows is about three times worse in position, so the lens should be written down.
- **`check` wants a fifth floor mark.** Exactly 4 marks fit exactly and prove nothing. Thresholds: 5 px reprojection RMS, 0.15 m on the floor.
- **The floor homography is fitted on normalised points** (Hartley). 4MP pixel coordinates next to metres made the raw fit badly conditioned as soon as the marks carried click noise.
- **The error model is the simulator's, on purpose.** Per camera, a 1 sigma miss across the line of sight of sqrt((pixel error / pixels per metre)^2 + (range x pointing error)^2). `NoiseModel.pixel_px` is the simulator's `pxSigma`, `cam_rot_deg` its `calibDeg`; `cam_pos_m` is added here because a hand-measured camera position is not exact. One definition means a layout scored in the simulator and a pick located in the pipeline are talking about the same error.
- **The defaults are not the simulator's, also on purpose.** The simulator's 0.29 px is a shelf point known to the nearest pixel, a floor for what optics allow. The pipeline locates a pose keypoint on a hand, which is a few pixels off at best: 5 px, 2 cm, 0.2 degrees. `NoiseModel.sim()` returns the simulator's values. The synthetic bench puts the tool's own calibration at 0.08 degrees median and 0.24 at the 90th percentile for 2 px clicks, so 0.2 is on the careful side and the simulator's 0.1 is near the median. All three are placeholders until measured on installed cameras.
- **Slot assignment weights each direction by how well it was measured.** A triangulated point is known well across both lines of sight and badly along them. Picking the nearest slot in plain metres ignores that: 38.6% exact against 56.0% at the default noise (synthetic bench).
- **Source order for the slot: two views of the hand, then the item's own pixel, then one view of the hand.** The second camera does not need the shelf zone. A shelf zone still belongs to one camera, because two cameras with the same zone both raise a PICK and the basket counts the item twice.
- **Store-wide closed world is one pool in `MultiCamIdentity`,** not one per camera. A per-camera pool takes a person walking in from another camera's area for someone lost in its own view. Births only in a door zone of a camera that sees it, deaths at a door camera's EXIT (dropped if another camera still sees the person a second later) or the timeout. The score and its weights are the single-camera ones (`CW_WEIGHTS`), not refitted.
- **A track at the door is never handed to someone lost inside, and never to someone whose track just ended.** A new track in a door zone takes an existing identity only when another camera has that person in its latest frame (at most `door_gap_s`, 0.5 s, old) at that spot. Otherwise it is a birth. The first version handed it to anyone seen up to 0.5 s ago, which swapped identities when one person walked in 0.2 or 0.4 s after another walked out: the newcomer took the leaver's basket and the leaver's EXIT was vetoed (audit 2026-10-05, `test_tailgating_at_the_door_is_two_people_and_the_exit_is_delivered`). Cost: a track that flickers inside the door zone of a single door camera becomes a second identity, and the first one gets an exit. No uncertain mark is set, because nothing was guessed: nobody was handed anybody's basket.
- **Default off,** by the same rule as the single-camera flag. Numbers and reasons in REPORT "Calibration, 3D slots and store-wide identity".

## Camera node first pass: Pi Zero 2 W + Camera Module 3 (2026-10-04)

**The node only answers "did something hand-sized move into a shelf zone", nothing more.** A Zero 2 W
cannot run a detector on 12 MP frames, and shipping every full-res frame from 45 cameras is the cost we
want to avoid. So the node runs a background model and a motion check on a 320 pixel wide stream and
sends full-res frames only around a trigger. All recognition stays on the hub. Code:
`src/bree/edge/trigger.py`.

**Tuned for few misses, not for few false triggers.** A false trigger costs upload. A miss loses the
pick for good, because the full-res frames are never sent. So the defaults are loose (30 low-res pixels
of change, 6 moving), and the zone is grown by 6 low-res pixels: on the MERL train split the drawn shelf
polygon alone missed reaches to the front edge of the shelf, and the margin fixed that. The price is
that a body in front of the shelf triggers as well. In practice the trigger means "someone is at this
shelf", and the saving comes from the time nobody is.

**Colour as well as brightness.** A hand that is as bright as the packaging behind it is invisible to a
grey-level difference. The background model keeps Cr and Cb too. Skin colour is only a shortcut that lets
a smaller blob through; it never blocks a trigger, because gloves and bad light break skin detection.

**Pre-roll is raw frames in RAM, encoded only when a trigger fires.** Encoding every full-res frame just
in case would spend the CPU the trigger is there to save. The ring holds the frames of the last `pre_roll_s` seconds by their own timestamps, at most
`fps x pre_roll_s` of them (the RAM bound), as YUV420. So a source slower than the configured rate still
gives `pre_roll_s` of pre-roll, and a faster one gives less (20 frames at a real 20 fps are 1 s, not 2). The frame size is a config value because the Zero 2 W has 512 MB: 2304x1296 (the sensor's binned
mode) is the starting point, the full 4608x2592 is untested.

**HTTP, not MQTT.** A burst is megabytes of JPEG: that is a POST body, and it is what MQTT brokers are
tuned against. HTTP needs nothing beyond the Python standard library on either end (no broker to install,
run and secure on the hub), the repo already takes POS payments by HTTP POST, and the hub can push back
with a status code (429 + Retry-After). The nodes are on wired PoE, so MQTT's strength on flaky links
buys little. What MQTT would give for free (queued delivery, last-will) is covered by the node's disk
spool and the heartbeat. If a later design wants many small telemetry messages from many more nodes,
MQTT is worth another look for those, not for the frames.

**The spool drops the oldest burst when the card is full.** After a long outage the newest evidence is
the one a person can still act on. Drops are counted and reported in every heartbeat.

**Back-pressure in two places.** The hub answers 429 when its queue for the vision pipeline is full, and
the node keeps the burst. As the node's spool fills, it sends less per burst: past half full every second
pre-roll and post-roll frame is left out, past 90 % only the frames with a zone active are sent.

**Frame times are the node's monotonic clock; the hub converts.** A Pi Zero has no battery clock, so its
wall time is wrong until NTP answers, and wall time can jump. Monotonic time cannot. Each message carries
a `boot_id` (monotonic time restarts at boot) and the node's monotonic time at send; the hub keeps the
smallest (receive time minus send time) over the last 30 requests per boot and adds it to every frame
time. A burst that was spooled before a node reboot and
sent after it gets the old boot's offset if the hub still has it, else no hub time (`hub_time: null`),
never the new boot's.

**One pipeline run per burst in the hub's own `--store` mode.** The hub writes each burst as `burst.mp4` and
runs the existing pipeline on it. A shopper therefore has no identity or basket across bursts. Joining
bursts into one ledger is done outside the hub by `BurstSource` (see "Integration" below).

## Human review feedback loop (2026-10-04)
- **SQLite, one file per store, standard library only.** The shadow-mode `labels.jsonl` stays as it is. The review store is a superset that can ingest `would_be_alerts.jsonl`, so nothing has to migrate.
- **Stdlib `http.server`, not FastAPI.** FastAPI is not a dependency of this repo, and the existing dashboard and shadow review page already use the stdlib server (JSON-only POST, byte ranges for Safari, 127.0.0.1). Same pattern, no new dependency.
- **Four decisions: confirmed theft, not theft, theft with the wrong item, unclear.** "Wrong item" means a real theft where the system named the wrong product, and the reviewer must give the right one. So precision counts it as a correct alert, and a second figure (item precision) counts only alerts where theft and item were both right.
- **Reviews are append-only, latest per reviewer counts.** Same rule as the shadow labels: a changed mind stays in the history.
- **Two reviewers who disagree make the alert "disputed".** Disputed alerts are left out of precision and out of the label export. Agreement is reported as the share of same decisions and Cohen's kappa over the first two reviewers of each alert.
- **Label rules are conservative.** Detector: only confirmed theft (class = predicted category) and wrong item (class = corrected item). "Not theft" says nothing about what the product was, so it gives no detector label. Pick: the reviewer's answer if given, otherwise 1 for a confirmed theft (marked `derived`). Conceal: only from the reviewer's answer, never derived, because a theft can be a plain walk-out.
- **YOLO label files carry the reviewed box first and the detector's other boxes as unreviewed pseudo-labels.** A file with only the reviewed box would teach the rest of the shelf as background. The manifest gives `pseudo_boxes` per example so a trainer can drop them or mask them.
- **Class names, not SKU ids.** The manifest lists class names as the alerts carry them, with ids that never shift between exports. The trainer maps names to its own ids.
- **Export is a rebuild, dedup by content hash.** Each export rebuilds the list from the store. A changed decision or a purged alert drops out and its files are deleted. The key is a hash of the image bytes plus class plus box (or of the keypoint window plus label), so the same evidence logged under two alert ids is one example.
- **Detector examples need a clean frame.** Alert clips are 640 px wide with boxes, zones and skeletons drawn on them, so they are not used as training images. A frame must come from `save_evidence_frame` (heads pixelated, no overlays, full resolution). The pipeline does not call it yet. That file belongs to another stream.
- **Retention deletes media, not decisions.** Past `retention_days` (default 30) the clip, frames, stored keypoints and exported examples are deleted. The decision row has no image and no identity, so it stays and the weekly numbers do not change after a purge. Retention uses today's setting, so shortening it also removes examples exported earlier.
- **Keypoints are stored for alerted people only, for the retention period.** They are pose, not appearance, and the conceal classifier trains on them. Whether pose over time counts as a biometric is already a question for counsel in the re-ID notes. If the answer is yes, ingest without `--frames-log` and no keypoints are stored.
- **Staged tests are matched by test id, or by time and camera** (30 s before to 300 s after the staged time, one alert per test). Their alerts are reported separately and left out of the customer precision figure, so running tests cannot make the system look better.
- **Plain pipeline alert ids restart at A00001 every run**, so the ingest command stores them as run folder name, alert number and a hash of the record. The same record twice is one alert. A different record under the same folder and number is a new alert, with a warning, because the earlier alert's clip file may have been overwritten. Shadow ids are already unique.
- **Retention follows the export.** Every folder a label export is written to is recorded in the store's `meta` table, and the purge walks all of them. Refusing folders outside the store would have been simpler, but the training stream may want the dataset on another disk.
- **A corrected item is stored only with "wrong item".** Text left in the field with any other decision is dropped in the store, not only in the page, so no other client can add a junk class either.
- **Staged test matching looks across the week boundary.** A test staged just before midnight Sunday UTC is caught by an alert a few minutes into Monday. That alert counts as a staged one in the next week's report, not as a customer alert.
- **The label export clips boxes to the image and drops boxes with no area.** A bad reviewed box is counted under `skipped.detector_bad_box`. A bad or unnamed unreviewed box is just left out of the label file.
- **No train / test split in the manifest.** The trainer splits by `alert_id`, so one alert's frame and windows stay on one side.
- **The page refuses decisions nobody could have looked at.** Verifier finding: holding key 1 for about a second decided a whole queue. Now a held key counts once, a decision is ignored while the previous one is being saved, and nothing can be decided in the first 0.4 s an alert is on screen. 0.4 s is far below the time a real look takes and only stops key bounce and double presses. There is no minimum watch time: a reviewer who knows the clip can still decide fast.
- **Back goes one alert back.** Key U reopens the alert just decided and a new decision replaces the old one (a new row, the latest per reviewer counts). A full history browser was not built. Older mistakes are fixed by a second reviewer or directly through `store.decide`.
- **A fully withdrawn alert leaves the queue and the metrics. A lowered one stays.** A late receipt that clears every item means staff were told to stand down, so asking a reviewer about it wastes their time and counting it as "sent" would charge the system for an alert it took back itself. It is counted on its own. An alert lowered to the review tier is still an open question, so it stays, with a line on the page. Shadow mode keeps tiers "as first decided" in its own summary. The two differ on purpose: shadow measures what staff would have been told at the time, the review store measures what was left for a person to judge.
- **One rule for "sent" in `metrics()` and the owner report.** Both leave out staged-test alerts and withdrawn alerts, through one shared function. Before, `metrics()` counted staged alerts and the same week had two precision numbers.
- **Every alert that names a staged test is a staged alert**, whether or not the test was logged. Unlogged ids are listed in the report so the owner can see a test was not written down.
- **Corrected items have one spelling** (trimmed, lower case, underscores). A name the pipeline has never used still becomes a class, because a new product is a legitimate correction, but it is marked `class_known: false` and the page asks for Enter twice.
- **Allow-list, not deny-list, for what is stored.** Items, receipt lines, frames and product boxes keep only named fields. The earlier filter matched identity-looking key names and missed names like `vector` or `crop`.
- **Media is accepted by folder name: clips from `alerts`, frame images from `frames`.** This is best effort and the docs say so. Checking pixels for blurred heads was not attempted.
- **One dataset folder, one store.** The manifest carries the store's id (a random id made when the store is created) and an export from another store is refused, because an export rebuilds the folder and would delete the other store's examples.
- **The page checks the Host header.** Only `127.0.0.1:<port>` and `localhost:<port>` are answered, so a web page that points its own name at this machine cannot read alerts or post decisions. POST bodies are capped at 64 kB.

## SKU detector trained on simulated frames (2026-10-04)

- **Train on renders of the store simulator, because no labelled real frames exist.** The point is to have a product backend that names SKUs so the rest of the chain can run on simulator views, and to put a first number on the pixel threshold the camera layouts assume. It is not a claim about real accuracy, and every result file says SIMULATED.
- **A private copy of the simulator, never the original.** `render_synth.mjs` copies `~/bree/software/sim-prototype` to `out/sim-copy` and adds one overlay file (`scripts/train/sim/synth.js`). The research repo's simulator is not run or edited by this repo.
- **Ground-truth boxes come from an instance-id render,** so a box is what is visible after hands, bodies, neighbours and shelves hide what they hide. A box is a label only if at least 30 pixels of the item show, its short side is at least 5 pixels and at least 25% of it is visible.
- **640 px tiles at native resolution.** A 4MP frame squeezed to 640 px turns a 25 px can into a 6 px one. Training and inference both tile; overlap 128 px. Ceiling: an item larger than the overlap that no tile holds whole is cut, which shows as the accuracy drop above 40 px across a can. Upgrade: a downscaled whole-frame pass next to the tiles.
- **Split by scene seed.** No scene, shopper or lighting draw is shared between train, val and test. It is still one store and one set of package art.
- **Small colour augmentation and no mirroring.** The SKU is in the label art.
- **In the pipeline the detector runs only on tiles around people** (`roi="people"`). The event engine uses products in or near a hand, not the stock on the shelf, and a whole frame is 15 tiles. Cost: no person box means no product box, which is where the end-to-end run on simulator views lost its picks (REPORT "End to end on simulated data").
- **No validation pass during training** (on MPS it costs about a fifth of an epoch); the kept weights are the last epoch's and the val split is scored once at the end.
- **The minimum-pixel rule and what it returned.** Rule, fixed in `evaluate.py` before the run: the lowest bucket edge from which every bucket with 30 or more items has the right SKU 90% of the time, for clearly visible front items. Every bucket passed, so the rule returns no lower limit. The first version of the report printed that as "0 px", which is not a recommendation; the text now says the 20 px default holds and nothing supports raising it, and gives the curve by the item's own box size, where the floor does show (81.7% at 10 to 15 px).
- **Documented default: 20 px across a 6.6 cm can.** It is the simulator's `pxMin`, the threshold the layout reports use, and it stays. Simulated evidence only; the first real labelled frames replace it.
- **Fine-tuning on real frames starts from the sim weights** with the backbone frozen and a low learning rate, and rebuilds the class head when the real SKU list differs. Not run: there are no real labelled frames.

## Integration of the four streams (2026-10-04)

- **Result files live in `results/`** (`calib_bench.json`, `edge_measure_*.json`, `sku_detector.*`, `px_vs_accuracy.png`, `e2e_sim_chain.*`), like every earlier benchmark. The stream fragments under `docs/fragments/` were merged and removed.
- **Edge numbers were re-measured at integration.** The edge stream's result files predated its last trigger change (brightness normalisation), so the tables in REPORT come from a re-run of every file with the final code: 429 triggers and 77 false ones where the fragment said 455 and 81.
- **One error model, two sets of defaults.** The triangulation in `bree.calib` and the simulator's two-camera 3D metric use the same formula, tested at the image centre. They do not give the same number: on the simulator's per-slot output for the 45 camera layout the pipeline's error is 0.891 to 0.999 of the simulator's (median 0.960), because the simulator adds a lens edge loss and uses range where the pipeline uses depth. Within about 10%, the simulator the more cautious (audit 2026-10-05; `bree/calib/slots.py` docstring). The defaults stay different because they describe different things (a shelf point in the simulator, a hand keypoint in the pipeline), and both documents now say so and give the simulator's numbers at the pipeline's values.
- **`camera.calibration` is loaded with the store YAML.** The calibration tool wrote it and nothing read it.
- **The review manifest is what the detector fine-tune reads** (`finetune_real.py --manifest`). Split by a hash of `alert_id`, so an alert never changes side between exports. Examples with `class_known: false` are left out, not guessed. The derived dataset is built in a temp folder and deleted after training, because the review store's retention only tracks the export folder.
- **Node bursts become a pipeline source (`BurstSource`) instead of changing the pipeline.** The hub already stores every frame with its time. Reading a camera's bursts back as one stream with gaps lets node cameras and continuous cameras share one ledger, one identity pool and the 3D slot logic with a two-line change in `pipeline.py`. Ceiling: it reads recorded bursts, and the tracker is not told about the gaps.
- **In the end-to-end run the node cameras are the item cameras** (layout kind shelf or cooler), as in the hardware plan. Checkout and overhead cameras stream every frame.
- **Sim-eval zone rules changed for the recommended layout.** A zone partly behind the camera is kept (rail cameras sit beside the gondola they watch; before, those cameras got no shelf zone at all), and the door and register zones go to a camera that watches people when one sees them.
- **The reviewer in the end-to-end run is the simulator's ground truth,** and the report says so. It exists to exercise ingest, decisions, metrics and the owner report on the pipeline's own alert records, not to measure reviewers.
- **`BREE_PERSON_CONF` exists for simulator runs only.** The simulator's figures seen close up score around the default 0.3 with COCO weights. The recorded scorecard uses the default; the lower setting is reported next to it and labelled as chosen on that clip.
- **Not wired, on purpose:** alerts are not written to the review store live, the pipeline does not save clean evidence frames, and the ledger does not read the 3D slot. Each is a change to `pipeline.py` or the ledger with its own tests, and none was needed to run the chain.

## Audit fixes (2026-10-05)

An independent audit re-ran the commands behind the reports and probed the code. What changed and why. Numbers marked "audit" were measured by the audit, not re-run here.

- **Hub bursts are deleted after 24 hours.** A stored burst is raw full-resolution JPEGs with faces not pixelated, plus `burst.mp4` and `pipeline/` when the hub runs the pipeline on it. Nothing deleted them, so they outlived every retention rule in the repo. `Hub(retain_s=86400)` (`--retain-hours 24`, 0 keeps everything) now sweeps at start and every 10 minutes and removes every burst folder received longer ago than that, processed or not. A burst removed before the pipeline ran on it is counted as a hub error, because it means the pipeline is a day behind. Hours, not the review store's 30 days: the hub holds raw input, the review store holds the evidence of an alert. Ceiling: the pipeline does not yet copy a clean evidence frame out of a burst at an alert, so after 24 hours the full-resolution frames of an alert are gone; raise `--retain-hours` on a pilot box if that matters before the copy is built.
- **The review database zeroes what it deletes.** `PRAGMA secure_delete=ON` on every connection, and a `VACUUM` after a purge that changed rows (which also cleans a file written by an older version). Before, the keypoints of purged alerts were still readable in freed pages of `review.sqlite` (audit: 56 marker hits in the file against 34 in live rows). `test_purged_keypoints_are_not_left_in_the_database_file` reads the file bytes.
- **An exited identity is not reused.** If a camera track that was mapped to someone who has exited (or timed out) is seen again, the mapping is dropped and the track is placed again: a door birth or a hand-off. Before, later picks were delivered under the exited person.
- **Floor position is not taken from a box cut by the frame bottom.** With the frame size known, a person whose box bottom is within 2 px of the bottom edge has their feet out of frame, and the box bottom is some point up the body. A known person then keeps the last position that had feet; a new track is still placed (dropping it would drop a shopper at the register) and counted in `placed_with_feet_out_of_frame`. `to_floor` returns None for a pixel at or above the horizon; such a track is not placed. Audit geometry check on the 45 camera layout: feet out of frame in 53.4% of sampled positions on the 2 checkout cameras, 12.7% on cooler cameras, 2.6% on shelf cameras, 0.2% on overhead cameras. Identity belongs on the overhead cameras.
- **`BurstSource` never mixes clocks.** When any frame of a camera has a hub time, frames without one are dropped and counted. Two hub-time sources without an explicit shared `t0` are refused by `run_store`. Hub time is good to tens of milliseconds per camera (audit: mean 10.6 ms, worst 37.0 ms in a simulated link), the same order as the 0.07 s pairing window of the slot step.
- **`make e2e-sim` reproduces the recorded clip and leaves `results/` alone.** It reads the clip `make sku-clip-door` renders (the camera list of the recorded run, door camera included) and only replaces `results/e2e_sim_chain.*` with `E2E_RESULTS=e2e_sim_chain`. Before, the documented one-liner rendered a different, door-less clip and overwrote the tracked result.
- **The pick funnel's columns are independent counts,** and the table now says so and carries the two intersections. The earlier text read "69 tracked, 69 found, 69 right" as nested; it is 58 of 69.
- **The simulator copy's version is written into every clip and dataset** (`out/sim-copy/COPIED_FROM.json`, `clip.json` `simulator`, `meta.json` `simulator`). The copy that rendered the training frames and the recorded clip is commit a4f479f of `~/bree` (2026-10-03), identified by comparing files; the simulator has changed since.
- **`motmetrics` is a declared dependency** (`dev` extra) and MOT16 is fetched by `make data`.
- **Not changed:** the first-order triangulation error is left as it is (audit Monte Carlo: about 4% low, up to about 12% off axis); the docstring says so. Four edge slots the simulator calls localisable project 2.4 to 3.0 px outside the frame in `from_layout` (audit); noted, no action.

## Improvement round 1 on DEV (2026-10-05)

- **The POS lag is a setting of the store, 1.5 to 4.5 s for the simulated register feed.** `LedgerConfig.pos_lag_s`
  defaults to (0, 0) (receipt stamped while the payer stands there, the old behaviour). `bree.shelf.store.POS_LAG_S`
  sets it from the delay written in `scripts/bench/render_clip.mjs`. It is a property of the feed, the same for
  every clip and split, not a value fitted to DEV; 1 to 5 s and 2 to 4 s give the same DEV result. A real store
  measures its register clock against the camera clock once.
- **Time before content when crediting a receipt.** The old order (basket match first) let a thief's unpaid item
  pull in the receipt of the next customer who bought the same product. The 56 existing ledger and payment tests
  pass unchanged.
- **Correct receipts kept although the wrong ones scored better on one count.** With the lag off DEV shows 2 honest
  reviews instead of 4 (and 16 thefts instead of 17), because unclaimed receipts of other shoppers pay for false
  items. That is an accident of which products are popular, so it was not used to meet the bar.
- **Puts must be confirmed, slot-watch-only takes are dropped, a count above 1 is one unit.** The evidence is from
  TRAIN-seed clips (`out/shelf/eval_all9.json` joined with the stored events: slot watch alone 0 of 6 takes true,
  one-cue puts 4 of 23 true) and agrees with DEV. Cost: a real put-back seen by one cue at a slot more than 0.25 m
  from the read take is ignored; two units taken in one reach are one.
- **"Same place" for a put is 0.25 m in 3D, not a reach.** Wider (or along the floor) lets false puts erase stolen
  items: DEV thefts fell from 17 to 13 and 11.
- **One reach is measured along the floor.** Cost: two takes from two shelf heights of one bay within 4 s by one
  shopper are one take.
- **`misread_factor` 0.5**, the same size as `ambiguous_factor`: one unmatched paid item halves one open unpaid
  pick, the weakest first. A thief who pays for an item whose pick was missed and openly carries one other item out
  is no longer reviewed on that item alone; a concealed item is never discounted.
- **Parties come from the floor tracks** (ENTER meta `party`); the 4 s entry window stays as the fallback for
  callers that give no party information (the single-camera engine). First values, not swept: 1.5 m, half the
  common time, at least 5 s together. No simulated clip has a real party, so only a unit test covers the positive
  case.
- **Pick confidence of one-cue takes stays 0.9.** 0.6 would match the TRAIN-seed precision (18 of 32 one-cue takes true) but changed
  nothing on DEV, so it was left alone.
- **The DEV result of this round reuses the stored shelf events and person boxes** (`--keep`), since no code before
  tracking changed.

## Integration of the shelf, association, hand detector, benchmark and plates streams (2026-10-05)

- **One runner.** `bree.shelf.store:run` is the multi-camera path and the default of `make bench-dev`. The
  shelf-events stream's first join code in that file (its own fuse, slot vote and nearest-shopper rule) was removed
  in favour of `bree.shelf.events.fuse_views` and the association stream's `bree.events.shelf.store_events`; three
  tests of the removed functions went with it. `bree.sim.bench:run_pipeline` (per-camera engine) stays as `--runner`.
- **Detector weights: `sim_sku_hands_v3` for the shelf cameras of this runner.** It is better than `sim_sku` on the
  held-out seeds 3000 to 3009 on every row of `results/hand_detector.md` (item in a hand, right SKU: 87.5 against
  67.0 percent). `BREE_SKU_WEIGHTS` overrides it; `sim_sku` stays the default everywhere else. Not checked: the shelf
  rules were tuned with `sim_sku` and the shelf evaluation was not repeated with the new weights.
- **The slot claim of a PICK is `meta.slot.id`** (the association stream wrote `meta.slot.slot`; the scorer reads `id`).
- **One act per reach, after association** (`one_act_per_reach`, 4 s and 1 m, the first values tried; 3 s and 0.6 m
  and 6 s and 1 m give 15 and 16 thefts on DEV). Known cost: one shopper taking two different products from the same
  metre of shelf within 4 s is read as one take. No simulated clip has that.
- **Clothing colour in the floor tracker.** The association stream kept the tracker to position and time. Passing
  shoppers were swapped too often for that (15 identities covering two shoppers on DEV), so detections may now carry
  hue and saturation histograms of the upper and lower body (`bree.track.reid.part_colors`, shoulders down). The
  thresholds (`app_near_m` 0.8, `app_frames` 5, `app_margin` 0.08, `app_scale` 0.15) are first values, not swept.
  Detections without `app` give the position-only tracker, so the association stream's own numbers are unchanged.
- **Privacy limit to fix before a real store.** The runner stores person boxes, keypoints and those colour histograms
  in `<out>/pipeline/people_<camera>.jsonl` so that tracking can be repeated without the models. The repo's rule is
  that appearance features stay in memory for the visit only. On simulated clips this is harmless; a store deployment
  must not write that file (or must delete it with the visit).
- **Strangers who enter within 4 s are still treated as a possible party** (`LedgerConfig.group_window_s`). On DEV
  this pools unrelated shoppers. Turning it off gives 15 thefts and 7 reviews on honest shoppers against 16 and 4, so
  it stays. The better rule (a party walks together on the floor tracks) is not built.
- **`no_receipt_factor` stays 0.6.** Setting it to 1.0 was tried on the first join: 13 thefts against 12, 6 reviews on
  honest shoppers either way.
- **No concealment cue.** All flags are review tier. Alert tier needs an item camera to report concealment; the
  ledger and `store_events(conceal=...)` already take it.
- **`--keep` on the bench** leaves the run folders in place so the runner reuses its stored shelf events and person
  boxes. The recorded result (`results/bench_dev.*`) is from a run without it.
- **Batch, not live.** The runner tracks the whole clip, then associates. Undoing a swap relabels the last seconds.
- **Fragments merged and removed.** `docs/fragments/` is gone; result files moved to `results/` (`assoc_*.json`,
  `hand_detector.*`, `plates_bench.md`). The plates stream was complete and is merged too, with a `plates` extra in
  `pyproject.toml`.

## 2026-10-05: fixed benchmark on simulated clips (benchmark stream)

- **Layout: recommended-3d-45.** Its files are in `~/bree/software/shared/layouts`, so it is used instead of
  recommended-47: 45 cameras, two-view coverage, one entrance camera.
- **The private simulator copy `out/sim-copy` was refreshed** from the simulator at commit bff1f32 (the old copy
  had no entrance camera kind and no 3D layouts). Renders of training seeds made after this come from the newer
  simulator. `clip.json` records the copy's commit and the hashes of the two overlay files.
- **Scenario overlay, not simulator edits.** `scripts/bench/sim/bench.js` is loaded into the copy on top of
  `scripts/train/sim/synth.js`. It scripts each shopper (zone of each pick, look first, put back, walk before
  concealing, a second shopper at the same shelf) from its own seeded generator. The simulator's own shopper has
  no put-back and picks slots at random; that was not enough for the task.
- **Splits by seed.** Train 1000 to 4999 (the SKU detector's scenes 1000 to 1239 are inside it), scratch 5000
  to 5999 (seed 5001 was the first end to end clip and was looked at during development, so it is in neither
  dev nor test), dev 7001 to 7006, test 9001 to 9006. A test in `scripts/bench/test_bench.py` fails if they overlap.
- **Video, not PNG frames.** A clip is one H.264 file per camera (libx264 crf 16 from JPEG quality 0.92) at the
  camera's full resolution. PNG frames of the first clip took 12 GB for 8 cameras and 53 s; a benchmark clip has
  16 to 20 cameras. A real camera delivers H.264 too. The adapter uses the files as they are (no second encode;
  the old path re-encoded PNG to mp4v).
- **Which cameras are rendered.** Always the entrance camera, the four overhead cameras and the register camera.
  The item camera with the best view of each pick is never dropped. Second views are added until there are 12
  item cameras, then 2 item cameras that see no pick. Not every camera that sees a pick is rendered: a pick is
  usually in view of 3 to 5 cameras and rendering all of them would double the frame count.
- **Clip length.** The arrival window is the first of 30, 22, 38, 16, 46, 10, 4 s that makes the visit end
  between 60 and 90 s; if none does, the one closest to 75 s (seed 9003 ends at 59 s, seed 9004 at 94 s).
- **Calibration is the true pose.** `calibration.json` has the pose the scene was rendered with (layout pose plus
  the mounting error of synth.js, roll up to 0.02 rad included). That is what a calibrated install knows. The
  old run used the layout pose without roll; on a quick slice that was off by a median of 5 px and up to 71 px
  at the feet of a shopper, the calibration file by at most 2 px.
- **The planogram is exact.** `layout.json` gives each slot the SKU that is in it, including the single
  misplaced items synth.js puts in (12 percent of slots). A real planogram would not know those. Known ceiling:
  a pipeline that reads the SKU from the slot alone is scored as if misplaced items did not exist.
- **Register feed.** One receipt per paying shopper, 1.5 to 4.5 s after the payment, no dropped or mis-rung
  receipts. Receipts carry no shopper identity.
- **Ground truth cannot reach the pipeline by accident.** Truth lives in `truth/` inside the clip; the benchmark
  hands the runner a folder of links without it. The scorer is the only reader.
- **Identity for scoring comes from boxes.** A pipeline identity is matched to a true shopper where its boxes
  overlap the true person box (IoU 0.3), near the time in question. The runner logs track id to store-wide id
  per frame (`person_ids.jsonl`) by subclassing the identity layer; the frame log itself does not carry it.
- **Funnel stage "right slot" is the exact slot id.** A pick with the right SKU but no slot is counted lost at
  "right slot"; the "passed on its own" column shows the SKU stage separately.
- **A theft counts as caught at shopper level**: an alert on the thief after the concealment. "Alerted with the
  right SKU" is reported next to it.
- **A camera with nothing moving in view repeats its last frame.** The renderer skips the draw when no shopper,
  held item, nearby floor (shadow) or open cooler door can be in a camera's image and the scene has not changed
  since that camera's last such frame. Checked on a 20 s slice of seed 7001 with 4 cameras: 800 of 800 JPEG
  frames and the truth rows are byte identical to the renderer without the skip (256 of them were repeats).
  Dev clips 7001 to 7004 were rendered before the skip, 7005, 7006 and all test clips with it; `clip.json`
  records the overlay hash and `truth/render.json` the number of repeated frames.
- **The split renderer starts a clip again (up to 3 times) when Chrome closes mid clip.** That happened to
  seeds 7006 and 9002 on a loaded machine ("browser has been closed"), cause not found. A clip is only complete
  when `clip.json` exists; it is written last.
- **The baseline is kept under its own name.** `results/bench_dev_baseline.*` is the per-camera engine as
  committed at 19f51dc (`--runner bree.sim.bench:run_pipeline --name baseline`). `results/bench_dev.*` is
  whatever the default runner is at the time, so later runs do not overwrite the baseline.
- **Default runner left at the per-camera engine in this commit.** Another stream changed the default in the
  working tree to `bree.shelf.store:run`, a module that is not committed yet. The benchmark commit does not
  include that change so that `make bench-dev` works at this commit; the integrator switches the default when
  the new runner lands (or pass `BENCH_ARGS="--runner module:function"`).
- **No TEST numbers recorded.** The test command was only smoke checked on one clip with 60 frames per camera;
  that writes to `out/bench/test_smoke` and never to `results/`.

## 2026-10-05: shelf events without a person box (shelf-events stream)

- **Tuning and held-out clips are both TRAIN seeds.** 4900 to 4902 were looked at and tuned on, 4903 to 4906 were
  run once at the end. No DEV or TEST clip was used. Rendered with the benchmark generator
  (`scripts/shelf/render_train.sh`, which calls `scripts/bench/render_clip.mjs`).
- **A second shelf cue was added: the slot watch.** In these clips a slot is a row of identical items. Taking the
  front one leaves a picture that is almost the same, and the hand often stays in front of the slot until the
  item is put back, so the pixel comparison alone found 71 percent of the picks on clip 4900. The SKU detector
  sees stock on the shelf with tight boxes, so the box that fits "row position k" of a slot says how many items
  are gone. That is also the unit count.
- **Slot watch is off for cooler slots.** Behind the glass doors the detector's boxes shift when a door swings;
  on clips 4900 and 4901 that gave false takes. The pixel comparison reads coolers.
- **When the two shelf cues disagree on the slot, the pixel comparison's slot is kept.** On the tuning clips the
  slot watch was the one that named the neighbouring slot of the same SKU.
- **Held-item-only events are off by default** (`ShelfConfig.hand_only`). On clip 4900 they were 14 false out of
  15. The held item still confirms the SKU of a shelf change (`source: both`).
- **"Front item gone" rule is off by default** (`SlotConfig.gone_looks = 0`). It found one more pick on the
  tuning clips and added 11 false takes.
- **A take and a put of one slot within 2 s are dropped as a pair.** This removes cooler door swings. A real
  take and put back that fast nets to nothing for the ledger.
- **No guard against a swaying camera.** Some clips sway the camera (scene parameter `jitter`). The pixel
  comparison then reads false takes and puts once the view settles (clip 4902). A guard that froze reading while
  many patches were present was tried and removed: it also froze on the pieces of a standing shopper and lost
  real takes (pick recall on the tuning clips was 0.76 with it and 0.83 without).
- **The event time of a take is the pixel comparison's** (the moment the slot stopped looking like the shelf
  picture) when it saw the change. It was closer to the truth time than the first sighting of the item in a hand.
- **Unit count** comes from row positions when the slot watch saw the change, else from patch area capped by the
  slot's facings (1 for single-facing slots). Every pick in these clips is one unit, so counts above 1 are tested
  only in unit tests.
- **Scoring rule** (`scripts/shelf/eval_shelf.py`): a pair is found when the same camera reports the same kind of
  event within 3 s and 0.6 m of the truth slot. Right slot needs the exact slot id.
- **Acceptance restated after review.** The first report claimed right slot 0.915 on four held-out clips. Two
  more TRAIN-seed clips (4950, 4951) were added to the held-out set. Pooled over six clips right slot is 0.851,
  at the bar, and cooler cameras are under it. The neighbour-facing choice was not changed: there were three
  wrong-slot cases on record to work from and two of them were in held-out clips.
- **A shift of the whole picture is undone, not absorbed.** The picture is moved back onto the reference, so
  the slot map from the calibration stays valid. A new estimate is used only when it makes the picture at least
  15 percent closer to the reference; otherwise the shift of the frame before is kept while it helps. Without
  that a walking shopper switched the correction on and off.
- **Brightness is undone over a wide range (0.25 to 4) and the reference keeps its original brightness.**
  Scaling the reference down to a dim picture was tried and lost picks (recall 0.692 on the dimming check),
  because the difference threshold is fixed.
- **More than half the picture changed means unreliable.** Nothing is read; after 1 s without motion a new
  reference is taken and earlier takes are forgotten (their puts cannot be matched any more). If the new
  picture sits more than 12 px (half resolution) from the first one, the camera stops and says
  `needs_recalibration`.
- **Status records are a separate list, not part of the event stream**, so consumers of takes and puts need no
  change. `unreliable_windows(status)` turns them into spans per camera.
- **Repeat drop is narrow on purpose.** Same camera, same kind, same or neighbouring facing (0.1 m), within
  4 s, no opposite event between. A wider radius would drop real picks from nearby slots. Two real units taken
  from one facing within 4 s read as one event.
- **Strict precision is the number to quote.** The earlier precision counted every event near an act.
- **The all-cameras clip was not rendered** (started, then stopped to free the laptop). The limit is stated in
  the report and the command is given.

## 2026-10-05 hand-detector stream (hand and held-item detector, SIMULATED)

- Seeds. The benchmark manifest was there, so its TRAIN range (1000 to 4999) is used: training scenes 2000 to 2041,
  held-out scenes 3000 to 3009. Neither overlaps the first detector's scenes (1000 to 1239), dev (7001 to 7006) or
  test (9001 to 9006). `render_hands.mjs` refuses a seed outside the TRAIN range.
- The held-out split is a seed range (`--heldout-from 3000`), not the hash split of the first dataset, so the old and
  the new weights are both scored on scenes neither has seen. An eighth of the training seeds (by hash) is val.
- Scenario. The frames use the benchmark's scripted shopper behaviour (scripts/bench/sim/bench.js: reach, look, hold,
  put back, conceal, counter, two shoppers at one shelf) in randomised scenes (scripts/train/sim/synth.js: lighting,
  stock, camera mounting error). Neither file is edited; the hand overlay is a separate file.
- Hand label. The simulator's figures have no fingers: a "hand" is the last 7 cm of the forearm mesh, boxed from an id
  render (visible pixels only). Each hand box carries `reaching` (the arm is up at a shelf, cooler or counter) and
  `holding`. On real footage this class has to be relabelled and fine-tuned; the number here says the detector finds the
  end of a reaching arm in the simulator's views without a person box, nothing more.
- held_item is an attribute, not a class. Item boxes keep their SKU class; `kind` in the labels says shelf, backstock,
  hand (held) or counter. A second detector class per SKU for "held" would halve the examples per class. Whether an
  item is held is decided downstream from the hand box next to it and the shelf plane.
- Training frames are picked for the failure: when an arm is up, the item camera with the most pixels on the hand and
  one other camera that sees it; carried items and counter items less often; a few random item cameras. Tiles are cut
  on the pipeline's 640 px grid at native resolution plus two crops per frame at a random offset around a held item or
  a reaching hand.
- Every fourth rendered frame is used (captures are 0.3 s apart and look alike): 1764 training frames from 37 scenes.
  All 42 + 10 scenes were rendered so the scene variety is kept.
- Fine-tune, not retrain. The new weights start from sim_sku.pt (the first run took 321 minutes). Ultralytics resets the
  class outputs of every class when the class count changes; `keep_old_classes` in bree/train/train.py copies the 33
  SKU outputs back so only the hand class starts from zero.
- The first dataset's frames (seeds 1000 to 1239) are not mixed in: they show hands that have no hand label, which would
  teach the detector that those hands are background.
- sim_sku.pt stays the default and is not touched. New weights are versioned files next to it and are picked by name.
- Hand recall is reported at IoU 0.5 (the acceptance number), at IoU 0.3 and as "a hand box holds the true hand
  centre", because the shelf event contract needs a hand point (`hand_px`), not a tight box.
- Version to use: sim_sku_hands_v3 (v2 trained 3 more epochs on twice the frames). It meets the held-item target and
  not the hand target. A v4 run (3 more epochs) was stopped part way: the machine was shared with the other streams'
  renders and it had slowed to between 2 and 12 seconds a step. It was not evaluated and left no weights. The default
  stays sim_sku; switching is an environment variable, so the integrator decides after the dev benchmark.
- Hand-centred second look is an option of `detect`, off by default: it adds up to 6 tiles a frame for about one point
  on the held-item score and under three on the hand score.
- `caffeinate -i` around long runs: training steps stalled for minutes at a time twice, which looked like the laptop
  idling; the stalls stopped once it was kept awake.

## Association stream (2026-10-05)

- **One tracker on the floor plan, not per-camera tracks merged afterwards.** Each people camera's detections are put
  on the floor with its calibration and one constant velocity Kalman tracker follows people in metres. The per-camera
  ByteTrack ids were the cause of most of the fragmentation (22 and 18 ids on two cameras for 4 shoppers). Item cameras
  take no part in identity.
- **Placed by the hips.** Hip keypoints at 0.93 m, else shoulders at 1.42 m, else the box. A detection placed by the box
  alone can continue a track but not start one. Measured, not assumed: 0.09 to 0.17 m against 0.14 to 0.63 m.
- **Closed world kept.** Births at the door or in the first second; a track that starts elsewhere takes back a lost
  identity it could have walked from (two lost people both likely: uncertain); a steady track that fits nobody becomes
  a person flagged "entry not seen"; last seen in the doorway means left; 60 minutes unseen means dropped.
- **Two identities within 0.25 m are both marked uncertain** (`FloorConfig.encounter_m`). Position cannot tell who
  came out of that as whom. On DEV it marks 18 of the 20 identities that do cover two shoppers, and 29 shopper
  identities in all (of 47 made, 7 on a clerk), because the simulated shoppers walk through each other. Default on: a swap between a thief and an honest
  shopper is the worst error this system can make. Set 0 to turn it off (13 instead of 6 of 20 thefts at alert tier
  with a perfect conceal cue, no false alert on DEV, but 17 swapped identities unmarked).
- **Staff are whoever spends 70 percent of their time behind the counter** (the strip between counter and backbar,
  read off the layout). They make no events and are not candidates for a shelf event. A real store marks this zone.
- **Association cost is geometry only.** Stand point in front of the shelf face (0.5 m out, 0.25 m spread, 0.3 m along),
  never behind the face (the other aisle), shoulder within 1.5 m of the point, plus the wrist line of sight when an
  overhead camera has it. The median over a second around the event, so someone walking past does not win. The values
  are round numbers, deliberately not the simulator's exact 0.42 m.
- **Events close in time are solved together** (within 1.5 s, sharing a candidate, exhaustive over at most 6 events and
  4 candidates each): one person cannot take from two places further apart than an arm span plus the walk, and two
  takes in one second from different slots by one person cost a little more than one each.
- **Uncertain when the next best explanation is within 4 cost units** (about 7 to 1). The event still goes to the best
  person, with the others in `candidates`. The ledger already handles that: it halves the item and waits for the
  others' receipts. Doubt about who a person is travels separately as `meta.identity_uncertain` and caps at review.
- **Ledger untouched.** Unpaid at exit reaches review, a conceal cue raises it to alert. One uncorroborated pick still
  never alerts. PICK confidence is 0.9 for one source and 1.0 when shelf diff and hand item agree.
- **No conceal cue from overhead pose.** Tested on DEV and dropped: the feature does not separate concealing from
  walking in this simulator. `store_events` takes cues from item cameras instead.
- **The single-view pick rule is switched off per camera, not removed** (`rules.picks_from_shelf_events`): MERL and the
  toy clips still use it.
- **Files taken.** New: `src/bree/track/{floor,associate,people,scenes,floor_bench}.py`, `src/bree/events/shelf.py`,
  `tests/test_association.py`. Edited: `src/bree/events/engine.py` (the flag only). `multicam.py`, `calib/` and the
  ledger are unchanged.
- **Benchmarks read ground truth only to score**, and to turn true picks into shelf events for the stage test, which is
  labelled as such everywhere.

## 2026-10-05: fuel drive-off module (plates stream)

- **Open models, not a home-made reader.** `fast-plate-ocr` 1.1.0 (text) and `open-image-models` 0.6.0 (plate
  detector) both installed and downloaded their weights from GitHub releases with no sign-up, and both packages
  are MIT. The task said to build a classical localiser plus a small character reader trained on synthetic plates
  only if nothing could be downloaded, so the character reader was not built. The OpenCV localiser was built
  anyway (`ClassicalDetector`), because the detector weights are the part with the open licence question below.
- **Licence caveat logged, not resolved.** The MIT licence files cover the two packages. Neither project page
  states a separate licence for the weights or names the training data, and the detector is a YOLOv9 model: the
  original YOLOv9 code is GPL-3.0, an MIT implementation also exists, and the project does not say which one
  trained these weights. Counsel or the authors need to confirm before this ships. See readme.md.
- **A pump is one fuelling position.** That is the unit a forecourt controller reports sales for. A zone is a
  polygon in one camera's image; a vehicle is in it when the bottom centre of its box is inside.
- **Sales and payments come from the controller or POS, never from video.** The module does not guess that fuel
  flowed from pixels. Without a dispenser feed there is no drive-off rule.
- **Always review tier.** A drive-off alert is never the "alert" tier. Confidence is 0.6, plus 0.2 with a plate
  read, minus 0.2 when two vehicles shared the zone or no vehicle was seen. These three numbers are a first
  guess, not fitted to anything.
- **Grace period runs from the moment the vehicle leaves the zone**, default 120 s, because people move the car
  off the pump and then walk in to pay. A payment after the alert retracts it (existing retraction record) for
  up to 24 hours.
- **A box back in the same place within 60 s while the sale is unpaid is the same visit.** Otherwise a person
  walking in front of the camera would read as "vehicle left". No appearance check. Measured limit: a car hidden
  after fuelling for longer than the grace period still gives an alert, then a retraction when it pays (scenario
  `camera_blocked_past_grace`).
- **Verifier fix: a visit is a list of spans, and the next car is split off.** The first version merged any box
  back in the same place into the owed visit, reads included. A verifier showed that a paying customer who took
  the spot within 60 s of a drive-off got their plate stored on the thief's alert. Now the gap stays on record,
  the visit is split at the gap when a new sale starts at the pump after it or when the plates before and after
  differ by more than 2 characters, and the stored plate and clip come only from spans that overlap the
  fuelling time of the sale. So a later car's plate can not reach the store under any of these paths. Remaining
  limit, stated in the readme: unreadable thief plate plus a second car that buys no fuel delays the alert
  until that car leaves (the alert names the gap, no plate is stored).
- **Verifier fix: a gap or a track id change while fuel is flowing is never a departure.** The rejoin used to
  need a sale that had already ended, so a 6 s occlusion late in fuelling raised an alert on a car still at the
  pump. Now a new id in the same place as a track that just stopped carries on the same visit, and while the
  sale is still dispensing a box in the same place rejoins with no time limit (the hose is in the car).
- **Sales without a transaction id are merged** by pump and start time, or into the sale still dispensing.
  Payment by pump id is an advertised path, so such controllers are in scope.
- **A running monitor purges the plate store hourly.** Deletion used to run only on open, write and lookup. The
  monitor ticks on every frame, so it is the cheapest timer this stream owns. `sync_reviews` from the review
  server's timer is still the integrator's to wire.
- **Plate text lives only in the plate store.** The alert, its JSON and the review store carry a record id, not
  the plate. Plates of vehicles that paid are never written to disk. A test checks each of these.
- **Retention defaults: 72 hours, 30 days once a reviewer confirms, deleted at once on "not theft" or a late
  payment.** 72 hours covers a weekend shift change; 30 days gives time to file a police report. Both are
  settings stored in the plate database. Every lookup needs a name and a purpose and is logged; the log never
  holds the plate text.
- **Evidence clips follow the review store's rule (30 days), not the plate rule.** The clip shows the plate in
  pixels. If counsel wants clips on the short clock too, set the review store retention for the forecourt.
- **Vehicle boxes are scripted in the bench.** The bench tests plate reading and the rule. Vehicle detection is
  the repo's COCO YOLO (`yolo_vehicles`), which was not measured here: there is no forecourt footage.
- **Bench results:** `results/plates_bench.md` (the tables) and `out/plates/bench.json` (every number).
- **Synthetic plate fonts are macOS system fonts** (DIN Condensed, Arial Narrow). The bench and one test need
  them or a DejaVu fallback on Linux. Real plates use embossed dies; none were used.
- **Default plate detector is both (open detector plus OpenCV localiser).** On the synthetic bench the mean
  whole-plate rate over the 20 cells was 0.44 for both, 0.38 for the open detector alone and 0.37 for OpenCV
  alone. Both is better in daylight (0.58 against 0.44) and a little worse at night (0.57 against 0.61). On the
  scripted drive-offs it read 4 of 5 plates against 2 of 5. The open detector misses about a third of the drawn
  daytime cars; that may be the crude drawing and has to be checked on real footage before this choice is final.
- **A plate is stored only at vote confidence 0.9 or more** (it was 0.6). In the first full run one drive-off plate
  was stored wrong at confidence 0.67 (RUN2002 read as RUNN2022), so the threshold was first moved because of
  one observed failure. The check behind 0.9 is now part of the bench (`vote_calibration` in
  `scripts/plates/bench.py`, seed 20261006, never the reader table's seed; table in results.md): on 360
  synthetic plates (48 to 96 px, day, motion, night; 5 frames each) votes at 0.9 and up were right for 313 of
  319 (98%), 0.8 to 0.9 for 9 of 20 (45%), under 0.8 for 4 of 21 (19%). An earlier hand-run version of this check
  was quoted here as 96%, 65% and "mostly wrong"; it had no script and is replaced by these numbers. The "0
  stored wrong" on the scripted drive-offs is weak evidence by itself: those are the same few plates that
  prompted the change. 6 wrong in 319 also means the threshold does not make a stored plate certain; the
  reviewer has the crop.
- **Tried and not kept: blur plus contrast stretch on dark crops.** On 60 crops per cell it lifted night_noisy at
  160 px from 13% to 45% whole plate, left 64 and 96 px at 0 to 2% and was
  neutral in day and night. One cell is not enough to add a step tuned on the bench's own noise model.

## Improvement round 2 (2026-10-05)

- **No rule change was kept, because none held on both sets.** Four join and ledger rules were measured on DEV and
  on the TRAIN-seed clips (REPORT.md, round 2). One of them (a put with the item seen in the hand returns the take
  from any slot the reach was read at) gives 16 of 20 thefts and 2 of 22 honest reviews on DEV, which meets every
  bar. It was reverted: it is right in 2 of 7 uses on DEV and costs 2 of 13 flagged thefts on the TRAIN-seed clips
  for 1 honest review. To overrule: the change is small (keep the places of merged readings in
  `one_act_per_reach`, match puts against them in `confirm_puts`).
- **The TRAIN-seed clips are a second tuning set (`make bench-train`), not a test set.** The per-camera shelf rules
  were tuned on 4900 to 4902 and the scenes are harder than DEV by construction, so its numbers are not comparable
  to DEV one to one. It is used to check that a rule chosen on DEV also helps on clips it was not chosen on. A rule
  is kept only when it does not make either set worse.
- **`bench train` reads whatever TRAIN-seed clips are rendered** under `data/synth/bench/train` (the manifest gives
  train as a seed range, not a list). The clip list is printed in the result.
- **`results/bench_dev.md` was written again from the stored shelf events and person boxes** (`--keep`): nothing
  before tracking changed in this round.

## Improvement round 3 (2026-10-05)

- **A put event is read as what it is: one camera saying one place looks as it did before that camera's own take.**
  It carries the name of that take (`undoes`), and the join removes exactly that reading from the act it was merged
  into. Before, a put was matched to a take by place and product after the cameras had been merged, which let an arm
  leaving a slot return a stolen item, or left a real put-back with nothing to return. Reason: on the TRAIN-seed
  clips the take's place is back in 82 of 85 pixel puts, real put-back or not, so the event is right and the
  reading of it was wrong.
- **An act another camera still reads stands, unless the item was seen going into the slot.** Measured both ways:
  never returning such an act leaves 5 honest reviews on each set; returning it whenever any item was seen near the
  put drops DEV to 12 of 20 thefts. "Going in" (the item track ends at the slot, from further away) is right for 19
  of 28 puts on the TRAIN-seed clips. Its two thresholds (`ShelfConfig.put_in`) come from those clips.
- **`from_last` is in the code but off, although it meets every DEV bar (17 of 20, 2 of 22).** It loses one flagged
  theft on the TRAIN-seed clips (15 to 14). Same standard as round 2: a rule is kept only when it makes neither set
  worse. To overrule: `one_act_per_reach(..., from_last=True)` in `bree.shelf.store.rejoin`, one line.
- **The door rule for the floor tracker was reverted.** It clears the 7001 review but loses a flagged theft on DEV
  and changes nothing on the TRAIN-seed clips.
- **Dropping a camera's own take and put pair before merging was not taken** (16 of 20 and 13 of 19): a take another
  camera still reads protects stolen items, and a live system cannot drop a take it has already reported.
- **Reading names are amended after an event was handed out** (a repeat dropped in the camera adds its name to the
  kept event). The clip runner writes events at the end, so this is free here; a camera node that streams events has
  to send the amendment. Marked in `ShelfCamera._absorb`.
- **The two results files were made from scratch**, not with `--keep`: the shelf events changed.

## Improvement round 4 (2026-10-05)

- **The stage worked on was identity, not the three honest reviews one by one.** Round 3 named it; checking every
  hand-back against the true shopper showed the cause (a second track on one person becomes "a new person", the
  original track with the picks is lost for ever). Fixed at the tracker: a track that starts inside within 1.0 m of
  a tracked person with nobody lost gets no identity. Threshold from the TRAIN-seed clips.
- **Pixel-only takes within 0.75 m of the pay point are not passed to the ledger.** Uses `poi.register` of the
  layout (where the payer stands; a real store marks it at install). 0 of 5 real on the TRAIN-seed clips, 0 of 4 on
  DEV. A take there with the item seen in a hand still counts.
- **The DEV goal is reported as not met although one setting meets every DEV bar.** `standing` plus
  `ambiguous_factor` 1.0 gives 17 of 20 and 2 of 22 on DEV, and 15 of 19 and 4 of 36 on the TRAIN-seed clips (3 of
  36 now). Same standard as rounds 2 and 3: a rule is kept only when neither set gets worse. To overrule:
  `one_act_per_reach(..., standing=True)` and `"ambiguous_factor": 1.0` in the ledger settings of
  `bree.shelf.store.rejoin`.
- **The clothing gate for hand-backs is in the code but off** (`FloorConfig.other_m`, set 2.0 to turn on). It is
  right about identity (12 of 25 wrong hand-backs refused, 0 of 45 right ones, TRAIN-seed) and costs one theft and
  one honest review there, because flags that an identity error had produced disappear. It should go on together
  with a look at the ledger's doubt discounts, not before.
- **"A put cannot return an act still read as taken afterwards" was reverted.** Reading times are late by seconds, so
  "afterwards" is not reliable.
- **DEV was run from scratch, the TRAIN-seed clips with `--keep`.** Nothing before tracking changed this round, so
  the stored shelf events and person boxes are the ones a fresh run would produce.


## Improvement round 5 (2026-10-05)

- **The stage worked on was the ledger's doubt discount**, named first by round 4. A crowded pick is now halved only
  while another candidate's receipts can still arrive; after that it scores in full and is review tier at most
  (`LedgerConfig.ambiguous_settles`, set False for the old rule).
- **A pick in doubt can be covered by anybody's paid item that nobody saw them take**
  (`LedgerConfig.doubt_takes_any_extra`). Risk accepted: a thief whose pick is in doubt is cleared if a stranger paid
  for the same product and their own pick of it was missed. It did not happen on DEV or the TRAIN-seed clips (9
  firings, no flagged theft lost). It applies only to picks in doubt, never to a pick one person clearly made.
- **`standing` is on.** With the two ledger rules neither set gets worse against round 4 (DEV 17 and 2 from 17 and
  3, TRAIN-seed 15 and 2 from 15 and 3), the standard of rounds 2 to 4. On the TRAIN-seed clips it trades one flagged
  theft for another (4901 P001 lost, 4900 P003 gained).
- **"A take after a put with the item seen going in is a new take" was not kept** (TRAIN-seed 16 and 3 against 15 and
  2). Fewer honest reviews was preferred over one more theft, because the honest bar is the one the goal was missing.
- **DEV was rejoined with `--keep`, not run from scratch.** Only the join and the ledger changed (git diff 875fa28
  to db103d3 touches `src/bree/ledger/ledger.py` and one default in `src/bree/shelf/store.py`), so the stored shelf
  events and person boxes of round 4's from-scratch run are what a fresh run would produce. Saves about 35 minutes.
- **Tests:** full suite on the kept code, 415 passed, 1 skipped, 0 failed (counted from the progress lines of `out/bench/round5/tests_full.log`).
- **goal_met is reported true for DEV** with the caveat that one case moves the honest rate by 0.45 per 10 and theft
  recall by 0.05, and that alert tier is 0. TEST had not been run at that point (it was run once afterwards, see the wrap-up below).


## Wrap-up (2026-10-05)

- **The audit's three major findings were fixed in the report, not by re-rendering the benchmark.** Re-exporting the
  clips with a nominal planogram would make a new benchmark version and a second look at TEST. The stored runs were
  re-scored where that needs no rerun.
- **Right SKU is now reported twice: exact planogram (as benchmarked) and nominal planogram**
  (`scripts/bench/nominal_sku.py`, scoring only). The simulator exports no "misplaced" flag, so the nominal SKU of a
  slot is the most common SKU of its planogram block, with block borders from the simulator's unshuffled layout
  (`/Users/kiromoussa/bree/software/shared/example-layout.json`, same 2,349 slots in the same order). A block with a
  tie keeps its exact SKU (10 slots on DEV, 0 on TEST). Result: DEV 0.838, TEST 0.859 against the 0.85 bar. The
  goal is therefore reported as met on TEST by one pick and missed on DEV by one pick for this bar under a nominal
  planogram; under the exact planogram it is met on both.
- **Re-scoring TEST this way is not tuning.** No code, rule or threshold in the pipeline changed after the TEST
  run; the script reads stored outputs and truth and changes only how one number is counted.
- **The camera choice limit is stated with a measured idle-camera rate and one all-camera clip** (12 PICK events against 11, no extra review) (`scripts/bench/idle_cameras.py`,
  `out/bench/allcams`). The all-camera clip is TRAIN seed 4903, not a DEV or TEST seed, so the benchmark sets stay as
  they were. One clip is one clip; the benchmark itself should be re-rendered with every camera in its next version.
- **Plates:** the fuel drive-off fragments were already merged into the shared docs and committed before this
  wrap-up; `docs/fragments` does not exist any more. Nothing left to merge.
- **No new Makefile targets** for the two scoring scripts; the commands are in their docstrings and in REPORT.md.
- **Tests at wrap-up:** full suite 416 passed, 0 failed, 0 skipped (the browser test ran, `BREE_PLAYWRIGHT` was set),
  counted from the progress lines of `out/bench/wrapup/tests_full.log` because pytest prints no closing summary line
  here; `scripts/bench/test_bench.py` 3 passed. No file under `src/` changed in the wrap-up.

## Benchmark version 2 (2026-10-05, benchmark agent)

SIMULATED data only. Code: `scripts/bench/`, `src/bree/sim/bench.py`. No pipeline file was changed.

- **Generator 2 is a second path, not a rewrite.** `render_clip.mjs --gen 2` and `setup(..., { gen: 2 })` in
  `sim/bench.js`. Generator 1 keeps its stream of random numbers: a dry run of TRAIN seed 4903 gives the stored
  truth events and the stored camera list. DEV and TEST stay as rendered, as regression sets.
- **The simulator copy was refreshed to commit 1f5586e** (camera pass, day and night). The old copy is kept in
  `out/sim-copy.bff1f32`. Other streams render from `out/sim-copy` too (`scripts/train`, `scripts/conceal`): with
  the camera pass off, which is the headless default, the simulator says it renders as before, but the store art
  changed (price tags on the shelf rails), so frames from before and after the refresh are not pixel equal.
- **Seeds.** dev2 is 11001 to 11020, the checkpoint pool 13001 to 13030. Scratch seeds 5101 to 5108 were used to
  build and look at the generator. No dev2 or checkpoint frame was looked at while building it.
- **Camera pass at strength 0.5 for every effect, people and shelf mess left to synth.js.** The simulator's
  "people" and "shelves" switches rebuild figures and shelves that synth.js already randomises and checks against
  the store build order; switching them on risked breaking the id render for no test value.
- **Lens distortion is in the picture and its coefficient is in `calibration.json`.** The pipeline's camera model
  is a pinhole. This will cost it accuracy until it undistorts, most on the 120 degree overhead cameras. That is
  the point of the item, and the coefficient a real calibration would measure is given, so it is fair.
- **Truth under the lens.** Boxes come from an id render at half size with the field widened, each id pixel moved
  through the lens model; hand, head and feet points go through the simulator's own `distortPoint`. Checked by
  drawing them on stills of scratch seed 5101 (overhead and register camera).
- **Idle frames are still repeated.** Measured on an idle machine: about 50 ms for a drawn 4 MP frame, of which
  28 ms is the JPEG and 16 ms the readback, so drawing an empty view again only for fresh noise would cost nearly
  as much as a busy frame. The noise of an empty view therefore stands still. Stated in the README.
- **Cameras by aisle, by main view.** First rule tried: a camera belongs to an aisle when it sees 8 slots of it.
  Ceiling and drop-rod cameras see slivers of several aisles, so three aisles pulled in 35 to 40 of 45 cameras.
  Rule kept: a camera belongs to the aisle it sees most slots of. Which aisles a clip uses is drawn from the seed
  before any shopper exists.
- **Nominal planogram is what the pipeline is given.** `layout.json` of a generator 2 clip holds the most common
  product of each planogram block; the exact content is in `truth/planogram.json`. So "right SKU" on dev2 needs
  no re-scoring script. `nominal_sku.py` stays for the generator 1 sets.
- **A dropped scan is scored as an honest shopper.** The shopper paid; the feed lost the line. A review on that
  shopper counts as a review on an honest shopper and is also shown under its own tag, because a pipeline cannot
  be expected to clear it without seeing the item on the counter. An alert on it would be wrong.
- **Staff come in through the front door** (a vendor restocking, or an employee starting a shift) in a work vest
  or the clerk's shirt. The uniform is a cue a real pipeline could also use. A PICK event on a staff take is
  counted as a true shelf event for pick precision and reported apart; it is "wrongly counted" when a record
  lists it as unpaid.
- **In a group the carrier is honest.** The item was paid for by the companion. A review on the carrier counts
  against the honest review rate. A paid pick counts as "pay classified" when the PAY event is on the payer.
- **Stress is applied to the pipeline's inputs, never to the clip.** Camera drop, pose noise, receipt delay and
  planogram errors are drawn from (stress seed, clip seed). The register camera is never dropped. Pose noise goes
  into `calibration.json` (what the pipeline reads) and noise of the same size into the camera poses of
  `layout.json`; the two are not the same draw.
- **Intervals.** Percentile bootstrap, 2000 draws, seed 0. Shoppers are the unit for rates about people and their
  picks; clips are the unit for pick precision, put-back precision and identity numbers.
- **Checkpoint seeds are marked used before they render.** A crashed render still spends the seed.
- **Not done:** no Makefile target (the Makefile is not this stream's file; commands are in
  `scripts/bench/README.md`), no change to README.md or REPORT.md at the top level.
