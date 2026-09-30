# BREE Vision: Overnight Build Prompt

You are building the first version of BREE's store vision system in this new, empty repo. Work autonomously through the night. I will not be around to answer questions, so when something is ambiguous, make the most reasonable call, write down what you decided in `DECISIONS.md`, and keep going. Commit early and often with clear messages.

## Context

BREE retrofits the coolers and shelves a store already owns so it can sell without a cashier. The same on-device vision that lets a shopper grab an item and walk out also tracks exactly what leaves the shelf, so theft gets caught automatically.

Our first real pilot is with an operator who owns 35 gas stations. We are piloting in 2 of them. **Theft detection is what he cares about most**, so it is the lead metric. If the pilot goes well, he connects us to 40 to 50 more operators. Everything tonight should move us toward a system we could put on a camera in a gas station convenience store.

The core idea: theft is not a separate product. It falls out of the same pipeline as cashierless checkout. If we know which items each person took off the shelf and we know what was paid for at the register (or via card/RFID tap at the cooler), then anything taken and not paid for is shrink.

## How this run works

This file lives in the repo as `BUILD_SPEC.md`. You are running under `/goal` with ultracode on and auto mode, and nobody is watching. Work like it:

- **Never stop to ask me anything.** Decide, log it in `DECISIONS.md`, keep going.
- **Keep `PROGRESS.md` current.** After every stage and every major decision, update it: what's done, what's in progress, what's blocked, and the exact next step. Your context will get compacted overnight. `PROGRESS.md` is how you pick back up without redoing work. When you start a turn and aren't sure where you are, read `PROGRESS.md` and `BUILD_SPEC.md` first.
- **Use workflows where they help.** Good fits: downloading and inspecting datasets in parallel with building the simulator; writing tests for each event type in parallel; an adversarial review pass at the end where independent agents try to break the ledger logic and check every number in `REPORT.md` against the actual benchmark output. Don't fan out work that has to happen in order.
- **Prove things in the transcript.** When you finish a stage, run the relevant command (`make test`, `make bench`, `make demo`) and show the output. The goal evaluator only sees what you print, so claims without output don't count.
- **Commit after every stage** with a clear message, so a crash never loses more than one stage.
- If the same approach fails twice, try a different one. If a whole stage is blocked for 45 minutes, write down why in `PROGRESS.md` and move to the next stage.

## What to build tonight

Build in this order. Each stage should work end to end before you move to the next. If you run out of time, a working stage 1 to 3 is far more valuable than a half-built stage 5.

### Stage 1: Repo setup
- Python 3.11+, `uv` or `pip` with a `pyproject.toml`
- Structure roughly like:
  ```
  bree-vision/
    src/bree/
      ingest/        # video files, RTSP streams, webcam
      detect/        # person, product, hand detection
      track/         # multi-object tracking
      pose/          # pose keypoints
      events/        # shelf zones, pick/put-back, concealment, exit
      ledger/        # per-person basket + reconciliation against payments
      alerts/        # theft alert output (JSON + annotated clip)
      sim/           # synthetic event + scene simulator
      eval/          # metrics and benchmark harness
    configs/         # store layouts, zones, camera configs (YAML)
    data/            # gitignored; datasets and clips
    scripts/         # download, run, benchmark
    tests/
  ```
- `README.md` with how to install, how to run on a video file, how to run the benchmark
- `Makefile` or `justfile` with `setup`, `test`, `demo`, `bench`
- Detect the hardware first (CUDA GPU, Apple Silicon MPS, or CPU only) and pick model sizes accordingly. Record what you found in `DECISIONS.md`.

### Stage 2: Baseline perception pipeline
- Detection: Ultralytics YOLO (latest available version, small/nano variant) for people and products
- Pose: YOLO pose model for body and wrist keypoints
- Tracking: ByteTrack (via Ultralytics or `supervision`) so each person keeps a stable ID
- Input: a video file path or a webcam/RTSP URL
- Output: annotated video with boxes, track IDs, and keypoints, plus a per-frame JSON log
- It must run on a single consumer machine. Log FPS.

### Stage 3: Store events and the ledger (the core of BREE)
This is the most important stage. Spend real effort here.
- **Zones**: store layout defined in YAML. Shelf zones, cooler zones, register zone, exit zone, as polygons in camera pixel coordinates. Write one example config for a small gas station convenience store: 2 to 3 shelf runs, 1 cooler bank, 1 register, 1 door.
- **Events** to detect per tracked person:
  - `pick`: wrist enters a shelf/cooler zone and leaves holding an item
  - `put_back`: item returned to a shelf zone
  - `conceal`: item disappears near the torso, pocket, or bag while the person is away from the register (use pose keypoints plus item track loss)
  - `pay`: person dwells at the register zone, or a payment event arrives from an external source
  - `exit`: person leaves through the exit zone
- **Ledger**: each person has a running basket (items picked minus items put back). On `exit`, reconcile the basket against payment events.
  - Paid for everything: no alert
  - Items taken and not paid for: theft alert, with confidence, the item(s), timestamps, and a short clip
  - Any `conceal` event raises confidence
- **Payment input**: build a simple interface for payment events (JSON over stdin, a file, or a local HTTP endpoint) so we can plug in a real POS or card/RFID reader later. Include a mock payment source for testing.
- Keep the event logic rule-based and readable for v1. It is fine if it is heuristic. It is not fine if it is a black box nobody can debug.

### Stage 4: Simulation and evaluation harness
We have no real store footage yet, so we need a way to test.
- **Event-level simulator** (`src/bree/sim/`): generate thousands of synthetic shopper sessions as event streams with ground truth. Include normal shoppers, put-backs, people who pay for some items but not all, concealment, crowded moments where two people reach into the same cooler, and people who linger without buying. Run the ledger on these and measure accuracy. This tests the logic independent of vision.
- **Video-level evaluation**:
  - Write `scripts/download_data.sh` that pulls public datasets. Try, in order:
    - Simuletic synthetic CCTV shoplifting dataset on Kaggle (`simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm`)
    - DCSASS shoplifting dataset (Mendeley)
    - PoseLift (pose-based real shoplifting dataset, check its paper/repo for access)
  - If a download needs credentials (Kaggle API key) or manual approval, do not get stuck. Write clear instructions in `data/README.md` for me to do it in the morning, and continue with whatever you can get.
  - If you cannot get any video data at all, generate a small set of synthetic test clips yourself (for example, rendered 2D scenes with moving boxes and a "product" object) just to prove the pipeline runs end to end, and label them clearly as toy data.
- **Metrics** (`make bench` prints a table and writes `results/bench.json`):
  - Theft event precision and recall
  - False alerts per hour of normal shopping (the operator will care about this a lot; false accusations kill a pilot)
  - Basket accuracy (did we get the right items per person)
  - Pipeline FPS and end-to-end latency from event to alert
- **Isaac Sim prep** (do not try to run it): write a `sim/isaac/README.md` and a draft NVIDIA Isaac Sim Replicator Agent config describing a gas station convenience store scene, camera positions, and shopper behaviors (normal, put-back, concealment). This is for later on a GPU machine.

### Stage 5: If you still have time
- A second pass on concealment detection: a small classifier over pose keypoint sequences, trained on whatever labeled data you got, compared against the rule-based version
- Multi-camera support (one person handed off between two cameras)
- A tiny local web dashboard showing live alerts and the per-person basket
- Export path for edge hardware (ONNX or TensorRT), with notes on what device we would ship

## Rules
- **Never fabricate results.** Every number in the report must come from a run you actually did, on data you name. If something did not work, say so plainly. We have been burned by inflated numbers before and I will not send anything to an operator or investor that is not real.
- **Privacy by default.** Do not store or log faces. Prefer pose keypoints and track IDs. Alert clips should be short and only for flagged events. Note any privacy-relevant choices in `DECISIONS.md`.
- Write tests for the ledger and event logic (`pytest`). The vision parts can have smoke tests.
- Pin dependency versions.
- Don't spend more than about 45 minutes stuck on any one blocker. Write it down and move on.
- Keep code readable. A new engineer should understand the event logic in one sitting.

## Deliverable for the morning

Write `REPORT.md` at the repo root with:
1. What works right now, and the exact commands to see it
2. Benchmark table with real numbers, what data they came from, and what they do and don't prove
3. What didn't work or got skipped, and why
4. Anything you need from me (API keys, dataset approvals, hardware, footage)
5. Your honest take on the top 3 things to do next before we put this in the 2 gas stations
6. A short list of what we should ask the operator for: camera angles and models, a few hours of existing CCTV footage, how his POS exports transactions, and which items get stolen most
