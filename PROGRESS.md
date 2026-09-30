# PROGRESS

Read this first after a context reset. Then BUILD_SPEC.md.

## Status
| Stage | State |
|---|---|
| 1 Repo setup | done (pyproject, Makefile, setup.sh, hw detection, DECISIONS) |
| 2 Perception pipeline | done: `bree run --source X` (YOLO26n detector + crop pose + ByteTrack people / centroid products, annotated mp4, frames.jsonl, FPS) |
| 3 Events + ledger | done: engine + ledger + payments (jsonl/stdin/http) + alerts w/ head-pixelated clips; `make demo` on 8 toy clips |
| 4 Sim + eval | done: event-level simulator + noise model + metrics + `make bench` (results/bench.json, results/bench.md); Isaac Sim README + draft IRA config in src/bree/sim/isaac/ |
| 5 Extras | dashboard (done), ONNX export (done), multi-camera handoff (library + tests), conceal classifier (skipped: no usable labelled data) |

## Environment notes
- venv: `.venv` (uv). Weights in `models/` (yolo26n.pt, yolo26n-pose.pt, + onnx).
- CPU only, 4 cores. Kaggle/Mendeley/HF/Google Drive blocked (403). GitHub + PyPI OK.
- Do NOT run CPU-heavy things concurrently with `make bench` (FPS numbers get contaminated).

## Data
- PoseLift: a subagent cloned a third-party mirror into data/poselift (+ data/poselift_repo). The auto-mode safety check then BLOCKED using the unofficial mirror. DO NOT use data/poselift in any result. Tell the user in REPORT.

## Status at end of night
- All stages done except the concealment classifier (skipped, no labelled data). 80 tests pass.
- Final `make bench` run done; REPORT.md written from results/bench.json + results/export.json and fact-checked by an independent reviewer; ledger + engine hardened after two adversarial reviews (see DECISIONS.md, last section).

## Next steps (for the pilot)
1. Get pilot footage + POS export; draw zones per camera; run `bree run` in shadow mode.
2. Hand-label a few hours -> measured VisionNoise rates -> re-run `make bench`.
3. Store-specific product detector; wire multicam into `bree run`; alert retraction on late receipts.
