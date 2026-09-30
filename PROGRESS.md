# PROGRESS

Read this first after a context reset. Then BUILD_SPEC.md.

## Status
| Stage | State |
|---|---|
| 1 Repo setup | done (pyproject, Makefile, setup.sh, hw detection, DECISIONS) |
| 2 Perception pipeline | done: `bree run --source X` (detector + crop pose + ByteTrack, annotated mp4, frames.jsonl, FPS). 8.9 FPS on vtest.avi (CPU) |
| 3 Events + ledger | engine + ledger written; tests pending |
| 4 Sim + eval | not started |
| 5 Extras | not started |

## Environment notes
- venv: `.venv` (uv). `source .venv/bin/activate`. Weights in `models/` (yolo26n.pt, yolo26n-pose.pt).
- CPU only, 4 cores. Kaggle/Mendeley/HF blocked (403). GitHub + PyPI OK.
- A background agent investigated dataset access (PoseLift etc.) -> results go into data/README.md + scripts/download_data.sh.

## Next step
Toy renderer + toy backend (src/bree/detect/toy.py, src/bree/sim/toy_render.py) -> demo; tests for ledger + engine; then event-level simulator.

## Data
- PoseLift: a subagent cloned a third-party mirror into data/poselift (+ data/poselift_repo). The auto-mode safety check then BLOCKED using the unofficial mirror. DO NOT use data/poselift in any result. Tell user in REPORT.
