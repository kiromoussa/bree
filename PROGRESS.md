# PROGRESS

Read this first after a context reset. Then BUILD_SPEC.md.

## Status
| Stage | State |
|---|---|
| 1 Repo setup | done (pyproject, Makefile, setup.sh, hw detection, DECISIONS) |
| 2 Perception pipeline | not started |
| 3 Events + ledger | engine + ledger written; tests pending |
| 4 Sim + eval | not started |
| 5 Extras | not started |

## Environment notes
- venv: `.venv` (uv). `source .venv/bin/activate`. Weights in `models/` (yolo26n.pt, yolo26n-pose.pt).
- CPU only, 4 cores. Kaggle/Mendeley/HF blocked (403). GitHub + PyPI OK.
- A background agent investigated dataset access (PoseLift etc.) -> results go into data/README.md + scripts/download_data.sh.

## Next step
Write tests for ledger + engine; then Stage 2 pipeline (ingest/detect/track/pose + annotated video + per-frame JSON).
