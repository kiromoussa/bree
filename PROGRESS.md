# PROGRESS

Read this first after a context reset. Then BUILD_SPEC.md. Phase 2 instructions: ~/Downloads/BREE_PHASE2.md.

## PHASE 2 BLOCKERS FOR KIRO (updated 2026-09-30 19:05 UTC)
1. **GPU quota is 0 in every region checked, and the automatic quota requests were REJECTED** (code `ContactSupport`, 19:3x UTC). Azure wants a support ticket for this subscription. Kiro: portal -> Help + support -> Create a support request -> Issue type "Service and subscription limits (quotas)" -> Quota type "Compute-VM (cores-vCPUs) subscription limit increases" -> East US -> `Standard NVADSA10v5 Family vCPUs` = 72 and `Standard NCADSA100v4 Family vCPUs` = 24 (or `Standard NCadsH100v5 Family vCPUs` = 40). Mention it's a sponsored (startup credits) subscription. Claude also filed smaller automatic requests (A10 36 / A100 24 in westus3, southcentralus, eastus2; T4 8 in eastus); check them with `az rest --method get --url "https://management.azure.com/subscriptions/<sub>/providers/Microsoft.Compute/locations/<region>/providers/Microsoft.Quota/quotaRequests?api-version=2023-02-01"`.
   Once quota exists: `NGC_API_KEY=... scripts/azure_gpu.sh sim <region>` and `scripts/azure_gpu.sh train <region>`; `scripts/azure_gpu.sh stop` deallocates everything.
   Original request details: (eastus, eastus2, westus2, westus3, southcentralus, northcentralus) on subscription "Azure subscription 1" (Sponsored, id dc81c28d...). Claude filed two quota requests via `az quota update` at 18:54 UTC, **East US**:
   - `StandardNVADSA10v5Family` -> 72 vCPUs (A10 for Isaac Sim) — status: InProgress
   - `StandardNCADSA100v4Family` -> 24 vCPUs (1x A100 80GB) — status: InProgress
   If they get rejected or sit in review: portal -> Quotas -> Compute -> East US -> request the same two (fallback `StandardNCadsH100v5Family` -> 40). Check: `az vm list-usage --location eastus -o table | grep -iE "NVADSA10|NCADSA100|NCADSH100"`.
2. **No `~/.kaggle/kaggle.json`** -> Simuletic and DCSASS not downloaded.
3. **No `$NGC_API_KEY`** -> Isaac Sim container can't be pulled even once quota exists.
4. No `data/staged/`, `data/cloudstation/`, `data/operator/` footage yet.

## Phase 2 run log
| Time (UTC) | Step | Result |
|---|---|---|
| 18:50 | Phase 0 preflight | az OK (1 sub, Sponsored). kaggle/gdown/tmux/wget installed. Build machine: Apple M1 Max, 10 cores, 64 GB, MPS, 287 GB free. |
| 18:54 | GPU quota | 0 everywhere; requests filed (above). Running Phase 1/4/5 locally on the Mac until quota lands. |
| 19:00 | Phase 1 downloads started | PoseLift (official Drive), RetailS (Drive), MERL (wget), UCF-Crime (Dropbox zip), SKU-110K (Ultralytics). Logs: data/logs/ |

| 19:10 | Shadow mode (Phase 5) | done by a subagent in a worktree, merged (commit 7d04317). 88 tests pass. |
| 19:30 | Downloads | gdown --folder gets rate-limited by Drive: use `scripts/drive_fetch.py` (embeddedfolderview + drive.usercontent direct URLs). PoseLift pickles done (151 files: 104 train + 47 test; README says 153). RetailS done (960 MB zip, 5.1 GB). MERL done (106 videos). UCF-Crime: official page dead; Dropbox per-file links used (Part-4 = Shoplifting, Testing_Normal_Videos), slow. SKU-110K paused (bandwidth). |
| 20:00 | Leak found | RetailS_test_realworld == PoseLift test (same 47 clips, identical keypoints). Never evaluate a PoseLift-trained model on it. |
| 20:30 | Concealment classifier | `src/bree/conceal.py`, `scripts/conceal_experiment.py` (MPS). PoseLift has NaN keypoints: cleaned at load. |

## Next steps (exact)
1. When `results/conceal_poselift.json` exists: `.venv/bin/python scripts/eval_retails.py 60` -> results/conceal_retails.json.
2. When `data/logs/merl_measure.log` ends: results/merl_measure.json.
3. When both UCF zips are complete (`unzip -t`): `.venv/bin/python scripts/eval_ucf.py 40` -> results/conceal_ucf.json.
4. `.venv/bin/python scripts/measured_error_rates.py`, then (machine idle) `.venv/bin/python scripts/speed.py`, then `make bench`, `make test`.
5. data/README.md, REPORT.md, adversarial review, commit + push.

## Azure VM hours
| VM | Size | Hours | State |
|---|---|---|---|
| (none created) | | 0 | |

## Phase 1 (last night) notes below

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
