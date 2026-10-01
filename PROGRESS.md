# PROGRESS

Read this first after a context reset. Then BUILD_SPEC.md. Phase 2 instructions: ~/Downloads/BREE_PHASE2.md.

## PHASE 2 BLOCKERS FOR KIRO

**Quota support ticket #2610010040000169 OPEN** (created by Kiro 2026-10-01 00:52 UTC, Developer plan): East US, NC A100 v4 Series -> 24, NVads A10 v5 Series -> 72. Check: `az vm list-usage --location eastus -o table | grep -iE "NVADSA10|NCADSA100"` and `az support in-subscription tickets show --ticket-name 06bfd9d3-e12e3d1d-c092d2b3-d295-46b1-9c08-0b564dde23b0`. When limits are > 0: `scripts/azure_gpu.sh train eastus` and `NGC_API_KEY=... scripts/azure_gpu.sh sim eastus`. (updated 2026-09-30 19:05 UTC)
1. **GPU quota is 0 in every region checked, and the automatic quota requests were REJECTED** (code `ContactSupport`, 19:3x UTC). Azure wants a support ticket for this subscription. Kiro: portal -> Help + support -> Create a support request -> Issue type "Service and subscription limits (quotas)" -> Quota type "Compute-VM (cores-vCPUs) subscription limit increases" -> East US -> `Standard NVADSA10v5 Family vCPUs` = 72 and `Standard NCADSA100v4 Family vCPUs` = 24 (or `Standard NCadsH100v5 Family vCPUs` = 40). Mention it's a sponsored (startup credits) subscription. Claude also filed smaller automatic requests (A10 36 / A100 24 in westus3, southcentralus, eastus2; T4 8 in eastus): **all 9 requests Failed** by 19:31 UTC, even the 8-vCPU T4, so automatic approval is off for this subscription and only a support ticket will work. Claude did not file the ticket because it needs Kiro's contact details. Check any request with `az rest --method get --url "https://management.azure.com/subscriptions/<sub>/providers/Microsoft.Compute/locations/<region>/providers/Microsoft.Quota/quotaRequests?api-version=2023-02-01"`.
   Once quota exists: `NGC_API_KEY=... scripts/azure_gpu.sh sim <region>` and `scripts/azure_gpu.sh train <region>`; `scripts/azure_gpu.sh stop` deallocates everything.
   Original request details: (eastus, eastus2, westus2, westus3, southcentralus, northcentralus) on subscription "Azure subscription 1" (Sponsored, id dc81c28d...). Claude filed two quota requests via `az quota update` at 18:54 UTC, **East US**:
   - `StandardNVADSA10v5Family` -> 72 vCPUs (A10 for Isaac Sim) — status: InProgress
   - `StandardNCADSA100v4Family` -> 24 vCPUs (1x A100 80GB) — status: InProgress
   If they get rejected or sit in review: portal -> Quotas -> Compute -> East US -> request the same two (fallback `StandardNCadsH100v5Family` -> 40). Check: `az vm list-usage --location eastus -o table | grep -iE "NVADSA10|NCADSA100|NCADSH100"`.
2. ~~No Kaggle token~~ DONE 2026-09-30: token at ~/.kaggle/access_token; Simuletic + DCSASS downloading (data/logs/simuletic.log, dcsass.log). Next: add both to data/README.md; Simuletic turned out to be an 8-clip sample (not useful for training); DCSASS eval only.
3. ~~No NGC key~~ DONE 2026-09-30: `NGC_API_KEY` in ~/.zshrc, verified (NGC API 200, nvcr.io isaac-sim pull auth 200).
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

## Status (2026-10-01 ~01:30 UTC)
Phases 0, 1, 4 (except GPU parts), 5, 6 done on the Mac. All real-data results re-run after the adversarial review (`scripts/phase2_rerun.sh`), REPORT.md Phase 2 section written from them, 98 tests pass. Phase 2 (Azure VMs) and Phase 3 (Isaac Sim) blocked on GPU quota (ticket above).

## OVERNIGHT RUN (2026-10-01 ~00:20 UTC start, until ~06:20 UTC, Kiro asleep)
Rules: one GPU job at a time; never use a waiter that greps its own command line (wait on PIDs); no YouTube or unlicensed footage for training.
Queue:
1. `scripts/tune_merl.sh` (MERL TRAIN split, 9 perception configs) -> results/tuning/. Then run the best config on the TEST split once -> results/merl_measure_tuned.json; if clearly better, make it the default and update measured_error_rates.
2. `scripts/conceal_select.py` (PoseLift leave-one-camera-out only) -> results/conceal_select.json. If v2/aug wins clearly, make it the default in conceal_experiment.py and rerun `scripts/phase2_rerun.sh`.
3. Merge the three agent worktrees as they finish (A: multicam + late-receipt retraction, B: ONNX/CoreML runtime, C: Isaac Sim 4.5 scene + runbook), run tests after each merge.
4. Every wakeup: check quota ticket #2610010040000169; if limits > 0, launch per BREE_PHASE2.md cost rules.
5. End: speed.py (machine idle), REPORT.md "Overnight" section, independent fact-check, push.

## Next steps (exact)
1. Check the quota ticket. When limits > 0: `scripts/azure_gpu.sh train eastus` (A100 VM, runs make test), then `NGC_API_KEY=... scripts/azure_gpu.sh sim eastus` (Isaac Automator, Isaac Sim 4.5.0 because of the GRID 570 driver), start from `src/bree/sim/isaac/`, pilot 50 clips, check overlays, then 2,000+ split by scene seed. Deallocate with `scripts/azure_gpu.sh stop`.
2. On the A100: train the concealment classifier on synthetic + PoseLift, rerun `scripts/phase2_rerun.sh`; `python scripts/speed.py` for PyTorch/ONNX/TensorRT FPS.
3. When operator/staged footage lands in data/operator, data/staged: draw zones, run `bree shadow` / `bree run`, hand-label, measure pick/put-back/register/exit rates, update measured_error_rates.py.

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
