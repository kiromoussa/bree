# data/

Everything in here is gitignored except this file and `LICENSES.md` (licenses and allowed use per dataset).
Counts below were measured on the downloaded copies on 2026-09-30.

| Dir | Dataset | Source | License / use | Size | Contents (measured) |
|---|---|---|---|---|---|
| `poselift/PoseLift/Pickle_files/` | PoseLift (WACV 2025) | official Google Drive folder (via `scripts/drive_fetch.py`) | Apache-2.0: **train + eval** | 1.5 MB json + pickles | 151 videos (104 Train, all normal, 91,598 frames; 47 Test, 4,846 frames, 1,534 shoplifting frames, 41 of 47 clips contain shoplifting). README says 153; 2 are not in the Drive folder. 6 cameras, 1920x1080 @ 15 fps, pose only. |
| `retails/RetailS/` | RetailS (TeCSAR-UNCC) | official Google Drive file | no license; **eval only** | 5.1 GB | `RetailS_train`: 942 normal pose files, 6 cameras. `RetailS_test_staged`: 624 clips, 40,913 frames, 20,335 shoplifting frames. `RetailS_test_realworld`: 47 clips = **the PoseLift test set** (identical keypoints), not used. |
| `merl/` | MERL Shopping | merl.com (official) | research only; **eval only** | 1.2 GB | 106 videos, 244.7 min, 920x680 @ 30 fps, overhead camera, 1 shopper each, 5 hand-action labels. Test split = subjects 27-41 (28 videos, 64.3 min). |
| `ucf_crime/` | UCF-Crime | authors' Dropbox (official page dead) | research only; **eval only** | ~11 GB | `Testing_Normal_Videos.zip`: 150 normal test videos (15 scored, seeded sample, 0.38 h). `Anomaly-Videos-Part-4.zip`: Shoplifting / Stealing / Vandalism. Official temporal annotations for test videos. Low-res (320x240) general CCTV. |
| `dcsass/` | DCSASS | Kaggle | derived from UCF-Crime; **eval only** | 1.5 GB | 13 classes, 34,318 short clips; Shoplifting: 896 labelled clips (155 shoplifting) from 28 UCF videos. |
| `simuletic/` | Simuletic CCTV shoplifting (free sample) | Kaggle | CC BY 4.0 | 482 MB | 8 synthetic clips (4 shoplifting, 4 normal), 544x544 @ 24 fps; 456 frames with person boxes + 17 keypoints; VLM captions. Too small to train on. |
| `real/vtest.avi` | OpenCV sample | github.com/opencv | Apache-2.0 repo | 8 MB | 795 frames of pedestrians, FPS smoke test only. |
| `toy/` | rendered toy clips | `bree render-toy` | ours | small | 8 scripted 2D scenes, TOY DATA. |
| `staged/`, `cloudstation/`, `operator/` | Kiro's own footage | n/a | ours | none yet | **Takes priority over everything above once it exists.** |

Not downloaded: SKU-110K (partial, paused for bandwidth; see DECISIONS.md), Mendeley shoplifting sets.

## How close is each to a gas-station convenience store?
- **PoseLift / RetailS**: closest. Real US retail store, ceiling cameras over aisles, real and staged pocket / bag / under-shirt concealment. But pose-only (no pixels, no products), bigger store, and poses come from a different detector/pose model (YOLOv8 + HRNet) than ours, so models trained on them see a domain shift on our pipeline's poses.
- **MERL Shopping**: lab mock-up of one shelf filmed straight from above, one shopper, no theft. Good for measuring hands at the shelf; the overhead angle is harder for a stock person detector than a real angled ceiling camera, so its tracking numbers are pessimistic.
- **UCF-Crime / DCSASS**: real CCTV, many stores, but low resolution and varied angles; labels are coarse (video-level, or start/end of the incident). Normal videos are general surveillance scenes, not stores.
- **Simuletic**: synthetic retail CCTV, right idea, but only 8 clips in the free release.
- **Kiro's footage**: the only data from the actual cameras, angles, products and lighting we will deploy on.

## Formats
- PoseLift pickles: `dict[frame] -> dict[person_id] -> [bbox_xyxy, kps(17,3)]`, keypoints stored **(y, x, conf)**, some NaN (treated as undetected). `GT/<cam>_<vid>.npy`: per-frame 0/1.
- RetailS JSON: `{person_id: {frame: {"keypoints": [51 floats, (y, x, c)]}}}`, no boxes. `gt/test_frame_mask/<name>.npy`.
- Loaders: `bree.conceal.load_poselift`, `bree.conceal.load_retails`.
