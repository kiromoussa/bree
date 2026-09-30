# data/

Everything in here is gitignored except this file. `make data` (= `scripts/download_data.sh`) fills what it can.

| Dataset | What it is | Status from the build box | How to get it |
|---|---|---|---|
| `real/vtest.avi` | OpenCV sample video: 795 frames of real pedestrians, 768x576 @ 10 fps. **No theft labels**. Used only for detector/tracker FPS and stability. | downloaded | automatic (raw.githubusercontent.com) |
| `toy/` | Rendered 2D toy clips with ground truth (`bree render-toy`). **Toy data**: proves the pipeline runs end to end, says nothing about real-world accuracy. | generated locally | automatic |
| `simuletic/` | Simuletic synthetic CCTV shoplifting (Kaggle, CC BY 4.0 per listing). Free subset ~400 images + 8 videos with YOLO boxes, 17-kpt pose, VLM captions. | **blocked** (kaggle.com returns 403 from the build environment) | `pip install kaggle`, put your API token in `~/.kaggle/kaggle.json`, then `kaggle datasets download -d simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm -p data/simuletic --unzip` (or just re-run `make data`) |
| `dcsass/` | DCSASS: short clips cut from UCF-Crime, 13 classes incl. Shoplifting. Research use only (UCF-Crime derived). (Note: this is on Kaggle, not Mendeley.) | **blocked** (Kaggle) | `kaggle datasets download -d mateohervas/dcsass-dataset -p data/dcsass --unzip` |
| `mendeley_shoplifting/` | Two Mendeley shoplifting video sets (MNNIT 2022 `r3yjf35hzr`; ~172 clips `3pn2rnv2wg`). | **blocked** (data.mendeley.com 403) | Browser: https://data.mendeley.com/datasets/r3yjf35hzr/1 and https://data.mendeley.com/datasets/3pn2rnv2wg/1 -> "Download All" -> unzip here. Check the license on the page. |
| `poselift_official/` | **PoseLift** (WACV 2025, TeCSAR-UNCC, Apache-2.0). Real retail CCTV, shoplifting vs normal, **pose keypoints only** (COCO-17), 1920x1080 @ 15 fps. Train: 104 videos, all normal. Test: 47 videos with per-frame labels. Code/README: https://github.com/TeCSAR-UNCC/PoseLift | **not used tonight.** Official data is on Google Drive (blocked here). A third-party GitHub copy exists, but the automated safety check blocked pulling from an unofficial mirror, so nothing from it is used in any result. | Browser: https://drive.google.com/drive/folders/1aEkENZlVE4ZvF_BZXV1VJOwuiXQq6trn -> download into `data/poselift_official/` |
| `retails/` | RetailS (TeCSAR-UNCC, PoseLift follow-up): ~20M normal pose frames + real and staged shoplifting test sets. No license file in the repo — ask the authors. | **blocked** (Google Drive) | https://drive.google.com/file/d/1uBCvDm7QdYjxwS9HT63lYaucATrecNuH |

## PoseLift format notes (from the dataset investigation; re-verify on the official copy)
- `Pickle_files/{Train,Test}/<cam>_<vid>.pkl`: `dict[frame] -> dict[person_id] -> [bbox, kps]`.
- `bbox` reportedly **xyxy** (README says xywh).
- `kps` float16 (17, 3), COCO-17 order, reportedly stored as **(y, x, conf)** — check and swap columns if so.
- `Pickle_files/GT/<cam>_<vid>.npy`: per-frame labels (0/1), test videos only. Some label files are longer than the pose file: index labels by frame number.
- Labels are per frame, not per person.
