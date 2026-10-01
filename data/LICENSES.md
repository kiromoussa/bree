# Dataset licenses and allowed use

Rule: only data with a clear commercial license may train the shipped model. Everything else is **evaluation only**.

| Dataset | Source we used | License / terms (as found 2026-09-30) | Our use |
|---|---|---|---|
| PoseLift | Official Google Drive folder linked from github.com/TeCSAR-UNCC/PoseLift | **Apache-2.0** (LICENSE file in the official repo) | **Train + eval.** Concealment classifier trained on it. Cite Hashemi et al., WACV 2025. |
| RetailS | Official Google Drive file linked from github.com/TeCSAR-UNCC/RetailS | **No license file.** The repo's `index.md` (an unfinished template) says "Academic use only / non-commercial". Authors not yet asked (see PROGRESS.md: Kiro to email nrashvan@charlotte.edu, syao@charlotte.edu). | **Evaluation only.** Note: its real-world test set is the PoseLift test set (same 47 clips, identical keypoints), so we only evaluate on the staged test set and the normal training footage. |
| MERL Shopping | merl.com/pub/tmarks/MERL_Shopping_Dataset (official) | (c) 2016 MERL. "Permission to use ... without fee for research and educational purposes". No commercial grant. | **Evaluation only** (pose/track layer measurements). |
| UCF-Crime | Official Dropbox folder of the authors (Chen Chen, UNC Charlotte); the old visionlab.uncc.edu page no longer resolves | Research dataset, no commercial license. | **Evaluation only.** |
| SKU-110K | Ultralytics mirror of the official release (trax-geometry S3) | Released for research (CVPR 2019, Trax). No clear commercial grant. | Not used yet (download paused for bandwidth). Would be **evaluation only** until checked. |
| Simuletic CCTV shoplifting | Kaggle | CC BY 4.0 per the Kaggle listing | Downloaded; the free release is only 8 clips, so not used for training. CC BY 4.0 would allow training with attribution. |
| DCSASS | Kaggle | Derived from UCF-Crime: research only | Downloaded. **Evaluation only** (896 Shoplifting-category clips). |
| OpenCV `vtest.avi` | github.com/opencv/opencv samples | OpenCV sample data (Apache-2.0 repo) | FPS / tracking smoke tests only. |
| Kiro's own (`staged/`, `cloudstation/`, `operator/`) | n/a | Ours, subject to consent of the people filmed | Train + eval once it exists. None yet. |
