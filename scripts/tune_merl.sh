#!/usr/bin/env bash
# Perception tuning grid on MERL's TRAIN split (6 videos: every 10th of subjects 1-20). Test split untouched.
cd "$(dirname "$0")/.."
run() { name=$1; shift; [ -f results/tuning/merl_$name.json ] || .venv/bin/python scripts/measure_merl.py --split train --every 10 --out results/tuning/merl_$name.json "$@" > data/logs/tune_$name.log 2>&1; }
run base_s640                                                  # current default (s models, 640, conf .3, buffer 2 s)
run n640        --det yolo26n.pt --pose yolo26n-pose.pt
run m640        --det yolo26m.pt --pose yolo26m-pose.pt
run s960        --imgsz 960
run s_conf15    --person-conf 0.15
run s_buf5      --buffer 5
run s_crop256   --crop 256
run s_conf15_buf5_ntt15  --person-conf 0.15 --buffer 5 --new-track-thresh 0.15
run m_conf15_buf5_ntt15  --det yolo26m.pt --pose yolo26m-pose.pt --person-conf 0.15 --buffer 5 --new-track-thresh 0.15
echo TUNE_DONE
