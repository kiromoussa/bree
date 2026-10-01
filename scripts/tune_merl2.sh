#!/usr/bin/env bash
# Trimmed second half of the MERL TRAIN-split grid (overnight time budget): combine the two signals seen so far
# (nano = more reaches, medium = fewer track splits) with lower detection threshold + longer track buffer.
cd "$(dirname "$0")/.."
run() { name=$1; shift; [ -f results/tuning/merl_$name.json ] || .venv/bin/python scripts/measure_merl.py --split train --every 10 --out results/tuning/merl_$name.json "$@" > data/logs/tune_$name.log 2>&1; }
run m_conf15_buf5_ntt15  --det yolo26m.pt --pose yolo26m-pose.pt --person-conf 0.15 --buffer 5 --new-track-thresh 0.15
run n_conf15_buf5_ntt15  --det yolo26n.pt --pose yolo26n-pose.pt --person-conf 0.15 --buffer 5 --new-track-thresh 0.15
echo TUNE2_DONE
