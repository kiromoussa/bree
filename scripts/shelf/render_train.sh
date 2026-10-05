#!/bin/sh
# Render TRAIN-seed benchmark clips for the shelf-event evaluation (SIMULATED). Skips finished clips.
#   BREE_PLAYWRIGHT=... scripts/shelf/render_train.sh 4900 4901 ...      (default: 4900 to 4906, three at a time)
# Seeds must be in the TRAIN range 1000:5000 (scripts/bench/manifest.json). Does not touch scripts/bench/clips.json.
cd "$(dirname "$0")/../.." || exit 1
[ $# -eq 0 ] && set -- 4900 4901 4902 4903 4904 4905 4906
for s in "$@"; do [ "$s" -ge 1000 ] && [ "$s" -lt 5000 ] || { echo "seed $s is not a TRAIN seed"; exit 2; }; done
printf '%s\n' "$@" | xargs -P 3 -I{} sh -c '
  d=data/synth/bench/train/clip_{}; [ -f $d/clip.json ] && exit 0; mkdir -p $d
  for try in 1 2 3; do node scripts/bench/render_clip.mjs --seed {} --out $d > $d/render.log 2>&1 && break; done'
