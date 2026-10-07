#!/bin/sh
# Held-out clips for the concealment cue, one job at a time: render (if missing), then the shelf and people stages
# through scripts/conceal/eval.py (cue on). A scan that lost detections to an overloaded machine fails on purpose
# (bree.concealment.scan) and is tried again after a pause; finished cameras are kept.
#   BREE_PLAYWRIGHT=... scripts/conceal/heldout_queue.sh 4960 4962 4963 ...     TRAIN-range seeds only
try() { n=0; until "$@"; do n=$((n + 1)); [ $n -ge 6 ] && { echo "GAVE UP: $*"; return 1; }; echo "retry $n in 60 s: $*"; sleep 60; done; }
for s in "$@"; do
  scripts/conceal/render_clips.sh heldout "$s"
  [ -f data/synth/conceal/heldout/clip_$s/truth/events.jsonl ] || continue
  try .venv/bin/python scripts/conceal/eval.py data/synth/conceal/heldout "$s" --out out/conceal/runs/heldout --variants on --json out/conceal/heldout_stage_$s.json && echo "staged $s"
done
echo "queue finished"
