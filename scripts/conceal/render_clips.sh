#!/bin/sh
# Render SIMULATED clips for the concealment cue with the benchmark's renderer, one after the other, into
# data/synth/conceal/<name>/clip_<seed> (not into the benchmark's folders, so make bench-train is not touched).
#   scripts/conceal/render_clips.sh heldout 4960 4962 ...      TRAIN-range seeds only (1000:5000)
: "${BREE_PLAYWRIGHT:?set BREE_PLAYWRIGHT to node_modules/playwright}"
name=$1; shift
for s in "$@"; do
  [ "$s" -ge 1000 ] && [ "$s" -lt 5000 ] || { echo "seed $s is not a TRAIN-range seed"; exit 2; }
  out=data/synth/conceal/$name/clip_$s
  [ -f "$out/truth/events.jsonl" ] && [ -f "$out/clip.json" ] && { echo "clip $s already rendered"; continue; }
  rm -rf "$out"; mkdir -p "$out"
  node scripts/bench/render_clip.mjs --seed "$s" --out "$out" > "$out/render.log" 2>&1 && echo "clip $s rendered" || echo "clip $s FAILED"
done
