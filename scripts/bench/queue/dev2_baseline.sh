#!/bin/sh
# The committed pipeline (worktree out/wt-baseline) on dev2, into out/bench/dev2_baseline: its own folder, because
# tuning runs of other streams write to out/bench/dev2. SIMULATED clips.
# usage: dev2_baseline.sh plain|stress [wait]     "wait": start only when the dev2 render is over (a second plain worker)
# Several workers can run: a clip whose run folder exists is left to the worker that made it.
WT=/Users/kiromoussa/bree-vision/out/wt-baseline
MAIN=/Users/kiromoussa/bree-vision
B=$MAIN/out/bench/dev2_baseline
cd $WT
export PYTHONPATH=$WT/src
[ "$2" = wait ] && while pgrep -f "render_split.py dev2" > /dev/null; do sleep 30; done
for seed in $(seq 11001 11020); do
  if [ "$1" = plain ]; then
    until [ -f $MAIN/data/synth/bench/dev2/clip_$seed/clip.json ]; do sleep 20; done
    [ -d $B/clip_$seed ] && continue
    $MAIN/.venv/bin/python -m bree.sim.bench dev2 --clips $seed --jobs 1 --out $B > ${B}_$seed.cmd.log 2>&1
  else
    until [ -f $B/clip_$seed/run.json ]; do sleep 20; done
    [ -d ${B}_stress/clip_$seed ] && continue
    $MAIN/.venv/bin/python -m bree.sim.bench dev2 --clips $seed --jobs 1 --stress --stress-only --out $B > ${B}_${seed}_stress.cmd.log 2>&1
  fi
  echo "$(date +%H:%M) $1 $seed done"
done
echo "ALL DONE $1"
