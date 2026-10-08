#!/bin/sh
# Re-score the committed-pipeline baseline of dev2 whenever another run finishes; keep the latest scorecard in results/bench_dev2_baseline.*
MAIN=$(cd "$(dirname "$0")/../../.." && pwd)
WT=$MAIN/out/wt-baseline
B=$MAIN/out/bench/dev2_baseline
cd $WT
export PYTHONPATH=$WT/src
last=-1
while :; do
  n=$(ls $B/clip_*/run.json ${B}_stress/clip_*/run.json 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" != "$last" ]; then
    $MAIN/.venv/bin/python -m bree.sim.bench dev2 --score-only --stress --name baseline --out $B > $B.score.log 2>&1 \
      && cp $B/bench.json $MAIN/results/bench_dev2_baseline.json && cp $B/bench.md $MAIN/results/bench_dev2_baseline.md
    rc=$?; echo "$(date +%H:%M) scored with $n finished runs (exit $rc)"
    last=$n
  fi
  [ "$n" -ge 40 ] && break
  sleep 60
done
