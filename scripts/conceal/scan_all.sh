#!/bin/sh
# scan the item cameras of several clips, one clip at a time: scan_all.sh <split folder under data/synth/bench> <seed> ...
split=$1; shift
for s in "$@"; do
  .venv/bin/python -m bree.concealment.scan data/synth/bench/$split/clip_$s out/conceal/$split/clip_$s --jobs 3 || exit 1
  echo "clip $s done"
done
