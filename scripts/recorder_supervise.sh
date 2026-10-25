#!/bin/sh
# Restart the recorder if it dies, until touch ~/Projects/bree-data/recorder.stop or the deadline (12 h) passes.
D=$HOME/Projects/bree-data
END=$(( $(date +%s) + 43200 ))
while [ ! -e $D/recorder.stop ] && [ $(date +%s) -lt $END ]; do
  H=$(python3 -c "print(max(0.05, ($END-$(date +%s))/3600))")
  nice -n 10 python3 $D/imou_record_run.py --hours $H >> $D/recorder.stdout 2>&1
  [ -e $D/recorder.stop ] && break
  sleep 20
done
