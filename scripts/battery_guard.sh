#!/usr/bin/env bash
# Pause heavy BREE jobs at <=10% battery (push first), resume them on AC.
cd "$(dirname "$0")/.."
paused=0
while true; do
  b=$(pmset -g batt)
  pct=$(echo "$b" | grep -oE "[0-9]+%" | head -1 | tr -d %)
  if echo "$b" | grep -q "AC Power"; then
    [ $paused = 1 ] && { pkill -CONT -f "scripts/(measure_merl|conceal_select|conceal_experiment|eval_|speed)"; paused=0; echo "$(date) resumed on AC"; }
  elif [ "${pct:-100}" -le 10 ] && [ $paused = 0 ]; then
    git add -A >/dev/null 2>&1; git commit -qm "auto: low battery checkpoint" >/dev/null 2>&1; git push -q >/dev/null 2>&1
    pkill -STOP -f "scripts/(measure_merl|conceal_select|conceal_experiment|eval_|speed)"; paused=1; echo "$(date) paused at ${pct}%"
  fi
  sleep 60
done
