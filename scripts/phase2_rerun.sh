#!/usr/bin/env bash
# Re-run every real-data result in dependency order, one GPU job at a time. Perception caches in out/ are reused.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/conceal_experiment.py > data/logs/conceal.log 2>&1
$PY scripts/eval_retails.py 60   > data/logs/retails_eval.log 2>&1
PYTHONPATH=scripts $PY scripts/eval_dcsass.py > data/logs/dcsass_eval.log 2>&1
$PY scripts/eval_ucf.py 40       > data/logs/ucf_eval.log 2>&1
$PY scripts/measured_error_rates.py > data/logs/mer.log 2>&1
make bench > data/logs/bench.log 2>&1
make test  > data/logs/test.log 2>&1
echo PHASE2_RERUN_DONE >> data/logs/test.log
