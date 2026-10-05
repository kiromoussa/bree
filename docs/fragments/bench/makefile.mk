# ---- fixed benchmark on SIMULATED clips (scripts/bench, src/bree/sim/bench.py) ----
# add to .PHONY: bench-clips bench-dev bench-test bench-smoke
BENCH_JOBS ?= 2
BENCH_ARGS ?=

bench-clips:      ## render the DEV and TEST clips of the fixed benchmark (10 to 45 min per clip depending on load; skips clips already rendered, retries a clip when Chrome closes). Needs Chrome, node, ffmpeg, BREE_PLAYWRIGHT
	$(PY) scripts/bench/render_split.py dev
	$(PY) scripts/bench/render_split.py test

bench-dev:        ## SIMULATED benchmark, DEV split (6 clips): pipeline -> scorecard and per-pick funnel in results/bench_dev.json and .md
	$(PY) -m bree.sim.bench dev --jobs $(BENCH_JOBS) $(BENCH_ARGS)

bench-test:       ## SIMULATED benchmark, TEST split: final numbers only, never for tuning -> results/bench_test.json and .md
	$(PY) -m bree.sim.bench test --jobs $(BENCH_JOBS) $(BENCH_ARGS)

bench-smoke:      ## one DEV clip, 60 frames per camera: proves the command runs, writes nothing to results/
	$(PY) -m bree.sim.bench dev --smoke
