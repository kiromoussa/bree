# BREE vision. Everything runs from the local venv.
PY      ?= .venv/bin/python
VIDEO   ?= data/toy/toy_walkout.mp4
STORE   ?= configs/store_gas_station_small.yaml

.PHONY: setup hw test test-fast demo demo-toy bench reid-bench sim sim-eval sim-fixture data export dashboard clean

setup:            ## create venv, install pinned deps (CPU torch unless CUDA is present), fetch weights
	./scripts/setup.sh

hw:               ## print detected hardware + chosen model sizes
	$(PY) -m bree.hw

test:             ## unit tests (ledger, events, sim, re-ID) + vision smoke tests + re-ID regression guard on MOT16
	$(PY) -m pytest

test-fast:        ## unit tests only, no model weights needed
	$(PY) -m pytest -m "not vision"

demo:             ## toy clips end to end: render -> detect -> track -> events -> ledger -> alerts
	$(PY) -m bree.cli demo

bench: reid-bench ## full benchmark: re-ID bench, then event-level sim + toy video + real-footage FPS; writes results/bench.json
	$(PY) -m bree.cli bench

reid-bench:       ## body re-ID: rank-1 / mAP, IDF1, ID switches, false merges, before vs after -> results/reid_bench.json
	$(PY) -m bree.eval.reid_bench

sim:              ## event-level simulator only
	$(PY) -m bree.cli sim

# Isaac Sim run folder (events.jsonl + per-camera rgb_*.png folders) and the layout JSON it was built from
SIM_OUT     ?=
LAYOUT      ?=
SIM_BACKEND ?= yolo
SIM_ARGS    ?=
sim-eval:         ## score the pipeline on simulator output: make sim-eval SIM_OUT=<isaac run dir> LAYOUT=<layout.json>
	@test -n "$(SIM_OUT)" || { echo "usage: make sim-eval SIM_OUT=<isaac run dir> [LAYOUT=<layout.json>] [SIM_BACKEND=yolo|toy] [SIM_ARGS='--pos-dropout 0.1']"; exit 2; }
	$(PY) -m bree.sim.sim_eval --sim-out $(SIM_OUT) $(if $(LAYOUT),--layout $(LAYOUT)) --backend $(SIM_BACKEND) $(SIM_ARGS)

sim-fixture:      ## same chain on a generated TOY fixture (no GPU, no Isaac Sim): proves the harness, not accuracy
	$(PY) -m bree.sim.sim_eval --fixture out/sim_fixture $(SIM_ARGS)

data:             ## download whatever public datasets are reachable
	./scripts/download_data.sh

export:           ## export YOLO models to ONNX for edge devices
	$(PY) -m bree.cli export

dashboard:        ## local web dashboard of live alerts + baskets
	$(PY) -m bree.cli dashboard

clean:
	rm -rf out runs .pytest_cache
