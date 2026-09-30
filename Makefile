# BREE vision. Everything runs from the local venv.
PY      ?= .venv/bin/python
VIDEO   ?= data/toy/toy_walkout.mp4
STORE   ?= configs/store_gas_station_small.yaml

.PHONY: setup hw test test-fast demo demo-toy bench sim data export dashboard clean

setup:            ## create venv, install pinned deps (CPU torch unless CUDA is present), fetch weights
	./scripts/setup.sh

hw:               ## print detected hardware + chosen model sizes
	$(PY) -m bree.hw

test:             ## unit tests (ledger, events, sim) + vision smoke tests
	$(PY) -m pytest

test-fast:        ## unit tests only, no model weights needed
	$(PY) -m pytest -m "not vision"

demo:             ## toy clips end to end: render -> detect -> track -> events -> ledger -> alerts
	$(PY) -m bree.cli demo

bench:            ## full benchmark: event-level sim + toy video + real-footage FPS; writes results/bench.json
	$(PY) -m bree.cli bench

sim:              ## event-level simulator only
	$(PY) -m bree.cli sim

data:             ## download whatever public datasets are reachable
	./scripts/download_data.sh

export:           ## export YOLO models to ONNX for edge devices
	$(PY) -m bree.cli export

dashboard:        ## local web dashboard of live alerts + baskets
	$(PY) -m bree.cli dashboard

clean:
	rm -rf out runs .pytest_cache
