# BREE vision. Everything runs from the local venv.
PY      ?= .venv/bin/python
VIDEO   ?= data/toy/toy_walkout.mp4
STORE   ?= configs/store_gas_station_small.yaml

.PHONY: setup hw test test-fast demo demo-toy bench reid-bench sim sim-eval sim-fixture data export dashboard clean \
        calibrate calib-check calib-bench edge-test edge-measure edge-hub review review-report review-demo \
        sku-data sku-train sku-eval sku-clip sku-clip-door sim-eval-sku sku-finetune e2e-sim gc

setup:            ## create venv, install pinned deps (CPU torch unless CUDA is present), fetch weights
	./scripts/setup.sh

hw:               ## print detected hardware + chosen model sizes
	$(PY) -m bree.hw

test:             ## unit tests (ledger, events, sim, re-ID, closed-world identity, calibration, camera node, review loop, SKU detector maths) + vision smoke tests + re-ID regression guard on MOT16
	$(PY) -m pytest

test-fast:        ## unit tests only, no model weights needed
	$(PY) -m pytest -m "not vision"

demo:             ## toy clips end to end: render -> detect -> track -> events -> ledger -> alerts
	$(PY) -m bree.cli demo

bench: reid-bench ## full benchmark: re-ID bench, then event-level sim + toy video + real-footage FPS; writes results/bench.json
	$(PY) -m bree.cli bench

reid-bench:       ## identity: re-ID rank-1 / mAP; off / re-ID / closed world / both on MOT16 (sanity), MERL, toy, scripted store -> results/reid_bench.json
	$(PY) -m bree.eval.reid_bench

sim:              ## event-level simulator only
	$(PY) -m bree.cli sim

# Isaac Sim run folder (events.jsonl + per-camera rgb_*.png folders) and the layout JSON it was built from
SIM_OUT     ?=
LAYOUT      ?=
SIM_BACKEND ?= yolo
SIM_ARGS    ?=
sim-eval:         ## score the pipeline on simulator output: make sim-eval SIM_OUT=<isaac run dir> LAYOUT=<layout.json>
	@test -n "$(SIM_OUT)" || { echo "usage: make sim-eval SIM_OUT=<isaac run dir> [LAYOUT=<layout.json>] [SIM_BACKEND=yolo|toy|sim_sku] [SIM_ARGS='--pos-dropout 0.1 --closed-world --slots']"; exit 2; }
	$(PY) -m bree.sim.sim_eval --sim-out $(SIM_OUT) $(if $(LAYOUT),--layout $(LAYOUT)) --backend $(SIM_BACKEND) $(SIM_ARGS)

sim-fixture:      ## same chain on a generated TOY fixture (no GPU, no Isaac Sim): proves the harness, not accuracy
	$(PY) -m bree.sim.sim_eval --fixture out/sim_fixture $(SIM_ARGS)

# ---- camera calibration, 3D slots, store-wide identity (SYNTHETIC bench) ----
# Store YAML(s) of the cameras to calibrate or check, the shared layout JSON, and extra arguments
CALIB_STORES ?=
CALIB_LAYOUT ?=
CALIB_ARGS   ?=
calibrate:        ## marks in a store YAML -> camera.calibration: make calibrate CALIB_STORES=configs/cam.yaml [CALIB_LAYOUT=layout.json] [CALIB_ARGS='--hfov 25 --in-place']
	@test -n "$(CALIB_STORES)" || { echo "usage: make calibrate CALIB_STORES=<store.yaml> [CALIB_LAYOUT=<layout.json>] [CALIB_ARGS='--camera-id ID --hfov 25 --in-place']"; exit 2; }
	PYTHONPATH=src $(PY) scripts/calibrate.py fit $(CALIB_STORES) $(if $(CALIB_LAYOUT),--layout $(CALIB_LAYOUT)) $(CALIB_ARGS)

calib-check:      ## flag cameras whose calibration error is over the thresholds (exit 1): make calib-check CALIB_STORES='configs/store1_*.yaml' [CALIB_ARGS='--max-px 5 --max-m 0.15']
	@test -n "$(CALIB_STORES)" || { echo "usage: make calib-check CALIB_STORES='<store.yaml> ...' [CALIB_ARGS='--max-px 5 --max-m 0.15']"; exit 2; }
	PYTHONPATH=src $(PY) scripts/calibrate.py check $(CALIB_STORES) $(CALIB_ARGS)

calib-bench:      ## SYNTHETIC: calibration error, 3D slot of a pick (two views vs one), store-wide identity (2 and 4 cameras) on the 45 camera layout -> results/calib_bench.json
	$(PY) -m bree.calib.bench --out results/calib_bench.json

# ---- camera nodes (Pi Zero 2 W) and hub ----
edge-test:        ## camera node tests: trigger, pre-roll bursts, wire format, spool, node -> hub over HTTP (no weights needed)
	$(PY) -m pytest tests/test_edge_node.py

edge-measure:     ## node first pass on recorded clips: trigger recall, false triggers/hour, bytes vs full-res streaming, CPU -> results/edge_measure_*.json
	PYTHONPATH=src $(PY) scripts/edge/measure.py merl --split test --out results/edge_measure_merl_test.json
	PYTHONPATH=src $(PY) scripts/edge/measure.py toy --fps 15 --out results/edge_measure_toy.json
	PYTHONPATH=src $(PY) scripts/edge/measure.py cpu --out results/edge_measure_cpu.json

edge-hub:         ## hub receiver for camera nodes: make edge-hub TOKEN=... [HUB_ARGS="--store cam=configs/cam.yaml"]
	$(PY) -m bree.edge.hub --out out/hub --token "$(TOKEN)" $(HUB_ARGS)

# ---- human review and the feedback loop ----
REVIEW_STORE ?= out/review

review:           ## local reviewer page on the review store (127.0.0.1:8090): make review REVIEW_STORE=out/review
	$(PY) -m bree.review --store $(REVIEW_STORE) serve

review-report:    ## weekly owner report (markdown + JSON): the seven full UTC days ending yesterday, today is not in it (other weeks: report --week-start YYYY-MM-DD)
	$(PY) -m bree.review --store $(REVIEW_STORE) report

review-demo:      ## SYNTHETIC end to end run of the review loop: alerts, decisions, label export, metrics, owner report
	$(PY) scripts/review/synthetic_e2e.py

# ---- SKU detector trained on SIMULATED store frames ----
SYNTH        ?= data/synth
SKU_SEEDS    ?= 1000:1240
SKU_EPOCHS   ?= 6
# the research repo (simulator, layouts) is expected next to the home folder: ~/bree. Elsewhere: pass SKU_LAYOUT
# and, for the renders, node scripts/train/render_synth.mjs --sim-src <bree>/software/sim-prototype
SKU_LAYOUT   ?= $(HOME)/bree/software/shared/layouts/recommended-47.json
CLIP_SEED    ?= 5001
REAL_DATA    ?=
# The recorded end to end run (REPORT "End to end on simulated data"): register camera, the overhead camera that
# sees the door, a second overhead, and five rail cameras that see the picks. E2E_RESULTS=e2e_sim_chain replaces
# the tracked results/e2e_sim_chain.*; empty (the default) leaves results/ alone.
E2E_CAMS     ?= REGISTER-top,TRACK-2,TRACK-3,G4L-rail-1,G2L-rail-4,G2L-rail-3,G1R-rail-3,G3R-rail-1
E2E_CLIP     ?= $(SYNTH)/clip_$(CLIP_SEED)_door
E2E_RESULTS  ?=
# needs Chrome, node 18+, Playwright (BREE_PLAYWRIGHT=/path/to/node_modules/playwright) and network for three.js

sku-data:         ## SIMULATED dataset: render the browser sim copy headless (out/sim-copy), then cut 640 px tiles, split by scene seed
	node scripts/train/render_synth.mjs --out $(SYNTH)/frames --seeds $(SKU_SEEDS) --layout $(SKU_LAYOUT)
	$(PY) -m bree.train.dataset --frames $(SYNTH)/frames --out $(SYNTH)/sku

sku-train:        ## train the SKU detector on the simulated tiles (YOLO26n, MPS or CPU) -> data/synth/weights/sim_sku.pt + .json record
	$(PY) -m bree.train.train --data $(SYNTH)/sku/data.yaml --epochs $(SKU_EPOCHS)

sku-eval:         ## held-out simulated scenes: mAP per SKU, pixels-vs-accuracy curve, variant confusion, speed -> results/sku_detector.json, .md, px_vs_accuracy.png
	$(PY) -m bree.train.evaluate --data $(SYNTH)/sku --out results

sku-clip:         ## render one simulated scenario as video frames in the folder shape sim-eval reads (cameras picked by the script: NO door camera; for sim-eval-sku)
	node scripts/train/render_synth.mjs --clip --seed $(CLIP_SEED) --out $(SYNTH)/clip_$(CLIP_SEED) --layout $(SKU_LAYOUT)

sku-clip-door:    ## the clip of the recorded end to end run: the same scenario on the E2E_CAMS list (door camera included) -> $(E2E_CLIP)
	node scripts/train/render_synth.mjs --clip --seed $(CLIP_SEED) --out $(E2E_CLIP) --layout $(SKU_LAYOUT) --cams $(E2E_CAMS)

sim-eval-sku:     ## sim-eval with the sim-trained SKU detector as the product backend, then the COCO baseline for comparison
	$(PY) -m bree.train.sim_eval_sku --sim-out $(SYNTH)/clip_$(CLIP_SEED) --backend sim_sku
	$(PY) -m bree.train.sim_eval_sku --sim-out $(SYNTH)/clip_$(CLIP_SEED) --backend yolo

sku-finetune:     ## fine-tune on REAL labelled frames once they exist: make sku-finetune REAL_DATA=data/real_sku/data.yaml
	@test -n "$(REAL_DATA)" || { echo "usage: make sku-finetune REAL_DATA=<YOLO data.yaml of real labelled frames>"; exit 2; }
	PYTHONPATH=src $(PY) scripts/train/finetune_real.py --data $(REAL_DATA)

e2e-sim:          ## SIMULATED end to end on the sku-clip-door clip: node trigger -> hub -> pipeline (sim_sku, closed world, 3D slots) -> alerts -> review store -> owner report -> scorecard in $(E2E_CLIP)/e2e/ (E2E_RESULTS=e2e_sim_chain also replaces results/e2e_sim_chain.*)
	@test -f $(E2E_CLIP)/events.jsonl || { echo "no clip at $(E2E_CLIP): run make sku-clip-door first"; exit 2; }
	$(PY) scripts/e2e_sim_chain.py --sim-out $(E2E_CLIP) --results-name "$(E2E_RESULTS)"

gc:               ## drop unreachable git objects (something large staged once and never committed); run when no other process is using the repo
	git gc --prune=now

data:             ## download whatever public datasets are reachable
	./scripts/download_data.sh

export:           ## export YOLO models to ONNX for edge devices
	$(PY) -m bree.cli export

dashboard:        ## local web dashboard of live alerts + baskets
	$(PY) -m bree.cli dashboard

clean:
	rm -rf out runs .pytest_cache
