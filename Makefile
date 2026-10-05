# BREE vision. Everything runs from the local venv.
PY      ?= .venv/bin/python
VIDEO   ?= data/toy/toy_walkout.mp4
STORE   ?= configs/store_gas_station_small.yaml

.PHONY: setup hw test test-fast demo demo-toy bench reid-bench sim sim-eval sim-fixture data export dashboard clean \
        calibrate calib-check calib-bench edge-test edge-measure edge-hub review review-report review-demo \
        sku-data sku-train sku-eval sku-clip sku-clip-door sim-eval-sku sku-finetune e2e-sim gc \
        assoc-dev assoc-door assoc-scripted assoc-test bench-clips bench-dev bench-smoke bench-test hands-data hands-eval \
        hands-train plates-bench plates-bench-quick plates-purge plates-test shelf-clips shelf-eval shelf-eval-tune shelf-robust shelf-test

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

# ---- fixed benchmark on SIMULATED clips (scripts/bench, src/bree/sim/bench.py) ----
BENCH_JOBS ?= 2
BENCH_ARGS ?=

bench-clips:      ## render the DEV and TEST clips of the fixed benchmark (10 to 45 min per clip depending on load; skips clips already rendered, retries a clip when Chrome closes). Needs Chrome, node, ffmpeg, BREE_PLAYWRIGHT
	$(PY) scripts/bench/render_split.py dev
	$(PY) scripts/bench/render_split.py test

bench-dev:        ## SIMULATED benchmark, DEV split (6 clips): bree.shelf.store (shelf events, floor tracks, association, ledger) -> scorecard and per-pick funnel in results/bench_dev.json and .md. BENCH_ARGS=--keep repeats only tracking, association and the ledger
	$(PY) -m bree.sim.bench dev --jobs $(BENCH_JOBS) $(BENCH_ARGS)

bench-test:       ## SIMULATED benchmark, TEST split: final numbers only, never for tuning -> results/bench_test.json and .md
	$(PY) -m bree.sim.bench test --jobs $(BENCH_JOBS) $(BENCH_ARGS)

bench-smoke:      ## one DEV clip, 60 frames per camera: proves the command runs, writes nothing to results/
	$(PY) -m bree.sim.bench dev --smoke

# ---- shelf events without a person box (src/bree/shelf, scripts/shelf). SIMULATED clips, TRAIN seeds only ----
SHELF_TUNE ?= 4900 4901 4902
SHELF_HELDOUT ?= 4903 4904 4905 4906 4950 4951
SHELF_JOBS ?= 3

shelf-clips:      ## render the TRAIN-seed clips of the shelf-event evaluation (about 15 min per clip on an idle laptop, three at a time; skips finished clips). Needs Chrome, node, ffmpeg, BREE_PLAYWRIGHT
	scripts/shelf/render_train.sh $(SHELF_TUNE) $(SHELF_HELDOUT)

shelf-eval:       ## SIMULATED: shelf events on the six held-out TRAIN-seed clips -> out/shelf/eval_heldout6.json and .md
	$(PY) scripts/shelf/eval_shelf.py $(SHELF_HELDOUT) --name heldout6 --jobs $(SHELF_JOBS)

shelf-eval-tune:  ## SIMULATED: the same on the clips the rules were tuned on -> out/shelf/eval_tune.json and .md
	$(PY) scripts/shelf/eval_shelf.py $(SHELF_TUNE) --name tune --jobs $(SHELF_JOBS)

shelf-robust:     ## SIMULATED: pixel comparison on clip 4903 with the picture shifted or the light changed -> out/shelf/robust_4903.md
	$(PY) scripts/shelf/robust_check.py 4903

shelf-test:       ## unit tests of the shelf-event code (synthetic pictures, no video needed)
	$(PY) -m pytest tests/test_shelf_events.py tests/test_shelf_store.py -q

# ---- hand and held-item detector (SIMULATED; stream hand-detector) ----
HANDS        ?= $(SYNTH)/hands
HANDS_SEEDS  ?= 2000:2042
HANDS_HELD   ?= 3000:3010
HANDS_TRAIN   = $(PY) -m bree.train.train --batch 16 --lr0 0.001

hands-data:       ## SIMULATED reach / held item / hand frames from TRAIN-range seeds, held-out seeds apart, then 640 px tiles -> $(HANDS)/sku (held-out frames + first tiles), $(HANDS)/sku_x2 (twice the training frames)
	node scripts/train/render_hands.mjs --out $(HANDS)/frames --seeds $(HANDS_SEEDS) --p-act 0.5 --p-carry 0.2
	node scripts/train/render_hands.mjs --out $(HANDS)/heldout_frames --seeds $(HANDS_HELD) --p-act 0.5 --p-carry 0.2
	$(PY) -m bree.train.dataset --frames $(HANDS)/frames $(HANDS)/heldout_frames --out $(HANDS)/sku --heldout-from 3000 --stride 4 --tiles-per-frame 2 --max-hard 1 --focus 2 --workers 5
	$(PY) -m bree.train.dataset --frames $(HANDS)/frames --out $(HANDS)/sku_x2 --heldout-from 3000 --stride 2 --tiles-per-frame 2 --max-hard 2 --focus 3 --workers 4

hands-train:      ## sim_sku.pt -> sim_sku_hands_v2.pt (hand class added, 2 epochs) -> sim_sku_hands_v3.pt (3 epochs on sku_x2). sim_sku.pt is not touched
	$(HANDS_TRAIN) --data $(HANDS)/sku/data.yaml --base $(SYNTH)/weights/sim_sku.pt --epochs 2 --warmup-epochs 0.15 --name sim_sku_hands_v2 --out $(SYNTH)/weights/sim_sku_hands_v2.pt
	$(HANDS_TRAIN) --data $(HANDS)/sku_x2/data.yaml --base $(SYNTH)/weights/sim_sku_hands_v2.pt --epochs 3 --warmup-epochs 0.1 --name sim_sku_hands_v3 --out $(SYNTH)/weights/sim_sku_hands_v3.pt

hands-eval:       ## held-out simulated frames, old and new weights side by side -> results/hand_detector.json and .md
	$(PY) -m bree.train.eval_hands --data $(HANDS)/sku --weights sim_sku sim_sku_hands_v2 sim_sku_hands_v3 --second-look 6

assoc-test:       ## SYNTHETIC scripted scenes: association, floor tracker, shelf events to the ledger (no weights needed)
	$(PY) -m pytest tests/test_association.py -q

assoc-scripted:   ## SYNTHETIC: 200 scripted multi-shopper scenes, right-shopper rate and identity -> results/assoc_scripted.json
	$(PY) -m bree.track.scenes --scenes 200 --write results/assoc_scripted.json

assoc-door:       ## SIMULATED: identities per shopper on data/synth/clip_5001_door (TRACK-2, TRACK-3) -> results/assoc_door.json
	$(PY) -m bree.track.floor_bench door --write results/assoc_door.json

assoc-dev:        ## SIMULATED DEV clips: identity, right-shopper rate, and the chain to alerts with the TRUE picks as shelf events -> results/assoc_dev.json
	$(PY) -m bree.track.floor_bench dev --chain --write results/assoc_dev.json

# ---- fuel drive-off: plate reader, pump zones, retention (src/bree/plates, scripts/plates) ----
PLATES_DIR ?= out/plates/store

plates-test:        ## drive-off unit tests on SYNTHETIC plates and SCRIPTED timelines (no model weights needed; add -m vision for the open models)
	$(PY) -m pytest tests/test_plates.py -m "not vision"

plates-bench:       ## SYNTHETIC plate read accuracy by plate width and light, and the drive-off rule on scripted timelines -> results/plates_bench.md, out/plates/bench.json (about 16 min measured on a busy laptop; --scenarios-only reruns the timelines and the vote calibration in about 4)
	$(PY) scripts/plates/bench.py

plates-bench-quick: ## the same on 2 widths and 2 light levels, a few samples (about 3 min), writes out/plates/quick.md and out/plates/bench_quick.json
	$(PY) scripts/plates/bench.py --quick --out out/plates/quick.md

plates-purge:       ## delete expired plate records now (also runs on every open, write and lookup of the plate store, and hourly from a running DriveOffMonitor)
	$(PY) -c "from bree.plates.retention import PlateStore; print(PlateStore('$(PLATES_DIR)').purge())"
