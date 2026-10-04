# BREE vision

On-device store vision for retrofitted coolers and shelves. The same pipeline that
lets a shopper grab an item and walk out (cashierless checkout) also tracks exactly
what leaves the shelf, so **anything taken and not paid for is shrink**.

```
camera ─► detect people + products ─► pose on each person ─► track (ByteTrack) ─►
  event engine (zones in YAML): enter · pick · put_back · conceal · pay · exit ─►
    ledger: per-person basket ◄── POS receipts / cooler card taps (JSONL, stdin, HTTP)
      └─► on exit: basket − paid ─► alert (JSON + short head-blurred clip) / review / nothing
```

Start with `REPORT.md` for what works, the benchmark numbers, and what they do and don't prove.
Design decisions and their reasons are in `DECISIONS.md`.

**Pilot:** step-by-step runbook for a gas-station shadow-mode pilot in [docs/PILOT_RUNBOOK.md](docs/PILOT_RUNBOOK.md).

## Install

Python 3.11+, [uv](https://github.com/astral-sh/uv).

```bash
make setup      # .venv with pinned deps (CPU torch if no NVIDIA GPU), downloads YOLO26 weights into models/
make hw         # what hardware was detected and which model sizes that picks
make data       # public data reachable from here (see data/README.md for the manual ones)
```

## Run on a video file / webcam / RTSP camera

```bash
.venv/bin/python -m bree.cli run --source path/to/clip.mp4 --out out/myrun
.venv/bin/python -m bree.cli run --source 0                                  # webcam
.venv/bin/python -m bree.cli run --source rtsp://user:pass@cam/stream1 --payments http:8765
```

Outputs in `--out`: `annotated.mp4` (zones, boxes, IDs, skeletons, heads pixelated), `frames.jsonl`
(per-frame boxes/IDs/keypoints, no images), `events.jsonl`, `alerts.jsonl` + `alerts/<id>.json` +
`alerts/<id>.mp4` evidence clips, `ledger_log.txt` (human-readable reasoning per person), `summary.json` (FPS, stage timings).

Store layout: `--store configs/your_store.yaml` (zones are polygons in camera pixels; copy
`configs/store_gas_station_small.yaml`). Draw them on a still frame from the real camera.

### Drawing zones (install day)
`scripts/draw_zones.py` grabs one frame (still image, video file or RTSP URL), opens it in an OpenCV
window, and writes the store YAML: click a polygon, Enter, type its name and kind (shelf, cooler,
register, exit = the door, entrance = a one-way way in); `f` toggles floor-point mode (click 4+ floor marks, type their x y in metres, for
multi-camera stores); `q` saves. Catalog, terminals, product classes, rules and ledger settings are
kept from the template (`--out` if it exists, else the example config). The result is loaded back
with the pipeline's own loader before the script says it is done.

```bash
PYTHONPATH=src .venv/bin/python scripts/draw_zones.py --source rtsp://user:pass@cam/stream1 \
  --out configs/store1_register_cam.yaml --camera-id register_cam --preview register_zones.png
# headless: --from-json zones.json ({"zones": [{"name", "kind", "polygon"}], "floor_points": [...]})
```
Only `--preview` writes an image, and it is the raw frame (not pixelated): keep it out of shared folders.

Several cameras of one store: repeat `--source` and `--store` (one YAML per camera, zones in that
camera's pixels, plus `camera.floor_points`: 4+ marks mapping its pixels to a shared floor plan in
metres). People keep one id across cameras by floor position and time, plus body appearance when
`reid` is on (see "Re-ID" below; never the face), and one ledger reconciles the store, so a pick on the cooler camera, payment at the
register camera and the exit at the door camera are one visit. Only door cameras need an `exit` zone.

```bash
.venv/bin/python -m bree.cli run --source rtsp://.../door --store configs/door_cam.yaml \
  --source rtsp://.../cooler --store configs/cooler_cam.yaml --source rtsp://.../register --store configs/register_cam.yaml
```

### Re-ID (same shopper across occlusions, cameras, shelf to register to door)
`rules: {reid: true}` in the store YAML turns on body re-identification (off by default: it helps on
angled, busy views and hurts on a straight-overhead camera, see REPORT.md "Re-ID without the face") (`src/bree/track/reid.py`): an
appearance embedding of the body below the shoulders (DINOv2 ViT-S/14, Apache-2.0, ONNX on CPU), clothing
colour per body part from the pose keypoints, body proportions and carried bags, fused into one match
score. It only extends the position/time rules: a stitch or camera handoff candidate that looks different
is rejected, and someone lost for up to a minute is relinked if they look the same and could have walked
there (lost at the cooler, picked up again at the register, so the receipt lands on the right basket).
The face is never used: the head is cut off before any feature is computed. Features live in memory for
the visit only (at most `reid_max_age_s`, default 2 h), are never written to disk or sent anywhere, and
there is no identity across visits. Some laws may still treat body or gait measurements as biometric
data: read DECISIONS.md "Re-identification without the face" and get legal advice before a pilot.
The first run exports `models/dinov2_vits14_reid.onnx` (downloads the DINOv2 weights once).

### Closed-world identity (a store is a room with a door)
`rules: {closed_world: true}` in the store YAML (off by default, see REPORT.md "Closed-world identity"
for the numbers and why). A shopper who has been seen stays the same shopper until they leave through
the door or have not been seen for `closed_world_timeout_s` (default 3600 s = 60 minutes). No new person
is created in the middle of the store:
- **Births only at the door**: a track that first appears in an `exit` or `entrance` zone, or within
  `entry_border_px` of the frame edge (set it for a camera whose frame edge is the way in; 0 = off).
  In the first `closed_world_warmup_s` (5 s) after start, people already inside may appear anywhere.
- **Any other new track is one of the people already inside and not visible right now.** Which one is
  solved as one assignment over all new tracks and all lost people (Hungarian), scored by appearance
  (the re-ID match, only when `reid` is on), walkability from where they were last seen (floor metres
  when the camera has `floor_points`, else body heights) and time since lost.
- **Deaths only at the exit zone or the timeout.** A lost shopper keeps their basket and ledger state.
- **Too close to call**: the track still gets exactly one identity, but both candidates are marked
  `identity_uncertain` (for `closed_world_uncertain_s`, default the whole visit), and the ledger caps
  any theft alert on them at the review tier, with the reason in `ledger_log.txt`.
- **Occupancy**: if a track appears inside while nobody is unaccounted for, it becomes a person flagged
  `missed_entry` (counted in `summary.json` under `identity`, logged in `engine_log.txt`). A second box
  on top of someone visible is ignored for up to 2 s instead.
Single camera only for now: with several cameras of one store it switches itself off (logged) and the
floor-plan handoff stays in charge. Works with or without `reid`; with `reid` on, appearance features of
a lost shopper stay in memory until they exit or time out, then are cleared (never on disk).

### Payments (POS / cooler taps)
One JSON object per payment, e.g.
`{"terminal": "pos_1", "ts": 1767000000.2, "txn_id": "T1042", "items": [{"sku": "COKE-20OZ", "qty": 1}]}`
(`ts` = unix time; or `t` = seconds since video start for recorded footage). Terminals map to zones in the store YAML.
- `--payments payments.jsonl` (file), `--payments stdin`, or `--payments http:8765` then
  `curl -X POST localhost:8765/payments -d '{...}'`.

### POS CSV exports
Real POS exports are CSV with their own columns, one row per line item, in local time. A mapping YAML
(`configs/pos_mapping_example.yaml`, matching `tests/fixtures/pos_sample.csv`) names the time column(s)
and format, receipt id, terminal (or a constant), item, qty and price columns, and maps POS item names
to catalog SKUs, to a category, or to `skip` (fuel, lottery). Rows become one payment per receipt.

```bash
.venv/bin/python -m bree.cli pos-convert --mapping configs/pos_mapping_example.yaml export.csv > payments.jsonl
```
In shadow mode set `pos_mapping:` and the POS can drop `*.csv` into `pos_export_dir` directly.

**Clock alignment.** The ledger runs on the edge box's clock (live: stream time = receipt unix time
minus the box's time at stream start). POS times are local wall time: `time.timezone` converts them
(DST included), then `time.offset_s` (edge box clock minus POS clock) is added. Keep the edge box on
NTP; at install ring up a test sale, note `date +%s` on the box when it prints, set
`offset_s = noted - printed`. Many POS print whole seconds or minutes only; minute stamps are too
coarse to place a receipt at the counter, so ask for seconds.

### ONNX Runtime (edge devices)
`--runtime onnx` (on `run`, `dashboard`, `shadow`; or `runtime: onnx` in the shadow YAML) runs the detector
and pose models from ONNX instead of PyTorch. Default stays `pytorch`.

```bash
.venv/bin/python -m bree.cli run --source clip.mp4 --runtime onnx
```

- Models: `models/<name>_<size>.onnx` at the sizes the pipeline uses (detector `--imgsz`, default 640; pose
  crop 160), static shape, batch 1. Exported on first use if missing.
- Execution provider: the best one ONNX Runtime has here, TensorRT > CUDA > CoreML (Apple silicon) > CPU,
  with the rest as fallback (`bree.edge.ort.ort_providers`). The log line `bree: ... switched to [...]` says
  which one loaded. TensorRT/CUDA need `onnxruntime-gpu` instead of `onnxruntime`; TensorRT engines are
  cached in `models/ort_cache/` (delete it after changing weights).
- Parity with PyTorch: `PYTHONPATH=src .venv/bin/python scripts/onnx_parity.py` -> `results/onnx_parity.json`.
  It also exports the concealment classifier to `models/conceal_poselift.onnx` (dynamic batch, normalisation
  and sigmoid inside the graph; `bree.conceal.onnx_track_scorer` runs it without torch).
- Speed: `scripts/speed.py` has a row per ONNX Runtime provider present (CPU, CoreML, CUDA, TensorRT).

## Shadow mode (pilot)

Runs the full pipeline on the store cameras and shows **nothing** to staff. Every would-be
alert (alert and review tier) goes to `<output_dir>/would_be_alerts.jsonl` with a head-pixelated
clip, the basket, unpaid items, confidence, whether concealment was seen, and the ledger's audit log.

```bash
cp configs/shadow_example.yaml configs/shadow_store1.yaml   # camera RTSP URLs, zones per camera, POS folder
.venv/bin/python -m bree.cli shadow --config configs/shadow_store1.yaml
# review page (127.0.0.1 only): http://127.0.0.1:8080/review
.venv/bin/python -m bree.cli shadow-labels --config configs/shadow_store1.yaml            # counts + precision so far
.venv/bin/python -m bree.cli shadow-labels --config configs/shadow_store1.yaml --export labelled.jsonl
.venv/bin/python -m bree.cli shadow-labels --config configs/shadow_store1.yaml --serve    # review page only
```

- **POS**: the POS drops export files (`*.jsonl` / `*.json`, the payment format above, with `ts`;
  or `*.csv` with `pos_mapping:` set, see POS CSV exports) into `pos_export_dir`; shadow mode polls it every 2 s. Files already there at start are skipped.
  A receipt that arrives after the decision (batched exports) still counts for `late_receipt_window_s`
  (default 300 s) after the exit: it retracts or downgrades the would-be alert, never raises one. The
  retraction is a second record in `would_be_alerts.jsonl` (`retracts` = the earlier id) and the
  review page marks the card RETRACTED. For batched exports also raise `exit_grace_s` to the batch delay.
- **Multi-camera store**: `multicam: true` in the shadow YAML runs every camera into one ledger
  (see the single-camera notes above; each camera's store YAML needs `camera.floor_points`).
- **Review**: on `/review`, Kiro and the operator mark each would-be alert *Real theft*, *False alert*
  or *Unsure*, with an optional note. Labels are appended to `labels.jsonl` with reviewer name and
  time; the latest label per alert counts. Precision = real / (real + false).
- **Cameras** reconnect forever; `--once` runs each source once (recorded footage, with `t` in receipts).
- **Raw recording** (`record_raw`) is off by default. Raw segments are **not** pixelated and contain faces.

## Demo, tests, benchmark

```bash
make demo       # 8 rendered TOY clips end to end (pixels -> tracker -> events -> ledger -> alerts) + scorecard
make test       # unit tests: ledger, event engine, simulator, payments, re-ID, closed-world identity
                # (incl. a scripted multi-shopper store guard); + vision smoke tests
                # + the re-ID regression guard (two MOT16 sequences; fails if IDF1 drops or false merges rise)
make sim        # event-level simulator: ledger accuracy under perfect / baseline / 2x vision noise
make reid-bench # identity: re-ID rank-1 / mAP; off / re-ID / closed world / both on MOT16 (sanity), MERL,
                # toy clips and a scripted store (false merges, splits, uncertain) -> results/reid_bench.json
make bench      # reid-bench, then everything else -> results/bench.json + results/bench.md
make dashboard  # local web page with live alerts + baskets (http://127.0.0.1:8080)
make export     # YOLO -> ONNX for edge devices
```

## Simulator scoring (`make sim-eval`)

Runs the pipeline on simulator camera frames and scores its alerts against the simulator's own
ground truth. The loop:

1. **Browser sim** (`~/bree/software/sim-prototype/`): place cameras, **Export layout**. That is the shared
   layout JSON (`~/bree/software/isaac-sim/layout_schema.json`). The browser sim's ground-truth JSON and
   single COCO frames are not used here: it exports no video.
2. **Isaac Sim on a cloud GPU** (`~/bree/software/isaac-sim/run.sh my_layout.json out_my_store`, see that
   kit's README): per-camera frames and `run1/events.jsonl` (one line per picked item: time, shopper,
   SKU, slot, `paid` or `concealed`, cameras that saw it). Copy `out_my_store/run1` back.
3. **Score**:
   ```bash
   make sim-eval SIM_OUT=~/bree/software/isaac-sim/out_my_store/run1 LAYOUT=~/bree/software/isaac-sim/my_layout.json
   # options: SIM_BACKEND=yolo|toy   SIM_ARGS="--pos-dropout 0.1 --pos-sku-noise 0.05 --fps 10 --zone-owner all --save-video"
   make sim-fixture      # the same chain on a generated TOY fixture, no GPU needed (about a minute)
   ```
4. **Scorecard** in `<SIM_OUT>/sim_eval/`: `scorecard.json`, `scorecard.md`, `inputs/` (what the adapter
   made), `pipeline/` (the usual `events.jsonl`, `alerts.jsonl`, `ledger_log.txt`). Change cameras or
   pipeline code, re-run, compare.

What each step does (`src/bree/sim/`):

- **Adapter** (`isaac_adapter.py`): finds every folder of `rgb_*.png` named like a layout camera (any
  nesting, `static/` skipped) and writes `<camera>.mp4` at the written frame rate (inferred from IRA's
  `config.yaml` and the events, else 10) plus `<camera>.store.yaml`:
  - zones projected through that camera's `camera_params` into pixel polygons, clipped to the frame.
    Shelf and cooler zones are the hull of the fixture's product slots (the event engine only counts
    an item as in a hand once it is outside the merch zone, so the zone is the stock, not the carcass).
    The register zone is a 1.2 m floor strip on the counter's customer side, the exit zone a 1.5 m floor
    strip inside the door. Zone name = fixture id;
  - `camera.floor_points` from a floor grid seen by that camera (floor plan metres = layout x, z);
  - one product class and one ledger category per SKU id.
  Each zone goes to the one camera that sees it largest (`--zone-owner best`): the pipeline does not
  merge one pick seen by two cameras, so `all` double-counts. Every camera is checked against the
  layout (a point on its optical axis must land on the image centre); transposed matrices are detected.
- **POS feed** (`pos_feed`): one receipt per paying shopper from the `paid` events, stamped at the last
  paid item plus a delay (`--pos-delay` 1.0 s, `--pos-delay-sd` 0.5), with optional lost receipts
  (`--pos-dropout`) and mis-rung lines (`--pos-sku-noise`), seeded.
- **Scorer** (`sim_eval.py`): an alert belongs to a shopper when one of its unpaid items was picked within
  3 s of one of that shopper's true picks at the same fixture (pipeline person ids are tracker ids, so
  identity is matched through the pick). A concealed item is caught when an alert-tier alert on its
  shopper is emitted after the concealment and within 300 s of it. Scorecard fields: `theft_recall`
  (also alert-or-review), `alert_precision`, `sku_correct_rate`, `time_to_alert_s` (alert minus
  concealment; the sim does not log exits), `false_alerts_per_hour`, review-tier count, false alerts
  split into honest-shopper vs unmatched; the same block `by_zone_kind`, `by_zone` (fixture) and
  `by_camera_kind` (thefts: kinds of camera that had the item in view; alerts: kind of the camera whose
  pick raised it); then one row per concealed event and per alert.

**What has and has not run.** Isaac Sim has not produced any output yet (no GPU), so none of this has
seen a real Isaac frame. It is tested on a synthetic fixture (`sim_fixture.py`, `tests/test_sim_eval.py`):
flat OpenCV drawings in the toy palette, written in the folder layout the Isaac kit documents, three
cameras, one paying shopper and one who conceals an energy drink. On it the chain gives 1 of 1 thefts
caught, 1 alert, 0 false, SKU correct; with the receipt dropped the honest shopper gets a review-tier
flag. That proves the harness, not accuracy. Expect to adjust on the first real run:

- Folder nesting, file names and frame numbering of IRA's writer may differ from what the adapter
  expects; video time 0 is assumed to be sim time 0.
- The default `yolo` backend uses COCO weights, which do not know the layout's SKUs: with no product
  detections there are no picks, so recall will read near zero until a product detector is trained on
  the Isaac labels (`bounding_box_2d_tight`), or an oracle-box backend is added.
- A shopper whose first box in a camera is cut by the frame edge gets a wrong floor position and so a
  second cross-camera id (seen while building the fixture, which is why its door faces all cameras).
- One zone per fixture: a long gondola watched by several shelf cameras is owned by one of them.

## Layout

```
src/bree/
  ingest/     video files, webcams, RTSP (with reconnect)
  detect/     YOLO backend (people + products in one pass), toy colour backend for toy clips
  pose/       top-down pose on person crops (COCO-17 keypoints)
  track/      ByteTrack for people, centroid tracker for small products, multi-camera handoff, body re-ID
  events/     zones, observations, rule-based event engine, event types
  ledger/     baskets, payment attribution, reconciliation + scoring, payment inputs
  alerts/     alert records, annotation + head pixelation, evidence clips
  sim/        event-level simulator (ground truth + vision noise), toy clip renderer, isaac/ (prep),
              Isaac kit adapter + POS feed + scorer + synthetic fixture (make sim-eval)
  eval/       metrics, benchmark harness, toy-clip scoring
  dashboard/  tiny local web dashboard
configs/      store layouts (YAML)
data/         gitignored datasets (see data/README.md)
scripts/      setup, data download
tests/        pytest
```

## Reading the event logic
`src/bree/events/engine.py` (one rule per event, thresholds in `EngineRules`) and
`src/bree/ledger/ledger.py` (reconciliation and the evidence scoring table in `LedgerConfig`)
are written to be read in one sitting. Every person's decision is explained in plain text in
`ledger_log.txt` and in each alert's `reasons`.
