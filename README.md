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
register, exit); `f` toggles floor-point mode (click 4+ floor marks, type their x y in metres, for
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
metres). People keep one id across cameras by floor position and time only (no appearance
features), and one ledger reconciles the store, so a pick on the cooler camera, payment at the
register camera and the exit at the door camera are one visit. Only door cameras need an `exit` zone.

```bash
.venv/bin/python -m bree.cli run --source rtsp://.../door --store configs/door_cam.yaml \
  --source rtsp://.../cooler --store configs/cooler_cam.yaml --source rtsp://.../register --store configs/register_cam.yaml
```

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
make test       # unit tests: ledger, event engine, simulator, payments; + vision smoke tests
make sim        # event-level simulator: ledger accuracy under perfect / baseline / 2x vision noise
make bench      # everything -> results/bench.json + results/bench.md (~10 min on 4 CPU cores)
make dashboard  # local web page with live alerts + baskets (http://127.0.0.1:8080)
make export     # YOLO -> ONNX for edge devices
```

## Layout

```
src/bree/
  ingest/     video files, webcams, RTSP (with reconnect)
  detect/     YOLO backend (people + products in one pass), toy colour backend for toy clips
  pose/       top-down pose on person crops (COCO-17 keypoints)
  track/      ByteTrack for people, centroid tracker for small products
  events/     zones, observations, rule-based event engine, event types
  ledger/     baskets, payment attribution, reconciliation + scoring, payment inputs
  alerts/     alert records, annotation + head pixelation, evidence clips
  sim/        event-level simulator (ground truth + vision noise), toy clip renderer, isaac/ (prep)
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
