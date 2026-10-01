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

- **POS**: the POS drops export files (`*.jsonl` / `*.json`, the payment format above, with `ts`)
  into `pos_export_dir`; shadow mode polls it every 2 s. Files already there at start are skipped.
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
