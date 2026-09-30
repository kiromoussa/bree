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

### Payments (POS / cooler taps)
One JSON object per payment, e.g.
`{"terminal": "pos_1", "ts": 1767000000.2, "txn_id": "T1042", "items": [{"sku": "COKE-20OZ", "qty": 1}]}`
(`ts` = unix time; or `t` = seconds since video start for recorded footage). Terminals map to zones in the store YAML.
- `--payments payments.jsonl` (file), `--payments stdin`, or `--payments http:8765` then
  `curl -X POST localhost:8765/payments -d '{...}'`.

## Demo, tests, benchmark

```bash
make demo       # 8 rendered TOY clips end to end (pixels -> tracker -> events -> ledger -> alerts) + scorecard
make test       # unit tests: ledger, event engine, simulator, payments; + vision smoke tests
make sim        # event-level simulator: ledger accuracy under perfect / baseline / 2x vision noise
make bench      # everything -> results/bench.json + results/bench.md (~8 min on 4 CPU cores)
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
