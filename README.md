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
make data       # public data reachable from here (see data/README.md for the manual ones); MOT16 is 1.95 GB
```

The research repo (browser simulator, camera layouts) is expected at `~/bree`. The simulator renders, `make sku-*`
and `make e2e-sim` read it from there; elsewhere, pass `SKU_LAYOUT=<layout.json>` to make and `--sim-src <dir>` to
`scripts/train/render_synth.mjs`. Nothing else in this repo needs it.

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
With several cameras of one store the pool is store-wide (one identity pool for all cameras, see "Camera
calibration, 3D slots and store-wide identity" below). Works with or without `reid`; with `reid` on, appearance
features of a lost shopper stay in memory until they exit or time out, then are cleared (never on disk).

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

## Camera calibration, 3D slots and store-wide identity

Three things that build on each other. All are off until a store is configured for them.

### 1. Calibrate each camera to the floor plan

Marks go in the camera's store YAML, under `camera:`.

```yaml
camera:
  id: cooler_cam
  resolution: [1520, 2688]
  hfov_deg: 25                 # the lens, if known (or pass --hfov, or --layout)
  floor_points:                # a pixel and the same spot on the floor plan, metres
    - [212, 2590, 0.0, 0.0]
    - [1368, 2590, 1.5, 0.0]
  marks_3d:                    # the same with a height: a shelf-edge corner, a fixture corner
    - [640, 1210, 0.4, 2.2, 1.37]
```

Floor plan metres are the layout file's `(x, z)`; a height is its `y`.

```
PYTHONPATH=src .venv/bin/python scripts/calibrate.py fit configs/store1_cooler_cam.yaml --hfov 25
PYTHONPATH=src .venv/bin/python scripts/calibrate.py fit configs/store1_cooler_cam.yaml \
    --layout ../bree/software/shared/layouts/recommended-47.json --camera-id COOLER-rod-1 --in-place
PYTHONPATH=src .venv/bin/python scripts/calibrate.py check configs/store1_*.yaml --max-px 5 --max-m 0.15
```

What `fit` can work out depends on what it is given:

| marks | lens | result |
|---|---|---|
| 4 or more on the floor | unknown | floor homography only (what the multi-camera handoff needs) |
| 4 or more, floor or raised | known (`--hfov`, `camera.hfov_deg`, or the camera's entry in `--layout`) | full pose (`cv2.solvePnP`) |
| 6 or more | unknown | pose and focal length together |
| none | `--layout` | the design pose from the layout file, written as method `layout` and reported as unverified by `check` |

`fit` prints the reprojection error (RMS and max, pixels), the floor error (metres, and leave-one-out when there are 5 or more floor marks) and, with `--layout`, how far the fitted pose is from the design pose. It writes `camera.calibration` into a copy of the YAML (`<name>.calibrated.yaml`, `--out`, or `--in-place`; YAML comments are not kept).

`check` exits 1 and prints one line per problem when a camera's reprojection error is over `--max-px` (default 5) or its floor error is over `--max-m` (default 0.15). It compares the marks in the YAML with the stored calibration, so after a camera is bumped: click the marks again on a fresh frame, then run `check`. Exactly 4 floor marks fit exactly and prove nothing, so `check` asks for a fifth.

Model limits: pinhole, square pixels, principal point at the image centre, no lens distortion.

### 2. 3D slot of a pick

With calibrated cameras and a slot layout (the layout JSON with `fixtures` and `slots`), every PICK event gets `meta["slot"]`: the slot id, its SKU, how it was found and from how many views.

```yaml
# shadow YAML
multicam:
  slots: ../bree/software/shared/layouts/baseline-47.json
  calibration: {cooler_cam: {...}, rail_cam: {...}}     # the camera.calibration blocks
```

or `rules: {slot_layout: path/to/layout.json}` in any camera's store YAML. In code: `StoreEvents(stores, {"calibration": {...}, "slots": layout})`.

How the slot is found, best source first:

- `hand_2view`: two or more calibrated cameras saw the same hand at the same moment (within 0.07 s). The wrist and elbow are triangulated and the hand is matched to a slot, with each direction weighted by how well it was measured.
- `item_1view`: the camera that raised the PICK first saw the item sitting on that shelf. The line of sight through that pixel meets the shelf face at the slot.
- `hand_1view`: one camera, hand only. The line of sight meets the shelf face; a hand held in front of the shelf is pushed sideways along that line, which is the error two views remove.

The second camera does not need the shelf zone. Give each shelf zone to one camera (the one that raises the PICK); any other calibrated camera that sees the shopper's hand adds the second view. If two cameras both carry the same shelf zone, both raise a PICK and the basket counts the item twice.

The error model has the same definition as the browser simulator's two-camera 3D metric (`TRI` in `~/bree/software/sim-prototype/js/cameras.js`): per camera, a 1 sigma miss across the line of sight of `sqrt((pixel error / pixels per metre)^2 + (range x pointing error)^2)`. `NoiseModel.pixel_px` is the simulator's `pxSigma`, `cam_rot_deg` is its `calibDeg`, and `cam_pos_m` (where the camera hangs) is a third term the simulator does not have. `tests/test_calib.py::test_error_model_is_the_simulators` checks the two-ray error against the simulator's formula at the image centre. Against the simulator's real output the two agree within about 10% and the simulator is the more cautious one (pipeline error 0.891 to 0.999 of the simulator's on the 45 camera layout, audit 2026-10-05): it adds a lens edge loss and uses range where this code uses depth. The predicted error itself is first order and about 4% low, up to about 12% off axis (audit Monte Carlo). The defaults differ on purpose: the simulator scores a shelf point known to the nearest pixel (0.29 px, 0.1 degree), the pipeline locates a pose keypoint on a hand (`NoiseModel()` = 5 px, 2 cm, 0.2 degree; `NoiseModel.sim()` gives the simulator's). Set them with `multicam: {noise: {pixel_px: 5, cam_pos_m: 0.02, cam_rot_deg: 0.2}}`. The simulator run at the pipeline's values is in `~/bree/research/camera-layouts-3d.md` (addendum).

The ledger still reconciles by category and does not read `meta["slot"]`. `make sim-eval SIM_ARGS=--slots` scores it against the simulator's slot.

### 3. Closed-world identity across cameras

`rules: {closed_world: true}` in a multi-camera store now means one identity pool for the whole store (before, the flag was switched off when a store had several cameras).

- A person is created only at a door: a track that starts in an `exit` or `entrance` zone of a camera that sees it.
- A track that starts anywhere else is someone already inside: a person another camera is watching at that floor position, or one of the people nobody sees right now. One assignment per frame over floor-plan position, time since lost and, with `reid: true`, appearance.
- An identity ends when a door camera reports the exit (dropped if another camera still sees the person afterwards) or after `closed_world_timeout_s` unseen.
- A call that is too close marks both identities uncertain. The mark stays on the store-wide identity, so every later event from any camera carries it and the ledger caps the alert at review.

`multicam: {closed_world: false}` in the shadow YAML keeps the plain floor-plan handoff for a store whose camera rules have the flag on. Default: off. Measured numbers and the reason are in REPORT, "Calibration, 3D slots and store-wide identity".

`make calib-bench` reproduces the numbers (SYNTHETIC geometry, 190 s on an M1 Max) -> `results/calib_bench.json`.

## Camera nodes (Raspberry Pi Zero 2 W + Camera Module 3)

Each item camera is a small node that watches its own shelf at low resolution and sends full-resolution
frames to the hub only when something moves into a shelf zone. The hub (the edge box) stores the bursts
and runs the usual pipeline on them.

```
Pi node:  camera ─► low-res stream (320 px wide) ─► trigger: change + motion inside the shelf polygons
                 └► full-res frames ─► 2 s ring buffer in RAM ─► on trigger: pre-roll + during + post-roll
                                                                  as JPEGs ─► disk spool ─► HTTP POST
hub:      POST /v1/burst ─► <out>/<camera>/burst/<boot>_<seq>/frame_0000.jpg ... + burst.json ─► burst.mp4 ─► pipeline
          POST /v1/heartbeat ─► health, clock offset          GET /v1/health
```

Every frame in `burst.json` has the camera id, the node's monotonic time, the hub's wall time for it,
the zone ids that were active, and whether it is `pre`, `during` or `post`.

**What the hub keeps, and for how long.** A stored burst is raw full-resolution JPEGs. Heads are NOT pixelated
in them (pixelation happens in what the pipeline writes: annotated video, alert clips). The hub deletes every
burst folder, with its `burst.mp4` and `pipeline/` output, 24 hours after it was received:
`--retain-hours 24` on `bree.edge.hub` (`Hub(retain_s=...)`), swept at start and every 10 minutes;
`--retain-hours 0` keeps everything and is for recorded test runs only. `/v1/health` reports `retain_s` and
`bursts_deleted_by_retention`. The review store's 30 days do not apply to these folders.

**Pre-roll** is a time span (`pre_roll_s`), with `fps x pre_roll_s` frames as the upper bound because the
frames sit raw in RAM. If the camera delivers faster than the configured `fps` the pre-roll is shorter
(2 s configured at 10 fps is 1 s at a real 20 fps), so set `fps` to what the camera really delivers.

Code: `src/bree/edge/trigger.py` (the detector), `capture.py` (camera sources, ring buffer, bursts),
`uplink.py` (wire format, spool, retries, heartbeat; the reasons for HTTP are at the top),
`node.py` (the loop and config), `hub.py` (receiver). Numbers: REPORT.md "Camera node first pass".

### Try it on this machine (no Pi, no camera)

```bash
make edge-test                                    # 32 tests, synthetic frames, real HTTP on localhost
# a hub, then a node that plays a toy clip as its camera (make demo renders the toy clips once):
.venv/bin/python -m bree.edge.hub --out out/hub --host 127.0.0.1 --port 8787 --token demo \
    --store cam_main=configs/store_gas_station_small.yaml --backend toy &
{ cat configs/store_gas_station_small.yaml; printf 'node:\n  source: data/toy/toy_put_back.mp4\n  fps: 15\n  hub_url: http://127.0.0.1:8787\n  token: demo\n  spool_dir: out/node_spool\n'; } > out/node_demo.yaml
PYTHONPATH=src .venv/bin/python -m bree.edge.node --config out/node_demo.yaml
curl -s -H "Authorization: Bearer demo" http://127.0.0.1:8787/v1/health
ls out/hub/cam_main/burst/*/            # frames, burst.json, burst.mp4, pipeline/events.jsonl
```

`node.source` can also be a webcam index (`"0"`). The node config is a store YAML (`camera`, `zones`,
so `scripts/draw_zones.py` can write it) plus a `node:` block.

### Install on a Pi

Not yet run on a Pi: the kit was written and tested on a Mac with a file as the camera. The picamera2
path (`Picamera2Source` in `capture.py`) follows the picamera2 manual and has never touched a sensor.
Expect to fix small things on the first boot.

1. Flash **Raspberry Pi OS Lite (64-bit, Bookworm)** with Raspberry Pi Imager. In the imager settings
   set a hostname per camera (for example `bree-g1l-a`), enable SSH with a key, and set the time zone.
   Connect the Camera Module 3 ribbon and the PoE hat or splitter, boot, and check the camera:
   `rpicam-hello --list-cameras` should list `imx708`.
2. Copy the repo to the Pi (only `src/bree/edge`, `src/bree/ingest` and `deploy/pi` are used):
   ```bash
   rsync -a --include='src/***' --include='deploy/***' --exclude='*' ./ pi@bree-g1l-a.local:bree-vision/
   ssh pi@bree-g1l-a.local 'sudo bree-vision/deploy/pi/install.sh'
   ```
   The script installs `python3-picamera2 python3-opencv python3-numpy python3-yaml chrony` with apt,
   copies the node to `/opt/bree-node`, creates the `bree` user, writes `/etc/bree/node.yaml` (only if it
   is not there) and enables `bree-node.service`. No pip and no virtualenv. Run it again to upgrade.
3. Draw the shelf zones on a still from this camera, at the resolution the node will capture:
   ```bash
   ssh pi@bree-g1l-a.local 'rpicam-still --width 2304 --height 1296 -o still.jpg' && scp pi@bree-g1l-a.local:still.jpg .
   PYTHONPATH=src .venv/bin/python scripts/draw_zones.py --source still.jpg --out configs/g1l_a.yaml --camera-id shelf_g1l_a
   ```
   Draw the shelf face itself; the trigger adds a margin (`trigger.margin_px`) for hands at the front edge.
4. Edit `/etc/bree/node.yaml` on the Pi (`deploy/pi/node.example.yaml` explains every field): paste the
   `camera` and `zones` blocks from step 3, set `node.hub_url` and `node.token`.
5. On the hub: `make edge-hub TOKEN=<the same token> HUB_ARGS="--store shelf_g1l_a=configs/g1l_a.yaml"`.
   Use a long random token and keep the camera network separate from the store's guest Wi-Fi: the link
   is plain HTTP, so anyone on that network could read frames. TLS is not built in.
6. Start and watch:
   ```bash
   sudo systemctl start bree-node && journalctl -u bree-node -f
   curl -s -H "Authorization: Bearer <token>" http://<hub>:8787/v1/health     # on the hub
   ```
   Wave a hand in front of the shelf: a folder appears under `out/hub/shelf_g1l_a/burst/`.
   `trigger_ms` in the health reply is the real per-frame trigger cost on that Pi.

**Tuning on site** (`node.trigger` in the config): if hands are missed, lower `min_blob_px` or raise
`margin_px`; if the node triggers on nothing (flicker, a cooler door reflection), raise `diff_thr` and
`chroma_thr`. If capture stalls or the node is killed for memory, lower `fps`, `pre_roll_s` or
`main_size`. `spool_files` and `spool_dropped` in the heartbeat show whether the hub keeps up.

**Clocks.** Frames carry the node's monotonic time and the hub converts it to its own wall clock, so a
node with a wrong wall clock still gets correct frame times. The install script still sets up chrony;
point it at the hub (`server <hub ip> iburst` in `/etc/chrony/chrony.conf`) so logs line up.
`clock_offset_s` in the heartbeat is the node's wall-clock error as seen from the hub.

**Several node cameras in one store.** `pipeline_feed` above runs the pipeline once per burst, which has no shopper identity
or basket across bursts. To run node cameras together with ordinary streams into one ledger, read each camera's stored
bursts back as one frame stream with gaps and hand it to the pipeline as that camera's source:

```python
from bree.edge.hub import BurstSource
from bree.pipeline import CameraInput, run_store
cams = [CameraInput("shelf_g1l_a", BurstSource("out/hub/shelf_g1l_a"), store_a),      # a node camera
        CameraInput("door_cam", "rtsp://.../door", store_door)]                        # a continuous stream
run_store(cams, backend, "out/store", handoff={"closed_world": True})
```

Each frame keeps the time the hub gave it, so floor-plan handoff, store-wide closed-world identity and the 3D slot
of a pick (which need the same moment from two cameras) work across node cameras. With real nodes (hub time) give
every `BurstSource` the same `t0=` (for example the store's opening time as a Unix time); `run_store` refuses two
hub-time sources without one, because each would start its clock at its own first frame. Frames that have no hub
time (spooled before a node reboot, delivered after a hub restart) are dropped and counted in
`dropped_no_hub_time`. It reads the bursts on disk when
it starts (recorded bursts); following a live hub is not built. `make e2e-sim` runs this path on SIMULATED views.

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

## Human review and the feedback loop

Every alert that goes to a person is recorded in a review store, a reviewer decides on it in a few
seconds, and the decisions become training labels, metrics and a weekly report for the owner.
Code: `src/bree/review/`. Everything runs on the box. Nothing is sent anywhere.

```bash
# 1. put alerts in the store (a pipeline run, or shadow mode's would_be_alerts.jsonl)
#    make demo writes one folder per clip: out/demo/<clip>/alerts.jsonl and frames.jsonl
.venv/bin/python -m bree.review --store out/review ingest out/demo/walkout/alerts.jsonl --camera cam1 \
    --frames-log out/demo/walkout/frames.jsonl  # optional: pose windows for pick / conceal labels
.venv/bin/python -m bree.review --store out/review ingest out/shadow/would_be_alerts.jsonl

# 2. review: http://127.0.0.1:8090/  (keys 1 to 4 decide and load the next alert;
#    key 3 = wrong item: type the right item, then Enter; key U = back to the alert just decided)
.venv/bin/python -m bree.review --store out/review serve

# 3. log a staged test theft, so the weekly report can say caught or missed
.venv/bin/python -m bree.review --store out/review staged TEST7 --camera cam1 --item energy_drink

# 4. labels, numbers, owner report
.venv/bin/python -m bree.review --store out/review export
.venv/bin/python -m bree.review --store out/review metrics
.venv/bin/python -m bree.review --store out/review report --week-start 2026-09-28
#    with no --week-start: the seven full UTC days ending yesterday (today is not in it)

# 5. retrain hook: run a training command once 50 new examples exist. The detector fine-tune script reads the manifest:
.venv/bin/python -m bree.review --store out/review retrain --min-new 50 \
    --command ".venv/bin/python scripts/train/finetune_real.py --manifest"

# the whole loop on SYNTHETIC alerts (no cameras, no models, about 2 s)
.venv/bin/python scripts/review/synthetic_e2e.py            # add --serve to click through the page
```

- **Store** (`<store>/review.sqlite`): per alert the clip path, frames, camera, zone, predicted item,
  register match, identity-uncertain flag and event time; per review the decision (confirmed theft,
  not theft, theft with the wrong item, unclear), corrected item, reviewer id, time of the decision
  and how long the reviewer looked. Reviews are append-only. The latest one per reviewer counts.
  A corrected item is only kept with the "wrong item" decision, and is stored trimmed, in lower
  case, with spaces as underscores, so "Chips " and "CHIPS" are both `chips`.
- **Event time**: shadow records carry a wall-clock time (when the alert was emitted, a few seconds
  after the exit) and that is used. Plain pipeline alerts carry only video seconds, so their event
  time is the time of ingest. Time to decision, week buckets and retention run from that time, so
  ingest pipeline alerts soon after the run. The fix is for the pipeline to write a wall-clock time
  when it is wired to the store.
- **Ingest**: pipeline alert ids restart every run, so the stored id is run folder name, alert number
  and a hash of the record. The same file twice adds nothing. A re-run into a folder with the same
  name is stored as new alerts with a warning. A clip path that cannot be found (tried next to the
  alerts file, in the current folder and in every folder above the file) gives a warning too.
- **Retractions** (a late receipt lowered or withdrew an alert): the record is never stored as an
  alert of its own. It marks the alert it points at, also when that alert was ingested from an
  earlier file. If the alert is not in the store, the record is skipped with a warning. An alert
  that was withdrawn completely is taken out of the review queue and out of every metric, and is
  counted on its own (`retracted_alerts`, "withdrawn after a late receipt" in the owner report).
  An alert only lowered to the review tier stays in the queue and the metrics, and the page says
  "Later lowered to review by a late receipt". Decisions already made on a withdrawn alert still
  go into the label export: the reviewer watched the clip.
- **Reviewer page**: one alert at a time. Plays the head-pixelated clip, shows the item, the register
  record and why the system alerted. Two optional answers (pick seen, concealment seen) make better
  labels for the classifiers. Guards against decisions nobody looked at: a held key counts once, a
  key is ignored while the last decision is being saved and for 0.4 s after an alert appears, and
  key U (or the Back button) reopens the alert just decided so a slip can be fixed. Back goes one
  alert back, not further. An item name that is not in the list needs Enter twice. The page
  answers only when it is opened as `127.0.0.1` or `localhost`.
- **Label export** (`<store>/datasets/review/`): detector examples (YOLO text files, the reviewed box
  first), pick and conceal keypoint windows (`.npz`), and `manifest.json`. Duplicates are dropped by a
  hash of the content. Alerts where two reviewers disagree are left out. The manifest format is
  documented at the top of `src/bree/review/labels.py`. The manifest has no train / test split:
  split by `alert_id`, so frames and windows from one alert stay on one side. A detector example
  whose class the pipeline has never named in this store has `class_known: false` and is counted
  under `counts.unknown_classes` (a typo, or a product the detector does not have yet): drop or map
  those before training. One dataset folder belongs to one store. An export into a folder that
  holds another store's manifest is refused.
- **Metrics**: precision per week, review rate, median time from the event to a decision, per zone and
  per camera, and agreement between two reviewers (share of same decisions, Cohen's kappa). Alerts
  caused by staged tests and alerts withdrawn by a late receipt are counted on their own
  (`staged_alerts`, `retracted_alerts`) and are in no other figure. The `metrics` command and the
  owner report use the same rule, so a week has one precision number. The same event logged under
  two alert ids still counts twice in the metrics (the label export dedups, the metrics do not).
- **Owner report**: alerts sent, confirmed, rejected, staged test thefts caught or missed. Alerts caused
  by staged tests are left out of the precision figure. A staged test matches an alert that names
  it, or one on the same camera from 30 s before to 300 s after the staged time, also across the
  week boundary. A test logged with no camera matches an alert on any camera in that window, so
  log the camera. An alert that names a test id nobody logged is still counted as staged, and the
  report lists the id.
- **Retrain hook**: `bree.review.labels.retrain_hook(store, min_new=..., command=...)` or the `retrain`
  command. It exports, counts examples not yet used, and runs `<command> <manifest.json>` when there are
  enough. It never trains anything itself. A command that cannot be started gives `ran: false` and
  the reason in `error`, and the examples stay unused.
- **Privacy**: the head-pixelation check is best effort, by folder name. A clip must be in a folder
  named `alerts` and a frame image in a folder named `frames` (where the pixelating writers put
  them), and any path with a folder starting with `raw`, or with `unblur` or `unpixel` in its name,
  is refused. Nothing inspects the pixels, so a raw file copied into an `alerts` folder would get
  through. Items, receipt lines, frames and product boxes are stored through a fixed list of
  fields (`ITEM_KEYS`, `PAID_KEYS`, `FRAME_KEYS`, `PRODUCT_KEYS` in `store.py`). Any other key is
  dropped, whatever it is called, so a re-ID vector or a crop cannot ride along under a new name.
  The keypoints field is pose and is kept (see the decisions note). Retention is
  `--retention-days` (default 30, kept in the store). Past it, the clip, the frames, the stored
  keypoints and the exported examples are deleted, every time the store is opened and every hour while
  the page is up. The decision row stays, so the counts stay. Exports written with `--out` to a
  folder outside the store are recorded in the store and purged the same way. A copy somebody makes
  of a dataset folder by hand is not tracked and is not purged. A dataset folder whose
  `manifest.json` cannot be read does not stop the store: the purge result lists it under
  `dataset_errors` and the rest is purged. Manifests are written through a temp file and renamed.

The detector side reads this export: `scripts/train/finetune_real.py --manifest <manifest.json>` builds a YOLO
dataset from the detector examples (`bree.train.dataset.manifest_to_yolo`: split by `alert_id`, examples with
`class_known: false` left out, class ids = the manifest's `classes`), in a temp folder that is deleted when
training ends so no copy outlives the store's retention. Checked on a SYNTHETIC export in
`tests/test_detector_train.py`; no training run has been made from review labels.

Known gaps: the pipeline does not yet save a clean frame with the item box when it alerts, so real
alerts give pick windows only (from `frames.jsonl`) and no detector examples. The hook for that is
`bree.review.store.save_evidence_frame`. Alert clips have boxes and zones drawn on them and are not
used as training images.

## Demo, tests, benchmark

```bash
make demo       # 8 rendered TOY clips end to end (pixels -> tracker -> events -> ledger -> alerts) + scorecard
make test       # unit tests: ledger, event engine, simulator, payments, re-ID, closed-world identity
                # (incl. a scripted multi-shopper store guard), calibration and 3D slots, store-wide identity,
                # camera node and hub, review loop, SKU detector maths; + vision smoke tests
                # + the re-ID regression guard (two MOT16 sequences; fails if IDF1 drops or false merges rise)
make sim        # event-level simulator: ledger accuracy under perfect / baseline / 2x vision noise
make reid-bench # identity: re-ID rank-1 / mAP; off / re-ID / closed world / both on MOT16 (sanity), MERL,
                # toy clips and a scripted store (false merges, splits, uncertain) -> results/reid_bench.json
make bench      # reid-bench, then everything else -> results/bench.json + results/bench.md
make dashboard  # local web page with live alerts + baskets (http://127.0.0.1:8080)
make export     # YOLO -> ONNX for edge devices
make calib-bench   # SYNTHETIC: calibration error, 3D slot of a pick, store-wide identity -> results/calib_bench.json
make edge-test     # camera node and hub tests;  make edge-measure -> results/edge_measure_*.json
make review-demo   # SYNTHETIC run of the review loop;  make review / make review-report on a real store
make sku-data sku-train sku-eval   # SIMULATED: render, train and score the SKU detector -> results/sku_detector.*
make sku-clip-door && make e2e-sim   # SIMULATED end to end chain -> data/synth/clip_5001_door/e2e/e2e.md (E2E_RESULTS=e2e_sim_chain: results/)
make gc            # drop unreachable git objects; run when nothing else is using the repo
```

`make test` runs everything: 313 tests, about 11 minutes on this Mac (most of it the MOT16 guard and the vision smoke
tests). `make test-fast` needs no model weights and takes about a minute (300 passed, 1 skipped, 12 deselected). The one
skipped test drives the reviewer page in headless Chrome: set `BREE_PLAYWRIGHT=<path to node_modules/playwright>` (and have
node and Chrome installed) to run it, then the fast suite is 301 passed and `make test` is 313 passed. Tests that need
data or weights that are not there (MOT16, the sim-trained detector, `~/bree`) skip and say why.

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
   # options: SIM_BACKEND=yolo|toy|sim_sku   SIM_ARGS="--pos-dropout 0.1 --pos-sku-noise 0.05 --fps 10 --zone-owner all --save-video"
   #          SIM_ARGS="--closed-world --slots"   store-wide closed-world identity, 3D slot of each pick (scored against the sim's slot)
   make sim-fixture      # the same chain on a generated TOY fixture, no GPU needed (about a minute)
   ```
   A real clip takes minutes, most of it before the pipeline starts: the adapter first turns each camera's PNGs
   into a video. One run on this Mac (M1 Max, other jobs running), 6 cameras x 530 frames of 4MP with
   `SIM_BACKEND=sim_sku`: 10 min 35 s, of which 8 min 42 s was the adapter and 1 min 52 s the pipeline. It prints
   a line per stage, per camera and per 100 frames, so silence for more than a few minutes is a hang.
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
  merge one pick seen by two cameras, so `all` double-counts. The door and the register go to a camera
  that watches people (overhead, entrance, checkout) when one sees them, not to an item camera. A zone
  partly behind the camera is kept (the points in front make the polygon): the recommended layout's rail
  cameras sit beside the gondola they watch. Every camera is checked against the
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
seen a real Isaac frame. It has run on clips rendered from the browser simulator (`make sku-clip`, the same
folder shape; see "SKU detector trained on simulated frames" and "End to end on simulated data" below),
and it is tested on a synthetic fixture (`sim_fixture.py`, `tests/test_sim_eval.py`):
flat OpenCV drawings in the toy palette, written in the folder layout the Isaac kit documents, three
cameras, one paying shopper and one who conceals an energy drink. On it the chain gives 1 of 1 thefts
caught, 1 alert, 0 false, SKU correct; with the receipt dropped the honest shopper gets a review-tier
flag. That proves the harness, not accuracy. Expect to adjust on the first real run:

- Folder nesting, file names and frame numbering of IRA's writer may differ from what the adapter
  expects; video time 0 is assumed to be sim time 0.
- The default `yolo` backend uses COCO weights, which do not know the layout's SKUs: with no product
  detections there are no picks. `SIM_BACKEND=sim_sku` uses the detector trained on browser-simulator
  renders (`make sku-train`); a detector for Isaac renders would be trained on the Isaac labels
  (`bounding_box_2d_tight`) the same way.
- A shopper whose first box in a camera is cut by the frame edge gets a wrong floor position and so a
  second cross-camera id (seen while building the fixture, which is why its door faces all cameras).
- One zone per fixture: a long gondola watched by several shelf cameras is owned by one of them.

## SKU detector trained on simulated frames

A product detector that names the SKU, trained on frames rendered from the browser store simulator. **SIMULATED data only:**
the classes are the simulator's invented SKUs, and on real footage it detects nothing useful until it is fine-tuned on real
labelled frames. Numbers and limits: REPORT.md "SKU detector trained on simulated frames".

```bash
export BREE_PLAYWRIGHT=<path to node_modules/playwright>     # rendering needs Chrome, node 18+, network for three.js
make sku-data       # render 240 randomised scenes from a private copy of the simulator, cut 640 px tiles, split by scene seed
make sku-data SKU_SEEDS=1000:1004 SYNTH=out/synth_smoke      # smoke run: 4 scenes, all three splits present
make sku-train      # YOLO26 nano, 6 epochs -> data/synth/weights/sim_sku.pt (+ .json record of what was trained)
make sku-eval       # held-out scenes: mAP per SKU, pixels against accuracy, variant confusion, speed -> results/sku_detector.*
make sku-clip       # one scenario as video frames in the folder shape sim-eval reads (add --cams to choose cameras)
make sim-eval SIM_OUT=data/synth/clip_5001 SIM_BACKEND=sim_sku
make sim-eval-sku   # the same with a product-recognition block, then the COCO baseline for comparison
make sku-finetune REAL_DATA=data/real_sku/data.yaml          # real labelled frames, once they exist
.venv/bin/python scripts/train/finetune_real.py --manifest out/review/datasets/review/manifest.json   # or the review store's labels
```

- `--backend sim_sku` (on `run`, `dashboard`, `shadow`, the hub and `sim-eval`): people and pose from the usual YOLO weights,
  products from this detector, run on 640 px tiles at the camera's native resolution, around people only. Weights:
  `data/synth/weights/sim_sku.pt`, or `BREE_SKU_WEIGHTS=<file>`.
- **Minimum pixels: 20 px across a 6.6 cm can**, the default threshold of the simulator and the camera layout reports. On
  held-out simulated scenes the detector has the right SKU for 97.3% of clearly visible shelf items at 20 to 25 px and 96.2% at
  15 to 20 px, so the default holds and nothing supports raising it. By the item's own box the floor is about 15 px on the
  short side (92.4% at 15 to 20 px, 81.7% at 10 to 15 px). Simulated evidence; real frames replace it.
- The simulator copy lives in `out/sim-copy` (made on first use; the original in `~/bree` is never run or edited). The one file
  kept in this repo is the overlay `scripts/train/sim/synth.js`. The copy is NOT refreshed when the simulator changes: which
  version it is stands in `out/sim-copy/COPIED_FROM.json` and is written into every `clip.json` and dataset `meta.json`. Delete
  `out/sim-copy` to render from the current simulator. Datasets, weights and clips are under `data/synth/` (gitignored;
  here 3.2 GB of frames, 2.9 GB of tiles and 8.1 GB for a 6 camera clip of 53 seconds).

## End to end on simulated data (`make e2e-sim`)

One command runs every stage on a simulator clip: camera nodes with the zone trigger (the item cameras), a hub on localhost,
one pipeline run over all cameras (sim-trained SKU detector, store-wide closed-world identity, 3D slot of each pick), alerts,
the review store with decisions taken from the simulator's ground truth, the weekly owner report, and the `make sim-eval`
scorecard. It also counts, on the simulator's ground-truth boxes of items in a hand, at which stage a pick is lost.

```bash
make sku-clip-door && make e2e-sim       # the recorded run's clip (8 cameras, door camera included) -> data/synth/clip_5001_door/e2e/e2e.md
make e2e-sim E2E_RESULTS=e2e_sim_chain   # the same, and replace the tracked results/e2e_sim_chain.md and .json
.venv/bin/python scripts/e2e_sim_chain.py --sim-out data/synth/clip_5001_door [--no-edge] [--no-closed-world] [--no-slots]
.venv/bin/python scripts/e2e_sim_chain.py --sim-out data/synth/clip_5001_door --refunnel      # recount the pick funnel of a finished run
```

`make e2e-sim` does not touch `results/` unless `E2E_RESULTS` is set. `make sku-clip` (no `-door`) renders a different clip:
the script picks the cameras and leaves the door camera out; it is what `make sim-eval-sku` reads.

The recorded run (REPORT.md "End to end on simulated data") caught 0 of 2 simulated thefts with 0 alerts: no pick became a PICK
event, because the item cameras rarely give a tracked person and the overhead cameras cannot see the items. The chain runs; the
recommended layout needs a pick decided across cameras, which is not built.

## One pipeline path for a multi-camera store (`bree.shelf.store`)

This is what `make bench-dev` runs. One call per clip folder (video per camera, `calibration.json`, `layout.json` with
the planogram, `register.jsonl`):

| step | cameras | module | output |
|---|---|---|---|
| shelf events | shelf rail, cooler, counter | `bree.shelf.events` with the `sim_sku_hands_v3` weights | takes and puts per slot, no person box needed |
| people | overhead, entrance | `bree.track.people`, `bree.track.floor` | one floor track per shopper, with clothing colour to keep two people apart who pass each other |
| who took it | | `bree.track.associate`, `bree.shelf.store.one_act_per_reach`, `bree.events.shelf` | PICK / PUT_BACK per shopper, one per reach, plus ENTER (with who they walk with), PAY (register visit), EXIT. Slot-watch-only takes and unconfirmed puts are not passed on (`confirm_puts`) |
| theft at exit | | `bree.ledger` with the register feed (`pos_lag_s`: how long after the payment the receipt is stamped, 1.5 to 4.5 s for the simulated feed, set per store) | alerts and reviews in `alerts.jsonl`, ingested into a review store (`<out>/review`) |

```
.venv/bin/python -m bree.shelf.store <clip folder> <out folder>            # everything
.venv/bin/python -m bree.shelf.store <clip folder> <out folder> --rejoin   # reuse the stored shelf events and person boxes
make bench-dev                        # SIMULATED DEV clips, about 10 minutes -> results/bench_dev.md
make bench-dev BENCH_ARGS=--keep      # the same without running the models again (about 20 s of tracking, association, ledger)
BREE_SKU_WEIGHTS=sim_sku make bench-dev BENCH_ARGS="--name old_weights"    # the first detector instead
```

The two slow steps keep their output in `<out>/pipeline` (`shelf_events.jsonl`, `people_<camera>.jsonl`). Nothing in
this path reads ground truth: the bench hands the runner a folder of links without `truth/`.

Limits. No concealment cue is built yet, so every flag is review tier and none is alert tier. The runner replays a
whole clip (tracks first, then association); a live version has to hold events for a few seconds. The stored person
boxes carry clothing colour histograms: fine for simulated clips, not to be written to disk in a real store (see
DECISIONS.md).

## Fixed benchmark (simulated clips)

Everything in this section is SIMULATED (browser store simulator copy). It is the one benchmark every pipeline
change is measured on.

```
make bench-clips     # render the DEV and TEST clips once (Chrome, node, ffmpeg, BREE_PLAYWRIGHT)
make bench-dev       # pipeline on the 6 DEV clips -> results/bench_dev.json and .md (scorecard + per-pick funnel)
make bench-test      # the 6 TEST clips: final numbers only, never for tuning
make bench-smoke     # one clip, 60 frames per camera, writes nothing to results/
```

- Splits are fixed by seed in `scripts/bench/manifest.json`: train 1000 to 4999, dev 7001 to 7006, test 9001 to
  9006. Training data may only come from train seeds.
- A clip is 4 to 8 shoppers over about 60 to 90 s on the recommended-3d-45 layout: honest shoppers and thieves,
  picks at gondolas, in the cooler and at the counter, put-backs, two shoppers at one shelf, register payments.
  16 to 20 cameras per clip as H.264 video, with calibration, the planogram, the register feed and ground truth.
- The scorer reports theft recall, alert precision, false alerts and reviews on honest shoppers, pick recall
  and precision, right SKU rate, time to alert, identities per shopper, and a funnel that says at which stage
  each true pick was lost. The pipeline never sees the ground truth: it is run on a view of the clip without it.
- Clip format, commands and the scorer's input contract: `scripts/bench/README.md` and the docstring of
  `src/bree/sim/bench.py`.
- The recorded baseline (per-camera engine at 19f51dc) is `results/bench_dev_baseline.md`: 0 of 20 thefts caught, 2 of
  70 picks, 4.0 identities per shopper. Rerun it with `make bench-dev BENCH_ARGS="--runner bree.sim.bench:run_pipeline --name baseline"`.

## Shelf events without a person box (`src/bree/shelf/`)

Item cameras (shelf rail, cooler, counter) report what left or came back to a shelf without any person box. Each
record follows the shared shelf event contract (`camera_id, t, t_start, t_end, kind, slot_id, sku_id, sku_conf,
count, source, hand_px, point_3d, point_sigma_m, evidence`). Inputs are pixels, the camera calibration and the
planogram (`layout.json`: which slot holds which SKU, slot position and row depth). No ground truth.

Three cues per camera, fused in `bree.shelf.events.ShelfCamera`:

| cue | module | what it reads |
|---|---|---|
| pixel comparison | `bree/shelf/diff.py` | a slot-sized patch that differs from the shelf picture and has stopped moving; the slot is the planogram slot whose projected front item it overlaps. Ignores big regions (a body, a cooler door) |
| slot watch | `bree/shelf/slots.py` | the SKU detector's boxes of stock on the shelf, matched to the predicted box of every row position of a slot: the front moved back by n positions is a take of n units, forward is a put. Works while an arm is still in view. Gondola and counter slots only |
| item in a hand | `bree/shelf/hand.py` | SKU detections on moving regions that are not shelf stock, and a hand point (the detector's hand class when the weights have one, a skin blob otherwise, or any callable passed as `hand=`). Confirms the SKU of a shelf change |

`source` is `shelf_diff` for one shelf cue alone, `both` when two cues agree or the item was seen in a hand,
`hand_item` for a held item alone (off by default, see the decisions). Extra fields: `cue`, `slots` (candidate
slots with weights), `row` (row position before and after), `count` (units, from the row positions).

```python
from bree.shelf.events import run_clip, fuse_views, ShelfCamera, camera_events
events = run_clip(clip_dir, jobs=3)              # every item camera of a clip folder, sorted by t
status = []; run_clip(clip_dir, status=status, evidence_dir="out/shelf/evidence")   # also camera health records and before/after crops
acts = fuse_views(events, layout)                # one record per act: views of several cameras merged, slot by vote
cam = ShelfCamera(camera_id, camera, layout, fps, detector=sku_detector)   # live: cam.update(frame) -> events, cam.finish()
```

```
.venv/bin/python -m bree.shelf.events data/synth/bench/train/clip_4903 --out shelf_events.jsonl [--fused] [--no-detector] [--evidence DIR] [--status status.jsonl]
make shelf-clips        # render TRAIN-seed clips 4900 to 4906, 4950, 4951 (SIMULATED)
make shelf-eval         # six held-out clips -> out/shelf/eval_heldout6.md
make shelf-robust       # clip 4903 with the picture shifted or the light changed -> out/shelf/robust_4903.md
make shelf-test
```

An event is reported about 3 s after the change (the wait for a second cue and for a put that cancels it).
`point_3d` is the slot's face centre from the planogram (no two-view triangulation); `point_sigma_m` is half the
slot width (at least 0.03 m). `evidence.before` and `evidence.after` are null unless an evidence folder is given, and
then only events the pixel comparison saw get crops.

Camera health. The pixel comparison undoes a small shift of the whole picture (up to 12 px at half resolution) and a
brightness change (gain 0.25 to 4) before comparing. When more than half the picture differs from the reference it
reads nothing, and once the view has been still for 1 s it takes a new reference. Each camera keeps status records
(`ShelfCamera.status`, `run_clip(status=[])`, `--status`): `{"camera_id", "t", "kind": "status", "status": ...}` with
`unreliable`, `reliable`, `reference_reset`, `brightness_changed`, `brightness_back`, `needs_recalibration`.
`bree.shelf.events.unreliable_windows(status)` gives the spans per camera in which "no shelf event" means nothing.
Status records are kept apart from the events, so the event stream holds takes and puts only.

For the association step:
- One camera never reports two takes (or two puts) of the same or the neighbouring facing within 4 s unless the
  opposite event lies between them. Different cameras still report the same act each: use `fuse_views`.
- Treat `count` above 1 as unconfirmed. No clip has a multi-unit pick yet.
- The slot can be the neighbouring facing (7 to 8 cm away), most often with the same SKU. Trust `sku_id` more than
  `slot_id` and read `slots` (candidates with weights).
- A put with no earlier take of that slot by the same shopper should not become a PUT_BACK on its own.

## Hand and held-item detector (SIMULATED, stream hand-detector)

A second version of the sim-trained SKU detector for the item cameras (rail, cooler, checkout). Same 33 SKU
classes plus a `hand` class, trained on frames where an arm is up at a shelf, an item is in a hand, or an item is
on the counter. It runs on the whole frame, so it does not need a person box. Simulated data only: the hands are
the block-shaped forearm ends of the simulator's figures.

Weights (all in `data/synth/weights/`, each with a `.json` record of how it was trained):

| name | what |
|---|---|
| `sim_sku` | the first detector, 33 SKUs. Still the default. Not changed. |
| `sim_sku_hands_v2` | `sim_sku` + hand class, 2 epochs on 5,788 tiles (15.5 minutes). Kept for the record. |
| `sim_sku_hands_v3` | `v2` + 3 epochs on 12,064 tiles (44 minutes). Use this one. |

### Switching

```
BREE_SKU_WEIGHTS=sim_sku_hands_v3 make sim-eval SIM_BACKEND=sim_sku          # any command that builds the sim_sku backend
BREE_SKU_WEIGHTS=sim_sku_hands_v3 .venv/bin/python -m bree.sim.bench dev     # the benchmark
BREE_SKU_WEIGHTS=sim_sku_hands_v3 BREE_SKU_ROI=full ...                      # also run every tile, not only tiles around people
unset BREE_SKU_WEIGHTS                                                       # back to sim_sku
```

`BREE_SKU_WEIGHTS` takes a version name from `data/synth/weights/` or a path to a `.pt` file. `BREE_SKU_ROI`
is `people` (default, the old behaviour) or `full`. With hand weights the backend never reports `hand` as a
product; the hand boxes of the last frame are on `backend.last_hands`.

### Calling it from code

```python
from bree.train.backend import SkuDetector
det = SkuDetector("sim_sku_hands_v3", device="mps", conf=0.25)
(item_boxes, item_confs, item_cls), (hand_boxes, hand_confs) = det.detect(frame)             # whole frame
... = det.detect(frame, rois=np.array([[x0, y0, x1, y1]]))        # only tiles touching these boxes
... = det.detect(frame, second_look=6)      # one more tile centred on each of the 6 best hands
sku_id = det.names[int(item_cls[i])]        # det.hand_cls is None for weights without the hand class
items_only = det(frame)                     # the old call, unchanged
```

Boxes are xyxy in frame pixels. `bree.shelf.hand` already uses `det.detect` and `det.hand_cls`, so the shelf
event stream picks the hand boxes up as soon as `BREE_SKU_WEIGHTS` names hand weights.

### Reproducing

```
export BREE_PLAYWRIGHT=/path/to/node_modules/playwright
make hands-data      # render seeds 2000:2042 (train) and 3000:3010 (held out), cut tiles -> data/synth/hands/sku and sku_x2
make hands-train     # sim_sku.pt -> sim_sku_hands_v2.pt (2 epochs on sku) -> sim_sku_hands_v3.pt (3 epochs on sku_x2)
make hands-eval      # old and new weights on the held-out frames -> results/hand_detector.md, .json
.venv/bin/python -m bree.train.eval_hands --report-only      # rewrite the .md from the saved .json
.venv/bin/python -m pytest tests/test_hand_detector.py tests/test_detector_train.py -q
```

The exact commands are in the Makefile. Results: `results/hand_detector.md` and `.json` (held-out simulated
frames), summary in `report.md` next to them.

## Who took it: store-wide people and shelf events (association stream)

Item cameras (rail, cooler) report shelf events in the shared contract without needing a person box. Three modules
turn them into the PICK / PUT_BACK events the ledger already reads:

| module | what it does |
|---|---|
| `bree.track.people` | person boxes with pose keypoints for the overhead and entrance cameras (the repo's detector plus crop pose, no tracker) |
| `bree.track.floor` | one tracker on the floor plan over all people cameras: one identity per shopper, closed world (births at the door, hand-back of lost people, second boxes, exits, timeout, uncertain marks) |
| `bree.track.associate` | attaches each shelf event to the person whose floor position and reach explain it; events close in time are solved together; marks doubt |
| `bree.events.shelf` | shelf events + people -> ENTER / PICK / PUT_BACK / PAY / EXIT (+ CONCEAL from cues) -> the unchanged ledger with the register feed |

```python
from bree.events.shelf import track_people, store_events, load_payments, run_ledger, write_run
tracker, ids = track_people(cams, layout, boxes, fps)        # cams: {id: bree.calib.camera.Camera} of the people cameras
                                                             # boxes[camera][frame] = [{"bbox", "kpts"}]
events, assocs = store_events(shelf_events, tracker.people(), layout, cams=cams, conceal=cues)
alerts, ledger = run_ledger(events, load_payments(clip / "register.jsonl"), layout)
write_run(out, events, alerts, boxes, ids, fps)              # the files bree.sim.bench scores
```

- `shelf_events`: the shared contract. `slot_id` or `point_3d` is needed (a slot gives the shelf face and its normal; a
  bare 3D point is matched by distance only). `sku_id` missing: the planogram's SKU of the slot is used.
- `cues` (optional): `{"t", "conf", "sku_id"?, and "person_id" or "point_3d"}`, one per concealment seen by an item camera.
- Each PICK / PUT_BACK carries `sku`, `zone` (the fixture), `meta.slot`, `meta.assoc` (cost, margin, why),
  `meta.uncertain`, `candidates` (the other people who fit) and `meta.identity_uncertain` (the tracker's doubt about who
  the person is; the ledger caps that person at review).
- Per-camera engines for item cameras: set `rules: {picks_from_shelf_events: true}` so the single-view PICK / PUT_BACK
  rule is off there.
- Knobs: `bree.track.floor.FloorConfig` (`encounter_m`: two identities this close are both marked uncertain, 0 turns it
  off), `bree.track.associate.AssocConfig` (`stand_m`, `sigma_*`, `reach_m`, `margin`).

Commands: `make assoc-test`, `make assoc-scripted` (SYNTHETIC), `make assoc-door`, `make assoc-dev` (SIMULATED; person
boxes are read from the frame logs of a pipeline run, default `out/bench/dev_baseline`, or made with
`python -m bree.track.people <clip> <out>` and passed with `--boxes`).

## Fuel drive-off (plates)

A vehicle fuels, leaves the pump, and nobody pays. `src/bree/plates/` links a vehicle and its plate to a pump
and a fuel sale, and raises a review alert when the sale stays unpaid after the vehicle has gone. Everything
measured so far is on SYNTHETIC plates and SCRIPTED timelines (`results/plates_bench.md`). No real
forecourt footage has been used.

```
make plates-test          # unit tests, no model weights needed
make plates-bench-quick   # about 3 minutes
make plates-bench         # the full tables, about 16 minutes -> results/plates_bench.md
make plates-purge         # delete expired plate records now
```

### Parts

| file | what it does |
|---|---|
| `reader.py` | plate detector + plate text reader behind two small interfaces (`PlateDetector`, `PlateOcr`), `make_reader()`, `vote()` over several frames |
| `pump.py` | `PumpZone`, `FuelSale`, `DriveOffMonitor` (pump-zone tracker and the drive-off rule), `IouTracker`, `yolo_vehicles` |
| `retention.py` | `PlateStore`: plate records with expiry, automatic deletion, access log |
| `synth.py`, `scenarios.py` | SYNTHETIC plates, forecourt frames and scripted timelines for tests and the bench |

### How to call it

```python
from bree.plates.reader import make_reader
from bree.plates.pump import DriveOffMonitor, PumpZone, FuelSale, IouTracker, yolo_vehicles
from bree.plates.retention import PlateStore

zones = [PumpZone("P1", [(40, 520), (600, 520), (600, 715), (40, 715)], camera_id="forecourt")]
plates = PlateStore("out/plates/store")                       # 72 h default, see Retention
mon = DriveOffMonitor(zones, reader=make_reader(), plate_store=plates, out_dir="out/run")   # fps=camera rate
tracker = IouTracker()

# per frame of the forecourt camera (t = seconds on the shared store clock)
alerts = mon.update(t, tracker.update(t, yolo_vehicles(yolo, frame)), frame, camera_id="forecourt")
# from the forecourt controller / POS
mon.on_sale(FuelSale("P1", t_start, t_end, amount=41.37, gallons=11.9, txn_id="T1042", prepaid=False))
alerts += mon.on_payment(t, txn_id="T1042")                    # or pump_id="P1"
# the existing alert and review path
for a in alerts:
    sink.emit(a)                                               # bree.alerts.writer.AlertSink
    review.add_alert(a.to_dict(), camera="forecourt")          # bree.review.store.ReviewStore
plates.sync_reviews(review)                                    # confirmed: keep 30 days; not theft: delete now
```

- A drive-off alert is a normal `bree.alerts.types.Alert` with `tier="review"`, one unpaid item of category
  `fuel` whose `zone` is the pump id, and `person_id` set to the vehicle track id. A late payment gives a
  retraction record (`retracts` set, tier `retracted`), the same shape the ledger already uses.
- A plate is stored only when the vote over the frames of the visit reaches confidence 0.9; otherwise the alert
  says "plate not read" and the reviewer has the clip.
- The alert never holds the plate text. It says "stored as plate record <id>". The text is read with
  `PlateStore.get(record_id, who, purpose)`, which is logged.
- `make_reader(detector="both" | "open" | "classical")`, default `both`. `DriveOffMonitor(reader=None)` runs the rule and the
  evidence clip with no plate reading at all: use that where plate reading is not allowed.
- `update(..., evidence=pixelated_frame)` takes the head-pixelated copy of the frame for the clip when people
  are in view. The module does not pixelate heads itself.
- Shelf events and shopper identity are not involved: this module shares only the alert and review path.

### The rule

A sale at pump P is a drive-off candidate when it is not prepaid, not paid, and fuel has stopped. The vehicle is
the one whose time in P's zone overlaps the fuelling time most. When that vehicle has been out of the zone for
3 s it has left; if no payment arrives within the grace period after that (default 120 s), a review alert is
raised. Stays under 5 s are ignored. Two vehicles in the zone during the sale lower the confidence and mark the
alert identity uncertain. An unpaid sale with no vehicle seen at all is flagged after 5 minutes at low confidence.

Gaps and track id changes (a person walks in front of the camera, the tracker renumbers the car):
- A new track id in the same place (box overlap 0.5 or more) as a track that has just stopped is the same visit.
- A box back in the same place after a gap is the same visit when that visit has an unpaid sale: within 60 s
  once fuel has stopped, with no time limit while the sale is still dispensing. The alert lists every such gap.
- The visit is split at the gap again, and the earlier part counts as left at the start of the gap, when a new
  sale starts at that pump after the gap, or when the plate read after the gap differs from the plate read
  before it by more than 2 characters. That is the next car taking the spot.
- The stored plate and the evidence clip come only from the time fuel was flowing for that sale. A car that
  arrives later can not put its plate on someone else's alert.
- Known limit: a drive-off with an unreadable plate, followed within 60 s by a car that takes the same spot
  and buys no fuel. The alert waits until that second car leaves and gives its leave time. It names the gap
  and stores no plate. Also: a customer who is hidden after fuelling and then starts a second sale is split
  like a new car, so the first sale is timed from the start of the gap.
- Sale updates match by `txn_id`. A controller with no transaction ids is supported: an update matches the
  sale at the same pump with the same start time, or the one still dispensing, and `on_payment(pump_id=...)`
  settles the oldest unpaid sale at the pump. Not handled: an update without a `txn_id` that arrives after the
  payment (it would open a new sale).

### Retention

| data | where | kept |
|---|---|---|
| plate reads of a vehicle that paid, or that had no sale | memory only, never on disk | at most 7 minutes after the vehicle leaves (grace period + 5 minutes) |
| plate text and plate crop of a drive-off alert | `PlateStore` (SQLite, `secure_delete`) | 72 hours from the event (`retention_hours`) |
| the same, after a reviewer confirms the drive-off | `PlateStore` | 30 days from the event (`confirmed_retention_days`) |
| the same, after "not theft" or a late payment | | deleted at once |
| access log (record id, who, purpose, time; no plate text) | `PlateStore` | 365 days (`access_log_days`) |
| evidence clip (shows the plate in pixels) | review store media | the review store's rule, 30 days by default |

Deletion is automatic: `purge()` runs when the store is opened, on every write and lookup, and once an hour
from any running `DriveOffMonitor` that holds the store (`purge_every_s`), so an idle store with a live camera
still loses its expired plates. With no monitor running, use `make plates-purge` from a timer. For the
integrator: also call `PlateStore.sync_reviews(review_store)` from the review server's hourly timer, so a
"not theft" decision deletes the plate within the hour. The evidence clip shows the plate in pixels and follows
the review store's 30 day rule, not the 72 hour plate rule; whether forecourt clips need the short clock is a
question for counsel. Not done: encryption at rest (use an encrypted volume).

### Components and licences

| component | version | licence | checked how |
|---|---|---|---|
| fast-plate-ocr (text reader, model `cct-s-v2-global-model`) | 1.1.0 | MIT | `LICENSE` in the installed package ("MIT License, Copyright (c) 2024 ankandrew"), package metadata `License-Expression: MIT`, https://github.com/ankandrew/fast-plate-ocr |
| open-image-models (plate detector, model `yolo-v9-t-384-license-plate-end2end`) | 0.6.0 | MIT | same two checks, https://github.com/ankandrew/open-image-models |
| ONNX Runtime | 1.30.0 | MIT | already a repo dependency |
| OpenCV (classical localiser, image handling) | 5.0.0 | Apache 2.0 | already a repo dependency |

Weights download on first use from the two projects' GitHub releases, with no sign-up, to `~/.cache/fast-plate-ocr`
and `~/.cache/open-image-models`.

Open points for counsel, not resolved here:
1. The MIT licence files cover the packages. Neither project page (read 2026-10-05) states a separate licence
   for the trained weights or names the training data.
2. The detector is a YOLOv9 model. The original YOLOv9 code (WongKinYiu/yolov9) is GPL-3.0; an MIT
   implementation (MultimediaTechLab/YOLO) also exists. The project page does not say which one produced these
   weights. If this can not be confirmed, use `make_reader(detector="classical")`: the OpenCV localiser has no
   weights. Its measured accuracy is in results.md.
3. There is no weight-free text reader. A small character reader trained on synthetic plates was not built,
   because the open reader could be downloaded.

### Legal notes on plate reading (not legal advice)

Plate reading is regulated in some states, and the rules are changing fast (several 2026 laws). Get counsel to
clear each state before turning plate reading on there. The list below was put together on 2026-10-05 from the
NCSL table "Automated License Plate Readers: State Statutes"
(https://www.ncsl.org/technology-and-communication/automated-license-plate-readers-state-statutes, page dated
2026-09-28) and the EFF list (https://www.eff.org/ALPRstatelaws). The statutes themselves were not read in
full. Treat every line as a pointer to what counsel must read, not as a conclusion.

States whose statutes reach private operators, or may:

| state | citation | why it matters to a store |
|---|---|---|
| Arkansas | Ark. Code 12-12-1801 to 12-12-1808 | Generally bars individuals and businesses from using plate readers, with listed exceptions. Assume plate reading is off unless counsel finds an exception that fits. |
| Maine | 29-A M.R.S. 2117-A | Use is limited to listed government purposes; a violation is a crime. Assume off. |
| New Hampshire | RSA 261:75-b, RSA 236:130 | Plate readers are limited to law enforcement with a purge within minutes; surveillance that identifies vehicles on public ways is restricted. Whether a camera on a private forecourt is covered is a question for counsel. Assume off until answered. |
| California | Cal. Civ. Code 1798.90.5 to 1798.90.55 | Applies to any "ALPR operator", private ones included: security procedures, a posted usage and privacy policy, access records, limits on sharing, and a private right of action. The access log in `PlateStore` is a start, not compliance. The CCPA (Cal. Civ. Code 1798.100 and following) may also treat a plate as personal information. |
| Utah | Utah Code 41-6a-2001 to 41-6a-2006 | Lists who may use plate readers and for what, and has rules for privately held plate data and how long it is kept. Counsel must check that a retail forecourt fits a permitted use. |
| Montana | Mont. Code 46-5-117 to 46-5-119 | Restricts plate readers on public highways. A forecourt camera that also sees the road needs a look. |
| Connecticut | S.B. 397 (2026) | Per the NCSL table: covers public and private vendors, deletion within 21 days, limits on sale and on some uses. Second source checked 2026-10-05 (a vendor summary and the amendment text on cga.ct.gov): signed 2026-05-04, 21 day retention with listed exceptions. |
| Washington | Wash. S.B. 6002 (2026) | Statewide rules on collection, access, sharing and retention of plate reader data. Second source checked 2026-10-05 (Washington State Standard, a vendor summary): 21 day retention with limited exceptions. Counsel must check the reach to a store. |
| Oregon, New Jersey | 2026 Or. Laws ch. 77; N.J. A.B. 4070 (2026) | NOT CONFIRMED beyond the NCSL table: a search on 2026-10-05 found no second source for either citation. Listed there as 2026 laws on plate reader data (retention near 21 to 30 days, bans on selling or sharing), mostly written for agencies and their vendors. Counsel must confirm that each exists as cited and whether it reaches a store, before this text is shown outside the team. |

States whose plate reader statutes are written for government or police use (retention figures as listed by
NCSL; they still show what lawmakers consider reasonable): Alabama (Ala. Code 41-9-620), Colorado (C.R.S.
24-72-113), Florida (Fla. Stat. 316.0777, 316.0778), Georgia (O.C.G.A. 35-1-22, 30 months), Idaho (Idaho Code
49-1432), Illinois (625 ILCS 5/2-130), Iowa (Iowa Code 321P.4, 30 days), Kansas (K.S.A. 45-221), Kentucky (KRS
189.632, 90 days), Maryland (Md. Code, Public Safety 3-509), Minnesota (Minn. Stat. 13.824, 60 days), Nebraska
(Neb. Rev. Stat. 60-3201 to 60-3209, 180 days), New Mexico (N.M. Stat. 29-22-1 to 29-22-5), North Carolina
(N.C. Gen. Stat. 20-183.30 to 20-183.33, 90 days), Tennessee (Tenn. Code 55-10-302, 90 days), Vermont (23
V.S.A. 1607, 1608), Virginia (Va. Code 2.2-5517, 21 days).

In every state:
- Federal Driver's Privacy Protection Act, 18 U.S.C. 2721 to 2725: a store can not turn a plate into an owner
  through motor vehicle records. The plate goes to the police with the report; BREE never looks anyone up.
- States with no plate reader statute still have general privacy, consumer protection and data breach laws.
- Many states have their own fuel theft (drive-off) statutes that say what a station may report and to whom.
- Product defaults that follow from this: plate reading is a separate switch (`reader=None`), nothing is kept
  for vehicles that paid, incident plates expire in 72 hours unless confirmed, every lookup is logged, plates
  are never shared or sold and never matched against any list. Signage at the forecourt and a written policy
  are the operator's job and are not in this repo.

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
  calib/      camera calibration to the floor plan, triangulation, 3D slot of a pick, synthetic bench
  edge/       ONNX export and runtime; camera node (trigger, bursts, uplink) and hub receiver
  review/     review store, reviewer page, label export, metrics, owner report, retention
  train/      SKU detector on SIMULATED frames: dataset, training, evaluation, pipeline backend
  eval/       metrics, benchmark harness, toy-clip scoring
  dashboard/  tiny local web dashboard
configs/      store layouts (YAML)
data/         gitignored datasets (see data/README.md)
deploy/pi/    camera node install kit (systemd unit, install script, example config)
scripts/      setup, data download, calibrate.py, e2e_sim_chain.py, edge/ review/ train/ tools
tests/        pytest
```

## Reading the event logic
`src/bree/events/engine.py` (one rule per event, thresholds in `EngineRules`) and
`src/bree/ledger/ledger.py` (reconciliation and the evidence scoring table in `LedgerConfig`)
are written to be read in one sitting. Every person's decision is explained in plain text in
`ledger_log.txt` and in each alert's `reasons`.
