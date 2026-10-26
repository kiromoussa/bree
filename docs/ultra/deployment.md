# Deployment: edge box packaging, watchdog, disk, clock, update and rollback

Everything here was written and exercised on the dev machine (x86-64 Linux, systemd 26x user manager). What was **not**
possible here is called out: no Jetson, no ARM, no GPU container.

## 1. Pieces

| Piece | File | Tested how |
|---|---|---|
| systemd user units: pipeline, purge (hourly timer), doctor (10 min timer) | `deploy/systemd/*` + `install_user.sh` | `systemd-analyze --user verify` parses all five (only complaint: the venv path does not exist on the dev box). Not started as services here. |
| Stall watchdog + `sd_notify` | `src/bree/trust/supervise.py` | unit tests: a hung child is killed and the wrapper exits 75 within seconds; a child that beats is left alone; exit codes are mirrored. Real systemd `WatchdogSec` interplay is not exercised here. |
| Heartbeat in the frame loop | `pipeline.py` (`Heartbeat.beat`, env `BREE_HEARTBEAT_FILE`) | no-op unless the wrapper sets the variable |
| Pre-flight / periodic doctor | `python -m bree.trust doctor` | tests + run on this machine |
| Disk-full guard | `trust/health.DiskGuard` | unit test of the four states. **Not yet called by the pipeline's writers** (see gaps). |
| Update with automatic rollback | `deploy/update.sh` | test runs it against a throwaway git repo: switch, failed health check, rollback to the first release. |
| Container image (x86-64 CPU) | `deploy/container/Containerfile` | see section 5 for the build result |
| Raspberry Pi camera node | `deploy/pi/` (existing) | not touched; hub token can now be `file:`/`env:` |

## 2. Install on a box (no root except where marked)

```
git clone <repo> ~/src/bree && cd ~/src/bree
deploy/update.sh                          # builds ~/bree/releases/<stamp>_<sha>, venv, smoke test, switches `current`
deploy/systemd/install_user.sh            # copies the units, enables purge + doctor timers
cp configs/shadow_example.yaml ~/.config/bree/shadow.yaml   # edit: {file:...}/{env:...} for camera passwords
install -m 600 /dev/null ~/.config/bree/imou.env            # then put the credentials in it
systemctl --user start bree-shadow
# root, once: sudo loginctl enable-linger $USER   (units start at boot)   and a running NTP client (chrony / timesyncd)
```

Credentials: `EnvironmentFile=` in the unit loads `~/.config/bree/imou.env`; `bree-trust doctor` fails the start when that
file is group/other readable. RTSP passwords go in the shadow YAML as `rtsp://user:{file:~/.config/bree/cam1.pw}@host/..`
(`expand_refs`), so the YAML can be shared.

## 3. Watchdog and restarts

* `Restart=on-failure`, `RestartSec=10`, at most 6 starts in 15 minutes, then systemd stops trying (and the doctor timer
  shows the failure in the journal). `ExecStartPre` runs the doctor, so a dead clock or a world-readable credential stops
  the start instead of recording bad data.
* `Type=notify` with `WatchdogSec=180`: `bree.trust.supervise` pings systemd only while the pipeline touches its heartbeat
  (every 5 s while frames flow). A hang (RTSP read that never returns, deadlock, GPU wedged) stops the pings; the wrapper
  also kills the child after 120 s without a heartbeat and exits 75. Either way the unit restarts.
* Caveat: a camera that is simply offline produces no frames, so the heartbeat stops and the unit restarts every ~2 minutes
  while the camera is down. That is noisy but harmless; if it bothers, the pipeline should beat while waiting for a source.

## 4. Disk full

* Retention first: the purge timer deletes by age every hour; `max_total_gb` is an emergency cap over evictable classes
  (raw, bursts, debug video), oldest first.
* `DiskGuard(path, low_gb=20, critical_gb=5)`: under low, run an early purge once; under critical, `allow_video()` returns
  False so writers stop opening video/image files, while `alerts.jsonl` and the ledger (KBs) keep working. State returns to
  ok as soon as space frees. The doctor reports `disk` warn/fail with the same thresholds.
* **Gap**: the guard is a tested building block; wiring `allow_video()` into `alerts/writer.py` and `shadow.RawRecorder` is
  left to the integrator because those files are shared. Until then a full disk makes `cv2.VideoWriter` fail and the
  pipeline's own error handling applies. The recorder (`imou_record.py`) has its own size cap (80 GB).

## 5. Container

`docker build -f deploy/container/Containerfile -t bree-edge:dev .` from the repo root. CPU-only torch is installed first from
the PyTorch CPU index so the 2 GB CUDA wheels are not pulled. Run read-only, all capabilities dropped, config mounted
read-only, models mounted (weights are not in git). `HEALTHCHECK` runs the doctor. Build result and image size: see
`trust.md` section "Measured".

* **GPU**: not built or tested. Use an `nvidia/cuda` runtime base and the default CUDA torch wheel, run with
  `--gpus all`. torch/ultralytics GPU inference is the fast path on the dev machine (ONNX Runtime has no CUDA provider here).
* **Jetson / ARM caveats**: the x86 image does not apply. Use NVIDIA's `l4t-pytorch` base matching JetPack; `torch` wheels
  come from NVIDIA, not PyPI; install `ultralytics` without letting pip replace torch; build TensorRT engines on the device
  (`yolo export format=engine half=True`); `opencv-python-headless` aarch64 wheels exist but without CUDA. Expect the YuNet
  face layer on CPU: at 1280 px it costs a few ms to tens of ms on x86; measure on the device. None of this is tested.
* Raspberry Pi Zero 2 W nodes only run `bree.edge.node` (512 MB, no torch); they do not run the face layer.
* Licence: the image bundles Ultralytics (AGPL-3.0). See `sbom.md`.

## 6. Clock and NTP

Receipt matching depends on wall time. `bree-trust doctor` checks `timedatectl` NTP sync (then `chronyc tracking`) and
fails when unsynchronised. `health.device_clock_check` compares a camera/DVR clock to the box and recognises whole-hour
offsets as timezone settings. Real finding on this footage: the DVR overlay clock reads about 2 h behind the file names
(the box is EDT; 16:29 on screen at 18:28 box time). Record a per-station `clock_offset_s` and verify it against a POS
timestamp before trusting any receipt join. Not resolved here: whether the store is Eastern or Central.

## 7. Update and rollback

`deploy/update.sh [ref]`: export the ref to `releases/<stamp>_<sha>` -> share the previous venv if `pyproject.toml` is
byte-identical, else build a new one -> smoke test (imports) -> atomically repoint `current` -> restart -> wait
`HEALTH_WAIT` (45 s) -> health check (`is-active`). Any failure after the switch repoints `current` to the previous release
and restarts it (exit 1). `deploy/update.sh --rollback` does the same by hand. The newest 3 releases are kept. Data and
config live outside the releases and are never modified, so a rollback cannot lose footage or labels. Limits: if a release
changes an on-disk schema (sqlite plate store, review store), the old code may not read the new files: there are none today.

## 8. Resource budget (this machine: 24 cores, RTX 4070 Laptop, 30 GB RAM)

All numbers are REAL-UNLABELLED timings on real 2560x1440 frames from `ch07`, `scripts/trust/budget.py`. The machine was
heavily shared during the second column (load average 60 to 75 on 24 cores), so treat it as a pessimistic bound.

| Item | Idle-ish probe | Loaded (load avg ~65) |
|---|---|---|
| YuNet face detect at 640 px wide | 8 ms | 61 ms |
| at 1280 px | 25 ms | 197 ms |
| at 2560 px (native) | 95 ms | 997 ms |
| Person detect + pose (yolo26s, GPU) | about 75 ms (foundation step) | 76 ms |
| Guard per frame, pose heads only (no face detector) | | 8 ms |
| Guard per frame, face detector every 2nd frame at 1280 | about 13 ms (computed: half of 25 ms + 1 ms) | 134 ms |
| Still guard (face detector at 2560, for a single evidence still) | about 100 ms | 1146 ms |
| Audit record (hash chain append) | | 34 microseconds |
| Audit verify, 2000 records | | 94 ms |
| Purge plan over 20,000 files | | 1.06 s |
| YuNet model on disk | 227 KB | |
| Detector + pose + decode, 2560x1440 (foundation step) | about 75 ms/frame | |

Disk per channel (recorder, main stream HEVC 2560x1440): about 2.1 Mbit/s = 0.94 GB per channel-hour; 7 channels = 6.3 GB/hour,
the 80 GB cap is reached in 12.7 hours. Pixelated alert clips are 640 px wide JPEG-in-mp4, a few hundred KB each.
The pipeline's CPU/GPU cost of the trust layer is the face detector; with `face_every=2` at 1280 it adds roughly 15% to the
75 ms per-frame perception cost at idle. Memory: the guard itself holds a few boxes; the audit log is append-only text.
