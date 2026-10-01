# BREE pilot runbook (gas-station convenience store, shadow mode)

Goal of the first pilot: **show staff nothing**. Run BREE silently on the store's cameras and POS for 1 to 2 weeks,
review every would-be alert, and turn that into the first labelled data from our own cameras. Only then decide
whether anything goes live. Numbers to beat are in REPORT.md (simulator with measured rates: about 1.1 false
alerts per hour; we need far fewer before anything is shown to staff).

## 0. Before the visit (remote)
1. **Ask the operator for:**
   - camera make/model, resolution, fps, and whether each camera exposes an RTSP stream (URL, user, password);
   - one still frame per camera (or 10 seconds of video);
   - a sample POS export (CSV) covering a few hours, with **seconds** in the timestamps if the POS allows it;
   - the POS's time zone, and whether it exports continuously, in batches, or once a day.
2. **Write the POS mapping** from the sample export: copy `configs/pos_mapping_example.yaml`, set the column names,
   time format and time zone, terminal names, and map their item names to our catalog SKUs or categories. Map
   fuel, lottery, car wash and other non-merchandise lines to `skip`. Check it:
   ```
   .venv/bin/python -m bree.cli pos-convert --mapping configs/store1_pos.yaml sample.csv > /tmp/p.jsonl
   ```
   It prints how many receipts it read and any POS item names that are not mapped yet.
3. **Draw zones** on each camera's still (shelves, coolers, register customer side, exit/door):
   ```
   PYTHONPATH=src .venv/bin/python scripts/draw_zones.py --source still_cam1.png \
       --out configs/store1_cam1.yaml --camera-id cam1 --preview cam1_zones.png
   ```
   Check `cam1_zones.png`. The preview is the raw frame (people in it are not pixelated): keep it on the box, out of shared folders. If several cameras should share one ledger, also click 4 or more floor points per
   camera (`f` key) with their positions in metres on a shared floor plan, and set `multicam: true` in the
   shadow config.
4. **Replay first, if the operator can send recorded footage + the matching POS export** (no hardware needed):
   ```
   .venv/bin/python -m bree.cli pos-convert --mapping configs/store1_pos.yaml day.csv \
       --video-start 2026-10-05T14:00:00 > out/day_pos.jsonl
   .venv/bin/python -m bree.cli run --source cam1_1400.mp4 --store configs/store1_cam1.yaml \
       --payments out/day_pos.jsonl --out out/replay_cam1 --runtime onnx
   ```
   `--video-start` is the wall time (POS clock) of the video's first frame. Read `out/replay_cam1/ledger_log.txt`:
   every flagged person has a plain-English reason.

## 1. Hardware at the store
- One edge computer per store. Measured so far only on an Apple M1 Max: about 77 FPS with `--runtime onnx`
  (CoreML), about 5 cameras at 15 fps. A Jetson Orin is the cheaper target but not measured yet.
- Wired Ethernet to the camera network. The box needs no inbound ports; the review page binds to 127.0.0.1 only.
- Keep the box on NTP. The POS clock offset is measured in step 2.4.

## 2. Install day
1. `git clone`, `make setup`, `make test` (all should pass).
2. Copy the store configs and POS mapping; write the shadow config from `configs/shadow_example.yaml`:
   cameras (name, RTSP URL, store YAML), `pos_export_dir` (where the POS drops its exports), `pos_mapping`,
   `runtime: onnx`, and the ledger settings below.
3. **POS timing.** If the POS exports in batches (every N seconds/minutes), set `ledger.exit_grace_s` to the batch
   interval, keep `late_receipt_window_s: 300`, and raise `payment_expiry_s` past 300 s if batches are long; otherwise receipts arrive after people leave and paying
   customers become would-be alerts (simulator: 2.5 false alerts/hour with 60 s batches and no retraction,
   about 0.45 with the grace period plus retraction; `results/late_receipts.json`).
4. **Clock offset.** Ring up a test sale and note the edge box's `date +%s` when the receipt prints. Set the
   mapping's `offset_s` to (box time) minus (printed time).
5. Start: `.venv/bin/python -m bree.cli shadow --config configs/store1_shadow.yaml` (run it under a service
   manager so it restarts on reboot). Raw recording stays **off** unless needed; raw segments contain faces.
6. Walk the store yourself: pick something, pay, walk out. Within `exit_grace_s` plus a few seconds of the exit,
   nothing should be logged for you (paid). Then pick something and walk out without paying (staff aware): a would-be alert should
   appear in `would_be_alerts.jsonl` and on the review page.

## 3. Daily, during the pilot
- Open the review page on the box (SSH tunnel: `ssh -L 8080:127.0.0.1:8080 box`, then http://127.0.0.1:8080/review).
- For each would-be alert, watch the clip (heads pixelated) and mark **Real theft**, **False alert** or **Unsure**,
  with a note. Retracted alerts (a late receipt cleared them) are shown as such.
- `bree shadow-labels --config ... --summary` prints counts and precision so far; `--export` writes the labelled
  data for training and for re-measuring the vision error rates.
- Note anything odd (camera moved, lighting change, POS outage) with the date; it explains bad days later.

## 4. What to measure before anything goes live
- **False alerts per hour** from the labels (target well under 0.1 per hour before any staff-facing alert).
- **Precision by evidence:** alerts with a concealment seen vs without; the first live mode, if any, should be
  "alert only with concealment seen", everything else to a manager review queue.
- **Missed thefts:** ask the operator for known shrink incidents in the period and check whether BREE flagged them.
- **Per-camera tracking:** how often one shopper becomes several visits (`engine_log.txt`: "stitched" and "track lost
  inside store" lines).

## 5. Privacy and data
- No face storage, no identity, no re-identification across visits. Evidence clips are written only for flagged
  people, with heads pixelated; everything else stays in memory.
- Get the operator's written OK for the pilot and post whatever signage local law requires for video analytics.
- Keep labelled clips only as long as needed for the pilot (suggested 30 days), on the box or our encrypted storage.
