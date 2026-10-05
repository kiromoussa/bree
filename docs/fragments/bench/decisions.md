## 2026-10-05: fixed benchmark on simulated clips (benchmark stream)

- **Layout: recommended-3d-45.** Its files are in `~/bree/software/shared/layouts`, so it is used instead of
  recommended-47: 45 cameras, two-view coverage, one entrance camera.
- **The private simulator copy `out/sim-copy` was refreshed** from the simulator at commit bff1f32 (the old copy
  had no entrance camera kind and no 3D layouts). Renders of training seeds made after this come from the newer
  simulator. `clip.json` records the copy's commit and the hashes of the two overlay files.
- **Scenario overlay, not simulator edits.** `scripts/bench/sim/bench.js` is loaded into the copy on top of
  `scripts/train/sim/synth.js`. It scripts each shopper (zone of each pick, look first, put back, walk before
  concealing, a second shopper at the same shelf) from its own seeded generator. The simulator's own shopper has
  no put-back and picks slots at random; that was not enough for the task.
- **Splits by seed.** Train 1000 to 4999 (the SKU detector's scenes 1000 to 1239 are inside it), scratch 5000
  to 5999 (seed 5001 was the first end to end clip and was looked at during development, so it is in neither
  dev nor test), dev 7001 to 7006, test 9001 to 9006. A test in `scripts/bench/test_bench.py` fails if they overlap.
- **Video, not PNG frames.** A clip is one H.264 file per camera (libx264 crf 16 from JPEG quality 0.92) at the
  camera's full resolution. PNG frames of the first clip took 12 GB for 8 cameras and 53 s; a benchmark clip has
  16 to 20 cameras. A real camera delivers H.264 too. The adapter uses the files as they are (no second encode;
  the old path re-encoded PNG to mp4v).
- **Which cameras are rendered.** Always the entrance camera, the four overhead cameras and the register camera.
  The item camera with the best view of each pick is never dropped. Second views are added until there are 12
  item cameras, then 2 item cameras that see no pick. Not every camera that sees a pick is rendered: a pick is
  usually in view of 3 to 5 cameras and rendering all of them would double the frame count.
- **Clip length.** The arrival window is the first of 30, 22, 38, 16, 46, 10, 4 s that makes the visit end
  between 60 and 90 s; if none does, the one closest to 75 s (seed 9003 ends at 59 s, seed 9004 at 94 s).
- **Calibration is the true pose.** `calibration.json` has the pose the scene was rendered with (layout pose plus
  the mounting error of synth.js, roll up to 0.02 rad included). That is what a calibrated install knows. The
  old run used the layout pose without roll; on a quick slice that was off by a median of 5 px and up to 71 px
  at the feet of a shopper, the calibration file by at most 2 px.
- **The planogram is exact.** `layout.json` gives each slot the SKU that is in it, including the single
  misplaced items synth.js puts in (12 percent of slots). A real planogram would not know those. Known ceiling:
  a pipeline that reads the SKU from the slot alone is scored as if misplaced items did not exist.
- **Register feed.** One receipt per paying shopper, 1.5 to 4.5 s after the payment, no dropped or mis-rung
  receipts. Receipts carry no shopper identity.
- **Ground truth cannot reach the pipeline by accident.** Truth lives in `truth/` inside the clip; the benchmark
  hands the runner a folder of links without it. The scorer is the only reader.
- **Identity for scoring comes from boxes.** A pipeline identity is matched to a true shopper where its boxes
  overlap the true person box (IoU 0.3), near the time in question. The runner logs track id to store-wide id
  per frame (`person_ids.jsonl`) by subclassing the identity layer; the frame log itself does not carry it.
- **Funnel stage "right slot" is the exact slot id.** A pick with the right SKU but no slot is counted lost at
  "right slot"; the "passed on its own" column shows the SKU stage separately.
- **A theft counts as caught at shopper level**: an alert on the thief after the concealment. "Alerted with the
  right SKU" is reported next to it.
