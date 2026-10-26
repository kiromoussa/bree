# Store intelligence: people flow (`bree intel`)

Track: intel. Code: `src/bree/intel/`, layouts: `configs/intel_*.yaml`, tests: `tests/test_intel_flow.py`.
Label file for the door camera: `docs/ultra/intel_door_labels.json` (times and directions only, no descriptions of people).

## Evidence labels used here

- SIMULATED: unit tests with synthetic walkers.
- REAL-UNLABELLED: ran on real Truman Rd. footage, no truth. Every dwell, queue, staff, heatmap and outcome number is this.
- REAL-MODEL-LABELLED: an agent viewed frames and wrote labels. Not ground truth. Door entries and exits (ch07) and people-in-view counts (ch29, ch12, from other tracks' label files) are this.
- REAL-GROUND-TRUTH: none exists.

The real footage is about 20 minutes per channel from one afternoon (about 18:28 to 18:56 EDT on 2026-10-25), channels 7, 10, 12, 14, 15, 20, 29. Anything per hour is an extrapolation and is reported per observed hour with its coverage.

## What it does

`bree intel run` reads recorded segments, caches person detections (YOLO at 960, stride 5, about 6 fps), tracks feet points, and writes `flow.jsonl` and `heat.npz`. `bree intel report` turns that into hourly and daily aggregates, a digest text, heatmaps and a schematic plan view. `bree intel eval` scores door events against labels, `bree intel counts` scores people-in-view counts, `bree intel anomalies` flags rules and hour-of-week outliers, `bree intel activity` is a labelling helper (frame-difference windows at the door, detector independent).

No face, no identity, no appearance model. A track id is one walk through one camera's view. A person who leaves view and returns gets a new id, so "repeat visitor" is not claimed; "repeat pattern" means the same hour-of-week slot repeating.

### Metrics

| Metric | How | Status |
|---|---|---|
| Entries / exits, live occupancy (ch07) | feet cross the door line in its span; weaker "born on mats" and "lost on mats" evidence flagged in `via`; occupancy = initial + entries - exits (floored at 0), `customers` in view is a lower bound | door events REAL-MODEL-LABELLED, occupancy REAL-UNLABELLED |
| Zone dwell | per zone visit, gap-tolerant (`zone_gap_s`), plus `still_s` | REAL-UNLABELLED |
| Register queue, time at register | customers inside register zone, still for >= 2 s counts as a register visit | REAL-UNLABELLED |
| Staff vs customer | tracks that spend most time in the behind-counter zone are staff; they never count as occupancy, queue or visits | REAL-UNLABELLED, weak (see negative results) |
| Busiest hour | entries per observed hour | REAL-UNLABELLED, one hour exists |
| Heatmap, plan view | foot points accumulated, homography to a schematic plan (units 0-100, not metres) | REAL-UNLABELLED, layout drawn by eye |
| Conversion proxies | per customer track: registered / left without register / browsed, not seen leaving / seen briefly / passed through | REAL-UNLABELLED, camera-local proxies |
| Anomalies | rules (long queue, loitering, door imbalance; unattended counter OFF) and hour-of-week robust z | SIMULATED only: no real anomaly day exists |
| Door-less cameras (ch12, ch14, ch15, ch29 ...) | `configs/intel_aisle_view.yaml`: one whole-view zone, traffic, dwell in view, heatmap, no door events; `--learn-mask` ignores low-confidence boxes that sit still all segment | counts vs labels below |

### Record schema (`flow.jsonl`, one JSON object per line)

- `segment`: `{video, t0, t1, ts, entries, exits, initial, camera}`; starts each segment's records.
- `door`: `{dir: in|out, t, ts, track, via: line|born|lost, camera}`.
- `zone`: `{track, zone, kind, t_in, t_out, dwell_s, still_s, camera}` written when a visit ends.
- `sample`: `{t, ts, in_view, customers, staff, queue, occupancy, camera}` every 5 s (occupancy is null until it can be anchored).
- `track`: `{track, role: customer|staff, t0, t1, ts, obs, path_px, entered, exited, browse_s, register_s, register_still_s, outcome, camera}` when a track closes.

`ts` is station time: file-name time plus the layout's `clock.file_name_offset_s` (-7200 here, matching the burned-in DVR overlay; the store's real time zone is unverified, settle it with a POS timestamp before joining receipts). Aggregates (`hourly.json`, `daily.json`) carry `coverage_s` and per-observed-hour rates; nothing is extrapolated to missing time. `bree intel report` writes one folder per camera when given several cameras' records.

## Door counting on ch07: REAL-MODEL-LABELLED

Method. An agent viewed 1 Hz crops of the door area as contact sheets, for windows chosen by a detector-independent frame-difference trigger (door and mats ROI, mean absolute grey diff >= 2.5 at 2 Hz, +/- 5 s). Each person crossing the threshold was labelled with direction and time (+/- 1 s). Events outside the viewed windows are assumed absent. Match tolerance 4 s, one-to-one.

Limits. One labeller (a model), no human check, 4 segments from the same 23 minutes, 27 labelled crossings in all (dev 11, dev2 9, heldout 7). Windows came from motion, so a crossing that did not change the ROI could be missing from the labels. Splits: `dev` (182851, 183357) was used for tuning; `dev2` (184620) was looked at after a v0 run showed a false out/in pair, so it is not clean for v1; `heldout` (185128) is clean and was scored once at the end.

Results (tolerance 4 s):

| split | labelled in/out | system in/out | TP | FP | missed | mean abs time error |
|---|---|---|---|---|---|---|
| dev | 5 / 6 | 5 / 6 | 11 | 0 | 0 | 0.83 s |
| dev2 (not clean) | 3 / 6 | 3 / 6 | 9 | 0 | 0 | 0.91 s |
| heldout (clean) | 4 / 3 | 3 / 2 | 5 | 0 | 2 | 0.57 s |

Heldout recall 5 of 7 (0.71) with 0 false positives. The two misses are one person who stood on the sill and stepped back out and in about 10 s later (a labeller-ambiguous case; the system saw neither crossing). The first version (v0) scored 8 TP, 2 FP, 1 miss on dev2; the later versions (v1, v2) fixed that, so the fix was tuned on a segment that was already seen (the commit history holds the changes, they were not ablated one by one). Treat all of this as a small-sample sanity check, not an accuracy claim: 7 heldout events cannot support a rate.

Whole 20 minutes: 11 entries and 14 exits counted against 12 labelled in and 15 labelled out (labelled 27, counted 25).

## People-in-view counts on door-less cameras: REAL-MODEL-LABELLED

Label files are from the perception tracks (`~/Projects/bree-data/real/labels/trumanrd_chNN_people.json`, 40 frames per segment about 7 s apart, agent-made from contact sheets, positions +/- 0.1, partial people counted). Same detector, same layout for every run, no tuning on the heldout segment.

| camera | split | frames | exact count | within one | presence agreement | system people vs labelled |
|---|---|---|---|---|---|---|
| ch29 overhead counter | dev (182853, 183358 labels) | 40 | 0.85 | 1.00 | 0.90 | 33 vs 37 |
| ch29 overhead counter | heldout (184621) | 40 | 0.475 | 0.90 | 0.75 | 26 vs 51 |

On ch29 heldout the flow engine under-counts by half, because the detector does not see people cropped to shoes or an arm at the edge. This reproduces the baseline, so the flow layer does not add error but also cannot fix the detector. `--learn-mask` changed nothing on ch29 (no ghost spots at this view).

CH12_PLACEHOLDER

## Negative results and failed ideas

- Unattended-counter rule (customers at the till, nobody behind): OFF by default. The door camera detects counter staff in under half of labelled frames, so on real footage it fired while the counter was staffed.
- Staff vs customer from the behind-counter zone is a soft prior, not a classifier. Staff who walk out onto the floor count as customers; staff glimpsed behind the glass for under a few seconds are lost. No staff numbers are validated.
- v0 door counter produced a false out/in pair on dev2 (a person lingering in the door glare); fixed in v1, at the cost of looking at that segment; which change did it was not isolated.
- Per-person claims (visited twice, repeat customer) are not made: no re-identification, by design.
- Heatmaps and the plan view use a hand-drawn four-point homography from one still. Shapes look plausible (the walking lane from door to register), but positions on the plan are not measured.
- Conversion proxies on one camera: the door camera sees the register, snack rack and perfume stand only. "Left without register" means "not seen at the register zone by this camera", and with 20 minutes it is 1 of 44 customer tracks. Not a conversion rate.
- Hour-of-week anomaly baseline needs weeks of history; there is none, so only the rules fire on real data and no real anomaly is known.

## Daily digest on real footage (REAL-UNLABELLED, 20 minutes)

11 entries and 14 exits at the door; median time at the register 16 s per visit, most people queued at once in view 2; 19 customer walks registered, 15 seen briefly, 5 passed through, 4 browsed not seen leaving, 1 left without register. Per-hour extrapolation (33 entries per observed hour) rests on 20 minutes of one afternoon.

## How to run

```
W=~/Projects/bree-wt/intel; cd $W
export PYTHONPATH=$W/src OMP_NUM_THREADS=4
PY=~/Projects/bree/.venv/bin/python
$PY -m bree.cli intel run ~/Projects/bree-data/real/trumanrd/ch07/*.mp4 --layout configs/intel_trumanrd_ch07.yaml --out OUT
$PY -m bree.cli intel report OUT/flow.jsonl --out OUT/report --layout configs/intel_trumanrd_ch07.yaml --background <video>
$PY -m bree.cli intel eval OUT/flow.jsonl --labels docs/ultra/intel_door_labels.json
$PY -m bree.cli intel counts <videos> --layout configs/intel_aisle_view.yaml --labels <people labels> --dets-dir <cached dets>
```

## Onboarding another station

A layout YAML is all a station needs: `door_line` (omit for door-less cameras), zones with kinds (`door`, `browse`, `register`, `staff`, `aisle`), `ignore` polygons, a clock offset and, for the plan view, four floor points. Drawing it from one still takes minutes, but the layout is then uncalibrated until someone checks it against footage; label a handful of door crossings with `bree intel activity` windows and score with `bree intel eval` before trusting entry counts.

## Known gaps

- No operator ground truth; no POS join (clock offset unresolved).
- Only one real hour-slice, so busiest hours and repeat patterns are placeholders.
- Door counts rest on 27 labelled crossings.
- Queue and register time are not validated against labels.
- Staff separation is geometric only.
