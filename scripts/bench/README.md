# Fixed benchmark (SIMULATED clips)

Everything here is simulated: the clips come from a private copy of the browser store simulator
(`out/sim-copy`, refreshed from `~/bree/software/sim-prototype`). Nothing here is real footage.

## Splits

`manifest.json` fixes the splits by scenario seed. They never overlap.

| split | seeds | use |
|---|---|---|
| train | 1000 to 4999 | training data only (1000 to 1239 are the scenes the sim_sku detector was trained on) |
| scratch | 5000 to 5999 | not part of the benchmark (5001 is the clip of the first end to end run) |
| dev | 7001 to 7006 | generator 1. Was the tuning set of rounds 0 to 5; now a regression set |
| test | 9001 to 9006 | generator 1. Was run once as the held-out set; now a regression set |
| dev2 | 11001 to 11020 | generator 2. The tuning set from 2026-10-05 on |
| checkpoint | 13001 to 13030 | generator 2. Held-out pool, rendered 6 seeds at a time; a batch is scored once |

`clips.json` lists, per rendered clip, the cameras, frame count, what happens and the render time.

## Generator 2 (dev2, checkpoint): bigger, harder, fairer

Generator 1 clips had an exact planogram, cameras picked by which camera saw a pick, a clean pinhole picture and
only shoppers who pick, pay, put back or conceal. Generator 2 (`render_clip.mjs --gen 2`, set per split in
`manifest.json`) changes that. Generator 1 is untouched: a dry run of TRAIN seed 4903 still gives the stored truth.

Every clip draws each scenario option from its seed (`FEATURES` in `sim/bench.js`), so a clip has a mix:

| option | chance | what happens |
|---|---|---|
| group | 0.5 | two shoppers enter together; one carries the items to the counter and puts them down, the other pays. In 3 of 10 groups the carrier also pockets something. Half of the carriers walk out before the payment is done |
| staff | 0.45 | a vendor in a work vest or an employee in the clerk's shirt comes in, puts new items into empty places, pulls items and either moves them to another slot or carries them out. None of this is theft |
| wrongSlot | 0.45 | a put-back goes into an empty place of another slot on the same shelf |
| shift | 0.5 | one or two shoppers reach in, shift an item a few centimetres and take nothing |
| bump | 0.3 | two cameras of the clip are knocked 1.5 to 3.5 degrees partway through; `calibration.json` keeps the old pose |
| block | 0.3 | one item camera is covered for 10 to 25 s |
| dropScan | 0.35 | one paid item never reaches `register.jsonl` (the shopper paid for it) |
| night | 0.25 | the simulator's night lighting |

Always on:

- **The camera pass of the simulator** at strength 0.5 of each effect: lens distortion and vignette, auto exposure
  and gain drift, motion blur (two exposures over 1/60 s by day, three over 1/30 s at night), sensor noise, codec
  blocks, white balance, glare on the cooler glass; then H.264 at crf 23. `calibration.json` gives the lens of each
  camera (`k_div`, and `dist` as OpenCV rational coefficients) and says how to undo it. A pipeline that treats the
  picture as a pinhole image is off by up to about 90 px at the edge of a 120 degree overhead camera.
- **Cameras by aisle.** A clip's store has cameras at the counter, in the cooler aisle (6 of 10 clips) and in two or
  three of the five gondola aisles, all drawn from the seed. Shoppers pick only in those aisles. Every people camera
  is rendered, and every item camera whose main view (the aisle it sees most slots of) is one of those aisles: 20 to
  23 item cameras in the clips so far, most of them watching nothing at any moment. No camera is chosen by what it saw.
- **A nominal planogram.** `layout.json` names the product each slot is meant to hold (the product of its
  planogram block). The single misplaced items (about 6 percent of slots) are only in `truth/planogram.json`, and
  the shelf changes during the clip (wrong-slot put-backs, staff). Right SKU scored on these clips is right SKU
  with a nominal planogram.
- **Visits of 40 to 70 s** with 4 to 7 shoppers, plus the companion and the staff member when the options add them.

New truth files: `truth/acts.jsonl` (touches, staff takes and puts), `truth/planogram.json`, `truth/faults.json` (the
options of the clip, which cameras were knocked or covered and when, the dropped scan). `clip.json` no longer holds
scene parameters: they moved to `truth/render.json`. `truth/events.jsonl` rows gain `group`, `paidBy`, `tHand`,
`putBackSlot`, `scanDropped`; `truth/shoppers.json` rows gain `role` (shopper or staff) and `group`.

A camera with nothing moving in view still repeats its last frame, so its sensor noise stands still then. That keeps
the render affordable; it means a trigger that fires on noise alone is not exercised on an empty view.

## The checkpoint pool

30 seeds nobody has rendered. One command takes the next 6 unused seeds, marks them used in `manifest.json` (before
rendering, so a seed is never handed out twice), renders them, and the benchmark scores the latest batch:

```
.venv/bin/python scripts/bench/render_split.py checkpoint --next 6 --jobs 3     # batch 1 is 13001 to 13006, and so on
.venv/bin/python -m bree.sim.bench checkpoint --stress                          # -> results/bench_checkpoint_1.json and .md
.venv/bin/python scripts/bench/render_split.py checkpoint                       # only to finish a batch whose render was interrupted
```

Rules: tune on dev2 only; run a batch once, on a commit made before the batch was rendered; do not open the frames
or the truth of a batch to fix the pipeline (that turns the batch into tuning data: say so and take the next one).
Commit `manifest.json` after `--next`.

Generator 2 render cost is in `clips.json` (`render_seconds`, `frames_drawn`) and in the report of the day it was rendered.
Generator 1: rendering takes about 9 to 16 minutes per clip with three clips at a time on a 10 core laptop with nothing else
running hard (16 to 20 cameras, 600 to 950 frames), and 35 to 47 minutes per clip when the machine was busy with
other jobs (dev 7001 to 7005). The 12 clips take 15 GB. A camera with nothing moving in view repeats its last
frame instead of rendering it again (checked byte identical on a slice). If Chrome closes mid clip the split
renderer starts that clip again, up to 3 times.

## The dev2 baseline (committed pipeline, its own folder)

The baseline of the committed pipeline on dev2 is kept apart from tuning runs, because tuning runs of other streams
write to `out/bench/dev2`. It runs in a worktree of the committed code (`git worktree add out/wt-baseline <commit>`,
with `models` linked in) and writes to `out/bench/dev2_baseline` and
`out/bench/dev2_baseline_stress`. The scorecard is `results/bench_dev2_baseline.md` and `.json`; its first line says
how many clips it covers.

```
nohup scripts/bench/queue/dev2_baseline.sh plain  > out/bench/dev2_baseline_plain.log 2>&1 &    # one clip at a time, as each clip is rendered
nohup scripts/bench/queue/dev2_baseline.sh stress > out/bench/dev2_baseline_stress.log 2>&1 &   # the stressed run of each clip whose plain run is done
nohup scripts/bench/queue/dev2_baseline_score.sh  > out/bench/dev2_baseline_score.log 2>&1 &    # scores again after every finished run
cd out/wt-baseline && PYTHONPATH=$PWD/src ../../.venv/bin/python -m bree.sim.bench dev2 --score-only --stress --name baseline --out ../../out/bench/dev2_baseline   # score by hand
```

The render and the queues stop while the laptop sleeps: `caffeinate -i -w <pid of a queue>` keeps it awake for as long
as that queue runs. The queue scripts hold the paths of this machine; read them before
running them again. `--score-only` scores the clips that are rendered and have a finished run, and leaves `results/`
alone until the whole split is there (the score loop copies the partial scorecard to `results/` itself).

### Result of the baseline (SIMULATED, commit b551623, scored 2026-10-06)

All 20 clips: 120 shoppers (44 thieves, 76 honest), 7 staff, 230 picks, 55 stolen items. Square brackets are the 95
percent bootstrap interval. The full scorecard, funnel and per clip table are in `results/bench_dev2_baseline.md`.

| | plain, 20 clips | plain, the 18 clips also stressed | stressed, 18 clips |
|---|---|---|---|
| Thefts flagged, alert tier | 0 of 55 | 0 of 50 | 0 of 50 |
| Thefts flagged, alert or review | 33 of 55 (60.0%) [0.44 to 0.74] | 29 of 50 (58.0%) [0.41 to 0.74] | 15 of 50 (30.0%) [0.16 to 0.46] |
| Honest shoppers flagged for review | 16 of 76 (21.1%) [0.13 to 0.30] | 15 of 68 (22.1%) [0.13 to 0.32] | 7 of 68 (10.3%) [0.03 to 0.18] |
| Staff members flagged | 4 of 7 | 4 of 7 | 4 of 7 |
| Staff takes listed as unpaid | 3 of 6 | 3 of 6 | 5 of 6 |
| Shifted items counted as a pick | 8 of 16 | 8 of 16 | 4 of 16 |
| Picks found | 204 of 230 (88.7%) [0.85 to 0.93] | 185 of 210 (88.1%) | 155 of 210 (73.8%) [0.68 to 0.80] |
| Right slot, of paired picks | 66.7% [0.60 to 0.73] | 67.0% | 44.5% [0.36 to 0.53] |
| Right SKU, nominal planogram | 75.5% [0.69 to 0.81] | 75.7% | 66.5% [0.58 to 0.74] |
| Put-back recall | 17 of 34 (50.0%) [0.33 to 0.67] | 16 of 32 | 18 of 32 (56.2%) |
| Put-back precision | 41.8% of 55 events [0.31 to 0.55] | 44.9% of 49 | 46.2% of 52 |
| Identities per person | 1.551 [1.46 to 1.65] | 1.539 | 1.878 [1.75 to 2.02] |
| Identities that cover two people | 84 of 191 (44.0%) [0.36 to 0.51] | 76 of 173 (43.9%) | 128 of 212 (60.4%) |

The stressed column has 18 clips because **the committed pipeline crashed on the stressed inputs of 11004 and 11007**
(`src/bree/track/floor.py` line 478, `self.tracks.remove(q)`: the track dataclass compares numpy arrays with `==`,
`ValueError: The truth value of an array with more than one element is ambiguous`). The logs are
`out/bench/dev2_baseline_stress/clip_11004.log` and `clip_11007.log`. A crash is worse than a miss: read the stressed
column as an upper bound until that is fixed. `--score-only` leaves a clip without a finished run out and says how many
clips it scored; the plain command (`bench dev2 --stress`) refuses to score when a clip fails.

Fewer honest shoppers are flagged under stress only because the stressed pipeline sees fewer picks at all.

Render cost of dev2 (in `clips.json`): 32.9 minutes of wall time per clip on average (11.1 to 73.7) with three clips at
a time on a 10 core laptop that other jobs were using; the clips rendered with the least competition took 11 to 19
minutes. 25.2 cameras and 639 frames per clip on average, 11129 camera frames drawn per clip (the rest repeated
because nothing moved), 42 GB for the 20 clips. A pipeline run of one clip took 5 to 6 minutes with two or three runs
at a time and nothing else heavy running, and over an hour when three renders and three runs shared the machine: run
the render first, or one pipeline run beside it.

## Commands

```
export BREE_PLAYWRIGHT=/path/to/node_modules/playwright     # any Playwright install works (the renderer drives the installed Chrome). The install in /private/tmp is gone after a restart
.venv/bin/python scripts/bench/render_split.py dev            # render (skips finished clips); test likewise
.venv/bin/python scripts/bench/render_split.py train --seeds 1240:1250     # training clips with the same truth files
node scripts/bench/render_clip.mjs --seed 7001 --out /tmp/x --dry          # scenario and cameras only
node scripts/bench/render_clip.mjs --seed 1240 --out /tmp/q --from 10 --to 16 --stills 10   # a quick slice with stills

.venv/bin/python scripts/bench/render_split.py dev2 --jobs 3               # the 20 dev2 clips (generator 2)
node scripts/bench/render_clip.mjs --seed 5101 --gen 2 --out /tmp/x --dry                       # generator 2: what a seed holds
node scripts/bench/render_clip.mjs --seed 5101 --gen 2 --out /tmp/q --features staff,-night --from 20 --to 25 --stills 10
node scripts/bench/render_clip.mjs --seed 5101 --gen 2 --out /tmp/c --realism off --aisles all  # clean picture, every camera

.venv/bin/python -m bree.sim.bench dev2 --stress            # the tuning set, plain and stressed -> results/bench_dev2.json and .md
.venv/bin/python -m bree.sim.bench dev2 --score-only --stress     # score the finished runs again
.venv/bin/python -m bree.sim.bench dev2 --drop-item-cameras 0.2 --stress-only    # one stress at a time, plain runs kept
.venv/bin/python -m bree.sim.bench checkpoint --stress      # the latest checkpoint batch (--batch K for an earlier one)
.venv/bin/python -m bree.sim.bench dev            # make bench-dev  -> results/bench_dev.json and .md
.venv/bin/python -m bree.sim.bench test           # make bench-test -> results/bench_test.json and .md
.venv/bin/python -m bree.sim.bench dev --smoke    # one clip, 60 frames per camera, results/ untouched
.venv/bin/python -m bree.sim.bench dev --score-only        # rescore the runs in out/bench/dev
.venv/bin/python -m bree.sim.bench dev --clips 7001 --name mytry --runner mymodule:run   # partial or named runs
.venv/bin/python -m bree.sim.bench dev --runner bree.sim.bench:run_pipeline --name baseline   # the recorded baseline: results/bench_dev_baseline.*
.venv/bin/python -m pytest scripts/bench/test_bench.py -q
```

## A clip

Each clip is one seeded visit of 4 to 8 shoppers, about 60 to 90 simulated seconds at 10 frames a second, on the
`recommended-3d-45` layout. Honest shoppers pick 1 to 3 items, sometimes look at the shelf first, sometimes hold
an item and put it back, then pay at the register. Thieves pick, conceal the item in a front pocket (on the spot
or after walking to another aisle), may also buy something, and leave. Picks happen at gondolas, in the cooler
and at the counter. In every clip one shopper goes to the shelf another shopper is at.

Cameras rendered per clip (the exact list is in `clip.json` and `clips.json`): the entrance camera, the four
overhead cameras and the register camera, always; the item camera with the best view of each pick; second
views of picks until there are 12 item cameras; and 2 item cameras that see no pick.

```
clip_7001/
  <camera>.mp4       H.264 at the camera's full resolution, 10 fps, frame i is time i / 10 s
  calibration.json   per camera: fx, fy, cx, cy, position, R (rows: right, down, forward), roll. True pose.
  layout.json        store, fixtures, skus, slots (id, skuId, fixtureId, position, size, face, normal, zone):
                     the planogram of this scene; cameras: all 45 of the layout with their true pose
  register.jsonl     receipts: {t, terminal, txn_id, items: [{sku, qty}]}, 1.5 to 4.5 s after the payment
  clip.json          seed, fps, frames, cameras and how they were chosen, scene parameters, simulator version
  truth/             GROUND TRUTH. Scoring only (and training data, for train seeds only).
    events.jsonl     one row per picked item: t, shopper, skuId, slotId, fixtureId, zone, outcome
                     (concealed, paid, put_back), tConceal, tPay, tPutBack, tExit, cameras that saw it
    shoppers.json    shopper, thief, tEnter, tExit, height, task list
    tracks.jsonl     per frame: floor position (x, z), heading, state and hand position of every shopper
    frames.jsonl     per camera and frame: persons [{shopper, bbox, vis_px, hand, head, feet, state}] and
                     items in a hand or on the counter [{sku, kind, bbox, vis_px, shopper, slot}]
```

Boxes in `frames.jsonl` come from an id render: a box covers only the pixels that are visible. `hand`, `head`
and `feet` are projected points and can fall outside the image. `shopper` is `clerk` for the clerk.

Reading a clip in Python:

```python
from bree.sim.bench import load_calibration      # {camera id: bree.calib.camera.Camera}
cams = load_calibration(clip)                    # cams[id].project(points), .ray(u, v), .floor_homography()
cap = cv2.VideoCapture(str(clip / "G3R-price-rail-1.mp4"))
```

Store frame: metres, y up, floor at y = 0, +z toward the front door (the layout frame). The planogram is exact
in generator 1 clips (the slot's SKU is what is in the slot). The camera pose is the true one, with the mounting error
and roll the scene was rendered with; `layout.json` cameras carry the same pose as yaw, pitch and roll.

## What the pipeline may use

Pixels, `calibration.json`, `layout.json` and `register.jsonl`. Never `truth/`. The benchmark command runs the
pipeline on a folder of links without `truth/`, and only the scorer opens it.

## What the scorecard reports

Alert tier and review tier apart, for thefts and for honest shoppers; staff members flagged and staff takes listed
as unpaid; shifted items counted as picks; picks found and pick precision (a PICK on a staff take is not a false
pick); right slot, right SKU (with the planogram the clip gave: nominal on generator 2) and right shopper; put-back
recall and precision (shopper put-backs and staff puts are the true puts); identities per person and the share of
identities that cover two people; the funnel; the same by scenario tag, by scenario option and by camera mount.
Headline rates carry a 95 percent bootstrap interval (shoppers drawn again 2000 times; clips for precision and
identity numbers, which have no shopper to hang on).

`--stress` runs the pipeline a second time on worse inputs and prints it as a second column: 20 percent of the item
cameras missing (drawn with `--stress-seed`, the register camera stays), every camera pose off by 0.5 degrees per
axis, receipts 5 s late, 5 percent of the planogram naming another product. Each has its own flag
(`--drop-item-cameras`, `--calib-noise-deg`, `--register-delay-s`, `--wrong-planogram`). The inputs are made worse
from the clip seed and the stress seed only; no ground truth is read. Runs go to `out/bench/<split>_stress`.

## What the scorer reads

See the docstring of `src/bree/sim/bench.py`: `events.jsonl`, `alerts.jsonl`, `frames_<camera>.jsonl` and
`person_ids.jsonl` in `<out>/pipeline`. To benchmark a changed pipeline, either keep `run_pipeline(clip, out)`
in `bree/sim/bench.py` pointing at it, or pass `--runner module:function`.
