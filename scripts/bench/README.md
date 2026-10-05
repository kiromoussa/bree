# Fixed benchmark (SIMULATED clips)

Everything here is simulated: the clips come from a private copy of the browser store simulator
(`out/sim-copy`, refreshed from `~/bree/software/sim-prototype`). Nothing here is real footage.

## Splits

`manifest.json` fixes the splits by scenario seed. They never overlap.

| split | seeds | use |
|---|---|---|
| train | 1000 to 4999 | training data only (1000 to 1239 are the scenes the sim_sku detector was trained on) |
| scratch | 5000 to 5999 | not part of the benchmark (5001 is the clip of the first end to end run) |
| dev | 7001 to 7006 | tuning and diagnosis |
| test | 9001 to 9006 | final numbers only; do not tune on it, do not look at its frames |

`clips.json` lists, per rendered clip, the cameras, frame count, what happens and the render time.

## Commands

```
export BREE_PLAYWRIGHT=/path/to/node_modules/playwright
.venv/bin/python scripts/bench/render_split.py dev            # render (skips finished clips); test likewise
.venv/bin/python scripts/bench/render_split.py train --seeds 1240:1250     # training clips with the same truth files
node scripts/bench/render_clip.mjs --seed 7001 --out /tmp/x --dry          # scenario and cameras only
node scripts/bench/render_clip.mjs --seed 1240 --out /tmp/q --from 10 --to 16 --stills 10   # a quick slice with stills

.venv/bin/python -m bree.sim.bench dev            # make bench-dev  -> results/bench_dev.json and .md
.venv/bin/python -m bree.sim.bench test           # make bench-test -> results/bench_test.json and .md
.venv/bin/python -m bree.sim.bench dev --smoke    # one clip, 60 frames per camera, results/ untouched
.venv/bin/python -m bree.sim.bench dev --score-only        # rescore the runs in out/bench/dev
.venv/bin/python -m bree.sim.bench dev --clips 7001 --name mytry --runner mymodule:run   # partial or named runs
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
in these clips (the slot's SKU is what is in the slot). The camera pose is the true one, with the mounting error
and roll the scene was rendered with; `layout.json` cameras carry the same pose as yaw, pitch and roll.

## What the pipeline may use

Pixels, `calibration.json`, `layout.json` and `register.jsonl`. Never `truth/`. The benchmark command runs the
pipeline on a folder of links without `truth/`, and only the scorer opens it.

## What the scorer reads

See the docstring of `src/bree/sim/bench.py`: `events.jsonl`, `alerts.jsonl`, `frames_<camera>.jsonl` and
`person_ids.jsonl` in `<out>/pipeline`. To benchmark a changed pipeline, either keep `run_pipeline(clip, out)`
in `bree/sim/bench.py` pointing at it, or pass `--runner module:function`.
