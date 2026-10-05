## 2026-10-05: fixed benchmark and baseline (SIMULATED)

Everything here is simulated: 12 clips from the browser store simulator copy on the recommended-3d-45 layout.
Nothing is real footage. Command: `.venv/bin/python -m bree.sim.bench dev --runner bree.sim.bench:run_pipeline --name baseline`,
result in `results/bench_dev_baseline.md` and `.json`. The TEST split was not run except a 60 frame smoke check on
one clip; no TEST number is recorded.

DEV split: 6 clips (seeds 7001 to 7006), 421 s of simulated time, 37 shoppers (15 thieves, 22 honest), 70 true
picks (41 paid, 20 concealed, 9 put back; 46 at gondolas, 15 in the cooler, 9 at the counter), 18 to 20 cameras
per clip.

Baseline: the per-camera engine as committed at 19f51dc (sim_sku detector, camera nodes and hub, closed-world
identity, two-view slots).

| Metric | Baseline, DEV |
|---|---|
| Theft recall, alert tier | 0 of 20 stolen items |
| Theft recall, alert or review | 0 of 20 |
| Alerts, reviews | 0, 0 (so alert precision is undefined, and no false alert on the 22 honest shoppers) |
| Pick recall | 2 of 70 (2.9 percent) |
| Pick precision | 2 of 2 PICK events |
| Right SKU | 1 of the 2 paired picks, 1 of 70 true picks |
| Right slot | 1 of 2 paired picks |
| Time to alert | none (no alert) |
| Store-wide identities per real shopper | 4.0 (148 identities for 37 shoppers; 93 of them cover two shoppers; 1 shopper never tracked) |

Funnel, all 70 true picks, counted at the first stage each one fails:

| stage | reached | lost here | passed on its own |
|---|---|---|---|
| in view of a rendered item camera | 70 | 0 | 70 |
| frame reached the pipeline | 70 | 0 | 70 |
| hand or item detected | 70 | 21 | 49 |
| shelf event emitted | 49 | 47 | 2 |
| right slot | 2 | 1 | 1 |
| right SKU | 1 | 1 | 1 |
| associated to a shopper | 0 | 0 | 2 |
| right shopper | 0 | 0 | 2 |
| conceal or pay classified | 0 | 0 | 1 |
| ledger basket | 0 | 0 | 50 |
| alert | 0 | 0 | 50 |

Reading: the frames arrive and in 49 of 70 picks something is detected on the hand or the item (item box on 46,
right SKU on 44, person box on the picker on 41, wrist near the hand on 32), but only 2 PICK events come out. The
loss is at the shelf event stage (47 picks), then at detection (21, of which 20 at gondolas). The 50 in the last
two rows are paid and put-back picks for which no alert is the right answer; with zero alerts that says nothing.
None of the 20 concealed picks produced a PICK event.

Pipeline wall time per clip, on a loaded 10 core laptop, 3 clips at a time: 11 to 20 minutes (camera nodes and
hub 200 to 520 s, pipeline 480 to 815 s).
