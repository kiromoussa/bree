# End-to-end chain on SIMULATED data

SIMULATED (browser store simulator copy, scripts/train/render_synth.mjs --clip). Not real footage.

Clip: 530 frames at 10 fps, cameras REGISTER-top, TRACK-2, TRACK-3, G4L-rail-1, G2L-rail-4, G2L-rail-3, G1R-rail-3, G3R-rail-1. Product weights: /Users/kiromoussa/bree-vision/data/synth/weights/sim_sku.pt. Person confidence threshold 0.15. Wall time 732.8 s.

## Edge: camera nodes and hub

| camera | kind | path | frames | triggers | bursts | frames at hub | share sent | burst MB | est. full stream MB |
|---|---|---|---|---|---|---|---|---|---|
| G1R-rail-3 | shelf | node -> hub | 530 | 2 | 4 | 197 | 0.372 | 89.9 | 241.7 |
| G2L-rail-3 | shelf | node -> hub | 530 | 3 | 5 | 234 | 0.442 | 110.3 | 249.9 |
| G2L-rail-4 | shelf | node -> hub | 530 | 4 | 5 | 221 | 0.417 | 95.9 | 230.0 |
| G3R-rail-1 | shelf | node -> hub | 530 | 4 | 6 | 281 | 0.53 | 154.0 | 290.4 |
| G4L-rail-1 | shelf | node -> hub | 530 | 3 | 5 | 231 | 0.436 | 112.5 | 258.1 |
| REGISTER-top | checkout | continuous stream | 530 |  |  |  |  |  |  |
| TRACK-2 | overhead | continuous stream | 530 |  |  |  |  |  |  |
| TRACK-3 | overhead | continuous stream | 530 |  |  |  |  |  |  |

Node cameras: 5. Frames sent to the hub: 1164 of 2650 (0.439). Hub errors: 0.

## Scorecard (make sim-eval scorer, through the whole chain)

SIMULATED (browser store simulator copy, scripts/train/render_synth.mjs --clip). Not real footage. Backend `sim_sku`, 8 camera(s), 53 s of sim time, 7 picks (2 concealed, 5 paid), 4 receipt(s) (0 dropped, 0 mis-rung).

| Metric | Value |
|---|---|
| Theft recall (alert tier) | 0.0% (0/2) |
| Theft recall (alert or review) | 0.0% |
| Alert precision | n/a (0/0) |
| SKU-correct rate (caught thefts) | n/a |
| Time to alert after concealment | n/a |
| False alerts per hour | 0.0 (0 in 53 s) |
| Review-tier alerts | 0 |

## By zone kind

| | concealed | caught | recall | alerts | false | precision | SKU correct |
|---|---|---|---|---|---|---|---|
| shelf | 2 | 0 | 0.0% | 0 | 0 | n/a | n/a |

## By zone (fixture)

| | concealed | caught | recall | alerts | false | precision | SKU correct |
|---|---|---|---|---|---|---|---|
| G2 | 1 | 0 | 0.0% | 0 | 0 | n/a | n/a |
| G3 | 1 | 0 | 0.0% | 0 | 0 | n/a | n/a |

## By camera kind

| | concealed | caught | recall | alerts | false | precision | SKU correct |
|---|---|---|---|---|---|---|---|
| overhead | 1 | 0 | 0.0% | 0 | 0 | n/a | n/a |
| shelf | 2 | 0 | 0.0% | 0 | 0 | n/a | n/a |

## Missed thefts

- P002 cocoa_crest at G2 (t=10.2 s, seen by overhead, shelf)
- P004 cocoa_crest at G3 (t=19.87 s, seen by shelf)

## Closed-world identity

| pool | assigned_lost | births_entrance | births_missed_entry | births_warmup | exits | handoffs_live | occupancy | uncertain_marks |
|---|---|---|---|---|---|---|---|---|
| store | 16 | 4 | 12 | 1 | 1 | 32 | 10 | 40 |

## 3D slot of each pick

No PICK event carried a slot.

## Product recognition

| | |
|---|---|
| frames | 2754 |
| product_boxes | 7333 |
| product_boxes_per_frame | 2.66 |
| pipeline_pick_events | 0 |
| ground_truth_picks | 7 |
| picks_matched | 0 |
| picks_sku_right | 0 |
| pick_recall | 0.0 |
| sku_right_of_matched | None |

## Where a pick is lost (ground-truth boxes of items in a hand, per camera kind)

| camera kind | item boxes | frame reached pipeline | person box in frame | product box on item | right sku | item inside a shelf zone of this camera |
|---|---|---|---|---|---|---|
| overhead | 602 | 602 | 602 | 2 | 0 | 0 |
| shelf | 296 | 296 | 79 | 83 | 83 | 203 |
| checkout | 1 | 1 | 1 | 0 | 0 | 0 |

## Review store and owner report

Reviewer: simulator ground truth (not a person). Ingested 0 alert(s) from 0 record(s); decisions {'confirmed_theft': 0, 'wrong_item': 0, 'not_theft': 0}.

# BREE weekly report: 2026-10-05 to 2026-10-11

- Alerts sent: **0** (0 from staged tests, 0 withdrawn after a late receipt, 0 from customers)
- Confirmed theft: **0**
- Rejected (not theft): **0**
- Unclear: 0. Reviewers disagreed: 0. Not reviewed yet: 0.
- No customer alert has a clear decision yet, so there is no real theft share to report.
- No customer alerts this week.

## Staged test thefts

Staged: 0. Caught: **0**. Missed: **0**.

## By zone (customer alerts)

| | Sent | Confirmed | Rejected | Real theft share |
|---|---|---|---|---|

## By camera (customer alerts)

| | Sent | Confirmed | Rejected | Real theft share |
|---|---|---|---|---|

