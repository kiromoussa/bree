# Benchmark, checkpoint batch 1 split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

6 clips (13001, 13002, 13003, 13004, 13005, 13006), 395 s of sim time, 39 shoppers (10 thieves, 29 honest), 2 staff, 75 picks, 12 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-07 02:53, commit .

## Scorecard

Square brackets: 95 percent bootstrap interval (shoppers drawn again 2000 times; clips for precision and identities). The sample is small, so read the interval before the point value.

Stressed: the same clips and pipeline with worse inputs, {"drop_item_cameras": 0.2, "seed": 1} (`python -m bree.sim.bench --help`). No ground truth is used to make them worse.

| Metric | Value | Stressed (6 clips) |
|---|---|---|
| Thefts flagged, alert tier | 0 of 12 (0.0%) [0.00 to 0.00] | 0 of 12 (0.0%) [0.00 to 0.00] |
| Thefts flagged, review tier only | 9 of 12 (75.0%) | 7 of 12 (58.3%) |
| Thefts flagged, alert or review | 9 of 12 (75.0%) [0.46 to 0.93] | 7 of 12 (58.3%) [0.20 to 0.85] |
| Stolen items listed as unpaid on a record | 9 of 12 (75.0%) | 7 of 12 (58.3%) |
| Alert precision | n/a (0/0) | n/a (0/0) |
| Honest shoppers flagged, alert or review | 5 of 29 (17.2%) [0.07 to 0.31] | 4 of 29 (13.8%) [0.03 to 0.28] |
| Honest shoppers flagged, alert tier | 0 of 29 (0.0%) [0.00 to 0.00] | 0 of 29 (0.0%) [0.00 to 0.00] |
| Honest shoppers flagged, review tier only | 5 of 29 (17.2%) | 4 of 29 (13.8%) |
| Staff members flagged, alert or review (of them alert tier) | 2 of 2 (0) | 0 of 2 (0) |
| Staff takes listed as unpaid on a record | 1 of 2 (a PICK event on 2) | 0 of 2 (a PICK event on 2) |
| Shifted items counted as a pick (of them listed as unpaid) | 0 of 2 (0) | 0 of 2 (0) |
| Picks found | 65 of 75 (86.7%) [0.79 to 0.93] | 61 of 75 (81.3%) [0.73 to 0.89] |
| Pick precision (staff takes count as true) | 73.6% of 91 PICK events [0.72 to 0.78] | 78.7% of 80 PICK events [0.75 to 0.83] |
| Right slot, of paired picks | 70.8% [0.59 to 0.82] | 72.1% [0.60 to 0.82] |
| Right SKU, of paired picks (nominal planogram given) | 73.8% [0.61 to 0.86]; of all true picks 64.0% | 77.0% [0.65 to 0.88]; of all true picks 62.7% |
| Right shopper, of paired picks | 76.9% [0.65 to 0.88] | 82.0% [0.72 to 0.91] |
| Put-backs found (recall) | 6 of 10 (60.0%) [0.27 to 1.00] | 6 of 10 (60.0%) [0.27 to 1.00] |
| Put-back precision (staff puts count as true) | 39.1% of 23 PUT_BACK events [0.11 to 0.71] | 35.0% of 20 PUT_BACK events [0.10 to 0.67] |
| Put-backs into another slot found | 3 of 7 | 3 of 7 |
| Identities per person | 1.488 (61 for 41) [1.39 to 1.60] | 1.488 (61 for 41) [1.39 to 1.60] |
| Identities that cover two people | 23 of 59 (39.0%) [0.11 to 0.61] | 23 of 59 (39.0%) [0.11 to 0.61] |
| People never tracked | 1 | 1 |
| False alerts per hour | 0.0 | 0.0 |
| Time to alert after concealment | n/a | n/a |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 75 picks, 18 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 75 | 0 | 75 |
| frame reached pipeline | 75 | 0 | 75 |
| hand or item detected | 75 | 16 | 59 |
| shelf event emitted | 59 | 4 | 65 |
| right slot | 55 | 11 | 46 |
| right sku | 44 | 3 | 48 |
| associated to a shopper | 41 | 0 | 65 |
| right shopper | 41 | 2 | 50 |
| conceal or pay classified | 39 | 13 | 57 |
| ledger basket | 26 | 2 | 63 |
| alert | 24 | 6 | 47 |

### All picks, stressed: 75 picks, 18 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 75 | 0 | 75 |
| frame reached pipeline | 75 | 0 | 75 |
| hand or item detected | 75 | 23 | 52 |
| shelf event emitted | 52 | 4 | 61 |
| right slot | 48 | 8 | 44 |
| right sku | 40 | 3 | 47 |
| associated to a shopper | 37 | 0 | 61 |
| right shopper | 37 | 1 | 50 |
| conceal or pay classified | 36 | 13 | 56 |
| ledger basket | 23 | 1 | 65 |
| alert | 22 | 4 | 49 |

### Outcome concealed: 12 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 12 | 0 | 12 |
| frame reached pipeline | 12 | 0 | 12 |
| hand or item detected | 12 | 1 | 11 |
| shelf event emitted | 11 | 0 | 12 |
| right slot | 11 | 1 | 10 |
| right sku | 10 | 2 | 8 |
| associated to a shopper | 8 | 0 | 12 |
| right shopper | 8 | 0 | 10 |
| conceal or pay classified | 8 | 8 | 0 |
| ledger basket | 0 | 0 | 9 |
| alert | 0 | 0 | 0 |

### Outcome paid: 53 picks, 17 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 53 | 0 | 53 |
| frame reached pipeline | 53 | 0 | 53 |
| hand or item detected | 53 | 14 | 39 |
| shelf event emitted | 39 | 4 | 43 |
| right slot | 35 | 9 | 28 |
| right sku | 26 | 1 | 32 |
| associated to a shopper | 25 | 0 | 43 |
| right shopper | 25 | 2 | 32 |
| conceal or pay classified | 23 | 1 | 51 |
| ledger basket | 22 | 2 | 48 |
| alert | 20 | 3 | 45 |

### Outcome put_back: 10 picks, 1 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 10 | 0 | 10 |
| frame reached pipeline | 10 | 0 | 10 |
| hand or item detected | 10 | 1 | 9 |
| shelf event emitted | 9 | 0 | 10 |
| right slot | 9 | 1 | 8 |
| right sku | 8 | 0 | 8 |
| associated to a shopper | 8 | 0 | 10 |
| right shopper | 8 | 0 | 8 |
| conceal or pay classified | 8 | 4 | 6 |
| ledger basket | 4 | 0 | 6 |
| alert | 4 | 3 | 2 |

### Zone checkout: 13 picks, 3 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 13 | 0 | 13 |
| frame reached pipeline | 13 | 0 | 13 |
| hand or item detected | 13 | 3 | 10 |
| shelf event emitted | 10 | 0 | 12 |
| right slot | 10 | 1 | 10 |
| right sku | 9 | 1 | 10 |
| associated to a shopper | 8 | 0 | 12 |
| right shopper | 8 | 1 | 9 |
| conceal or pay classified | 7 | 2 | 10 |
| ledger basket | 5 | 0 | 10 |
| alert | 5 | 2 | 8 |

### Zone cooler: 13 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 13 | 0 | 13 |
| frame reached pipeline | 13 | 0 | 13 |
| hand or item detected | 13 | 8 | 5 |
| shelf event emitted | 5 | 1 | 9 |
| right slot | 4 | 3 | 2 |
| right sku | 1 | 0 | 4 |
| associated to a shopper | 1 | 0 | 9 |
| right shopper | 1 | 0 | 5 |
| conceal or pay classified | 1 | 1 | 11 |
| ledger basket | 0 | 0 | 12 |
| alert | 0 | 0 | 9 |

### Zone gondola: 49 picks, 15 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 49 | 0 | 49 |
| frame reached pipeline | 49 | 0 | 49 |
| hand or item detected | 49 | 5 | 44 |
| shelf event emitted | 44 | 3 | 44 |
| right slot | 41 | 7 | 34 |
| right sku | 34 | 2 | 34 |
| associated to a shopper | 32 | 0 | 44 |
| right shopper | 32 | 1 | 36 |
| conceal or pay classified | 31 | 10 | 36 |
| ledger basket | 21 | 2 | 41 |
| alert | 19 | 4 | 30 |

What a stage means:

- in view: a rendered shelf, cooler or counter camera sees the item in the hand, or the reaching hand, around the pick
- frame reached pipeline: one of those frames is in the pipeline's frame log (the camera node sent it)
- hand or item detected: in such a frame: a product box on the item (IoU 0.3), or a person box on the picker with a wrist or hand point near the true hand
- shelf event emitted: a PICK event within 3 s of the true pick (each PICK is paired with one true pick at most)
- right slot: that PICK's meta.slot.id is the true slot
- right sku: that PICK's sku (or item) is the true SKU
- associated to a shopper: that PICK has a person_id that appears in the frame logs
- right shopper: the boxes of that person_id around the pick sit on the true shopper
- conceal or pay classified: a CONCEAL (stolen), PAY (paid) or PUT_BACK (put back) event on the true shopper near the true time
- ledger basket: stolen: an alert or review lists this pick as unpaid. Paid or put back: none does
- alert: stolen: that record is alert tier. Paid or put back: no alert or review on this shopper

## What makes it hard

A pick, a theft or a shopper can carry several tags; "none" has none of them (night aside).

| picks | n | found | right slot | right SKU | right shopper | through every stage |
|---|---|---|---|---|---|---|
| best camera knocked | 1 | 1 | 0 | 1 | 1 | 0 |
| group | 8 | 6 | 6 | 6 | 6 | 6 |
| scan dropped | 2 | 1 | 1 | 1 | 1 | 1 |
| slot holds another product | 3 | 3 | 3 | 0 | 3 | 0 |
| none | 61 | 54 | 36 | 40 | 39 | 11 |

| stolen items | n | alert tier | alert or review |
|---|---|---|---|
| slot holds another product | 2 | 0 | 2 |
| none | 10 | 0 | 7 |

| honest shoppers | n | flagged, alert tier | flagged, alert or review |
|---|---|---|---|
| group carries not pays | 3 | 0 | 0 |
| group pays for another | 3 | 0 | 0 |
| put back | 7 | 0 | 5 |
| put back other slot | 5 | 0 | 3 |
| scan dropped | 2 | 0 | 0 |
| touched a shelf | 1 | 0 | 1 |
| none | 14 | 0 | 0 |

Clips with a scenario option against clips without it (whole clips, so other options are mixed in: a hint, not a measurement):

| option | clips with, without | thefts flagged | honest shoppers flagged | picks found | identities per person |
|---|---|---|---|---|---|
| block | 2, 4 | 5/5, 4/7 | 1/10, 4/19 | 23/26, 42/49 | 1.357, 1.556 |
| bump | 1, 5 | 1/1, 8/11 | 0/7, 5/22 | 13/16, 52/59 | 1.375, 1.515 |
| dropScan | 2, 4 | 1/3, 8/9 | 2/8, 3/21 | 18/21, 47/54 | 1.636, 1.433 |
| group | 3, 3 | 7/7, 2/5 | 3/17, 2/12 | 37/41, 28/34 | 1.435, 1.556 |
| shift | 1, 5 | 0/2, 9/10 | 1/5, 4/24 | 10/12, 55/63 | 1.714, 1.441 |
| staff | 2, 4 | 3/4, 6/8 | 2/11, 3/18 | 24/28, 41/47 | 1.5, 1.48 |
| wrongSlot | 5, 1 | 7/10, 2/2 | 3/22, 2/7 | 51/60, 14/15 | 1.469, 1.556 |

## By camera kind (the mount of the item camera with the best view of the pick)

| mount | picks | found | right slot | right SKU | right shopper | false PICK events from this mount |
|---|---|---|---|---|---|---|
| ceiling | 7 | 5 | 2 | 3 | 4 | 3 |
| counter-front | 2 | 2 | 2 | 2 | 2 | 1 |
| drop-rod | 14 | 13 | 12 | 11 | 13 | 5 |
| endcap | 3 | 2 | 1 | 1 | 1 | 1 |
| none | 13 | 9 | 4 | 4 | 4 | 0 |
| pole-on-gondola | 3 | 1 | 1 | 1 | 1 | 1 |
| price-rail | 29 | 29 | 21 | 23 | 23 | 13 |
| wall | 4 | 4 | 3 | 3 | 2 | 0 |

## Detection detail (independent counts over all true picks)

| | picks |
|---|---|
| item detected | 0 |
| right sku detected | 0 |
| person detected | 0 |
| hand detected | 59 |
| right fixture | 58 |
| picks with item frames | 75 |
| picks with item frames reached | 75 |

## Per clip

| clip | options | cameras | shoppers | thieves | picks | PICK events | stolen | flagged | alerts | false | honest flagged | ids | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13001 | wrongSlot shift dropScan | 27 | 7 | 2 | 12 | 14 | 2 | 0 | 0 | 0 | 1 of 5 | 12 | 1475.1 |
| 13002 | wrongSlot dropScan | 32 | 4 | 1 | 9 | 9 | 1 | 1 | 0 | 0 | 1 of 3 | 6 | 1446.1 |
| 13003 | group wrongSlot bump block | 36 | 8 | 1 | 16 | 18 | 1 | 1 | 0 | 0 | 0 of 7 | 11 | 1045.3 |
| 13004 | group staff | 27 | 8 | 1 | 15 | 22 | 2 | 2 | 0 | 0 | 2 of 7 | 14 | 803.3 |
| 13005 | group wrongSlot block | 19 | 6 | 3 | 10 | 14 | 4 | 4 | 0 | 0 | 1 of 3 | 8 | 794.7 |
| 13006 | staff wrongSlot | 17 | 6 | 2 | 13 | 14 | 2 | 1 | 0 | 0 | 0 of 4 | 10 | 658.5 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | tags | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13001 | 9.28 | P001 | gondola | G2L-S2-34 | torqueline_5w30 | paid |  | shelf event emitted | 76 (76) |  |  |  |  |  |
| 13001 | 12.93 | P005 | checkout | CK-2 | cocoa_crest | paid |  | right slot | 29 (29) |  |  |  | cocoa_crest | P006 |
| 13001 | 14.18 | P003 | gondola | G1L-S1-19 | sierra_nacho | paid |  | right shopper | 44 (44) |  |  |  | sierra_nacho | P002 |
| 13001 | 15.23 | P002 | gondola | G2L-S1-5 | oatfield_cookies | paid |  | none | 85 (85) |  |  |  | oatfield_cookies | P002 |
| 13001 | 15.97 | P007 | checkout | CK-14 | peanut_pilot | concealed |  | conceal or pay classified | 39 (39) |  |  |  | peanut_pilot | P007 |
| 13001 | 18.2 | P004 | cooler | D3-S4-7 | pacer_glacier | paid |  | right slot | 84 (84) |  |  |  | pacer_punch | P004 |
| 13001 | 23.5 | P002 | gondola | G1L-S4-28 | relieva_ibu | paid | scan_dropped | none | 27 (27) |  |  |  | relieva_ibu | P002 |
| 13001 | 27.98 | P005 | gondola | G1L-S1-15 | crunchly_classic | paid |  | none | 64 (64) |  |  |  | crunchly_classic | P005 |
| 13001 | 29.77 | P006 | gondola | G1L-S4-23 | voltlink_usbc | put_back |  | conceal or pay classified | 80 (80) |  |  |  | voltlink_usbc | P006 |
| 13001 | 32.38 | P004 | checkout | CK-7 | peanut_pilot | concealed |  | hand or item detected | 48 (48) |  |  |  | ridgeline_original | P007 |
| 13001 | 47.08 | P006 | cooler | D10-S5-3 | arcwave_berry | paid |  | shelf event emitted | 57 (57) |  |  |  |  |  |
| 13001 | 57.57 | P006 | checkout | CK-11 | peanut_pilot | put_back |  | conceal or pay classified | 52 (52) |  |  |  | peanut_pilot | P006 |
| 13002 | 11.1 | P001 | gondola | G2L-S2-21 | relieva_ibu | paid |  | none | 105 (105) |  |  |  | relieva_ibu | P001 |
| 13002 | 19 | P002 | cooler | D9-S2-5 | lumen_citrus | paid |  | hand or item detected | 50 (50) |  |  |  | fizzo_cola | P002 |
| 13002 | 19.43 | P004 | gondola | G3L-S4-42 | caramel_crest | paid |  | ledger basket | 41 (41) |  |  |  | caramel_crest | P004 |
| 13002 | 23.98 | P003 | cooler | D10-S4-5 | arcwave_berry | put_back |  | conceal or pay classified | 55 (55) |  |  |  | arcwave_berry | P003 |
| 13002 | 24.27 | P001 | checkout | CK-2 | peanut_pilot | paid | slot_holds_another_product | right sku | 40 (40) |  |  |  | caramel_crest | P001 |
| 13002 | 24.53 | P002 | cooler | D4-S1-8 | pacer_punch | paid | scan_dropped | hand or item detected | 58 (58) |  |  |  |  |  |
| 13002 | 24.93 | P004 | gondola | G2R-S1-28 | voltlink_micro | concealed |  | conceal or pay classified | 48 (48) |  |  |  | voltlink_micro | P004 |
| 13002 | 38.88 | P001 | cooler | D5-S3-3 | fizzo_cola | paid |  | hand or item detected | 37 (37) |  |  |  | torqueline_5w30 | P003 |
| 13002 | 41.83 | P003 | gondola | G1R-S2-38 | torqueline_5w30 | paid |  | right slot | 101 (101) |  |  |  | morning_hoops | P003 |
| 13003 | 13.73 | P001 | cooler | D8-S1-2 | fizzo_cherry | paid |  | hand or item detected | 52 (52) |  |  |  | fizzo_cherry | P002 |
| 13003 | 15.18 | P004 | gondola | G2L-S1-39 | caramel_crest | put_back |  | none | 93 (93) |  |  |  | caramel_crest | P004 |
| 13003 | 15.58 | P002 | cooler | D7-S3-2 | fizzo_cola | paid |  | right slot | 51 (51) |  |  |  | fizzo_cola | P007 |
| 13003 | 16.58 | P005 | gondola | G3R-S4-42 | ridgeline_original | paid | group | none | 70 (70) |  |  |  | ridgeline_original | P005 |
| 13003 | 17.43 | P003 | cooler | D13-S3-8 | orchard_peach | paid |  | hand or item detected | 30 (30) |  |  |  |  |  |
| 13003 | 17.83 | P006 | gondola | G3R-S4-37 | torqueline_10w30 | paid | group | none | 103 (103) |  |  |  | torqueline_10w30 | P006 |
| 13003 | 18.77 | P001 | cooler | D7-S4-4 | fizzo_cola | paid | best_camera_knocked | right slot | 65 (65) |  |  |  | fizzo_cola | P001 |
| 13003 | 21.48 | P007 | gondola | G1R-S3-46 | peanut_pilot | concealed |  | conceal or pay classified | 92 (92) |  |  |  | peanut_pilot | P007 |
| 13003 | 22.67 | P003 | gondola | G4R-S4-10 | peanut_pilot | paid |  | conceal or pay classified | 32 (32) |  |  |  | peanut_pilot | P003 |
| 13003 | 22.97 | P005 | checkout | CK-6 | caramel_crest | paid | group | none | 39 (39) |  |  |  | caramel_crest | P005 |
| 13003 | 25.37 | P008 | gondola | G1R-S2-25 | voltlink_micro | paid |  | right slot | 95 (95) |  |  |  | caramel_crest | P007 |
| 13003 | 27.83 | P002 | checkout | CK-13 | peanut_pilot | paid |  | hand or item detected | 27 (27) |  |  |  | peanut_pilot | P002 |
| 13003 | 30.03 | P004 | gondola | G4L-S2-9 | crunchly_bbq | paid |  | none | 103 (103) |  |  |  | crunchly_bbq | P004 |
| 13003 | 31.33 | P005 | gondola | G3R-S1-40 | peanut_pilot | paid | group | shelf event emitted | 28 (28) |  |  |  |  |  |
| 13003 | 35.73 | P008 | gondola | G3R-S4-11 | relieva_aceta | paid |  | none | 59 (59) |  |  |  | relieva_aceta | P008 |
| 13003 | 43.68 | P008 | cooler | D2-S1-4 | clearwell_water | paid |  | hand or item detected | 38 (38) |  |  |  |  |  |
| 13004 | 10.63 | P001 | gondola | G3R-S3-8 | peanut_pilot | paid |  | none | 68 (68) |  |  |  | peanut_pilot | P001 |
| 13004 | 13.88 | P003 | gondola | G3R-S1-11 | cocoa_crest | put_back |  | alert | 83 (83) |  |  |  | cocoa_crest | P003 |
| 13004 | 14.28 | P005 | checkout | CK-13 | peanut_pilot | paid | group | hand or item detected | 41 (41) |  |  |  |  |  |
| 13004 | 16.38 | P004 | checkout | CK-3 | caramel_crest | paid | group | none | 40 (40) |  |  |  | caramel_crest | P004 |
| 13004 | 20.03 | P001 | checkout | CK-12 | peanut_pilot | paid |  | right shopper | 34 (34) |  |  |  | peanut_pilot | P005 |
| 13004 | 20.08 | P003 | gondola | G3R-S4-11 | relieva_aceta | paid |  | ledger basket | 66 (66) |  |  |  | relieva_aceta | P003 |
| 13004 | 21.07 | P006 | gondola | G3R-S4-1 | torqueline_10w30 | concealed |  | conceal or pay classified | 66 (66) |  |  |  | torqueline_10w30 | P006 |
| 13004 | 22.8 | P004 | gondola | G4R-S3-19 | relieva_aceta | paid | group | none | 43 (43) |  |  |  | relieva_aceta | P004 |
| 13004 | 23.83 | P007 | gondola | G4R-S4-9 | cocoa_crest | paid |  | right slot | 41 (41) |  |  |  | cocoa_crest | P007 |
| 13004 | 26.98 | P008 | cooler | D11-S2-9 | arcwave_original | put_back |  | hand or item detected | 31 (31) |  |  |  | ridgeline_original | P001 |
| 13004 | 27 | P009 | gondola | G4L-S3-12 | cocoa_crest | paid |  | right slot | 61 (61) |  |  |  | cocoa_crest | P009 |
| 13004 | 27.5 | P003 | cooler | D12-S1-2 | arcwave_tropical | paid |  | hand or item detected | 33 (33) |  |  |  | arcwave_original | P003 |
| 13004 | 29.78 | P001 | gondola | G4L-S3-22 | ridgeline_original | paid |  | hand or item detected | 38 (38) |  |  |  | cocoa_crest | STAFF1 |
| 13004 | 32.83 | P006 | gondola | G4R-S2-17 | cocoa_crest | concealed |  | conceal or pay classified | 66 (66) |  |  |  | cocoa_crest | P006 |
| 13004 | 39.97 | P008 | gondola | G4L-S3-24 | ridgeline_original | put_back |  | alert | 7 (7) |  |  |  | ridgeline_original | P008 |
| 13005 | 9.17 | P001 | gondola | G4L-S3-31 | morning_hoops | paid | group | none | 68 (68) |  |  |  | morning_hoops | P001 |
| 13005 | 11.97 | P003 | gondola | G4L-S1-31 | relieva_ibu | concealed |  | conceal or pay classified | 38 (38) |  |  |  | relieva_ibu | P003 |
| 13005 | 16.58 | P004 | gondola | G3R-S1-2 | peanut_pilot | concealed |  | conceal or pay classified | 25 (25) |  |  |  | peanut_pilot | P004 |
| 13005 | 20.03 | P005 | gondola | G4R-S4-42 | caramel_crest | put_back |  | alert | 75 (75) |  |  |  | caramel_crest | P005 |
| 13005 | 23.27 | P006 | gondola | G4R-S2-46 | caramel_crest | concealed | slot_holds_another_product | right sku | 32 (32) |  |  |  | cocoa_crest | P006 |
| 13005 | 28.8 | P005 | gondola | G4R-S3-16 | relieva_aceta | paid |  | alert | 62 (62) |  |  |  | relieva_aceta | P005 |
| 13005 | 37.5 | P005 | checkout | CK-2 | caramel_crest | paid |  | alert | 42 (42) |  |  |  | caramel_crest | P005 |
| 13005 | 37.93 | P006 | gondola | G3R-S4-7 | torqueline_5w30 | concealed |  | conceal or pay classified | 50 (50) |  |  |  | torqueline_5w30 | P006 |
| 13005 | 47.73 | P005 | gondola | G4L-S3-1 | oatfield_cookies | put_back |  | conceal or pay classified | 25 (25) |  |  |  | oatfield_cookies | P005 |
| 13005 | 61.88 | P005 | checkout | CK-7 | caramel_crest | paid |  | alert | 38 (38) |  |  |  | caramel_crest | P005 |
| 13006 | 11.9 | P001 | gondola | G3L-S4-25 | ridgeline_original | put_back |  | right slot | 52 (52) |  |  |  | caramel_crest | STAFF1 |
| 13006 | 17.43 | P003 | gondola | G1L-S3-20 | relieva_aceta | paid |  | none | 56 (56) |  |  |  | relieva_aceta | P003 |
| 13006 | 18.83 | P004 | gondola | G3L-S2-3 | cocoa_crest | paid |  | hand or item detected | 14 (14) |  |  |  |  |  |
| 13006 | 20.23 | P001 | gondola | G3L-S1-1 | torqueline_5w30 | paid |  | hand or item detected | 15 (15) |  |  |  | relieva_aceta | P006 |
| 13006 | 20.27 | P003 | gondola | G1L-S4-23 | voltlink_micro | paid |  | shelf event emitted | 66 (66) |  |  |  |  |  |
| 13006 | 23.58 | P003 | gondola | G1L-S4-33 | relieva_ibu | paid |  | hand or item detected | 35 (35) |  |  |  | voltlink_micro | P007 |
| 13006 | 24.33 | P007 | gondola | G3L-S4-30 | morning_hoops | paid |  | none | 43 (43) |  |  |  | morning_hoops | P007 |
| 13006 | 25.43 | P006 | gondola | G3L-S2-32 | relieva_aceta | concealed |  | right slot | 39 (39) |  |  |  | relieva_ibu | P003 |
| 13006 | 26 | P005 | gondola | G1L-S4-6 | cocoa_crest | concealed | slot_holds_another_product | right sku | 43 (43) |  |  |  | caramel_crest | P005 |
| 13006 | 27.28 | P001 | gondola | G3L-S1-40 | ridgeline_teriyaki | paid |  | right slot | 29 (29) |  |  |  | torqueline_5w30 | STAFF1 |
| 13006 | 35.68 | P007 | gondola | G1L-S3-8 | torqueline_10w30 | paid |  | none | 50 (50) |  |  |  | torqueline_10w30 | P007 |
| 13006 | 44.5 | P006 | gondola | G3L-S2-9 | cocoa_crest | paid |  | hand or item detected | 24 (24) |  |  |  |  |  |
| 13006 | 52.37 | P007 | checkout | CK-7 | caramel_crest | paid |  | none | 35 (35) |  |  |  | caramel_crest | P007 |
