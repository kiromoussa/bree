# Benchmark, checkpoint batch 2 split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

6 clips (13007, 13008, 13009, 13010, 13011, 13012), 413 s of sim time, 42 shoppers (15 thieves, 27 honest), 2 staff, 79 picks, 20 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-07 15:49, commit e5a9830.

## Scorecard

Square brackets: 95 percent bootstrap interval (shoppers drawn again 2000 times; clips for precision and identities). The sample is small, so read the interval before the point value.

Stressed: the same clips and pipeline with worse inputs, {"drop_item_cameras": 0.2, "calib_noise_deg": 0.5, "register_delay_s": 5.0, "wrong_planogram": 0.05, "seed": 1} (`python -m bree.sim.bench --help`). No ground truth is used to make them worse.

| Metric | Value | Stressed (6 clips) |
|---|---|---|
| Thefts flagged, alert tier | 1 of 20 (5.0%) [0.00 to 0.17] | 0 of 20 (0.0%) [0.00 to 0.00] |
| Thefts flagged, review tier only | 11 of 20 (55.0%) | 9 of 20 (45.0%) |
| Thefts flagged, alert or review | 12 of 20 (60.0%) [0.32 to 0.85] | 9 of 20 (45.0%) [0.21 to 0.70] |
| Stolen items listed as unpaid on a record | 12 of 20 (60.0%) | 8 of 20 (40.0%) |
| Alert precision | 50.0% (1/2) | 0.0% (0/1) |
| Honest shoppers flagged, alert or review | 7 of 27 (25.9%) [0.11 to 0.41] | 5 of 27 (18.5%) [0.04 to 0.33] |
| Honest shoppers flagged, alert tier | 0 of 27 (0.0%) [0.00 to 0.00] | 0 of 27 (0.0%) [0.00 to 0.00] |
| Honest shoppers flagged, review tier only | 7 of 27 (25.9%) | 5 of 27 (18.5%) |
| Staff members flagged, alert or review (of them alert tier) | 1 of 2 (1) | 2 of 2 (1) |
| Staff takes listed as unpaid on a record | 0 of 1 (a PICK event on 1) | 0 of 1 (a PICK event on 1) |
| Shifted items counted as a pick (of them listed as unpaid) | 5 of 10 (4) | 3 of 10 (1) |
| Picks found | 64 of 79 (81.0%) [0.73 to 0.90] | 55 of 79 (69.6%) [0.59 to 0.79] |
| Pick precision (staff takes count as true) | 75.6% of 86 PICK events [0.68 to 0.84] | 77.8% of 72 PICK events [0.69 to 0.90] |
| Right slot, of paired picks | 73.4% [0.61 to 0.85] | 47.3% [0.35 to 0.60] |
| Right SKU, of paired picks (nominal planogram given) | 84.4% [0.75 to 0.93]; of all true picks 68.4% | 78.2% [0.67 to 0.88]; of all true picks 54.4% |
| Right shopper, of paired picks | 78.1% [0.68 to 0.88] | 76.4% [0.66 to 0.86] |
| Put-backs found (recall) | 7 of 17 (41.2%) [0.20 to 0.67] | 8 of 17 (47.1%) [0.22 to 0.75] |
| Put-back precision (staff puts count as true) | 63.2% of 19 PUT_BACK events [0.33 to 0.83] | 50.0% of 20 PUT_BACK events [0.33 to 0.83] |
| Put-backs into another slot found | 5 of 9 | 6 of 9 |
| Identities per person | 1.25 (55 for 44) [1.14 to 1.33] | 1.773 (78 for 44) [1.60 to 1.90] |
| Identities that cover two people | 22 of 55 (40.0%) [0.26 to 0.54] | 34 of 75 (45.3%) [0.39 to 0.51] |
| People never tracked | 0 | 0 |
| False alerts per hour | 8.7 | 8.7 |
| Time to alert after concealment | 14.43 s median, 14.43 s max | n/a |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 79 picks, 20 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 79 | 0 | 79 |
| frame reached pipeline | 79 | 0 | 79 |
| hand or item detected | 79 | 13 | 66 |
| shelf event emitted | 66 | 6 | 64 |
| right slot | 60 | 14 | 47 |
| right sku | 46 | 1 | 54 |
| associated to a shopper | 45 | 0 | 64 |
| right shopper | 45 | 4 | 50 |
| conceal or pay classified | 41 | 12 | 48 |
| ledger basket | 29 | 1 | 65 |
| alert | 28 | 8 | 36 |

### All picks, stressed: 79 picks, 12 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 79 | 0 | 79 |
| frame reached pipeline | 79 | 0 | 79 |
| hand or item detected | 79 | 28 | 51 |
| shelf event emitted | 51 | 4 | 55 |
| right slot | 47 | 23 | 26 |
| right sku | 24 | 1 | 43 |
| associated to a shopper | 23 | 0 | 55 |
| right shopper | 23 | 1 | 42 |
| conceal or pay classified | 22 | 6 | 48 |
| ledger basket | 16 | 3 | 58 |
| alert | 13 | 1 | 40 |

### Outcome concealed: 20 picks, 1 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 20 | 0 | 20 |
| frame reached pipeline | 20 | 0 | 20 |
| hand or item detected | 20 | 3 | 17 |
| shelf event emitted | 17 | 1 | 18 |
| right slot | 16 | 3 | 13 |
| right sku | 13 | 1 | 13 |
| associated to a shopper | 12 | 0 | 18 |
| right shopper | 12 | 2 | 13 |
| conceal or pay classified | 10 | 9 | 1 |
| ledger basket | 1 | 0 | 12 |
| alert | 1 | 0 | 1 |

### Outcome paid: 42 picks, 17 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 42 | 0 | 42 |
| frame reached pipeline | 42 | 0 | 42 |
| hand or item detected | 42 | 6 | 36 |
| shelf event emitted | 36 | 1 | 37 |
| right slot | 35 | 8 | 28 |
| right sku | 27 | 0 | 32 |
| associated to a shopper | 27 | 0 | 37 |
| right shopper | 27 | 2 | 29 |
| conceal or pay classified | 25 | 0 | 42 |
| ledger basket | 25 | 1 | 39 |
| alert | 24 | 7 | 29 |

### Outcome put_back: 17 picks, 2 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 17 | 0 | 17 |
| frame reached pipeline | 17 | 0 | 17 |
| hand or item detected | 17 | 4 | 13 |
| shelf event emitted | 13 | 4 | 9 |
| right slot | 9 | 3 | 6 |
| right sku | 6 | 0 | 9 |
| associated to a shopper | 6 | 0 | 9 |
| right shopper | 6 | 0 | 8 |
| conceal or pay classified | 6 | 3 | 5 |
| ledger basket | 3 | 0 | 14 |
| alert | 3 | 1 | 6 |

### Zone checkout: 18 picks, 5 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 18 | 0 | 18 |
| frame reached pipeline | 18 | 0 | 18 |
| hand or item detected | 18 | 5 | 13 |
| shelf event emitted | 13 | 3 | 11 |
| right slot | 10 | 1 | 9 |
| right sku | 9 | 0 | 10 |
| associated to a shopper | 9 | 0 | 11 |
| right shopper | 9 | 1 | 9 |
| conceal or pay classified | 8 | 0 | 13 |
| ledger basket | 8 | 1 | 14 |
| alert | 7 | 2 | 9 |

### Zone cooler: 9 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 9 | 0 | 9 |
| frame reached pipeline | 9 | 0 | 9 |
| hand or item detected | 9 | 5 | 4 |
| shelf event emitted | 4 | 0 | 5 |
| right slot | 4 | 3 | 2 |
| right sku | 1 | 0 | 2 |
| associated to a shopper | 1 | 0 | 5 |
| right shopper | 1 | 0 | 3 |
| conceal or pay classified | 1 | 1 | 5 |
| ledger basket | 0 | 0 | 8 |
| alert | 0 | 0 | 6 |

### Zone gondola: 52 picks, 15 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 52 | 0 | 52 |
| frame reached pipeline | 52 | 0 | 52 |
| hand or item detected | 52 | 3 | 49 |
| shelf event emitted | 49 | 3 | 48 |
| right slot | 46 | 10 | 36 |
| right sku | 36 | 1 | 42 |
| associated to a shopper | 35 | 0 | 48 |
| right shopper | 35 | 3 | 38 |
| conceal or pay classified | 32 | 11 | 30 |
| ledger basket | 21 | 0 | 43 |
| alert | 21 | 6 | 21 |

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
| best camera knocked | 1 | 1 | 0 | 0 | 0 | 0 |
| group | 10 | 6 | 3 | 5 | 4 | 0 |
| night | 15 | 10 | 5 | 8 | 6 | 0 |
| slot holds another product | 2 | 1 | 1 | 0 | 1 | 0 |
| none | 66 | 56 | 43 | 49 | 45 | 20 |

| stolen items | n | alert tier | alert or review |
|---|---|---|---|
| group | 2 | 0 | 1 |
| night | 5 | 0 | 2 |
| slot holds another product | 1 | 0 | 0 |
| none | 17 | 1 | 11 |

| honest shoppers | n | flagged, alert tier | flagged, alert or review |
|---|---|---|---|
| group carries not pays | 1 | 0 | 1 |
| group pays for another | 2 | 0 | 0 |
| night | 5 | 0 | 3 |
| put back | 11 | 0 | 7 |
| put back other slot | 7 | 0 | 3 |
| touched a shelf | 8 | 0 | 4 |
| none | 10 | 0 | 0 |

Clips with a scenario option against clips without it (whole clips, so other options are mixed in: a hint, not a measurement):

| option | clips with, without | thefts flagged | honest shoppers flagged | picks found | identities per person |
|---|---|---|---|---|---|
| bump | 1, 5 | 0/1, 12/19 | 2/6, 5/21 | 15/16, 49/63 | 1.286, 1.243 |
| group | 3, 3 | 8/12, 4/8 | 4/14, 3/13 | 32/41, 32/38 | 1.217, 1.286 |
| night | 1, 5 | 2/5, 10/15 | 3/5, 4/22 | 10/15, 54/64 | 1.0, 1.306 |
| shift | 5, 1 | 9/15, 3/5 | 7/24, 0/3 | 56/67, 8/12 | 1.25, 1.25 |
| staff | 2, 4 | 8/10, 4/10 | 1/8, 6/19 | 23/27, 41/52 | 1.294, 1.222 |
| wrongSlot | 4, 2 | 5/10, 7/10 | 3/17, 4/10 | 39/49, 25/30 | 1.296, 1.176 |

## By camera kind (the mount of the item camera with the best view of the pick)

| mount | picks | found | right slot | right SKU | right shopper | false PICK events from this mount |
|---|---|---|---|---|---|---|
| ceiling | 15 | 12 | 8 | 9 | 10 | 3 |
| counter-front | 4 | 4 | 4 | 4 | 4 | 2 |
| drop-rod | 7 | 7 | 4 | 5 | 5 | 3 |
| endcap | 1 | 1 | 0 | 0 | 0 | 0 |
| none | 11 | 10 | 6 | 7 | 5 | 0 |
| pole-on-gondola | 4 | 4 | 4 | 4 | 4 | 0 |
| price-rail | 29 | 19 | 16 | 19 | 18 | 13 |
| wall | 8 | 7 | 5 | 6 | 4 | 0 |

## Detection detail (independent counts over all true picks)

| | picks |
|---|---|
| item detected | 0 |
| right sku detected | 0 |
| person detected | 0 |
| hand detected | 66 |
| right fixture | 59 |
| picks with item frames | 79 |
| picks with item frames reached | 79 |

## Per clip

| clip | options | cameras | shoppers | thieves | picks | PICK events | stolen | flagged | alerts | false | honest flagged | ids | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13007 | group staff shift | 28 | 8 | 3 | 15 | 22 | 5 | 5 | 0 | 0 | 1 of 5 | 12 | 3669.4 |
| 13008 | group shift night | 18 | 8 | 3 | 15 | 13 | 5 | 2 | 0 | 0 | 3 of 5 | 8 | 3463.2 |
| 13009 | group wrongSlot shift | 28 | 6 | 2 | 11 | 9 | 2 | 1 | 1 | 0 | 0 of 4 | 8 | 1959.1 |
| 13010 | wrongSlot shift | 26 | 6 | 2 | 10 | 13 | 2 | 1 | 0 | 0 | 1 of 4 | 8 | 1762.5 |
| 13011 | staff wrongSlot | 17 | 7 | 4 | 12 | 13 | 5 | 3 | 1 | 1 | 0 of 3 | 10 | 1517.7 |
| 13012 | wrongSlot shift bump | 25 | 7 | 1 | 16 | 16 | 1 | 0 | 0 | 0 | 2 of 6 | 9 | 1656.4 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | tags | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 13007 | 12.27 | P001 | gondola | G4R-S4-1 | peanut_pilot | paid |  | none | 28 (28) |  |  |  | peanut_pilot | P001 |
| 13007 | 18.03 | P004 | gondola | G4R-S3-37 | torqueline_5w30 | paid |  | right slot | 35 (35) |  |  |  | torqueline_5w30 | P004 |
| 13007 | 20.1 | P002 | cooler | D3-S1-1 | pacer_punch | paid |  | hand or item detected | 50 (50) |  |  |  | pacer_punch | P002 |
| 13007 | 22.77 | P004 | gondola | G4R-S2-9 | peanut_pilot | put_back |  | alert | 43 (43) |  |  |  | peanut_pilot | P004 |
| 13007 | 26.3 | P005 | cooler | D12-S5-5 | arcwave_original | concealed |  | right slot | 46 (46) |  |  |  | arcwave_berry | P005 |
| 13007 | 27.08 | P006 | gondola | G4R-S3-10 | relieva_aceta | paid |  | right slot | 40 (40) |  |  |  | relieva_aceta | P004 |
| 13007 | 31.03 | P007 | gondola | G2L-S3-26 | relieva_aceta | concealed |  | right shopper | 90 (90) |  |  |  | relieva_aceta | P008 |
| 13007 | 32.4 | P005 | cooler | D11-S2-3 | arcwave_original | concealed |  | right slot | 52 (52) |  |  |  | cocoa_crest | P007 |
| 13007 | 33.87 | P006 | cooler | D7-S2-9 | fizzo_cola | paid |  | right slot | 62 (62) |  |  |  | voltlink_micro | STAFF1 |
| 13007 | 34.2 | P004 | checkout | CK-2 | peanut_pilot | paid |  | alert | 38 (38) |  |  |  | peanut_pilot | P004 |
| 13007 | 35.5 | P009 | gondola | G2L-S1-5 | oatfield_cookies | paid | group | right slot | 79 (79) |  |  |  | caramel_crest | P008 |
| 13007 | 36.73 | P008 | gondola | G2L-S1-8 | peanut_pilot | put_back | group | right slot | 42 (42) |  |  |  | peanut_pilot | P008 |
| 13007 | 42.63 | P007 | gondola | G2L-S3-20 | voltlink_micro | concealed |  | conceal or pay classified | 105 (105) |  |  |  | voltlink_micro | P007 |
| 13007 | 45 | P008 | gondola | G2L-S2-4 | torqueline_5w30 | paid | group | alert | 98 (98) |  |  |  | torqueline_5w30 | P008 |
| 13007 | 54.47 | P008 | cooler | D11-S4-6 | arcwave_berry | concealed | group | conceal or pay classified | 29 (29) |  |  |  | arcwave_berry | P008 |
| 13008 | 9.03 | P003 | checkout | CK-9 | caramel_crest | paid | night | hand or item detected | 35 (35) |  |  |  |  |  |
| 13008 | 11.2 | P001 | gondola | G3R-S2-5 | crunchly_classic | put_back | night group | hand or item detected | 31 (31) |  |  |  |  |  |
| 13008 | 15.88 | P007 | checkout | CK-3 | caramel_crest | put_back | night | hand or item detected | 52 (52) |  |  |  |  |  |
| 13008 | 18.28 | P005 | gondola | G1L-S3-24 | voltlink_micro | concealed | night | conceal or pay classified | 49 (49) |  |  |  | voltlink_micro | P005 |
| 13008 | 18.53 | P004 | gondola | G1L-S4-5 | cocoa_crest | concealed | night | conceal or pay classified | 40 (40) |  |  |  | cocoa_crest | P004 |
| 13008 | 21.03 | P008 | gondola | G4L-S4-2 | torqueline_5w30 | concealed | night | hand or item detected | 8 (8) |  |  |  | peanut_pilot | P006 |
| 13008 | 27.37 | P008 | gondola | G4L-S1-34 | relieva_ibu | concealed | night | conceal or pay classified | 25 (25) |  |  |  | relieva_ibu | P008 |
| 13008 | 28.37 | P001 | gondola | G1L-S4-26 | relieva_aceta | paid | night group | right shopper | 45 (45) |  |  |  | relieva_aceta | P006 |
| 13008 | 28.58 | P007 | gondola | G3R-S3-34 | caramel_crest | paid | night | alert | 49 (49) |  |  |  | caramel_crest | P007 |
| 13008 | 31.6 | P006 | gondola | G1L-S3-30 | voltlink_micro | put_back | night | shelf event emitted | 52 (52) |  |  |  |  |  |
| 13008 | 35.13 | P005 | checkout | CK-12 | caramel_crest | concealed | night | hand or item detected | 21 (21) |  |  |  | voltlink_micro | P006 |
| 13008 | 35.23 | P007 | checkout | CK-6 | caramel_crest | put_back | night | shelf event emitted | 50 (50) |  |  |  |  |  |
| 13008 | 38.17 | P006 | gondola | G1L-S4-38 | caramel_crest | put_back | night | right slot | 26 (26) |  |  |  | caramel_crest | P002 |
| 13008 | 41.18 | P007 | checkout | CK-6 | caramel_crest | paid | night | right slot | 38 (38) |  |  |  | caramel_crest | P007 |
| 13008 | 55.7 | P006 | gondola | G3R-S1-16 | voltlink_micro | paid | night | right slot | 34 (34) |  |  |  | voltlink_micro | P006 |
| 13009 | 10.03 | P002 | checkout | CK-1 | cocoa_crest | paid |  | none | 34 (34) |  |  |  | cocoa_crest | P002 |
| 13009 | 14.37 | P001 | gondola | G4R-S4-13 | peanut_pilot | paid |  | none | 35 (35) |  |  |  | peanut_pilot | P001 |
| 13009 | 18.03 | P003 | gondola | G2L-S2-16 | relieva_aceta | concealed |  | none | 84 (84) |  |  |  | relieva_aceta | P003 |
| 13009 | 20.87 | P004 | cooler | D14-S4-7 | orchard_peach | paid |  | hand or item detected | 24 (24) |  |  |  |  |  |
| 13009 | 23.6 | P005 | cooler | D13-S3-3 | orchard_peach | put_back | group | hand or item detected | 49 (49) |  |  |  |  |  |
| 13009 | 25.48 | P006 | cooler | D14-S1-6 | orchard_peach | paid | group | hand or item detected | 26 (26) |  |  |  |  |  |
| 13009 | 27.77 | P001 | gondola | G2L-S3-42 | caramel_crest | paid |  | none | 71 (71) |  |  |  | caramel_crest | P001 |
| 13009 | 30.5 | P005 | cooler | D13-S5-5 | orchard_peach | put_back | group | hand or item detected | 32 (32) |  |  |  |  |  |
| 13009 | 35.47 | P004 | gondola | G2L-S2-37 | torqueline_5w30 | paid |  | none | 85 (85) |  |  |  | torqueline_5w30 | P004 |
| 13009 | 39.5 | P001 | checkout | CK-8 | cocoa_crest | paid |  | none | 34 (34) |  |  |  | cocoa_crest | P001 |
| 13009 | 50.1 | P005 | gondola | G1R-S2-41 | ridgeline_teriyaki | concealed | group | right slot | 61 (61) |  |  |  | ridgeline_teriyaki | P005 |
| 13010 | 12.97 | P001 | gondola | G2L-S1-6 | cocoa_crest | put_back |  | conceal or pay classified | 47 (47) |  |  |  | cocoa_crest | P001 |
| 13010 | 13.9 | P002 | gondola | G2L-S3-23 | voltlink_usbc | concealed |  | conceal or pay classified | 73 (73) |  |  |  | voltlink_usbc | P002 |
| 13010 | 14.63 | P003 | gondola | G2L-S2-32 | torqueline_5w30 | paid |  | none | 86 (86) |  |  |  | torqueline_5w30 | P003 |
| 13010 | 16.43 | P004 | gondola | G4R-S2-34 | peanut_pilot | concealed | slot_holds_another_product | right sku | 44 (44) |  |  |  | cocoa_crest | P004 |
| 13010 | 22.5 | P005 | gondola | G2L-S2-26 | voltlink_micro | paid |  | none | 98 (98) |  |  |  | voltlink_micro | P005 |
| 13010 | 23.33 | P006 | checkout | CK-13 | cocoa_crest | paid |  | none | 34 (34) |  |  |  | cocoa_crest | P006 |
| 13010 | 26.97 | P003 | gondola | G3R-S1-47 | peanut_pilot | put_back |  | none | 76 (76) |  |  |  | peanut_pilot | P003 |
| 13010 | 30.78 | P002 | gondola | G4R-S4-30 | relieva_aceta | paid |  | alert | 48 (48) |  |  |  | relieva_aceta | P002 |
| 13010 | 32.38 | P001 | checkout | CK-2 | cocoa_crest | paid |  | ledger basket | 41 (41) |  |  |  | cocoa_crest | P001 |
| 13010 | 41.67 | P003 | checkout | CK-10 | caramel_crest | paid | slot_holds_another_product | shelf event emitted | 34 (34) |  |  |  |  |  |
| 13011 | 8.53 | P001 | checkout | CK-6 | cocoa_crest | concealed |  | shelf event emitted | 39 (39) |  |  |  |  |  |
| 13011 | 11.7 | P002 | checkout | CK-7 | cocoa_crest | concealed |  | hand or item detected | 39 (39) |  |  |  |  |  |
| 13011 | 13.77 | P004 | checkout | CK-11 | caramel_crest | paid |  | none | 37 (37) |  |  |  | caramel_crest | P004 |
| 13011 | 18.67 | P005 | gondola | G3L-S3-8 | crunchly_classic | put_back |  | conceal or pay classified | 47 (47) |  |  |  | crunchly_classic | P005 |
| 13011 | 19.33 | P006 | gondola | G2R-S2-25 | relieva_ibu | paid |  | none | 30 (30) |  |  |  | relieva_ibu | P006 |
| 13011 | 23.37 | P007 | gondola | G4R-S3-8 | torqueline_5w30 | concealed |  | conceal or pay classified | 36 (36) |  |  |  | torqueline_5w30 | P007 |
| 13011 | 24.33 | P004 | gondola | G3L-S4-23 | ridgeline_original | put_back |  | shelf event emitted | 43 (43) |  |  |  |  |  |
| 13011 | 27.98 | P005 | gondola | G2R-S2-53 | oatfield_cookies | paid |  | none | 4 (4) |  |  |  | oatfield_cookies | P005 |
| 13011 | 28.17 | P008 | gondola | G4R-S4-3 | caramel_crest | concealed |  | conceal or pay classified | 46 (46) |  |  |  | caramel_crest | P008 |
| 13011 | 36.77 | P005 | checkout | CK-3 | cocoa_crest | paid |  | hand or item detected | 41 (41) |  |  |  |  |  |
| 13011 | 41.33 | P004 | checkout | CK-13 | caramel_crest | paid |  | right shopper | 33 (33) |  |  |  | caramel_crest | P005 |
| 13011 | 42.87 | P008 | gondola | G4R-S4-51 | peanut_pilot | concealed |  | conceal or pay classified | 24 (24) |  |  |  | peanut_pilot | P008 |
| 13012 | 14.2 | P001 | gondola | G1L-S2-37 | cocoa_crest | paid |  | none | 24 (24) |  |  |  | cocoa_crest | P001 |
| 13012 | 14.27 | P002 | gondola | G2R-S2-13 | peanut_pilot | put_back |  | none | 46 (46) |  |  |  | peanut_pilot | P002 |
| 13012 | 18.27 | P004 | gondola | G3L-S2-22 | voltlink_micro | concealed |  | right shopper | 27 (27) |  |  |  | voltlink_micro | P003 |
| 13012 | 19.1 | P003 | gondola | G3L-S4-12 | peanut_pilot | put_back |  | shelf event emitted | 49 (49) |  |  |  |  |  |
| 13012 | 21.83 | P001 | gondola | G1L-S4-12 | peanut_pilot | paid | best_camera_knocked | right slot | 38 (38) |  |  |  | crunchly_classic | P005 |
| 13012 | 21.97 | P005 | gondola | G1L-S1-22 | crunchly_classic | paid |  | right slot | 15 (15) |  |  |  | peanut_pilot | P001 |
| 13012 | 23.03 | P006 | checkout | CK-14 | peanut_pilot | paid |  | none | 34 (34) |  |  |  | peanut_pilot | P006 |
| 13012 | 23.2 | P002 | gondola | G2R-S4-20 | peanut_pilot | paid |  | none | 61 (61) |  |  |  | peanut_pilot | P002 |
| 13012 | 27.28 | P003 | gondola | G2R-S2-49 | caramel_crest | paid |  | alert | 37 (37) |  |  |  | caramel_crest | P003 |
| 13012 | 28.57 | P007 | gondola | G1L-S2-32 | morning_hoops | put_back |  | conceal or pay classified | 26 (26) |  |  |  | morning_hoops | P007 |
| 13012 | 31.6 | P006 | gondola | G3R-S2-11 | crunchly_classic | paid |  | none | 70 (70) |  |  |  | crunchly_classic | P006 |
| 13012 | 33.68 | P001 | gondola | G2R-S1-40 | ridgeline_original | paid |  | none | 42 (42) |  |  |  | ridgeline_original | P001 |
| 13012 | 33.93 | P003 | gondola | G4L-S2-24 | sierra_ranch | put_back |  | right slot | 21 (21) |  |  |  | sierra_ranch | P003 |
| 13012 | 39.43 | P007 | gondola | G1L-S2-18 | caramel_crest | paid |  | alert | 50 (50) |  |  |  | caramel_crest | P007 |
| 13012 | 44.07 | P003 | gondola | G3R-S1-4 | cocoa_crest | paid |  | hand or item detected | 13 (13) |  |  |  | peanut_pilot | P001 |
| 13012 | 55.07 | P007 | checkout | CK-4 | peanut_pilot | paid |  | alert | 40 (40) |  |  |  | peanut_pilot | P007 |
