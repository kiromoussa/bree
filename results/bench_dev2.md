# Benchmark, dev2 split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

20 clips (11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010, 11011, 11012, 11013, 11014, 11015, 11016, 11017, 11018, 11019, 11020), 1278 s of sim time, 120 shoppers (44 thieves, 76 honest), 7 staff, 230 picks, 55 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-06 20:38, commit eacf7ad.

## Scorecard

Square brackets: 95 percent bootstrap interval (shoppers drawn again 2000 times; clips for precision and identities). The sample is small, so read the interval before the point value.

Stressed: the same clips and pipeline with worse inputs, {"drop_item_cameras": 0.2, "calib_noise_deg": 0.5, "register_delay_s": 5.0, "wrong_planogram": 0.05, "seed": 1} (`python -m bree.sim.bench --help`). No ground truth is used to make them worse.

| Metric | Value | Stressed (20 clips) |
|---|---|---|
| Thefts flagged, alert tier | 0 of 55 (0.0%) [0.00 to 0.00] | 0 of 55 (0.0%) [0.00 to 0.00] |
| Thefts flagged, review tier only | 34 of 55 (61.8%) | 17 of 55 (30.9%) |
| Thefts flagged, alert or review | 34 of 55 (61.8%) [0.45 to 0.76] | 17 of 55 (30.9%) [0.17 to 0.45] |
| Stolen items listed as unpaid on a record | 28 of 55 (50.9%) | 11 of 55 (20.0%) |
| Alert precision | n/a (0/0) | n/a (0/0) |
| Honest shoppers flagged, alert or review | 12 of 76 (15.8%) [0.08 to 0.24] | 11 of 76 (14.5%) [0.08 to 0.22] |
| Honest shoppers flagged, alert tier | 0 of 76 (0.0%) [0.00 to 0.00] | 0 of 76 (0.0%) [0.00 to 0.00] |
| Honest shoppers flagged, review tier only | 12 of 76 (15.8%) | 11 of 76 (14.5%) |
| Staff members flagged, alert or review (of them alert tier) | 4 of 7 (0) | 3 of 7 (0) |
| Staff takes listed as unpaid on a record | 4 of 6 (a PICK event on 5) | 3 of 6 (a PICK event on 5) |
| Shifted items counted as a pick (of them listed as unpaid) | 8 of 16 (5) | 5 of 16 (2) |
| Picks found | 205 of 230 (89.1%) [0.85 to 0.93] | 170 of 230 (73.9%) [0.69 to 0.79] |
| Pick precision (staff takes count as true) | 75.8% of 277 PICK events [0.70 to 0.82] | 77.8% of 225 PICK events [0.71 to 0.84] |
| Right slot, of paired picks | 64.4% [0.58 to 0.71] | 49.4% [0.41 to 0.58] |
| Right SKU, of paired picks (nominal planogram given) | 74.1% [0.68 to 0.80]; of all true picks 66.1% | 70.6% [0.64 to 0.77]; of all true picks 52.2% |
| Right shopper, of paired picks | 77.6% [0.72 to 0.83] | 74.1% [0.68 to 0.80] |
| Put-backs found (recall) | 16 of 34 (47.1%) [0.30 to 0.65] | 15 of 34 (44.1%) [0.29 to 0.61] |
| Put-back precision (staff puts count as true) | 46.8% of 47 PUT_BACK events [0.33 to 0.63] | 44.0% of 50 PUT_BACK events [0.29 to 0.60] |
| Put-backs into another slot found | 4 of 15 | 3 of 15 |
| Identities per person | 1.417 (180 for 127) [1.35 to 1.50] | 1.819 (231 for 127) [1.68 to 1.98] |
| Identities that cover two people | 77 of 175 (44.0%) [0.33 to 0.54] | 131 of 224 (58.5%) [0.49 to 0.68] |
| People never tracked | 3 | 4 |
| False alerts per hour | 0.0 | 0.0 |
| Time to alert after concealment | n/a | n/a |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 230 picks, 47 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 230 | 1 | 229 |
| frame reached pipeline | 229 | 0 | 229 |
| hand or item detected | 229 | 44 | 185 |
| shelf event emitted | 185 | 9 | 205 |
| right slot | 176 | 54 | 132 |
| right sku | 122 | 6 | 152 |
| associated to a shopper | 116 | 0 | 205 |
| right shopper | 116 | 8 | 159 |
| conceal or pay classified | 108 | 41 | 143 |
| ledger basket | 67 | 7 | 175 |
| alert | 60 | 13 | 124 |

### All picks, stressed: 230 picks, 31 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 230 | 1 | 229 |
| frame reached pipeline | 229 | 1 | 228 |
| hand or item detected | 228 | 91 | 137 |
| shelf event emitted | 137 | 9 | 170 |
| right slot | 128 | 49 | 84 |
| right sku | 79 | 4 | 120 |
| associated to a shopper | 75 | 0 | 170 |
| right shopper | 75 | 6 | 126 |
| conceal or pay classified | 69 | 22 | 142 |
| ledger basket | 47 | 7 | 158 |
| alert | 40 | 9 | 138 |

### Outcome concealed: 55 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 55 | 0 | 55 |
| frame reached pipeline | 55 | 0 | 55 |
| hand or item detected | 55 | 9 | 46 |
| shelf event emitted | 46 | 0 | 52 |
| right slot | 46 | 15 | 33 |
| right sku | 31 | 3 | 37 |
| associated to a shopper | 28 | 0 | 52 |
| right shopper | 28 | 0 | 43 |
| conceal or pay classified | 28 | 28 | 0 |
| ledger basket | 0 | 0 | 28 |
| alert | 0 | 0 | 0 |

### Outcome paid: 141 picks, 46 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 141 | 1 | 140 |
| frame reached pipeline | 140 | 0 | 140 |
| hand or item detected | 140 | 27 | 113 |
| shelf event emitted | 113 | 7 | 122 |
| right slot | 106 | 26 | 85 |
| right sku | 80 | 2 | 96 |
| associated to a shopper | 78 | 0 | 122 |
| right shopper | 78 | 8 | 93 |
| conceal or pay classified | 70 | 5 | 133 |
| ledger basket | 65 | 6 | 124 |
| alert | 59 | 13 | 103 |

### Outcome put_back: 34 picks, 1 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 34 | 0 | 34 |
| frame reached pipeline | 34 | 0 | 34 |
| hand or item detected | 34 | 8 | 26 |
| shelf event emitted | 26 | 2 | 31 |
| right slot | 24 | 13 | 14 |
| right sku | 11 | 1 | 19 |
| associated to a shopper | 10 | 0 | 31 |
| right shopper | 10 | 0 | 23 |
| conceal or pay classified | 10 | 8 | 10 |
| ledger basket | 2 | 1 | 23 |
| alert | 1 | 0 | 21 |

### Zone checkout: 43 picks, 12 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 43 | 0 | 43 |
| frame reached pipeline | 43 | 0 | 43 |
| hand or item detected | 43 | 4 | 39 |
| shelf event emitted | 39 | 3 | 38 |
| right slot | 36 | 7 | 31 |
| right sku | 29 | 0 | 34 |
| associated to a shopper | 29 | 0 | 38 |
| right shopper | 29 | 2 | 33 |
| conceal or pay classified | 27 | 10 | 30 |
| ledger basket | 17 | 1 | 35 |
| alert | 16 | 4 | 25 |

### Zone cooler: 41 picks, 2 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 41 | 0 | 41 |
| frame reached pipeline | 41 | 0 | 41 |
| hand or item detected | 41 | 25 | 16 |
| shelf event emitted | 16 | 1 | 31 |
| right slot | 15 | 8 | 11 |
| right sku | 7 | 1 | 18 |
| associated to a shopper | 6 | 0 | 31 |
| right shopper | 6 | 2 | 19 |
| conceal or pay classified | 4 | 2 | 24 |
| ledger basket | 2 | 0 | 28 |
| alert | 2 | 0 | 21 |

### Zone gondola: 146 picks, 33 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 146 | 1 | 145 |
| frame reached pipeline | 145 | 0 | 145 |
| hand or item detected | 145 | 15 | 130 |
| shelf event emitted | 130 | 5 | 136 |
| right slot | 125 | 39 | 90 |
| right sku | 86 | 5 | 100 |
| associated to a shopper | 81 | 0 | 136 |
| right shopper | 81 | 4 | 107 |
| conceal or pay classified | 77 | 29 | 89 |
| ledger basket | 48 | 6 | 112 |
| alert | 42 | 9 | 78 |

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
| group | 44 | 35 | 18 | 24 | 23 | 6 |
| night | 127 | 112 | 67 | 80 | 86 | 21 |
| scan dropped | 6 | 5 | 4 | 5 | 3 | 2 |
| slot holds another product | 13 | 11 | 7 | 0 | 9 | 0 |
| none | 171 | 158 | 106 | 126 | 126 | 40 |

| stolen items | n | alert tier | alert or review |
|---|---|---|---|
| group | 5 | 0 | 3 |
| night | 31 | 0 | 20 |
| slot holds another product | 4 | 0 | 2 |
| none | 46 | 0 | 29 |

| honest shoppers | n | flagged, alert tier | flagged, alert or review |
|---|---|---|---|
| group carries not pays | 8 | 0 | 3 |
| group pays for another | 13 | 0 | 0 |
| night | 38 | 0 | 5 |
| put back | 29 | 0 | 11 |
| put back other slot | 13 | 0 | 7 |
| scan dropped | 8 | 0 | 3 |
| touched a shelf | 7 | 0 | 3 |
| none | 25 | 0 | 0 |

Clips with a scenario option against clips without it (whole clips, so other options are mixed in: a hint, not a measurement):

| option | clips with, without | thefts flagged | honest shoppers flagged | picks found | identities per person |
|---|---|---|---|---|---|
| block | 6, 14 | 12/15, 22/40 | 4/21, 8/55 | 59/64, 146/166 | 1.385, 1.432 |
| bump | 4, 16 | 8/9, 26/46 | 2/15, 10/61 | 42/46, 163/184 | 1.56, 1.382 |
| dropScan | 6, 14 | 11/13, 23/42 | 3/22, 9/54 | 45/52, 160/178 | 1.371, 1.435 |
| group | 13, 7 | 24/38, 10/17 | 8/51, 4/25 | 133/150, 72/80 | 1.442, 1.366 |
| night | 11, 9 | 20/31, 14/24 | 5/38, 7/38 | 112/127, 93/103 | 1.42, 1.414 |
| shift | 11, 9 | 20/28, 14/27 | 8/44, 4/32 | 110/122, 95/108 | 1.366, 1.482 |
| staff | 7, 13 | 14/17, 20/38 | 5/26, 7/50 | 78/82, 127/148 | 1.34, 1.462 |
| wrongSlot | 10, 10 | 18/25, 16/30 | 10/39, 2/37 | 104/116, 101/114 | 1.492, 1.348 |

## By camera kind (the mount of the item camera with the best view of the pick)

| mount | picks | found | right slot | right SKU | right shopper | false PICK events from this mount |
|---|---|---|---|---|---|---|
| ceiling | 39 | 29 | 12 | 17 | 17 | 12 |
| counter-front | 8 | 7 | 3 | 3 | 3 | 1 |
| drop-rod | 34 | 34 | 24 | 28 | 29 | 11 |
| endcap | 6 | 6 | 2 | 4 | 5 | 1 |
| none | 21 | 17 | 6 | 9 | 9 | 0 |
| pole-on-gondola | 10 | 10 | 7 | 9 | 7 | 4 |
| price-rail | 102 | 92 | 69 | 76 | 80 | 38 |
| wall | 10 | 10 | 9 | 6 | 9 | 0 |

## Detection detail (independent counts over all true picks)

| | picks |
|---|---|
| item detected | 0 |
| right sku detected | 0 |
| person detected | 0 |
| hand detected | 185 |
| right fixture | 186 |
| picks with item frames | 229 |
| picks with item frames reached | 229 |

## Per clip

| clip | options | cameras | shoppers | thieves | picks | PICK events | stolen | flagged | alerts | false | honest flagged | ids | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11001 | group wrongSlot shift bump night | 18 | 6 | 1 | 9 | 10 | 1 | 1 | 0 | 0 | 0 of 5 | 9 | 100.8 |
| 11002 | group staff block dropScan night | 28 | 5 | 2 | 7 | 11 | 2 | 2 | 0 | 0 | 0 of 3 | 8 | 183.2 |
| 11003 | wrongSlot | 27 | 5 | 1 | 14 | 18 | 2 | 0 | 0 | 0 | 2 of 4 | 8 | 169.4 |
| 11004 | group wrongSlot dropScan | 26 | 5 | 1 | 8 | 12 | 2 | 2 | 0 | 0 | 1 of 4 | 7 | 144.3 |
| 11005 | group wrongSlot shift night | 28 | 6 | 3 | 12 | 17 | 4 | 2 | 0 | 0 | 2 of 3 | 8 | 216.2 |
| 11006 | group shift dropScan | 19 | 8 | 2 | 10 | 12 | 2 | 2 | 0 | 0 | 0 of 6 | 10 | 109.8 |
| 11007 | group block night | 26 | 7 | 3 | 12 | 10 | 3 | 1 | 0 | 0 | 0 of 4 | 10 | 148.8 |
| 11008 | group wrongSlot shift dropScan | 27 | 6 | 3 | 11 | 10 | 4 | 3 | 0 | 0 | 0 of 3 | 9 | 155.4 |
| 11009 | group staff wrongSlot shift block | 25 | 6 | 3 | 15 | 20 | 4 | 4 | 0 | 0 | 1 of 3 | 9 | 160.4 |
| 11010 | group staff wrongSlot shift block | 23 | 7 | 1 | 12 | 18 | 1 | 0 | 0 | 0 | 2 of 6 | 12 | 158.6 |
| 11011 | group night | 22 | 7 | 5 | 17 | 15 | 7 | 3 | 0 | 0 | 0 of 2 | 9 | 156.2 |
| 11012 | staff wrongSlot shift block night | 22 | 5 | 2 | 11 | 12 | 3 | 3 | 0 | 0 | 0 of 3 | 8 | 184.8 |
| 11013 | group | 18 | 8 | 4 | 13 | 12 | 4 | 1 | 0 | 0 | 0 of 4 | 12 | 102.1 |
| 11014 | staff shift bump block dropScan night | 32 | 4 | 2 | 7 | 15 | 2 | 2 | 0 | 0 | 1 of 2 | 7 | 219.1 |
| 11015 | staff bump night | 23 | 6 | 2 | 15 | 20 | 3 | 2 | 0 | 0 | 0 of 4 | 9 | 208.3 |
| 11016 | shift | 28 | 6 | 2 | 11 | 11 | 4 | 2 | 0 | 0 | 0 of 4 | 8 | 186.1 |
| 11017 | group wrongSlot bump night | 31 | 7 | 3 | 15 | 16 | 3 | 3 | 0 | 0 | 1 of 4 | 14 | 253.1 |
| 11018 | night | 24 | 4 | 1 | 7 | 6 | 1 | 0 | 0 | 0 | 0 of 3 | 6 | 133.9 |
| 11019 | group wrongSlot shift dropScan | 34 | 5 | 1 | 9 | 11 | 1 | 0 | 0 | 0 | 1 of 4 | 7 | 205.9 |
| 11020 | staff shift night | 23 | 7 | 2 | 15 | 21 | 2 | 1 | 0 | 0 | 1 of 5 | 10 | 176.3 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | tags | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11001 | 9.9 | P002 | gondola | G3R-S1-52 | oatfield_cookies | paid | night | none | 69 (69) |  |  |  | oatfield_cookies | P002 |
| 11001 | 9.98 | P001 | checkout | CK-2 | caramel_crest | paid | night | none | 30 (30) |  |  |  | caramel_crest | P001 |
| 11001 | 17.33 | P001 | gondola | G3R-S4-33 | torqueline_10w30 | paid | night | none | 116 (116) |  |  |  | torqueline_10w30 | P001 |
| 11001 | 17.78 | P003 | gondola | G4L-S1-15 | cocoa_crest | paid | night group | right slot | 23 (23) |  |  |  | crunchly_classic | P003 |
| 11001 | 18.27 | P005 | gondola | G4R-S4-51 | peanut_pilot | concealed | night | conceal or pay classified | 45 (45) |  |  |  | peanut_pilot | P005 |
| 11001 | 19.03 | P004 | gondola | G4L-S1-4 | cocoa_crest | paid | night group | hand or item detected | 9 (9) |  |  |  |  |  |
| 11001 | 20.53 | P003 | gondola | G4L-S2-7 | crunchly_classic | paid | night group | shelf event emitted | 50 (50) |  |  |  |  |  |
| 11001 | 21.33 | P006 | gondola | G4R-S2-40 | caramel_crest | put_back | night | conceal or pay classified | 50 (50) |  |  |  | caramel_crest | P006 |
| 11001 | 30.1 | P006 | checkout | CK-1 | caramel_crest | paid | night | none | 31 (31) |  |  |  | caramel_crest | P006 |
| 11002 | 10.48 | P001 | checkout | CK-11 | caramel_crest | paid | night group | shelf event emitted | 38 (38) |  |  |  |  |  |
| 11002 | 11.53 | P002 | checkout | CK-3 | peanut_pilot | paid | night group scan_dropped | right shopper | 34 (34) |  |  |  | peanut_pilot | P001 |
| 11002 | 12.97 | P001 | checkout | CK-12 | caramel_crest | paid | night group | right slot | 31 (31) |  |  |  | caramel_crest | P002 |
| 11002 | 19.63 | P003 | cooler | D7-S3-1 | lumen_citrus | put_back | night | hand or item detected | 34 (34) |  |  |  | lumen_citrus | P003 |
| 11002 | 21.78 | P004 | cooler | D8-S3-6 | fizzo_zero | concealed | night slot_holds_another_product | right sku | 47 (47) |  |  |  | fizzo_cherry | P004 |
| 11002 | 24.93 | P006 | gondola | G4R-S2-9 | caramel_crest | concealed | night | conceal or pay classified | 45 (45) |  |  |  | caramel_crest | P006 |
| 11002 | 30.97 | P003 | gondola | G4R-S1-11 | sierra_ranch | paid | night | none | 49 (49) |  |  |  | sierra_ranch | P003 |
| 11003 | 10.2 | P002 | gondola | G4R-S1-24 | sierra_ranch | paid |  | hand or item detected | 32 (32) |  |  |  |  |  |
| 11003 | 10.9 | P001 | gondola | G3R-S3-25 | ridgeline_teriyaki | put_back |  | ledger basket | 130 (130) |  |  |  | ridgeline_teriyaki | P001 |
| 11003 | 13.77 | P003 | gondola | G4R-S2-32 | morning_hoops | put_back |  | conceal or pay classified | 47 (47) |  |  |  | morning_hoops | P003 |
| 11003 | 19.13 | P002 | cooler | D10-S4-3 | arcwave_original | concealed |  | right slot | 46 (46) |  |  |  | arcwave_original | P002 |
| 11003 | 19.23 | P005 | checkout | CK-13 | cocoa_crest | paid |  | hand or item detected | 43 (43) |  |  |  | cocoa_crest | P005 |
| 11003 | 20.13 | P001 | gondola | G3R-S2-4 | sierra_ranch | paid |  | alert | 90 (90) |  |  |  | sierra_ranch | P001 |
| 11003 | 21.57 | P004 | cooler | D13-S2-3 | orchard_peach | paid |  | hand or item detected | 43 (43) |  |  |  | morning_hoops | P003 |
| 11003 | 23.43 | P003 | gondola | G4R-S4-16 | voltlink_micro | paid |  | alert | 61 (61) |  |  |  | voltlink_micro | P003 |
| 11003 | 25.37 | P002 | gondola | G3R-S1-1 | peanut_pilot | concealed |  | conceal or pay classified | 26 (26) |  |  |  | peanut_pilot | P002 |
| 11003 | 29.83 | P004 | gondola | G4L-S2-9 | sierra_ranch | paid |  | none | 79 (79) |  |  |  | sierra_ranch | P004 |
| 11003 | 30.43 | P005 | cooler | D9-S4-1 | fizzo_cola | paid |  | hand or item detected | 51 (51) |  |  |  | fizzo_cherry | P005 |
| 11003 | 30.8 | P003 | gondola | G3R-S4-4 | torqueline_10w30 | paid |  | alert | 62 (62) |  |  |  | torqueline_10w30 | P003 |
| 11003 | 33.77 | P005 | cooler | D7-S4-8 | fizzo_cola | put_back |  | shelf event emitted | 69 (69) |  |  |  |  |  |
| 11003 | 45.8 | P005 | gondola | G4R-S2-29 | morning_hoops | paid |  | none | 58 (58) |  |  |  | morning_hoops | P005 |
| 11004 | 12.23 | P001 | gondola | G4R-S3-26 | voltlink_micro | concealed |  | right slot | 55 (55) |  |  |  | caramel_crest | P002 |
| 11004 | 12.87 | P002 | gondola | G4R-S2-18 | caramel_crest | put_back | group | right slot | 56 (56) |  |  |  | crunchly_classic | P001 |
| 11004 | 21.13 | P004 | cooler | D2-S1-7 | clearwell_water | paid |  | hand or item detected | 30 (30) |  |  |  | morning_hoops | P005 |
| 11004 | 23.5 | P002 | gondola | G4R-S2-43 | cocoa_crest | paid | group | alert | 42 (42) |  |  |  | cocoa_crest | P002 |
| 11004 | 23.67 | P005 | gondola | G4R-S3-9 | torqueline_10w30 | paid |  | none | 59 (59) |  |  |  | torqueline_10w30 | P005 |
| 11004 | 30.27 | P002 | gondola | G3R-S3-30 | morning_hoops | paid | group scan_dropped | right shopper | 88 (88) |  |  |  | morning_hoops | P001 |
| 11004 | 32.63 | P001 | gondola | G3R-S3-44 | peanut_pilot | concealed |  | right slot | 62 (62) |  |  |  | peanut_pilot | P005 |
| 11004 | 46.77 | P001 | gondola | G3R-S1-18 | voltlink_micro | paid |  | ledger basket | 88 (88) |  |  |  | voltlink_micro | P001 |
| 11005 | 11.63 | P002 | gondola | G1R-S2-40 | ridgeline_original | paid | night group | right slot | 83 (83) |  |  |  | caramel_crest | P001 |
| 11005 | 12.43 | P001 | gondola | G1R-S1-47 | caramel_crest | paid | night group | right slot | 30 (30) |  |  |  | ridgeline_original | P003 |
| 11005 | 15.3 | P003 | gondola | G1R-S4-21 | crunchly_bbq | paid | night | hand or item detected | 99 (99) |  |  |  |  |  |
| 11005 | 19.93 | P004 | gondola | G1R-S2-7 | torqueline_10w30 | concealed | night | right slot | 72 (72) |  |  |  | torqueline_10w30 | P004 |
| 11005 | 21.53 | P005 | gondola | G1R-S4-16 | sierra_nacho | put_back | night | conceal or pay classified | 102 (102) |  |  |  | sierra_nacho | P005 |
| 11005 | 25.27 | P006 | cooler | D3-S4-1 | pacer_glacier | paid | night | hand or item detected | 49 (49) |  |  |  | pacer_glacier | P006 |
| 11005 | 27.03 | P001 | cooler | D12-S2-8 | arcwave_berry | put_back | night group | hand or item detected | 48 (48) |  |  |  | fizzo_zero | P003 |
| 11005 | 27.8 | P003 | cooler | D10-S4-8 | arcwave_original | concealed | night | hand or item detected | 25 (25) |  |  |  | voltlink_usbc | P005 |
| 11005 | 28.9 | P005 | gondola | G2L-S3-22 | voltlink_usbc | paid | night | right slot | 125 (125) |  |  |  | voltlink_usbc | P005 |
| 11005 | 33.57 | P006 | gondola | G1L-S3-25 | voltlink_micro | concealed | night slot_holds_another_product | right sku | 54 (54) |  |  |  | voltlink_usbc | P006 |
| 11005 | 44.1 | P001 | gondola | G1L-S2-19 | peanut_pilot | paid | night group | ledger basket | 67 (67) |  |  |  | peanut_pilot | P001 |
| 11005 | 45.28 | P003 | gondola | G1L-S3-38 | torqueline_10w30 | concealed | night | right slot | 46 (46) |  |  |  | torqueline_10w30 | P003 |
| 11006 | 11.28 | P001 | gondola | G3R-S3-9 | cocoa_crest | paid |  | none | 63 (63) |  |  |  | cocoa_crest | P001 |
| 11006 | 16.7 | P002 | gondola | G4R-S4-11 | cocoa_crest | put_back |  | right slot | 45 (45) |  |  |  | cocoa_crest | P002 |
| 11006 | 16.8 | P004 | gondola | G3R-S2-18 | sierra_ranch | paid | group | none | 90 (90) |  |  |  | sierra_ranch | P004 |
| 11006 | 18.58 | P006 | gondola | G4R-S3-43 | ridgeline_original | paid |  | none | 46 (46) |  |  |  | ridgeline_original | P006 |
| 11006 | 18.88 | P003 | gondola | G4R-S4-1 | cocoa_crest | paid |  | none | 30 (30) |  |  |  | cocoa_crest | P003 |
| 11006 | 21.57 | P007 | gondola | G3R-S3-20 | cocoa_crest | concealed |  | right slot | 47 (47) |  |  |  | voltlink_micro | P005 |
| 11006 | 25.27 | P002 | gondola | G4R-S1-13 | crunchly_classic | paid |  | none | 51 (51) |  |  |  | crunchly_classic | P002 |
| 11006 | 25.27 | P006 | checkout | CK-9 | caramel_crest | paid | scan_dropped | none | 26 (26) |  |  |  | caramel_crest | P006 |
| 11006 | 26.93 | P008 | gondola | G3R-S1-26 | relieva_aceta | paid |  | hand or item detected | 73 (73) |  |  |  |  |  |
| 11006 | 35.4 | P008 | checkout | CK-13 | peanut_pilot | concealed |  | conceal or pay classified | 44 (44) |  |  |  | peanut_pilot | P008 |
| 11007 | 8.63 | P001 | gondola | G2R-S2-40 | peanut_pilot | paid | night | hand or item detected | 37 (37) |  |  |  |  |  |
| 11007 | 14.38 | P004 | gondola | G3L-S2-32 | relieva_aceta | concealed | night | conceal or pay classified | 61 (61) |  |  |  | relieva_aceta | P004 |
| 11007 | 14.98 | P005 | gondola | G3L-S4-31 | morning_hoops | put_back | night | conceal or pay classified | 30 (30) |  |  |  | morning_hoops | P005 |
| 11007 | 15.08 | P002 | cooler | D5-S5-3 | fizzo_zero | paid | night group | hand or item detected | 52 (52) |  |  |  |  |  |
| 11007 | 15.83 | P003 | cooler | D6-S5-4 | fizzo_cola | paid | night group | right slot | 35 (35) |  |  |  | fizzo_zero | P002 |
| 11007 | 16.63 | P006 | checkout | CK-7 | peanut_pilot | paid | night | none | 37 (37) |  |  |  | peanut_pilot | P006 |
| 11007 | 23.3 | P007 | gondola | G1L-S3-36 | torqueline_5w30 | concealed | night | conceal or pay classified | 23 (23) |  |  |  | torqueline_5w30 | P007 |
| 11007 | 23.43 | P002 | gondola | G3L-S4-38 | caramel_crest | concealed | night group | right slot | 14 (14) |  |  |  | caramel_crest | P002 |
| 11007 | 29.77 | P005 | checkout | CK-8 | peanut_pilot | paid | night | none | 34 (34) |  |  |  | peanut_pilot | P005 |
| 11007 | 31.97 | P006 | gondola | G1L-S1-14 | crunchly_bbq | paid | night | none | 46 (46) |  |  |  | crunchly_bbq | P006 |
| 11007 | 34.33 | P007 | cooler | D3-S5-3 | orchard_peach | paid | night | hand or item detected | 78 (78) |  |  |  |  |  |
| 11007 | 35.53 | P006 | gondola | G1L-S1-19 | sierra_nacho | paid | night slot_holds_another_product | right sku | 19 (19) |  |  |  | crunchly_bbq | P006 |
| 11008 | 13.53 | P001 | cooler | D9-S5-1 | fizzo_cherry | put_back | group | hand or item detected | 33 (33) |  |  |  |  |  |
| 11008 | 14.2 | P002 | cooler | D9-S2-8 | fizzo_cherry | paid | group | hand or item detected | 34 (34) |  |  |  |  |  |
| 11008 | 14.37 | P004 | gondola | G3R-S4-41 | ridgeline_original | concealed |  | conceal or pay classified | 72 (72) |  |  |  | ridgeline_original | P004 |
| 11008 | 17.5 | P005 | gondola | G4R-S2-45 | caramel_crest | paid |  | alert | 23 (23) |  |  |  | caramel_crest | P005 |
| 11008 | 19.33 | P003 | gondola | G3R-S2-3 | crunchly_bbq | paid |  | none | 71 (71) |  |  |  | crunchly_bbq | P003 |
| 11008 | 19.83 | P006 | gondola | G4R-S3-41 | ridgeline_original | paid | slot_holds_another_product | right slot | 55 (55) |  |  |  | caramel_crest | P002 |
| 11008 | 21.47 | P005 | checkout | CK-5 | peanut_pilot | concealed |  | conceal or pay classified | 38 (38) |  |  |  | peanut_pilot | P005 |
| 11008 | 22.53 | P001 | gondola | G4L-S4-10 | relieva_ibu | paid | group scan_dropped | none | 52 (52) |  |  |  | relieva_ibu | P001 |
| 11008 | 23.13 | P006 | gondola | G4R-S4-41 | caramel_crest | concealed |  | right slot | 67 (67) |  |  |  | lumen_citrus | P004 |
| 11008 | 31.73 | P003 | checkout | CK-1 | peanut_pilot | paid |  | none | 22 (22) |  |  |  | peanut_pilot | P003 |
| 11008 | 35.23 | P005 | cooler | D10-S2-2 | arcwave_berry | concealed |  | hand or item detected | 82 (82) |  |  |  |  |  |
| 11009 | 12.57 | P002 | checkout | CK-10 | cocoa_crest | concealed |  | conceal or pay classified | 45 (45) |  |  |  | cocoa_crest | P002 |
| 11009 | 13.43 | P001 | gondola | G1L-S4-38 | caramel_crest | concealed |  | conceal or pay classified | 39 (39) |  |  |  | caramel_crest | P001 |
| 11009 | 18.27 | P004 | gondola | G2R-S2-18 | voltlink_usbc | paid | group | right slot | 31 (31) |  |  |  | torqueline_10w30 | P006 |
| 11009 | 19.53 | P006 | gondola | G2R-S4-23 | ridgeline_teriyaki | put_back |  | right slot | 42 (42) |  |  |  | voltlink_usbc | STAFF1 |
| 11009 | 19.63 | P005 | gondola | G2R-S4-3 | oatfield_cookies | paid | group | none | 21 (21) |  |  |  | oatfield_cookies | P005 |
| 11009 | 21.73 | P002 | gondola | G3R-S4-30 | voltlink_usbc | concealed |  | conceal or pay classified | 106 (106) |  |  |  | voltlink_usbc | P002 |
| 11009 | 29 | P006 | gondola | G2R-S4-41 | cocoa_crest | paid |  | ledger basket | 24 (24) |  |  |  | cocoa_crest | P006 |
| 11009 | 29.8 | P004 | checkout | CK-9 | cocoa_crest | paid | group | alert | 39 (39) |  |  |  | cocoa_crest | P004 |
| 11009 | 30.88 | P007 | gondola | G1L-S3-4 | torqueline_5w30 | put_back |  | conceal or pay classified | 32 (32) |  |  |  | torqueline_5w30 | P007 |
| 11009 | 37.57 | P006 | checkout | CK-1 | caramel_crest | paid |  | ledger basket | 42 (42) |  |  |  | caramel_crest | P006 |
| 11009 | 38.63 | P004 | gondola | G2R-S2-53 | oatfield_cookies | paid | group | ledger basket | 58 (58) |  |  |  | oatfield_cookies | P004 |
| 11009 | 41.67 | P004 | gondola | G2R-S2-47 | peanut_pilot | concealed | group | right slot | 39 (39) |  |  |  | ridgeline_original | P004 |
| 11009 | 47.83 | P006 | gondola | G3L-S4-25 | ridgeline_teriyaki | paid |  | ledger basket | 55 (55) |  |  |  | ridgeline_teriyaki | P006 |
| 11009 | 53.73 | P007 | checkout | CK-14 | peanut_pilot | paid |  | right shopper | 45 (45) |  |  |  | peanut_pilot | P005 |
| 11009 | 56.53 | P007 | checkout | CK-8 | cocoa_crest | paid |  | none | 35 (35) |  |  |  | cocoa_crest | P007 |
| 11010 | 15.33 | P005 | gondola | G2R-S2-48 | cocoa_crest | paid | group | right slot | 40 (40) |  |  |  | ridgeline_original | P003 |
| 11010 | 16.27 | P002 | gondola | G2R-S1-11 | relieva_aceta | paid |  | none | 28 (28) |  |  |  | relieva_aceta | P002 |
| 11010 | 17.38 | P003 | gondola | G3L-S3-22 | crunchly_classic | put_back | group slot_holds_another_product | right slot | 44 (44) |  |  |  | sierra_ranch | P003 |
| 11010 | 18.23 | P006 | checkout | CK-2 | peanut_pilot | paid |  | right slot | 31 (31) |  |  |  | peanut_pilot | P006 |
| 11010 | 19.53 | P007 | checkout | CK-7 | peanut_pilot | put_back |  | right slot | 49 (49) |  |  |  | crunchly_classic | STAFF1 |
| 11010 | 20.48 | P001 | gondola | G2L-S4-18 | sierra_nacho | paid |  | alert | 92 (92) |  |  |  | sierra_nacho | P001 |
| 11010 | 28 | P002 | checkout | CK-11 | cocoa_crest | paid |  | none | 35 (35) |  |  |  | cocoa_crest | P002 |
| 11010 | 28.27 | P006 | gondola | G2R-S1-30 | voltlink_usbc | paid |  | right slot | 33 (33) |  |  |  | torqueline_10w30 | STAFF1 |
| 11010 | 29.17 | P008 | gondola | G1R-S2-3 | torqueline_10w30 | concealed |  | conceal or pay classified | 51 (51) |  |  |  | torqueline_10w30 | P008 |
| 11010 | 30.78 | P003 | checkout | CK-8 | peanut_pilot | paid | group | alert | 36 (36) |  |  |  | peanut_pilot | P003 |
| 11010 | 32.63 | P006 | gondola | G3L-S2-56 | oatfield_cookies | paid |  | none | 25 (25) |  |  |  | oatfield_cookies | P006 |
| 11010 | 39.47 | P007 | gondola | G1R-S4-2 | crunchly_classic | paid |  | none | 80 (80) |  |  |  | crunchly_classic | P007 |
| 11011 | 11.93 | P004 | checkout | CK-4 | peanut_pilot | concealed | night | hand or item detected | 35 (35) |  |  |  |  |  |
| 11011 | 13.27 | P002 | gondola | G1L-S4-49 | peanut_pilot | paid | night group | shelf event emitted | 12 (12) |  |  |  |  |  |
| 11011 | 14.4 | P001 | gondola | G1L-S1-16 | crunchly_bbq | paid | night group | right shopper | 35 (35) |  |  |  | crunchly_bbq | P002 |
| 11011 | 14.43 | P005 | gondola | G4R-S2-35 | cocoa_crest | concealed | night | right slot | 40 (40) |  |  |  | cocoa_crest | P005 |
| 11011 | 17.67 | P003 | cooler | D11-S2-8 | arcwave_tropical | put_back | night | hand or item detected | 50 (50) |  |  |  | peanut_pilot | P001 |
| 11011 | 17.67 | P006 | gondola | G4R-S2-44 | cocoa_crest | paid | night | alert | 27 (27) |  |  |  | cocoa_crest | P006 |
| 11011 | 19.03 | P001 | gondola | G1L-S2-8 | cocoa_crest | put_back | night group | none | 39 (39) |  |  |  | cocoa_crest | P001 |
| 11011 | 22.48 | P007 | gondola | G1L-S4-27 | relieva_ibu | paid | night slot_holds_another_product | right sku | 47 (47) |  |  |  | relieva_aceta | P007 |
| 11011 | 28.1 | P006 | cooler | D11-S5-8 | arcwave_original | concealed | night | conceal or pay classified | 43 (43) |  |  |  | arcwave_original | P006 |
| 11011 | 30.8 | P003 | gondola | G4R-S4-37 | peanut_pilot | paid | night | none | 42 (42) |  |  |  | peanut_pilot | P003 |
| 11011 | 35.6 | P001 | cooler | D11-S4-6 | arcwave_original | paid | night group | none | 44 (44) |  |  |  | arcwave_original | P001 |
| 11011 | 38.43 | P003 | checkout | CK-14 | peanut_pilot | put_back | night | conceal or pay classified | 50 (50) |  |  |  | peanut_pilot | P003 |
| 11011 | 40.03 | P007 | gondola | G4R-S3-33 | torqueline_10w30 | concealed | night | hand or item detected | 44 (44) |  |  |  | cocoa_crest | P001 |
| 11011 | 40.33 | P001 | gondola | G4R-S4-8 | cocoa_crest | concealed | night group | right slot | 22 (22) |  |  |  | lumen_citrus | P003 |
| 11011 | 42.9 | P006 | cooler | D12-S5-7 | arcwave_original | concealed | night | hand or item detected | 48 (48) |  |  |  | arcwave_original | P006 |
| 11011 | 46 | P003 | checkout | CK-3 | peanut_pilot | paid | night | right slot | 43 (43) |  |  |  | peanut_pilot | P003 |
| 11011 | 52.83 | P007 | checkout | CK-12 | peanut_pilot | concealed | night | conceal or pay classified | 46 (46) |  |  |  | peanut_pilot | P007 |
| 11012 | 8.27 | P001 | gondola | G2R-S1-31 | torqueline_5w30 | paid | night | none | 48 (48) |  |  |  | torqueline_5w30 | P001 |
| 11012 | 11.18 | P002 | checkout | CK-8 | caramel_crest | paid | night slot_holds_another_product | shelf event emitted | 47 (47) |  |  |  |  |  |
| 11012 | 15.08 | P002 | checkout | CK-1 | cocoa_crest | concealed | night | right slot | 37 (37) |  |  |  | peanut_pilot | P002 |
| 11012 | 20.38 | P003 | gondola | G1R-S3-13 | peanut_pilot | put_back | night | conceal or pay classified | 52 (52) |  |  |  | peanut_pilot | P003 |
| 11012 | 23.68 | P004 | gondola | G1R-S3-22 | voltlink_micro | concealed | night | conceal or pay classified | 74 (74) |  |  |  | voltlink_micro | P004 |
| 11012 | 23.87 | P006 | gondola | G3L-S1-4 | torqueline_5w30 | paid | night | hand or item detected | 47 (47) |  |  |  | voltlink_micro | P003 |
| 11012 | 34.2 | P003 | gondola | G3L-S2-51 | peanut_pilot | paid | night | none | 32 (32) |  |  |  | peanut_pilot | P003 |
| 11012 | 36.27 | P006 | checkout | CK-9 | peanut_pilot | paid | night | none | 36 (36) |  |  |  | peanut_pilot | P006 |
| 11012 | 40.13 | P004 | gondola | G2L-S2-29 | voltlink_micro | concealed | night | conceal or pay classified | 100 (100) |  |  |  | voltlink_micro | P004 |
| 11012 | 45.18 | P006 | gondola | G3L-S1-40 | ridgeline_teriyaki | paid | night | none | 29 (29) |  |  |  | ridgeline_teriyaki | P006 |
| 11012 | 64.48 | P004 | checkout | CK-14 | cocoa_crest | paid | night | alert | 49 (49) |  |  |  | cocoa_crest | P004 |
| 11013 | 8.13 | P002 | checkout | CK-9 | peanut_pilot | paid |  | conceal or pay classified | 36 (36) |  |  |  | peanut_pilot | P002 |
| 11013 | 14.17 | P003 | gondola | G3L-S2-17 | voltlink_micro | put_back |  | hand or item detected | 25 (25) |  |  |  | voltlink_micro | P003 |
| 11013 | 14.3 | P005 | checkout | CK-12 | cocoa_crest | paid | group | none | 39 (39) |  |  |  | cocoa_crest | P005 |
| 11013 | 15.3 | P004 | checkout | CK-1 | peanut_pilot | paid | group | alert | 39 (39) |  |  |  | peanut_pilot | P004 |
| 11013 | 15.83 | P001 | gondola | G1L-S3-2 | torqueline_5w30 | concealed | slot_holds_another_product | right sku | 39 (39) |  |  |  | torqueline_10w30 | P001 |
| 11013 | 18.73 | P006 | gondola | G3L-S1-25 | voltlink_usbc | paid |  | none | 49 (49) |  |  |  | voltlink_usbc | P006 |
| 11013 | 19.1 | P004 | checkout | CK-6 | peanut_pilot | paid | group | shelf event emitted | 41 (41) |  |  |  |  |  |
| 11013 | 20.57 | P007 | gondola | G3L-S2-22 | voltlink_micro | concealed |  | hand or item detected | 32 (32) |  |  |  | voltlink_micro | P007 |
| 11013 | 21.18 | P002 | gondola | G1L-S1-25 | sierra_ranch | paid |  | in view | 0 (0) |  |  |  | voltlink_micro | P001 |
| 11013 | 22.13 | P003 | gondola | G2R-S2-9 | peanut_pilot | paid |  | conceal or pay classified | 35 (35) |  |  |  | peanut_pilot | P003 |
| 11013 | 22.43 | P008 | gondola | G2R-S1-18 | relieva_ibu | paid |  | none | 52 (52) |  |  |  | relieva_ibu | P008 |
| 11013 | 30.4 | P006 | checkout | CK-7 | peanut_pilot | concealed |  | conceal or pay classified | 46 (46) |  |  |  | peanut_pilot | P006 |
| 11013 | 30.63 | P004 | gondola | G2R-S4-22 | ridgeline_original | concealed | group | conceal or pay classified | 65 (65) |  |  |  | ridgeline_original | P004 |
| 11014 | 10 | P001 | gondola | G2L-S3-47 | caramel_crest | put_back | night | right slot | 73 (73) |  |  |  | caramel_crest | P001 |
| 11014 | 20 | P002 | cooler | D4-S4-3 | pacer_punch | paid | night | right shopper | 68 (68) |  |  |  | pacer_punch | P003 |
| 11014 | 20.53 | P003 | cooler | D5-S1-2 | lumen_citrus | concealed | night | hand or item detected | 32 (32) |  |  |  | caramel_crest | STAFF1 |
| 11014 | 23.13 | P005 | gondola | G1R-S1-7 | caramel_crest | concealed | night | right slot | 64 (64) |  |  |  | caramel_crest | P005 |
| 11014 | 26.63 | P001 | checkout | CK-5 | cocoa_crest | paid | night slot_holds_another_product | hand or item detected | 43 (43) |  |  |  | caramel_crest | P001 |
| 11014 | 37.93 | P001 | cooler | D9-S1-9 | fizzo_zero | paid | night scan_dropped | right slot | 42 (42) |  |  |  | fizzo_zero | P001 |
| 11014 | 38.67 | P003 | gondola | G2R-S1-27 | voltlink_usbc | paid | night | alert | 76 (76) |  |  |  | voltlink_usbc | P003 |
| 11015 | 10.03 | P001 | checkout | CK-8 | peanut_pilot | paid | night | hand or item detected | 37 (37) |  |  |  |  |  |
| 11015 | 13.73 | P005 | checkout | CK-9 | peanut_pilot | concealed | night | conceal or pay classified | 37 (37) |  |  |  | peanut_pilot | P005 |
| 11015 | 13.9 | P002 | gondola | G3L-S2-37 | caramel_crest | put_back | night | hand or item detected | 60 (60) |  |  |  | caramel_crest | P002 |
| 11015 | 15.43 | P004 | gondola | G3L-S1-22 | relieva_aceta | paid | night | none | 52 (52) |  |  |  | relieva_aceta | P004 |
| 11015 | 20.08 | P006 | gondola | G3L-S4-18 | peanut_pilot | paid | night | conceal or pay classified | 59 (59) |  |  |  | peanut_pilot | P006 |
| 11015 | 20.2 | P007 | gondola | G2L-S3-54 | oatfield_cookies | paid | night | right slot | 97 (97) |  |  |  | voltlink_usbc | P001 |
| 11015 | 20.53 | P001 | gondola | G2R-S1-26 | voltlink_usbc | paid | night | right slot | 71 (71) |  |  |  | oatfield_cookies | P007 |
| 11015 | 24.03 | P001 | gondola | G2R-S4-10 | peanut_pilot | put_back | night | right slot | 52 (52) |  |  |  | sierra_ranch | P006 |
| 11015 | 25.37 | P006 | gondola | G3L-S1-46 | ridgeline_original | paid | night | hand or item detected | 24 (24) |  |  |  | ridgeline_teriyaki | P007 |
| 11015 | 27.13 | P002 | checkout | CK-5 | peanut_pilot | paid | night | none | 39 (39) |  |  |  | peanut_pilot | P002 |
| 11015 | 29.47 | P007 | gondola | G3L-S2-15 | peanut_pilot | concealed | night | right slot | 13 (13) |  |  |  | voltlink_usbc | P007 |
| 11015 | 30.2 | P006 | gondola | G3L-S2-48 | caramel_crest | paid | night | conceal or pay classified | 25 (25) |  |  |  | caramel_crest | P006 |
| 11015 | 40.28 | P002 | gondola | G2L-S2-21 | relieva_aceta | put_back | night | right slot | 49 (49) |  |  |  | voltlink_usbc | P002 |
| 11015 | 46.68 | P007 | gondola | G3L-S1-24 | voltlink_usbc | concealed | night | conceal or pay classified | 53 (53) |  |  |  | voltlink_usbc | P007 |
| 11015 | 46.77 | P002 | gondola | G2L-S3-14 | peanut_pilot | paid | night | none | 53 (53) |  |  |  | peanut_pilot | P002 |
| 11016 | 12.1 | P003 | checkout | CK-10 | cocoa_crest | concealed |  | conceal or pay classified | 46 (46) |  |  |  | cocoa_crest | P003 |
| 11016 | 13.7 | P001 | cooler | D9-S3-1 | fizzo_cherry | put_back |  | right slot | 46 (46) |  |  |  | fizzo_zero | P002 |
| 11016 | 16.53 | P004 | gondola | G3R-S3-8 | caramel_crest | paid |  | none | 81 (81) |  |  |  | caramel_crest | P004 |
| 11016 | 17.63 | P002 | cooler | D8-S4-2 | fizzo_cola | paid |  | hand or item detected | 25 (25) |  |  |  | peanut_pilot | P005 |
| 11016 | 19.3 | P005 | gondola | G3R-S1-39 | peanut_pilot | put_back |  | right slot | 108 (108) |  |  |  | ridgeline_original | P004 |
| 11016 | 22.48 | P001 | cooler | D6-S5-8 | fizzo_cola | paid |  | none | 49 (49) |  |  |  | fizzo_cola | P001 |
| 11016 | 25.93 | P003 | cooler | D10-S3-2 | arcwave_berry | concealed |  | hand or item detected | 57 (57) |  |  |  | orchard_peach | P006 |
| 11016 | 26.53 | P006 | cooler | D14-S3-3 | orchard_peach | paid |  | hand or item detected | 27 (27) |  |  |  |  |  |
| 11016 | 33.3 | P006 | gondola | G4L-S4-3 | torqueline_5w30 | concealed |  | conceal or pay classified | 45 (45) |  |  |  | torqueline_5w30 | P006 |
| 11016 | 33.57 | P005 | cooler | D4-S2-3 | pacer_punch | paid |  | hand or item detected | 69 (69) |  |  |  | pacer_punch | P005 |
| 11016 | 40.23 | P006 | gondola | G3R-S4-17 | relieva_aceta | concealed |  | conceal or pay classified | 73 (73) |  |  |  | relieva_aceta | P006 |
| 11017 | 14.7 | P002 | gondola | G4L-S1-11 | caramel_crest | paid | night | right slot | 29 (29) |  |  |  | caramel_crest | P002 |
| 11017 | 15.43 | P001 | cooler | D4-S5-8 | orchard_peach | put_back | night | hand or item detected | 31 (31) |  |  |  | fizzo_zero | P001 |
| 11017 | 16.2 | P003 | gondola | G1L-S3-36 | torqueline_5w30 | put_back | night slot_holds_another_product | right sku | 26 (26) |  |  |  | torqueline_10w30 | P003 |
| 11017 | 18.67 | P004 | gondola | G1L-S2-34 | caramel_crest | concealed | night | conceal or pay classified | 9 (9) |  |  |  | caramel_crest | P004 |
| 11017 | 19.6 | P006 | gondola | G3R-S1-54 | oatfield_cookies | put_back | night group | hand or item detected | 80 (80) |  |  |  | oatfield_cookies | P006 |
| 11017 | 20.68 | P002 | cooler | D12-S5-3 | arcwave_berry | paid | night | right slot | 49 (49) |  |  |  | arcwave_berry | P002 |
| 11017 | 23.83 | P005 | cooler | D10-S5-6 | arcwave_tropical | concealed | night | hand or item detected | 31 (31) |  |  |  |  |  |
| 11017 | 25.67 | P001 | cooler | D11-S5-1 | arcwave_original | paid | night | right shopper | 53 (53) |  |  |  | arcwave_original | P005 |
| 11017 | 26.33 | P006 | gondola | G4L-S1-42 | peanut_pilot | put_back | night group best_camera_knocked | right slot | 50 (50) |  |  |  | peanut_pilot | P006 |
| 11017 | 32.63 | P003 | cooler | D4-S2-4 | pacer_glacier | paid | night | hand or item detected | 70 (70) |  |  |  | pacer_glacier | P003 |
| 11017 | 35.57 | P004 | gondola | G4L-S3-12 | peanut_pilot | paid | night | alert | 65 (65) |  |  |  | peanut_pilot | P004 |
| 11017 | 38.38 | P006 | cooler | D6-S3-1 | lumen_citrus | paid | night group | right slot | 44 (44) |  |  |  | lumen_citrus | P006 |
| 11017 | 44.13 | P003 | gondola | G4R-S1-20 | crunchly_bbq | paid | night | right shopper | 50 (50) |  |  |  | crunchly_bbq | P007 |
| 11017 | 44.97 | P006 | cooler | D10-S5-4 | arcwave_tropical | paid | night group | right slot | 78 (78) |  |  |  | arcwave_tropical | P006 |
| 11017 | 49.2 | P006 | cooler | D10-S2-6 | arcwave_tropical | concealed | night group | conceal or pay classified | 88 (88) |  |  |  | arcwave_tropical | P006 |
| 11018 | 11.77 | P001 | gondola | G1R-S4-12 | sierra_ranch | paid | night slot_holds_another_product | right slot | 82 (82) |  |  |  | oatfield_cookies | P002 |
| 11018 | 18.48 | P002 | gondola | G1R-S3-15 | cocoa_crest | concealed | night slot_holds_another_product | right slot | 44 (44) |  |  |  | voltlink_micro | P002 |
| 11018 | 18.67 | P003 | gondola | G1R-S1-21 | ridgeline_original | paid | night | shelf event emitted | 92 (92) |  |  |  |  |  |
| 11018 | 22.5 | P004 | gondola | G1R-S3-17 | voltlink_micro | put_back | night | shelf event emitted | 71 (71) |  |  |  |  |  |
| 11018 | 29.33 | P003 | gondola | G4L-S4-17 | relieva_aceta | paid | night | none | 21 (21) |  |  |  | relieva_aceta | P003 |
| 11018 | 37.43 | P002 | gondola | G4L-S1-29 | relieva_aceta | paid | night | none | 40 (40) |  |  |  | relieva_aceta | P002 |
| 11018 | 39.4 | P004 | gondola | G3R-S1-54 | oatfield_cookies | paid | night | conceal or pay classified | 78 (78) |  |  |  | oatfield_cookies | P004 |
| 11019 | 15.7 | P001 | gondola | G1L-S3-20 | relieva_ibu | concealed |  | conceal or pay classified | 53 (53) |  |  |  | relieva_ibu | P001 |
| 11019 | 15.93 | P002 | gondola | G1L-S4-16 | voltlink_usbc | paid |  | none | 42 (42) |  |  |  | voltlink_usbc | P002 |
| 11019 | 17.37 | P004 | cooler | D3-S2-2 | pacer_glacier | paid | group | hand or item detected | 81 (81) |  |  |  |  |  |
| 11019 | 17.57 | P003 | cooler | D2-S4-4 | clearwell_water | paid | group | hand or item detected | 58 (58) |  |  |  | clearwell_water | P003 |
| 11019 | 20.37 | P005 | cooler | D3-S4-4 | pacer_orange | put_back |  | right slot | 54 (54) |  |  |  | pacer_orange | P005 |
| 11019 | 28.93 | P001 | gondola | G4L-S1-7 | caramel_crest | paid |  | none | 44 (44) |  |  |  | caramel_crest | P001 |
| 11019 | 31.4 | P002 | gondola | G3R-S3-32 | morning_hoops | paid |  | none | 51 (51) |  |  |  | morning_hoops | P002 |
| 11019 | 35.1 | P005 | gondola | G1L-S4-46 | caramel_crest | paid |  | right slot | 28 (28) |  |  |  | caramel_crest | P005 |
| 11019 | 54.38 | P005 | cooler | D14-S2-5 | orchard_peach | paid | scan_dropped | hand or item detected | 25 (25) |  |  |  |  |  |
| 11020 | 12.23 | P002 | gondola | G4L-S3-21 | ridgeline_original | put_back | night | conceal or pay classified | 45 (45) |  |  |  | ridgeline_original | P002 |
| 11020 | 12.97 | P001 | gondola | G2L-S4-8 | crunchly_classic | paid | night slot_holds_another_product | shelf event emitted | 62 (62) |  |  |  |  |  |
| 11020 | 17.3 | P001 | gondola | G1R-S3-3 | peanut_pilot | put_back | night | right slot | 45 (45) |  |  |  | peanut_pilot | P001 |
| 11020 | 20.47 | P002 | gondola | G4L-S1-54 | oatfield_cookies | paid | night | right shopper | 23 (23) |  |  |  | oatfield_cookies | P003 |
| 11020 | 21.83 | P003 | gondola | G3R-S4-38 | torqueline_5w30 | paid | night | right slot | 61 (61) |  |  |  | caramel_crest | P002 |
| 11020 | 22.13 | P005 | gondola | G4L-S1-1 | peanut_pilot | paid | night | hand or item detected | 2 (2) |  |  |  | crunchly_bbq | P007 |
| 11020 | 24 | P007 | gondola | G4L-S3-6 | peanut_pilot | concealed | night | conceal or pay classified | 21 (21) |  |  |  | peanut_pilot | P007 |
| 11020 | 26.68 | P002 | checkout | CK-12 | cocoa_crest | paid | night | right slot | 35 (35) |  |  |  | voltlink_micro | STAFF1 |
| 11020 | 26.83 | P001 | gondola | G1R-S1-37 | caramel_crest | paid | night | right slot | 44 (44) |  |  |  | oatfield_cookies | P001 |
| 11020 | 28.77 | P008 | gondola | G1R-S1-2 | oatfield_cookies | paid | night | hand or item detected | 59 (59) |  |  |  | oatfield_cookies | P005 |
| 11020 | 28.83 | P004 | gondola | G3R-S1-16 | voltlink_usbc | paid | night | right slot | 22 (22) |  |  |  | voltlink_usbc | P004 |
| 11020 | 28.88 | P003 | checkout | CK-14 | cocoa_crest | paid | night | right slot | 30 (30) |  |  |  | cocoa_crest | P003 |
| 11020 | 32.2 | P005 | gondola | G2L-S1-14 | peanut_pilot | paid | night | hand or item detected | 64 (64) |  |  |  | sierra_ranch | P005 |
| 11020 | 39.9 | P004 | checkout | CK-13 | cocoa_crest | concealed | night | conceal or pay classified | 48 (48) |  |  |  | cocoa_crest | P004 |
| 11020 | 43.53 | P008 | checkout | CK-8 | peanut_pilot | paid | night | none | 35 (35) |  |  |  | peanut_pilot | P008 |
