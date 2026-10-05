# Benchmark, dev split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

6 clips (7001, 7002, 7003, 7004, 7005, 7006), 421 s of sim time, 37 shoppers (15 thieves, 22 honest), 70 picks, 20 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-05 16:09, commit 28989f3 plus uncommitted changes in src.

## Scorecard

| Metric | Value |
|---|---|
| Theft recall, alert tier | 0.0% (0/20) |
| Theft recall, alert or review | 85.0% (17/20) |
| Stolen items alerted with the right SKU | 0/20 |
| Alert precision | n/a (0/0) |
| False alerts on honest shoppers | 0 (of 22 honest shoppers); on nobody: 0 |
| Reviews on honest shoppers | 3 (of 15 reviews) |
| False alerts per hour | 0.0 |
| Pick recall | 95.7% (67/70) |
| Pick precision | 84.8% (of 79 PICK events) |
| Right SKU, of paired picks | 94.0%; of all true picks 90.0% |
| Right slot, of paired picks | 88.1% |
| Right shopper, of paired picks | 94.0% |
| Time to alert after concealment | n/a |
| Store-wide identities per real shopper | 1.27 (47 ids for 37 shoppers) |
| Identities sitting on one shopper | mean 1.08, max 2; shoppers never tracked 0; ids covering two shoppers 6 |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 70 picks, 27 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 70 | 0 | 70 |
| frame reached pipeline | 70 | 0 | 70 |
| hand or item detected | 70 | 8 | 62 |
| shelf event emitted | 62 | 0 | 67 |
| right slot | 62 | 8 | 59 |
| right sku | 54 | 0 | 63 |
| associated to a shopper | 54 | 0 | 67 |
| right shopper | 54 | 0 | 63 |
| conceal or pay classified | 54 | 17 | 49 |
| ledger basket | 37 | 2 | 61 |
| alert | 35 | 8 | 37 |

### Outcome concealed: 20 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 20 | 0 | 20 |
| frame reached pipeline | 20 | 0 | 20 |
| hand or item detected | 20 | 2 | 18 |
| shelf event emitted | 18 | 0 | 19 |
| right slot | 18 | 1 | 18 |
| right sku | 17 | 0 | 19 |
| associated to a shopper | 17 | 0 | 19 |
| right shopper | 17 | 0 | 19 |
| conceal or pay classified | 17 | 17 | 0 |
| ledger basket | 0 | 0 | 15 |
| alert | 0 | 0 | 0 |

### Outcome paid: 41 picks, 21 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 41 | 0 | 41 |
| frame reached pipeline | 41 | 0 | 41 |
| hand or item detected | 41 | 6 | 35 |
| shelf event emitted | 35 | 0 | 39 |
| right slot | 35 | 5 | 34 |
| right sku | 30 | 0 | 36 |
| associated to a shopper | 30 | 0 | 39 |
| right shopper | 30 | 0 | 36 |
| conceal or pay classified | 30 | 0 | 41 |
| ledger basket | 30 | 2 | 38 |
| alert | 28 | 7 | 30 |

### Outcome put_back: 9 picks, 6 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 9 | 0 | 9 |
| frame reached pipeline | 9 | 0 | 9 |
| hand or item detected | 9 | 0 | 9 |
| shelf event emitted | 9 | 0 | 9 |
| right slot | 9 | 2 | 7 |
| right sku | 7 | 0 | 8 |
| associated to a shopper | 7 | 0 | 9 |
| right shopper | 7 | 0 | 8 |
| conceal or pay classified | 7 | 0 | 8 |
| ledger basket | 7 | 0 | 8 |
| alert | 7 | 1 | 7 |

### Zone checkout: 9 picks, 6 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 9 | 0 | 9 |
| frame reached pipeline | 9 | 0 | 9 |
| hand or item detected | 9 | 1 | 8 |
| shelf event emitted | 8 | 0 | 9 |
| right slot | 8 | 0 | 9 |
| right sku | 8 | 0 | 9 |
| associated to a shopper | 8 | 0 | 9 |
| right shopper | 8 | 0 | 9 |
| conceal or pay classified | 8 | 2 | 7 |
| ledger basket | 6 | 0 | 8 |
| alert | 6 | 0 | 7 |

### Zone cooler: 15 picks, 6 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 15 | 0 | 15 |
| frame reached pipeline | 15 | 0 | 15 |
| hand or item detected | 15 | 5 | 10 |
| shelf event emitted | 10 | 0 | 14 |
| right slot | 10 | 2 | 12 |
| right sku | 8 | 0 | 13 |
| associated to a shopper | 8 | 0 | 14 |
| right shopper | 8 | 0 | 14 |
| conceal or pay classified | 8 | 1 | 12 |
| ledger basket | 7 | 1 | 13 |
| alert | 6 | 0 | 11 |

### Zone gondola: 46 picks, 15 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 46 | 0 | 46 |
| frame reached pipeline | 46 | 0 | 46 |
| hand or item detected | 46 | 2 | 44 |
| shelf event emitted | 44 | 0 | 44 |
| right slot | 44 | 6 | 38 |
| right sku | 38 | 0 | 41 |
| associated to a shopper | 38 | 0 | 44 |
| right shopper | 38 | 0 | 40 |
| conceal or pay classified | 38 | 14 | 30 |
| ledger basket | 24 | 1 | 40 |
| alert | 23 | 8 | 19 |

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

## Detection detail (independent counts over all true picks)

| | picks |
|---|---|
| item detected | 0 |
| right sku detected | 0 |
| person detected | 0 |
| hand detected | 62 |
| right fixture | 67 |
| picks with item frames | 70 |
| picks with item frames reached | 70 |

## Per clip

| clip | cameras | shoppers | thieves | picks | PICK events | stolen | alerted | alerts | false | ids | wall s (edge + pipeline) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7001 | 20 | 7 | 2 | 13 | 12 | 2 | 0 | 0 | 0 | 9 |  + 181.7 |
| 7002 | 20 | 5 | 3 | 9 | 9 | 5 | 0 | 0 | 0 | 6 |  + 206.2 |
| 7003 | 18 | 6 | 2 | 10 | 10 | 2 | 0 | 0 | 0 | 8 |  + 190.9 |
| 7004 | 20 | 4 | 1 | 9 | 10 | 2 | 0 | 0 | 0 | 5 |  + 216.9 |
| 7005 | 20 | 8 | 5 | 14 | 19 | 6 | 0 | 0 | 0 | 10 |  + 204.6 |
| 7006 | 20 | 7 | 2 | 15 | 19 | 3 | 0 | 0 | 0 | 9 |  + 192.7 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7001 | 14.23 | P001 | gondola | G1L-S2-17 | peanut_pilot | concealed | conceal or pay classified | 46 (46) |  |  |  | peanut_pilot | P001 |
| 7001 | 14.83 | P002 | gondola | G2R-S4-33 | cocoa_crest | paid | none | 23 (23) |  |  |  | cocoa_crest | P002 |
| 7001 | 18.27 | P003 | gondola | G4R-S2-39 | peanut_pilot | concealed | conceal or pay classified | 51 (51) |  |  |  | peanut_pilot | P003 |
| 7001 | 22.9 | P002 | gondola | G4L-S4-30 | voltlink_micro | paid | right slot | 37 (37) |  |  |  | sierra_nacho | P005 |
| 7001 | 26.47 | P004 | cooler | D10-S2-5 | arcwave_tropical | paid | hand or item detected | 13 (13) |  |  |  | arcwave_tropical | P004 |
| 7001 | 29.07 | P005 | cooler | D9-S2-7 | lumen_citrus | put_back | none | 23 (23) |  |  |  | lumen_citrus | P005 |
| 7001 | 31.27 | P007 | gondola | G3R-S1-39 | caramel_crest | put_back | alert | 93 (93) |  |  |  | caramel_crest | P007 |
| 7001 | 32.93 | P006 | cooler | D8-S4-3 | lumen_citrus | paid | hand or item detected | 18 (18) |  |  |  | lumen_citrus | P006 |
| 7001 | 36.23 | P003 | gondola | G1L-S2-9 | peanut_pilot | paid | ledger basket | 51 (51) |  |  |  | peanut_pilot | P003 |
| 7001 | 36.73 | P004 | checkout | CK-6 | cocoa_crest | paid | hand or item detected | 43 (43) |  |  |  | cocoa_crest | P004 |
| 7001 | 37.3 | P007 | gondola | G4L-S1-46 | caramel_crest | paid | alert | 31 (31) |  |  |  | caramel_crest | P007 |
| 7001 | 41 | P005 | gondola | G3L-S3-9 | crunchly_classic | paid | right slot | 36 (36) |  |  |  | ridgeline_original | P007 |
| 7001 | 46.33 | P005 | gondola | G3L-S4-42 | peanut_pilot | paid | hand or item detected | 15 (15) |  |  |  |  |  |
| 7002 | 9.5 | P001 | gondola | G2L-S2-34 | torqueline_10w30 | put_back | none | 51 (51) |  |  |  | torqueline_10w30 | P001 |
| 7002 | 14.13 | P002 | gondola | G4L-S3-34 | cocoa_crest | paid | none | 38 (38) |  |  |  | cocoa_crest | P002 |
| 7002 | 23.3 | P004 | gondola | G3R-S4-23 | voltlink_micro | paid | alert | 61 (61) |  |  |  | voltlink_micro | P004 |
| 7002 | 23.6 | P003 | gondola | G3R-S4-10 | relieva_aceta | concealed | conceal or pay classified | 25 (25) |  |  |  | relieva_aceta | P003 |
| 7002 | 24.3 | P001 | cooler | D8-S1-9 | fizzo_zero | paid | none | 10 (10) |  |  |  | fizzo_zero | P001 |
| 7002 | 29.53 | P005 | gondola | G3L-S4-11 | cocoa_crest | concealed | conceal or pay classified | 52 (52) |  |  |  | cocoa_crest | P005 |
| 7002 | 30.43 | P004 | cooler | D11-S2-8 | arcwave_tropical | concealed | hand or item detected | 11 (11) |  |  |  | arcwave_tropical | P004 |
| 7002 | 40.17 | P005 | gondola | G3R-S1-9 | caramel_crest | concealed | conceal or pay classified | 44 (44) |  |  |  | caramel_crest | P005 |
| 7002 | 45.93 | P004 | gondola | G2R-S1-28 | voltlink_usbc | concealed | conceal or pay classified | 37 (37) |  |  |  | voltlink_usbc | P004 |
| 7003 | 7.7 | P001 | checkout | CK-4 | peanut_pilot | put_back | none | 52 (52) |  |  |  | peanut_pilot | P001 |
| 7003 | 16.47 | P002 | gondola | G3R-S1-11 | peanut_pilot | concealed | conceal or pay classified | 47 (47) |  |  |  | peanut_pilot | P002 |
| 7003 | 19.43 | P001 | gondola | G3R-S4-18 | relieva_ibu | paid | none | 45 (45) |  |  |  | relieva_ibu | P001 |
| 7003 | 23.8 | P003 | gondola | G2R-S1-3 | torqueline_5w30 | concealed | conceal or pay classified | 37 (37) |  |  |  | torqueline_5w30 | P003 |
| 7003 | 27.07 | P004 | gondola | G4L-S3-16 | caramel_crest | put_back | none | 39 (39) |  |  |  | caramel_crest | P004 |
| 7003 | 32.07 | P005 | gondola | G4L-S2-6 | crunchly_classic | paid | none | 64 (64) |  |  |  | crunchly_classic | P005 |
| 7003 | 36.9 | P006 | gondola | G4L-S4-26 | voltlink_usbc | paid | none | 46 (46) |  |  |  | voltlink_usbc | P006 |
| 7003 | 38.23 | P004 | cooler | D13-S2-6 | orchard_lemon | paid | hand or item detected | 30 (30) |  |  |  | orchard_lemon | P004 |
| 7003 | 41.5 | P005 | checkout | CK-4 | peanut_pilot | paid | none | 20 (20) |  |  |  | peanut_pilot | P005 |
| 7003 | 44.53 | P006 | gondola | G3L-S2-56 | oatfield_cookies | paid | none | 17 (17) |  |  |  | oatfield_cookies | P006 |
| 7004 | 13 | P001 | gondola | G1R-S2-4 | torqueline_5w30 | put_back | none | 32 (32) |  |  |  | torqueline_5w30 | P001 |
| 7004 | 14.73 | P002 | gondola | G1R-S4-7 | sierra_nacho | paid | none | 68 (68) |  |  |  | sierra_nacho | P002 |
| 7004 | 23.2 | P001 | gondola | G2R-S2-23 | voltlink_micro | paid | none | 82 (82) |  |  |  | voltlink_micro | P001 |
| 7004 | 23.43 | P003 | cooler | D8-S3-4 | fizzo_cherry | paid | none | 27 (27) |  |  |  | fizzo_cherry | P003 |
| 7004 | 29.5 | P001 | cooler | D4-S2-1 | pacer_glacier | paid | right slot | 57 (57) |  |  |  | pacer_punch | P001 |
| 7004 | 30.77 | P004 | gondola | G1L-S1-11 | crunchly_sco | paid | alert | 59 (59) |  |  |  | crunchly_sco | P004 |
| 7004 | 39 | P004 | gondola | G1R-S2-10 | relieva_aceta | concealed | conceal or pay classified | 43 (43) |  |  |  | relieva_aceta | P004 |
| 7004 | 43.47 | P001 | checkout | CK-10 | cocoa_crest | paid | none | 38 (38) |  |  |  | cocoa_crest | P001 |
| 7004 | 58.83 | P004 | gondola | G3L-S4-19 | caramel_crest | concealed | conceal or pay classified | 20 (20) |  |  |  | caramel_crest | P004 |
| 7005 | 8.57 | P001 | checkout | CK-2 | peanut_pilot | concealed | conceal or pay classified | 43 (43) |  |  |  | peanut_pilot | P001 |
| 7005 | 17.93 | P002 | cooler | D3-S2-8 | pacer_orange | paid | none | 59 (59) |  |  |  | pacer_orange | P002 |
| 7005 | 19.43 | P003 | cooler | D2-S2-7 | clearwell_water | concealed | conceal or pay classified | 48 (48) |  |  |  | clearwell_water | P003 |
| 7005 | 20.5 | P005 | gondola | G1R-S3-49 | peanut_pilot | paid | right slot | 32 (32) |  |  |  | peanut_pilot | P005 |
| 7005 | 23.17 | P004 | cooler | D2-S4-3 | clearwell_water | paid | hand or item detected | 56 (56) |  |  |  |  |  |
| 7005 | 25.43 | P006 | gondola | G3R-S1-19 | voltlink_micro | paid | alert | 52 (52) |  |  |  | voltlink_micro | P006 |
| 7005 | 27.7 | P002 | gondola | G4R-S4-3 | cocoa_crest | paid | none | 26 (26) |  |  |  | cocoa_crest | P002 |
| 7005 | 28.87 | P007 | gondola | G4L-S4-14 | relieva_ibu | concealed | hand or item detected | 26 (26) |  |  |  |  |  |
| 7005 | 29.4 | P005 | gondola | G2R-S4-23 | ridgeline_original | concealed | conceal or pay classified | 42 (42) |  |  |  | ridgeline_original | P005 |
| 7005 | 30.93 | P006 | gondola | G3R-S4-29 | voltlink_micro | concealed | conceal or pay classified | 47 (47) |  |  |  | voltlink_micro | P006 |
| 7005 | 35.97 | P002 | checkout | CK-9 | cocoa_crest | paid | none | 38 (38) |  |  |  | cocoa_crest | P002 |
| 7005 | 36.87 | P008 | cooler | D5-S3-4 | lumen_citrus | put_back | right slot | 26 (26) |  |  |  | lumen_citrus | P008 |
| 7005 | 37.47 | P006 | gondola | G3R-S4-1 | torqueline_10w30 | concealed | conceal or pay classified | 49 (49) |  |  |  | torqueline_10w30 | P006 |
| 7005 | 49.9 | P008 | gondola | G4R-S3-20 | relieva_ibu | paid | alert | 30 (30) |  |  |  | relieva_ibu | P008 |
| 7006 | 11.6 | P002 | gondola | G3R-S3-46 | peanut_pilot | paid | right slot | 37 (37) |  |  |  | peanut_pilot | P003 |
| 7006 | 13.73 | P003 | gondola | G3R-S4-42 | ridgeline_original | put_back | right slot | 75 (75) |  |  |  | peanut_pilot | P005 |
| 7006 | 15.1 | P001 | gondola | G1L-S2-14 | peanut_pilot | paid | alert | 44 (44) |  |  |  | peanut_pilot | P001 |
| 7006 | 21.17 | P005 | checkout | CK-12 | cocoa_crest | paid | none | 35 (35) |  |  |  | cocoa_crest | P005 |
| 7006 | 21.9 | P004 | gondola | G2L-S3-30 | relieva_ibu | put_back | none | 52 (52) |  |  |  | relieva_ibu | P004 |
| 7006 | 23 | P002 | gondola | G1L-S3-39 | torqueline_5w30 | concealed | conceal or pay classified | 23 (23) |  |  |  | torqueline_5w30 | P002 |
| 7006 | 23.27 | P003 | checkout | CK-1 | cocoa_crest | paid | none | 33 (33) |  |  |  | cocoa_crest | P003 |
| 7006 | 25.93 | P001 | cooler | D4-S3-2 | pacer_orange | paid | ledger basket | 58 (58) |  |  |  | pacer_orange | P001 |
| 7006 | 27.7 | P006 | gondola | G4L-S3-24 | ridgeline_teriyaki | paid | alert | 30 (30) |  |  |  | ridgeline_teriyaki | P006 |
| 7006 | 34.83 | P004 | gondola | G2R-S1-19 | relieva_ibu | paid | none | 33 (33) |  |  |  | relieva_ibu | P004 |
| 7006 | 36.07 | P007 | cooler | D4-S3-3 | pacer_orange | paid | none | 62 (62) |  |  |  | pacer_orange | P007 |
| 7006 | 37.57 | P006 | checkout | CK-9 | peanut_pilot | concealed | conceal or pay classified | 39 (39) |  |  |  | peanut_pilot | P006 |
| 7006 | 39.83 | P002 | gondola | G1L-S4-12 | peanut_pilot | concealed | right slot | 39 (39) |  |  |  | peanut_pilot | P002 |
| 7006 | 42.57 | P004 | gondola | G3L-S4-47 | caramel_crest | paid | none | 4 (4) |  |  |  | caramel_crest | P004 |
| 7006 | 51.73 | P004 | cooler | D4-S2-1 | pacer_orange | paid | none | 57 (57) |  |  |  | pacer_orange | P004 |
