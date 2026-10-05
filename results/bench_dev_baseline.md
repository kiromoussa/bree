# Benchmark, dev split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

6 clips (7001, 7002, 7003, 7004, 7005, 7006), 421 s of sim time, 37 shoppers (15 thieves, 22 honest), 70 picks, 20 stolen items. Runner `bree.sim.bench:run_pipeline`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-05 10:40, commit 19f51dc plus uncommitted changes in src.

## Scorecard

| Metric | Value |
|---|---|
| Theft recall, alert tier | 0.0% (0/20) |
| Theft recall, alert or review | 0.0% (0/20) |
| Stolen items alerted with the right SKU | 0/20 |
| Alert precision | n/a (0/0) |
| False alerts on honest shoppers | 0 (of 22 honest shoppers); on nobody: 0 |
| Reviews on honest shoppers | 0 (of 0 reviews) |
| False alerts per hour | 0.0 |
| Pick recall | 2.9% (2/70) |
| Pick precision | 100.0% (of 2 PICK events) |
| Right SKU, of paired picks | 50.0%; of all true picks 1.4% |
| Right slot, of paired picks | 50.0% |
| Right shopper, of paired picks | 100.0% |
| Time to alert after concealment | n/a |
| Store-wide identities per real shopper | 4.0 (148 ids for 37 shoppers) |
| Identities sitting on one shopper | mean 3.19, max 8; shoppers never tracked 1; ids covering two shoppers 93 |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 70 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 70 | 0 | 70 |
| frame reached pipeline | 70 | 0 | 70 |
| hand or item detected | 70 | 21 | 49 |
| shelf event emitted | 49 | 47 | 2 |
| right slot | 2 | 1 | 1 |
| right sku | 1 | 1 | 1 |
| associated to a shopper | 0 | 0 | 2 |
| right shopper | 0 | 0 | 2 |
| conceal or pay classified | 0 | 0 | 1 |
| ledger basket | 0 | 0 | 50 |
| alert | 0 | 0 | 50 |

### Outcome concealed: 20 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 20 | 0 | 20 |
| frame reached pipeline | 20 | 0 | 20 |
| hand or item detected | 20 | 11 | 9 |
| shelf event emitted | 9 | 9 | 0 |
| right slot | 0 | 0 | 0 |
| right sku | 0 | 0 | 0 |
| associated to a shopper | 0 | 0 | 0 |
| right shopper | 0 | 0 | 0 |
| conceal or pay classified | 0 | 0 | 0 |
| ledger basket | 0 | 0 | 0 |
| alert | 0 | 0 | 0 |

### Outcome paid: 41 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 41 | 0 | 41 |
| frame reached pipeline | 41 | 0 | 41 |
| hand or item detected | 41 | 8 | 33 |
| shelf event emitted | 33 | 32 | 1 |
| right slot | 1 | 0 | 1 |
| right sku | 1 | 1 | 0 |
| associated to a shopper | 0 | 0 | 1 |
| right shopper | 0 | 0 | 1 |
| conceal or pay classified | 0 | 0 | 1 |
| ledger basket | 0 | 0 | 41 |
| alert | 0 | 0 | 41 |

### Outcome put_back: 9 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 9 | 0 | 9 |
| frame reached pipeline | 9 | 0 | 9 |
| hand or item detected | 9 | 2 | 7 |
| shelf event emitted | 7 | 6 | 1 |
| right slot | 1 | 1 | 0 |
| right sku | 0 | 0 | 1 |
| associated to a shopper | 0 | 0 | 1 |
| right shopper | 0 | 0 | 1 |
| conceal or pay classified | 0 | 0 | 0 |
| ledger basket | 0 | 0 | 9 |
| alert | 0 | 0 | 9 |

### Zone checkout: 9 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 9 | 0 | 9 |
| frame reached pipeline | 9 | 0 | 9 |
| hand or item detected | 9 | 0 | 9 |
| shelf event emitted | 9 | 9 | 0 |
| right slot | 0 | 0 | 0 |
| right sku | 0 | 0 | 0 |
| associated to a shopper | 0 | 0 | 0 |
| right shopper | 0 | 0 | 0 |
| conceal or pay classified | 0 | 0 | 1 |
| ledger basket | 0 | 0 | 7 |
| alert | 0 | 0 | 7 |

### Zone cooler: 15 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 15 | 0 | 15 |
| frame reached pipeline | 15 | 0 | 15 |
| hand or item detected | 15 | 1 | 14 |
| shelf event emitted | 14 | 13 | 1 |
| right slot | 1 | 0 | 1 |
| right sku | 1 | 1 | 0 |
| associated to a shopper | 0 | 0 | 1 |
| right shopper | 0 | 0 | 1 |
| conceal or pay classified | 0 | 0 | 0 |
| ledger basket | 0 | 0 | 13 |
| alert | 0 | 0 | 13 |

### Zone gondola: 46 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 46 | 0 | 46 |
| frame reached pipeline | 46 | 0 | 46 |
| hand or item detected | 46 | 20 | 26 |
| shelf event emitted | 26 | 25 | 1 |
| right slot | 1 | 1 | 0 |
| right sku | 0 | 0 | 1 |
| associated to a shopper | 0 | 0 | 1 |
| right shopper | 0 | 0 | 1 |
| conceal or pay classified | 0 | 0 | 0 |
| ledger basket | 0 | 0 | 30 |
| alert | 0 | 0 | 30 |

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
| item detected | 46 |
| right sku detected | 44 |
| person detected | 41 |
| hand detected | 32 |
| right fixture | 2 |
| picks with item frames | 70 |
| picks with item frames reached | 70 |

## Per clip

| clip | cameras | shoppers | thieves | picks | PICK events | stolen | alerted | alerts | false | ids | wall s (edge + pipeline) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7001 | 20 | 7 | 2 | 13 | 1 | 2 | 0 | 0 | 0 | 33 | 391.1 + 814.5 |
| 7002 | 20 | 5 | 3 | 9 | 0 | 5 | 0 | 0 | 0 | 19 | 520.6 + 665.1 |
| 7003 | 18 | 6 | 2 | 10 | 0 | 2 | 0 | 0 | 0 | 17 | 378.7 + 557.3 |
| 7004 | 20 | 4 | 1 | 9 | 1 | 2 | 0 | 0 | 0 | 16 | 224.6 + 581.7 |
| 7005 | 20 | 8 | 5 | 14 | 0 | 6 | 0 | 0 | 0 | 29 | 199.7 + 479.9 |
| 7006 | 20 | 7 | 2 | 15 | 0 | 3 | 0 | 0 | 0 | 34 | 231.7 + 753.1 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7001 | 14.23 | P001 | gondola | G1L-S2-17 | peanut_pilot | concealed | shelf event emitted | 46 (46) | y | y | y |  |  |
| 7001 | 14.83 | P002 | gondola | G2R-S4-33 | cocoa_crest | paid | hand or item detected | 23 (23) |  |  |  |  |  |
| 7001 | 18.27 | P003 | gondola | G4R-S2-39 | peanut_pilot | concealed | shelf event emitted | 51 (51) | y | y | y |  |  |
| 7001 | 22.9 | P002 | gondola | G4L-S4-30 | voltlink_micro | paid | shelf event emitted | 37 (37) | y | y | y |  |  |
| 7001 | 26.47 | P004 | cooler | D10-S2-5 | arcwave_tropical | paid | shelf event emitted | 13 (13) | y | y |  |  |  |
| 7001 | 29.07 | P005 | cooler | D9-S2-7 | lumen_citrus | put_back | shelf event emitted | 23 (23) | y | y | y |  |  |
| 7001 | 31.27 | P007 | gondola | G3R-S1-39 | caramel_crest | put_back | right slot | 93 (93) | y |  | y | caramel_crest | P007 |
| 7001 | 32.93 | P006 | cooler | D8-S4-3 | lumen_citrus | paid | shelf event emitted | 18 (18) | y | y |  |  |  |
| 7001 | 36.23 | P003 | gondola | G1L-S2-9 | peanut_pilot | paid | shelf event emitted | 51 (51) | y | y | y |  |  |
| 7001 | 36.73 | P004 | checkout | CK-6 | cocoa_crest | paid | shelf event emitted | 43 (43) | y | y | y |  |  |
| 7001 | 37.3 | P007 | gondola | G4L-S1-46 | caramel_crest | paid | shelf event emitted | 31 (31) | y | y | y |  |  |
| 7001 | 41 | P005 | gondola | G3L-S3-9 | crunchly_classic | paid | shelf event emitted | 36 (36) | y | y |  |  |  |
| 7001 | 46.33 | P005 | gondola | G3L-S4-42 | peanut_pilot | paid | hand or item detected | 15 (15) |  |  |  |  |  |
| 7002 | 9.5 | P001 | gondola | G2L-S2-34 | torqueline_10w30 | put_back | hand or item detected | 51 (51) |  |  |  |  |  |
| 7002 | 14.13 | P002 | gondola | G4L-S3-34 | cocoa_crest | paid | shelf event emitted | 38 (38) | y |  |  |  |  |
| 7002 | 23.3 | P004 | gondola | G3R-S4-23 | voltlink_micro | paid | shelf event emitted | 61 (61) | y | y | y |  |  |
| 7002 | 23.6 | P003 | gondola | G3R-S4-10 | relieva_aceta | concealed | hand or item detected | 25 (25) |  |  |  |  |  |
| 7002 | 24.3 | P001 | cooler | D8-S1-9 | fizzo_zero | paid | shelf event emitted | 10 (10) | y | y | y |  |  |
| 7002 | 29.53 | P005 | gondola | G3L-S4-11 | cocoa_crest | concealed | hand or item detected | 52 (52) |  |  |  |  |  |
| 7002 | 30.43 | P004 | cooler | D11-S2-8 | arcwave_tropical | concealed | shelf event emitted | 11 (11) | y | y | y |  |  |
| 7002 | 40.17 | P005 | gondola | G3R-S1-9 | caramel_crest | concealed | hand or item detected | 44 (44) |  |  |  |  |  |
| 7002 | 45.93 | P004 | gondola | G2R-S1-28 | voltlink_usbc | concealed | hand or item detected | 37 (37) |  |  |  |  |  |
| 7003 | 7.7 | P001 | checkout | CK-4 | peanut_pilot | put_back | shelf event emitted | 52 (52) | y | y | y |  |  |
| 7003 | 16.47 | P002 | gondola | G3R-S1-11 | peanut_pilot | concealed | hand or item detected | 47 (47) |  |  |  |  |  |
| 7003 | 19.43 | P001 | gondola | G3R-S4-18 | relieva_ibu | paid | shelf event emitted | 45 (45) |  |  | y |  |  |
| 7003 | 23.8 | P003 | gondola | G2R-S1-3 | torqueline_5w30 | concealed | hand or item detected | 37 (37) |  |  |  |  |  |
| 7003 | 27.07 | P004 | gondola | G4L-S3-16 | caramel_crest | put_back | shelf event emitted | 39 (39) | y | y | y |  |  |
| 7003 | 32.07 | P005 | gondola | G4L-S2-6 | crunchly_classic | paid | shelf event emitted | 64 (64) | y | y | y |  |  |
| 7003 | 36.9 | P006 | gondola | G4L-S4-26 | voltlink_usbc | paid | hand or item detected | 46 (46) |  |  | y |  |  |
| 7003 | 38.23 | P004 | cooler | D13-S2-6 | orchard_lemon | paid | hand or item detected | 30 (30) |  |  |  |  |  |
| 7003 | 41.5 | P005 | checkout | CK-4 | peanut_pilot | paid | shelf event emitted | 20 (20) | y | y | y |  |  |
| 7003 | 44.53 | P006 | gondola | G3L-S2-56 | oatfield_cookies | paid | shelf event emitted | 17 (17) |  |  | y |  |  |
| 7004 | 13 | P001 | gondola | G1R-S2-4 | torqueline_5w30 | put_back | shelf event emitted | 32 (32) | y | y |  |  |  |
| 7004 | 14.73 | P002 | gondola | G1R-S4-7 | sierra_nacho | paid | shelf event emitted | 68 (68) | y | y |  |  |  |
| 7004 | 23.2 | P001 | gondola | G2R-S2-23 | voltlink_micro | paid | shelf event emitted | 82 (82) | y | y | y |  |  |
| 7004 | 23.43 | P003 | cooler | D8-S3-4 | fizzo_cherry | paid | shelf event emitted | 27 (27) | y | y | y |  |  |
| 7004 | 29.5 | P001 | cooler | D4-S2-1 | pacer_glacier | paid | right sku | 57 (57) | y | y | y | pacer_punch | P001 |
| 7004 | 30.77 | P004 | gondola | G1L-S1-11 | crunchly_sco | paid | shelf event emitted | 59 (59) | y | y |  |  |  |
| 7004 | 39 | P004 | gondola | G1R-S2-10 | relieva_aceta | concealed | hand or item detected | 43 (43) |  |  |  |  |  |
| 7004 | 43.47 | P001 | checkout | CK-10 | cocoa_crest | paid | shelf event emitted | 38 (38) | y | y | y |  |  |
| 7004 | 58.83 | P004 | gondola | G3L-S4-19 | caramel_crest | concealed | hand or item detected | 20 (20) |  |  |  |  |  |
| 7005 | 8.57 | P001 | checkout | CK-2 | peanut_pilot | concealed | shelf event emitted | 43 (43) | y | y |  |  |  |
| 7005 | 17.93 | P002 | cooler | D3-S2-8 | pacer_orange | paid | shelf event emitted | 59 (59) | y | y | y |  |  |
| 7005 | 19.43 | P003 | cooler | D2-S2-7 | clearwell_water | concealed | shelf event emitted | 48 (48) | y | y | y |  |  |
| 7005 | 20.5 | P005 | gondola | G1R-S3-49 | peanut_pilot | paid | hand or item detected | 32 (32) |  |  |  |  |  |
| 7005 | 23.17 | P004 | cooler | D2-S4-3 | clearwell_water | paid | shelf event emitted | 56 (56) | y | y | y |  |  |
| 7005 | 25.43 | P006 | gondola | G3R-S1-19 | voltlink_micro | paid | shelf event emitted | 52 (52) | y | y | y |  |  |
| 7005 | 27.7 | P002 | gondola | G4R-S4-3 | cocoa_crest | paid | shelf event emitted | 26 (26) | y | y |  |  |  |
| 7005 | 28.87 | P007 | gondola | G4L-S4-14 | relieva_ibu | concealed | shelf event emitted | 26 (26) |  |  | y |  |  |
| 7005 | 29.4 | P005 | gondola | G2R-S4-23 | ridgeline_original | concealed | shelf event emitted | 42 (42) | y | y | y |  |  |
| 7005 | 30.93 | P006 | gondola | G3R-S4-29 | voltlink_micro | concealed | shelf event emitted | 47 (47) | y | y |  |  |  |
| 7005 | 35.97 | P002 | checkout | CK-9 | cocoa_crest | paid | shelf event emitted | 38 (38) | y | y | y |  |  |
| 7005 | 36.87 | P008 | cooler | D5-S3-4 | lumen_citrus | put_back | shelf event emitted | 26 (26) | y | y | y |  |  |
| 7005 | 37.47 | P006 | gondola | G3R-S4-1 | torqueline_10w30 | concealed | hand or item detected | 49 (49) |  |  |  |  |  |
| 7005 | 49.9 | P008 | gondola | G4R-S3-20 | relieva_ibu | paid | shelf event emitted | 30 (30) | y | y |  |  |  |
| 7006 | 11.6 | P002 | gondola | G3R-S3-46 | peanut_pilot | paid | shelf event emitted | 37 (37) | y | y | y |  |  |
| 7006 | 13.73 | P003 | gondola | G3R-S4-42 | ridgeline_original | put_back | hand or item detected | 75 (75) |  |  | y |  |  |
| 7006 | 15.1 | P001 | gondola | G1L-S2-14 | peanut_pilot | paid | shelf event emitted | 44 (44) | y | y | y |  |  |
| 7006 | 21.17 | P005 | checkout | CK-12 | cocoa_crest | paid | shelf event emitted | 35 (35) | y | y | y |  |  |
| 7006 | 21.9 | P004 | gondola | G2L-S3-30 | relieva_ibu | put_back | shelf event emitted | 52 (52) | y | y | y |  |  |
| 7006 | 23 | P002 | gondola | G1L-S3-39 | torqueline_5w30 | concealed | hand or item detected | 23 (23) |  |  |  |  |  |
| 7006 | 23.27 | P003 | checkout | CK-1 | cocoa_crest | paid | shelf event emitted | 33 (33) | y | y | y |  |  |
| 7006 | 25.93 | P001 | cooler | D4-S3-2 | pacer_orange | paid | shelf event emitted | 58 (58) | y | y | y |  |  |
| 7006 | 27.7 | P006 | gondola | G4L-S3-24 | ridgeline_teriyaki | paid | hand or item detected | 30 (30) |  |  |  |  |  |
| 7006 | 34.83 | P004 | gondola | G2R-S1-19 | relieva_ibu | paid | hand or item detected | 33 (33) |  |  | y |  |  |
| 7006 | 36.07 | P007 | cooler | D4-S3-3 | pacer_orange | paid | shelf event emitted | 62 (62) | y | y | y |  |  |
| 7006 | 37.57 | P006 | checkout | CK-9 | peanut_pilot | concealed | shelf event emitted | 39 (39) | y | y |  |  |  |
| 7006 | 39.83 | P002 | gondola | G1L-S4-12 | peanut_pilot | concealed | hand or item detected | 39 (39) |  |  |  |  |  |
| 7006 | 42.57 | P004 | gondola | G3L-S4-47 | caramel_crest | paid | hand or item detected | 4 (4) |  |  | y |  |  |
| 7006 | 51.73 | P004 | cooler | D4-S2-1 | pacer_orange | paid | shelf event emitted | 57 (57) | y | y | y |  |  |
