# Benchmark, test split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

6 clips (9001, 9002, 9003, 9004, 9005, 9006), 466 s of sim time, 31 shoppers (9 thieves, 22 honest), 66 picks, 14 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-05 17:51, commit 9068c1a.

## Scorecard

| Metric | Value |
|---|---|
| Theft recall, alert tier | 0.0% (0/14) |
| Theft recall, alert or review | 100.0% (14/14) |
| Stolen items alerted with the right SKU | 0/14 |
| Alert precision | n/a (0/0) |
| False alerts on honest shoppers | 0 (of 22 honest shoppers); on nobody: 0 |
| Reviews on honest shoppers | 2 (of 11 reviews) |
| False alerts per hour | 0.0 |
| Pick recall | 97.0% (64/66) |
| Pick precision | 90.1% (of 71 PICK events) |
| Right SKU, of paired picks | 92.2%; of all true picks 89.4% |
| Right slot, of paired picks | 82.8% |
| Right shopper, of paired picks | 92.2% |
| Time to alert after concealment | n/a |
| Store-wide identities per real shopper | 1.387 (43 ids for 31 shoppers) |
| Identities sitting on one shopper | mean 1.19, max 2; shoppers never tracked 1; ids covering two shoppers 14 |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 66 picks, 26 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 66 | 0 | 66 |
| frame reached pipeline | 66 | 0 | 66 |
| hand or item detected | 66 | 6 | 60 |
| shelf event emitted | 60 | 2 | 64 |
| right slot | 58 | 8 | 53 |
| right sku | 50 | 0 | 59 |
| associated to a shopper | 50 | 0 | 64 |
| right shopper | 50 | 3 | 59 |
| conceal or pay classified | 47 | 11 | 51 |
| ledger basket | 36 | 0 | 63 |
| alert | 36 | 10 | 42 |

### Outcome concealed: 14 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 14 | 0 | 14 |
| frame reached pipeline | 14 | 0 | 14 |
| hand or item detected | 14 | 0 | 14 |
| shelf event emitted | 14 | 0 | 14 |
| right slot | 14 | 4 | 10 |
| right sku | 10 | 0 | 12 |
| associated to a shopper | 10 | 0 | 14 |
| right shopper | 10 | 0 | 14 |
| conceal or pay classified | 10 | 10 | 0 |
| ledger basket | 0 | 0 | 12 |
| alert | 0 | 0 | 0 |

### Outcome paid: 37 picks, 20 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 37 | 0 | 37 |
| frame reached pipeline | 37 | 0 | 37 |
| hand or item detected | 37 | 5 | 32 |
| shelf event emitted | 32 | 0 | 37 |
| right slot | 32 | 2 | 32 |
| right sku | 30 | 0 | 34 |
| associated to a shopper | 30 | 0 | 37 |
| right shopper | 30 | 2 | 33 |
| conceal or pay classified | 28 | 1 | 36 |
| ledger basket | 27 | 0 | 36 |
| alert | 27 | 7 | 30 |

### Outcome put_back: 15 picks, 6 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 15 | 0 | 15 |
| frame reached pipeline | 15 | 0 | 15 |
| hand or item detected | 15 | 1 | 14 |
| shelf event emitted | 14 | 2 | 13 |
| right slot | 12 | 2 | 11 |
| right sku | 10 | 0 | 13 |
| associated to a shopper | 10 | 0 | 13 |
| right shopper | 10 | 1 | 12 |
| conceal or pay classified | 9 | 0 | 15 |
| ledger basket | 9 | 0 | 15 |
| alert | 9 | 3 | 12 |

### Zone checkout: 10 picks, 3 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 10 | 0 | 10 |
| frame reached pipeline | 10 | 0 | 10 |
| hand or item detected | 10 | 0 | 10 |
| shelf event emitted | 10 | 1 | 9 |
| right slot | 9 | 3 | 6 |
| right sku | 6 | 0 | 8 |
| associated to a shopper | 6 | 0 | 9 |
| right shopper | 6 | 0 | 8 |
| conceal or pay classified | 6 | 0 | 10 |
| ledger basket | 6 | 0 | 10 |
| alert | 6 | 3 | 7 |

### Zone cooler: 16 picks, 5 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 16 | 0 | 16 |
| frame reached pipeline | 16 | 0 | 16 |
| hand or item detected | 16 | 1 | 15 |
| shelf event emitted | 15 | 0 | 16 |
| right slot | 15 | 3 | 12 |
| right sku | 12 | 0 | 14 |
| associated to a shopper | 12 | 0 | 16 |
| right shopper | 12 | 1 | 15 |
| conceal or pay classified | 11 | 3 | 10 |
| ledger basket | 8 | 0 | 13 |
| alert | 8 | 3 | 8 |

### Zone gondola: 40 picks, 18 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 40 | 0 | 40 |
| frame reached pipeline | 40 | 0 | 40 |
| hand or item detected | 40 | 5 | 35 |
| shelf event emitted | 35 | 1 | 39 |
| right slot | 34 | 2 | 35 |
| right sku | 32 | 0 | 37 |
| associated to a shopper | 32 | 0 | 39 |
| right shopper | 32 | 2 | 36 |
| conceal or pay classified | 30 | 8 | 31 |
| ledger basket | 22 | 0 | 40 |
| alert | 22 | 4 | 27 |

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
| hand detected | 60 |
| right fixture | 63 |
| picks with item frames | 66 |
| picks with item frames reached | 66 |

## Per clip

| clip | cameras | shoppers | thieves | picks | PICK events | stolen | alerted | alerts | false | ids | wall s (edge + pipeline) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 9001 | 18 | 5 | 1 | 10 | 10 | 1 | 0 | 0 | 0 | 6 |  + 294.6 |
| 9002 | 20 | 5 | 3 | 11 | 13 | 5 | 0 | 0 | 0 | 6 |  + 508.4 |
| 9003 | 20 | 8 | 2 | 13 | 13 | 3 | 0 | 0 | 0 | 10 |  + 444.6 |
| 9004 | 20 | 4 | 1 | 11 | 12 | 2 | 0 | 0 | 0 | 7 |  + 415.8 |
| 9005 | 16 | 4 | 1 | 9 | 9 | 2 | 0 | 0 | 0 | 6 |  + 341.9 |
| 9006 | 20 | 5 | 1 | 12 | 14 | 1 | 0 | 0 | 0 | 8 |  + 233.7 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9001 | 11.53 | P001 | gondola | G1R-S1-27 | ridgeline_teriyaki | put_back | shelf event emitted | 75 (75) |  |  |  |  |  |
| 9001 | 13.03 | P002 | gondola | G1R-S4-20 | crunchly_bbq | paid | hand or item detected | 51 (51) |  |  |  | ridgeline_teriyaki | P001 |
| 9001 | 18.2 | P002 | gondola | G2L-S2-11 | relieva_ibu | paid | none | 53 (53) |  |  |  | relieva_ibu | P002 |
| 9001 | 18.3 | P003 | gondola | G3R-S4-31 | torqueline_5w30 | concealed | conceal or pay classified | 49 (49) |  |  |  | torqueline_5w30 | P003 |
| 9001 | 25.43 | P004 | gondola | G2L-S2-8 | torqueline_10w30 | put_back | alert | 75 (75) |  |  |  | torqueline_10w30 | P004 |
| 9001 | 26.17 | P005 | checkout | CK-11 | cocoa_crest | put_back | shelf event emitted | 42 (42) |  |  |  |  |  |
| 9001 | 28.77 | P001 | checkout | CK-8 | caramel_crest | paid | right slot | 35 (35) |  |  |  | cocoa_crest | P005 |
| 9001 | 35.9 | P005 | checkout | CK-7 | caramel_crest | paid | none | 39 (39) |  |  |  | caramel_crest | P005 |
| 9001 | 38.97 | P004 | gondola | G2R-S1-37 | torqueline_5w30 | paid | alert | 42 (42) |  |  |  | torqueline_5w30 | P004 |
| 9001 | 48.47 | P004 | checkout | CK-11 | cocoa_crest | put_back | alert | 52 (52) |  |  |  | cocoa_crest | P004 |
| 9002 | 14.97 | P001 | cooler | D11-S2-1 | arcwave_original | concealed | conceal or pay classified | 43 (43) |  |  |  | arcwave_original | P001 |
| 9002 | 16.7 | P002 | gondola | G2L-S4-22 | sierra_nacho | paid | none | 52 (52) |  |  |  | sierra_nacho | P002 |
| 9002 | 21.17 | P001 | cooler | D12-S4-5 | arcwave_berry | concealed | right slot | 24 (24) |  |  |  | arcwave_tropical | P001 |
| 9002 | 28.23 | P002 | gondola | G3R-S4-11 | relieva_ibu | paid | hand or item detected | 26 (26) |  |  |  | peanut_pilot | P002 |
| 9002 | 30.73 | P003 | cooler | D11-S3-5 | arcwave_original | paid | alert | 40 (40) |  |  |  | arcwave_original | P003 |
| 9002 | 32.37 | P004 | cooler | D12-S4-8 | arcwave_berry | paid | alert | 33 (33) |  |  |  | arcwave_berry | P004 |
| 9002 | 36.4 | P005 | gondola | G1L-S2-28 | morning_hoops | put_back | right shopper | 46 (46) |  |  |  | morning_hoops | P001 |
| 9002 | 41.63 | P004 | gondola | G3L-S1-1 | torqueline_10w30 | concealed | conceal or pay classified | 23 (23) |  |  |  | torqueline_10w30 | P004 |
| 9002 | 44.97 | P005 | gondola | G1L-S4-7 | peanut_pilot | paid | right shopper | 45 (45) |  |  |  | peanut_pilot | P003 |
| 9002 | 46.43 | P003 | gondola | G1L-S2-21 | ridgeline_original | concealed | conceal or pay classified | 63 (63) |  |  |  | ridgeline_original | P003 |
| 9002 | 59.3 | P004 | gondola | G3R-S4-34 | torqueline_10w30 | concealed | right slot | 69 (69) |  |  |  | torqueline_10w30 | P004 |
| 9003 | 8.17 | P001 | checkout | CK-4 | peanut_pilot | paid | none | 42 (42) |  |  |  | peanut_pilot | P001 |
| 9003 | 17.13 | P002 | gondola | G3R-S1-10 | cocoa_crest | paid | hand or item detected | 33 (33) |  |  |  | cocoa_crest | P002 |
| 9003 | 18.2 | P003 | gondola | G3R-S4-14 | relieva_ibu | paid | none | 38 (38) |  |  |  | relieva_ibu | P003 |
| 9003 | 19.1 | P001 | cooler | D8-S1-9 | fizzo_cherry | paid | none | 27 (27) |  |  |  | fizzo_cherry | P001 |
| 9003 | 26.07 | P004 | cooler | D14-S1-6 | orchard_lemon | put_back | none | 26 (26) |  |  |  | orchard_lemon | P004 |
| 9003 | 26.63 | P002 | cooler | D2-S1-1 | clearwell_water | paid | hand or item detected | 50 (50) |  |  |  | clearwell_water | P002 |
| 9003 | 26.83 | P005 | gondola | G4R-S3-17 | relieva_aceta | concealed | conceal or pay classified | 41 (41) |  |  |  | relieva_aceta | P005 |
| 9003 | 30.77 | P007 | gondola | G3R-S4-40 | ridgeline_original | concealed | conceal or pay classified | 47 (47) |  |  |  | ridgeline_original | P007 |
| 9003 | 32.3 | P004 | cooler | D13-S1-5 | orchard_peach | paid | none | 26 (26) |  |  |  | orchard_peach | P004 |
| 9003 | 32.8 | P006 | gondola | G1L-S1-14 | sierra_ranch | paid | none | 26 (26) |  |  |  | sierra_ranch | P006 |
| 9003 | 35.43 | P007 | gondola | G4L-S1-46 | caramel_crest | concealed | conceal or pay classified | 25 (25) |  |  |  | caramel_crest | P007 |
| 9003 | 36.4 | P002 | gondola | G4L-S3-1 | oatfield_cookies | put_back | none | 26 (26) |  |  |  | oatfield_cookies | P002 |
| 9003 | 42.43 | P008 | gondola | G4R-S4-10 | peanut_pilot | paid | none | 26 (26) |  |  |  | peanut_pilot | P008 |
| 9004 | 13.27 | P002 | gondola | G1R-S1-19 | peanut_pilot | concealed | conceal or pay classified | 43 (43) |  |  |  | peanut_pilot | P002 |
| 9004 | 13.6 | P001 | cooler | D9-S2-6 | fizzo_cola | put_back | alert | 53 (53) |  |  |  | fizzo_cola | P001 |
| 9004 | 15.67 | P003 | gondola | G1R-S3-29 | relieva_aceta | paid | none | 58 (58) |  |  |  | relieva_aceta | P003 |
| 9004 | 20.23 | P004 | gondola | G1L-S3-8 | torqueline_10w30 | put_back | none | 50 (50) |  |  |  | torqueline_10w30 | P004 |
| 9004 | 27.63 | P002 | gondola | G4L-S1-5 | cocoa_crest | concealed | conceal or pay classified | 25 (25) |  |  |  | cocoa_crest | P002 |
| 9004 | 31.1 | P001 | gondola | G2L-S1-30 | morning_hoops | paid | alert | 55 (55) |  |  |  | morning_hoops | P001 |
| 9004 | 36.6 | P004 | cooler | D8-S3-6 | fizzo_cola | paid | none | 36 (36) |  |  |  | fizzo_cola | P004 |
| 9004 | 39.3 | P002 | checkout | CK-10 | caramel_crest | paid | alert | 37 (37) |  |  |  | caramel_crest | P002 |
| 9004 | 48.57 | P004 | gondola | G1R-S2-46 | ridgeline_teriyaki | put_back | none | 27 (27) |  |  |  | ridgeline_teriyaki | P004 |
| 9004 | 64.63 | P004 | gondola | G1L-S4-9 | peanut_pilot | paid | right slot | 37 (37) |  |  |  | peanut_pilot | P004 |
| 9004 | 75.93 | P004 | gondola | G3L-S1-27 | voltlink_usbc | paid | hand or item detected | 25 (25) |  |  |  | voltlink_usbc | P004 |
| 9005 | 8.43 | P001 | checkout | CK-14 | peanut_pilot | put_back | right slot | 50 (50) |  |  |  | peanut_pilot | P001 |
| 9005 | 20.13 | P002 | gondola | G1R-S1-13 | caramel_crest | paid | none | 42 (42) |  |  |  | caramel_crest | P002 |
| 9005 | 22.47 | P003 | gondola | G1R-S1-20 | caramel_crest | put_back | none | 52 (52) |  |  |  | caramel_crest | P003 |
| 9005 | 22.8 | P001 | gondola | G2L-S1-46 | peanut_pilot | paid | none | 17 (17) |  |  |  | peanut_pilot | P001 |
| 9005 | 35.97 | P004 | cooler | D11-S1-9 | arcwave_tropical | concealed | right slot | 49 (49) |  |  |  | arcwave_tropical | P004 |
| 9005 | 39.33 | P003 | checkout | CK-14 | peanut_pilot | put_back | right slot | 50 (50) |  |  |  | peanut_pilot | P003 |
| 9005 | 42.1 | P004 | cooler | D12-S1-7 | arcwave_tropical | concealed | right slot | 35 (35) |  |  |  | arcwave_original | P004 |
| 9005 | 53.3 | P004 | checkout | CK-2 | cocoa_crest | paid | alert | 49 (49) |  |  |  | cocoa_crest | P004 |
| 9005 | 55.4 | P003 | cooler | D7-S4-2 | fizzo_zero | paid | none | 45 (45) |  |  |  | fizzo_zero | P003 |
| 9006 | 12.43 | P001 | gondola | G1R-S3-30 | relieva_ibu | paid | alert | 50 (50) |  |  |  | relieva_ibu | P001 |
| 9006 | 19 | P002 | gondola | G1R-S2-37 | torqueline_5w30 | put_back | none | 72 (72) |  |  |  | torqueline_5w30 | P002 |
| 9006 | 24.77 | P001 | cooler | D12-S2-7 | arcwave_tropical | concealed | conceal or pay classified | 45 (45) |  |  |  | arcwave_tropical | P001 |
| 9006 | 25.07 | P003 | gondola | G2R-S2-32 | relieva_aceta | put_back | hand or item detected | 40 (40) |  |  |  | relieva_aceta | P003 |
| 9006 | 28.77 | P004 | gondola | G2R-S4-18 | cocoa_crest | paid | none | 52 (52) |  |  |  | cocoa_crest | P004 |
| 9006 | 30.1 | P005 | checkout | CK-5 | caramel_crest | paid | none | 41 (41) |  |  |  | caramel_crest | P005 |
| 9006 | 32.9 | P002 | cooler | D6-S3-5 | fizzo_cola | paid | conceal or pay classified | 34 (34) |  |  |  | fizzo_cola | P002 |
| 9006 | 36.67 | P005 | gondola | G3R-S4-42 | ridgeline_teriyaki | paid | none | 22 (22) |  |  |  | ridgeline_teriyaki | P005 |
| 9006 | 37.17 | P004 | cooler | D10-S5-8 | arcwave_original | paid | right shopper | 68 (68) |  |  |  | arcwave_original | P001 |
| 9006 | 40.93 | P003 | gondola | G4R-S4-40 | caramel_crest | paid | none | 16 (16) |  |  |  | caramel_crest | P003 |
| 9006 | 52.57 | P003 | gondola | G2R-S4-23 | ridgeline_original | paid | none | 52 (52) |  |  |  | ridgeline_original | P003 |
| 9006 | 60.47 | P003 | gondola | G3R-S1-11 | caramel_crest | paid | none | 26 (26) |  |  |  | caramel_crest | P003 |
