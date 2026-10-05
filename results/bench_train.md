# Benchmark, train split

SIMULATED (browser store simulator copy, scripts/bench/render_clip.mjs). Not real footage.

9 clips (4900, 4901, 4902, 4903, 4904, 4905, 4906, 4950, 4951), 653 s of sim time, 53 shoppers (17 thieves, 36 honest), 106 picks, 19 stolen items. Runner `bree.shelf.store:run`, options {"backend": "sim_sku", "edge": true, "max_frames": null}. Scored 2026-10-05 17:07, commit 65f4103.

## Scorecard

| Metric | Value |
|---|---|
| Theft recall, alert tier | 0.0% (0/19) |
| Theft recall, alert or review | 78.9% (15/19) |
| Stolen items alerted with the right SKU | 0/19 |
| Alert precision | n/a (0/0) |
| False alerts on honest shoppers | 0 (of 36 honest shoppers); on nobody: 0 |
| Reviews on honest shoppers | 3 (of 16 reviews) |
| False alerts per hour | 0.0 |
| Pick recall | 91.5% (97/106) |
| Pick precision | 89.8% (of 108 PICK events) |
| Right SKU, of paired picks | 93.8%; of all true picks 85.8% |
| Right slot, of paired picks | 83.5% |
| Right shopper, of paired picks | 96.9% |
| Time to alert after concealment | n/a |
| Store-wide identities per real shopper | 1.377 (73 ids for 53 shoppers) |
| Identities sitting on one shopper | mean 1.17, max 2; shoppers never tracked 0; ids covering two shoppers 21 |

## Funnel: where each true pick is lost

Each true pick walks the stages in order and is counted at the first one it fails. "Passed on its own" counts the stage for every pick, whatever happened before it.

### All picks: 106 picks, 48 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 106 | 2 | 104 |
| frame reached pipeline | 104 | 0 | 104 |
| hand or item detected | 104 | 8 | 96 |
| shelf event emitted | 96 | 2 | 97 |
| right slot | 94 | 16 | 81 |
| right sku | 78 | 0 | 91 |
| associated to a shopper | 78 | 0 | 97 |
| right shopper | 78 | 2 | 94 |
| conceal or pay classified | 76 | 14 | 81 |
| ledger basket | 62 | 8 | 92 |
| alert | 54 | 6 | 72 |

### Outcome concealed: 19 picks, 0 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 19 | 0 | 19 |
| frame reached pipeline | 19 | 0 | 19 |
| hand or item detected | 19 | 0 | 19 |
| shelf event emitted | 19 | 0 | 19 |
| right slot | 19 | 3 | 16 |
| right sku | 16 | 0 | 17 |
| associated to a shopper | 16 | 0 | 19 |
| right shopper | 16 | 2 | 16 |
| conceal or pay classified | 14 | 14 | 0 |
| ledger basket | 0 | 0 | 14 |
| alert | 0 | 0 | 0 |

### Outcome paid: 62 picks, 32 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 62 | 1 | 61 |
| frame reached pipeline | 61 | 0 | 61 |
| hand or item detected | 61 | 6 | 55 |
| shelf event emitted | 55 | 1 | 56 |
| right slot | 54 | 8 | 48 |
| right sku | 46 | 0 | 53 |
| associated to a shopper | 46 | 0 | 56 |
| right shopper | 46 | 0 | 56 |
| conceal or pay classified | 46 | 0 | 62 |
| ledger basket | 46 | 8 | 53 |
| alert | 38 | 6 | 48 |

### Outcome put_back: 25 picks, 16 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 25 | 1 | 24 |
| frame reached pipeline | 24 | 0 | 24 |
| hand or item detected | 24 | 2 | 22 |
| shelf event emitted | 22 | 1 | 22 |
| right slot | 21 | 5 | 17 |
| right sku | 16 | 0 | 21 |
| associated to a shopper | 16 | 0 | 22 |
| right shopper | 16 | 0 | 22 |
| conceal or pay classified | 16 | 0 | 19 |
| ledger basket | 16 | 0 | 25 |
| alert | 16 | 0 | 24 |

### Zone checkout: 18 picks, 12 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 18 | 0 | 18 |
| frame reached pipeline | 18 | 0 | 18 |
| hand or item detected | 18 | 0 | 18 |
| shelf event emitted | 18 | 0 | 18 |
| right slot | 18 | 0 | 18 |
| right sku | 18 | 0 | 18 |
| associated to a shopper | 18 | 0 | 18 |
| right shopper | 18 | 0 | 18 |
| conceal or pay classified | 18 | 2 | 16 |
| ledger basket | 16 | 1 | 16 |
| alert | 15 | 3 | 12 |

### Zone cooler: 33 picks, 9 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 33 | 2 | 31 |
| frame reached pipeline | 31 | 0 | 31 |
| hand or item detected | 31 | 3 | 28 |
| shelf event emitted | 28 | 1 | 30 |
| right slot | 27 | 11 | 19 |
| right sku | 16 | 0 | 26 |
| associated to a shopper | 16 | 0 | 30 |
| right shopper | 16 | 0 | 29 |
| conceal or pay classified | 16 | 5 | 22 |
| ledger basket | 11 | 1 | 29 |
| alert | 10 | 1 | 22 |

### Zone gondola: 55 picks, 27 through every stage

| stage | reached this stage | lost here | passed on its own |
|---|---|---|---|
| in view | 55 | 0 | 55 |
| frame reached pipeline | 55 | 0 | 55 |
| hand or item detected | 55 | 5 | 50 |
| shelf event emitted | 50 | 1 | 49 |
| right slot | 49 | 5 | 44 |
| right sku | 44 | 0 | 47 |
| associated to a shopper | 44 | 0 | 49 |
| right shopper | 44 | 2 | 47 |
| conceal or pay classified | 42 | 7 | 43 |
| ledger basket | 35 | 6 | 47 |
| alert | 29 | 2 | 38 |

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
| hand detected | 96 |
| right fixture | 96 |
| picks with item frames | 104 |
| picks with item frames reached | 104 |

## Per clip

| clip | cameras | shoppers | thieves | picks | PICK events | stolen | alerted | alerts | false | ids | wall s (edge + pipeline) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4900 | 19 | 4 | 1 | 8 | 7 | 1 | 0 | 0 | 0 | 5 |  + 0.4 |
| 4901 | 20 | 6 | 4 | 12 | 12 | 4 | 0 | 0 | 0 | 8 |  + 0.6 |
| 4902 | 19 | 5 | 1 | 10 | 11 | 2 | 0 | 0 | 0 | 6 |  + 0.6 |
| 4903 | 18 | 4 | 2 | 8 | 11 | 2 | 0 | 0 | 0 | 6 |  + 0.4 |
| 4904 | 20 | 8 | 2 | 12 | 12 | 2 | 0 | 0 | 0 | 11 |  + 0.8 |
| 4905 | 15 | 5 | 3 | 11 | 10 | 4 | 0 | 0 | 0 | 6 |  + 0.6 |
| 4906 | 20 | 6 | 1 | 13 | 12 | 1 | 0 | 0 | 0 | 9 |  + 0.7 |
| 4950 | 20 | 7 | 1 | 15 | 17 | 1 | 0 | 0 | 0 | 9 |  + 0.8 |
| 4951 | 21 | 8 | 2 | 17 | 16 | 2 | 0 | 0 | 0 | 13 |  + 0.8 |

## Every true pick

| clip | t | shopper | zone | slot | sku | outcome | lost at | item cam frames (reached) | item det | sku det | person det | PICK sku | PICK shopper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4900 | 13.13 | P001 | gondola | G1L-S3-33 | torqueline_5w30 | put_back | none | 50 (50) |  |  |  | torqueline_5w30 | P001 |
| 4900 | 17.2 | P002 | gondola | G3L-S4-7 | cocoa_crest | put_back | hand or item detected | 25 (25) |  |  |  |  |  |
| 4900 | 18.83 | P003 | gondola | G3L-S2-21 | voltlink_usbc | concealed | conceal or pay classified | 50 (50) |  |  |  | voltlink_usbc | P003 |
| 4900 | 25.47 | P004 | gondola | G2L-S4-23 | sierra_ranch | put_back | none | 52 (52) |  |  |  | sierra_ranch | P004 |
| 4900 | 27.1 | P002 | cooler | D8-S5-2 | fizzo_cola | paid | none | 23 (23) |  |  |  | fizzo_cola | P002 |
| 4900 | 31.37 | P001 | gondola | G4L-S2-7 | crunchly_bbq | paid | none | 60 (60) |  |  |  | crunchly_bbq | P001 |
| 4900 | 37.93 | P004 | gondola | G3R-S4-33 | torqueline_10w30 | paid | none | 53 (53) |  |  |  | torqueline_10w30 | P004 |
| 4900 | 45.37 | P004 | cooler | D8-S4-9 | lumen_citrus | paid | none | 24 (24) |  |  |  | lumen_citrus | P004 |
| 4901 | 14.67 | P001 | cooler | D10-S5-3 | arcwave_berry | concealed | right slot | 56 (56) |  |  |  | arcwave_original | P001 |
| 4901 | 22.17 | P002 | cooler | D12-S1-6 | arcwave_berry | paid | none | 27 (27) |  |  |  | arcwave_berry | P002 |
| 4901 | 23.27 | P003 | gondola | G2L-S1-9 | cocoa_crest | concealed | right shopper | 18 (18) |  |  |  | cocoa_crest | P004 |
| 4901 | 25.8 | P004 | gondola | G2L-S3-9 | peanut_pilot | put_back | shelf event emitted | 49 (49) |  |  |  |  |  |
| 4901 | 26.73 | P005 | gondola | G4L-S2-21 | sierra_ranch | paid | alert | 57 (57) |  |  |  | sierra_ranch | P005 |
| 4901 | 30.63 | P002 | gondola | G4L-S4-31 | torqueline_5w30 | paid | hand or item detected | 34 (34) |  |  |  |  |  |
| 4901 | 33.53 | P005 | gondola | G3R-S1-27 | relieva_aceta | concealed | conceal or pay classified | 28 (28) |  |  |  | relieva_aceta | P005 |
| 4901 | 34.53 | P002 | gondola | G4L-S2-18 | crunchly_bbq | put_back | none | 27 (27) |  |  |  | crunchly_bbq | P002 |
| 4901 | 37.53 | P006 | gondola | G1L-S3-21 | relieva_aceta | concealed | conceal or pay classified | 49 (49) |  |  |  | relieva_aceta | P006 |
| 4901 | 39.37 | P004 | gondola | G3L-S4-20 | cocoa_crest | paid | ledger basket | 19 (19) |  |  |  | cocoa_crest | P004 |
| 4901 | 51.87 | P004 | checkout | CK-8 | peanut_pilot | paid | alert | 37 (37) |  |  |  | peanut_pilot | P004 |
| 4901 | 52.33 | P006 | cooler | D9-S5-6 | fizzo_zero | paid | right slot | 39 (39) |  |  |  | fizzo_cola | P006 |
| 4902 | 7.87 | P001 | checkout | CK-5 | peanut_pilot | paid | none | 47 (47) |  |  |  | peanut_pilot | P001 |
| 4902 | 14.67 | P003 | checkout | CK-10 | cocoa_crest | concealed | conceal or pay classified | 40 (40) |  |  |  | cocoa_crest | P003 |
| 4902 | 16.33 | P002 | gondola | G4R-S1-1 | crunchly_bbq | put_back | none | 25 (25) |  |  |  | crunchly_bbq | P002 |
| 4902 | 17.8 | P001 | cooler | D14-S3-5 | orchard_lemon | put_back | none | 25 (25) |  |  |  | orchard_lemon | P001 |
| 4902 | 26.97 | P004 | cooler | D9-S1-2 | lumen_citrus | paid | right slot | 37 (37) |  |  |  | lumen_citrus | P004 |
| 4902 | 29.03 | P005 | cooler | D10-S2-6 | arcwave_original | put_back | right slot | 51 (51) |  |  |  | arcwave_original | P005 |
| 4902 | 35.33 | P002 | gondola | G1L-S4-41 | peanut_pilot | paid | none | 40 (40) |  |  |  | peanut_pilot | P002 |
| 4902 | 42.37 | P005 | gondola | G1R-S3-3 | caramel_crest | paid | right slot | 53 (53) |  |  |  | peanut_pilot | P005 |
| 4902 | 42.97 | P003 | gondola | G4R-S3-40 | ridgeline_original | concealed | conceal or pay classified | 60 (60) |  |  |  | ridgeline_original | P003 |
| 4902 | 54.53 | P003 | checkout | CK-3 | peanut_pilot | paid | alert | 41 (41) |  |  |  | peanut_pilot | P003 |
| 4903 | 10.67 | P001 | gondola | G4R-S1-10 | crunchly_bbq | put_back | none | 52 (52) |  |  |  | crunchly_bbq | P001 |
| 4903 | 19.83 | P002 | cooler | D9-S5-3 | fizzo_zero | put_back | right slot | 48 (48) |  |  |  | fizzo_zero | P002 |
| 4903 | 22.43 | P001 | gondola | G3R-S2-22 | crunchly_classic | paid | none | 53 (53) |  |  |  | crunchly_classic | P001 |
| 4903 | 23.03 | P003 | cooler | D10-S5-4 | arcwave_original | concealed | conceal or pay classified | 37 (37) |  |  |  | arcwave_original | P003 |
| 4903 | 27.77 | P004 | checkout | CK-1 | caramel_crest | paid | alert | 35 (35) |  |  |  | caramel_crest | P004 |
| 4903 | 37.67 | P002 | gondola | G1R-S4-23 | crunchly_sco | paid | none | 44 (44) |  |  |  | crunchly_sco | P002 |
| 4903 | 38.47 | P004 | cooler | D11-S5-2 | arcwave_berry | concealed | conceal or pay classified | 52 (52) |  |  |  | arcwave_berry | P004 |
| 4903 | 39.63 | P003 | cooler | D6-S4-1 | lumen_citrus | paid | alert | 25 (25) |  |  |  | lumen_citrus | P003 |
| 4904 | 10.53 | P001 | gondola | G3L-S1-42 | ridgeline_original | concealed | right shopper | 33 (33) |  |  |  | ridgeline_original | P002 |
| 4904 | 12.53 | P002 | gondola | G2R-S4-20 | peanut_pilot | concealed | conceal or pay classified | 63 (63) |  |  |  | peanut_pilot | P002 |
| 4904 | 16.8 | P003 | checkout | CK-13 | cocoa_crest | put_back | none | 51 (51) |  |  |  | cocoa_crest | P003 |
| 4904 | 22.53 | P004 | gondola | G2R-S4-15 | cocoa_crest | paid | ledger basket | 65 (65) |  |  |  | cocoa_crest | P004 |
| 4904 | 24.4 | P003 | checkout | CK-13 | cocoa_crest | paid | none | 38 (38) |  |  |  | cocoa_crest | P003 |
| 4904 | 26.27 | P005 | gondola | G2R-S2-11 | cocoa_crest | paid | right slot | 51 (51) |  |  |  | cocoa_crest | P005 |
| 4904 | 28.17 | P006 | gondola | G3L-S4-42 | peanut_pilot | paid | hand or item detected | 21 (21) |  |  |  |  |  |
| 4904 | 31.6 | P004 | gondola | G1R-S2-45 | ridgeline_original | paid | ledger basket | 53 (53) |  |  |  | ridgeline_original | P004 |
| 4904 | 35.3 | P008 | checkout | CK-2 | peanut_pilot | put_back | none | 52 (52) |  |  |  | peanut_pilot | P008 |
| 4904 | 37.1 | P006 | cooler | D6-S5-8 | lumen_citrus | put_back | none | 26 (26) |  |  |  | lumen_citrus | P006 |
| 4904 | 40.43 | P007 | gondola | G1L-S4-6 | peanut_pilot | paid | none | 40 (40) |  |  |  | peanut_pilot | P007 |
| 4904 | 48.8 | P008 | gondola | G4L-S3-5 | oatfield_cookies | paid | none | 48 (48) |  |  |  | oatfield_cookies | P008 |
| 4905 | 10.3 | P001 | checkout | CK-5 | cocoa_crest | concealed | conceal or pay classified | 47 (47) |  |  |  | cocoa_crest | P001 |
| 4905 | 14.47 | P001 | checkout | CK-2 | cocoa_crest | paid | ledger basket | 43 (43) |  |  |  | cocoa_crest | P001 |
| 4905 | 24.07 | P002 | cooler | D12-S5-7 | arcwave_berry | concealed | conceal or pay classified | 27 (27) |  |  |  | arcwave_berry | P002 |
| 4905 | 27.8 | P003 | cooler | D7-S1-3 | fizzo_zero | put_back | right slot | 33 (33) |  |  |  | fizzo_zero | P003 |
| 4905 | 29.23 | P004 | cooler | D6-S4-1 | fizzo_cola | paid | hand or item detected | 25 (25) |  |  |  | fizzo_cola | P004 |
| 4905 | 31.13 | P005 | gondola | G2R-S1-12 | relieva_ibu | concealed | conceal or pay classified | 39 (39) |  |  |  | relieva_ibu | P005 |
| 4905 | 34.13 | P004 | gondola | G3R-S4-1 | torqueline_5w30 | paid | hand or item detected | 25 (25) |  |  |  |  |  |
| 4905 | 42.7 | P003 | checkout | CK-3 | cocoa_crest | paid | none | 42 (42) |  |  |  | cocoa_crest | P003 |
| 4905 | 47.27 | P002 | cooler | D11-S5-4 | arcwave_tropical | concealed | conceal or pay classified | 43 (43) |  |  |  | arcwave_tropical | P002 |
| 4905 | 54.1 | P003 | cooler | D10-S3-1 | arcwave_berry | put_back | none | 26 (26) |  |  |  | arcwave_berry | P003 |
| 4905 | 70.23 | P003 | checkout | CK-13 | peanut_pilot | paid | none | 34 (34) |  |  |  | peanut_pilot | P003 |
| 4906 | 14.33 | P001 | cooler | D10-S1-8 | arcwave_original | concealed | right slot | 35 (35) |  |  |  | arcwave_original | P001 |
| 4906 | 15.17 | P002 | gondola | G1R-S1-16 | peanut_pilot | paid | none | 56 (56) |  |  |  | peanut_pilot | P002 |
| 4906 | 19.13 | P003 | gondola | G4R-S2-22 | ridgeline_original | put_back | none | 48 (48) |  |  |  | ridgeline_original | P003 |
| 4906 | 27.87 | P006 | checkout | CK-6 | caramel_crest | put_back | none | 52 (52) |  |  |  | caramel_crest | P006 |
| 4906 | 29.5 | P002 | gondola | G4R-S2-29 | morning_hoops | put_back | none | 51 (51) |  |  |  | morning_hoops | P002 |
| 4906 | 29.57 | P005 | cooler | D14-S3-1 | orchard_lemon | put_back | in view | 0 (0) |  |  |  |  |  |
| 4906 | 30.1 | P004 | cooler | D14-S5-7 | orchard_peach | paid | in view | 0 (0) |  |  |  |  |  |
| 4906 | 32.07 | P003 | gondola | G3R-S2-12 | sierra_nacho | paid | none | 43 (43) |  |  |  | sierra_nacho | P003 |
| 4906 | 43.33 | P005 | checkout | CK-10 | caramel_crest | paid | none | 40 (40) |  |  |  | caramel_crest | P005 |
| 4906 | 46.17 | P006 | gondola | G3L-S1-6 | torqueline_5w30 | paid | none | 30 (30) |  |  |  | torqueline_5w30 | P006 |
| 4906 | 53.03 | P005 | gondola | G2R-S2-43 | caramel_crest | paid | none | 41 (41) |  |  |  | caramel_crest | P005 |
| 4906 | 53.73 | P006 | gondola | G1R-S2-22 | relieva_aceta | paid | none | 55 (55) |  |  |  | relieva_aceta | P006 |
| 4906 | 64.87 | P005 | checkout | CK-3 | peanut_pilot | paid | none | 44 (44) |  |  |  | peanut_pilot | P005 |
| 4950 | 13.97 | P002 | checkout | CK-14 | cocoa_crest | paid | none | 39 (39) |  |  |  | cocoa_crest | P002 |
| 4950 | 15.37 | P001 | cooler | D14-S5-6 | orchard_peach | paid | right slot | 25 (25) |  |  |  | orchard_peach | P001 |
| 4950 | 22.63 | P003 | gondola | G4R-S3-1 | torqueline_5w30 | put_back | right slot | 42 (42) |  |  |  | torqueline_5w30 | P003 |
| 4950 | 24.4 | P002 | gondola | G2R-S4-37 | peanut_pilot | paid | right slot | 35 (35) |  |  |  | peanut_pilot | P002 |
| 4950 | 25.43 | P001 | checkout | CK-7 | cocoa_crest | put_back | none | 50 (50) |  |  |  | cocoa_crest | P001 |
| 4950 | 26.77 | P004 | cooler | D10-S5-2 | arcwave_tropical | paid | none | 55 (55) |  |  |  | arcwave_tropical | P004 |
| 4950 | 26.87 | P005 | gondola | G4L-S1-28 | relieva_ibu | paid | ledger basket | 34 (34) |  |  |  | relieva_ibu | P005 |
| 4950 | 30.3 | P006 | gondola | G4L-S3-10 | caramel_crest | paid | none | 51 (51) |  |  |  | caramel_crest | P006 |
| 4950 | 30.33 | P003 | cooler | D10-S1-2 | arcwave_tropical | paid | shelf event emitted | 43 (43) |  |  |  |  |  |
| 4950 | 37.53 | P005 | gondola | G2L-S1-2 | oatfield_cookies | paid | ledger basket | 25 (25) |  |  |  | oatfield_cookies | P005 |
| 4950 | 38.27 | P007 | cooler | D11-S4-7 | arcwave_tropical | concealed | right slot | 51 (51) |  |  |  | arcwave_original | P001 |
| 4950 | 40.2 | P001 | cooler | D9-S2-1 | fizzo_cola | paid | right slot | 30 (30) |  |  |  | fizzo_cola | P001 |
| 4950 | 40.77 | P003 | checkout | CK-7 | cocoa_crest | put_back | none | 52 (52) |  |  |  | cocoa_crest | P003 |
| 4950 | 57.8 | P003 | gondola | G3L-S4-17 | peanut_pilot | paid | none | 21 (21) |  |  |  | peanut_pilot | P003 |
| 4950 | 65.57 | P003 | cooler | D9-S2-9 | fizzo_cola | paid | none | 54 (54) |  |  |  | fizzo_cola | P003 |
| 4951 | 11.83 | P003 | gondola | G3R-S2-24 | crunchly_bbq | paid | alert | 38 (38) |  |  |  | crunchly_bbq | P003 |
| 4951 | 13.23 | P002 | gondola | G4R-S4-19 | voltlink_usbc | paid | none | 49 (49) |  |  |  | voltlink_usbc | P002 |
| 4951 | 14.7 | P001 | cooler | D2-S5-5 | orchard_peach | paid | right slot | 39 (39) |  |  |  | orchard_lemon | P001 |
| 4951 | 17.07 | P002 | gondola | G4R-S4-34 | relieva_ibu | paid | none | 70 (70) |  |  |  | relieva_ibu | P002 |
| 4951 | 19.3 | P004 | gondola | G4L-S1-8 | caramel_crest | put_back | right slot | 18 (18) |  |  |  | voltlink_micro | P004 |
| 4951 | 20.9 | P003 | cooler | D10-S3-6 | arcwave_berry | concealed | conceal or pay classified | 24 (24) |  |  |  | arcwave_berry | P003 |
| 4951 | 24.1 | P001 | gondola | G2R-S4-43 | cocoa_crest | paid | none | 17 (17) |  |  |  | cocoa_crest | P001 |
| 4951 | 27.03 | P005 | cooler | D8-S4-3 | fizzo_cola | paid | ledger basket | 20 (20) |  |  |  | fizzo_cola | P005 |
| 4951 | 32.33 | P004 | gondola | G2L-S2-8 | torqueline_10w30 | put_back | none | 50 (50) |  |  |  | torqueline_10w30 | P004 |
| 4951 | 32.33 | P008 | gondola | G4L-S4-6 | torqueline_10w30 | paid | none | 19 (19) |  |  |  | torqueline_10w30 | P008 |
| 4951 | 32.87 | P005 | gondola | G3R-S1-20 | voltlink_micro | concealed | conceal or pay classified | 39 (39) |  |  |  | voltlink_micro | P005 |
| 4951 | 33.13 | P006 | cooler | D12-S1-6 | arcwave_original | put_back | hand or item detected | 25 (25) |  |  |  | arcwave_original | P006 |
| 4951 | 34.07 | P007 | cooler | D11-S5-5 | arcwave_tropical | paid | hand or item detected | 36 (36) |  |  |  | arcwave_tropical | P007 |
| 4951 | 41.33 | P004 | gondola | G1R-S2-40 | ridgeline_teriyaki | paid | none | 37 (37) |  |  |  | ridgeline_teriyaki | P004 |
| 4951 | 41.83 | P006 | cooler | D8-S3-6 | lumen_citrus | paid | none | 12 (12) |  |  |  | lumen_citrus | P006 |
| 4951 | 41.93 | P008 | gondola | G2L-S1-19 | peanut_pilot | paid | ledger basket | 39 (39) |  |  |  | peanut_pilot | P008 |
| 4951 | 50.53 | P004 | gondola | G3L-S4-1 | oatfield_cookies | paid | hand or item detected | 34 (34) |  |  |  |  |  |
