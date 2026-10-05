# Sim-trained SKU detector: results on held-out SIMULATED scenes

All numbers below are on simulated frames from the browser store simulator (one store model, invented package art). They are not accuracy on real footage.

Trained: yolo26n.pt fine-tuned for 6 epochs on 13596 tiles of 640 px, batch 32, device mps, 320.8 minutes wall time.

Test split: 965 frames from 45 scene seeds never used in training (3299 train frames from 151 seeds).

## mAP on the test tiles

All classes: mAP50 0.972, mAP50-95 0.882 (3991 tiles, 49424 boxes).

| SKU | boxes | mAP50 | mAP50-95 |
|---|---|---|---|
| fizzo_cola | 1945 | 0.979 | 0.871 |
| fizzo_zero | 1646 | 0.972 | 0.861 |
| fizzo_cherry | 1916 | 0.976 | 0.862 |
| lumen_citrus | 1268 | 0.976 | 0.823 |
| arcwave_original | 1505 | 0.981 | 0.891 |
| arcwave_tropical | 1501 | 0.981 | 0.891 |
| arcwave_berry | 1429 | 0.974 | 0.884 |
| clearwell_water | 925 | 0.965 | 0.819 |
| orchard_lemon | 1149 | 0.916 | 0.823 |
| orchard_peach | 916 | 0.903 | 0.821 |
| pacer_glacier | 705 | 0.953 | 0.812 |
| pacer_orange | 568 | 0.972 | 0.860 |
| pacer_punch | 558 | 0.971 | 0.865 |
| crunchly_classic | 770 | 0.976 | 0.936 |
| crunchly_sco | 781 | 0.974 | 0.926 |
| crunchly_bbq | 736 | 0.978 | 0.943 |
| sierra_nacho | 815 | 0.979 | 0.936 |
| sierra_ranch | 768 | 0.982 | 0.939 |
| ridgeline_original | 1005 | 0.977 | 0.924 |
| ridgeline_teriyaki | 1276 | 0.978 | 0.925 |
| oatfield_cookies | 1488 | 0.983 | 0.918 |
| morning_hoops | 822 | 0.972 | 0.915 |
| torqueline_5w30 | 1389 | 0.959 | 0.872 |
| torqueline_10w30 | 1357 | 0.963 | 0.873 |
| relieva_ibu | 2091 | 0.980 | 0.891 |
| relieva_aceta | 1958 | 0.983 | 0.893 |
| voltlink_usbc | 1507 | 0.980 | 0.929 |
| voltlink_micro | 1201 | 0.983 | 0.933 |
| cocoa_crest | 4376 | 0.978 | 0.855 |
| peanut_pilot | 4254 | 0.984 | 0.877 |
| caramel_crest | 4134 | 0.980 | 0.859 |
| aldermoor_gold | 1171 | 0.982 | 0.818 |
| aldermoor_menthol | 1494 | 0.978 | 0.848 |

## Accuracy by pixels across the item (whole frames, tiled inference, confidence 0.25, IoU 0.5)

965 test frames, 84873 labelled items, box precision 90.9%. Pixels = pixels across a 6.6 cm can at the item, the simulator's effective-pixel metric without its lens edge term.

### Shelf, front item, at least 80% visible

| px across a can | items | found | found with the right SKU | right SKU when found |
|---|---|---|---|---|
| 0-10 | 93 | 100.0% | 100.0% | 100.0% |
| 10-15 | 124 | 96.0% | 94.3% | 98.3% |
| 15-20 | 1238 | 97.8% | 96.2% | 98.4% |
| 20-25 | 2048 | 98.3% | 97.3% | 98.9% |
| 25-30 | 2357 | 98.5% | 98.4% | 99.8% |
| 30-40 | 12070 | 98.7% | 98.3% | 99.6% |
| 40-60 | 5743 | 92.2% | 91.8% | 99.6% |
| 60-100 | 1958 | 91.8% | 91.8% | 100.0% |
| 100+ | 64 | 92.2% | 92.2% | 100.0% |

### Shelf, front item, 25 to 80% visible

| px across a can | items | found | found with the right SKU | right SKU when found |
|---|---|---|---|---|
| 0-10 | 267 | 97.4% | 96.6% | 99.2% |
| 10-15 | 804 | 93.9% | 92.3% | 98.3% |
| 15-20 | 4060 | 94.2% | 91.6% | 97.3% |
| 20-25 | 5788 | 96.8% | 94.7% | 97.8% |
| 25-30 | 6006 | 95.7% | 94.4% | 98.6% |
| 30-40 | 10353 | 96.2% | 95.6% | 99.3% |
| 40-60 | 9268 | 90.2% | 89.8% | 99.6% |
| 60-100 | 2922 | 83.9% | 83.7% | 99.8% |
| 100+ | 81 | 92.6% | 90.1% | 97.3% |

### Shelf, item behind the front one

| px across a can | items | found | found with the right SKU | right SKU when found |
|---|---|---|---|---|
| 0-10 | 392 | 97.2% | 95.7% | 98.4% |
| 10-15 | 261 | 86.6% | 84.3% | 97.4% |
| 15-20 | 1299 | 89.8% | 87.5% | 97.4% |
| 20-25 | 1996 | 93.1% | 91.8% | 98.6% |
| 25-30 | 2214 | 93.0% | 91.6% | 98.6% |
| 30-40 | 7277 | 92.7% | 91.3% | 98.6% |
| 40-60 | 4589 | 87.9% | 86.9% | 98.9% |
| 60-100 | 1160 | 80.3% | 79.9% | 99.5% |
| 100+ | 29 | 75.9% | 75.9% | 100.0% |

### In a hand or on the counter

| px across a can | items | found | found with the right SKU | right SKU when found |
|---|---|---|---|---|
| 30-40 | 5 | 80.0% | 60.0% | 75.0% |
| 40-60 | 144 | 79.9% | 58.3% | 73.0% |
| 60-100 | 165 | 89.1% | 83.6% | 93.9% |
| 100+ | 98 | 62.2% | 50.0% | 80.3% |

## Minimum pixels

Lowest bucket edge from which every higher bucket (30+ items) has the right SKU at the target rate:

| items | 80% | 90% | 95% |
|---|---|---|---|
| Shelf, front item, at least 80% visible | 0 | 0 | None |
| Shelf, front item, 25 to 80% visible | 0 | 100 | None |
| Shelf, item behind the front one | None | None | None |
| In a hand or on the counter | None | None | None |

Minimum pixels (simulated): no lower limit shows up for clearly visible shelf items. Every bucket with 30 or more items has the right SKU 90% of the time or better, so the simulator's default of 20 px across a can holds on this data and nothing here supports raising it. Right-SKU rate by pixels across a can, clearly visible shelf items: 10-15 px 94.3%, 15-20 px 96.2%, 20-25 px 97.3%, 25-30 px 98.4%, 30-40 px 98.3%, 40-60 px 91.8%. By the short side of the item's own box in pixels, all items: 0-10 px 50.0%, 10-15 px 81.7%, 15-20 px 92.4%, 20-25 px 93.5%, 25-30 px 93.6%.

## Near-identical variants (same shape and size, only the label art differs)

**aldermoor_gold/aldermoor_menthol**: 4055 found, 22 given a sibling's SKU (0.5%).

| true \ predicted | aldermoor_gold | aldermoor_menthol | other | missed |
|---|---|---|---|---|
| aldermoor_gold | 1823 | 14 | 0 | 26 |
| aldermoor_menthol | 8 | 2210 | 0 | 26 |

**arcwave_original/arcwave_tropical/arcwave_berry**: 9856 found, 45 given a sibling's SKU (0.5%).

| true \ predicted | arcwave_original | arcwave_tropical | arcwave_berry | other | missed |
|---|---|---|---|---|---|
| arcwave_original | 3202 | 4 | 11 | 2 | 116 |
| arcwave_tropical | 6 | 3083 | 5 | 2 | 137 |
| arcwave_berry | 9 | 10 | 3519 | 3 | 169 |

**cocoa_crest/peanut_pilot/caramel_crest**: 17366 found, 172 given a sibling's SKU (1.0%).

| true \ predicted | cocoa_crest | peanut_pilot | caramel_crest | other | missed |
|---|---|---|---|---|---|
| cocoa_crest | 5616 | 34 | 32 | 1 | 438 |
| peanut_pilot | 24 | 5927 | 31 | 0 | 396 |
| caramel_crest | 29 | 22 | 5649 | 1 | 325 |

**crunchly_classic/crunchly_sco/crunchly_bbq/sierra_nacho/sierra_ranch**: 6242 found, 10 given a sibling's SKU (0.2%).

| true \ predicted | crunchly_classic | crunchly_sco | crunchly_bbq | sierra_nacho | sierra_ranch | other | missed |
|---|---|---|---|---|---|---|---|
| crunchly_classic | 1170 | 2 | 2 | 1 | 0 | 0 | 124 |
| crunchly_sco | 0 | 1171 | 0 | 0 | 1 | 0 | 108 |
| crunchly_bbq | 1 | 0 | 1314 | 0 | 0 | 2 | 80 |
| sierra_nacho | 0 | 0 | 2 | 1302 | 0 | 0 | 115 |
| sierra_ranch | 0 | 0 | 1 | 0 | 1273 | 0 | 127 |

**fizzo_cola/fizzo_zero/fizzo_cherry/lumen_citrus**: 11697 found, 92 given a sibling's SKU (0.8%).

| true \ predicted | fizzo_cola | fizzo_zero | fizzo_cherry | lumen_citrus | other | missed |
|---|---|---|---|---|---|---|
| fizzo_cola | 3147 | 2 | 38 | 0 | 1 | 61 |
| fizzo_zero | 3 | 2757 | 16 | 2 | 2 | 103 |
| fizzo_cherry | 20 | 7 | 3342 | 1 | 6 | 68 |
| lumen_citrus | 1 | 1 | 1 | 2347 | 3 | 53 |

**orchard_lemon/orchard_peach**: 3770 found, 284 given a sibling's SKU (7.5%).

| true \ predicted | orchard_lemon | orchard_peach | other | missed |
|---|---|---|---|---|
| orchard_lemon | 1995 | 49 | 3 | 70 |
| orchard_peach | 235 | 1481 | 7 | 70 |

**pacer_glacier/pacer_orange/pacer_punch**: 3840 found, 17 given a sibling's SKU (0.4%).

| true \ predicted | pacer_glacier | pacer_orange | pacer_punch | other | missed |
|---|---|---|---|---|---|
| pacer_glacier | 1334 | 1 | 3 | 1 | 73 |
| pacer_orange | 5 | 1232 | 7 | 1 | 44 |
| pacer_punch | 1 | 0 | 1255 | 0 | 49 |

**relieva_ibu/relieva_aceta**: 6327 found, 31 given a sibling's SKU (0.5%).

| true \ predicted | relieva_ibu | relieva_aceta | other | missed |
|---|---|---|---|---|
| relieva_ibu | 3188 | 19 | 3 | 295 |
| relieva_aceta | 12 | 3105 | 0 | 209 |

**ridgeline_original/ridgeline_teriyaki**: 3357 found, 21 given a sibling's SKU (0.6%).

| true \ predicted | ridgeline_original | ridgeline_teriyaki | other | missed |
|---|---|---|---|---|
| ridgeline_original | 1523 | 8 | 3 | 245 |
| ridgeline_teriyaki | 13 | 1808 | 2 | 295 |

**torqueline_5w30/torqueline_10w30**: 4153 found, 83 given a sibling's SKU (2.0%).

| true \ predicted | torqueline_5w30 | torqueline_10w30 | other | missed |
|---|---|---|---|---|
| torqueline_5w30 | 2006 | 43 | 0 | 304 |
| torqueline_10w30 | 40 | 2064 | 0 | 328 |

**voltlink_usbc/voltlink_micro**: 3810 found, 16 given a sibling's SKU (0.4%).

| true \ predicted | voltlink_usbc | voltlink_micro | other | missed |
|---|---|---|---|---|
| voltlink_usbc | 2121 | 10 | 1 | 343 |
| voltlink_micro | 6 | 1671 | 1 | 252 |

## Speed

| device | ms per 640 px tile | tiles per s | ms per whole frame | frames per s |
|---|---|---|---|---|
| mps | 8.2 | 121.5 | 48.2 (1520x2688, 15 tiles) | 20.73 |
| cpu | 39.8 | 25.1 | 300.3 (1520x2688, 15 tiles) | 3.33 |
