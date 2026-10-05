# Hand and held-item detector: held-out SIMULATED frames

All numbers are on simulated frames from the browser store simulator (one store model, invented package art, block-shaped hands). They are not accuracy on real footage.

Held-out split: 484 whole frames from seeds 3000 to 3009 (10 scenes), none used for training. Tiled inference on the whole frame, confidence 0.25, no person box.

| measure | sim_sku | sim_sku_hands_v2 | sim_sku_hands_v3 | sim_sku_hands_v3 + second look |
|---|---|---|---|---|
| item in a hand, 20 px or more: right SKU (target 85%) | 67.0% (235 of 351) | 74.1% (260 of 351) | 87.5% (307 of 351) | 88.3% (310 of 351) |
| item in a hand, 20 px or more: found | 75.2% (264 of 351) | 78.9% (277 of 351) | 90.6% (318 of 351) | 91.7% (322 of 351) |
| hand on a reaching arm found, IoU 0.5 (target 90%) | 0.0% (0 of 301) | 56.8% (171 of 301) | 76.4% (230 of 301) | 79.1% (238 of 301) |
| hand on a reaching arm found, IoU 0.3 | 0.0% (0 of 301) | 61.1% (184 of 301) | 80.1% (241 of 301) | 81.7% (246 of 301) |
| a hand box holds the true hand centre (reaching arm) | 0.0% (0 of 301) | 65.5% (197 of 301) | 83.4% (251 of 301) | 84.7% (255 of 301) |
| every labelled hand found, IoU 0.5 | 0.0% (0 of 995) | 63.9% (636 of 995) | 78.9% (785 of 995) | 80.6% (802 of 995) |
| hand boxes that are a hand (IoU 0.3) | n/a | 80.7% | 81.2% | 80.2% |
| on counter, 20 px or more: right SKU | 65.9% (58 of 88) | 70.5% (62 of 88) | 76.1% (67 of 88) | 77.3% (68 of 88) |
| shelf front clear, 20 px or more: right SKU | 97.9% (12144 of 12405) | 98.6% (12233 of 12405) | 99.0% (12275 of 12405) | 99.0% (12280 of 12405) |
| shelf front partly hidden, 20 px or more: right SKU | 95.7% (18418 of 19252) | 97.0% (18675 of 19252) | 97.8% (18829 of 19252) | 98.0% (18861 of 19252) |
| backstock, 20 px or more: right SKU | 90.2% (11041 of 12240) | 93.2% (11412 of 12240) | 95.3% (11666 of 12240) | 95.5% (11689 of 12240) |
| item boxes that are an item | 90.2% | 88.6% | 89.0% | 88.8% |

## Item in a hand: right SKU by pixels across a 6.6 cm can

| px | items | sim_sku | sim_sku_hands_v2 | sim_sku_hands_v3 | sim_sku_hands_v3 + second look |
|---|---|---|---|---|---|
| 25-30 | 3 | 33.3% | 66.7% | 33.3% | 33.3% |
| 30-40 | 17 | 35.3% | 52.9% | 70.6% | 70.6% |
| 40-60 | 122 | 55.7% | 76.2% | 87.7% | 89.3% |
| 60-100 | 148 | 83.8% | 83.8% | 93.9% | 94.6% |
| 100+ | 61 | 59.0% | 52.5% | 78.7% | 78.7% |

## sim_sku

Item in a hand at 20 px or more, right SKU, by camera kind: checkout 72.4% (63 of 87), cooler 67.5% (27 of 40), shelf 64.7% (145 of 224).
By share of the item that shows: 25 to 50% 51.4% (55 of 107), 50 to 80% 75.3% (119 of 158), 80%+ 70.9% (61 of 86).
Reaching hand found (IoU 0.5) by the short side of the hand box: 5-15 px 0.0% (0 of 12), 15-30 px 0.0% (0 of 36), 30-60 px 0.0% (0 of 68), 60+ px 0.0% (0 of 185).
Near-identical variants held in a hand, given a sibling's SKU: arcwave_original/arcwave_tropical/arcwave_berry 0 of 19 found, cocoa_crest/peanut_pilot/caramel_crest 4 of 105 found, crunchly_classic/crunchly_sco/crunchly_bbq/sierra_nacho/sierra_ranch 0 of 13 found, fizzo_cola/fizzo_zero/fizzo_cherry/lumen_citrus 0 of 18 found, orchard_lemon/orchard_peach 1 of 1 found, pacer_glacier/pacer_orange/pacer_punch 0 of 2 found, relieva_ibu/relieva_aceta 0 of 27 found, ridgeline_original/ridgeline_teriyaki 0 of 12 found, torqueline_5w30/torqueline_10w30 1 of 13 found, voltlink_usbc/voltlink_micro 0 of 14 found.

| device | ms per 640 px tile | ms per whole frame | frames per s |
|---|---|---|---|
| mps | 10.3 | 51.0 (1520x2688, 15 tiles) | 19.59 |
| cpu | 72.8 | 555.7 (1520x2688, 15 tiles) | 1.8 |

## sim_sku_hands_v2

Item in a hand at 20 px or more, right SKU, by camera kind: checkout 80.5% (70 of 87), cooler 82.5% (33 of 40), shelf 70.1% (157 of 224).
By share of the item that shows: 25 to 50% 54.2% (58 of 107), 50 to 80% 81.0% (128 of 158), 80%+ 86.1% (74 of 86).
Reaching hand found (IoU 0.5) by the short side of the hand box: 5-15 px 0.0% (0 of 12), 15-30 px 2.8% (1 of 36), 30-60 px 35.3% (24 of 68), 60+ px 78.9% (146 of 185).
Near-identical variants held in a hand, given a sibling's SKU: arcwave_original/arcwave_tropical/arcwave_berry 0 of 23 found, cocoa_crest/peanut_pilot/caramel_crest 1 of 109 found, crunchly_classic/crunchly_sco/crunchly_bbq/sierra_nacho/sierra_ranch 0 of 13 found, fizzo_cola/fizzo_zero/fizzo_cherry/lumen_citrus 0 of 20 found, orchard_lemon/orchard_peach 0 of 1 found, pacer_glacier/pacer_orange/pacer_punch 0 of 4 found, relieva_ibu/relieva_aceta 0 of 25 found, ridgeline_original/ridgeline_teriyaki 1 of 14 found, torqueline_5w30/torqueline_10w30 0 of 12 found, voltlink_usbc/voltlink_micro 0 of 14 found.

Held-out tiles: mAP50 0.949, mAP50-95 0.821.

| class | AP50 | AP50-95 |
|---|---|---|
| fizzo_cola | 0.975 | 0.865 |
| fizzo_zero | 0.957 | 0.796 |
| fizzo_cherry | 0.957 | 0.822 |
| lumen_citrus | 0.943 | 0.783 |
| arcwave_original | 0.967 | 0.852 |
| arcwave_tropical | 0.966 | 0.852 |
| arcwave_berry | 0.975 | 0.847 |
| clearwell_water | 0.956 | 0.759 |
| orchard_lemon | 0.930 | 0.842 |
| orchard_peach | 0.955 | 0.846 |
| pacer_glacier | 0.910 | 0.738 |
| pacer_orange | 0.974 | 0.866 |
| pacer_punch | 0.951 | 0.829 |
| crunchly_classic | 0.947 | 0.840 |
| crunchly_sco | 0.958 | 0.880 |
| crunchly_bbq | 0.967 | 0.869 |
| sierra_nacho | 0.950 | 0.856 |
| sierra_ranch | 0.970 | 0.891 |
| ridgeline_original | 0.959 | 0.848 |
| ridgeline_teriyaki | 0.971 | 0.868 |
| oatfield_cookies | 0.928 | 0.834 |
| morning_hoops | 0.919 | 0.766 |
| torqueline_5w30 | 0.955 | 0.812 |
| torqueline_10w30 | 0.935 | 0.792 |
| relieva_ibu | 0.960 | 0.834 |
| relieva_aceta | 0.952 | 0.816 |
| voltlink_usbc | 0.967 | 0.868 |
| voltlink_micro | 0.953 | 0.831 |
| cocoa_crest | 0.967 | 0.812 |
| peanut_pilot | 0.959 | 0.794 |
| caramel_crest | 0.961 | 0.789 |
| aldermoor_gold | 0.977 | 0.856 |
| aldermoor_menthol | 0.986 | 0.862 |
| hand | 0.702 | 0.487 |

| device | ms per 640 px tile | ms per whole frame | frames per s |
|---|---|---|---|
| mps | 7.9 | 47.8 (1520x2688, 15 tiles) | 20.9 |
| cpu | 56.8 | 694.5 (1520x2688, 15 tiles) | 1.44 |

## sim_sku_hands_v3

Item in a hand at 20 px or more, right SKU, by camera kind: checkout 92.0% (80 of 87), cooler 85.0% (34 of 40), shelf 86.2% (193 of 224).
By share of the item that shows: 25 to 50% 72.9% (78 of 107), 50 to 80% 92.4% (146 of 158), 80%+ 96.5% (83 of 86).
Reaching hand found (IoU 0.5) by the short side of the hand box: 5-15 px 0.0% (0 of 12), 15-30 px 30.6% (11 of 36), 30-60 px 64.7% (44 of 68), 60+ px 94.6% (175 of 185).
Near-identical variants held in a hand, given a sibling's SKU: arcwave_original/arcwave_tropical/arcwave_berry 0 of 23 found, cocoa_crest/peanut_pilot/caramel_crest 3 of 123 found, crunchly_classic/crunchly_sco/crunchly_bbq/sierra_nacho/sierra_ranch 0 of 15 found, fizzo_cola/fizzo_zero/fizzo_cherry/lumen_citrus 0 of 22 found, orchard_lemon/orchard_peach 0 of 1 found, pacer_glacier/pacer_orange/pacer_punch 0 of 7 found, relieva_ibu/relieva_aceta 3 of 34 found, ridgeline_original/ridgeline_teriyaki 0 of 18 found, torqueline_5w30/torqueline_10w30 0 of 14 found, voltlink_usbc/voltlink_micro 0 of 14 found.

Held-out tiles: mAP50 0.973, mAP50-95 0.872.

| class | AP50 | AP50-95 |
|---|---|---|
| fizzo_cola | 0.983 | 0.903 |
| fizzo_zero | 0.982 | 0.844 |
| fizzo_cherry | 0.980 | 0.880 |
| lumen_citrus | 0.978 | 0.844 |
| arcwave_original | 0.984 | 0.894 |
| arcwave_tropical | 0.983 | 0.902 |
| arcwave_berry | 0.987 | 0.894 |
| clearwell_water | 0.981 | 0.832 |
| orchard_lemon | 0.965 | 0.895 |
| orchard_peach | 0.987 | 0.907 |
| pacer_glacier | 0.960 | 0.834 |
| pacer_orange | 0.988 | 0.920 |
| pacer_punch | 0.970 | 0.877 |
| crunchly_classic | 0.958 | 0.876 |
| crunchly_sco | 0.979 | 0.923 |
| crunchly_bbq | 0.979 | 0.914 |
| sierra_nacho | 0.964 | 0.891 |
| sierra_ranch | 0.972 | 0.916 |
| ridgeline_original | 0.978 | 0.906 |
| ridgeline_teriyaki | 0.981 | 0.910 |
| oatfield_cookies | 0.956 | 0.874 |
| morning_hoops | 0.966 | 0.872 |
| torqueline_5w30 | 0.978 | 0.877 |
| torqueline_10w30 | 0.957 | 0.841 |
| relieva_ibu | 0.974 | 0.873 |
| relieva_aceta | 0.972 | 0.872 |
| voltlink_usbc | 0.983 | 0.917 |
| voltlink_micro | 0.972 | 0.876 |
| cocoa_crest | 0.985 | 0.859 |
| peanut_pilot | 0.984 | 0.848 |
| caramel_crest | 0.981 | 0.834 |
| aldermoor_gold | 0.984 | 0.850 |
| aldermoor_menthol | 0.988 | 0.854 |
| hand | 0.848 | 0.645 |

| device | ms per 640 px tile | ms per whole frame | frames per s |
|---|---|---|---|
| mps | 11.8 | 51.4 (1520x2688, 15 tiles) | 19.46 |
| cpu | 65.5 | 543.8 (1520x2688, 15 tiles) | 1.84 |

## sim_sku_hands_v3 + second look

Item in a hand at 20 px or more, right SKU, by camera kind: checkout 92.0% (80 of 87), cooler 85.0% (34 of 40), shelf 87.5% (196 of 224).
By share of the item that shows: 25 to 50% 75.7% (81 of 107), 50 to 80% 93.0% (147 of 158), 80%+ 95.3% (82 of 86).
Reaching hand found (IoU 0.5) by the short side of the hand box: 5-15 px 8.3% (1 of 12), 15-30 px 36.1% (13 of 36), 30-60 px 69.1% (47 of 68), 60+ px 95.7% (177 of 185).
Near-identical variants held in a hand, given a sibling's SKU: arcwave_original/arcwave_tropical/arcwave_berry 0 of 24 found, cocoa_crest/peanut_pilot/caramel_crest 1 of 124 found, crunchly_classic/crunchly_sco/crunchly_bbq/sierra_nacho/sierra_ranch 0 of 14 found, fizzo_cola/fizzo_zero/fizzo_cherry/lumen_citrus 0 of 22 found, orchard_lemon/orchard_peach 0 of 1 found, pacer_glacier/pacer_orange/pacer_punch 0 of 7 found, relieva_ibu/relieva_aceta 3 of 35 found, ridgeline_original/ridgeline_teriyaki 0 of 19 found, torqueline_5w30/torqueline_10w30 0 of 14 found, voltlink_usbc/voltlink_micro 0 of 15 found.

## Acceptance

- sim_sku: in_hand_sku_right_20px: NOT met: 0.6695 against 0.85
- sim_sku: hand_recall_reach: NOT met: 0.0 against 0.9
- sim_sku_hands_v2: in_hand_sku_right_20px: NOT met: 0.7407 against 0.85
- sim_sku_hands_v2: hand_recall_reach: NOT met: 0.5681 against 0.9
- sim_sku_hands_v3: in_hand_sku_right_20px: met: 0.8746 against 0.85
- sim_sku_hands_v3: hand_recall_reach: NOT met: 0.7641 against 0.9
- sim_sku_hands_v3 + second look: in_hand_sku_right_20px: met: 0.8832 against 0.85
- sim_sku_hands_v3 + second look: hand_recall_reach: NOT met: 0.7907 against 0.9
