# ugi_e1c_ps1m against ugi_r1_none, seed 0: pre-registered reading (CLAUDE.md 13.8)

**Reading (centre sq8km, primary, 13.8): no gain.**

Gain in AP inside the centre: +0.0102 (`runs_itareo\ugi\summary_r/summary_e1c_centre_sq8km.json`), against the threshold 0.0540, three times the SD 0.0180 of the 81 one-seed-per-fold baseline realisations inside it (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`; CLAUDE.md 13.8). Positive in 3 of 4 folds: -0.0184, +0.0274, +0.0345, +0.0264. Mean per-fold AP difference: +0.0175; same sign as the pooled gain, which decides (13.8).

Block interval (13.8: the block bootstrap does not cover seed noise here, so it is no test of a pass; below the threshold it separates undecided from no gain, as the undecided clause is written): [-0.0182, +0.0322]; with 50,000 resamples [-0.0174, +0.0330].

Tests: gain at least 0.0540: no; positive in at least 3 of 4 folds: yes.

The block interval holds no training-seed variance, so the threshold is the test that covers it. Between baseline seeds inside this centre (b0, b1, b2: seeds 0, 1, 2), the same interval gives b1 - b0 +0.0230 [-0.0149, +0.0506]; b2 - b0 -0.0293 [-0.0515, -0.0068]; b2 - b1 -0.0523 [-0.0840, -0.0096] (`docs/evidence_phase2/ugi_centre/seed_pairs.csv`).

| score, inside the centre | baseline | arm | difference | 95 % block interval |
|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.6530 | 0.6632 | +0.0102 | [-0.0182, +0.0322] |
| AP | 0.6530 | 0.6632 | +0.0102 | [-0.0182, +0.0322] |
| AP, vegetated pixels | 0.7068 | 0.7130 | +0.0062 | [-0.0196, +0.0305] |
| AUROC, CIRCUNVALACION excluded | 0.9255 | 0.9001 | -0.0254 | [-0.0443, -0.0072] |
| AUROC | 0.9255 | 0.9001 | -0.0254 | [-0.0443, -0.0072] |
| IoU, CIRCUNVALACION excluded | 0.3817 | 0.4441 | +0.0624 | [+0.0088, +0.1075] |
| IoU | 0.3817 | 0.4441 | +0.0624 | [+0.0088, +0.1075] |
| F1, CIRCUNVALACION excluded | 0.5525 | 0.6150 | +0.0625 | [+0.0080, +0.1109] |
| F1 | 0.5525 | 0.6150 | +0.0625 | [+0.0080, +0.1109] |
| Object recall, tolerant | 0.4624 | 0.5066 | +0.0442 | [+0.0148, +0.0753] |
| Object recall, strict | 0.4256 | 0.4728 | +0.0471 | [+0.0159, +0.0813] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4624 | 0.5066 | +0.0442 | [+0.0148, +0.0753] |
| Object recall, tolerant, under 6 m wide | 0.1017 | 0.0508 | -0.0508 | [-0.1346, +0.0192] |
| Object recall, strict, under 6 m wide | 0.0847 | 0.0339 | -0.0508 | [-0.1316, +0.0189] |

Precision by area, inside the centre: strict 0.8470 against 0.8427, tolerant 0.8669 against 0.8635 (965 and 1079 components). Classes: `objects_by_class_centre.csv`.

## The ejido, secondary

**Secondary reading (rule of 13.7 on the tests of 13.8): no gain.** All tiles: no gain. B4 tile set: no gain.

Gain in AP with CIRCUNVALACION excluded, inside the ejido: -0.0015 [-0.0276, +0.0232] (`runs_itareo\ugi\summary_r/summary_e1c_ejido.json`); threshold 0.0372, three times the SD 0.0124 of the 81 one-seed-per-fold baseline realisations (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`; CLAUDE.md 13.8). With 50,000 resamples the interval is [-0.0283, +0.0232]. Positive in 2 of 4 folds: +0.0506, -0.0176, -0.0287, +0.0008. Mean per-fold AP difference: +0.0013; its sign DISAGREES with the pooled gain, which decides (13.8).

Tests: gain at least 0.0372: no; interval above zero: no; positive in at least 3 of 4 folds: no.

**B4 tile set** (1,240 tiles with all three dates, `runs_itareo\ugi\summary_r/summary_e1c_ejido_b4.json`): gain -0.0026 [-0.0326, +0.0228]; with 50,000 resamples [-0.0310, +0.0228]; positive in 3 of 4 folds: +0.0012, -0.0145, +0.0009, +0.0347. Threshold 0.0324, three times the SD 0.0108 computed the same way on these tiles. Mean per-fold AP difference: +0.0056; its sign DISAGREES with the pooled gain, which decides (13.8). By the same tests: no gain.

**What E1c holds and its registration (13.8).** The three PlanetScope dates with the plane per date of E1b removed inside one bilinear resampling to 1 m, into the stem as zero-initialised conv1 channels with one availability channel per date (the mechanism of E2). Residual per 1,024 m block with the plane, low-pass, median and 90th percentile, measured for E1b at 4 m: 20220424 0.85 and 2.72 m, 20220910 1.0 and 2.28 m, 20221028 0.89 and 1.89 m (`docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json`).

| score | baseline | arm | difference | 95 % block interval | full grid | B4 tiles |
|---|---|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.4494 | 0.4479 | -0.0015 | [-0.0276, +0.0232] | +0.0013 [-0.0242, +0.0257] | -0.0026 [-0.0326, +0.0228] |
| AP | 0.6159 | 0.6162 | +0.0002 | [-0.0190, +0.0185] | +0.0026 [-0.0162, +0.0205] | -0.0043 [-0.0241, +0.0120] |
| AP, vegetated pixels | 0.5250 | 0.5166 | -0.0083 | [-0.0255, +0.0097] | -0.0061 [-0.0230, +0.0121] | +0.0004 [-0.0162, +0.0168] |
| AUROC, CIRCUNVALACION excluded | 0.8882 | 0.8828 | -0.0054 | [-0.0146, +0.0035] | -0.0043 [-0.0135, +0.0045] | -0.0088 [-0.0193, +0.0011] |
| AUROC | 0.9116 | 0.9072 | -0.0044 | [-0.0115, +0.0025] | -0.0035 [-0.0107, +0.0033] | -0.0069 [-0.0145, +0.0003] |
| IoU, CIRCUNVALACION excluded | 0.3188 | 0.3254 | +0.0066 | [-0.0120, +0.0259] | +0.0077 [-0.0106, +0.0267] | +0.0202 [-0.0013, +0.0411] |
| IoU | 0.4271 | 0.4282 | +0.0011 | [-0.0152, +0.0177] | +0.0024 [-0.0137, +0.0191] | +0.0088 [-0.0101, +0.0263] |
| F1, CIRCUNVALACION excluded | 0.4835 | 0.4910 | +0.0075 | [-0.0137, +0.0300] | +0.0088 [-0.0120, +0.0310] | +0.0221 [-0.0014, +0.0451] |
| F1 | 0.5985 | 0.5996 | +0.0011 | [-0.0148, +0.0177] | +0.0023 [-0.0131, +0.0190] | +0.0080 [-0.0090, +0.0246] |
| Object recall, tolerant | 0.4932 | 0.5077 | +0.0145 | [-0.0024, +0.0305] | +0.0145 [-0.0024, +0.0305] | +0.0389 [+0.0240, +0.0534] |
| Object recall, strict | 0.4607 | 0.4787 | +0.0180 | [+0.0010, +0.0343] | +0.0180 [+0.0010, +0.0343] | +0.0389 [+0.0242, +0.0535] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4872 | 0.5019 | +0.0147 | [-0.0024, +0.0309] | +0.0147 [-0.0024, +0.0309] | +0.0394 [+0.0244, +0.0542] |
| Object recall, tolerant, under 6 m wide | 0.0994 | 0.0795 | -0.0199 | [-0.0572, +0.0150] | -0.0199 [-0.0572, +0.0150] | -0.0033 [-0.0366, +0.0310] |
| Object recall, strict, under 6 m wide | 0.0767 | 0.0682 | -0.0085 | [-0.0438, +0.0256] | -0.0085 [-0.0438, +0.0256] | +0.0098 [-0.0229, +0.0447] |

Precision by area, inside the ejido: strict 0.7907 against 0.7571, tolerant 0.8034 against 0.7703 (baseline against arm; 5645 and 6036 components; `components_rule7.csv` of each fold). Full grid: strict 0.7844 against 0.7527, tolerant 0.7976 against 0.7668 (5750 and 6060 components).

The class table is inside the ejido. On the full grid no recall or coverage in it differs by more than 0.0000, because all but 263 inventory pixels lie inside the ejido.

| class | polygons | tolerant recall, baseline | arm | area-weighted coverage, baseline | arm |
|---|---|---|---|---|---|
| all | 3167 | 0.4932 | 0.5077 | 0.5409 | 0.5651 |
| group: Road greens | 1166 | 0.3696 | 0.3611 | 0.8082 | 0.8174 |
| group: Squares | 1174 | 0.6346 | 0.6601 | 0.6394 | 0.6818 |
| group: Parks and green areas | 479 | 0.4885 | 0.5219 | 0.2182 | 0.2969 |
| group: Water corridors | 93 | 0.4516 | 0.4946 | 0.4296 | 0.4296 |
| group: Rail verges | 107 | 0.6075 | 0.6916 | 0.4873 | 0.5239 |
| group: NO DEFINIDO | 148 | 0.3041 | 0.2838 | 0.6486 | 0.6143 |
| size: <500 m2 | 811 | 0.2072 | 0.2133 | 0.1987 | 0.2050 |
| size: 500-2000 m2 | 1092 | 0.4881 | 0.4927 | 0.4603 | 0.4679 |
| size: 2000-10000 m2 | 953 | 0.7261 | 0.7482 | 0.6605 | 0.6844 |
| size: >10000 m2 | 311 | 0.5434 | 0.5916 | 0.5275 | 0.5526 |
| width: <3 m | 103 | 0.0388 | 0.0291 | 0.0474 | 0.0614 |
| width: 3-6 m | 249 | 0.1245 | 0.1004 | 0.1210 | 0.1423 |
| width: 6-12 m | 558 | 0.2240 | 0.2473 | 0.2593 | 0.2814 |
| width: 12-25 m | 958 | 0.5240 | 0.5240 | 0.4212 | 0.4069 |
| width: >25 m | 1299 | 0.6928 | 0.7236 | 0.5647 | 0.5947 |

Files: `paired_table.csv`, `per_fold.csv`, `objects_by_class.csv` in this folder. The per-fold values come from the fine histograms of the runs' `ejido/boot_rule7.npz` (B4 rows: `ejido_b4/boot_rule7.npz`); `metrics_rule7.json` holds them at 5 decimals. Seed 0. One seed: a helping input is confirmed with seeds 1-2 before it enters the fusion design. Coverage is the pixel-weighted mean of per-polygon values stored at 4 decimals (`objects_rule7.csv`), so its last digit can differ by one from the exact ratio. The values inside the centre come from the runs' `centre_sq8km/boot_rule7.npz`, `metrics_rule7.json` and `objects_rule7.csv`.
