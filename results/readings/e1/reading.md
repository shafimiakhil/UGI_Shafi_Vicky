# ugi_e1_ps against ugi_r1_none, seed 0: pre-registered reading (CLAUDE.md 13.6)

**Reading (rule of 13.7): no gain.** All tiles does not pass the 13.6 rule; the B4 tile set does not pass it.

Gain in AP with CIRCUNVALACION excluded, inside the ejido: +0.0164 [-0.0015, +0.0346] (`runs_itareo\ugi\summary_r/summary_e1_ejido.json`); the baseline's SD over seeds is 0.0030. With 50,000 resamples the interval is [-0.0012, +0.0338]. Positive in 3 of 4 folds: +0.0439, -0.0319, +0.0142, +0.0086.

Tests: gain at least 0.010: yes; interval above zero: no; positive in at least 3 of 4 folds: yes.

**B4 tile set** (1,240 tiles with all three dates, `runs_itareo\ugi\summary_r/summary_e1_ejido_b4.json`): gain +0.0161 [-0.0006, +0.0335]; with 50,000 resamples [-0.0008, +0.0335]; positive in 3 of 4 folds: +0.0185, -0.0155, +0.0352, +0.0246. By the same tests: no gain.

**Registration caveat (13.7).** E1 removes one constant per date, which removes the median only. Residual per 1 km block, median and 90th percentile: 20220424 2.39 and 4.15 m, 20220910 1.76 and 3.31 m, 20221028 1.06 and 2.45 m. A plane per axis would cut 20220424 to 0.86 m (`docs/evidence_phase2/ugi_seg/e1_ps_residuals/summary.json`). E1b tests that.

| score | baseline | arm | difference | 95 % block interval | full grid | B4 tiles |
|---|---|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.4494 | 0.4658 | +0.0164 | [-0.0015, +0.0346] | +0.0195 [+0.0019, +0.0377] | +0.0161 [-0.0006, +0.0335] |
| AP | 0.6159 | 0.6223 | +0.0064 | [-0.0067, +0.0193] | +0.0089 [-0.0042, +0.0218] | +0.0006 [-0.0109, +0.0128] |
| AP, vegetated pixels | 0.5250 | 0.5417 | +0.0167 | [+0.0017, +0.0315] | +0.0198 [+0.0045, +0.0345] | +0.0072 [-0.0079, +0.0215] |
| AUROC, CIRCUNVALACION excluded | 0.8882 | 0.8781 | -0.0101 | [-0.0165, -0.0035] | -0.0089 [-0.0153, -0.0024] | -0.0102 [-0.0176, -0.0026] |
| AUROC | 0.9116 | 0.9031 | -0.0085 | [-0.0136, -0.0033] | -0.0075 [-0.0126, -0.0025] | -0.0085 [-0.0142, -0.0028] |
| IoU, CIRCUNVALACION excluded | 0.3188 | 0.3289 | +0.0100 | [-0.0037, +0.0234] | +0.0113 [-0.0024, +0.0246] | +0.0215 [+0.0081, +0.0348] |
| IoU | 0.4271 | 0.4324 | +0.0053 | [-0.0064, +0.0162] | +0.0067 [-0.0049, +0.0176] | +0.0106 [-0.0005, +0.0223] |
| F1, CIRCUNVALACION excluded | 0.4835 | 0.4949 | +0.0114 | [-0.0043, +0.0268] | +0.0129 [-0.0026, +0.0281] | +0.0234 [+0.0087, +0.0379] |
| F1 | 0.5985 | 0.6037 | +0.0052 | [-0.0062, +0.0160] | +0.0066 [-0.0048, +0.0173] | +0.0096 [-0.0005, +0.0206] |
| Object recall, tolerant | 0.4932 | 0.4632 | -0.0300 | [-0.0469, -0.0136] | -0.0300 [-0.0469, -0.0136] | -0.0096 [-0.0250, +0.0058] |
| Object recall, strict | 0.4607 | 0.4361 | -0.0246 | [-0.0408, -0.0085] | -0.0246 [-0.0408, -0.0085] | -0.0070 [-0.0218, +0.0077] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4872 | 0.4569 | -0.0304 | [-0.0475, -0.0138] | -0.0304 [-0.0475, -0.0138] | -0.0098 [-0.0254, +0.0059] |
| Object recall, tolerant, under 6 m wide | 0.0994 | 0.0682 | -0.0312 | [-0.0619, -0.0033] | -0.0312 [-0.0619, -0.0033] | -0.0195 [-0.0456, +0.0070] |
| Object recall, strict, under 6 m wide | 0.0767 | 0.0511 | -0.0256 | [-0.0564, +0.0000] | -0.0256 [-0.0564, +0.0000] | -0.0098 [-0.0303, +0.0095] |

Precision by area, inside the ejido: strict 0.7907 against 0.7890, tolerant 0.8034 against 0.8015 (baseline against arm; 5645 and 5556 components; `components_rule7.csv` of each fold). Full grid: strict 0.7844 against 0.7850, tolerant 0.7976 against 0.7982 (5750 and 5606 components).

The class table is inside the ejido. On the full grid no recall or coverage in it differs by more than 0.0000, because the inventory lies inside the ejido.

| class | polygons | tolerant recall, baseline | arm | area-weighted coverage, baseline | arm |
|---|---|---|---|---|---|
| all | 3167 | 0.4932 | 0.4632 | 0.5409 | 0.5417 |
| group: Road greens | 1166 | 0.3696 | 0.3105 | 0.8082 | 0.7905 |
| group: Squares | 1174 | 0.6346 | 0.6201 | 0.6394 | 0.6531 |
| group: Parks and green areas | 479 | 0.4885 | 0.5073 | 0.2182 | 0.2579 |
| group: Water corridors | 93 | 0.4516 | 0.3548 | 0.4296 | 0.4195 |
| group: Rail verges | 107 | 0.6075 | 0.5981 | 0.4873 | 0.4831 |
| group: NO DEFINIDO | 148 | 0.3041 | 0.2500 | 0.6486 | 0.6288 |
| size: <500 m2 | 811 | 0.2072 | 0.1714 | 0.1987 | 0.1616 |
| size: 500-2000 m2 | 1092 | 0.4881 | 0.4441 | 0.4603 | 0.4277 |
| size: 2000-10000 m2 | 953 | 0.7261 | 0.7083 | 0.6605 | 0.6481 |
| size: >10000 m2 | 311 | 0.5434 | 0.5402 | 0.5275 | 0.5325 |
| width: <3 m | 103 | 0.0388 | 0.0291 | 0.0474 | 0.0413 |
| width: 3-6 m | 249 | 0.1245 | 0.0843 | 0.1210 | 0.1221 |
| width: 6-12 m | 558 | 0.2240 | 0.1756 | 0.2593 | 0.2164 |
| width: 12-25 m | 958 | 0.5240 | 0.4666 | 0.4212 | 0.3776 |
| width: >25 m | 1299 | 0.6928 | 0.6913 | 0.5647 | 0.5729 |

Files: `paired_table.csv`, `per_fold.csv`, `objects_by_class.csv` in this folder. The per-fold values are the runs' `ejido/metrics_rule7.json` (B4 rows: `ejido_b4/metrics_rule7.json`), seed 0. One seed: a helping input is confirmed with seeds 1-2 before it enters the fusion design. Coverage is the pixel-weighted mean of per-polygon values stored at 4 decimals (`objects_rule7.csv`), so its last digit can differ by one from the exact ratio.

## Diagnostics of CLAUDE.md 13.8

The reading stands: no gain. D1 and D3 neither change nor explain it. Paths from `docs/evidence_phase2/ugi_seg/diagnostics/`. What the review established after D2 ran, E1 included: `d2_blank/after_d2.md`.

- **D1, the gate.** Open at the checkpoint each run kept: weight norm 0.86, 0.91, 0.85, 0.94, added term 0.18, 0.20, 0.16, 0.16 of the stage features' RMS, folds 1-4 (`d1_gate/d1_gate.csv`).
- **D1, the output.** On 64 tiles per fold with all three dates, replacing the added term by its tile mean changes P by 0.028, 0.015, 0.027, 0.013, mostly a fall, and the decision on 2.9, 1.9, 3.1, 1.3 % of pixels; baseline seeds 0 and 1 differ there by 0.022, 0.014, 0.031, 0.016 and on 2.1, 1.7, 4.2, 1.5 % (`review_logs/d1/d1_output_effect.csv`). So the PlanetScope signal reaches the output. Sizes, not scores.
- **D2.** E1 has no D2: 13.8 names the E1b and E2 checkpoints; E1b's numbers belong to other weights (`d2_blank/d2_table.csv`).
- **D3.** AUROC on a class-balanced draw of vegetated pixels, with E1's constant registration. Highest of five PlanetScope features: ejido 0.6101 (24 Apr, auc_min 0.5684, in fold 4 0.4030), sq8km 0.8339 on 32 tiles (Neo NDVI in 33 px 0.8761, `README.md`); the other four 0.5357-0.5630, 0.7015-0.8297 (`d3_screen/d3_auroc.csv`, rows `psconst_*`). The registered statistic reads E1b's plane: 0.6091, so E1c (`d3_screen/d3.json`).
- **What 13.8's rule gives.** The highest is not near 0.5, so the rule does not stop PlanetScope experiments. In the ejido it just reaches 0.60 (plane values: in 66.9 % of new 30-tile draws, `README.md`), but E1 has no D2 to call unchanged, so the second branch cannot be read. Neither branch applies.
- **What it does not give.** Why E1 shows no gain, or whether PlanetScope adds anything the model lacks. The rise of +0.0164 is 1.32 times the 81-based SD 0.0124 and AUROC falls by 0.0101, SD 0.0069 (`../e1/paired_table.csv`, `../e1b/registration_check/combos.csv`); one seed does not separate these from seed noise.
- **D3b, an addition to 13.8's list, on E1b's plane values; not run on E1's constant.** Gain over Neo pixel NDVI + CHM + Neo NDVI in 33 px, the same-place fit of `README.md`, read inside tiles: ejido +0.0122 [+0.0058, +0.0189] for 24 Apr, -0.0066 to -0.0017 the other four; sq8km -0.0067 to +0.0163. The window was chosen after the results; over 17 and 65 px: ejido -0.0147 to +0.0300, sq8km -0.0029 to +0.0393. Over the baseline's score, same fit: ejido -0.0043 to -0.0015, sq8km -0.0109 to +0.0038 (`README.md`, `review_logs/d3/TABLE_d3b_gains.csv`, `sameplace_*`). A late add-on in AUROC: not AP, not what a network trained with the input does.

