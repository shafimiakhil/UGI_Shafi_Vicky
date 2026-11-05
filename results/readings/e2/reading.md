# ugi_e2_tex against ugi_r1_none, seed 0: pre-registered reading (CLAUDE.md 13.6)

**Reading: no gain.**

Gain in AP with CIRCUNVALACION excluded, inside the ejido: -0.0054 [-0.0229, +0.0137] (`runs_itareo\ugi\summary_r/summary_e2_ejido.json`); the baseline's SD over seeds is 0.0030. With 50,000 resamples the interval is [-0.0236, +0.0136]. Positive in 1 of 4 folds: -0.0118, -0.0348, -0.0234, +0.0432.

Tests: gain at least 0.010: no; interval above zero: no; positive in at least 3 of 4 folds: no.

**What E2 held.** Eight new channels in conv1: six layers and two masks. The PAN lawn layers var3, var9, hom3 and hom9 (`work_ms/derived/features/neo2022_v3_pan_lawn_grid1m.tif`), and ndvi_orientcons_41m and ndvi_glcmcorr_41m (`work_ms/derived/features/neo2022_v3_ndvi_texture_grid1m.tif`). 13.6 leaves the GLCM out if it cannot be computed for every tile within about 1 h of CPU. The texture script's own check stopped at 5,482 CPU s, a sum that also holds the reads and the other layers, in eight processes beside E1 (`work_ms/derived/features/neo2022_v3_ndvi_texture_log.json`). The GLCM stayed on an estimate, before any E2 score (`work_ms/CHANGES.md` rows 94 and 98). Timed apart afterwards, the four GLCMs took 2,802 CPU s over the 36 blocks, about 2,623 s for the 2,116 tiles, 73 % of the hour (`review_logs/glcm_cost_verify.json`).

The two NDVI layers come from the texture screen, whose keep list is one list for all folds, so each fold's labels enter it through the sets that hold that fold. By the gain over NDVI + CHM per set, one of the eight choices would differ: without fold 1, ndvi_coherence_41m 0.0564 against ndvi_orientcons_41m 0.0554 (`docs/evidence_phase2/texture_screen/auroc.csv`).

**Inside the centre**, for information, since E2 keeps this reading (13.7): `centre_rescoring/README.md`; why it falls: `centre_fall/README.md`.

| score | baseline | arm | difference | 95 % block interval | full grid |
|---|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.4494 | 0.4440 | -0.0054 | [-0.0229, +0.0137] | -0.0049 [-0.0222, +0.0141] |
| AP | 0.6159 | 0.6056 | -0.0103 | [-0.0239, +0.0045] | -0.0101 [-0.0238, +0.0048] |
| AP, vegetated pixels | 0.5250 | 0.5261 | +0.0011 | [-0.0200, +0.0230] | +0.0009 [-0.0202, +0.0228] |
| AUROC, CIRCUNVALACION excluded | 0.8882 | 0.8834 | -0.0049 | [-0.0200, +0.0115] | -0.0045 [-0.0199, +0.0118] |
| AUROC | 0.9116 | 0.9077 | -0.0039 | [-0.0159, +0.0085] | -0.0036 [-0.0158, +0.0086] |
| IoU, CIRCUNVALACION excluded | 0.3188 | 0.3048 | -0.0140 | [-0.0264, -0.0016] | -0.0136 [-0.0260, -0.0013] |
| IoU | 0.4271 | 0.4137 | -0.0134 | [-0.0248, -0.0025] | -0.0130 [-0.0243, -0.0021] |
| F1, CIRCUNVALACION excluded | 0.4835 | 0.4672 | -0.0163 | [-0.0310, -0.0018] | -0.0159 [-0.0304, -0.0015] |
| F1 | 0.5985 | 0.5852 | -0.0133 | [-0.0248, -0.0025] | -0.0129 [-0.0243, -0.0020] |
| Object recall, tolerant | 0.4932 | 0.4869 | -0.0063 | [-0.0243, +0.0108] | -0.0063 [-0.0243, +0.0108] |
| Object recall, strict | 0.4607 | 0.4452 | -0.0155 | [-0.0308, +0.0006] | -0.0155 [-0.0308, +0.0006] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4872 | 0.4808 | -0.0064 | [-0.0245, +0.0110] | -0.0064 [-0.0245, +0.0110] |
| Object recall, tolerant, under 6 m wide | 0.0994 | 0.1278 | +0.0284 | [-0.0076, +0.0646] | +0.0284 [-0.0076, +0.0646] |
| Object recall, strict, under 6 m wide | 0.0767 | 0.0909 | +0.0142 | [-0.0266, +0.0548] | +0.0142 [-0.0266, +0.0548] |

Precision by area, inside the ejido: strict 0.7907 against 0.7777, tolerant 0.8034 against 0.7916 (baseline against arm; 5645 and 6518 components; `components_rule7.csv` of each fold). Full grid: strict 0.7844 against 0.7728, tolerant 0.7976 against 0.7856 (5750 and 6598 components).

The class table is inside the ejido. On the full grid no recall or coverage in it differs by more than 0.0000, because the inventory lies inside the ejido.

| class | polygons | tolerant recall, baseline | arm | area-weighted coverage, baseline | arm |
|---|---|---|---|---|---|
| all | 3167 | 0.4932 | 0.4869 | 0.5409 | 0.5284 |
| group: Road greens | 1166 | 0.3696 | 0.3774 | 0.8082 | 0.7992 |
| group: Squares | 1174 | 0.6346 | 0.6278 | 0.6394 | 0.6272 |
| group: Parks and green areas | 479 | 0.4885 | 0.4802 | 0.2182 | 0.2026 |
| group: Water corridors | 93 | 0.4516 | 0.3871 | 0.4296 | 0.3886 |
| group: Rail verges | 107 | 0.6075 | 0.5888 | 0.4873 | 0.4909 |
| group: NO DEFINIDO | 148 | 0.3041 | 0.2432 | 0.6486 | 0.7002 |
| size: <500 m2 | 811 | 0.2072 | 0.2047 | 0.1987 | 0.1968 |
| size: 500-2000 m2 | 1092 | 0.4881 | 0.5000 | 0.4603 | 0.4655 |
| size: 2000-10000 m2 | 953 | 0.7261 | 0.6988 | 0.6605 | 0.6331 |
| size: >10000 m2 | 311 | 0.5434 | 0.5273 | 0.5275 | 0.5166 |
| width: <3 m | 103 | 0.0388 | 0.1068 | 0.0474 | 0.1373 |
| width: 3-6 m | 249 | 0.1245 | 0.1365 | 0.1210 | 0.1668 |
| width: 6-12 m | 558 | 0.2240 | 0.2258 | 0.2593 | 0.2681 |
| width: 12-25 m | 958 | 0.5240 | 0.4885 | 0.4212 | 0.3868 |
| width: >25 m | 1299 | 0.6928 | 0.6952 | 0.5647 | 0.5549 |

Files: `paired_table.csv`, `per_fold.csv`, `objects_by_class.csv` in this folder. The per-fold values come from the fine histograms of the runs' `ejido/boot_rule7.npz`; `metrics_rule7.json` holds them at 5 decimals, and `per_fold.csv` holds differences of those. Seed 0. One seed: a helping input is confirmed with seeds 1-2 before it enters the fusion design. Coverage is the pixel-weighted mean of per-polygon values stored at 4 decimals (`objects_rule7.csv`), so its last digit can differ by one from the exact ratio: here Rail verges, arm 0.4910; width <3 m, arm 0.1374; width 12-25 m, arm 0.3869 (`review_logs/v9_cov.out`).

## Diagnostics of CLAUDE.md 13.8

The reading stands: no gain. D1 and D2 do not change it and do not explain it; D3 was not run on texture. What the review established after D2 ran: `docs/evidence_phase2/ugi_seg/diagnostics/d2_blank/after_d2.md`.

- **D1.** E2 has no gate. At the kept checkpoint, per channel the eight new weights have norm 0.148, 0.166, 0.142, 0.147 of the six Neo channels, folds 1-4 (`docs/evidence_phase2/ugi_seg/diagnostics/d1_gate/d1_conv1.csv`). On 64 tiles per fold, replacing conv1's output on them by its tile mean changes P by 0.027, 0.038, 0.033, 0.028 and the decision on 2.4, 5.1, 3.8, 2.8 % of the pixels (`docs/evidence_phase2/ugi_seg/diagnostics/review_logs/d1/d1_output_effect.csv`). These are sizes, not scores.
- **D2, skill.** Same weights and thresholds, the six values at their training mean, both masks at 0. Seed SDs: 81-based (13.8), not the 0.0030 above. Centre (sq8km): AP 0.5792 to 0.5639, -0.0153 [-0.1141, +0.0593], seed SD 0.0180; AUROC 0.8959 to 0.9188, +0.0229, seed SD 0.0116. Ejido, CIRCUNVALACION excluded: AP 0.4440 to 0.2917, -0.1523 [-0.1845, -0.1234], seed SD 0.0124; AUROC 0.8834 to 0.8367, -0.0467, seed SD 0.0069 (`docs/evidence_phase2/ugi_seg/diagnostics/d2_blank/d2_table.csv`; bare names: same folder).
- **The centre's pooled numbers hide the folds.** AP by fold +0.0379, -0.1063, -0.0788, -0.1141, AUROC +0.0190, -0.0211, -0.0046, -0.0086 (`d2_folds.csv`); mean of folds AP -0.0653, AP at equal FPR -0.0405. Each fold's median logit rises, +1.936, +0.809, +0.623, +0.680, and these offsets alone move the pooled AP by +0.0358 with no pixel re-ranked inside a fold (`d2_table.csv`). So the small pooled AP change does not mean no use, and the rise in pooled AUROC does not mean a better model.
- **At the run's thresholds.** P rises, so these are no reading of skill: pooled IoU 0.3369 to 0.4060 in the centre, 0.3048 to 0.1629 in the ejido (`d2_table.csv`); tolerant object recall +0.18 in both (`work_ms/runs_itareo/ugi/summary_r/summary_d2_e2_*.json`).
- **Outputs.** Mean |dP| per fold 0.0305-0.0919 in the centre and 0.0457-0.1327 in the ejido, Spearman 0.5669-0.7932 and 0.4666-0.6275, 6.25 % and 11.56 % of pixels change side; arithmetic alone gives at most 0.0005 and at least 0.9994 (`README.md`).
- **What the blank is.** A whole window with both masks at 0 lies outside training: both masks are 0 on 0.0017 to 0.0045 % of training pixels, and the NDVI mask, a constant 1 in training, works as an offset in conv1 (`README.md`).
- **The values, on windows.** Windows holding 92 % of sq8km's pixels, CPU, not the D2 number. NDVI mask alone at 0, all texture kept: AP +0.0090, -0.0036, +0.0186, +0.0004. Values at the training mean, masks real: -0.0140, -0.0260, -0.0574, -0.0729. True tensor from 1,024 m away: -0.0835, -0.1504, -0.1535, -0.1282 (`docs/evidence_phase2/ugi_seg/diagnostics/review_logs/d2_blank/d2_window_folds.csv`). So the output depends on the values, and D2's change, small or large, cannot be read as what the texture contributes.
- **Not the contribution.** The blanked model is no stand-in for one without texture: AP 0.5639 against baseline seeds 0.6530, 0.6760, 0.6237 and a lowest of 81 realisations of 0.6232 in the centre, 0.2917 against 0.4494, 0.4544, 0.4488 and 0.4285 in the ejido; AUROC 0.9188 inside the realisations' 0.8932-0.9324 in the centre, 0.8367 below their 0.8741 in the ejido (`d2_table.csv`; `docs/evidence_phase2/ugi_seg/e1b/registration_check/{scores_by_run,combos}.csv`).
- **D3.** 13.8 registers D3 for PlanetScope and Sentinel-2 only. The texture screen is not D3 (all samples, no region, an earlier mosaic): `ndvi_glcmcorr_41m` auc_mean 0.6799, auc_min 0.6573; `ndvi_orientcons_41m` 0.5876, 0.5704; the PAN layers were not screened (`docs/evidence_phase2/texture_screen/{auroc.csv,README.md,run_log.json}`).
- **E2's layers by D3's statistic.** An addition of the review, not registered and not D3, so no branch reads it. E2's six layers (mosaic v3) on D3's rows, oriented by the sign of the mean of the four sets, read in the ejido var3 0.5003, var9 0.5315, hom3 0.5026, hom9 0.5042, orientcons 0.5860, glcmcorr 0.6812, and inside sq8km, on 32 tiles, 0.7067, 0.7246, 0.7035, 0.7596, 0.7139, 0.7125, there with var3 and var9 lower in the inventory and the others higher. The gain over the model's score, a late add-on fit on three folds and read in the fourth, mean of folds, is -0.0056 to +0.0031 in the ejido (single folds -0.0119 to +0.0211) and -0.0014 to +0.0036 inside sq8km (single folds -0.0058 to +0.0164) (`docs/evidence_phase2/ugi_seg/diagnostics/review_logs/d2_round/reading/s3_e2_layers.log`); random patterns under that read reach +0.0038, +0.0029 in the ejido and +0.0072, +0.0109 inside sq8km at their 97.5 % points (`docs/evidence_phase2/ugi_seg/diagnostics/review_logs/d2_round/refute_10/s2_d3rows_tex.log`). With every tile, weighted as D3, the statistic reads 0.5102, 0.5078, 0.5128, 0.5186, 0.6117, 0.6692 in the ejido (var3 lower) and 0.6699, 0.7011, 0.6628, 0.7117, 0.6939, 0.6862 inside sq8km (`docs/evidence_phase2/ugi_seg/diagnostics/review_logs/d2_round/rule/u1c_every.log`).
- **What 13.8's rule gives.** Ejido: no change is below the seed SD, so 'did not use the input' does not apply. Centre: the AP change is below the seed SD and the AUROC change is not. Read on AP alone the words would apply, and the outputs and folds above show they would be false. With no D3 for E2, neither D3 branch can be read. No branch of the rule applies to E2.
- **What it does not give.** Whether the texture adds anything the model would not otherwise have, or repeats what it gets from Neo. D3b, an addition to 13.8's list, was not run on texture. Registration is not in question: the NDVI layers come from Neo v3 itself, and PAN against Neo v3 reads 0.00, -0.05 m on strip 1_2 and +0.05, 0.00 m on 1_5, the centre's two (`review_logs/cd_panreg.log`), +0.05, -0.025 on 1_3 (`work_ms/derived/features/README.md`); 1_6 [to be verified].
