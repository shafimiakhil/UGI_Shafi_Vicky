# ugi_e1b_ps_plane against ugi_r1_none, seed 0: pre-registered reading (CLAUDE.md 13.7)

**Reading (centre sq8km, primary, 13.7): no gain.**

Gain in AP inside the centre: -0.0084 [-0.0403, +0.0256] (`runs_itareo\ugi\summary_r/summary_e1b_centre_sq8km.json`), against the threshold 0.0787, three times the seed SD 0.0262 inside it (`runs_itareo\ugi\summary_r/summary_centre_sq8km.json`). With 50,000 resamples the interval is [-0.0394, +0.0258]. Positive in 3 of 4 folds: -0.0644, +0.0354, +0.0523, +0.0200.

Tests: gain at least 0.0787: no; interval above zero: no; positive in at least 3 of 4 folds: yes.

The block interval holds no training-seed variance, so the threshold is the test that covers it. Between baseline seeds inside this centre (b0, b1, b2: seeds 0, 1, 2), the same interval gives b1 - b0 +0.0230 [-0.0149, +0.0506]; b2 - b0 -0.0293 [-0.0515, -0.0068]; b2 - b1 -0.0523 [-0.0840, -0.0096] (`docs/evidence_phase2/ugi_centre/seed_pairs.csv`).

| score, inside the centre | baseline | arm | difference | 95 % block interval |
|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.6530 | 0.6446 | -0.0084 | [-0.0403, +0.0256] |
| AP | 0.6530 | 0.6446 | -0.0084 | [-0.0403, +0.0256] |
| AP, vegetated pixels | 0.7068 | 0.6950 | -0.0118 | [-0.0427, +0.0219] |
| AUROC, CIRCUNVALACION excluded | 0.9255 | 0.9033 | -0.0221 | [-0.0393, -0.0037] |
| AUROC | 0.9255 | 0.9033 | -0.0221 | [-0.0393, -0.0037] |
| IoU, CIRCUNVALACION excluded | 0.3817 | 0.4162 | +0.0345 | [-0.0126, +0.0771] |
| IoU | 0.3817 | 0.4162 | +0.0345 | [-0.0126, +0.0771] |
| F1, CIRCUNVALACION excluded | 0.5525 | 0.5878 | +0.0353 | [-0.0123, +0.0791] |
| F1 | 0.5525 | 0.5878 | +0.0353 | [-0.0123, +0.0791] |
| Object recall, tolerant | 0.4624 | 0.4492 | -0.0133 | [-0.0383, +0.0139] |
| Object recall, strict | 0.4256 | 0.4094 | -0.0162 | [-0.0410, +0.0085] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4624 | 0.4492 | -0.0133 | [-0.0383, +0.0139] |
| Object recall, tolerant, under 6 m wide | 0.1017 | 0.0847 | -0.0169 | [-0.1176, +0.0909] |
| Object recall, strict, under 6 m wide | 0.0847 | 0.0508 | -0.0339 | [-0.1071, +0.0270] |

Precision by area, inside the centre: strict 0.8470 against 0.8486, tolerant 0.8669 against 0.8657 (965 and 918 components). Classes: `objects_by_class_centre.csv`.

## The ejido, secondary

**Secondary reading (rule of 13.7): no gain.** All tiles: does not pass the 13.6 rule. B4 tile set: does not pass it.

Gain in AP with CIRCUNVALACION excluded, inside the ejido: -0.0120 [-0.0307, +0.0070] (`runs_itareo\ugi\summary_r/summary_e1b_ejido.json`); the baseline's SD over seeds is 0.0030. With 50,000 resamples the interval is [-0.0311, +0.0066]. Positive in 2 of 4 folds: +0.0219, -0.0368, -0.0310, +0.0412.

Tests: gain at least 0.010: no; interval above zero: no; positive in at least 3 of 4 folds: no.

**B4 tile set** (1,240 tiles with all three dates, `runs_itareo\ugi\summary_r/summary_e1b_ejido_b4.json`): gain -0.0109 [-0.0317, +0.0090]; with 50,000 resamples [-0.0307, +0.0087]; positive in 1 of 4 folds: -0.0025, -0.0194, -0.0039, +0.0333. By the same tests: no gain.

**Registration (13.7).** E1b removes a plane per date. Residual per 1,024 m block, low-pass, median and 90th percentile, with the plane (with the constant, on the same blocks): 20220424 0.85 and 2.72 m (2.01 and 3.6 m), 20220910 1.0 and 2.28 m (1.56 and 3.04 m), 20221028 0.89 and 1.89 m (1.2 and 2.62 m) (`docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json`).

| score | baseline | arm | difference | 95 % block interval | full grid | B4 tiles |
|---|---|---|---|---|---|---|
| AP, CIRCUNVALACION excluded | 0.4494 | 0.4374 | -0.0120 | [-0.0307, +0.0070] | -0.0086 [-0.0271, +0.0104] | -0.0109 [-0.0317, +0.0090] |
| AP | 0.6159 | 0.6007 | -0.0153 | [-0.0293, -0.0009] | -0.0126 [-0.0266, +0.0017] | -0.0173 [-0.0316, -0.0035] |
| AP, vegetated pixels | 0.5250 | 0.5124 | -0.0126 | [-0.0290, +0.0033] | -0.0091 [-0.0256, +0.0071] | -0.0132 [-0.0304, +0.0027] |
| AUROC, CIRCUNVALACION excluded | 0.8882 | 0.8725 | -0.0157 | [-0.0247, -0.0065] | -0.0145 [-0.0235, -0.0053] | -0.0152 [-0.0258, -0.0055] |
| AUROC | 0.9116 | 0.8986 | -0.0130 | [-0.0203, -0.0060] | -0.0120 [-0.0193, -0.0050] | -0.0124 [-0.0204, -0.0053] |
| IoU, CIRCUNVALACION excluded | 0.3188 | 0.3157 | -0.0031 | [-0.0179, +0.0114] | -0.0017 [-0.0165, +0.0127] | +0.0135 [-0.0025, +0.0290] |
| IoU | 0.4271 | 0.4190 | -0.0080 | [-0.0210, +0.0045] | -0.0065 [-0.0196, +0.0058] | +0.0034 [-0.0104, +0.0172] |
| F1, CIRCUNVALACION excluded | 0.4835 | 0.4799 | -0.0036 | [-0.0208, +0.0133] | -0.0020 [-0.0191, +0.0146] | +0.0149 [-0.0028, +0.0320] |
| F1 | 0.5985 | 0.5906 | -0.0079 | [-0.0206, +0.0045] | -0.0065 [-0.0192, +0.0058] | +0.0031 [-0.0095, +0.0158] |
| Object recall, tolerant | 0.4932 | 0.4799 | -0.0133 | [-0.0310, +0.0033] | -0.0133 [-0.0310, +0.0033] | +0.0070 [-0.0089, +0.0235] |
| Object recall, strict | 0.4607 | 0.4481 | -0.0126 | [-0.0302, +0.0036] | -0.0126 [-0.0302, +0.0036] | +0.0048 [-0.0100, +0.0200] |
| Object recall, tolerant, CIRCUNVALACION excluded | 0.4872 | 0.4738 | -0.0134 | [-0.0314, +0.0034] | -0.0134 [-0.0314, +0.0034] | +0.0071 [-0.0090, +0.0239] |
| Object recall, tolerant, under 6 m wide | 0.0994 | 0.0795 | -0.0199 | [-0.0542, +0.0092] | -0.0199 [-0.0542, +0.0092] | -0.0033 [-0.0323, +0.0254] |
| Object recall, strict, under 6 m wide | 0.0767 | 0.0653 | -0.0114 | [-0.0431, +0.0167] | -0.0114 [-0.0431, +0.0167] | +0.0065 [-0.0170, +0.0321] |

Precision by area, inside the ejido: strict 0.7907 against 0.7662, tolerant 0.8034 against 0.7778 (baseline against arm; 5645 and 5793 components; `components_rule7.csv` of each fold). Full grid: strict 0.7844 against 0.7639, tolerant 0.7976 against 0.7758 (5750 and 5838 components).

The class table is inside the ejido. On the full grid no recall or coverage in it differs by more than 0.0000, because the inventory lies inside the ejido.

| class | polygons | tolerant recall, baseline | arm | area-weighted coverage, baseline | arm |
|---|---|---|---|---|---|
| all | 3167 | 0.4932 | 0.4799 | 0.5409 | 0.5404 |
| group: Road greens | 1166 | 0.3696 | 0.3422 | 0.8082 | 0.7960 |
| group: Squares | 1174 | 0.6346 | 0.6295 | 0.6394 | 0.6503 |
| group: Parks and green areas | 479 | 0.4885 | 0.5010 | 0.2182 | 0.2595 |
| group: Water corridors | 93 | 0.4516 | 0.3978 | 0.4296 | 0.3936 |
| group: Rail verges | 107 | 0.6075 | 0.6449 | 0.4873 | 0.4963 |
| group: NO DEFINIDO | 148 | 0.3041 | 0.2432 | 0.6486 | 0.6564 |
| size: <500 m2 | 811 | 0.2072 | 0.1985 | 0.1987 | 0.1926 |
| size: 500-2000 m2 | 1092 | 0.4881 | 0.4615 | 0.4603 | 0.4476 |
| size: 2000-10000 m2 | 953 | 0.7261 | 0.7177 | 0.6605 | 0.6588 |
| size: >10000 m2 | 311 | 0.5434 | 0.5498 | 0.5275 | 0.5278 |
| width: <3 m | 103 | 0.0388 | 0.0388 | 0.0474 | 0.0865 |
| width: 3-6 m | 249 | 0.1245 | 0.0964 | 0.1210 | 0.1559 |
| width: 6-12 m | 558 | 0.2240 | 0.2133 | 0.2593 | 0.2497 |
| width: 12-25 m | 958 | 0.5240 | 0.4927 | 0.4212 | 0.3858 |
| width: >25 m | 1299 | 0.6928 | 0.6936 | 0.5647 | 0.5694 |

Files: `paired_table.csv`, `per_fold.csv`, `objects_by_class.csv` in this folder. The per-fold values come from the fine histograms of the runs' `ejido/boot_rule7.npz` (B4 rows: `ejido_b4/boot_rule7.npz`); `metrics_rule7.json` holds them at 5 decimals. Seed 0. One seed: a helping input is confirmed with seeds 1-2 before it enters the fusion design. Coverage is the pixel-weighted mean of per-polygon values stored at 4 decimals (`objects_rule7.csv`), so its last digit can differ by one from the exact ratio.

## Independent review (workflow `wf_f48ff81d-00c`; logs and results in `review_logs/`, `workflow_result.json`; corrected after the review of 13.8, workflow `wf_fa0360bb-211`, whose scripts and logs are the `f*` files there)

The reading stands: no number, test or headline above changes.

- **Numbers.** Reproduced from the stored rasters by own code that imports nothing from this project: all point values, per-fold values, class rows, precision by area, and every interval by the documented estimator within 8e-8. Under exact ranks instead of the 1,000 coarse bins, 6 of 56 interval cells move in the 4th decimal (largest 0.00015, `review_logs/ci_cells.log`); no interval changes whether it excludes zero. Monte Carlo spread of the limits over resampling seeds is larger (0.0014).
- **Wording, corrected here.** The registration sentence says "with the constant, on the same blocks": on 24 Apr the constant has 257 blocks and the plane 256; on the shared 256 the constant's 90th percentile is 3.59 m, not 3.6. "Because the inventory lies inside the ejido": 263 inventory pixels lie outside it. The centre table and per-fold values come from each run's `centre_sq8km/boot_rule7.npz`.
- **The plane inside the centre.** The cut of the residual is scene-wide. Inside sq8km, median constant to plane: 24 Apr 1.53 to 0.59 m, 28 Oct 1.30 to 1.04 m, 10 Sep 0.89 to 0.80 m, not distinguishable from zero (`review_logs/v3d_regions.out`, `review_logs/refute3/`). One reviewer read this as limiting the test of registration; the check that tried to refute it did: models trained mostly outside the centre, with inputs at fixed weights registered tile by tile to about 0.1 m, change AP by at most 0.0013 as a mean of the four folds and by at most 0.0027 in one fold (`review_logs/refute3/p4_fixed.real.out`). The 53 tiles with all three dates within 1 m (13 % of the inventory pixels of the centre's 196 tiles; per fold 1, 4, 20, 28) show no gain over the baseline either: E1b - b0 -0.0165 [-0.0516, +0.0076], against -0.0247 [-0.0511, +0.0049] on the other 143 tiles (`review_logs/refute3/p6c_shares.out`, `review_logs/f2_p6c_rest.log`). Pooled they lose a little less than the rest, and the interval of that difference spans zero (+0.0082 [-0.0355, +0.0431]); in folds 3 and 4, which hold 48 of the 53, they gain less (-0.0341 against +0.0694, -0.0017 against +0.0514; `review_logs/f2_strata.log`). A clause, not a limit on the reading.
- **Registration as the cause, tests added by the review** (one seed each). At fixed weights, swapping plane and constant moves AP, as a mean of the four folds, by at most 0.0003 inside the centre and 0.0016 outside it, and in one fold by at most 0.0013 inside the centre and 0.0055 outside it; displacing E1's input by 2, 4, 8 and 16 m moves the mean of the folds in the centre by -0.0003, -0.0011, -0.0038, -0.0116 (`review_logs/a8b_analysis.log`). On a larger draw of 100 tiles per fold outside the centre, the swap moves the mean of the folds by -0.0020 with E1's weights and +0.0004 with E1b's (`review_logs/f2_fixed.log`). E1b - E1 is as large on pixels more than 520 m from any PlanetScope pixel, where both runs get the same input (AP -0.0314, against -0.0290 on three-date pixels of the same folds; `review_logs/a7d_placebo_zone.log`). Limit: no run trains on inputs registered tile by tile, so a benefit from training at under 0.2 m is not excluded.
- **Where dates are missing (not registration, and it explains part of the ejido).** Pixels with fewer than three dates hold 18.4 % of the inventory without CIRCUNVALACION (14.2 % with it) and none of sq8km. There, against the baseline mean: E1 AP -0.0434 [-0.0929, -0.0013], F1 -0.0598 [-0.0991, -0.0233]; E1b AP -0.0763 [-0.1240, -0.0350], F1 -0.0908 [-0.1287, -0.0559] (the reason of `refute:2` in `review_logs/workflow_result.json`). Not everywhere: fold 1 shows no loss, the 0- and 2-date pixels exist only in folds 3-4, and fold 3's part is a logit shift (recall 0.21 against 0.60-0.65). A candidate cause: the availability channels can act as a prior on prevalence, since the UGI share of training tiles is 0.0032-0.0267 with no date and 0.0695-0.0844 with all three; setting availability to 1 on those tiles raises F1 from 0.3935 to 0.4160 (E1: 0.4007 to 0.4489; baseline mean 0.4169). On three-date pixels E1b's F1 gain (+0.0168) is mostly threshold placement (+0.0035 at each fold's best threshold) and its AP is -0.0096 [-0.0270, +0.0069]; E1's AP is +0.0167 [+0.0036, +0.0303]. Replacing the arm by the baseline where dates are missing moves ejido AP by little (E1 0.4658 to 0.4664, E1b 0.4374 to 0.4463). A design warning for any fusion, not a change of this reading.
- **Why 3 of 4 folds rise while the pooled AP falls.** The weights turn the sign and the score scales add to the fall; this corrects what `registration_check/README.md` first said. Pooled AP is additive over positive pixels and fold 1 holds 43.2 % of them inside sq8km: pooled E1b - b0 -0.0084 splits by fold as -0.0182, +0.0046, +0.0025, +0.0027; plain mean of folds +0.0108; positive-weighted mean -0.0072; at equal FPR in every fold, which keeps the weights and removes the scales, -0.0025 (`review_logs/f1_scale.log`). The scale step, -0.0059, is no larger than between baseline seeds (+0.0010, +0.0065, +0.0055), so this describes one seed and names no cause. Two polygons hold 62.1 % of fold 1's positives (3049, 37.2 %; 3048, 24.9 %); without them the seed SD inside sq8km falls from 0.0262 to 0.0111 and the 81-based one from 0.0148 (uncorrected) to 0.0074 (`review_logs/a4_fold1.log`, `a3b_contrib` numbers in `workflow_result.json`).
- **Why IoU and F1 rise inside the centre.** Against b0, mostly one polygon; threshold placement in part (this line first said only that). Polygon 3048 (CANTERO, 96.8 ha, folds 1 and 2, 18.2 % of the centre's positives) holds +178,113 of E1b's net +225,085 true positives over b0, coverage 0.439 against 0.255 (b1 0.453, b2 0.347). With its pixels left out, a diagnostic and not a score, E1b - b0 falls from +0.0353 to +0.0070 in F1 (IoU +0.0345 to +0.0070): b0 is the lowest seed with it (F1: b1 0.5726, b2 0.5633) and the highest without it (E1b 0.5836 against seeds 0.5765, 0.5625, 0.5719, above 97.5 % of the 81 baseline realisations, 0.5538-0.5849). Of the 703 polygons with pixels there the next largest drop leaves +0.0269 (3049); no single removal turns E1b - b0 negative, 3048 and 3049 together do (-0.0059; folds 3 and 4 hold neither and rise +0.0193, +0.0150). The thresholds as numbers do not make the rise: E1b at b0's has F1 0.5879, its own 0.5878 (b0 at E1b's 0.5587). E1b flags more in every fold, in folds 1 and 2 only on 3048. b0 at E1b's flagged area per fold reaches 0.5680 from 0.5525 (b1 0.5881), 63 % of the added true positives off 3048: at equal flagged area E1b - b0 is +0.0198 to +0.0237; with 3048 left out -0.0033 to -0.0007 at those thresholds, +0.0072 to +0.0074 with the area matched on the rest. At each fold's best threshold (fold 1: 0.003-0.005, internal 0.315-0.51) it is -0.0356 in F1, another part of the curve, no test of this rise (`docs/evidence_phase2/ugi_seg/e3/review_logs/`: `d2_round_cause/s6_e1b.log`, `d2_round_draft_7/v3_flagged_and_pairs.log`; `review_logs/a9_misc.log`). One seed, and the rule reads AP.

## Diagnostics of CLAUDE.md 13.8

The reading stands: no gain. D1, D2 and D3 neither change nor explain it. Paths start at `docs/evidence_phase2/ugi_seg/`. What the review established after D2 ran: `diagnostics/d2_blank/after_d2.md`.

- **D1.** The gate is open at each run's kept checkpoint: weight norm 0.92, 0.94, 0.93, 0.92, bias norm 0.27, 0.27, 0.19, 0.22, added term 0.17, 0.19, 0.16, 0.16 of the stage features' RMS, folds 1-4 (`diagnostics/d1_gate/d1_gate.csv`). On 64 tiles per fold with all three dates, replacing the added term by its tile mean changes P by 0.026, 0.015, 0.019, 0.013 and the decision on 2.5, 1.9, 2.6, 1.2 % of pixels (`diagnostics/review_logs/d1/d1_output_effect.csv`). These are sizes, not scores.
- **D2, skill.** Same weights; 24 PlanetScope values at their training mean, availability 0. SD: 81-based, not the three-run 0.0262, 0.0030 above. Centre: AP 0.6446 to 0.5266, -0.1180 [-0.1828, -0.0564], SD 0.0180; AUROC 0.9033 to 0.8736, -0.0297, SD 0.0116. Ejido, CIRCUNVALACION excluded: AP 0.4374 to 0.3144, -0.1230 [-0.1461, -0.1006], SD 0.0124; AUROC 0.8725 to 0.8297, -0.0428, SD 0.0069 (`diagnostics/d2_blank/d2_table.csv`).
- **D2, folds and outputs.** AP by fold: centre -0.0182 (inside its seed SD, 0.0554), -0.1809, -0.1914, -0.0536, mean -0.1110; ejido -0.0516, -0.0739, -0.0601, -0.0684, mean -0.0635, at equal FPR -0.0607, so about half the pooled fall is between-fold score scale (`diagnostics/d2_blank/d2_folds.csv`, `d2_table.csv`). Mean |dP| per fold 0.0181-0.0507 and 0.0126-0.0343, Spearman 0.5866-0.8647 and 0.5597-0.7493; two baseline seeds 0.011-0.032 and 0.43-0.67, arithmetic alone at most 0.0005 and at least 0.9994 (`diagnostics/d2_blank/README.md`).
- **What the blank is.** The no-date state occurs on 4.83 to 11.93 % of the training cells, only where PlanetScope does not reach, on land with fewer positives, and on no training tile with at least half its area in sq8km (same README). So the fall includes the model's response to a state it learned on peripheral land.
- **Not PlanetScope's contribution.** The blanked model is no stand-in for one that never had the input: its AP lies below every baseline score, centre 0.5266 against seeds 0.6530, 0.6760, 0.6237 and lowest realisation 0.6232, ejido 0.3144 against 0.4494, 0.4544, 0.4488 and 0.4285 (`e1b/registration_check/scores_by_run.csv`, `combos.csv`). What PlanetScope adds over the baseline is what the reading above measures.
- **The values, on windows.** Windows holding 92 % of sq8km, CPU, not the D2 number: values at the training mean with availability real change AP by -0.0311, -0.1324, -0.0612, -0.0645, and the true tensor from 1,024 m away by -0.0635, -0.1699, -0.0821, -0.0742; on 17 % of the ejido the first rises in fold 1, +0.0221 (`diagnostics/review_logs/d2_blank/d2_window_folds.csv`). So the output depends on the values too: the model leans on them, which does not say they help.
- **D3.** AUROC on a class-balanced sample of vegetated pixels. Ejido: registered statistic (highest of five features) 0.6091, 24 Apr, auc_min 0.5677, inside folds 0.5604, 0.7067, 0.7246, 0.3976; the others 0.5346-0.5618; floor NDVI 0.5092, CHM 0.5633 (`diagnostics/d3_screen/d3.json`, `d3_auroc.csv`). At or above 0.60 in 66.9 % of repeated draws, 24 Apr in 56.1 % (`diagnostics/README.md`). Sq8km 0.7006-0.8337 on 32 tiles (`d3_auroc.csv`), 0.7239-0.8000 with every tile (same README).
- **What 13.8's rule gives.** No pooled AP or AUROC change is below the SD in either region, so 'did not use the input' does not apply. D3 is not near 0.5, so the rule does not stop PlanetScope experiments. D3 is at or above 0.60 but D2 changed, so the second branch does not apply. 13.8 has no branch for this; 'the fusion design is the question' does not follow.
- **What it does not give.** Whether PlanetScope adds anything the model would not otherwise have, or repeats what the network gets from Neo: a fall under the blank fits both. D2 has no bearing on registration; see the review above.
- **D3b, added to 13.8's list.** Gain over Neo pixel NDVI + CHM + Neo NDVI in 33 px, window chosen after the results, fitted inside tiles: ejido +0.0122 [+0.0058, +0.0189] for 24 Apr, -0.0066 to -0.0017 for the other four; sq8km -0.0067 to +0.0163. All five over 17 px: -0.0147 to +0.0172, -0.0029 to +0.0393; over 65 px: +0.0088 to +0.0300, +0.0060 to +0.0319. Over the baseline's score, same fit: ejido -0.0043 to -0.0015, sq8km -0.0109 to +0.0038 (`diagnostics/README.md`, `diagnostics/review_logs/d3/TABLE_d3b_gains.csv`). A late add-on read in AUROC: not AP, and not what a network trained with the input does.

