# Diagnostics of CLAUDE.md 13.8: D1 gate norms, D2 input blanked, D3 information screen, seed 0

13.8 asks these diagnostics to say which of three explanations of E1, E1b and E2 holds; it does not list the three, and no branch of its reading fits what they show (see D2 and `d2_blank/after_d2.md`). The readings of E1, E1b and E2 stand as ruled: "no gain".

Inside the ejido, AP with CIRCUNVALACION excluded, against baseline seed 0: E1 +0.0164 [-0.0015, +0.0346], E1b -0.0120 [-0.0307, +0.0070], E2 -0.0054 [-0.0229, +0.0137] (`runs_itareo/ugi/summary_r/summary_e1_ejido.json`, `summary_e1b_ejido.json`, `summary_e2_ejido.json`). The 81-based SD is 0.0124 (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`), so E1 lies 1.32 SD above the baseline, above 92.6 % of the 81 baseline scores, and E1b and E2 lie 0.97 and 0.44 SD below it. AP with CIRCUNVALACION: +0.0064, -0.0153, -0.0103. All three read no gain, under the rule of 13.6 and 13.7 and under that of 13.8 (threshold 0.0372). CLAUDE.md 13.8 quotes -0.012, -0.012, -0.010, all slightly negative: no report, table or summary of the project holds that triple and no single metric gives it; in the rule's metric E1 is positive and more than one SD from the baseline.

Everything below was checked by own code that imports nothing from the project (workflows `wf_4d1a36b3-79f`, `wf_fa0360bb-211`; the D2 section and the E3 subsection, `wf_67b286a8-6be`); scripts, logs and tables are in `review_logs/` (`d1/`, `d2_blank/`, `d2_round/`, `d3/`, and the overall results `workflow_result_*.json`). A sentence naming a table without a path means the one in that subfolder: `d3/TABLE_d3_statistic.csv`, `d3/TABLE_d3b_gains.csv`, `d1/d1_review_table.csv` (columns: `d1/d1_review_columns.csv`). Original outputs: `d1_gate/`, `d1_gate_e3/` (E3, an addition to 13.8's list, checked in `../e3/review_logs/`), `d1_gate_e1c/` (E1c, written after this note, not checked here), `d3_screen/`. Paths under `runs_itareo/`, `scripts/` and `itareo/` are relative to `work_ms/`, paths under `docs/` and `work_ms/` to the project root.

## Summary, as of this note

- **D1.** The gate is open in all eight runs and the branch passes on a signal that follows the PlanetScope values; the same holds for E2's new channels. That says nothing about gain.
- **D3.** The rule written before the run reads 0.6091, so the conditional run is E1c, which Shafi confirmed knowing the four sets and the lowest of them. Neither half of 13.8's reading follows from D3 and D3b alone; see below.
- **D2.** Blanking the added input moves the outputs of E1b and E2 far beyond arithmetic noise. The change exceeds the seed SD in AP and AUROC in every row but one: E2 in the centre, AP -0.0153 against 0.0180, AUROC +0.0229 against 0.0116, where folds 2-4 fall beyond their SDs (`d2_blank/d2_table.csv`). So 'did not use the input' fits no row; with D3, no branch of 13.8's reading fits PlanetScope. The fall is not what the input adds: blanked AP lies below all 81 baseline scores, E2 in the centre already as run (`../e1b/registration_check/combos.csv`). See below.

## D1

Is the gate open, does the branch pass anything on? Yes to both, and D1 goes no further. In E1 and E1b the 1 x 1 gate starts at zero. At the kept checkpoint it has weight norm 0.86, 0.91, 0.85, 0.94 (E1) and 0.92, 0.94, 0.93, 0.92 (E1b), and bias norm 0.25, 0.26, 0.21, 0.24 and 0.27, 0.27, 0.19, 0.22. On tiles with all three dates the branch passes on a signal that follows the PlanetScope values, inside a tile 0.10-0.13 and between tiles 0.05-0.11 of the RMS of the stage features, beside a constant that says nothing about the tile. It reaches the output. On 64 tiles per run, replacing the added term by its tile mean per channel changes P by 0.028, 0.015, 0.027, 0.013 on average in E1 and 0.026, 0.015, 0.019, 0.013 in E1b. It changes the decision at the run's threshold on 2.9, 1.9, 3.1, 1.3 % and 2.5, 1.9, 2.6, 1.2 % of the pixels; the models call 5.4-8.2 % of these pixels UGI. For scale, seed 0 and seed 1 of the baseline differ on the same tiles by 0.022, 0.014, 0.031, 0.016 and on 2.1, 1.7, 4.2, 1.5 % of the pixels. E2 has no gate, and its new channels behave the same way. On 64 tiles per fold, replacing conv1's output on them by its tile mean changes P by 0.027, 0.038, 0.033, 0.028 and the decision on 2.4, 5.1, 3.8, 2.8 % of the pixels, where two baseline seeds differ by 0.021, 0.010, 0.020, 0.015 and on 2.0, 1.0, 2.7, 1.4 %. These are sizes, not scores. A closed gate does not explain the 'no gain' of E1, E1b and E2, whose readings stand. D1 does not say whether what is passed on helps. D2 asks whether the score depends on it, and its answer mixes the values with the no-data state (see D2 below). None of these sizes is read as the importance of a sensor (13.4, rule 8).

All of D1 reads the checkpoint each run kept (best.pt), seed 0: epochs 54, 55, 56, 48 in E1, 47, 60, 54, 48 in E1b and 54, 52, 58, 50 in E2, of 60. 13.8 words D1 as the end of training; the runs store no later checkpoint, so the state at epoch 60 is known only for E1b fold 2.

### The columns of `d1_gate/d1_gate.csv`

E1, E1b (seed 0, the checkpoint each run kept). The column ratio_input_dependent of d1_gate.csv is the root mean square (RMS) of the added term minus the added term for the no-data input (all 27 channels zero), over the RMS of the stage features (layer1 output, 4 m). It is the distance from the no-data output, not how much the content of PlanetScope moves the features. It exceeds ratio_added in 7 of 8 rows because the no-data output is about as large as the output on data and only partly aligned with it (cosine 0.30-0.56): RMS 0.138, 0.152, 0.170, 0.135 in E1 and 0.150, 0.128, 0.217, 0.131 in E1b, against 0.134, 0.165, 0.136, 0.127 and 0.139, 0.153, 0.139, 0.128 on data. On the tiles with all three dates at least 90 % valid (394, 362, 286, 198 tiles in folds 1-4, the set D1 draws from), the split of its mean square is as follows. A shift common to all tiles is 50, 28, 59, 41 % in E1 and 51, 21, 74, 45 % in E1b. Variation between tiles is 16, 23, 8, 15 % and 15, 24, 5, 14 %. Variation inside a tile is 34, 49, 33, 44 % and 34, 55, 21, 41 %.

What the branch adds, on the same tiles, is 0.180, 0.203, 0.158, 0.153 (E1) and 0.176, 0.189, 0.157, 0.155 (E1b) of the RMS of the stage features, as the median over tiles; the 32-tile rows of d1_gate.csv lie within 0.01 of these. Of its mean square, 32, 30, 44, 44 % and 33, 30, 37, 41 % is a constant per channel, the same on every tile, which says nothing about the tile. Another 22, 22, 12, 15 % and 21, 21, 13, 15 % varies between tiles, and 46, 48, 45, 42 % and 46, 48, 51, 44 % varies inside a tile.

The part of the branch's output that varies inside a tile, over the RMS of the stage features, median over tiles: on all held-out tiles (625, 525, 525, 441) E1 0.111, 0.139, 0.097, 0.100 and E1b 0.111, 0.133, 0.109, 0.106; on the tiles with all three dates E1 0.118, 0.134, 0.102, 0.100 and E1b 0.118, 0.130, 0.110, 0.106. The stage features are non-negative and 53-60 % of their mean square is a constant per channel. Over the part of them that varies inside a tile, the same quantity on the tiles with all three dates is E1 0.196, 0.216, 0.149, 0.167 and E1b 0.193, 0.214, 0.162, 0.177. It comes from the PlanetScope values: with the values at the training mean and the availability as it is, it falls to 0.005-0.009. The part that varies between tiles also follows the values (availability gives at most 0.002 on these tiles). Pooled over the tiles it is 0.088, 0.111, 0.053, 0.061 and 0.084, 0.102, 0.056, 0.062 of the RMS of the stage features, which is 0.71, 0.75, 0.62, 0.50 and 0.66, 0.58, 0.60, 0.46 of their own between-tile part. Both parts together, pooled, are 0.156, 0.198, 0.118, 0.120 and 0.151, 0.184, 0.126, 0.122. On the 57 and 75 tiles of folds 3 and 4 with no date at all the branch adds a constant only; the inside-tile part is 0.010, 0.008 in E1 and 0.014, 0.009 in E1b.

### E2's conv1

E2, the new conv1 channels (seed 0, the checkpoint each run kept). The weights of the eight new channels start at zero. Per channel their norm is now 0.148, 0.166, 0.142, 0.147 of that of the six Neo channels in folds 1-4. Before the batch norm, conv1's output on the new channels against its output on the Neo channels, as the median of the tile ratios on all held-out tiles (625, 525, 525, 441), is 0.542, 0.657, 0.327, 0.621; tile quartiles 0.28-1.14, 0.26-1.17, 0.26-0.44, 0.35-0.89. The values 1.020, 0.535, 0.330, 0.495 in d1_conv1.csv are correct for the 32 tiles drawn there but are not the folds' values. The median of 32 tiles moves too much: 95 % of 20,000 draws lie in 0.33-0.89, 0.33-1.04, 0.28-0.39, 0.47-0.79, and fold 1's 1.020 is reached by 0.3 % of them (56 % of its 32 tiles have a ratio above 1, against 30 % of the fold). So in fold 1 the texture does not equal Neo at conv1. The tile ratio follows its denominator, conv1's output on Neo (Spearman -0.99, -0.91, -0.94, -0.97). That output has median 2.512, 2.889, 4.154, 2.443 (1.525 on fold 1's 32 tiles), while the output on the new channels has median 1.332, 1.760, 1.265, 1.497. Half of the spread between folds in the file remains on all tiles (SD over folds 0.297 there, 0.148 here): fold 3 stays low because Neo's output is larger there, not because of the draw. Pooled over the tiles of a fold, the ratio of the root mean squares is 0.483, 0.475, 0.310, 0.412. On tiles at least half inside the ejido (615, 525, 506, 441) the median ratio is 0.534, 0.657, 0.324, 0.621.

Part of the E2 ratio is a constant. The NDVI-texture mask is 1 on at least 99.7 % of every tile, so conv1's output on it is a constant per channel: RMS 1.045, 0.891, 0.635, 0.829 in folds 1-4, of which 0.058, 0.051, 0.038, 0.046 varies inside a tile. The constant part of the new channels' output is 54, 36, 38, 39 % of its mean square, and this mask makes 78, 50, 64, 73 % of that constant. The batch norm that follows takes it out again: its running mean has RMS 1.124, 0.947, 0.717, 0.917 over the 64 channels, correlates at 0.98, 0.98, 0.97, 0.97 with the sum of that filter's weights, and differs from it by 0.246, 0.204, 0.185, 0.244. Without this channel the ratio on all held-out tiles is 0.338, 0.508, 0.298, 0.472. What the texture passes on is better measured by the part that varies inside a tile. That part is 0.285, 0.441, 0.233, 0.418 of the RMS of conv1 on Neo and 0.323, 0.524, 0.237, 0.490 of the part of conv1 on Neo that varies inside a tile. With each channel scaled as the batch norm scales it, which is what layer1 receives, it is 0.504, 0.833, 0.386, 0.725. These numbers and those of E1 and E1b are at different stages and not on one scale. At the 4 m stage of E1 and E1b, replacing conv1's output on the new channels by its tile mean moves the layer1 output by 0.263, 0.357, 0.278, 0.365 of its RMS.

### E3: `docs/evidence_phase2/ugi_seg/diagnostics/d1_gate_e3/`

An addition to 13.8's list, whose D1 names the PlanetScope branch and E2's conv1, not E3. `d1_gate_e3/d1_gate.csv`, written by `work_ms/scripts/diag_d1.py`, holds one row per fold, seed 0, at the checkpoint each run kept (best.pt; epochs 46, 55, 44, 50 of 60): the weight and bias norms of the 1 x 1 gate (0.99, 1.02, 0.94, 1.13 and 0.29, 0.34, 0.35, 0.32) and, on 32 held-out tiles, the median RMS of the stage features it adds to (layer2 output, 8 m), of the added term and of that term minus its value on the all-zero input (`input_dependent_rms_median`), with the ratios `ratio_added` and `ratio_input_dependent`. No test of E3's reading uses it; `docs/evidence_phase2/ugi_seg/e3/reading.md` quotes the weight norms only.

Both input-dependent columns need a caveat. In E3 the availability channel is 1 on every cell of all 2,116 cached tiles (`work_ms/runs_itareo/ugi/cache/s2_2022_stage8m/s2.npy` holds no NaN), so the all-zero input, z = 0 with availability 0 (`diag_d1.py` line 68), is a state training never held. Against z = 0 with availability 1 (every band uniform at its training mean, which no training tile is, availability as in training), the median per-tile rms of the difference on the same 32 tiles is 0.117, 0.145, 0.110, 0.129 in folds 1-4, against 0.168, 0.136, 0.093, 0.135 in the CSV (`docs/evidence_phase2/ugi_seg/e3/review_logs/refute_4/a_cache_gate.log`). Neither is only what depends on a tile's Sentinel-2 content: against z = 0 with availability 1, of the mean square on those tiles 58, 56, 39, 35 % (49, 53, 35, 39 % on all 625, 525, 525, 441 held-out tiles) is a constant per channel, the same on every tile, and the added term minus its own constant per channel has median rms 0.079, 0.100, 0.080, 0.099 on all held-out tiles (`docs/evidence_phase2/ugi_seg/e3/review_logs/refute_4/e_split32.log`). Both logs come from E3's review, not the workflows named above.

## D2

Same weights and thresholds, added input at its no-data value in every window, seed 0: `scripts/diag_d2.py`, `itareo/ugi_seg/blank_input.py`; runs `runs_itareo/ugi/diag_d2_e1b_blank/`, `diag_d2_e2_blank/`; report `d2_blank/README.md`, with `d2_notes.md`, written before it ran, appended, and `d2_blank/after_d2.md`, what the review established afterwards; self-tests `d2_selftest/` (failed on its own limit), `d2_selftest_v2/` (passed: same maxima, wider limit, `scripts/diag_d2.py`). AP and AUROC without CIRCUNVALACION (`d2_blank/d2_table.csv`, `d2_blank/d2_folds.csv`):

| arm | pixels | AP as run | blanked | change | seed SD | AUROC change | seed SD | AP change by fold |
|---|---|---|---|---|---|---|---|---|
| E1b | sq8km | 0.6446 | 0.5266 | -0.1180 | 0.0180 | -0.0297 | 0.0116 | -0.0182, -0.1809, -0.1914, -0.0536 |
| E1b | ejido | 0.4374 | 0.3144 | -0.1230 | 0.0124 | -0.0428 | 0.0069 | -0.0516, -0.0739, -0.0601, -0.0684 |
| E2 | sq8km | 0.5792 | 0.5639 | -0.0153 | 0.0180 | +0.0229 | 0.0116 | +0.0379, -0.1063, -0.0788, -0.1141 |
| E2 | ejido | 0.4440 | 0.2917 | -0.1523 | 0.0124 | -0.0467 | 0.0069 | -0.0584, -0.2942, -0.1515, -0.0836 |

- **13.8's test.** Only E2's AP in the centre is below the seed SD. A cancellation: per-fold score scales alone move pooled AP by +0.0358 (`d2_blank/d2_table.csv`) and pooled AUROC by +0.0362 (`review_logs/d2_round/refute_5/s1_e2_centre_exact.log`), while folds 2-4 fall in both beyond their SDs. Pooling also inflates E1b's ejido fall: mean of folds -0.0635. Outputs moved in every row and fold: mean |dP| at least 0.0126, Spearman at most 0.8647; arithmetic alone reaches 0.0005, 0.9994 (`d2_blank/README.md`). 'Did not use the input' fits no row.
- **What the blank is.** E2: both masks at 0, 0.0017 to 0.0045 % of training pixels, never a window. E1b: no date, 4.83 to 11.93 % of training cells, 45 to 120 near-whole tiles per fold, none in tiles at least half inside sq8km (`d2_blank/README.md`). So the change mixes the values with the model's response to that state.
- **The values.** On windows with 92 % of sq8km's pixels, CPU, not D2's number, values at the training mean with masks or availability real change AP by -0.0140, -0.0260, -0.0574, -0.0729 (E2) and -0.0311, -0.1324, -0.0612, -0.0645 (E1b); fold 1 stays within its SD and rises on 17 % of the ejido (`review_logs/d2_blank/d2_window_folds.csv`). The output depends on the values: reliance, not benefit.
- **Not the contribution.** Blanked AP lies below the lowest of the 81 baseline scores: sq8km 0.5266 and 0.5639 against 0.6232 (E2 as run too: 0.5792), ejido 0.3144 and 0.2917 against 0.4285 (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`). A blanked model is not a model without the input; what an input adds is the experiment's reading.
- **13.8's branches.** D3's PlanetScope rule reads 0.6091, its other four features 0.5346 to 0.5618 (`d3_screen/d3_auroc.csv`); 13.8 sets no limit for 'near 0.5'. With D2 changed the second branch does not apply to E1b, and 13.8's reading has no such case; E1c follows from D3 alone. E1 and E3 have no D2, E2 no D3. D2 does not say whether an input adds to what the model has or repeats what it gets from Neo, and has no bearing on registration. The readings stand: no gain.

## D3 and D3b

**D3 and D3b: what they say.** All numbers are AUROC on a class-balanced sample of pixels with Neo NDVI >= 0.3 and the edge band out, which is 37.9 % of the valid inventory pixels in the ejido and 35.6 % in sq8km. They are not AP at the true prevalence. D3 reads the 4 m and 8 m stage values, not the 1 m input of E1c. This part was written before D2 ran and does not use its results, except one sentence marked as added after it. The readings of E1, E1b and E2 ('no gain') stand as ruled.

The rule written before the run reads the highest auc_mean among the five PlanetScope features in the ejido against 0.60: 24 Apr reads 0.6091 (auc_min 0.5677), so the screen reached 0.60. 13.8 goes on to read D3 near 0.5 as no information, and D3 at or above 0.60 with D2 unchanged as 'the model did not use the input, and the fusion design is the question'. Neither follows from D3 and D3b alone.

**Inside the ejido a reader may conclude:**
- 24 Apr's value is not a fluke of the draw (0.6089 with every tile, auc_min 0.5760), but it does not clear 0.60 by a margin: a new 30-tile draw gives [0.5384, 0.6741] and reaches 0.60 in 56.1 % of draws; the highest of the five does so in 66.9 %.
- Across the region a green inventory pixel is only weakly greener than a green pixel elsewhere, and in fold 4 it is less green: 24 Apr reads 0.3976 there (0.3900 with every tile, below 0.5 in 91.9 % of draws).
- Inside one 512 m tile the inventory pixel is the greener one: 0.5989-0.6690 for the five PlanetScope features, 0.6693 for Neo NDVI in 33 px, 0.8907 for P(UGI).
- Over the model's score (base: logit P refitted, read in fold) no input adds AUROC: -0.0245 to -0.0008 on D3's rows, -0.0101 to -0.0007 with every tile, -0.0044 to +0.0012 with a same-place fit.

**Inside the ejido a reader may not conclude:**
- That 10 Sep and 28 Oct hold no information because they read 0.5346 and 0.5407: inside tiles they read 0.6321 and 0.6639.
- That reaching 0.60 is information Neo or the model lacks. On rows holding all three dates, 24 Apr reads 0.5794 against Neo NDVI in 33 px 0.5869 on D3's draw, and 0.6289 against 0.5946 with every tile. At the same place 24 Apr adds +0.0122 [+0.0058, +0.0189] over NDVI + CHM + Neo 33 px, and -0.0015 [-0.0052, +0.0020] over the model.
- That D3 at or above 0.60 with D2 unchanged shows the model did not use the input and that the fusion design is the question. An unchanged D2 is also what an input would give that repeats what the model already has from Neo; D3 and D3b do not tell these apart. Added after D2 ran: D2 did not come out unchanged for E1b or E2, since in every row of `docs/evidence_phase2/ugi_seg/diagnostics/d2_blank/d2_table.csv` AP or AUROC changes by more than its seed SD, and a fall under the blank does not exclude such an input either: a model trained with it can come to lean on it (`docs/evidence_phase2/ugi_seg/diagnostics/d2_blank/after_d2.md`, section (f)).
- That a region-wide gain over NDVI, CHM and Neo measures anything over Neo: that base reads at chance (0.4516-0.5557).

**Inside sq8km a reader may conclude:**
- PlanetScope NDVI at the 4 m stage separates inventory green from other green through the neighbourhood: 0.7006-0.8337 on D3's 32 tiles, 0.7239-0.8000 with every tile, 0.7260-0.7888 inside tiles; Neo NDVI in 33 px reads 0.8761, 0.8048, 0.8042; P(UGI) 0.9657, 0.9309, 0.9325. It is no defect of the input (correlation with Neo NDVI blurred at sigma 8 m 0.952 on 28 Oct).
- Over NDVI + CHM + Neo NDVI in 33 x 33 px, the five PlanetScope features add little: -0.0081 to +0.0043 on D3's rows, -0.0017 to +0.0088 with every tile, -0.0067 to +0.0163 with a same-place fit.
- Over that same base, Sentinel-2's summer quarter and amplitude add more: +0.0152 to +0.0247 with a same-place fit.
- Over the model's score nothing adds AUROC beyond chance-sized values: -0.0094 to +0.0046 on D3's rows, -0.0038 to +0.0025 with every tile, -0.0109 to +0.0038 with a same-place fit.

**Inside sq8km a reader may not conclude:**
- Anything from the 0.60 of 13.8. Neo's own NDVI in a 9 px window reads 0.6399 (0.6176 with every tile), so any coarse NDVI clears it, and an unchanged D2 there would not mean what 13.8 says.
- A statement about one fold from D3's rows: positive against negative tiles are 3/1, 3/2, 6/3, 10/4, no tile holds both classes, and fold 1's largest gain over the model, +0.0352, is +0.0084 with every tile.
- The 32-tile values as the centre's value: they lie 0.03 to 0.07 above every tile for all features but the amplitude.
- That Sentinel-2 adds 0.02-0.03 over the model, as D3b's pooled column suggests (+0.0075 to +0.0267): that column compares three separately fitted models, its base reads 0.9286 against 0.9657 for P itself, and with the feature it stays at most 0.9554.

**How D3's samples are drawn.** D3 reuses the 160,000 rows of the texture screen, which drew 30 tiles per fold for positives and, separately, 30 for negatives. So the two classes almost never share a tile: of 232 tiles, 115 hold only positive rows, 114 only negative rows and 3 both (8192/4096, 15872/13312, 22528/13824; 1,584 positive and 2,707 negative rows). 0.027 % of the positive-negative pairs lie in one tile. Inside sq8km there are 32 tiles, 22 positive only, 10 negative only, none with both; positive against negative tiles by fold: 3/1, 3/2, 6/3, 10/4, so fold 1 has one negative tile. The samples concentrate: the top 5 tiles hold 49.3, 55.5, 67.5 and 35.5 % of the positives of folds 1-4, and the top 3 and top 5 tiles hold 42.8 % and 58.3 % of sq8km's positives. Between tile clusters (tile and class) lie 44-59 % of the variance of the five PlanetScope features in the ejido and 21-42 % inside sq8km.

**What the statistic is.** D3's AUROC compares a green inventory pixel with a green pixel anywhere else in the region. That is a property of any pooled AUROC, not of the disjoint draw: with every one of the 2,116 tiles only 0.042 % of the pairs in the ejido and 0.347 % in sq8km lie in one tile. The draw does not bias the ejido value (24 Apr: 0.6091 on the draw, 0.6089 with every tile); it makes it uncertain.

**The same-place contrast**, which D3's rows cannot give, was computed from the rasters with the same eligibility: AUROC inside each 512 m tile with at least 50 eligible pixels of each class (1,178 tiles in the ejido, 210 in sq8km), weighted inside a fold by pairs of pixels, then the mean of the four folds. Interval: tiles resampled inside folds. 'Repeating the draw' is the 2.5-97.5 % range of D3's auc_mean when the 30-tile draw is repeated 2,000 times from all tiles. All 19 features: TABLE_d3_statistic.csv.

| region | feature | D3 auc_mean | repeating the draw | every tile | same tile [tile interval] |
|---|---|---|---|---|---|
| ejido | ps_ndvi_20220424 | 0.6091 | [0.5384, 0.6741] | 0.6089 | 0.6690 [0.6496, 0.6895] |
| ejido | ps_ndvi_20220910 | 0.5346 | [0.4893, 0.6213] | 0.5592 | 0.6321 [0.6112, 0.6513] |
| ejido | ps_ndvi_20221028 | 0.5407 | [0.4823, 0.6338] | 0.5602 | 0.6639 [0.6438, 0.6833] |
| ejido | ps_ndvi_min | 0.5618 | [0.5328, 0.6682] | 0.6037 | 0.6519 [0.6328, 0.6719] |
| ejido | ps_ndvi_amp | 0.5441 | [0.4908, 0.6303] | 0.5646 | 0.5989 [0.5793, 0.6197] |
| ejido | s2_ndmi_q2 | 0.6395 | [0.5613, 0.6811] | 0.6225 | 0.6482 [0.6296, 0.6685] |
| ejido | s2_ndmi_q1 | 0.6149 | [0.5512, 0.6797] | 0.6197 | 0.7106 [0.6913, 0.7301] |
| ejido | Neo NDVI, pixel | 0.5092 | | 0.5078 | 0.5516 [0.5371, 0.5657] |
| ejido | Neo NDVI, 33 px | 0.5310 | | 0.5423 | 0.6693 [0.6511, 0.6871] |
| ejido | P(UGI), ugi_r1_none seed 0 | 0.8922 | | 0.8911 | 0.8907 [0.8803, 0.9014] |
| sq8km | ps_ndvi_20220424 | 0.8337 | [0.6671, 0.8937] | 0.8000 | 0.7888 [0.7567, 0.8237] |
| sq8km | ps_ndvi_20220910 | 0.7844 | [0.6186, 0.8308] | 0.7398 | 0.7260 [0.6965, 0.7566] |
| sq8km | ps_ndvi_20221028 | 0.8306 | [0.6475, 0.8857] | 0.7847 | 0.7871 [0.7566, 0.8178] |
| sq8km | ps_ndvi_min | 0.7940 | [0.6296, 0.8479] | 0.7538 | 0.7404 [0.7084, 0.7729] |
| sq8km | ps_ndvi_amp | 0.7006 | [0.6232, 0.8142] | 0.7239 | 0.7276 [0.7034, 0.7529] |
| sq8km | s2_ndre_q1 | 0.9082 | [0.6975, 0.9400] | 0.8452 | 0.8421 [0.8112, 0.8767] |
| sq8km | Neo NDVI, 33 px | 0.8761 | | 0.8048 | 0.8042 [0.7730, 0.8367] |
| sq8km | P(UGI), ugi_r1_none seed 0 | 0.9657 | | 0.9309 | 0.9325 [0.9143, 0.9518] |

The between-place contrast does not inflate D3. In the ejido the same-tile values are higher than D3 (0.5989-0.6690 against 0.5346-0.6091), because green pixels outside the inventory are greener in some places than in others. Inside sq8km they are close to the every-tile values (0.7260-0.7888 against 0.7239-0.8000).

**The 0.60 of the rule.** 24 Apr's 0.6091 has a tile bootstrap interval of [0.5252, 0.6813]. Repeating the draw gives [0.5384, 0.6741], at or above 0.60 in 56.1 % of the draws; the highest of the five features reaches 0.60 in 66.9 % of them, its auc_min for 24 Apr in 16.2 %.

**The fold 4 reversal.** 24 Apr reads 0.3976 inside fold 4 (mean NDVI 0.417 for positives, 0.457 for negatives; 25 positive and 15 negative tiles). It is not a small-sample accident: 0.3900 with every tile, below 0.5 in 91.9 % of the repeated draws. It is a between-place effect: inside the tiles of fold 4 the inventory pixel is still the greener one (0.5334).

**Sq8km is one draw.** Inside sq8km D3 holds 11,412 rows from the 32 of its 232 tiles that fall in the square; 240 tiles touch it. With every tile, fold and class weighted as in D3, the five PlanetScope features read 0.8000, 0.7398, 0.7847, 0.7538, 0.7239 against D3's 0.8337, 0.7844, 0.8306, 0.7940, 0.7006, so D3 is +0.0337, +0.0446, +0.0459, +0.0402 and -0.0233 away. The Sentinel-2 features read +0.0273 to +0.0705 above every tile, Neo NDVI in 33 px 0.8761 against 0.8048, and P(UGI) of ugi_r1_none, seed 0, 0.9657 against 0.9309. All of these lie inside the range of a repeated draw: for 24 Apr mean 0.7999, SD 0.0613, 2.5-97.5 % range [0.6671, 0.8937]. So the sq8km values on D3's rows are quoted with the every-tile value beside them. A bootstrap of the 32 tiles alone ([0.7777, 0.8819] for 24 Apr) is too narrow, because it cannot vary which tiles fall in the square.

**A D3 value near 0.5.** 13.8 reads a D3 value near 0.5 as no information. On these samples a value near 0.5 does not show that. D3's auc_mean averages over folds and compares pixels of different places, so it can be near 0.5 when the feature separates the classes in opposite directions by fold, or when it separates them inside places. PlanetScope 28 Oct reads 0.5407, and 0.5755, 0.5738, 0.6617, 0.3611 inside the folds, and 0.6639 [0.6438, 0.6833] inside tiles. PlanetScope 10 Sep reads 0.5346, and 0.5485, 0.5383, 0.6450, 0.3788 inside the folds, and 0.6321 [0.6112, 0.6513] inside tiles. s2_ndvi_amp reads 0.5126, and 0.6368, 0.3373, 0.6825, 0.3971 inside the folds (every tile 0.6297, 0.3966, 0.7020, 0.4244), and 0.6060 [0.5860, 0.6263] inside tiles. Such a value says that one relation is not shared across the region. It does not say the input is empty.

**PlanetScope against Neo inside sq8km.** PlanetScope NDVI reads 0.8306 (28 Oct) and 0.8337 (24 Apr) where Neo NDVI on the same 4 m cell reads 0.5336. This is no defect. Band 6 is red and band 8 NIR (file descriptions); the raw mosaics give 0.826 (28 Oct) and 0.793-0.795 (10 Sep), with correlation 0.992 and 0.995-0.996 to the normalised NDVI; the correlation with Neo peaks at lag (0, 0) within 16 m on all three dates.

PlanetScope at the 4 m stage measures the neighbourhood. On all 524,288 cells of the 32 tiles its 28 Oct NDVI correlates with Neo NDVI at 0.74 without blur and at 0.946, 0.952 and 0.919 after a Gaussian of sigma 6, 8 and 12 m; 24 Apr reaches 0.887 at 6-8 m, 10 Sep 0.887 at 6 m. That scale includes registration residual and view geometry, not only the sensor.

Neo NDVI by window (NDVI of the window's mean red and NIR, mosaic v3), D3's statistic: 4 m cell 0.5336, 9 px 0.6399, 17 px 0.7783, 33 px 0.8761, 65 px 0.8796, 129 px 0.8288 on D3's rows; 0.5294, 0.6176, 0.7281, 0.8048, 0.8067, 0.7752 with every tile, where PlanetScope reads 0.8000 (24 Apr) and 0.7847 (28 Oct). Each tile as one unit, 22 against 10 tiles: PlanetScope 28 Oct 0.905 [0.745, 1.0], Neo 33 px 0.905 [0.741, 1.0], Neo 4 m 0.632 [0.423, 0.832].

The fine Neo scales fail because the samples are chosen at Neo pixel NDVI >= 0.3, which truncates Neo at 1-4 m. On all valid clear pixels inside sq8km, without that cut: pixel NDVI 0.7098, Neo 4 m 0.7156, Neo 33 px 0.8235, PlanetScope 28 Oct 0.8342. With the cut: 0.5475 in the opposite direction, 0.5296, 0.8036, 0.7836. Inventory green sits in green neighbourhoods, other green in built-up blocks: green fraction in 33 px 0.506 against 0.301 on D3's rows, 0.507 against 0.330 with every tile.

**Gain over a Neo neighbourhood.** The base is Neo pixel NDVI + Meta CHM + Neo NDVI in a 33 x 33 px window (33 m) centred on the pixel, fitted through the screen's add-on model. The gain is the AUROC of that model with the PlanetScope feature minus the AUROC without it, on the same rows. The window was chosen after the results (correlation with PlanetScope 28 Oct: 0.893 at 33 px, 0.891 at 17 px, 0.765 at 65 px); it is not registered, so 17 and 65 px stand beside it. Fold columns: D3's rows and D3b's fits, each fold read by the three fits that leave it out, mean of the three. 'Every tile': fit on three folds, read in the held-out fold, on 50,000 rows per fold and class from all tiles (ejido) or 20,000 from inside sq8km. 'Same-place fit', added in this review: conditional logistic regression on pairs of rows of one tile, fit on three folds, read inside the tiles of the fourth, weighted by pairs. Intervals: tiles resampled, fits fixed.

Ejido:

| feature | fold 1 | fold 2 | fold 3 | fold 4 | mean | base in fold | every tile [interval] | base | same-place fit [interval] | base |
|---|---|---|---|---|---|---|---|---|---|---|
| ps_ndvi_20220424 | +0.0059 | -0.0599 | +0.0650 | +0.0209 | +0.0080 | 0.4950 | +0.0243 [+0.0159, +0.0328] | 0.4907 | +0.0122 [+0.0058, +0.0189] | 0.6432 |
| ps_ndvi_20220910 | -0.0046 | -0.0028 | -0.0104 | +0.0299 | +0.0031 | 0.5117 | +0.0229 [+0.0143, +0.0305] | 0.5103 | -0.0017 [-0.0045, +0.0013] | 0.6704 |
| ps_ndvi_20221028 | +0.0130 | +0.0119 | -0.0075 | +0.0679 | +0.0213 | 0.5116 | +0.0195 [+0.0086, +0.0284] | 0.5105 | -0.0037 [-0.0057, -0.0017] | 0.6705 |
| ps_ndvi_min | -0.0137 | -0.0729 | -0.0058 | +0.0217 | -0.0177 | 0.5356 | +0.0011 [-0.0050, +0.0072] | 0.5295 | -0.0066 [-0.0136, +0.0001] | 0.6618 |
| ps_ndvi_amp | +0.0024 | +0.0370 | -0.0008 | -0.0246 | +0.0035 | 0.5356 | +0.0124 [+0.0051, +0.0206] | 0.5295 | -0.0025 [-0.0132, +0.0068] | 0.6618 |

Sq8km:

| feature | fold 1 | fold 2 | fold 3 | fold 4 | mean | base in fold | every tile [interval] | base | same-place fit [interval] | base |
|---|---|---|---|---|---|---|---|---|---|---|
| ps_ndvi_20220424 | +0.0078 | -0.0016 | +0.0180 | -0.0069 | +0.0043 | 0.8928 | +0.0084 [+0.0031, +0.0149] | 0.8349 | +0.0083 [+0.0024, +0.0146] | 0.8083 |
| ps_ndvi_20220910 | +0.0029 | -0.0015 | +0.0042 | -0.0272 | -0.0054 | 0.8928 | -0.0010 [-0.0036, +0.0016] | 0.8349 | -0.0062 [-0.0139, -0.0005] | 0.8083 |
| ps_ndvi_20221028 | +0.0019 | -0.0018 | +0.0056 | -0.0034 | +0.0006 | 0.8928 | +0.0026 [-0.0003, +0.0053] | 0.8349 | +0.0056 [+0.0002, +0.0116] | 0.8083 |
| ps_ndvi_min | -0.0025 | -0.0072 | -0.0008 | -0.0220 | -0.0081 | 0.8928 | -0.0017 [-0.0038, +0.0003] | 0.8349 | -0.0067 [-0.0111, -0.0027] | 0.8083 |
| ps_ndvi_amp | -0.0028 | -0.0133 | +0.0086 | +0.0075 | -0.0000 | 0.8928 | +0.0088 [+0.0050, +0.0129] | 0.8349 | +0.0163 [+0.0028, +0.0305] | 0.8083 |

Other windows, five features in the same order. Over 17 px, D3's rows in fold: ejido +0.0293, +0.0041, +0.0080, -0.0221, +0.0069; sq8km +0.0375, +0.0144, +0.0566, +0.0190, +0.0028. Same-place fit: ejido +0.0172, -0.0018, -0.0147, +0.0035, +0.0015; sq8km +0.0331, -0.0026, +0.0393, -0.0029, +0.0299. Over 65 px, D3's rows in fold: ejido +0.0296, -0.0150, +0.0028, -0.0191, +0.0002; sq8km +0.0129, +0.0031, +0.0090, +0.0020, +0.0027. Same-place fit: ejido +0.0300, +0.0088, +0.0263, +0.0195, +0.0123; sq8km +0.0316, +0.0060, +0.0319, +0.0079, +0.0307. Every base and reading: TABLE_d3b_gains.csv.

Inside sq8km the five features add little over the 33 px window: -0.0081 to +0.0043 on D3's rows, -0.0017 to +0.0088 with every tile, -0.0067 to +0.0163 with the same-place fit. It is small, not zero: the intervals lie above zero for 24 Apr and the amplitude in both every-tile columns. The size depends on the window, up to +0.0566 over 17 px. Pooled over the three fits the same gains read -0.0026, -0.0127, -0.0078, -0.0130, -0.0022 (base 0.8689); these are not the numbers to quote. D3b's dauc_over_neo_stage in sq8km, +0.2542, +0.1907, +0.2658, +0.2066, +0.0799 (read in fold +0.2066, +0.1722, +0.2336, +0.1858, +0.0617; base 0.5570 pooled, 0.6587 in fold), is a gain over Neo at the 4 m cell. The model sees a 512 m window of Neo, so this column is not what PlanetScope adds to what the model has.

After a straight line on Neo NDVI blurred at sigma 8 m, what is left of 28 Oct reads 0.4815 pooled on D3's rows, but 0.765, 0.564, 0.755, 0.467 inside the folds, and 0.523 inside tiles with every tile. What is left of 24 Apr reads 0.6269 pooled and 0.615 inside tiles. Another date holds a little that Neo's October image does not.

D3 numbers must be compared on the same rows. On 28 Oct's rows in the ejido PlanetScope reads 0.5407, Neo 17 px 0.5676 and 33 px 0.5661. On rows holding all three dates: D3's 105,110 rows, 24 Apr 0.5794, 10 Sep 0.5429, 28 Oct 0.5549, minimum 0.5618, amplitude 0.5441, Neo 17, 33, 65 px 0.5829, 0.5869, 0.5485; every tile, 278,468 rows, 0.6289, 0.5754, 0.5785, 0.6037, 0.5646, and 0.5837, 0.5946, 0.5633.

Sentinel-2 over the same 33 px base, same-place fit: sq8km Q1 NDVI, NDRE, NDMI +0.0238, +0.0226, +0.0152, Q4 NDVI and NDRE +0.0121, +0.0132, amplitude +0.0247, the others -0.0043 to +0.0073 (base 0.8083); ejido Q1 +0.0325, +0.0325, +0.0378, Q4 NDVI and NDRE +0.0278, +0.0099, amplitude +0.0241, the others -0.0087 to +0.0043 (base 0.6610). So the summer quarter and the amplitude carry something Neo's October neighbourhood does not. This is a gain over Neo, not over the model.

**Gains over NDVI, CHM and Neo in the ejido.** Region-wide, these gains are measured over a base at chance. The base of dauc_over_neo_stage (NDVI + CHM + Neo NDVI on the input's stage cell) reads 0.4693-0.5259 pooled and 0.5143-0.5301 in fold. The base of dauc_ndvi_chm (NDVI + CHM) reads 0.4516-0.5489 pooled and 0.5238-0.5557 in fold. The base with Neo NDVI in 33 px reads 0.4950-0.5356 in fold and 0.4907-0.5295 with every tile. The relation flips between folds (Neo 33 px reads 0.5787, 0.4144, 0.7685, 0.3682 inside the folds), so a fit on other folds predicts a fold at chance. A gain such as s2_ndvi_amp +0.0878 lifts a fit from 0.5259 and does not say what the input adds to what Neo knows. For that, read the same-place fit, whose base reads 0.6432-0.6705: PlanetScope 24 Apr +0.0122 [+0.0058, +0.0189], 10 Sep -0.0017, 28 Oct -0.0037, minimum -0.0066, amplitude -0.0025; Sentinel-2 NDVI of Q1 +0.0325 [+0.0253, +0.0399], amplitude +0.0241 [+0.0159, +0.0324].

**What D3b's over-model column measures.** dauc_over_model (scripts/diag_d3b.py) fits a rank-spline logistic regression on two folds, predicts the third, and pools the predictions of the three folds. Three separately fitted models put the same score on different scales, so the base loses skill the score has: a refit of logit P alone reads 0.9286 pooled against 0.9657 for P itself in sq8km, and 0.8676 against 0.8922 in the ejido. Read inside each fold there is no such loss: 0.9680 against 0.9698, and 0.9111 against 0.9110. Aligning each fold's predictions to that fold's negatives before pooling removes it too: 0.9760 against 0.9778, and 0.9111 against 0.9110. Class share is a small part: balanced fits raise the pooled base in sq8km only to 0.9369.

So the positive pooled gains in sq8km, +0.0043 to +0.0267, are not gains over the model's score. With the feature the pooled value reaches at most 0.9554, below P's own 0.9657 (ejido, Sentinel-2 rows: 0.8675 against 0.8922). After alignment every gain is negative: sq8km -0.0109 to -0.0032, ejido -0.0245 to -0.0008. Features with no relation to UGI reach the same size in the pooled column: over 300 draws a random spatial pattern (plane waves, wavelengths 1-6 km) has mean -0.0219 and 97.5 % point +0.0277, a random value per tile mean -0.0271 and 97.5 % point +0.0258; read in fold their 97.5 % points are +0.0041 and +0.0058.

**Gains read inside each fold.** The base is logit P(UGI) of ugi_r1_none, seed 0, from the sample's own fold's prob_oof, refitted alone through the same add-on model on the same rows. The gain is the AUROC of the refit of logit P + feature minus the AUROC of the refit of logit P alone, inside the fold. Fold columns: D3's rows and D3b's own fits; each fold is read by the three fits that leave it out, each fit on two folds, and the three are averaged. 'Every tile': fit on the other three folds, read in the held-out fold, 50,000 rows per fold and class from all tiles (ejido) or 20,000 from inside sq8km. 'Same-place fit', added in this review: conditional logistic regression on pairs of rows of one tile, read inside the tiles of the held-out fold. Intervals: tiles resampled, fits fixed. On D3's rows of sq8km a stratum with one tile cannot be resampled, so no interval is shown; the fit-on-three-folds reading with intervals is in TABLE_d3b_gains.csv.

Ejido:

| feature | fold 1 | fold 2 | fold 3 | fold 4 | mean | pooled, as published | every tile [interval] | same-place fit [interval] |
|---|---|---|---|---|---|---|---|---|
| ps_ndvi_20220424 | -0.0253 | -0.0099 | -0.0091 | -0.0291 | -0.0184 | -0.0061 | -0.0049 [-0.0077, -0.0021] | -0.0015 [-0.0052, +0.0020] |
| ps_ndvi_20220910 | -0.0107 | -0.0184 | -0.0110 | -0.0317 | -0.0180 | -0.0120 | -0.0060 [-0.0075, -0.0046] | -0.0025 [-0.0044, -0.0007] |
| ps_ndvi_20221028 | -0.0143 | -0.0150 | -0.0161 | -0.0339 | -0.0198 | -0.0122 | -0.0093 [-0.0111, -0.0076] | -0.0043 [-0.0060, -0.0026] |
| ps_ndvi_min | -0.0139 | -0.0192 | -0.0112 | -0.0309 | -0.0188 | -0.0076 | -0.0038 [-0.0057, -0.0018] | -0.0035 [-0.0068, -0.0009] |
| ps_ndvi_amp | -0.0021 | +0.0017 | -0.0062 | -0.0226 | -0.0073 | -0.0009 | -0.0046 [-0.0062, -0.0030] | -0.0036 [-0.0060, -0.0015] |
| s2_ndvi_q1 | -0.0097 | -0.0072 | -0.0163 | -0.0253 | -0.0146 | -0.0065 | -0.0057 [-0.0076, -0.0037] | -0.0006 [-0.0027, +0.0016] |
| s2_ndre_q1 | -0.0206 | -0.0134 | -0.0157 | -0.0272 | -0.0192 | -0.0094 | -0.0069 [-0.0088, -0.0051] | -0.0019 [-0.0041, +0.0004] |
| s2_ndmi_q1 | -0.0179 | -0.0287 | -0.0138 | -0.0246 | -0.0212 | -0.0126 | -0.0042 [-0.0063, -0.0020] | +0.0012 [-0.0014, +0.0037] |
| s2_ndvi_q2 | -0.0191 | -0.0233 | -0.0146 | -0.0363 | -0.0233 | -0.0145 | -0.0086 [-0.0107, -0.0065] | -0.0011 [-0.0039, +0.0015] |
| s2_ndre_q2 | -0.0211 | -0.0262 | -0.0160 | -0.0347 | -0.0245 | -0.0156 | -0.0101 [-0.0122, -0.0080] | -0.0017 [-0.0044, +0.0007] |
| s2_ndmi_q2 | -0.0135 | -0.0178 | -0.0121 | -0.0273 | -0.0177 | -0.0115 | -0.0076 [-0.0098, -0.0053] | -0.0019 [-0.0044, +0.0003] |
| s2_ndvi_q3 | -0.0081 | -0.0028 | -0.0193 | -0.0259 | -0.0140 | -0.0086 | -0.0035 [-0.0052, -0.0019] | -0.0025 [-0.0040, -0.0011] |
| s2_ndre_q3 | -0.0120 | -0.0063 | -0.0227 | -0.0278 | -0.0172 | -0.0119 | -0.0054 [-0.0072, -0.0039] | -0.0023 [-0.0040, -0.0007] |
| s2_ndmi_q3 | -0.0074 | -0.0062 | -0.0399 | -0.0118 | -0.0163 | -0.0101 | -0.0048 [-0.0074, -0.0024] | -0.0042 [-0.0058, -0.0026] |
| s2_ndvi_q4 | -0.0162 | -0.0012 | -0.0133 | -0.0306 | -0.0153 | -0.0064 | -0.0054 [-0.0068, -0.0040] | -0.0009 [-0.0028, +0.0008] |
| s2_ndre_q4 | -0.0232 | -0.0019 | -0.0107 | -0.0288 | -0.0161 | -0.0067 | -0.0075 [-0.0092, -0.0060] | -0.0016 [-0.0034, +0.0002] |
| s2_ndmi_q4 | -0.0111 | -0.0028 | -0.0180 | -0.0171 | -0.0123 | -0.0037 | -0.0053 [-0.0075, -0.0033] | -0.0012 [-0.0033, +0.0008] |
| s2_ndvi_drymin | -0.0123 | -0.0105 | -0.0151 | -0.0265 | -0.0161 | -0.0087 | -0.0056 [-0.0072, -0.0041] | -0.0024 [-0.0044, -0.0006] |
| s2_ndvi_amp | -0.0074 | +0.0065 | +0.0072 | -0.0095 | -0.0008 | -0.0001 | -0.0007 [-0.0021, +0.0008] | -0.0044 [-0.0071, -0.0022] |

Sq8km:

| feature | fold 1 | fold 2 | fold 3 | fold 4 | mean | pooled, as published | every tile [interval] | same-place fit [interval] |
|---|---|---|---|---|---|---|---|---|
| ps_ndvi_20220424 | +0.0104 | +0.0001 | +0.0038 | -0.0220 | -0.0019 | +0.0097 | -0.0008 [-0.0043, +0.0024] | -0.0043 [-0.0108, +0.0016] |
| ps_ndvi_20220910 | +0.0097 | +0.0001 | +0.0022 | -0.0214 | -0.0024 | +0.0104 | -0.0022 [-0.0051, +0.0003] | -0.0109 [-0.0239, -0.0016] |
| ps_ndvi_20221028 | -0.0018 | -0.0024 | -0.0008 | -0.0199 | -0.0062 | +0.0043 | -0.0009 [-0.0042, +0.0021] | -0.0032 [-0.0071, +0.0003] |
| ps_ndvi_min | +0.0089 | -0.0002 | +0.0035 | -0.0203 | -0.0021 | +0.0090 | -0.0020 [-0.0050, +0.0006] | -0.0109 [-0.0227, -0.0018] |
| ps_ndvi_amp | -0.0308 | -0.0010 | -0.0016 | -0.0040 | -0.0094 | -0.0030 | +0.0006 [-0.0011, +0.0024] | +0.0038 [-0.0013, +0.0107] |
| s2_ndvi_q1 | +0.0203 | -0.0009 | +0.0061 | -0.0162 | +0.0023 | +0.0183 | +0.0012 [-0.0020, +0.0044] | -0.0006 [-0.0049, +0.0033] |
| s2_ndre_q1 | +0.0352 | -0.0008 | +0.0020 | -0.0181 | +0.0046 | +0.0255 | +0.0021 [-0.0016, +0.0057] | +0.0018 [-0.0006, +0.0043] |
| s2_ndmi_q1 | +0.0300 | +0.0030 | +0.0016 | -0.0204 | +0.0036 | +0.0267 | -0.0013 [-0.0062, +0.0035] | -0.0057 [-0.0160, +0.0015] |
| s2_ndvi_q2 | +0.0132 | -0.0018 | +0.0041 | -0.0199 | -0.0011 | +0.0127 | -0.0008 [-0.0035, +0.0018] | -0.0071 [-0.0171, +0.0005] |
| s2_ndre_q2 | +0.0224 | -0.0032 | +0.0033 | -0.0262 | -0.0009 | +0.0171 | -0.0006 [-0.0039, +0.0028] | -0.0039 [-0.0095, +0.0010] |
| s2_ndmi_q2 | +0.0171 | -0.0036 | -0.0009 | -0.0240 | -0.0028 | +0.0187 | -0.0038 [-0.0069, -0.0009] | -0.0067 [-0.0140, -0.0005] |
| s2_ndvi_q3 | +0.0153 | -0.0005 | +0.0027 | -0.0191 | -0.0004 | +0.0168 | -0.0017 [-0.0039, +0.0004] | -0.0074 [-0.0150, -0.0012] |
| s2_ndre_q3 | +0.0221 | +0.0005 | +0.0020 | -0.0225 | +0.0005 | +0.0195 | -0.0018 [-0.0051, +0.0011] | -0.0039 [-0.0081, -0.0004] |
| s2_ndmi_q3 | +0.0130 | -0.0029 | -0.0035 | -0.0159 | -0.0023 | +0.0219 | -0.0022 [-0.0034, -0.0011] | -0.0046 [-0.0089, -0.0012] |
| s2_ndvi_q4 | +0.0087 | -0.0003 | +0.0019 | -0.0162 | -0.0015 | +0.0128 | +0.0002 [-0.0027, +0.0030] | -0.0025 [-0.0069, +0.0011] |
| s2_ndre_q4 | +0.0226 | -0.0008 | -0.0003 | -0.0202 | +0.0003 | +0.0183 | +0.0009 [-0.0027, +0.0043] | -0.0013 [-0.0046, +0.0017] |
| s2_ndmi_q4 | -0.0063 | -0.0003 | -0.0080 | -0.0165 | -0.0078 | +0.0083 | -0.0013 [-0.0050, +0.0021] | -0.0065 [-0.0154, -0.0004] |
| s2_ndvi_drymin | +0.0144 | -0.0008 | +0.0027 | -0.0186 | -0.0006 | +0.0158 | -0.0016 [-0.0038, +0.0005] | -0.0073 [-0.0150, -0.0013] |
| s2_ndvi_amp | +0.0027 | -0.0027 | +0.0032 | -0.0059 | -0.0007 | +0.0075 | +0.0025 [-0.0003, +0.0053] | +0.0034 [+0.0008, +0.0064] |

Reading. Inside the ejido the gain is -0.0245 to -0.0008 on D3's rows, -0.0101 to -0.0007 with every tile, -0.0044 to +0.0012 with the same-place fit. Inside sq8km it is -0.0094 to +0.0046 on D3's rows, -0.0038 to +0.0025 with every tile, -0.0109 to +0.0038 with the same-place fit. The sq8km mean on D3's rows hides opposite folds: fold 1 reads -0.0308 to +0.0352, fold 4 -0.0262 to -0.0040. Fold 1 has 3 positive tiles against 1 negative tile and P reads 0.9274 there; fold 4 has P at 0.9931. With every tile fold 1's +0.0352 is +0.0084. Measured over P's own score instead of its refit (fit on three folds, read in fold): sq8km -0.0097 to +0.0055 on D3's rows and -0.0041 to +0.0021 with every tile; ejido -0.0195 to +0.0006 and -0.0103 to -0.0009.

The inputs together do not change this. Logit P with the 14 Sentinel-2 indices, the three Sentinel-2 Q1 indices, the three PlanetScope dates, or those dates and the 14 indices, each through the spline logistic and through gradient boosting (which allows interactions), fit on three folds and read in fold: sq8km -0.0071 to +0.0054 on D3's rows and -0.0037 to +0.0012 with every tile; ejido -0.0544 to -0.0158 and -0.0244 to -0.0060.

So these samples do not support the reading that Sentinel-2 adds 0.02-0.03 over the model in the centre. Limits: this is AUROC on a class-balanced sample of pixels with NDVI >= 0.3 and the edge band out (37.9 % of the valid inventory pixels in the ejido, 35.6 % in sq8km), and a late add-on to the model's final score. It is not AP at the true prevalence and not what a network trained with the input would do. E3 is read on its own registered metric.
