# Was registration the cause of E1's result? (CLAUDE.md 13.7)

**Finding, at one seed: no sign that it was.** The plane cuts the residual, yet E1b scores no better than E1, and the blocks where the plane moves PlanetScope most gain no more than a placebo pair of baseline seeds shows. The reading of E1b is in `../reading.md` and does not depend on this note.

`python scripts/e1b_registration_check.py --out ../docs/evidence_phase2/ugi_seg/e1b/registration_check` (from `work_ms/`; reads stored files only). Runs, seed 0 unless stated: `ugi_r1_none` seeds 0, 1, 2 (b0, b1, b2), `ugi_e1_ps` (E1), `ugi_e1b_ps_plane` (E1b). AP is without CIRCUNVALACION.

## Registration

On the 306 blocks with a measured displacement, the median residual falls from 1.69 m with the constant to 1.02 m with the plane; the plane moves PlanetScope relative to the constant by a median 1.35 m, 90th percentile 2.96 m, maximum 4.75 m (`block_tests.json`). Per date, on 1,024 m blocks: `../../e1b_ps_plane_run2/residuals.json`, quoted in `../reading.md`.

## E1b against E1 (`e1b_vs_e1.csv`)

| region | E1 - b0 | E1b - b0 | E1b - E1 | share of pairs of the 81 baseline scores differing by at least that |
|---|---|---|---|---|
| sq8km | -0.0036 [-0.0322, +0.0195] | -0.0084 [-0.0403, +0.0256] | -0.0048 [-0.0300, +0.0194] | 0.82 |
| ejido | +0.0164 [-0.0015, +0.0346] | -0.0120 [-0.0307, +0.0070] | -0.0284 [-0.0408, -0.0160] | 0.04 |
| ejido outside sq8km | | | 0.0353 lower | 0.01 |

The plane does not help. Outside the centre E1b is lower than E1, near the edge of what seeds alone give; one seed of each cannot separate the two.

Internal-validation AP at the best epoch, E1b (three seeds): fold 1 0.7990 (0.7966-0.8050), fold 2 0.7301 (0.7143-0.7404), fold 3 0.6756 (0.6824-0.6918), fold 4 0.7434 (0.7390-0.7462) (`fit.csv`).

## Block test (`blocks.csv`, `block_tests.json`)

Rank correlation within fold over 1 km blocks with at least 5,000 inventory pixels. If registration mattered, the first two rows should be clearly positive and the next two clearly negative.

| size per block | change in block score | r, AP | p | r, AUROC | p | blocks |
|---|---|---|---|---|---|---|
| plane's correction | E1b - E1 | +0.073 | 0.147 | -0.028 | 0.583 | 388 |
| measured improvement | E1b - E1 | +0.045 | 0.435 | +0.001 | 0.982 | 306 |
| residual with the constant | E1 - b0 | -0.091 | 0.115 | -0.100 | 0.085 | 306 |
| residual with the plane | E1b - b0 | -0.014 | 0.823 | -0.028 | 0.624 | 306 |
| plane's correction, placebo | b1 - b0 | +0.029 | 0.573 | +0.041 | 0.421 | 388 |
| plane's correction, placebo | b2 - b0 | -0.051 | 0.328 | -0.079 | 0.121 | 388 |
| plane's correction, placebo | b2 - b1 | -0.006 | 0.909 | -0.149 | 0.003 | 388 |

The third row points weakly the expected way. The p-values permute blocks within fold and ignore spatial structure, as the last placebo shows, so they are not calibrated. The SD of the change in block AP is 0.089 for E1b - E1 against 0.090-0.109 for the seed pairs.

## Seed noise (`combos.csv`)

Each fold trains alone, so one seed per fold gives 81 pooled baseline scores. Their SD estimates the seed noise of a pooled score better than the SD of three whole runs, which has 2 degrees of freedom. Three values per fold give a plug-in variance of about 2/3 of the true one, so the SD is multiplied by sqrt(3/2).

| region | 81 scores: min, median, max | SD of the 81, corrected | SD of 3 whole runs (used in the rule) | 3 x SD, 81 | 3 x SD, rule | E1 (share of the 81 below) | E1b |
|---|---|---|---|---|---|---|---|
| sq8km | 0.6232, 0.6530, 0.6768 | 0.0180 | 0.0262 | 0.0540 | 0.0787 | 0.6494 (0.38) | 0.6446 (0.32) |
| ejido | 0.4285, 0.4494, 0.4750 | 0.0124 | 0.0030 | 0.0372 | 0.010 | 0.4658 (0.93) | 0.4374 (0.09) |
| ejido outside sq8km | 0.3925, 0.4138, 0.4412 | 0.0128 | | 0.0384 | | 0.4366 (0.99) | 0.4013 (0.10) |

No reading changes with these values: E1, E1b and E2 read no gain under either. The rule as written stands unless Shafi changes it before E3's result exists (13.3, rule 6).

## Pooled scores and folds (`scores_by_run.csv`, `fold_ap.csv`)

Pooled AP and AUROC rank pixels across four separately trained models with their own score scales; IoU and F1 use each fold's own threshold.

| region | run | AP, pooled | AP, mean of folds | IoU | F1 |
|---|---|---|---|---|---|
| sq8km | b0 | 0.6530 | 0.7113 | 0.3817 | 0.5525 |
| sq8km | b1 | 0.6760 | 0.7152 | 0.4011 | 0.5726 |
| sq8km | b2 | 0.6237 | 0.6817 | 0.3921 | 0.5633 |
| sq8km | E1 | 0.6494 | 0.7153 | 0.4004 | 0.5719 |
| sq8km | E1b | 0.6446 | 0.7222 | 0.4162 | 0.5878 |
| ejido | b0 | 0.4494 | 0.4851 | 0.4271 | 0.5985 |
| ejido | b1 | 0.4544 | 0.4782 | 0.4213 | 0.5928 |
| ejido | b2 | 0.4488 | 0.4806 | 0.4289 | 0.6003 |
| ejido | E1 | 0.4658 | 0.4938 | 0.4324 | 0.6037 |
| ejido | E1b | 0.4374 | 0.4839 | 0.4190 | 0.5906 |

**Corrected after the review:** this first said the score scales are why E1b's pooled AP falls inside the centre while 3 of 4 folds, IoU and F1 rise. The scales account for 36-69 % of pooled minus mean of folds inside sq8km (b0, b1, b2, E1, E1b: 48, 69, 37, 36, 44 %; AP at equal FPR in every fold 0.6812, 0.7032, 0.6454, 0.6733, 0.6787, summed as the project's AP, `../review_logs/f1_scale.log`). The weights turn the sign of the fall against the rise: fold 1 holds 43.2 % of the positives, pooled E1b - b0 -0.0084 splits by fold as -0.0182, +0.0046, +0.0025, +0.0027, and the mean of folds goes from +0.0108 plain to -0.0072 weighted by positives. The scales add to its size: at equal FPR E1b - b0 is -0.0025, a scale step of -0.0059, no larger than between baseline seeds (+0.0010, +0.0065, +0.0055). An earlier version of this note gave 26-60 %: the trapezoid of `../review_logs/a3_pooling.py` starts at the recall reached at FPR 0 and so leaves 0.0022 to 0.0140 out of the AP; the equal-FPR columns of `a3_runs.csv` and `a3_differences.csv` and check:cause [9] of `workflow_result.json` keep that error. IoU and F1 rise mostly through polygon 3048 (`../reading.md`, Independent review). Fold C, one model, still has no such cost: its number and the four-quadrant column of `../../foldc/README.md` are not on the same footing. Shifting each fold's logits by its internal-validation threshold does not close the gap (`ap_threshold_aligned`): those thresholds run from 0.05 to 0.54 and are too noisy to serve as a scale.

## Limits

One seed each of E1 and E1b. Blocks of this table are observational. The cut of the residual is scene-wide: inside sq8km it is large on 24 Apr, small on 28 Oct and not distinguishable from zero on 10 Sep. The independent review, workflow `wf_f48ff81d-00c`, adds tests at fixed weights and is summarised in `../reading.md`; its logs are in `../review_logs/`. No run trains on inputs registered tile by tile, so a benefit from training at under 0.2 m is not excluded.
