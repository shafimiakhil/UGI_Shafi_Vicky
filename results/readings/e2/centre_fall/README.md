# Why E2 falls inside the centre (CLAUDE.md 13.7: for information; E2 keeps its ejido reading)

**Finding.** No defect was found. The fall is a property of this one E2 run, mainly its fold 2 model and one 97 ha polygon. One seed cannot fix its size. The reading of E2 does not depend on it.

`python scripts/e2_centre_check.py --out ../docs/evidence_phase2/ugi_seg/e2/centre_fall` (from `work_ms/`) recomputes the tables below from the stored rasters and histograms, without the scoring code; its counts equal each run's `boot.npz` or it stops. Runs, seed 0 unless stated: `ugi_r1_none` seeds 0, 1, 2 (b0, b1, b2), `ugi_e1_ps` (E1), `ugi_e2_tex` (E2). Reviewers' results, marked "review", are in `../review_logs/` (scripts, logs, and `workflow_result.json`, workflow `wf_11c76268-525`) and were not recomputed here.

## Size (`combos.csv`, `e2_vs_seeds.csv`)

Each fold trains alone, so one seed per fold gives 3^4 = 81 pooled baseline scores. AP without CIRCUNVALACION (no such pixel lies inside a candidate):

| region | 81 baseline scores: min | median | max | E1 | E2 | share of the 81 below E2 |
|---|---|---|---|---|---|---|
| ring | 0.5858 | 0.6121 | 0.6277 | 0.6276 | 0.5681 | 0.00 |
| sq3km | 0.5852 | 0.6272 | 0.6927 | 0.5816 | 0.5715 | 0.00 |
| sq5km | 0.6205 | 0.6368 | 0.6797 | 0.6365 | 0.5913 | 0.00 |
| sq8km | 0.6232 | 0.6530 | 0.6768 | 0.6494 | 0.5792 | 0.00 |
| ejido | 0.4285 | 0.4494 | 0.4750 | 0.4658 | 0.4440 | 0.26 |
| ejido outside sq8km | 0.3925 | 0.4138 | 0.4412 | 0.4366 | 0.4248 | 0.81 |

E2 against the three seeds inside sq8km: -0.0738 [-0.1050, -0.0405], -0.0968 [-0.1469, -0.0389], -0.0445 [-0.0792, -0.0070]. Against b0 it stays under 3 x seed SD = 0.0787. AUROC inside sq8km is less clear: E2 0.8959, baseline scores 0.8932 to 0.9324, 2.5 % of them below E2.

The block interval holds no training-seed variance. Between baseline seeds it excludes zero in 8 of 12 candidate cells and in none of 3 in the ejido; inside sq8km: b1 - b0 +0.0230 [-0.0149, +0.0506], b2 - b0 -0.0293 [-0.0515, -0.0068], b2 - b1 -0.0523 [-0.0840, -0.0096] (`docs/evidence_phase2/ugi_centre/seed_pairs.csv`).

## Where inside sq8km

Per fold (`fold_ap.csv`), AP:

| fold | b0 | b1 | b2 | E1 | E2 | E2 minus lowest seed |
|---|---|---|---|---|---|---|
| 1 | 0.69072 | 0.71759 | 0.61109 | 0.64769 | 0.60544 | -0.00565 |
| 2 | 0.76146 | 0.78912 | 0.75097 | 0.76215 | 0.66169 | -0.08928 |
| 3 | 0.68183 | 0.67092 | 0.66709 | 0.74727 | 0.69251 | +0.02542 |
| 4 | 0.71121 | 0.68327 | 0.69761 | 0.70410 | 0.68688 | +0.00361 |

At each run's own thresholds (`precision_recall.csv`), E2 loses true positives and gains false positives, both outside the range of the seeds:

| run | TP | FP | precision | recall |
|---|---|---|---|---|
| b0 | 2,371,014 | 889,561 | 0.7272 | 0.4455 |
| b1 | 2,442,938 | 767,988 | 0.7608 | 0.4590 |
| b2 | 2,436,459 | 891,659 | 0.7321 | 0.4578 |
| E1 | 2,458,265 | 816,742 | 0.7506 | 0.4619 |
| E2 | 2,124,277 | 983,503 | 0.6835 | 0.3991 |

By polygon size inside sq8km (`by_size_sq8km.csv`), change in TP against b0. The five polygons over 100,000 px hold 56 % of the inventory pixels and 84 % of E2's net loss; the other runs gain there:

| polygons | n | pixels | TP b0 | b1 - b0 | b2 - b0 | E1 - b0 | E2 - b0 |
|---|---|---|---|---|---|---|---|
| < 10,000 px | 644 | 1,238,691 | 694,105 | -43,351 | -25,718 | -23,955 | -17,969 |
| 10,000-100,000 px | 54 | 1,090,358 | 530,325 | -29,200 | +570 | +4,627 | -22,693 |
| > 100,000 px | 5 | 2,993,224 | 1,146,584 | +144,475 | +90,593 | +106,579 | -206,075 |

The largest, polygon 3048 (967,840 px inside sq8km), has coverage 0.2553, 0.4533, 0.3468 (b0, b1, b2), E1 0.4126, E2 0.1366 (`polygons_sq8km.csv`; `type` is blank where a polygon has less than half its area inside). Without the five, AP is 0.5808, 0.5738, 0.5621, E1 0.5753, E2 0.5424: the fall against b0 halves to -0.0384 and E2 stays 0.0197 below the lowest seed.

## Fit (`fit.csv`)

| fold | internal-validation AP at the best epoch: b0, b1, b2 | E2 | training loss, mean of epochs 51-60: b0, b1, b2 | E2 |
|---|---|---|---|---|
| 1 | 0.80503, 0.79916, 0.79657 | 0.78056 | 0.7896, 0.8028, 0.6843 | 0.8270 |
| 2 | 0.73724, 0.74038, 0.71428 | 0.69423 | 0.9735, 0.9179, 0.9061 | 1.1621 |
| 3 | 0.68891, 0.69177, 0.68236 | 0.69369 | 0.9399, 0.8956, 0.8757 | 0.9735 |
| 4 | 0.73846, 0.74619, 0.73902 | 0.72946 | 0.8807, 0.8493, 0.8747 | 0.9240 |

E2's training loss is above all three seeds in all four folds, most in fold 2. Review: on every third internal training tile of fold 2, AP is 0.692 for E2 against 0.760, 0.789, 0.760 (`cd_val2.log`), so that model fits worse on its own training tiles too; fold 1 does not (0.779 against 0.755-0.812). Muting channels at inference does not bring fold 2 back (`cd_ablate_score.log`), which places its shortfall in the weights.

## Ruled out (review; `../review_logs/workflow_result.json`)

- The numbers: 368 reported values recomputed, largest gap 5.4e-05; 1,632 raster-level checks of the scoring files, 0 failures.
- The comparison: the config differs from the baseline's only in its name and texture keys; same tiles, internal blocks, Neo statistics and class weights per fold.
- The texture path: training and inference give the same tensors on every window tried; each fold's own GLCM band is the closest to the stored map in 8 of 8 windows; normalisation statistics recomputed from the internal training tiles, difference 0.0; the checkpoint comes from internal validation only.
- The map: no lattice, tile-border, fold-line or seam pattern; the new conv1 channels moved from zero in every fold; no NaN or inf.
- Out-of-range input: inside sq8km at most 3.0 % of the values of any texture layer fall outside the training [0.5, 99.5] percentiles, against 1 % nominal. What differs is the relation to the label: on vegetated pixels hom9 separates UGI with AUROC 0.69-0.73 inside sq8km and 0.45-0.54 on the training tiles (`s2_texture.log`).

## Not done

No second E2 seed, so E2's own seed spread is unknown. 13.6 runs seeds 1-2 only for an input that helps.
