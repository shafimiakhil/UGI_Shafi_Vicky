# Urban green infrastructure in Córdoba, results and how they were obtained

This folder holds the code, the result tables, the maps and the figures of the work on task D1 of Dr. Marinelli's plan, the mapping of urban green infrastructure (UGI) in Córdoba from Pléiades Neo 2022 and optical series of 2021 and 2022. It holds no source data, no checkpoints and no caches. Every number below comes from a file in `results/`.

## Data scope

The target is the municipal UGI inventory as delivered, 3,169 polygons of 14 types covering 32.63 km2, an administrative definition of public green in which every polygon and type counts, including the ring-road reserves (CIRCUNVALACION, 37 polygons, 7.43 km2, carriageways included) and the university campus. It is rasterised on a 1 m grid in EPSG:32720, with origin 375,327 E, 6,535,833 N and 24,297 by 24,381 pixels. The reference area is the municipal ejido, 575.72 km2, inside which every detection outside the inventory counts as a false positive. The test area is the 8 km square `sq8km` centred on 388,282.5 E, 6,525,025.5 N, with 679 evaluated polygons and no CIRCUNVALACION pixel, so AP and AP without CIRCUNVALACION coincide there.

| input | use |
|---|---|
| Pléiades Neo, 23 and 28 Oct 2022, 6 bands at 1 m, four strips registered and mosaicked | the model's image |
| Pléiades Neo PAN 0.3 m, NDVI texture | experiment E2 |
| PlanetScope, 24 Apr, 10 Sep and 28 Oct 2022, 3 m | experiments E1, E1b, E1c |
| Sentinel-2 quarterly medians, 2022 | experiment E3 |
| Pléiades 1A, 12 Feb 2017, with the GEOBIA land-cover map | encoder pretraining of A2 only, closed |

## Protocol

The baseline `ugi_r1_none` is a U-Net with a ResNet-18 encoder started from ImageNet, conv1 widened to the six Neo bands, and two heads, binary UGI (BCE plus Dice) and five groups (cross-entropy with median-frequency weights). It trains with AdamW, learning rate 3e-4, weight decay 1e-4, cosine decay, batch 8, tiles of 512 px, flips, 90-degree rotations and photometric jitter, for 60 epochs. Folds are the four quadrants of the grid, with straddling tiles left out. Internal validation lies in 2 km blocks (15 %) with a 128 m buffer inside the training folds. Normalisation comes from the training tiles, and checkpoint and threshold from internal-validation AP. Inference uses sliding windows of 512 px every 256 px with raised-cosine blending, and each run stores its out-of-fold probability raster. Every experiment keeps this protocol, changes one thing, and uses seed 0 against baseline seed 0. Two label-free fits for the PlanetScope inputs use the whole scene, the normalisation of each date to Sentinel-2 and the registration plane per date.

Scores are pooled over the four folds. Intervals are paired block bootstraps with 1 km blocks and 2,000 resamples. The seed noise of the pooled AP is the SD of the 81 one-seed-per-fold combinations of the three baseline seeds, 0.0124 in the ejido (AP without CIRCUNVALACION) and 0.0180 in `sq8km`. An input helps when its AP gain reaches three times that SD (0.0372 and 0.0540), its interval lies above zero, and it is positive in at least 3 of 4 folds. Inside `sq8km` the interval holds no seed noise, so the threshold is the test there and the interval is reported only. Any other outcome is "no gain", meaning not detectable at this power.

## Results

| what | region | result | reading | files |
|---|---|---|---|---|
| Baseline `ugi_r1_none`, 3 seeds | ejido | AP 0.6157 +- 0.0002; AP no C 0.4509 +- 0.0030; IoU 0.4258 +- 0.0040; tolerant object recall 0.4763 +- 0.0172 | the yardstick | `results/summary_r/summary_baseline_ejido.csv` |
| Stage M2 ensemble | ejido | AUROC 0.926, AP 0.647, AP no C 0.484, IoU 0.438, P 0.698, R 0.541, tolerant recall 0.481; against Stage M (30 epochs) AP +0.061 [+0.049, +0.072] | the map | `results/maps/before_after_stage_m_m2.csv`, `maps/m2/`, `figures/m2/` |
| A2, GEOBIA encoder (`ugi_r3_a2_none`) | ejido | AP 0.592 vs 0.616; AP no C 0.434 vs 0.449; paired AP -0.0235 [-0.0352, -0.0112] | below A1; closed | `results/maps/pair_a2_vs_a1_seed0.csv`, `results/summary_r/summary_r3_ejido.json`, `maps/a2/`, `maps/a1_seed0/`, `figures/a2/`, `figures/a1_seed0/` |
| R1 edge band (both, outer, inner, none) and R1b 120 epochs | full grid | none wins (AP 0.611); 120 epochs -0.003 | protocol fixed | `results/summary_r/summary_r1.json` |
| E1 PlanetScope 3 dates, 4 m branch, constant offset | ejido no C / B4 / sq8km | +0.0164 [-0.0015, +0.0346] / +0.0161 / -0.004 | no gain | `results/readings/e1/reading.md` |
| E1b same, plane per date | sq8km / ejido all / B4 | -0.0084 [-0.0403, +0.0256] / -0.0120 / -0.0109 | no gain; registration ruled out | `results/readings/e1b/reading.md` |
| E1c PlanetScope into the stem at 1 m | sq8km / ejido / B4 | +0.0102 [-0.0182, +0.0322] / -0.0015 / -0.0026 | no gain | `results/readings/e1c/reading.md` |
| E2 optical texture (PAN lawn, NDVI orientation, GLCM) | ejido no C / sq8km | -0.0054 [-0.0229, +0.0137] / -0.0738 [-0.1050, -0.0405] | no gain | `results/readings/e2/reading.md` |
| E3 Sentinel-2 2022 quarters at the 8 m stage | sq8km / ejido no C | +0.0020 [-0.0360, +0.0339] / -0.0216 [-0.0421, -0.0007] | no gain | `results/readings/e3/reading.md` |
| Fold C, baseline with `sq8km` held out (512 m buffer) | sq8km | AP 0.6212 +- 0.0073 (seeds 0.6281, 0.6136, 0.6219); quadrant models on the same pixels 0.6509 | the test of the paper; holding the centre out costs 0.035-0.056 AP | `results/foldc/README.md`, `results/foldc/seeds.csv` |
| Diagnostics D1-D3 | - | gates open; blanking the input lowers AP far below the baseline (E1b -0.12, E2 -0.15 in the ejido); screens, PlanetScope best 0.609, Sentinel-2 best 0.640, NDVI 0.509, CHM 0.563; over the model's own score the features add -0.016 to 0.000 | inputs used, redundant with Neo at this power | `results/diagnostics/README.md`, `results/diagnostics/d2_blank/`, `results/diagnostics/d3_screen/` |

The minimal detectable effects at one seed are 0.037 AP in the ejido and 0.054 in `sq8km`. At this power no 2021-2022 optical series or texture layer adds to the 1 m Neo image for this target. Misses are concentrated in large mixed spaces. At Stage M parks and water corridors held about two thirds of the missed area (area-weighted coverage 0.22 and 0.42 at M2), spaces under 6 m wide reach tolerant recall 0.07 to 0.10, and the ring road held 42 % of the correctly mapped area.

## How each result is reproduced

Every command runs from `code/`, which stands for the project's working folder, with the environment of `code/docs/ENVIRONMENT.md`. The job scripts name one local interpreter in their `PY` line, which needs changing. Source rasters and the inventory go under `raw_data/` and `gee_exports/` as the scripts name them. `itareo/grid.py` imports one function from `preprocessing/align_sar_hs_ms_common_grid.py` of Luigi Russo's public repository `luigirusso99/Slums_Mapping_SAR_HS_MS`, which has to sit beside `itareo/`.

| step | commands | writes |
|---|---|---|
| Neo mosaic | `python scripts/a6_pneo2022.py`, `python scripts/u03_neo_register.py` (flags in its docstring) | `derived/pneo2022_registered_v3/` |
| Labels, ejido, tiles | `python scripts/u01_ugi_labels.py`, `python scripts/u01b_polygon_attributes.py`, `python scripts/u01c_ejido_raster.py`, `python scripts/u08_tile_index.py` | `derived/ugi_labels/`, `derived/ugi_tiles/` |
| Features | `python scripts/u04_features.py`, `python scripts/u06_planetscope.py`, `python scripts/u06b_b4_inputs.py`, `bash jobs/e2_features.sh`, `itareo/gee/s2_quarterly.py` for the Sentinel-2 medians | `derived/features/`, `derived/planetscope2022/` |
| Tile cache | `python -m itareo.ugi_seg.prepare --cfg itareo/configs/ugi_r1_none.yaml` | `runs_itareo/ugi/cache/` |
| Any run | `python -m itareo.ugi_seg.train --cfg itareo/configs/<config>.yaml --fold <f> --seed <s> --out runs_itareo/ugi/<config>/fold<f>_seed<s>` | `split.json`, `epochs.csv`, `best.pt`, `prob_oof.tif`, `metrics.json`, `run_log.md` |
| Scores of a run | `python -m itareo.ugi_seg.rescore7 --runs runs_itareo/ugi/<config>`, `python -m itareo.ugi_seg.rescore9 --runs runs_itareo/ugi/<config>`, `python -m itareo.ugi_seg.rescore_centre --regions sq8km --runs runs_itareo/ugi/<config>` | `ejido/`, `centre_sq8km/` of each run |
| Summary | `python -m itareo.ugi_seg.summarize_r --arms ugi_r1_none <config> --seeds 0 --ejido --tag <tag>_ejido`, and with `--sub centre_sq8km --tag <tag>_centre_sq8km` | `runs_itareo/ugi/summary_r/` |
| Reading of an experiment | `python scripts/exp_report.py --arm <config> --tag <tag> --out <folder> --centre sq8km` (add `--b4` for PlanetScope) | `reading.md`, `paired_table*.csv`, `per_fold.csv`, `objects_by_class*.csv` |

| result | config | job | report |
|---|---|---|---|
| Baseline, R1, R1b | `ugi_r1_none`, `ugi_r1_both`, `ugi_r1_outer`, `ugi_r1_inner`, `ugi_r1b_both_e120` | `jobs/ugi_r1.sh`, `jobs/ugi_r1_post.sh`, `jobs/ugi_night_queue.sh` | `scripts/night_report.py` |
| A2 | `d1_l1a_main` for the encoder, then `ugi_r3_a2_none` | `jobs/d1_l1a_b3.sh`, `jobs/ugi_night_queue.sh` | `scripts/r3_interval_sensitivity.py` |
| Stage M2 | three seeds of `ugi_r1_none` | `bash jobs/stage_m2.sh` | `results/maps/README_m2.md` |
| A1 seed 0 and A2 maps, pair table | seed 0 of `ugi_r1_none` and `ugi_r3_a2_none` | `bash jobs/maps_a2_pair.sh` | `results/maps/pair_a2_vs_a1_seed0.csv` |
| E1 | `ugi_e1_ps` | `jobs/ugi_e1.sh`, `jobs/ugi_e1_post.sh` | `scripts/exp_report.py` |
| E1b | `ugi_e1b_ps_plane` | `jobs/night_queue_13_7.sh`, `scripts/night_post_13_7.py` | `scripts/exp_report.py`, `scripts/e1b_registration_check.py` |
| E2 | `ugi_e2_tex` | `jobs/ugi_e2.sh`, `jobs/ugi_e2_post.sh` | `scripts/exp_report.py`, `scripts/e2_centre_check.py` |
| E3, E1c | `ugi_e3_s2`, `ugi_e1c_ps1m` | `jobs/night_queue_13_8.sh`, `scripts/night_post_13_8.py` | `scripts/exp_report.py` |
| Fold C | `python -m itareo.ugi_seg.fold_c --prepare sq8km`, then `ugi_foldc_sq8km` with `--fold 5`, seeds 0-2 | `jobs/night_queue_13_7.sh` | `scripts/foldc_report.py` |
| Centre candidates | none, stored rasters | `python scripts/centre_candidates.py` | `results/centre/README.md` |
| D1, D2, D3 | the runs of E1, E1b, E2 | `python scripts/diag_d1.py`, `python scripts/diag_d2.py`, `python scripts/diag_d3.py`, `python scripts/diag_d3b.py` | `scripts/diag_d2_report.py`, `results/diagnostics/README.md` |

## Layout

`code/` holds the code as run. `results/` holds the summaries, the readings, the map tables, fold C and the diagnostics tables. `maps/` holds the QGIS layers and their styles, one folder per map with its README. `figures/` holds the city and zoom JPGs of each map. The readings and the diagnostics notes were written inside the project, so they cite its section numbers and folders such as `review_logs/` and `archive/`, which are not part of this folder. The probability rasters (about 1.2 GB each) and the checkpoints are not here, and the map job rebuilds the rasters from the runs' stored outputs.
