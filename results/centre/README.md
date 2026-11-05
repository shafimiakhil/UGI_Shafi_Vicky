# The centre of the city: candidates (CLAUDE.md 13.7, step 1)

`python scripts/centre_candidates.py` (from `work_ms/`; docstring for the steps). Inside the ejido; every candidate is limited to it. **Chosen: `sq8km`** (Shafi; `work_ms/derived/ugi_centre/chosen.txt`). It is the primary region from E1b on, with the AP threshold 3 x 0.0262 = 0.0787 (`runs_itareo/ugi/summary_r/summary_centre_sq8km.json`).

| | ring | sq3km | sq5km | sq8km |
|---|---|---|---|---|
| Area, km2 | 148.33 | 9.01 | 25.01 | 64.02 |
| UGI pixel share | 0.0756 | 0.0606 | 0.0518 | 0.0831 |
| Inventory polygons (>= 50 % inside) | 1762 | 62 | 227 | 679 |
| Inventory area inside, km2 | 11.21 | 0.55 | 1.30 | 5.32 |
| CIRCUNVALACION inside, km2 | 0.00 | 0.00 | 0.00 | 0.00 |
| Cached tiles wholly inside | 457 | 16 | 56 | 182 |
| Cached tiles touching | 569 | 36 | 90 | 240 |
| 1 km blocks | 176 | 16 | 36 | 81 |
| M2 AP | 0.6421 | 0.6684 | 0.6825 | 0.6870 |
| M2 AP, no C | 0.6421 | 0.6684 | 0.6825 | 0.6870 |
| M2 AUROC, no C | 0.9349 | 0.9559 | 0.9596 | 0.9401 |
| M2 IoU, no C | 0.3949 | 0.4169 | 0.4412 | 0.3893 |
| M2 F1, no C | 0.5663 | 0.5885 | 0.6122 | 0.5604 |
| M2 AP, vegetated pixels | 0.7058 | 0.6934 | 0.7502 | 0.7401 |
| NDVI alone, AP, vegetated | 0.1492 | 0.0796 | 0.0839 | 0.1268 |
| M2 AUROC, vegetated pixels | 0.9314 | 0.9328 | 0.9459 | 0.9340 |
| NDVI alone, AUROC, vegetated | 0.5234 | 0.3931 | 0.4234 | 0.4567 |
| M2 object recall, tolerant | 0.4972 | 0.3387 | 0.3568 | 0.4418 |
| M2 object recall, strict | 0.4608 | 0.3065 | 0.3172 | 0.4109 |
| M2 precision by area, tolerant | 0.8165 | 0.7990 | 0.8516 | 0.8909 |
| M2 block MAE, pp | 3.31 | 2.99 | 1.85 | 4.50 |
| Seeds 0-2: AP no C, mean | 0.6091 | 0.6333 | 0.6447 | 0.6509 |
| Seeds 0-2: AP no C, SD | 0.0193 | 0.0435 | 0.0253 | 0.0262 |
| 3 x SD (the AP threshold of 13.7) | 0.0580 | 0.1306 | 0.0760 | 0.0787 |
| 95 % block interval low, AP no C | 0.5681 | 0.4503 | 0.5330 | 0.5975 |
| 95 % block interval high, AP no C | 0.6494 | 0.7649 | 0.7386 | 0.7124 |
| Seeds 0-2: IoU, mean | 0.3907 | 0.4148 | 0.4376 | 0.3916 |
| Seeds 0-2: IoU, SD | 0.0092 | 0.0492 | 0.0228 | 0.0097 |
| Seeds 0-2: tolerant recall, mean | 0.4915 | 0.3279 | 0.3598 | 0.4394 |
| Seeds 0-2: tolerant recall, SD | 0.0187 | 0.0434 | 0.0313 | 0.0201 |

Whole ejido, for comparison (`runs_itareo/ugi/summary_r/summary_baseline_ejido.json`): AP no C 0.4509, SD 0.0030; IoU 0.4258, SD 0.0040; tolerant recall 0.4763, SD 0.0172.

## By group (`by_group.csv`)

| group | ring: polygons, km2 | sq3km: polygons, km2 | sq5km: polygons, km2 | sq8km: polygons, km2 |
|---|---|---|---|---|
| Road greens | 551, 1.43 | 12, 0.01 | 67, 0.11 | 218, 1.17 |
| Squares | 762, 2.55 | 27, 0.11 | 95, 0.30 | 285, 0.94 |
| Parks and green areas | 277, 3.55 | 5, 0.08 | 22, 0.11 | 85, 1.29 |
| Water corridors | 38, 2.63 | 3, 0.20 | 8, 0.51 | 20, 1.33 |
| Rail verges | 73, 0.90 | 15, 0.15 | 31, 0.27 | 46, 0.48 |
| NO DEFINIDO | 61, 0.16 | 0, 0.00 | 4, 0.00 | 25, 0.11 |

## Geometry

- **The ring (`derived/ugi_centre/`).** The 37 polygons close at a buffer of 10 m (`regions_log.json`). Inside it there is no CIRCUNVALACION polygon, so AP and AP no C coincide.
- **The historic centre lies 1624 m from the fold crossing:** Plaza San Martin at -782 m east, -1423 m north; the Cathedral at -892, -1413. Coordinates as printed by Wikipedia (https://en.wikipedia.org/wiki/Plaza_San_Mart%C3%ADn,_C%C3%B3rdoba, https://en.wikipedia.org/wiki/Cathedral_of_C%C3%B3rdoba,_Argentina), checked on `historic_centre.jpg`. Inventory polygon at the plaza's point: 1928.
- **All four folds meet at the crossing,** so a square on it takes a quarter from each quadrant, and its pixels were held out by four different models. Tiles straddling a fold line never trained any model and are not in the cache; `sq3km` has the largest share on them: 31 % of its pixels (13 of the 49 tiles it touches), against 19 % for `sq5km`, 12 % for `sq8km` and 9 % for `ring`; those pixels are still scored.
- **Figures:** `candidates_map.jpg`, `historic_centre.jpg`.

## Seeds

Each of `ugi_r1_none`'s 12 runs is rescored inside each candidate (`runs_itareo/ugi/ugi_r1_none/fold*_seed*/centre_<name>/`), and pooled per seed by `itareo.ugi_seg.summarize_r` (`runs_itareo/ugi/summary_r/summary_centre_<name>.json`). M2 is the ensemble of the three seeds (`runs_itareo/ugi/maps_m2/centre_<name>/`): its AP and AUROC are above the seeds' mean in all four candidates. Its IoU and object recall come from the vote map and can fall on either side (`sq8km` IoU and `sq5km` tolerant recall are lower). In `sq3km` M2 counts 62 polygons and the seeds 61: polygon 3226 lies across folds 1 and 2 and reaches 50 % inside the square only when they are pooled.

## What fold C would leave to train on (`foldc_preview.csv`, 512 m buffer)

| | ring | sq3km | sq5km | sq8km |
|---|---|---|---|---|
| Cached tiles left to train | 1453 | 2054 | 1984 | 1810 |
| UGI in them, km2 | 12.131 | 29.43 | 28.142 | 23.387 |
| UGI scored, km2 | 11.215 | 0.546 | 1.296 | 5.322 |
| Road greens left, km2 | 0.654 | 8.757 | 8.317 | 6.992 |
| Squares left, km2 | 1.174 | 3.619 | 3.37 | 2.547 |
| Parks left, km2 | 2.9 | 6.503 | 6.434 | 4.58 |
| Water corridors left, km2 | 4.23 | 6.292 | 5.902 | 5.575 |
| Rail verges left, km2 | 0.987 | 1.733 | 1.653 | 1.309 |

With `ring` held out, every training tile lies outside the ring road, in the peri-urban part of the city. With a square, the urban fabric around it stays in training.

`seed_pairs.csv` (`scripts/e2_centre_check.py`): the AP difference between baseline seeds inside each candidate and in the ejido, with the paired block bootstrap. Inside the candidates the interval excludes zero in 8 of 12 cells, in the ejido in none of 3. It holds no training-seed variance, so inside the centre the threshold 3 x seed SD is the test that covers it.
