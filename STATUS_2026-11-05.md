# UGI mapping in Córdoba, status on 5 November 2026

Prepared by Shafi Ullah Miakhil for Dr. María Victoria Marinelli and Prof. Paolo Gamba. Every number comes from a file
of the project; the file paths are in the README of the shared repository (`UGI_Shafi_Vicky`), which also holds the
code, the result tables, the maps and the figures.

## 1. What was built

A segmentation model that maps the municipal UGI inventory (3,169 polygons, 32.63 km2, every polygon and type counting
as UGI) from the Pléiades Neo image of October 2022 at 1 m. The model is a U-Net with a ResNet-18 encoder started
from ImageNet and widened to the six Neo bands, trained and evaluated under spatial cross-validation (four quadrants
of the city, tiles on a fold line excluded), with checkpoints, thresholds and normalisation taken only from the
training quadrants. The reference area is the municipal ejido. Inside it, every detection outside the inventory is a
false positive. The 8 km square around the historic centre is the test area agreed with Prof. Gamba.

## 2. What the model reaches

| map | region | AUROC | AP | IoU | precision | recall | inventory spaces found (tolerant rule) |
|---|---|---|---|---|---|---|---|
| First baseline, 30 epochs, three seeds (Stage M) | ejido | 0.915 | 0.587 | 0.386 | 0.642 | 0.492 | 0.448 |
| Converged baseline, 60 epochs, three seeds (Stage M2, the current map) | ejido | 0.926 | 0.647 | 0.438 | 0.698 | 0.541 | 0.481 |
| Single model, seed 0 | ejido | 0.912 | 0.616 | 0.427 | | | 0.493 |
| Single model trained with the centre held out (fold C, three seeds) | centre | 0.911 | 0.621 +- 0.007 | 0.332 | | | |

The gain from Stage M to Stage M2 (AP +0.061, interval +0.049 to +0.072) came from the training protocol, no ignored
boundary band, the re-registered mosaic and 60 instead of 30 epochs, and from averaging three seeds. Water corridors
rose most (object recall 0.30 to 0.42), road greens next (0.29 to 0.35). Spaces narrower than 6 m are still found in
fewer than one case in ten, and large mixed spaces, parks and water corridors, hold about two thirds of the missed
area. A model trained without ever seeing the centre reaches AP 0.621 there, 0.035 to 0.056 below the quadrant models.

The encoder pretrained on the 2017 Pléiades land-cover map (A2) scores below the ImageNet start (AP 0.592 against
0.616 in the ejido, paired difference -0.024, interval -0.035 to -0.011), so ImageNet is kept.

## 3. What the other sensors add, tested one at a time

Each input was added alone to the baseline under the same protocol, on the same tiles, folds and seed, and read by a
rule written before the run. The smallest gain the design can detect at one seed is 0.037 AP in the ejido and 0.054
in the centre, which is three times the seed-to-seed spread of the baseline.

| input | how it entered | AP change, centre | AP change, ejido | reading |
|---|---|---|---|---|
| PlanetScope 24 Apr, 10 Sep, 28 Oct 2022 | branch at the 4 m stage, constant registration | -0.004 | +0.016 | no gain |
| PlanetScope, registration plane per date | branch at the 4 m stage | -0.008 | -0.012 | no gain |
| PlanetScope upsampled into the stem at 1 m | extra input channels | +0.010 | -0.002 | no gain |
| Optical texture (0.3 m PAN lawn layers, NDVI orientation and GLCM) | extra input channels | -0.074 | -0.005 | no gain |
| Sentinel-2 2022 quarterly indices | branch at the 8 m stage | +0.002 | -0.022 | no gain |

Three diagnostics make the readings defensible. The fusion gates opened and the signal reached the output; blanking
the added input at inference lowers the scores far below the baseline, so the models did use it; and the best single
feature of each series separates inventory UGI from other vegetation at AUROC 0.61 (PlanetScope) and 0.64
(Sentinel-2) where Neo's NDVI gives 0.51, yet over the model's own score the features add -0.016 to 0.000. The
information in the series is largely contained in the 1 m image for this target. The target is administrative, public
green, not vegetation. Of the inventory area, 23 % is vegetated in every 2022 observation, 67 % part of the year and
9 % never (27 % of the ring-road reserves, 16 % of the campus), so seasonality separates green from bare but not
public from private green.

Sentinel-1 2021 was downloaded and checked and is kept as a record. The SAOCOM 2021 file turned out to be a composite
of three acquisitions whose content lies about 95 m, 14 m and 950 m apart, so it cannot be registered; the per-date
products are requested. Prof. Gamba's advice of 4 November, optical only, stands.

## 4. What comes next

1. A context model that sees 2,048 m around each pixel at 4 m, fused with the 1 m model (training started 5 Nov).
   It targets the large spaces that hold most of the missed area.
2. Late fusion, as Prof. Gamba suggested. One model per source (Neo at 1 m, Neo at 4 m, PlanetScope at 3 m,
   Sentinel-2 at 10 m), each a result on its own, then a learned fusion head trained on out-of-fold maps, always
   against the ensemble of baseline seeds as the control.
3. One Transformer encoder arm under the same protocol, to test whether attention gives the context that the
   convolutional model lacks.
4. A dedicated 0.3 m member for narrow road greens, where the current model fails.
5. The product for D3. UGI extent at 1 m together with the surface persistence class of every pixel, joined to the
   thermal archive, with the hypothesis that persistent green cools and seasonal green cools little in the dry
   season (after Li et al. 2026, Urban Forestry and Urban Greening).
