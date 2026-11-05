# Stage M2, the map

This is the three-seed ensemble of `ugi_r1_none`. The probability is the mean of seeds 0, 1 and 2, and a pixel is mapped when at least two seeds exceed their own thresholds. Figures are in `../../figures/m2/`.

| file | what it holds |
|---|---|
| `ugi_m2_r1_none_vs_inventory.tif` | uint8. Inside the ejido, 1 = in the inventory and mapped, 2 = mapped outside the inventory, 3 = in the inventory, not mapped. 4 = mapped outside the ejido, where there is no reference. 0 = nodata, transparent |
| `ugi_m2_r1_none_predicted.tif` | uint8. 1 = mapped as UGI, 0 = nodata, transparent |
| `ugi_m2_r1_none_vs_inventory.qml`, `ugi_m2_r1_none_predicted.qml` | styles, which QGIS applies on loading because they share the raster's name and folder |
| `counts.json` | pixel counts of the comparison raster |

Rasters lie on the 1 m grid in EPSG:32720, tiled, DEFLATE, with an embedded colour table and internal overviews. Pixels inside the ejido are 17,641,169 in the inventory and mapped, 7,648,611 mapped outside it and 14,969,399 in the inventory and not mapped; 234,172 are mapped outside the ejido.

| pixels | AP | AUROC | IoU | precision | recall | tolerant object recall |
|---|---|---|---|---|---|---|
| ejido | 0.6471 | 0.9263 | 0.4382 | 0.6976 | 0.5410 | 0.4809 |
| sq8km | 0.6870 | 0.9401 | 0.3893 | 0.7560 | 0.4452 | 0.4418 |

The probability raster `ugi_r1_none_prob_mean.tif` (about 1.2 GB) is not here. From `code/`, `bash jobs/stage_m2.sh` rebuilds it, these layers and the figures from the runs' stored `prob_oof.tif`.
