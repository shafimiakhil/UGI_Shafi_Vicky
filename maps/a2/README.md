# A2, seed 0

This is `ugi_r3_a2_none`, seed 0, with the encoder started from the GEOBIA land-cover model. The probability is the out-of-fold raster of the four folds, and a pixel is mapped when it reaches its fold's own threshold. Figures are in `../../figures/a2/`.

This map and its pair are built exactly as Stage M2, with one seed, so that A1 and A2 compare like for like. The three-seed map in `../m2/` stays the map. The table of the pair is `../../results/maps/pair_a2_vs_a1_seed0.csv`. A polygon that crosses a fold line is scored whole on a map and by its part in one fold in a run, so object recall here can differ from the runs' summaries by a few polygons.

| file | what it holds |
|---|---|
| `ugi_a2_seed0_vs_inventory.tif` | uint8. Inside the ejido, 1 = in the inventory and mapped, 2 = mapped outside the inventory, 3 = in the inventory, not mapped. 4 = mapped outside the ejido, where there is no reference. 0 = nodata, transparent |
| `ugi_a2_seed0_predicted.tif` | uint8. 1 = mapped as UGI, 0 = nodata, transparent |
| `ugi_a2_seed0_vs_inventory.qml`, `ugi_a2_seed0_predicted.qml` | styles, which QGIS applies on loading because they share the raster's name and folder |
| `counts.json` | pixel counts of the comparison raster |

Rasters lie on the 1 m grid in EPSG:32720, tiled, DEFLATE, with an embedded colour table and internal overviews. Pixels inside the ejido are 17,586,174 in the inventory and mapped, 9,419,828 mapped outside it and 15,024,394 in the inventory and not mapped; 224,094 are mapped outside the ejido.

| pixels | AP | AUROC | IoU | precision | recall | tolerant object recall |
|---|---|---|---|---|---|---|
| ejido | 0.5924 | 0.9007 | 0.4184 | 0.6512 | 0.5393 | 0.4850 |
| sq8km | 0.6088 | 0.9075 | 0.3593 | 0.7119 | 0.4204 | 0.4551 |

The probability raster `ugi_r3_a2_none_seed0_prob.tif` (about 1.2 GB) is not here. From `code/`, `bash jobs/maps_a2_pair.sh` rebuilds it, these layers and the figures from the runs' stored `prob_oof.tif`.
