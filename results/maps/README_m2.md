# Stage M2, the converged map (CLAUDE.md 13.6)

**Code** (from `work_ms/`): `bash jobs/stage_m2.sh`, which runs `itareo/ugi_seg/stage_m2.py` on the functions of `stage_m.py` and `rescore9.py`. CPU only, no inference.

## Inputs

- **Runs:** `runs_itareo/ugi/ugi_r1_none/fold{1-4}_seed{0-2}/prob_oof.tif`, with each run's internal-validation threshold from `prob_oof.json`.
- **Image, indices, strips:** `derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif`, `derived/features/neo2022_v3_indices_grid1m.tif`, `derived/pneo2022_registered_v3/pneo2022_strip_id_grid1m.tif` (v3).
- **Labels:** `derived/ugi_labels/`. **Reference area:** `derived/ugi_labels/ejido_grid1m.tif`.

## Build (`ensemble_build.json`)

The rules of Stage M: the mean P of the three seeds of the fold that holds each pixel out; a pixel is voted UGI when at least two seeds exceed their own threshold. 592,385,157 pixels evaluated, 25,523,952 voted UGI.

## Scores

- **Inside the ejido, primary:** `ejido/metrics.json`, `ejido/metrics_rule7.json` and the object, component and block files beside them.
- **Full grid, secondary:** `ensemble_metrics.json`, `ensemble_metrics_rule7.json`, `objects.csv`, `components.csv`, `blocks_1km.csv`.
- **Against Stage M:** `before_after.csv`, with the protocol differences first.

Inside the ejido: pixel scores {"iou": 0.43819, "f1": 0.60936, "precision": 0.69756, "recall": 0.54096, "tp": 17641169, "fp": 7648611, "fn": 14969399, "tn": 535459598, "auroc": 0.9263, "ap": 0.64709}; AP and IoU with CIRCUNVALACION excluded 0.48391, 0.32807.
Full grid: pixel scores {"iou": 0.43566, "f1": 0.60691, "precision": 0.69117, "recall": 0.54096, "tp": 17641283, "fp": 7882669, "fn": 14969548, "tn": 551891657, "auroc": 0.92638, "ap": 0.64255}.

## Figures (`docs/evidence_phase2/ugi_seg/maps_m2/`)

`city_prob.jpg`, `city_outcome.jpg`, `zoom1_squares.jpg` to `zoom5_random.jpg`, `blocks_scatter.jpg`: the code and the five windows of Stage M (`zooms.csv` is a copy of `../maps/zooms.csv`; all five lie wholly inside the ejido). `ejido/city_outcome.jpg`, `ejido/blocks_scatter.jpg`: the outcome and the blocks inside the ejido, the reference area. QGIS layers: `qgis/`.
