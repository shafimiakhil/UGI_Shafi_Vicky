"""unet_ugi: D1 L1a semantic segmentation of urban green infrastructure and other land cover classes from 4-band VHR
imagery (R, G, B, NIR) with a U-Net on a torchvision ResNet-18 encoder (CLAUDE.md 12.3).

Self-contained: depends only on numpy, scipy, rasterio, pyyaml, torch and torchvision, never on the rest of itareo or
on Luigi's code, so that it can be shared as is (CLAUDE.md 12.5).

Modules: data (tiles, folds, internal validation, boundary band, tile cache, dataset), model (U-Net, conv1 init),
metrics (confusion matrices and scores), train (one fold and seed), infer (sliding-window inference), selftest.
"""
