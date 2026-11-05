"""End-to-end smoke test on synthetic data (no real data needed): tile index and report, tile cache, one fold and seed
of training (2 epochs), held-out evaluation, map prediction; then a check that only this package and its listed
dependencies were imported (no other project code).

  python -m <package>.smoke_test [--keep DIR]     (files stay in DIR or in a new temporary folder; nothing is deleted)

The synthetic scene: 768 x 768 px at 1 m (EPSG:32720), 7 classes as smooth random regions, 4 bands whose values depend
on the class plus noise, a 20 px unlabelled border and two "strips" (left and right halves). Every metric of this test
is meaningless by construction; it only shows that the pipeline runs.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter

from .data import Paths, build_cache, tile_index, write_csv
from .prepare import report
from .train import run

FORBIDDEN = ("training", "dataset_builder", "preprocessing", "inference", "work_ms")


def synthetic(d, n=768, K=7, seed=0):
    rng = np.random.default_rng(seed)
    f = np.stack([uniform_filter(rng.normal(size=(n, n)), 40, mode="wrap") for _ in range(K)])
    lab = (f.argmax(0) + 1).astype(np.uint8)
    lab[:20], lab[-20:], lab[:, :20], lab[:, -20:] = 0, 0, 0, 0
    means = rng.uniform(500, 4000, size=(K + 1, 4))
    img = np.clip(means[lab].transpose(2, 0, 1) + rng.normal(0, 300, size=(4, n, n)), 1, 10000).astype(np.uint16)
    img[:, 300:310, 400:450] = 65535
    strip = np.where(np.arange(n)[None, :] < n // 2, 1, 2).repeat(n, 0).astype(np.uint8)
    tr = from_origin(375000.0, 6536000.0, 1.0, 1.0)
    common = dict(driver="GTiff", width=n, height=n, crs="EPSG:32720", transform=tr, tiled=True, blockxsize=128,
                  blockysize=128)
    with rasterio.open(d / "image.tif", "w", count=4, dtype="uint16", nodata=65535, **common) as o:
        o.write(img)
    with rasterio.open(d / "labels.tif", "w", count=1, dtype="uint8", nodata=0, **common) as o:
        o.write(lab, 1)
    with rasterio.open(d / "strips.tif", "w", count=1, dtype="uint8", nodata=0, **common) as o:
        o.write(strip, 1)
    return 375000.0 + n / 2 + 0.5, 6536000.0 - n / 2 - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    a = ap.parse_args()
    t0 = time.time()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="unet_ugi_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True)
    xl, yl = synthetic(tmp)
    cfg = {"name": "smoke", "grid_res": 1.0,
           "paths": {"image": str(tmp / "image.tif"), "labels": str(tmp / "labels.tif"), "strips": str(tmp / "strips.tif"),
                     "tiles": str(tmp / "tiles" / "tiles.csv"), "cache": str(tmp / "cache"), "runs": str(tmp / "runs")},
           "classes": ["Trees", "Low vegetation", "Water", "Buildings", "Asphalt", "Bare soil", "Shadows"],
           "strip_names": {0: "none", 1: "left", 2: "right"},
           "folds": {"x_line": xl, "y_line": yl},
           "tiles": {"size": 128, "stride": 128, "min_share": 0.5, "cache_min_share": 0.3},
           "internal_val": {"block_m": 256, "frac": 0.3, "seed": 1000, "buffer_m": 16},
           "train": {"epochs": 2, "batch": 4, "lr": 1e-3, "wd": 1e-4, "dice_weight": 1.0, "ignore_band": True,
                     "workers": 0, "photometric": {"global": [0.9, 1.1], "per_band": 0.03}},
           "eval": {"window": 128, "band_rows": 256}}
    p = cfg["paths"]
    paths = Paths(p["image"], p["labels"], p["strips"])
    tiles = tile_index(paths, 128, 128, xl, yl, 7)
    Path(p["tiles"]).parent.mkdir(parents=True, exist_ok=True)
    write_csv(tiles, p["tiles"])
    rep = report(cfg, tiles)
    keep = [t for t in tiles if t["share"] >= 0.3 and not t["straddle"]]
    build_cache(paths, keep, p["cache"])
    write_csv(keep, Path(p["cache"]) / "tiles.csv")
    res = run(cfg, 1, 0, Path(p["runs"]) / "fold1_seed0")
    from .infer import predict_to_raster
    from .model import UNetResNet18
    import torch
    ck = torch.load(Path(p["runs"]) / "fold1_seed0" / "best.pt", map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = UNetResNet18(7, pretrained=False).to(dev)
    m.load_state_dict(ck["model"])
    predict_to_raster(m, p["image"], tmp / "map.tif", (0, 0, 768, 768), ck["mean"], ck["sd"], 7, device=dev, n=128,
                      band_rows=256)
    with rasterio.open(tmp / "map.tif") as s:
        arg = s.read(1)
    bad = sorted(m_ for m_ in sys.modules if m_.split(".")[0] in FORBIDDEN
                 or (m_.split(".")[0] == "itareo" and __package__.split(".")[0] != "itareo"))
    out = {"package": __package__, "tiles": len(tiles), "tiles_kept_50": rep["thresholds"]["0.5"]["tiles_kept"],
           "held_out_miou_full": res["held_out"]["full"]["miou"], "held_out_miou_eroded": res["held_out"]["eroded"]["miou"],
           "map_classes_present": sorted(int(v) for v in np.unique(arg)), "forbidden_modules_imported": bad,
           "seconds": round(time.time() - t0, 1), "workdir": str(tmp)}
    out["passed"] = bool(not bad and np.isfinite(out["held_out_miou_full"]) and len(tiles) > 0
                         and set(np.unique(arg)) <= set(range(1, 8)) | {255} and (arg == 255).any()
                         and "held_out_auroc_ap" in res)          # 255 = invalid input (the synthetic hole)
    print(json.dumps(out, indent=1))
    assert out["passed"], "smoke test failed"


if __name__ == "__main__":
    main()
