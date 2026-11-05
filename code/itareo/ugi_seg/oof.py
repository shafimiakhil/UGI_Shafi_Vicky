"""Out-of-fold probability rasters (CLAUDE.md 13.3, rule 1).

Each run writes prob_oof.tif beside metrics.json: P(UGI) over its held-out quadrant from the sliding-window inference of
itareo.ugi_seg.evaluate.predict_region (windows of n px every n/2 px, raised-cosine blending), float32 stored with
NBITS=16 (float16 precision, as the PAN lawn layers), COG, deflate, NaN outside the quadrant and on invalid image pixels.
Beside it, oof_groups.csv: per inventory polygon the evaluated pixels and the sum of the blended group probabilities
(softmax of channels 1-5) over them, so type scores need no new inference. Rescoring reads these files and never infers
again.
  python -m itareo.ugi_seg.oof --runs runs_itareo/ugi/ugi_a1        (from work_ms/; skips runs that have prob_oof.tif)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.windows import Window

from .evaluate import predict_region
from .model import build

NID = 4000


def quadrant(image, folds, fold):
    """(in_r, in_c, region) of the held-out quadrant of `fold`, exactly as train.run defines it."""
    with rasterio.open(image) as src:
        Tr, Hh, Ww = src.transform, src.height, src.width
    if folds.get("centre"):
        from .fold_c import region_c
        return region_c(Hh, Ww, folds["centre"])
    left = Tr.c + (np.arange(Ww) + 0.5) * Tr.a <= folds["x_line"]
    south = Tr.f + (np.arange(Hh) + 0.5) * Tr.e <= folds["y_line"]
    in_c = left if fold in (1, 3) else ~left
    in_r = south if fold in (1, 2) else ~south
    ci, ri = np.nonzero(in_c)[0], np.nonzero(in_r)[0]
    region = (int(ri.min()), int(ci.min()), int(ri.max() - ri.min() + 1), int(ci.max() - ci.min() + 1))
    return in_r, in_c, region


def write_prob_cog(path, arr, transform, crs, description="P(UGI)"):
    """float32 array -> COG with NBITS=16 (float16 precision), deflate, nodata NaN (in-memory copy, no temp file)."""
    prof = dict(driver="COG", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32", nodata=np.nan, crs=crs,
                transform=transform, compress="DEFLATE", NBITS=16, BLOCKSIZE=512, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr[None])
        d.set_band_description(1, description)
    return Path(path).stat().st_size


def infer_oof(model, ck, cfg, fold, run_dir, device="cuda"):
    """Writes run_dir/prob_oof.tif and oof_groups.csv; returns (prob as read back from prob_oof.tif, region, seconds)."""
    t0 = time.time()
    image, labels = cfg["paths"]["image"], Path(cfg["paths"]["labels"])
    in_r, in_c, region = quadrant(image, cfg["folds"], fold)
    r_lo, c_lo, R, Cn = region
    prob = np.full((R, Cn), np.nan, np.float32)
    poly_px = np.zeros(NID, np.int64)
    poly_g = np.zeros((NID, 5))
    extra = None
    if cfg["model"].get("planetscope"):
        from .ps_branch import WindowPS
        extra = WindowPS(ck["ps_mean"], ck["ps_sd"], cfg["model"]["planetscope"].get("registration", "constant"))
    if cfg["model"].get("texture"):
        from .tex_branch import WindowTex
        extra = WindowTex(cfg, fold, ck["tex_mean"], ck["tex_sd"], region)
    if cfg["model"].get("plugin"):
        import importlib
        extra = importlib.import_module(cfg["model"]["plugin"]["module"]).window(cfg, fold, ck, region)
    centre = None
    if cfg["folds"].get("centre"):
        from .fold_c import centre_window
        centre = centre_window(cfg["folds"]["centre"])
    with rasterio.open(image) as im, rasterio.open(labels / "ugi_id_grid1m.tif") as ui:
        crs, G_T = im.crs, im.transform
        for b0, pu, pg in predict_region(model, image, region, ck["mean"], ck["sd"], n=cfg["eval"]["window"],
                                         band_rows=cfg["eval"]["band_rows"], device=device, extra=extra):
            rows = pu.shape[0]
            w = Window(c_lo, r_lo + b0, Cn, rows)
            ok = (im.read(window=w) != im.nodata).all(0) & in_r[r_lo + b0:r_lo + b0 + rows][:, None] & in_c[c_lo:c_lo + Cn][None, :]
            if centre is not None:
                ok &= centre.read(1, window=w) == 1
            prob[b0:b0 + rows] = np.where(ok, pu, np.nan)
            ids = ui.read(1, window=w).astype(np.int64)
            sel = ok & (ids > 0)
            ii = ids[sel]
            poly_px += np.bincount(ii, minlength=NID)
            for g in range(5):
                poly_g[:, g] += np.bincount(ii, weights=pg[g][sel], minlength=NID)
    if centre is not None:
        centre.close()
    run_dir = Path(run_dir)
    nbytes = write_prob_cog(run_dir / "prob_oof.tif", prob, G_T * Affine.translation(c_lo, r_lo), crs)
    with rasterio.open(run_dir / "prob_oof.tif") as s:      # score exactly what is stored (GDAL's NBITS=16 truncates)
        prob = s.read(1)
    with open(run_dir / "oof_groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pixels_evaluated"] + [f"sum_p_group{g}" for g in range(1, 6)])
        for pid in np.nonzero(poly_px)[0]:
            w.writerow([int(pid), int(poly_px[pid])] + [round(float(v), 4) for v in poly_g[pid]])
    secs = round(time.time() - t0, 1)
    json.dump({"fold": fold, "region_rows_cols": region, "bytes": nbytes, "seconds": secs, "threshold": ck["threshold"],
               "image": image, "nbits": 16}, open(run_dir / "prob_oof.json", "w"), indent=1)
    return prob, region, secs


def run_dir_oof(run_dir, device="cuda"):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    fold = int(run_dir.name.split("_")[0].replace("fold", ""))
    model = build("imagenet")             # architecture only; weights from the checkpoint
    model.load_state_dict(ck["model"])
    model = model.to(device)
    _, region, secs = infer_oof(model, ck, cfg, fold, run_dir, device)
    print(run_dir, region, secs, "s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    for d in a.runs:
        for run in sorted(Path(d).glob("fold*_seed*")):
            if (run / "best.pt").exists() and (run / "metrics.json").exists() and not (run / "prob_oof.json").exists():
                run_dir_oof(run)


if __name__ == "__main__":
    main()
