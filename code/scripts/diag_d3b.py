"""Beyond the registered D3 of CLAUDE.md 13.8: is what the PlanetScope and Sentinel-2 features separate already known to the model? Added because a coarse NDVI measures the neighbourhood: on the screen's samples a 4 m or 8 m NDVI may separate inventory from other vegetated pixels while a U-Net with a 512 m window over the 1 m Neo image already sees that neighbourhood. The rule of 13.8 is unchanged and reads d3.json only; this adds columns for the person who reads it. CPU only; reads stored files; refuses an existing output file.

On the samples of scripts/diag_d3.py (d3_samples.csv):
  * neo_ndvi_4m, neo_ndvi_8m: Neo v3 NDVI (derived/features/neo2022_v3_indices_grid1m.tif) averaged over the valid pixels of the 4 m and 8 m stage cells of the sample's tile: what the model can compute from its own input. Same AUROC as the screen, in the ejido and in sq8km.
  * baseline_p: P(UGI) of ugi_r1_none, seed 0, from the prob_oof.tif of the sample's own fold, so out of fold. Its AUROC on the same samples is the skill the input has to add to.
  * dauc_over_neo_stage_{mean,min}: T.inner_auc of NDVI + CHM + the Neo NDVI at the input's own stage + feature, minus the same without the feature (the add-on check of the screen, spline logistic regression, inner cross-validation over the three training folds of each set).
  * dauc_over_model_{mean,min}: T.inner_auc of logit(baseline_p) + feature, minus logit(baseline_p) alone.
Both per region, for the five ps_ndvi_* features and the 14 s2_* features, on the rows where everything exists. Sets of a region with fewer than 50 samples of a class in a held-out set are left out.

Output in <d3 folder>: d3b_over_neo_and_model.csv, d3b_floors.csv. From work_ms/: python scripts/diag_d3b.py --d3 ../docs/evidence_phase2/ugi_seg/diagnostics/d3_screen [--workers 6]
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import roc_auc_score

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import u04_texture_screen as T  # noqa: E402

from diag_d3 import MIN_N, PS_NAMES, S2_NAMES, auc_sets, oriented  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402

FOLDS = (1, 2, 3, 4)
NEO_NDVI = Path("derived/features/neo2022_v3_indices_grid1m.tif")
RUNS = Path("runs_itareo/ugi/ugi_r1_none")
_D = {}


def _init(d):
    _D.update(d)


def job(a):
    region, name = a
    m, y, fold = _D["regions"][region], _D["y"], _D["fold"]
    x = _D["X"][name]
    stage = _D["X"]["neo_ndvi_4m" if name.startswith("ps_") else "neo_ndvi_8m"]
    ndv, chm, lp = _D["X"]["ndvi"], _D["X"]["chm"], _D["X"]["logit_p"]
    fin = np.isfinite(x) & np.isfinite(stage) & np.isfinite(ndv) & np.isfinite(chm) & np.isfinite(lp) & m
    over_neo, over_model, base_model = [], [], []
    for k in FOLDS:
        tr = fin & (fold != k)
        if min((y[tr] == 1).sum(), (y[tr] == 0).sum()) < 3 * MIN_N or min(((y[tr] == 1) & (fold[tr] == j)).sum() for j in FOLDS if j != k) < MIN_N or min(((y[tr] == 0) & (fold[tr] == j)).sum() for j in FOLDS if j != k) < MIN_N:
            continue
        a0 = T.inner_auc(np.c_[ndv[fin], chm[fin], stage[fin]], y[fin], fold[fin], k)
        a1 = T.inner_auc(np.c_[ndv[fin], chm[fin], stage[fin], x[fin]], y[fin], fold[fin], k)
        b0 = T.inner_auc(lp[fin][:, None], y[fin], fold[fin], k)
        b1 = T.inner_auc(np.c_[lp[fin], x[fin]], y[fin], fold[fin], k)
        over_neo.append(a1 - a0)
        over_model.append(b1 - b0)
        base_model.append(b0)
    if not over_neo:
        return region, name, {"sets": 0}
    return region, name, {"sets": len(over_neo), "rows": int(fin.sum()), "dauc_over_neo_stage_mean": round(float(np.mean(over_neo)), 4), "dauc_over_neo_stage_min": round(float(np.min(over_neo)), 4), "dauc_over_model_mean": round(float(np.mean(over_model)), 4), "dauc_over_model_min": round(float(np.min(over_model)), 4), "inner_auc_of_the_model_alone": round(float(np.mean(base_model)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    d3 = Path(a.d3)
    refuse_existing(d3 / "d3b_over_neo_and_model.csv", d3 / "d3b_floors.csv")
    df = pd.read_csv(d3 / "d3_samples.csv")
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
    y, fold = df["cls"].to_numpy(), df["fold"].to_numpy()
    # ---- the model's own score, out of fold ---------------------------------------------------------------------------------------------------------
    p = np.full(len(df), np.nan)
    for f in FOLDS:
        run = RUNS / f"fold{f}_seed0"
        r0, c0, H, W_ = json.load(open(run / "prob_oof.json"))["region_rows_cols"]
        with rasterio.open(run / "prob_oof.tif") as s:
            a_ = s.read(1)
        ix = np.nonzero(fold == f)[0]
        if not ((rows[ix] >= r0) & (rows[ix] < r0 + H) & (cols[ix] >= c0) & (cols[ix] < c0 + W_)).all():
            raise SystemExit(f"a sample of fold {f} lies outside its own held-out region")
        p[ix] = a_[rows[ix] - r0, cols[ix] - c0]
        del a_
    df["baseline_p"] = p
    df["logit_p"] = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    # ---- Neo NDVI at the stage grids -----------------------------------------------------------------------------------------------------------------
    n4, n8 = np.full(len(df), np.nan), np.full(len(df), np.nan)
    with rasterio.open(NEO_NDVI) as s:
        nd = s.nodata
        for (r0, c0), ix in df.groupby(["tile_row0", "tile_col0"]).indices.items():
            a_ = s.read(1, window=Window(int(c0), int(r0), 512, 512)).astype(np.float32)
            a_[a_ == nd] = np.nan
            a_ /= 1e4
            with np.errstate(invalid="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    g4 = np.nanmean(a_.reshape(128, 4, 128, 4), axis=(1, 3))
                    g8 = np.nanmean(a_.reshape(64, 8, 64, 8), axis=(1, 3))
            n4[ix] = g4[(rows[ix] - r0) // 4, (cols[ix] - c0) // 4]
            n8[ix] = g8[(rows[ix] - r0) // 8, (cols[ix] - c0) // 8]
    df["neo_ndvi_4m"], df["neo_ndvi_8m"] = n4, n8
    regions = {"ejido": df["in_ejido"].to_numpy(bool), "sq8km": df["in_sq8km"].to_numpy(bool)}
    # ---- floors -------------------------------------------------------------------------------------------------------------------------------------
    fl = []
    for region, m in regions.items():
        for name in ("ndvi", "chm", "neo_ndvi_4m", "neo_ndvi_8m", "baseline_p"):
            wo, inside, npos, nneg = auc_sets(df[name].to_numpy(float), y, fold, m)
            ori, sign = oriented(wo)
            fl.append({"region": region, "feature": name, "auc_mean": round(float(np.nanmean(ori)), 4), "auc_min": round(float(np.nanmin(ori)), 4), "sign": sign, **{f"auc_wo_fold{k}": round(float(v), 4) for k, v in zip(FOLDS, ori)}, "nan_share": round(float(1 - np.isfinite(df[name].to_numpy(float)[m]).mean()), 4)})
    with open(d3 / "d3b_floors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fl[0]))
        w.writeheader()
        w.writerows(fl)
    print(pd.DataFrame(fl).to_string(index=False), flush=True)
    # ---- what the inputs add ------------------------------------------------------------------------------------------------------------------------
    names = PS_NAMES + S2_NAMES
    data = {"X": {n: df[n].to_numpy(float) for n in names + ["ndvi", "chm", "neo_ndvi_4m", "neo_ndvi_8m", "logit_p"]}, "y": y, "fold": fold, "regions": regions}
    jobs = [(r, n) for r in regions for n in names]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(data,)) as ex:
        res = [{"region": r, "feature": n, **v} for r, n, v in ex.map(job, jobs)]
    out = pd.DataFrame(res)
    out.to_csv(d3 / "d3b_over_neo_and_model.csv", index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    # roc_auc_score is imported for diag_d3's helpers through the same environment; nothing else here uses it directly
    _ = roc_auc_score
    main()
