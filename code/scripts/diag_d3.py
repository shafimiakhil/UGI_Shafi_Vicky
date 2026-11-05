"""D3 of CLAUDE.md 13.8, the information screen: do the PlanetScope and Sentinel-2 inputs hold anything that separates inventory UGI from other vegetated pixels? CPU only; reads stored files; refuses an existing output folder.

As the texture screen (docs/evidence_phase2/texture_screen/README.md, scripts/u04_texture_screen.py), same samples and same AUROC: the 160,000 pixels of samples.csv (20,000 per fold and class; UGI = 1 against UGI = 0, edge band out, Neo NDVI >= 0.3), for each held-out fold k roc_auc_score on the pooled samples of the other three folds (auc_wo_fold<k>), oriented by the sign of the mean over the four sets; auc_mean and auc_min over the sets. Also auc_in_fold<k>, on the samples of fold k alone with the same orientation, so a feature whose sign flips by fold shows it. A set with fewer than 50 samples of a class gives no value.
Regions: all samples (to reproduce the screen), the ejido (samples inside derived/ugi_labels/ejido_grid1m.tif) and the centre sq8km (derived/ugi_centre/sq8km_grid1m.tif).

Features, read from what the models read (so a sample takes the value of its stage-grid cell):
  * PlanetScope, 4 m stage of E1b, ps_stage.stage_tile_plane (the plane per date removed inside the single resampling): ps_ndvi_20220424, ps_ndvi_20220910, ps_ndvi_20221028, ps_ndvi_min, ps_ndvi_amp (minimum and max - min over the three dates, only where all three exist). NDVI = (NIR - red) / (NIR + red), bands 8 and 6 of the 8-band product. psconst_*: the same with the constant of E1 (ps_stage.stage_tile), for E1's report;
  * Sentinel-2, 8 m stage of E3, s2_branch.stage_tile: the 14 indices of derived/features/s2_2022_indices_10m.tif (s2_ndvi_q1 ... s2_ndmi_q4, s2_ndvi_drymin, s2_ndvi_amp);
  * floor: ndvi and chm of samples.csv (Neo NDVI, Meta CHM). floor_ndvi_mean and floor_chm_mean are their auc_mean on the rows where the feature exists, because PlanetScope does not cover every sample.
Add-on check of the screen (T.inner_auc, on the rows where the feature, NDVI and CHM exist): dauc_ndvi_chm_mean and _min, the gain of NDVI + CHM + feature over NDVI + CHM. Inside the ejido only.

The number the queue reads, written before this script first ran: the PlanetScope screen reaches 0.60 (13.8) when the highest auc_mean among the five ps_ndvi_* features inside the ejido is at least 0.60. auc_mean, because the screen's numbers quoted in 13.3 (NDVI 0.509, best texture 0.680) are auc_mean; auc_min stands beside it. d3.json holds it; the 13.8 readings "near 0.5" and "at or above 0.60" are read by a person together with D2.

Self-checks, the run stops if one fails: the screen's auc_mean of ndvi, chm and ndvi_glcmcorr_41m reproduced to 4 decimals from samples.csv; the PlanetScope and Sentinel-2 values of 12 sampled tiles equal to E1b's and E3's caches where the tile is cached.

Outputs in <out>: d3_samples.csv, d3_auroc.csv, d3_by_group.csv, d3.json. Run from work_ms/: python scripts/diag_d3.py --out ../docs/evidence_phase2/ugi_seg/diagnostics/d3_screen [--workers 6]
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import u04_texture_screen as T  # noqa: E402

from itareo.ugi_seg import ps_stage, s2_branch  # noqa: E402

SCREEN = Path("../docs/evidence_phase2/texture_screen")
EJIDO = Path("derived/ugi_labels/ejido_grid1m.tif")
CENTRE = Path("derived/ugi_centre/sq8km_grid1m.tif")
S2 = Path("derived/features/s2_2022_indices_10m.tif")
PS_CACHE = Path("runs_itareo/ugi/cache/ps2022_stage4m_plane")
S2_CACHE = Path("runs_itareo/ugi/cache/s2_2022_stage8m")
NEO_TILES = Path("runs_itareo/ugi/cache/neo2022_registered_v3/tiles.csv")
S2_DX, S2_DY = -1.42, 3.98
FOLDS = (1, 2, 3, 4)
MIN_N = 50
S2_NAMES = [f"s2_{n}_q{q}" for q in (1, 2, 3, 4) for n in ("ndvi", "ndre", "ndmi")] + ["s2_ndvi_drymin", "s2_ndvi_amp"]
PS_NAMES = [f"ps_ndvi_{d}" for d in ps_stage.DATES] + ["ps_ndvi_min", "ps_ndvi_amp"]
PSC_NAMES = [n.replace("ps_", "psconst_") for n in PS_NAMES]


def ndvi(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a[7] - a[5]) / (a[7] + a[5])


def min_amp(stack):
    """(3, H, W) -> minimum and max - min, NaN unless all three dates exist."""
    bad = np.isnan(stack).any(0)
    lo, hi = np.where(bad, np.nan, stack.min(0)), np.where(bad, np.nan, stack.max(0))
    return lo, hi - lo


def read_features(df):
    planes, offs = ps_stage.load_planes(), ps_stage.load_offsets()
    src = {d: rasterio.open(ps_stage.PS.format(d)) for d in ps_stage.DATES}
    s2 = rasterio.open(S2)
    assert [d.split(" x ")[0] for d in s2.descriptions][:3] == ["NDVI 2022Q1", "NDRE 2022Q1", "NDMI 2022Q1"] and s2.count == 14
    out = {n: np.full(len(df), np.nan) for n in PS_NAMES + PSC_NAMES + S2_NAMES}
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
    groups = df.groupby(["tile_row0", "tile_col0"]).indices
    per_tile = {}
    for (r0, c0), ix in groups.items():
        r, c = rows[ix] - r0, cols[ix] - c0
        assert r.min() >= 0 and c.min() >= 0 and r.max() < 512 and c.max() < 512
        p = np.stack([ps_stage.stage_tile_plane(src[d], int(r0), int(c0), planes[d]) for d in ps_stage.DATES])     # (3, 8, 128, 128)
        q = np.stack([ps_stage.stage_tile(src[d], int(r0), int(c0), offs[d]["dx_east_m"], offs[d]["dy_north_m"]) for d in ps_stage.DATES])
        s = s2_branch.stage_tile(s2, int(r0), int(c0), S2_DX, S2_DY)
        per_tile[(int(r0), int(c0))] = (p, s)
        for names, stack in ((PS_NAMES, p), (PSC_NAMES, q)):
            nd = np.stack([ndvi(stack[k]) for k in range(3)])
            lo, amp = min_amp(nd)
            for name, layer in zip(names, [nd[0], nd[1], nd[2], lo, amp]):
                out[name][ix] = layer[r // 4, c // 4]
        for k, name in enumerate(S2_NAMES):
            out[name][ix] = s[k][r // 8, c // 8] / 1e4
    for v in src.values():
        v.close()
    s2.close()
    return out, per_tile


def check_caches(per_tile):
    """The values read here equal the caches E1b and E3 train on, on 12 sampled tiles that are cached."""
    index = {(int(t["row0"]), int(t["col0"])): i for i, t in enumerate(csv.DictReader(open(NEO_TILES)))}
    ps, s2 = np.load(PS_CACHE / "ps.npy", mmap_mode="r"), np.load(S2_CACHE / "s2.npy", mmap_mode="r")
    keys = [k for k in per_tile if k in index]
    rng = np.random.default_rng(0)
    res = {"tiles_sampled": len(per_tile), "tiles_cached": len(keys), "checked": 0}
    for j in rng.choice(len(keys), 12, replace=False):
        k = keys[j]
        p, s = per_tile[k]
        if not (np.array_equal(p.reshape(24, 128, 128), np.asarray(ps[index[k]]), equal_nan=True) and np.array_equal(s, np.asarray(s2[index[k]]), equal_nan=True)):
            raise SystemExit(f"tile {k}: the values read here differ from the caches")
        res["checked"] += 1
    return res


def auc_sets(x, y, fold, m):
    """Raw AUROC on the pooled other folds and on the fold itself, inside mask m; NaN where a class is thin."""
    wo, inside, npos, nneg = {}, {}, {}, {}
    fin = np.isfinite(x) & m
    for k in FOLDS:
        for store, sel in ((wo, fin & (fold != k)), (inside, fin & (fold == k))):
            yy = y[sel]
            np_, nn_ = int((yy == 1).sum()), int((yy == 0).sum())
            store[k] = roc_auc_score(yy, x[sel]) if min(np_, nn_) >= MIN_N else np.nan
            if store is wo:
                npos[k], nneg[k] = np_, nn_
    return wo, inside, npos, nneg


def oriented(wo):
    raw = np.array([wo[k] for k in FOLDS])
    sign = 1 if np.nanmean(raw) >= 0.5 else -1
    return raw if sign > 0 else 1 - raw, sign


def row_for(name, x, y, fold, m, tiles, floors):
    wo, inside, npos, nneg = auc_sets(x, y, fold, m)
    ori, sign = oriented(wo)
    r = {"feature": name, **{f"auc_wo_fold{k}": round(float(a), 4) for k, a in zip(FOLDS, ori)}, "auc_mean": round(float(np.nanmean(ori)), 4), "auc_min": round(float(np.nanmin(ori)), 4), "sign": sign,
         "sign_consistent": bool(np.all((np.array([wo[k] for k in FOLDS]) - 0.5) * sign > 0)), **{f"auc_in_fold{k}": round(float(inside[k] if sign > 0 else 1 - inside[k]), 4) for k in FOLDS},
         "n_pos_min": min(npos.values()), "n_neg_min": min(nneg.values()), "nan_share": round(float(1 - np.isfinite(x[m]).mean()), 4), "tiles": int(pd.unique(tiles[m & np.isfinite(x)]).size)}
    fin = np.isfinite(x)
    for fl, v in floors.items():
        w_fl, _, _, _ = auc_sets(v, y, fold, m & fin)
        r[f"floor_{fl}_mean"] = round(float(np.nanmean(oriented(w_fl)[0])), 4)
    return r


_D = {}


def _init(d):
    _D.update(d)


def addon(job):
    """The add-on check of the screen: inner AUROC of NDVI + CHM + feature against NDVI + CHM, per held-out fold."""
    name = job
    x, y, fold, m = _D["X"][name], _D["y"], _D["fold"], _D["m"]
    ndv, chm = _D["X"]["ndvi"], _D["X"]["chm"]
    fin = np.isfinite(x) & np.isfinite(ndv) & np.isfinite(chm) & m
    d = []
    for k in FOLDS:
        a = T.inner_auc(np.c_[ndv[fin], chm[fin], x[fin]], y[fin], fold[fin], k)
        b = T.inner_auc(np.c_[ndv[fin], chm[fin]], y[fin], fold[fin], k)
        d.append(a - b)
    return name, round(float(np.mean(d)), 4), round(float(np.min(d)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cols = ["fold", "cls", "tile_row0", "tile_col0", "row", "col", "group", "ndvi", "chm", "ndvi_glcmcorr_41m"]
    df = pd.read_csv(SCREEN / "samples.csv", usecols=cols)
    y, fold = df["cls"].to_numpy(), df["fold"].to_numpy()
    tiles = (df["tile_row0"].astype(np.int64) * 100000 + df["tile_col0"]).to_numpy()
    # ---- the screen reproduced ----------------------------------------------------------------------------------------------------------------
    scr = pd.read_csv(SCREEN / "auroc.csv").set_index("feature")
    every = np.ones(len(df), bool)
    for n in ("ndvi", "chm", "ndvi_glcmcorr_41m"):
        got = round(float(oriented(auc_sets(df[n].to_numpy(float), y, fold, every)[0])[0].mean()), 4)
        if abs(got - float(scr.loc[n, "auc_mean"])) > 1e-4:
            raise SystemExit(f"{n}: auc_mean {got} here, {scr.loc[n, 'auc_mean']} in the screen")
    # ---- features and regions -----------------------------------------------------------------------------------------------------------------
    feats, per_tile = read_features(df)
    check = check_caches(per_tile)
    with rasterio.open(EJIDO) as s:
        ej = s.read(1)
    with rasterio.open(CENTRE) as s:
        ce = s.read(1)
    assert ej.shape == ce.shape == (24381, 24297)
    rows, cols_ = df["row"].to_numpy(), df["col"].to_numpy()
    in_e, in_c = ej[rows, cols_] == 1, ce[rows, cols_] == 1
    del ej, ce
    for n, v in feats.items():
        df[n] = v
    df["in_ejido"], df["in_sq8km"] = in_e, in_c
    df.drop(columns=["ndvi_glcmcorr_41m"]).to_csv(out / "d3_samples.csv", index=False, float_format="%.6g")
    floors = {"ndvi": df["ndvi"].to_numpy(float), "chm": df["chm"].to_numpy(float)}
    names = ["ndvi", "chm"] + PS_NAMES + PSC_NAMES + S2_NAMES
    table = []
    for region, m in (("all samples", every), ("ejido", in_e), ("sq8km", in_c)):
        for n in names:
            table.append({"region": region, **row_for(n, df[n].to_numpy(float), y, fold, m, tiles, floors)})
    res = pd.DataFrame(table)
    # ---- add-on check inside the ejido --------------------------------------------------------------------------------------------------------
    data = {"X": {n: df[n].to_numpy(float) for n in names}, "y": y, "fold": fold, "m": in_e}
    todo = PS_NAMES + S2_NAMES
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(data,)) as ex:
        add = {n: (m_, mn) for n, m_, mn in ex.map(addon, todo)}
    res["dauc_ndvi_chm_mean"] = [add[r.feature][0] if r.region == "ejido" and r.feature in add else np.nan for r in res.itertuples()]
    res["dauc_ndvi_chm_min"] = [add[r.feature][1] if r.region == "ejido" and r.feature in add else np.nan for r in res.itertuples()]
    res.to_csv(out / "d3_auroc.csv", index=False)
    # ---- by group, inside the ejido -----------------------------------------------------------------------------------------------------------
    grp = df["group"].to_numpy()
    gr = []
    sgn = {r.feature: r.sign for r in res[res.region == "ejido"].itertuples()}
    for n in names:
        x = df[n].to_numpy(float)
        fin = np.isfinite(x) & in_e
        for code, gname in T.GROUPS.items():
            vals = []
            for k in FOLDS:
                m = fin & (fold != k) & ((y == 0) | (grp == code))
                if (y[m] == 1).sum() >= MIN_N and (y[m] == 0).sum() >= MIN_N:
                    v = roc_auc_score(y[m], x[m])
                    vals.append(v if sgn[n] > 0 else 1 - v)
            if vals:
                gr.append({"feature": n, "group": gname, "auc_mean": round(float(np.mean(vals)), 4)})
    pd.DataFrame(gr).to_csv(out / "d3_by_group.csv", index=False)
    # ---- what the queue reads ------------------------------------------------------------------------------------------------------------------
    ej_rows = res[res.region == "ejido"].set_index("feature")
    ps = ej_rows.loc[PS_NAMES]
    best = ps["auc_mean"].idxmax()
    s2_ = ej_rows.loc[S2_NAMES]
    best_s2 = s2_["auc_mean"].idxmax()
    d3 = {"rule": "the PlanetScope screen reaches 0.60 when the highest auc_mean among the five ps_ndvi_* features inside the ejido is at least 0.60 (written before the first run, scripts/diag_d3.py)",
          "ps_best_feature": best, "ps_best_auc_mean": float(ps.loc[best, "auc_mean"]), "ps_best_auc_min": float(ps.loc[best, "auc_min"]), "ps_reaches_0_60": bool(ps.loc[best, "auc_mean"] >= 0.60),
          "s2_best_feature": best_s2, "s2_best_auc_mean": float(s2_.loc[best_s2, "auc_mean"]), "s2_best_auc_min": float(s2_.loc[best_s2, "auc_min"]),
          "floor_ejido": {"ndvi_auc_mean": float(ej_rows.loc["ndvi", "auc_mean"]), "chm_auc_mean": float(ej_rows.loc["chm", "auc_mean"])},
          "samples": {"all": int(len(df)), "ejido": int(in_e.sum()), "sq8km": int(in_c.sum()), "sq8km_positives": int((y[in_c] == 1).sum()), "sq8km_tiles": int(pd.unique(tiles[in_c]).size)},
          "cache_check": check, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(d3, open(out / "d3.json", "w"), indent=1)
    print(json.dumps(d3, indent=1), flush=True)
    show = ["feature", "auc_mean", "auc_min", "auc_wo_fold1", "auc_wo_fold2", "auc_wo_fold3", "auc_wo_fold4", "sign", "sign_consistent", "nan_share", "tiles", "floor_ndvi_mean", "floor_chm_mean", "dauc_ndvi_chm_mean", "dauc_ndvi_chm_min"]
    for region in ("ejido", "sq8km"):
        print(region)
        print(res[res.region == region][show].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
