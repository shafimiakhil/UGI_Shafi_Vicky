"""E1c (CLAUDE.md 13.8): the three plane-corrected PlanetScope dates into the stem of the UGI U-Net at 1 m, the mechanism of E2. One change against ugi_r1_none. A plugin of itareo.ugi_seg.train and oof (config model.plugin).

  * input: each date's normalised mosaic (derived/planetscope2022/ps<date>_sr_norm_s2_3m.tif, 3 m, SR x 10000, nodata 0), 8 bands, resampled once to the 512 px window of GRID_1M by bilinear interpolation, with the plane per date of E1b (runs_itareo/ugi/b4_ps_planes.json) removed inside that one resampling. Plus one availability channel per date: 27 channels at 1 m. A 1 m cell holds a value only where all four 3 m neighbours are valid in all 8 bands (the mosaics' nodata is 0 in any band), so no value is drawn from a partly empty kernel;
  * model: conv1 of the encoder gains the 27 channels, their weights initialised at zero (tex_branch.UNetTex), so at step 0 the model is the baseline whatever PlanetScope holds. They enter no other layer;
  * normalisation: mean and SD per band over the valid 1 m cells of the internal training tiles of the fold; invalid cells are 0 after it, and the availability channels say so (ps_branch.ps_tensor);
  * augmentation: the geometric draws of UGIDataset (same seed, same order). The photometric jitter stays on Neo only;
  * cache: ps_native.npy (tiles of the Neo cache, 24, 176, 176), uint16, the 3 m windows that hold each tile, read as they are, and origins.npy (tiles, 3, 2), the window corner per date. The 1 m bilinear step runs in the data loader: 24 bands at 1 m over 2,116 tiles would take 26.6 GB as float16 and the disk must stay above 30 GB, so the cache holds the source and the one resampling stays one. Inference reads the same windows from the mosaics.
  * workers: 8 in the config, because the resampling runs in the workers. The augmentation depends on the seed only, not on the number of workers (itareo.unet_ugi.data.SeededSampler), so this changes the speed and nothing else.

  python -m itareo.ugi_seg.ps1m_branch --build-cache --cfg itareo/configs/ugi_e1c_ps1m.yaml
  python -m itareo.ugi_seg.ps1m_branch --selftest --cfg itareo/configs/ugi_e1c_ps1m.yaml --out <a new folder>       (exit 1 if it fails)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from itareo.safe_write import refuse_existing
from itareo.unet_ugi.data import read_csv

from .data import IGNORE, UGIDataset
from .model import build as build_base
from .ps_branch import N_BANDS, N_DATES, N_PS, PS_CHANNELS, ps_tensor
from .ps_stage import DATES, GRID1M_ORIGIN, PS, load_planes, plane_transform, plane_value
from .tex_branch import UNetTex

TILE = 512
NATIVE = int(TILE / 3) + 6                  # the source window of ps_stage.stage_tile_plane
_SHAPE = {}


def window_corner(st, p, row0, col0):
    """The corner (col, row) in the 3 m source of the window that holds the tile; follows the displacement at the tile's centre, as ps_stage.stage_tile_plane."""
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    de, dn = plane_value(p, x0 + TILE / 2, y0 - TILE / 2)
    return int((x0 + de - st.c) // 3) - 2, int((st.f - y0 - dn) // 3) - 2


def sample_grid(st, p, corner, row0, col0, n=TILE):
    """(n, n, 2) grid for torch grid_sample, align_corners=True: the positions in the source window of the centres of the 1 m cells, with the plane removed. plane_transform maps source (col, row) to the corrected position; its inverse maps a 1 m cell centre back. The map is affine, so the grid is its value at the first cell plus an array that depends on the date only."""
    c, r = corner
    inv = ~plane_transform(st * Affine.translation(c, r), p)
    x00, y00 = GRID1M_ORIGIN[0] + col0 + 0.5, GRID1M_ORIGIN[1] - row0 - 0.5
    u00 = inv.a * x00 + inv.b * y00 + inv.c - 0.5         # index of the source pixel centres
    v00 = inv.d * x00 + inv.e * y00 + inv.f - 0.5
    key = (inv.a, inv.b, inv.d, inv.e, n)
    if key not in _SHAPE:
        jj, ii = np.meshgrid(np.arange(n, dtype=np.float64), -np.arange(n, dtype=np.float64))      # x grows with the column, y falls with the row
        _SHAPE[key] = np.stack([inv.a * jj + inv.b * ii, inv.d * jj + inv.e * ii], -1) * (2 / (NATIVE - 1))
    return (_SHAPE[key] + np.array([2 * u00 / (NATIVE - 1) - 1, 2 * v00 / (NATIVE - 1) - 1])).astype(np.float32)


def bilinear(native, grid):
    """native (8, N, N) uint16, 0 = nodata; grid from sample_grid -> (8, n, n) float32, SR x 10000, NaN unless all four neighbours are valid in all bands (and inside the window)."""
    a = torch.from_numpy(native.astype(np.float32))
    ok = (a > 0).all(0, keepdim=True)
    if not ok.any():
        return np.full((native.shape[0],) + grid.shape[:2], np.nan, np.float32)
    g = torch.from_numpy(grid)[None]
    o = F.grid_sample(a[None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    if not ok.all() or grid.min() < -1 or grid.max() > 1:
        m = F.grid_sample(ok.float()[None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        o[:, m < 1 - 1e-4] = float("nan")
    return o.numpy()


def planes_and_transform():
    pl = load_planes()
    with rasterio.open(PS.format(DATES[0])) as s:
        st = s.transform
    for d in DATES[1:]:
        with rasterio.open(PS.format(d)) as s:
            assert s.transform == st and s.count == N_BANDS and s.nodata == 0
    return [pl[d] for d in DATES], st


def raw_from_native(native, corners, planes, st, row0, col0):
    """native (24, N, N), corners (3, 2) -> (24, 512, 512) float32, NaN where invalid."""
    return np.concatenate([bilinear(native[k * N_BANDS:(k + 1) * N_BANDS], sample_grid(st, planes[k], corners[k], row0, col0)) for k in range(N_DATES)])


def build_cache(cfg):
    out = Path(cfg["paths"]["ps1m_cache"])
    refuse_existing(out)
    out.mkdir(parents=True)
    t0 = time.time()
    tiles = read_csv(Path(cfg["paths"]["cache"]) / "tiles.csv")
    planes, st = planes_and_transform()
    arr = np.lib.format.open_memmap(out / "ps_native.npy", "w+", np.uint16, (len(tiles), N_PS, NATIVE, NATIVE))
    org = np.lib.format.open_memmap(out / "origins.npy", "w+", np.int32, (len(tiles), N_DATES, 2))
    for k, d in enumerate(DATES):
        with rasterio.open(PS.format(d)) as src:
            for i, t in enumerate(tiles):
                c, r = window_corner(st, planes[k], t["row0"], t["col0"])
                org[i, k] = (c, r)
                arr[i, k * N_BANDS:(k + 1) * N_BANDS] = src.read(window=Window(c, r, NATIVE, NATIVE), boundless=True, fill_value=0)
        print(d, round(time.time() - t0), "s", flush=True)
    arr.flush()
    org.flush()
    json.dump({"n": len(tiles), "dates": DATES, "bands": N_BANDS, "native_px": NATIVE, "tiles": str(Path(cfg["paths"]["cache"]) / "tiles.csv"), "planes": "runs_itareo/ugi/b4_ps_planes.json", "source": PS, "seconds": round(time.time() - t0, 1)}, open(out / "cache.json", "w"), indent=1)


class Native:
    """The cached source of the tiles of the Neo cache, opened lazily in each worker."""

    def __init__(self, cache_dir, neo_cache):
        self.dir, self.neo = Path(cache_dir), Path(neo_cache)
        self.arr, self.org, self.tiles, self.planes, self.st = None, None, None, None, None

    def raw(self, i):
        if self.arr is None:
            self.arr, self.org = np.load(self.dir / "ps_native.npy", mmap_mode="r"), np.load(self.dir / "origins.npy", mmap_mode="r")
            self.tiles = read_csv(self.neo / "tiles.csv")
            self.planes, self.st = planes_and_transform()
        t = self.tiles[i]
        return raw_from_native(np.asarray(self.arr[i]), np.asarray(self.org[i]), self.planes, self.st, t["row0"], t["col0"])


class UGIPS1mDataset(UGIDataset):
    """UGIDataset, whose items are unchanged, with the 1 m PlanetScope tensor beside the Neo tensor: ([x, ps], y_bin, y_grp)."""

    def __init__(self, ps1m_cache, ps_mean, ps_sd, cache_dir, *args, **kw):
        super().__init__(cache_dir, *args, **kw)
        self.native, self.ps_mean, self.ps_sd = Native(ps1m_cache, cache_dir), ps_mean, ps_sd

    def __getitem__(self, key):
        x, yb, yg = super().__getitem__(key)
        j, seed = key if isinstance(key, tuple) else (key, None)
        p = ps_tensor(self.native.raw(self.idx[j]), self.ps_mean, self.ps_sd)
        if self.augment:
            rng = np.random.default_rng(seed)                     # the draws of UGIDataset.__getitem__, in its order
            rng.uniform(*self.photo["global"])
            rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"], size=(x.shape[0], 1, 1))
            p = np.rot90(p, int(rng.integers(4)), (1, 2))
            if rng.random() < 0.5:
                p = p[:, :, ::-1]
        return [x, torch.from_numpy(np.ascontiguousarray(p))], yb, yg


class WindowPS1m:
    """The tensor of a sliding window whose corner is (row0, col0) of GRID_1M, read from the mosaics; the same functions as the data loader."""

    def __init__(self, ps_mean, ps_sd):
        self.mean, self.sd = ps_mean, ps_sd
        self.planes, self.st = planes_and_transform()
        self.src = [rasterio.open(PS.format(d)) for d in DATES]

    def native(self, row0, col0):
        corners = [window_corner(self.st, self.planes[k], row0, col0) for k in range(N_DATES)]
        a = np.concatenate([s.read(window=Window(c, r, NATIVE, NATIVE), boundless=True, fill_value=0) for s, (c, r) in zip(self.src, corners)])
        return a, corners

    def raw(self, row0, col0):
        a, corners = self.native(row0, col0)
        return raw_from_native(a, corners, self.planes, self.st, row0, col0)

    def __call__(self, row0, col0):
        return torch.from_numpy(ps_tensor(self.raw(row0, col0), self.mean, self.sd))

    def close(self):
        for s in self.src:
            s.close()


# ---- the plugin interface of itareo.ugi_seg.train and oof ---------------------------------------------------------------------------------------------

def build(cfg):
    """The baseline's model first, drawn as the baseline draws it; conv1 widened after it."""
    return UNetTex(build_base(cfg["model"]["init"]), channels=PS_CHANNELS)


def stats(cfg, fold, tr_i):
    """Mean and SD per band over the valid 1 m cells of the internal training tiles of the fold; stored in split.json and best.pt."""
    nat = Native(cfg["paths"]["ps1m_cache"], cfg["paths"]["cache"])
    n, s1, s2 = np.zeros(N_PS), np.zeros(N_PS), np.zeros(N_PS)
    for i in tr_i:
        v = nat.raw(i).astype(np.float64)
        ok = np.isfinite(v)
        n += ok.sum(axis=(1, 2))
        s1 += np.where(ok, v, 0).sum(axis=(1, 2))
        s2 += np.where(ok, v * v, 0).sum(axis=(1, 2))
    if (n == 0).any():
        raise ValueError("a PlanetScope band has no valid cell in these tiles")
    mean = s1 / n
    return {"ps1m_mean": mean.tolist(), "ps1m_sd": np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12)).tolist()}


def dataset(cfg, fold, st, cache, *args, **kw):
    return UGIPS1mDataset(cfg["paths"]["ps1m_cache"], st["ps1m_mean"], st["ps1m_sd"], cache, *args, **kw)


def window(cfg, fold, ck, region):
    return WindowPS1m(ck["ps1m_mean"], ck["ps1m_sd"])


# ---- self-test -----------------------------------------------------------------------------------------------------------------------------------------

def _transforms(a):
    """The eight results of the augmentation's geometry on (..., H, W)."""
    out = []
    for k in range(4):
        r = np.rot90(a, k, (-2, -1))
        out += [r, r[..., ::-1]]
    return out


def selftest(cfg_path, out):
    """Before E1c trains. Geometry: a field linear in the corrected coordinates comes back exactly at the 1 m cell centres, which tests the inversion of the plane and the half-pixel convention without GDAL. The sign itself comes from ps_stage.plane_transform and is not tested here; the review of 13.8 measured it directly against Neo v3 on 30 tiles (red band, median displacement E, N in m, 24 Apr: 2.33, 2.91 without the plane; 0.57, 0.08 with it; 4.91, 5.25 with the opposite sign). The resampling against rasterio's bilinear warp on real tiles. The 1 m data against E1b's 4 m stage on the same tiles (averaged to 4 m). The tensors of the dataset and of the window are equal. The Neo items equal the baseline's. The geometry of the augmentation agrees on both tensors. Step 0 equals the baseline for the same weights, whatever the input holds. The new weights take a gradient. Writes <out>/e1c_selftest.json. Returns True if all passed."""
    import yaml

    from itareo.unet_ugi.data import band_stats
    from itareo.unet_ugi.train import seed_all

    from .ps_stage import stage_tile_plane
    from .train import split
    out = Path(out)
    refuse_existing(out / "e1c_selftest.json")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    res = {"config": str(cfg_path)}
    cache = Path(cfg["paths"]["cache"])
    cinfo = json.load(open(cache / "cache.json"))
    tiles = read_csv(cache / "tiles.csv")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(tiles)}
    planes, st = planes_and_transform()
    # ---- 1. a linear field comes back exactly, with a plane exaggerated tenfold so that a sign error cannot hide ----------------------------------------------
    rows = []
    for k, d in enumerate(DATES):
        for scale in (1.0, 10.0):
            p = {"e": [v * scale for v in planes[k]["e"]], "n": [v * scale for v in planes[k]["n"]]}
            row0, col0 = 10807, 5120
            c, r = window_corner(st, p, row0, col0)
            T = plane_transform(st * Affine.translation(c, r), p)
            cc, rr = np.meshgrid(np.arange(NATIVE) + 0.5, np.arange(NATIVE) + 0.5)
            xc, yc = T.a * cc + T.b * rr + T.c, T.d * cc + T.e * rr + T.f                 # corrected position of each source pixel centre
            x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
            field = 3000 + 3 * (xc - x0) - 2 * (y0 - yc)                                   # positive, inside uint16 range as float below
            g = sample_grid(st, p, (c, r), row0, col0)
            o = F.grid_sample(torch.from_numpy(field.astype(np.float32))[None, None], torch.from_numpy(g)[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0].numpy()
            xx, yy = np.meshgrid(x0 + np.arange(TILE) + 0.5, y0 - np.arange(TILE) - 0.5)
            want = 3000 + 3 * (xx - x0) - 2 * (y0 - yy)
            rows.append({"date": d, "plane_scale": scale, "max_abs_error_in_units_of_the_field": float(np.abs(o - want).max()), "max_error_m": float(np.abs(o - want).max() / 3)})
    res["linear_field"] = rows
    res["linear_field_passed"] = all(r["max_error_m"] < 0.02 for r in rows)
    # ---- 2. against rasterio's bilinear warp; against E1b's 4 m stage -------------------------------------------------------------------------------------------
    b4 = read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv")
    full = [t for t in b4 if not t["straddle"] and t["avail_ps_all3"] >= 0.99 and (t["row0"], t["col0"]) in index]
    part = [t for t in b4 if not t["straddle"] and 0.3 < t["avail_ps_all3"] < 0.9 and (t["row0"], t["col0"]) in index]
    rng = np.random.default_rng(0)
    sel = [full[k] for k in rng.choice(len(full), 4, replace=False)] + [part[k] for k in rng.choice(len(part), 2, replace=False)]
    win = WindowPS1m(np.zeros(N_PS), np.ones(N_PS))
    rows = []
    t_w = time.time()
    for t in sel:
        r0, c0 = t["row0"], t["col0"]
        a, corners = win.native(r0, c0)
        raw = raw_from_native(a, corners, planes, st, r0, c0)
        for k, d in enumerate(DATES):
            src_a = a[k * N_BANDS:(k + 1) * N_BANDS].astype(np.float32)
            src_a[:, (src_a == 0).any(0)] = np.nan
            q = np.full((N_BANDS, TILE, TILE), np.nan, np.float32)
            reproject(src_a, q, src_transform=plane_transform(st * Affine.translation(*corners[k]), planes[k]), src_crs=win.src[k].crs, dst_transform=Affine(1, 0, GRID1M_ORIGIN[0] + c0, 0, -1, GRID1M_ORIGIN[1] - r0), dst_crs=win.src[k].crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            mine = raw[k * N_BANDS:(k + 1) * N_BANDS]
            both = np.isfinite(mine) & np.isfinite(q)
            st4 = stage_tile_plane(win.src[k], r0, c0, planes[k])
            m4 = mine.reshape(N_BANDS, 128, 4, 128, 4)
            with np.errstate(invalid="ignore"):
                av = np.nanmean(m4, axis=(2, 4)) if np.isfinite(m4).any() else m4[:, :, 0, :, 0]
            ok4 = np.isfinite(av) & np.isfinite(st4)
            rows.append({"date": d, "row0": r0, "col0": c0, "cells_mine": int(np.isfinite(mine[0]).sum()), "cells_gdal": int(np.isfinite(q[0]).sum()), "cells_mine_not_gdal": int((np.isfinite(mine[0]) & ~np.isfinite(q[0])).sum()),
                         "max_rel_diff_to_gdal": float((np.abs(mine - q)[both] / np.maximum(np.abs(q[both]), 1)).max()) if both.any() else None, "mean_abs_diff_to_gdal": float(np.abs(mine - q)[both].mean()) if both.any() else None,
                         "corr_with_4m_stage": float(np.corrcoef(av[ok4], st4[ok4])[0, 1]) if ok4.sum() > 100 else None, "mean_rel_diff_to_4m_stage": float((np.abs(av - st4)[ok4] / np.maximum(st4[ok4], 1)).mean()) if ok4.sum() > 100 else None})
    res["seconds_per_window_raw"] = round((time.time() - t_w) / (len(sel) * 2), 4)
    res["against_gdal_and_stage"] = rows
    res["gdal_passed"] = all(r["cells_mine_not_gdal"] == 0 and (r["max_rel_diff_to_gdal"] is None or r["max_rel_diff_to_gdal"] < 1e-3) for r in rows)
    res["stage_passed"] = all(r["corr_with_4m_stage"] is None or r["corr_with_4m_stage"] > 0.995 for r in rows)
    # ---- 3. dataset and window; normalisation on a training subset ---------------------------------------------------------------------------------------------
    _, tr, val, _, _ = split(cfg, 1)
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    mean, sd = band_stats(cache, tr_i[:200], cinfo["nodata"])
    t0 = time.time()
    full_keys = {(t["row0"], t["col0"]) for t in full}
    tr_full = [index[(t["row0"], t["col0"])] for t in tr if (t["row0"], t["col0"]) in full_keys]
    st_small = stats(cfg, 1, tr_full[:40])                  # tiles that hold all three dates: a subset of the first tiles may hold none
    res["seconds_per_tile_stats"] = round((time.time() - t0) / 40, 4)
    pm, ps_ = st_small["ps1m_mean"], st_small["ps1m_sd"]
    res["ps_mean_range"] = [round(min(pm), 1), round(max(pm), 1)]
    res["ps_sd_range"] = [round(min(ps_), 1), round(max(ps_), 1)]
    ds_plain = dataset(cfg, 1, st_small, cache, [index[(t["row0"], t["col0"])] for t in sel], mean, sd, cinfo["nodata"], False, edge_band="none")
    w2 = WindowPS1m(pm, ps_)
    rows = []
    for j, t in enumerate(sel):
        [_, p], _, _ = ds_plain[j]
        rows.append({"row0": t["row0"], "col0": t["col0"], "window_equals_dataset": bool(torch.equal(w2(t["row0"], t["col0"]), p)), "availability_per_date": [round(float(p[N_PS + k].mean()), 4) for k in range(N_DATES)], "max_abs_z": round(float(p[:N_PS].abs().max()), 2)})
    res["tensors"] = rows
    res["window_equals_dataset"] = all(r["window_equals_dataset"] for r in rows)
    corner = w2(-256, -256)
    res["outside_grid_is_zero"] = bool((corner[:, :200, :] == 0).all() and (corner[:, :, :200] == 0).all())
    # ---- 4. the Neo items equal the baseline's, with augmentation; the geometry agrees ---------------------------------------------------------------------------
    pho = cfg["train"].get("photometric")
    d_base = UGIDataset(cache, tr_i[:12], mean, sd, cinfo["nodata"], True, None, pho, "none")
    d_ps = dataset(cfg, 1, st_small, cache, tr_i[:12], mean, sd, cinfo["nodata"], True, None, pho, "none")
    same = True
    for j in range(12):
        key = (j, 1000 + j)
        x0, y0, g0 = d_base[key]
        [x1, _], y1, g1 = d_ps[key]
        same &= bool(torch.equal(x0, x1) and torch.equal(y0, y1) and torch.equal(g0, g1))
    d_speed = dataset(cfg, 1, st_small, cache, tr_full[:30], mean, sd, cinfo["nodata"], True, None, pho, "none")
    t0 = time.time()
    for j in range(30):
        d_speed[(j, 1500 + j)]
    res["seconds_per_item_with_augmentation_all_dates"] = round((time.time() - t0) / 30, 4)
    res["neo_items_equal_baseline"] = same
    full_i = tr_full[:1]
    d_a = dataset(cfg, 1, st_small, cache, full_i, mean, sd, cinfo["nodata"], True, None, {"global": [1.0, 1.0], "per_band": 0.0}, "none")
    d_p = dataset(cfg, 1, st_small, cache, full_i, mean, sd, cinfo["nodata"], False, edge_band="none")
    [xp, pp], _, _ = d_p[0]
    xs, ps_all = _transforms(xp.numpy()), _transforms(pp.numpy())
    agree, seen = True, set()
    for s in range(48):
        [x, p], _, _ = d_a[(0, s)]
        m = [k for k in range(8) if np.allclose(xs[k], x.numpy())]
        agree &= len(m) >= 1 and any(np.array_equal(ps_all[k], p.numpy()) for k in m)
        seen.add(m[0] if m else -1)
    res["geometry_agrees"] = bool(agree)
    res["geometry_distinct_transforms_seen"] = len(seen)
    # ---- 5. step 0 ---------------------------------------------------------------------------------------------------------------------------------------------
    batch = [d_ps[(j, 2000 + j)] for j in range(2)]
    x = torch.stack([b[0][0] for b in batch])
    p = torch.stack([b[0][1] for b in batch])
    res["batch_availability_per_date"] = [round(float(p[:, N_PS + k].mean()), 4) for k in range(N_DATES)]
    res["step0"] = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        seed_all(0)
        base = build_base(cfg["model"]["init"]).to(device)
        seed_all(0)
        m = build(cfg).to(device)
        sd_b, sd_m = base.state_dict(), m.base.state_dict()
        c = m.base.enc.conv1.weight
        r = {"same_weights_as_baseline": bool(all(torch.equal(sd_b[k], sd_m[k]) for k in sd_b if k != "enc.conv1.weight") and torch.equal(sd_b["enc.conv1.weight"], c[:, :6])), "new_conv1_weights_zero": bool((c[:, 6:] == 0).all()), "conv1_shape_ok": list(c.shape[:2]) == [64, 6 + PS_CHANNELS]}
        xd, pd = x.to(device), p.to(device)
        with torch.no_grad():
            for mode in ("eval", "train"):
                base.train(mode == "train")
                m.train(mode == "train")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    o0, o1 = base(xd), m(xd, pd)
                    o2 = m(xd, torch.randn_like(pd) * 50)
                r[f"outputs_equal_{mode}"] = bool(torch.equal(o0.float(), o1.float()))
                r[f"outputs_equal_{mode}_whatever_planetscope_holds"] = bool(torch.equal(o0.float(), o2.float()))
        res["step0"][device] = r
    # ---- 6. it can learn ---------------------------------------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(0)
    m = build(cfg).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    d_g = dataset(cfg, 1, st_small, cache, full_i * 2, mean, sd, cinfo["nodata"], True, None, pho, "none")
    bt = [d_g[(j, 3000 + j)] for j in range(2)]
    xg, pg = torch.stack([b[0][0] for b in bt]).to(device), torch.stack([b[0][1] for b in bt]).to(device)
    yb = torch.stack([b[1] for b in bt]).to(device)
    valid = yb != IGNORE
    opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(m(xg, pg)[:, 0][valid], yb[valid].float()).backward()
    g = m.base.enc.conv1.weight.grad
    res["learning"] = {"grad_new_channels": [round(float(v), 4) for v in g[:, 6:].abs().sum(dim=(0, 2, 3))]}
    opt.step()
    res["learning"]["new_weights_moved"] = bool((m.base.enc.conv1.weight[:, 6:] != 0).any())
    res["learning_passed"] = bool((g[:, 6:6 + N_PS].abs().sum(dim=(0, 2, 3)) > 0).all() and res["learning"]["new_weights_moved"])
    res["passed"] = bool(res["linear_field_passed"] and res["gdal_passed"] and res["stage_passed"] and res["window_equals_dataset"] and res["outside_grid_is_zero"] and same and agree and len(seen) == 8 and res["learning_passed"] and all(all(v for v in r.values()) for r in res["step0"].values()))
    json.dump(res, open(out / "e1c_selftest.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])
    return res["passed"]


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.build_cache:
        build_cache(yaml.safe_load(open(a.cfg, encoding="utf-8")))
    if a.selftest:
        raise SystemExit(0 if selftest(a.cfg, a.out) else 1)
