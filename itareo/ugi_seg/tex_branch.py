"""E2 (CLAUDE.md 13.6): optical texture in the stem of the UGI U-Net at 1 m. One change against ugi_r1_none.

  * input: six layers on GRID_1M. The PAN lawn layers var3, var9, hom3, hom9 (derived/features/neo2022_v3_pan_lawn_grid1m.tif; vegetated pixels only, strip 1_3 recomputed on v3, scripts/e2_pan_lawn_v3.py) and
    ndvi_orientcons_41m, ndvi_glcmcorr_41m from the Neo v3 NDVI (derived/features/neo2022_v3_ndvi_texture_grid1m.tif, scripts/e2_ndvi_texture.py; the correlation of the held-out fold, whose quantisation range comes from
    that fold's internal training tiles), plus two mask channels: where the PAN layers have a value, and where both NDVI layers have one. 8 new channels;
  * model: conv1 of the encoder gains the 8 channels, their weights initialised at zero, so at step 0 the model is the baseline whatever the texture holds. They enter no other layer;
  * values: var3 and var9 are heavy-tailed (variance; over the 2,116 cached tiles medians 0.49 and 1.6, 99th percentiles 10 and 50, maxima 1,650 and 2,660), so log1p is applied before normalising, decided before any E2 result (the constant LOG1P; the config key model.texture.log1p records it, and build_e2 stops if they differ. Without the key, `texture: {}` reads as false and train.py builds the baseline). Mean and SD per layer from the internal training tiles of the
    fold, over valid pixels; invalid pixels are 0 after normalising, and the masks say so;
  * augmentation: the geometric draws of UGIDataset (same seed, same order). The layers are rotated and flipped with the tile, not recomputed. var and orientcons give the same values on a transformed image. hom and the 4-direction correlation have no direction term, but each pair is stored at its first pixel, so their windows sit half a pair off-centre and recomputing them on a transformed image gives somewhat different values (at about 1 m, hom3 21-30 % of its SD, hom9 about 10 %; the correlation most at the edge of flat windows; review of 4 Nov). The photometric jitter stays on Neo only;
  * cache: tex.npy (tiles of the Neo cache, 9, 512, 512) float16, NaN where invalid, read from the two rasters: var3, var9, hom3, hom9, orientcons, glcmcorr of folds 1-4. Inference reads the same rasters over the region (WindowTex).

  python -m itareo.ugi_seg.tex_branch --build-cache --cfg itareo/configs/ugi_e2_tex.yaml
  python -m itareo.ugi_seg.tex_branch --selftest --cfg itareo/configs/ugi_e2_tex.yaml --out <a new folder>       (exit 1 if it fails)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.windows import Window

from itareo.safe_write import refuse_existing
from itareo.unet_ugi.data import read_csv

from .data import IGNORE, UGIDataset
from .model import build

N_PAN, N_LAYERS, N_CACHE = 4, 6, 9
TEX_CHANNELS = N_LAYERS + 2
LAYERS = ["var3", "var9", "hom3", "hom9", "orientcons", "glcmcorr"]
LOG1P = 2                                  # the first two layers


def fold_bands(fold):
    """Cache bands (also the order pan 1-4, ndvi 1 and 1 + fold of the rasters) of the six layers for a held-out fold."""
    return [0, 1, 2, 3, 4, 4 + fold]


def transform(raw):
    """(6, H, W) float32, NaN where invalid: log1p on var3 and var9."""
    t = np.array(raw, np.float32)
    t[:LOG1P] = np.log1p(t[:LOG1P])
    return t


def tex_tensor(raw, mean, sd):
    """(6, H, W), NaN where invalid -> (8, H, W): normalised, 0 where invalid, then the two masks."""
    t = transform(raw)
    ok = np.isfinite(t)
    m, s = np.asarray(mean, np.float32)[:, None, None], np.asarray(sd, np.float32)[:, None, None]
    z = np.where(ok, (np.nan_to_num(t) - m) / s, 0.0)
    return np.concatenate([z, ok[:N_PAN].all(0, keepdims=True), ok[N_PAN:].all(0, keepdims=True)]).astype(np.float32)


def build_tex_cache(cfg):
    out = Path(cfg["paths"]["tex_cache"])
    refuse_existing(out)
    out.mkdir(parents=True)
    t0 = time.time()
    tiles = read_csv(Path(cfg["paths"]["cache"]) / "tiles.csv")
    arr = np.lib.format.open_memmap(out / "tex.npy", "w+", np.float16, (len(tiles), N_CACHE, 512, 512))
    with rasterio.open(cfg["paths"]["pan_lawn"]) as p, rasterio.open(cfg["paths"]["ndvi_texture"]) as n:
        for i, t in enumerate(tiles):
            w = Window(t["col0"], t["row0"], 512, 512)
            arr[i, :N_PAN] = p.read(window=w)
            arr[i, N_PAN:] = n.read(window=w)
    arr.flush()
    json.dump({"n": len(tiles), "bands": ["var3", "var9", "hom3", "hom9", "orientcons", "glcmcorr_fold1", "glcmcorr_fold2", "glcmcorr_fold3", "glcmcorr_fold4"], "tiles": str(Path(cfg["paths"]["cache"]) / "tiles.csv"),
               "pan_lawn": cfg["paths"]["pan_lawn"], "ndvi_texture": cfg["paths"]["ndvi_texture"], "seconds": round(time.time() - t0, 1)}, open(out / "cache.json", "w"), indent=1)
    print("texture cache", len(tiles), "tiles", round(time.time() - t0), "s", flush=True)


def tex_stats(cache_dir, fold, idx):
    """Mean and SD of the six layers (after log1p) of the held-out fold over the valid pixels of tiles idx."""
    a = np.load(Path(cache_dir) / "tex.npy", mmap_mode="r")
    b = fold_bands(fold)
    n, s1, s2 = np.zeros(N_LAYERS), np.zeros(N_LAYERS), np.zeros(N_LAYERS)
    for i in idx:
        v = transform(a[i][b]).astype(np.float64)
        ok = np.isfinite(v)
        n += ok.sum(axis=(1, 2))
        s1 += np.where(ok, v, 0).sum(axis=(1, 2))
        s2 += np.where(ok, v * v, 0).sum(axis=(1, 2))
    mean = s1 / n
    return mean, np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))


class UGITexDataset(UGIDataset):
    """UGIDataset, whose items are unchanged, with the texture tensor beside the Neo tensor: ([x, tex], y_bin, y_grp)."""

    def __init__(self, tex_cache, fold, tex_mean, tex_sd, *args, **kw):
        super().__init__(*args, **kw)
        self.tex_path, self.bands, self.tex_mean, self.tex_sd, self.tex = str(Path(tex_cache) / "tex.npy"), fold_bands(fold), tex_mean, tex_sd, None

    def __getitem__(self, key):
        x, yb, yg = super().__getitem__(key)
        if self.tex is None:
            self.tex = np.load(self.tex_path, mmap_mode="r")
        j, seed = key if isinstance(key, tuple) else (key, None)
        p = tex_tensor(self.tex[self.idx[j]][self.bands], self.tex_mean, self.tex_sd)
        if self.augment:
            rng = np.random.default_rng(seed)                     # the draws of UGIDataset.__getitem__, in its order
            rng.uniform(*self.photo["global"])
            rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"], size=(x.shape[0], 1, 1))
            p = np.rot90(p, int(rng.integers(4)), (1, 2))
            if rng.random() < 0.5:
                p = p[:, :, ::-1]
        return [x, torch.from_numpy(np.ascontiguousarray(p))], yb, yg


class WindowTex:
    """The texture tensor of a sliding window of n px whose corner is (row0, col0) of GRID_1M. The region, with the reach of the windows of predict_region, is read once; NaN outside the grid."""

    def __init__(self, cfg, fold, mean, sd, region, n=512):
        self.mean, self.sd, self.n = mean, sd, n
        r_lo, c_lo, R, Cn = region
        h = n // 2
        self.r0, self.c0 = r_lo - h, c_lo - h
        rows, cols = R + h + n, Cn + h + n
        self.arr = np.full((N_LAYERS, rows, cols), np.nan, np.float32)
        with rasterio.open(cfg["paths"]["pan_lawn"]) as p, rasterio.open(cfg["paths"]["ndvi_texture"]) as t:
            a0, b0 = max(self.r0, 0), max(self.c0, 0)
            a1, b1 = min(self.r0 + rows, p.height), min(self.c0 + cols, p.width)
            w = Window(b0, a0, b1 - b0, a1 - a0)
            sl = (slice(a0 - self.r0, a1 - self.r0), slice(b0 - self.c0, b1 - self.c0))
            self.arr[(slice(0, N_PAN),) + sl] = p.read(window=w)
            self.arr[(N_PAN,) + sl] = t.read(1, window=w)
            self.arr[(N_PAN + 1,) + sl] = t.read(1 + fold, window=w)

    def raw(self, row0, col0):
        a, b = row0 - self.r0, col0 - self.c0
        return self.arr[:, a:a + self.n, b:b + self.n]

    def __call__(self, row0, col0):
        return torch.from_numpy(tex_tensor(self.raw(row0, col0), self.mean, self.sd))


class UNetTex(nn.Module):
    def __init__(self, base, channels=TEX_CHANNELS):
        super().__init__()
        self.base = base
        old = base.enc.conv1
        new = nn.Conv2d(old.in_channels + channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_channels] = old.weight
        base.enc.conv1 = new

    def forward(self, x, tex):
        return self.base(torch.cat([x, tex], 1))


def build_e2(cfg):
    """The baseline's model first, drawn as the baseline draws it; conv1 widened after it."""
    if list(cfg["model"]["texture"].get("log1p") or []) != LAYERS[:LOG1P]:
        raise ValueError(f"model.texture.log1p must be {LAYERS[:LOG1P]}: the code applies LOG1P, it does not read the key")
    return UNetTex(build(cfg["model"]["init"]))


def selftest(cfg_path, out):
    """Before E2 trains: the feature files passed their own checks; the tensors of the cache, of the dataset and of the window equal what the rasters hold, with the band of the right fold; the Neo items equal the baseline's; the geometry agrees on the two tensors;
    the step-0 outputs equal the baseline's for the same weights; the new weights take a gradient. Writes <out>/e2_selftest.json. Returns True if all passed."""
    import yaml

    from itareo.unet_ugi.data import band_stats
    from itareo.unet_ugi.train import seed_all

    from .train import split
    out = Path(out)
    refuse_existing(out / "e2_selftest.json", out / "geometry_cache")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    res = {"config": str(cfg_path)}
    cache, tc = Path(cfg["paths"]["cache"]), Path(cfg["paths"]["tex_cache"])
    cinfo = json.load(open(cache / "cache.json"))
    tiles = read_csv(cache / "tiles.csv")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(tiles)}
    # ---- 1. the feature files ------------------------------------------------------------------------------------------------------------------------------------
    pan_log = json.load(open("derived/features/neo2022_v3_pan_lawn_log.json"))
    tex_log = json.load(open("derived/features/neo2022_v3_ndvi_texture_log.json"))
    pan_st = json.load(open("../docs/evidence_phase2/ugi_seg/selftest_e2/pan_lawn_v3_selftest_run2.json"))
    res["pan_selftest_passed"] = bool(pan_st["passed"])
    res["ndvi_texture_check_passed"] = bool(tex_log["check_passed"])
    res["pan_values_outside_vegetated_pixels"] = pan_log["values_outside_vegetated_pixels"]
    # ---- 2. tensors against the rasters, per fold ------------------------------------------------------------------------------------------------------------------
    rows = []
    a = np.load(tc / "tex.npy", mmap_mode="r")
    for fold in (1, 2, 3, 4):
        _, tr, val, _, _ = split(cfg, fold)
        tr_i = [index[(t["row0"], t["col0"])] for t in tr]
        tm, ts = tex_stats(tc, fold, tr_i[:200])
        res[f"fold{fold}_layer_mean_on_200_tiles"] = [round(float(v), 4) for v in tm]
        res[f"fold{fold}_layer_sd_on_200_tiles"] = [round(float(v), 4) for v in ts]
        held = [t for t in tiles if t["fold"] == fold and not t["straddle"]]
        rng = np.random.default_rng(fold)
        sel = [held[k] for k in rng.choice(len(held), 3, replace=False)] + [tr[k] for k in rng.choice(len(tr), 3, replace=False)]
        mean, sd = band_stats(cache, tr_i[:50], cinfo["nodata"])
        for t in sel:
            r0, c0, i = t["row0"], t["col0"], index[(t["row0"], t["col0"])]
            w = Window(c0, r0, 512, 512)
            with rasterio.open(cfg["paths"]["pan_lawn"]) as p, rasterio.open(cfg["paths"]["ndvi_texture"]) as n:
                raw = np.concatenate([p.read(window=w), n.read(1, window=w)[None], n.read(1 + fold, window=w)[None]])
            ds = UGITexDataset(tc, fold, tm, ts, cache, [i], mean, sd, cinfo["nodata"], False, edge_band="none")
            [_, q], _, _ = ds[0]
            win = WindowTex(cfg, fold, tm, ts, (r0, c0, 512, 512))
            rows.append({"fold": fold, "row0": r0, "col0": c0, "cache_equals_rasters": bool(np.array_equal(np.asarray(a[i][fold_bands(fold)], np.float32), raw, equal_nan=True)),
                         "dataset_equals_tensor_of_rasters": bool(np.array_equal(q.numpy(), tex_tensor(raw, tm, ts))), "window_equals_dataset": bool(torch.equal(win(r0, c0), q)),
                         "pan_valid_share": round(float(q[6].mean()), 4), "ndvi_valid_share": round(float(q[7].mean()), 4), "max_abs_z": round(float(q[:6].abs().max()), 2)})
    res["windows"] = rows
    res["tensors_equal_rasters"] = all(r["cache_equals_rasters"] and r["dataset_equals_tensor_of_rasters"] and r["window_equals_dataset"] for r in rows)
    # outside the grid: NaN, so z = 0 and masks 0, as Neo is 0 there
    w0 = WindowTex(cfg, 1, tm, ts, (0, 0, 512, 512))
    corner = w0(-256, -256)
    res["outside_grid_is_zero"] = bool((corner[:, :256, :] == 0).all() and (corner[:, :, :256] == 0).all())
    # ---- 3. the Neo items and the targets equal the baseline's, with augmentation ---------------------------------------------------------------------------------
    fold = 1
    _, tr, val, _, _ = split(cfg, fold)
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    mean, sd = band_stats(cache, tr_i, cinfo["nodata"])
    tm, ts = tex_stats(tc, fold, tr_i)
    pho = cfg["train"].get("photometric")
    d_base = UGIDataset(cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    d_tex = UGITexDataset(tc, fold, tm, ts, cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    same = True
    for j in range(40):
        key = (j, 1000 + j)
        x0, y0, g0 = d_base[key]
        [x1, _], y1, g1 = d_tex[key]
        same &= bool(torch.equal(x0, x1) and torch.equal(y0, y1) and torch.equal(g0, g1))
    res["neo_items_equal_baseline"] = same
    # ---- 4. the geometry agrees: one pattern carried by both tensors ------------------------------------------------------------------------------------------------
    gc = out / "geometry_cache"
    gc.mkdir()
    rng = np.random.default_rng(0)
    pat = rng.integers(1000, 9000, (2, 512, 512)).astype(np.uint16)
    np.save(gc / "img.npy", pat[:, None].repeat(6, 1))
    for k in ("ugi", "grp", "edge"):
        np.save(gc / f"{k}.npy", np.zeros((2, 512, 512), np.uint8))
    sp = gc / "tex"
    sp.mkdir()
    np.save(sp / "tex.npy", pat[:, None].astype(np.float32).astype(np.float16).repeat(N_CACHE, 1))
    d_g = UGITexDataset(sp, 1, np.zeros(N_LAYERS), np.ones(N_LAYERS), gc, [0, 1], np.zeros(6), np.ones(6), 65535, True, None, {"global": [1.0, 1.0], "per_band": 0.0}, "none")
    agree, seen = True, set()
    for s in range(64):
        [x, p], _, _ = d_g[(s % 2, s)]
        agree &= bool(torch.allclose(x[0].float(), p[2].float(), atol=8.0))          # float16 steps of 4 at these values; hom3 has no log1p
        seen.add(tuple(np.asarray(x[0][0, :2]).tolist()))
    res["geometry_agrees"] = agree
    res["geometry_distinct_cases_seen"] = len(seen)                                 # 2 tiles x 8 transforms; the first run, selftest_e2/, expected 8 and so failed
    # ---- 5. step 0: the outputs equal the baseline's for the same weights ----------------------------------------------------------------------------------------
    green = [i for i in tr_i if np.isfinite(a[i, 0]).mean() > 0.2][:8]                 # training tiles that hold PAN layers: an empty batch would give the new channels no gradient
    d_b = UGITexDataset(tc, fold, tm, ts, cache, green, mean, sd, cinfo["nodata"], True, None, pho, "none")
    batch = [d_b[(j, 2000 + j)] for j in range(4)]
    x = torch.stack([b[0][0] for b in batch])
    p = torch.stack([b[0][1] for b in batch])
    res["batch_pan_valid_share"] = round(float(p[:, 6].mean()), 4)
    res["step0"] = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        seed_all(0)
        base = build(cfg["model"]["init"]).to(device)
        seed_all(0)
        m = build_e2(cfg).to(device)
        sd_b, sd_m = base.state_dict(), m.base.state_dict()
        c = m.base.enc.conv1.weight
        weights_same = all(torch.equal(sd_b[k], sd_m[k]) for k in sd_b if k != "enc.conv1.weight") and torch.equal(sd_b["enc.conv1.weight"], c[:, :6])
        r = {"same_weights_as_baseline": bool(weights_same), "new_conv1_weights_zero": bool((c[:, 6:] == 0).all()), "conv1_shape_ok": list(c.shape[:2]) == [64, 6 + TEX_CHANNELS]}
        xd, pd = x.to(device), p.to(device)
        with torch.no_grad():
            for mode in ("eval", "train"):
                base.train(mode == "train")
                m.train(mode == "train")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    o0, o1 = base(xd), m(xd, pd)
                    o2 = m(xd, torch.randn_like(pd) * 50)
                r[f"outputs_equal_{mode}"] = bool(torch.equal(o0.float(), o1.float()))
                r[f"outputs_equal_{mode}_whatever_texture_holds"] = bool(torch.equal(o0.float(), o2.float()))
        res["step0"][device] = r
    # ---- 6. it can learn: the new weights take a gradient and move -------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(0)
    m = build_e2(cfg).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    yb = torch.stack([b[1] for b in batch]).to(device)
    valid = yb != IGNORE
    opt.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(m(x.to(device), p.to(device))[:, 0][valid], yb[valid].float()).backward()
    g = m.base.enc.conv1.weight.grad
    res["learning"] = {"grad_new_channels": [round(float(v), 4) for v in g[:, 6:].abs().sum(dim=(0, 2, 3))]}
    opt.step()
    res["learning"]["new_weights_moved"] = bool((m.base.enc.conv1.weight[:, 6:] != 0).any())
    res["learning_passed"] = bool((g[:, 6:12].abs().sum(dim=(0, 2, 3)) > 0).all() and res["learning"]["new_weights_moved"])
    res["passed"] = bool(res["pan_selftest_passed"] and res["ndvi_texture_check_passed"] and res["pan_values_outside_vegetated_pixels"] == 0 and res["tensors_equal_rasters"] and res["outside_grid_is_zero"] and same and agree and len(seen) == 16
                         and res["learning_passed"] and all(all(v for v in r.values()) for r in res["step0"].values()))
    json.dump(res, open(out / "e2_selftest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "windows"}, indent=1))
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
        build_tex_cache(yaml.safe_load(open(a.cfg, encoding="utf-8")))
    if a.selftest:
        raise SystemExit(0 if selftest(a.cfg, a.out) else 1)
