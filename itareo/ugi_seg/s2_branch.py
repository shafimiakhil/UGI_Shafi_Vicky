"""E3 (CLAUDE.md 13.7): Sentinel-2 2022 quarters at the 8 m stage of the UGI U-Net. One change against ugi_r1_none. A plugin of itareo.ugi_seg.train and oof (config model.plugin).

  * input: the 14 bands of derived/features/s2_2022_indices_10m.tif (NDVI, NDRE, NDMI of each 2022 quarter, the dry-season NDVI minimum, the NDVI amplitude; int16 x 10000), the indices rather than the bands (the choice is said in the config). The constant
    displacement against Neo v3, -1.42 E / +3.98 N m (docs/evidence_phase2/ugi_surface/surface_s2_displacement.json; convention of the PlanetScope offsets: content at Neo position (x, y) lies at (x + dx, y + dy)), is removed inside the single resampling, area averaging to the 8 m stage grid of the 512 m window, 64 x 64 px, plus one availability channel: 15 channels.
    Residual per block: docs/evidence_phase2/ugi_seg/e3_s2_residuals/summary.json;
  * model: the layer2 output of the encoder (the 8 m stage, 128 channels) gains, as a sum, a small branch (two 3 x 3 conv-BN-ReLU, 64 channels) through a 1 x 1 convolution initialised at zero, the gate, closed at the start: at step 0 the model is the baseline, whatever Sentinel-2 holds. layer3 and the 1/8 skip of the decoder both see the sum. No modality dropout;
  * normalisation: mean and SD per band from the internal training tiles of the fold, over valid cells; invalid cells are 0 after it, and the availability channel says so;
  * augmentation: the geometric draws of UGIDataset (same seed, same order) applied to the 8 m tensor. The photometric jitter stays on Neo only;
  * cache: s2.npy (tiles of the Neo cache, 14, 64, 64) float32, NaN where invalid, from stage_tile. Inference calls stage_tile for each sliding window.

  python -m itareo.ugi_seg.s2_branch --build-cache --cfg itareo/configs/ugi_e3_s2.yaml
  python -m itareo.ugi_seg.s2_branch --selftest --cfg itareo/configs/ugi_e3_s2.yaml --out <a new folder>       (exit 1 if it fails)
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
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from itareo.safe_write import refuse_existing
from itareo.unet_ugi.data import read_csv

from .data import IGNORE, UGIDataset
from .model import build as build_base

N_S2 = 14
S2_CHANNELS = N_S2 + 1
STAGE, TILE, STAGE_PX = 8.0, 512, 64
GRID1M_ORIGIN = (375327.0, 6535833.0)


def stage_tile(src, row0, col0, dx, dy):
    """src: the open index raster, 10 m. Returns (14, 64, 64) float32, values x 10000, NaN where invalid, on the 8 m stage grid of the window whose corner is (row0, col0) of GRID_1M, with the content shifted by (-dx, -dy) m."""
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    dst_t = Affine(STAGE, 0, x0, 0, -STAGE, y0)
    st = src.transform
    c = int((x0 + dx - st.c) // 10) - 2
    r = int((st.f - y0 - dy) // 10) - 2
    n = int(TILE / 10) + 6
    w = Window(c, r, n, n)
    a = src.read(window=w, boundless=True, fill_value=src.nodata).astype(np.float32)
    a[a == src.nodata] = np.nan
    s = src.window_transform(w)
    out = np.full((a.shape[0], STAGE_PX, STAGE_PX), np.nan, np.float32)
    reproject(a, out, src_transform=Affine(s.a, s.b, s.c - dx, s.d, s.e, s.f - dy), src_crs=src.crs, dst_transform=dst_t, dst_crs=src.crs, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def displacement(cfg):
    d = cfg["model"]["plugin"]["displacement_m"]
    return d["dx_east"], d["dy_north"]


def build_cache(cfg):
    out = Path(cfg["paths"]["s2_cache"])
    refuse_existing(out)
    out.mkdir(parents=True)
    t0 = time.time()
    tiles = read_csv(Path(cfg["paths"]["cache"]) / "tiles.csv")
    dx, dy = displacement(cfg)
    arr = np.lib.format.open_memmap(out / "s2.npy", "w+", np.float32, (len(tiles), N_S2, STAGE_PX, STAGE_PX))
    with rasterio.open(cfg["paths"]["s2"]) as src:
        for i, t in enumerate(tiles):
            arr[i] = stage_tile(src, t["row0"], t["col0"], dx, dy)
    arr.flush()
    json.dump({"n": len(tiles), "stage_px": STAGE_PX, "source": cfg["paths"]["s2"], "displacement_m": cfg["model"]["plugin"]["displacement_m"], "seconds": round(time.time() - t0, 1)}, open(out / "cache.json", "w"), indent=1)
    print("s2 cache", len(tiles), "tiles", round(time.time() - t0), "s", flush=True)


def s2_tensor(raw, mean, sd):
    """(14, H, W), NaN where invalid -> (15, H, W): normalised, 0 where invalid, then the availability channel."""
    ok = np.isfinite(raw)
    m, s = np.asarray(mean, np.float32)[:, None, None], np.asarray(sd, np.float32)[:, None, None]
    z = np.where(ok, (np.nan_to_num(raw) - m) / s, 0.0)
    return np.concatenate([z, ok.all(0, keepdims=True)]).astype(np.float32)


class UGIS2Dataset(UGIDataset):
    """UGIDataset, whose items are unchanged, with the Sentinel-2 tensor beside the Neo tensor: ([x, s2], y_bin, y_grp)."""

    def __init__(self, s2_cache, s2_mean, s2_sd, *args, **kw):
        super().__init__(*args, **kw)
        self.s2_path, self.s2_mean, self.s2_sd, self.s2 = str(Path(s2_cache) / "s2.npy"), s2_mean, s2_sd, None

    def __getitem__(self, key):
        x, yb, yg = super().__getitem__(key)
        if self.s2 is None:
            self.s2 = np.load(self.s2_path, mmap_mode="r")
        j, seed = key if isinstance(key, tuple) else (key, None)
        p = s2_tensor(np.asarray(self.s2[self.idx[j]]), self.s2_mean, self.s2_sd)
        if self.augment:
            rng = np.random.default_rng(seed)                     # the draws of UGIDataset.__getitem__, in its order
            rng.uniform(*self.photo["global"])
            rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"], size=(x.shape[0], 1, 1))
            p = np.rot90(p, int(rng.integers(4)), (1, 2))
            if rng.random() < 0.5:
                p = p[:, :, ::-1]
        return [x, torch.from_numpy(np.ascontiguousarray(p))], yb, yg


class WindowS2:
    """The Sentinel-2 tensor of a sliding window whose corner is (row0, col0) of GRID_1M, from stage_tile."""

    def __init__(self, path, dx, dy, mean, sd):
        self.src, self.dx, self.dy, self.mean, self.sd = rasterio.open(path), dx, dy, mean, sd

    def raw(self, row0, col0):
        return stage_tile(self.src, row0, col0, self.dx, self.dy)

    def __call__(self, row0, col0):
        return torch.from_numpy(s2_tensor(self.raw(row0, col0), self.mean, self.sd))


class UNetS2(nn.Module):
    def __init__(self, base, channels=64, in_channels=S2_CHANNELS, stage_channels=128):
        super().__init__()
        self.base = base
        self.s2_branch = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                       nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.s2_gate = nn.Conv2d(channels, stage_channels, 1)
        nn.init.zeros_(self.s2_gate.weight)
        nn.init.zeros_(self.s2_gate.bias)

    def forward(self, x, s2):
        b, e = self.base, self.base.enc
        x0 = e.relu(e.bn1(e.conv1(x)))
        x1 = e.layer1(e.maxpool(x0))
        x2 = e.layer2(x1) + self.s2_gate(self.s2_branch(s2))
        x3 = e.layer3(x2)
        y = e.layer4(x3)
        for blk, skip in zip(b.blocks, (x3, x2, x1, x0)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return b.head(b.final(y))


# ---- the plugin interface of itareo.ugi_seg.train and oof ---------------------------------------------------------------------------------------------

def build(cfg):
    """The baseline's model first, drawn as the baseline draws it; the branch after it."""
    return UNetS2(build_base(cfg["model"]["init"]), cfg["model"]["plugin"]["channels"])


def stats(cfg, fold, tr_i):
    """Mean and SD per band over the valid cells of the internal training tiles of the fold; stored in split.json and best.pt."""
    a = np.load(Path(cfg["paths"]["s2_cache"]) / "s2.npy", mmap_mode="r")
    n, s1, s2 = np.zeros(N_S2), np.zeros(N_S2), np.zeros(N_S2)
    for i in tr_i:
        v = np.asarray(a[i], np.float64)
        ok = np.isfinite(v)
        n += ok.sum(axis=(1, 2))
        s1 += np.where(ok, v, 0).sum(axis=(1, 2))
        s2 += np.where(ok, v * v, 0).sum(axis=(1, 2))
    mean = s1 / n
    return {"s2_mean": mean.tolist(), "s2_sd": np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12)).tolist()}


def dataset(cfg, fold, st, *args, **kw):
    return UGIS2Dataset(cfg["paths"]["s2_cache"], st["s2_mean"], st["s2_sd"], *args, **kw)


def window(cfg, fold, ck, region):
    dx, dy = displacement(cfg)
    return WindowS2(cfg["paths"]["s2"], dx, dy, ck["s2_mean"], ck["s2_sd"])


# ---- self-test -----------------------------------------------------------------------------------------------------------------------------------------

def _overlap(dst_edges, src_edges):
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0, None)


def _reference(src, row0, col0, dx, dy):
    """Independent stage-grid value: separable overlap-length weights of the shifted 10 m pixels on the 8 m pixels, invalid pixels excluded (float64)."""
    st = src.transform
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    c, r, n = int((x0 - st.c) // 10) - 4, int((st.f - y0) // 10) - 4, 62
    a = src.read(window=Window(c, r, n, n), boundless=True, fill_value=src.nodata).astype(np.float64)
    v = ~(a == src.nodata)
    xs = st.c + 10 * (c + np.arange(n + 1)) - dx
    ys = st.f - 10 * (r + np.arange(n + 1)) - dy
    Wx = _overlap(x0 + STAGE * np.arange(STAGE_PX + 1), xs)
    Wy = _overlap(-(y0 - STAGE * np.arange(STAGE_PX + 1)), -ys)
    out = np.full((a.shape[0], STAGE_PX, STAGE_PX), np.nan)
    for b in range(a.shape[0]):
        num = Wy @ np.where(v[b], a[b], 0) @ Wx.T
        den = Wy @ v[b].astype(float) @ Wx.T
        out[b] = num / np.where(den > 1e-9, den, 1)
        out[b][den <= 1e-9] = np.nan
    return out


def selftest(cfg_path, out):
    """Before E3 trains: stage_tile against an independent average; the tensors of the cache, of the dataset and of the window equal stage_tile; the residual file exists; the Neo items equal the baseline's; the geometry agrees on both grids; the step-0 outputs equal the baseline's; the branch learns."""
    import yaml

    from itareo.unet_ugi.data import band_stats
    from itareo.unet_ugi.train import seed_all

    from .train import split
    out = Path(out)
    refuse_existing(out / "e3_selftest.json", out / "geometry_cache")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    res = {"config": str(cfg_path)}
    cache, sc = Path(cfg["paths"]["cache"]), Path(cfg["paths"]["s2_cache"])
    cinfo = json.load(open(cache / "cache.json"))
    tiles = read_csv(cache / "tiles.csv")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(tiles)}
    _, tr, val, _, _ = split(cfg, 1)
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    mean, sd = band_stats(cache, tr_i, cinfo["nodata"])
    st = stats(cfg, 1, tr_i)
    dx, dy = displacement(cfg)
    res["displacement_m"] = [dx, dy]
    res["residual_summary_exists"] = Path("../docs/evidence_phase2/ugi_seg/e3_s2_residuals/summary.json").exists()
    a = np.load(sc / "s2.npy", mmap_mode="r")
    win = window(cfg, 1, st, None)
    rng = np.random.default_rng(0)
    rows = []
    sel = [tiles[k] for k in rng.choice(len(tiles), 12, replace=False)]
    d_plain = UGIS2Dataset(sc, st["s2_mean"], st["s2_sd"], cache, [index[(t["row0"], t["col0"])] for t in sel], mean, sd, cinfo["nodata"], False, edge_band="none")
    for j, t in enumerate(sel):
        r0, c0 = t["row0"], t["col0"]
        raw = win.raw(r0, c0)
        q = _reference(win.src, r0, c0, dx, dy)
        both = np.isfinite(raw) & np.isfinite(q)
        rel = np.abs(raw - q)[both] / np.maximum(np.abs(q[both]), 100)
        [_, p], _, _ = d_plain[j]
        rows.append({"row0": r0, "col0": c0, "cache_equals_stage_tile": bool(np.array_equal(raw, np.asarray(a[index[(r0, c0)]]), equal_nan=True)), "dataset_equals_tensor": bool(np.array_equal(p.numpy(), s2_tensor(raw, st["s2_mean"], st["s2_sd"]))), "window_equals_dataset": bool(torch.equal(win(r0, c0), p)),
                     "same_nan_as_reference": bool(np.array_equal(np.isfinite(raw), np.isfinite(q))), "max_rel_diff_against_reference": float(rel.max()) if rel.size else None, "valid_share": round(float(p[N_S2].mean()), 4)})
    res["windows"] = rows
    res["tensors_equal_stage_tile"] = all(r["cache_equals_stage_tile"] and r["dataset_equals_tensor"] and r["window_equals_dataset"] for r in rows)
    res["stage_tile_equals_reference"] = all(r["same_nan_as_reference"] and (r["max_rel_diff_against_reference"] or 0) < 1e-3 for r in rows)
    t0 = time.time()
    for k in range(20):
        win(10807 + 256 * k, 256 * k)
    res["seconds_per_inference_window"] = round((time.time() - t0) / 20, 4)
    # outside the grid: NaN, so z = 0 and availability 0
    corner = win(-256, -256)
    res["outside_grid_is_zero"] = bool((corner[:, :24, :] == 0).all() and (corner[:, :, :24] == 0).all())
    # the Neo items and the targets equal the baseline's, with augmentation
    pho = cfg["train"].get("photometric")
    d_base = UGIDataset(cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    d_s2 = dataset(cfg, 1, st, cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    same = True
    for j in range(40):
        key = (j, 1000 + j)
        x0, y0, g0 = d_base[key]
        [x1, _], y1, g1 = d_s2[key]
        same &= bool(torch.equal(x0, x1) and torch.equal(y0, y1) and torch.equal(g0, g1))
    res["neo_items_equal_baseline"] = same
    # the geometry agrees on both grids: a pattern on the 8 m cells, carried by both tensors
    gc = out / "geometry_cache"
    gc.mkdir()
    grng = np.random.default_rng(0)
    pat = grng.integers(1000, 9000, (2, 64, 64)).astype(np.uint16)
    np.save(gc / "img.npy", np.repeat(np.repeat(pat[:, None], 8, 2), 8, 3).repeat(6, 1))
    for k in ("ugi", "grp", "edge"):
        np.save(gc / f"{k}.npy", np.zeros((2, 512, 512), np.uint8))
    sp = gc / "s2"
    sp.mkdir()
    np.save(sp / "s2.npy", np.repeat(pat[:, None].astype(np.float32), N_S2, 1))
    d_g = UGIS2Dataset(sp, np.zeros(N_S2), np.ones(N_S2), gc, [0, 1], np.zeros(6), np.ones(6), 65535, True, None, {"global": [1.0, 1.0], "per_band": 0.0}, "none")
    agree, seen = True, set()
    for s in range(64):
        [x, p], _, _ = d_g[(s % 2, s)]
        cell = x[0].reshape(64, 8, 64, 8)
        agree &= bool((cell == cell[:, :1, :, :1]).all() and np.array_equal(cell[:, 0, :, 0].numpy(), p[0].numpy()))
        seen.add((s % 2, tuple(np.asarray(p[0][0, :2]).tolist())))
    res["geometry_agrees_on_both_grids"] = agree
    res["geometry_distinct_cases_seen"] = len(seen)           # 2 tiles x 8 transforms
    # step 0: the outputs equal the baseline's for the same weights
    d_b = dataset(cfg, 1, st, cache, tr_i[:8], mean, sd, cinfo["nodata"], True, None, pho, "none")
    batch = [d_b[(j, 2000 + j)] for j in range(4)]
    x = torch.stack([b[0][0] for b in batch])
    p = torch.stack([b[0][1] for b in batch])
    res["batch_availability_share"] = round(float(p[:, N_S2].mean()), 4)
    res["step0"] = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        seed_all(0)
        base = build_base(cfg["model"]["init"]).to(device)
        seed_all(0)
        m = build(cfg).to(device)
        sd_b, sd_m = base.state_dict(), m.base.state_dict()
        r = {"same_weights_as_baseline": all(torch.equal(sd_b[k], sd_m[k]) for k in sd_b), "gate_closed": bool((m.s2_gate.weight == 0).all() and (m.s2_gate.bias == 0).all())}
        xd, pd = x.to(device), p.to(device)
        with torch.no_grad():
            for mode in ("eval", "train"):
                base.train(mode == "train")
                m.train(mode == "train")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    o0, o1 = base(xd), m(xd, pd)
                    o2 = m(xd, torch.randn_like(pd) * 50)
                r[f"outputs_equal_{mode}"] = bool(torch.equal(o0.float(), o1.float()))
                r[f"outputs_equal_{mode}_whatever_sentinel2_holds"] = bool(torch.equal(o0.float(), o2.float()))
        res["step0"][device] = r
    # it can learn: the gate takes a gradient at once, the branch after the gate has moved
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(0)
    m = build(cfg).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    yb = torch.stack([b[1] for b in batch]).to(device)
    valid = yb != IGNORE

    def step():
        opt.zero_grad(set_to_none=True)
        o = m(x.to(device), p.to(device))
        F.binary_cross_entropy_with_logits(o[:, 0][valid], yb[valid].float()).backward()
        g = float(m.s2_gate.weight.grad.abs().sum())
        br = m.s2_branch[0].weight.grad
        opt.step()
        return g, 0.0 if br is None else float(br.abs().sum())
    g1, b1 = step()
    g2, b2 = step()
    res["learning"] = {"step1_gate_grad": g1, "step1_branch_grad": b1, "step2_gate_grad": g2, "step2_branch_grad": b2}
    res["learning_passed"] = g1 > 0 and b1 == 0 and b2 > 0
    res["passed"] = bool(res["residual_summary_exists"] and res["tensors_equal_stage_tile"] and res["stage_tile_equals_reference"] and res["outside_grid_is_zero"] and same and agree and len(seen) == 16 and res["learning_passed"] and all(all(v for v in r.values()) for r in res["step0"].values()))
    json.dump(res, open(out / "e3_selftest.json", "w"), indent=1)
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
        build_cache(yaml.safe_load(open(a.cfg, encoding="utf-8")))
    if a.selftest:
        raise SystemExit(0 if selftest(a.cfg, a.out) else 1)
