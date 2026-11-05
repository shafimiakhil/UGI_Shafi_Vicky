"""E1 (CLAUDE.md 13.6): PlanetScope 2022 seasons at the 4 m stage of the UGI U-Net. One change against ugi_r1_none.

  * input: the B4 stage of itareo.ugi_seg.ps_stage: three dates (24 Apr, 10 Sep, 28 Oct 2022) x 8 bands, each date's measured displacement
    to the registered Neo removed inside its single resampling to the 4 m stage grid of the 512 m window, plus one availability
    channel per date: 27 channels, 128 x 128 px;
  * model: the layer1 output of the encoder (the 4 m stage, 64 channels) gains, as a sum, a small branch (two 3 x 3 conv-BN-ReLU) through a 1 x 1
    convolution initialised at zero, the gate, closed at the start: at step 0 the model is the baseline, whatever PlanetScope holds. layer2 and
    the 1/4 skip of the decoder both see the sum. No modality dropout;
  * normalisation: mean and SD per PlanetScope band from the internal training tiles of the fold, over valid cells; invalid cells are 0 after
    it, and the availability channels say so;
  * augmentation: the geometric draws of UGIDataset (same seed, same order) applied to the 4 m tensor. The photometric jitter stays on Neo only;
  * cache: ps.npy (tiles of the Neo cache, 24, 128, 128) float32, NaN where invalid, from ps_stage.stage_tile, so training reads
    what ps_stage builds. Inference calls stage_tile for each sliding window, whose origin is not a tile corner.

  * E1b (CLAUDE.md 13.7): model.planetscope.registration: plane, the plane per date of itareo.ugi_seg.ps_stage instead of the constant, inside the same single resampling; its own cache. Everything else as E1.

  python -m itareo.ugi_seg.ps_branch --build-cache --tiles runs_itareo/ugi/cache/neo2022_registered_v3/tiles.csv --out runs_itareo/ugi/cache/ps2022_stage4m [--registration plane]
  python -m itareo.ugi_seg.ps_branch --selftest --cfg itareo/configs/ugi_e1_ps.yaml --out <a new folder>       (exit 1 if it fails)
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

from itareo.safe_write import refuse_existing
from itareo.unet_ugi.data import read_csv

from .data import IGNORE, UGIDataset
from .model import build
from .ps_stage import DATES, PS, load_offsets, load_planes, stage_tile, stage_tile_plane

N_BANDS = 8
N_DATES = len(DATES)
N_PS = N_BANDS * N_DATES
PS_CHANNELS = N_PS + N_DATES
STAGE_PX = 128


def build_ps_cache(tiles_csv, out_dir, registration="constant"):
    out = Path(out_dir)
    refuse_existing(out)
    out.mkdir(parents=True)
    t0 = time.time()
    tiles = read_csv(tiles_csv)
    off = load_offsets() if registration == "constant" else load_planes()
    arr = np.lib.format.open_memmap(out / "ps.npy", "w+", np.float32, (len(tiles), N_PS, STAGE_PX, STAGE_PX))
    for k, d in enumerate(DATES):
        with rasterio.open(PS.format(d)) as src:
            for i, t in enumerate(tiles):
                arr[i, k * N_BANDS:(k + 1) * N_BANDS] = (stage_tile(src, t["row0"], t["col0"], off[d]["dx_east_m"], off[d]["dy_north_m"]) if registration == "constant" else stage_tile_plane(src, t["row0"], t["col0"], off[d]))
        print(d, round(time.time() - t0), "s", flush=True)
    arr.flush()
    json.dump({"n": len(tiles), "dates": DATES, "bands": N_BANDS, "stage_px": STAGE_PX, "tiles": str(tiles_csv), "registration": registration, "offsets": off, "source": PS, "seconds": round(time.time() - t0, 1)},
              open(out / "cache.json", "w"), indent=1)


def ps_stats(cache_dir, idx):
    """Mean and SD per PlanetScope channel over the valid cells of tiles idx."""
    a = np.load(Path(cache_dir) / "ps.npy", mmap_mode="r")
    n, s1, s2 = np.zeros(N_PS), np.zeros(N_PS), np.zeros(N_PS)
    for i in idx:
        v = np.asarray(a[i], np.float64)
        ok = np.isfinite(v)
        n += ok.sum(axis=(1, 2))
        s1 += np.where(ok, v, 0).sum(axis=(1, 2))
        s2 += np.where(ok, v * v, 0).sum(axis=(1, 2))
    mean = s1 / n
    return mean, np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))


def ps_tensor(raw, mean, sd):
    """(24, H, W) with NaN where invalid -> (27, H, W): normalised, 0 where invalid, then one availability channel per date."""
    ok = np.isfinite(raw)
    m, s = np.asarray(mean, np.float32)[:, None, None], np.asarray(sd, np.float32)[:, None, None]
    z = np.where(ok, (np.nan_to_num(raw) - m) / s, 0.0)
    av = np.stack([ok[k * N_BANDS:(k + 1) * N_BANDS].all(0) for k in range(N_DATES)])
    return np.concatenate([z, av]).astype(np.float32)


class UGIPSDataset(UGIDataset):
    """UGIDataset, whose items are unchanged, with the PlanetScope tensor beside the Neo tensor: ([x, ps], y_bin, y_grp)."""

    def __init__(self, ps_cache, ps_mean, ps_sd, *args, **kw):
        super().__init__(*args, **kw)
        self.ps_path, self.ps_mean, self.ps_sd, self.ps = str(Path(ps_cache) / "ps.npy"), ps_mean, ps_sd, None

    def __getitem__(self, key):
        x, yb, yg = super().__getitem__(key)
        if self.ps is None:
            self.ps = np.load(self.ps_path, mmap_mode="r")
        j, seed = key if isinstance(key, tuple) else (key, None)
        p = ps_tensor(np.asarray(self.ps[self.idx[j]]), self.ps_mean, self.ps_sd)
        if self.augment:
            rng = np.random.default_rng(seed)                     # the draws of UGIDataset.__getitem__, in its order
            rng.uniform(*self.photo["global"])
            rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"], size=(x.shape[0], 1, 1))
            p = np.rot90(p, int(rng.integers(4)), (1, 2))
            if rng.random() < 0.5:
                p = p[:, :, ::-1]
        return [x, torch.from_numpy(np.ascontiguousarray(p))], yb, yg


class WindowPS:
    """The PlanetScope tensor of a sliding window whose corner is (row0, col0) of GRID_1M, from ps_stage.stage_tile."""

    def __init__(self, ps_mean, ps_sd, registration="constant"):
        self.mean, self.sd, self.registration = ps_mean, ps_sd, registration
        off = load_offsets() if registration == "constant" else load_planes()
        self.src = [(rasterio.open(PS.format(d)), off[d]) for d in DATES]

    def raw(self, row0, col0):
        if self.registration == "constant":
            return np.concatenate([stage_tile(s, row0, col0, o["dx_east_m"], o["dy_north_m"]) for s, o in self.src])
        return np.concatenate([stage_tile_plane(s, row0, col0, o) for s, o in self.src])

    def __call__(self, row0, col0):
        return torch.from_numpy(ps_tensor(self.raw(row0, col0), self.mean, self.sd))

    def close(self):
        for s, _ in self.src:
            s.close()


class UNetPS(nn.Module):
    def __init__(self, base, channels=64, in_channels=PS_CHANNELS, stage_channels=64):
        super().__init__()
        self.base = base
        self.ps_branch = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                                       nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.ps_gate = nn.Conv2d(channels, stage_channels, 1)
        nn.init.zeros_(self.ps_gate.weight)
        nn.init.zeros_(self.ps_gate.bias)

    def forward(self, x, ps):
        b, e = self.base, self.base.enc
        x0 = e.relu(e.bn1(e.conv1(x)))
        x1 = e.layer1(e.maxpool(x0)) + self.ps_gate(self.ps_branch(ps))
        x2 = e.layer2(x1)
        x3 = e.layer3(x2)
        y = e.layer4(x3)
        for blk, skip in zip(b.blocks, (x3, x2, x1, x0)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return b.head(b.final(y))


def build_e1(cfg):
    """The baseline's model first, drawn as the baseline draws it; the branch after it."""
    return UNetPS(build(cfg["model"]["init"]), cfg["model"]["planetscope"]["channels"])


def selftest(cfg_path, out):
    """Before E1 trains: the step-0 outputs equal the baseline's for the same weights; the PlanetScope tensors equal ps_stage on the self-test windows;
    the Neo items equal the baseline's; the geometry agrees on both grids. Writes <out>/e1_selftest.json. Returns True if all passed."""
    import yaml

    from itareo.unet_ugi.train import seed_all

    from .train import split
    out = Path(out)
    refuse_existing(out / "e1_selftest.json", out / "geometry_cache")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    res = {"config": str(cfg_path)}
    cache, ps_cache = Path(cfg["paths"]["cache"]), Path(cfg["paths"]["ps_cache"])
    cinfo = json.load(open(cache / "cache.json"))
    tiles = read_csv(cache / "tiles.csv")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(tiles)}
    _, tr, val, _, _ = split(cfg, 1)
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    from itareo.unet_ugi.data import band_stats
    mean, sd = band_stats(cache, tr_i, cinfo["nodata"])
    pm, ps_ = ps_stats(ps_cache, tr_i)
    res["ps_valid_mean_range"] = [round(float(pm.min()), 1), round(float(pm.max()), 1)]
    res["ps_sd_range"] = [round(float(ps_.min()), 1), round(float(ps_.max()), 1)]
    # ---- 1. PlanetScope tensors against ps_stage on its self-test windows -----------------------------------------------------------
    reg = cfg["model"]["planetscope"].get("registration", "constant")
    res["registration"] = reg
    win = WindowPS(pm, ps_, reg)
    a = np.load(ps_cache / "ps.npy", mmap_mode="r")
    st = json.load(open(Path("runs_itareo/ugi/b4_ps_stage_selftest.json")))
    sel = sorted({(r["row0"], r["col0"]) for r in st["rows"]})
    res["ps_stage_selftest_passed"] = bool(st["passed"])
    ds_plain = UGIPSDataset(ps_cache, pm, ps_, cache, [index[w] for w in sel], mean, sd, cinfo["nodata"], False, edge_band="none")
    rows = []
    for j, (r0, c0) in enumerate(sel):
        raw = win.raw(r0, c0)
        c = np.asarray(a[index[(r0, c0)]])
        [_, p], _, _ = ds_plain[j]
        rows.append({"row0": r0, "col0": c0, "cache_equals_stage_tile": bool(np.array_equal(raw, c, equal_nan=True)), "dataset_equals_tensor_of_stage_tile": bool(np.array_equal(p.numpy(), ps_tensor(raw, pm, ps_))),
                     "window_equals_tensor": bool(torch.equal(win(r0, c0), p)), "valid_cells_per_date": [int(p[N_PS + k].sum()) for k in range(N_DATES)]})
    res["ps_windows"] = rows
    res["ps_tensors_equal_ps_stage"] = all(r["cache_equals_stage_tile"] and r["dataset_equals_tensor_of_stage_tile"] and r["window_equals_tensor"] for r in rows)
    t0 = time.time()
    for k in range(20):
        win(10807 + 256 * k, 256 * k)
    res["seconds_per_inference_window"] = round((time.time() - t0) / 20, 4)
    win.close()
    # ---- 2. the Neo items and the targets equal the baseline's, with augmentation ----------------------------------------------------
    pho = cfg["train"].get("photometric")
    d_base = UGIDataset(cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    d_ps = UGIPSDataset(ps_cache, pm, ps_, cache, tr_i[:40], mean, sd, cinfo["nodata"], True, None, pho, "none")
    same = True
    for j in range(40):
        key = (j, 1000 + j)
        x0, y0, g0 = d_base[key]
        [x1, _], y1, g1 = d_ps[key]
        same &= bool(torch.equal(x0, x1) and torch.equal(y0, y1) and torch.equal(g0, g1))
    res["neo_items_equal_baseline"] = same
    # ---- 3. the geometry agrees on both grids: a pattern on the 4 m cells, carried by both tensors --------------------------------
    gc = out / "geometry_cache"
    gc.mkdir()
    rng = np.random.default_rng(0)
    pat = rng.integers(1000, 9000, (2, 128, 128)).astype(np.uint16)
    np.save(gc / "img.npy", np.repeat(np.repeat(pat[:, None], 4, 2), 4, 3).repeat(6, 1))
    for k in ("ugi", "grp", "edge"):
        np.save(gc / f"{k}.npy", np.zeros((2, 512, 512), np.uint8))
    sp = gc / "ps"
    sp.mkdir()
    np.save(sp / "ps.npy", np.repeat(pat[:, None].astype(np.float32), N_PS, 1))
    d_g = UGIPSDataset(sp, np.zeros(N_PS), np.ones(N_PS), gc, [0, 1], np.zeros(6), np.ones(6), 65535, True, None, {"global": [1.0, 1.0], "per_band": 0.0}, "none")
    agree, seen = True, set()
    for s in range(64):
        [x, p], _, _ = d_g[(s % 2, s)]
        cell = x[0].reshape(128, 4, 128, 4)
        agree &= bool((cell == cell[:, :1, :, :1]).all() and np.array_equal(cell[:, 0, :, 0].numpy(), p[0].numpy()))
        seen.add(tuple(np.asarray(x[0][0, :2]).tolist()))
    res["geometry_agrees_on_both_grids"] = agree
    res["geometry_distinct_transforms_seen"] = len(seen)
    # ---- 4. step 0: the outputs equal the baseline's for the same weights ----------------------------------------------------------------
    b4 = {(t["row0"], t["col0"]): t["avail_ps_all3"] for t in read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv")}
    full = [index[(t["row0"], t["col0"])] for t in tr if b4[(t["row0"], t["col0"])] >= 0.99][:8]     # training tiles that hold all three dates
    d_b = UGIPSDataset(ps_cache, pm, ps_, cache, full, mean, sd, cinfo["nodata"], True, None, pho, "none")
    batch = [d_b[(j, 2000 + j)] for j in range(4)]
    x = torch.stack([b[0][0] for b in batch])
    p = torch.stack([b[0][1] for b in batch])
    res["batch_availability_share_per_date"] = [round(float(p[:, N_PS + k].mean()), 4) for k in range(N_DATES)]
    res["step0"] = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        seed_all(0)
        base = build(cfg["model"]["init"]).to(device)
        seed_all(0)
        m = build_e1(cfg).to(device)
        sd_b, sd_m = base.state_dict(), m.base.state_dict()
        weights_same = all(torch.equal(sd_b[k], sd_m[k]) for k in sd_b)
        r = {"same_weights_as_baseline": weights_same, "gate_closed": bool((m.ps_gate.weight == 0).all() and (m.ps_gate.bias == 0).all())}
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
    # ---- 5. it can learn: the gate takes a gradient at once, the branch after the gate has moved ----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(0)
    m = build_e1(cfg).to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    yb = torch.stack([b[1] for b in batch]).to(device)
    valid = yb != IGNORE

    def step():
        opt.zero_grad(set_to_none=True)
        o = m(x.to(device), p.to(device))
        F.binary_cross_entropy_with_logits(o[:, 0][valid], yb[valid].float()).backward()
        g = float(m.ps_gate.weight.grad.abs().sum())
        br = m.ps_branch[0].weight.grad
        opt.step()
        return g, 0.0 if br is None else float(br.abs().sum())
    g1, b1 = step()
    g2, b2 = step()
    res["learning"] = {"step1_gate_grad": g1, "step1_branch_grad": b1, "step2_gate_grad": g2, "step2_branch_grad": b2}
    res["learning_passed"] = g1 > 0 and b1 == 0 and b2 > 0
    res["passed"] = bool(res["ps_stage_selftest_passed"] and res["ps_tensors_equal_ps_stage"] and res["neo_items_equal_baseline"] and agree and len(seen) == 8 and res["learning_passed"]
                         and all(all(v for k, v in r.items()) for r in res["step0"].values()))
    json.dump(res, open(out / "e1_selftest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "ps_windows"}, indent=1))
    return res["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tiles")
    ap.add_argument("--registration", default="constant")
    ap.add_argument("--cfg")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.build_cache:
        build_ps_cache(a.tiles, a.out, a.registration)
    if a.selftest:
        raise SystemExit(0 if selftest(a.cfg, a.out) else 1)
