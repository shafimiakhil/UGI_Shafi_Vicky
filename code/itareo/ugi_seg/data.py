"""Tile cache and dataset for UGI segmentation.

Cache (memory-mappable .npy, row i = tile i of tiles.csv): img (N, B, T, T) uint16 from the image raster; ugi (N, T, T)
uint8 binary; grp (N, T, T) uint8 group code 0-6; edge (N, T, T) uint8. Image pixels with nodata in any band are
invalid (ignored in both heads).
"""
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

IGNORE = 255
N_GROUPS = 5          # codes 1-5; 6 = NO DEFINIDO (UGI, no type)


def build_cache(image, labels_dir, tiles, out_dir, tile=512):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab = Path(labels_dir)
    with rasterio.open(image) as im, rasterio.open(lab / "ugi_binary_grid1m.tif") as ub, \
            rasterio.open(lab / "ugi_group_grid1m.tif") as ug, rasterio.open(lab / "ugi_edge_grid1m.tif") as ue:
        assert im.transform == ub.transform and im.shape == ub.shape
        N = len(tiles)
        arr = {"img": np.lib.format.open_memmap(out / "img.npy", "w+", np.uint16, (N, im.count, tile, tile))}
        for k in ("ugi", "grp", "edge"):
            arr[k] = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.uint8, (N, tile, tile))
        for i, t in enumerate(tiles):
            w = Window(int(t["col0"]), int(t["row0"]), tile, tile)
            arr["img"][i] = im.read(window=w)
            arr["ugi"][i] = ub.read(1, window=w)
            arr["grp"][i] = ug.read(1, window=w)
            arr["edge"][i] = ue.read(1, window=w)
        for a in arr.values():
            a.flush()
        nodata = im.nodata
        bands = im.count
    json.dump({"n": N, "tile": tile, "nodata": nodata, "bands": bands, "image": str(image), "labels": str(labels_dir)},
              open(out / "cache.json", "w"), indent=1)


EDGE_BANDS = ("both", "outer", "inner", "none")


def targets(ugi, grp, edge, ok, buffer=None, edge_band="both"):
    """Binary and group targets with IGNORE for invalid image and buffer pixels and for the 1 m edge band as set by
    edge_band (CLAUDE.md 13.3 F3, 13.4): both = the band on both sides of every boundary ignored (A1/A2 round); outer =
    only band pixels outside the inventory ignored; inner = only band pixels inside the inventory ignored; none = no band.
    Both heads follow it."""
    if edge_band not in EDGE_BANDS:
        raise ValueError(f"edge_band must be one of {EDGE_BANDS}, not {edge_band!r}")
    ign = ~ok
    if edge_band == "both":
        ign = ign | (edge > 0)
    elif edge_band == "outer":
        ign = ign | ((edge > 0) & (ugi == 0))
    elif edge_band == "inner":
        ign = ign | ((edge > 0) & (ugi == 1))
    if buffer is not None:
        ign |= buffer
    yb = np.where(ign, IGNORE, ugi).astype(np.int64)
    yg = np.where(ign | (ugi == 0) | (grp < 1) | (grp > N_GROUPS), IGNORE, grp.astype(np.int64) - 1)
    return yb, yg


def label_counts(cache_dir, idx, nodata, buffer_masks=None, edge_band="both"):
    """Pixels entering the loss: binary positives / negatives and group pixels (codes 1-5), over tiles idx."""
    c = Path(cache_dir)
    a = {k: np.load(c / f"{k}.npy", mmap_mode="r") for k in ("img", "ugi", "grp", "edge")}
    buffer_masks = buffer_masks or {}
    pos = neg = 0
    gcnt = np.zeros(N_GROUPS, np.int64)
    gpres = np.zeros(N_GROUPS, np.int64)
    for i in idx:
        ok = (np.asarray(a["img"][i]) != nodata).all(0)
        yb, yg = targets(np.asarray(a["ugi"][i]), np.asarray(a["grp"][i]), np.asarray(a["edge"][i]), ok,
                         buffer_masks.get(i), edge_band)
        pos += int((yb == 1).sum())
        neg += int((yb == 0).sum())
        g = np.bincount(yg[yg != IGNORE].ravel(), minlength=N_GROUPS)
        gcnt += g
        gpres += (g > 0) * g.sum()
    return pos, neg, gcnt, gpres


class UGIDataset(torch.utils.data.Dataset):
    """x (B, T, T) normalised; y_bin (T, T) int64 in {0, 1, IGNORE}; y_grp (T, T) int64 in {0..4, IGNORE}.
    With augmentation the key is (position, seed) from itareo.unet_ugi.data.SeededSampler."""

    def __init__(self, cache_dir, idx, mean, sd, nodata, augment, buffer_masks=None, photometric=None, edge_band="both"):
        self.cache, self.idx = str(cache_dir), list(idx)
        self.edge_band = edge_band
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.sd = np.asarray(sd, np.float32)[:, None, None]
        self.nodata, self.augment = nodata, augment
        self.buffer = buffer_masks or {}
        self.photo = photometric or {"global": [0.9, 1.1], "per_band": 0.03}
        self.arr = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        if self.arr is None:
            c = Path(self.cache)
            self.arr = {k: np.load(c / f"{k}.npy", mmap_mode="r") for k in ("img", "ugi", "grp", "edge")}
        j, seed = key if isinstance(key, tuple) else (key, None)
        i = self.idx[j]
        raw = np.asarray(self.arr["img"][i])
        ok = (raw != self.nodata).all(0)
        x = raw.astype(np.float32)
        yb, yg = targets(np.asarray(self.arr["ugi"][i]), np.asarray(self.arr["grp"][i]), np.asarray(self.arr["edge"][i]),
                         ok, self.buffer.get(i), self.edge_band)
        if self.augment:
            if seed is None:
                raise ValueError("augmentation needs (index, seed) keys from SeededSampler")
            rng = np.random.default_rng(seed)
            g = rng.uniform(*self.photo["global"]) * rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"],
                                                                 size=(x.shape[0], 1, 1))
            x = x * g.astype(np.float32)
            k = int(rng.integers(4))
            x, yb, yg, ok = np.rot90(x, k, (1, 2)), np.rot90(yb, k), np.rot90(yg, k), np.rot90(ok, k)
            if rng.random() < 0.5:
                x, yb, yg, ok = x[:, :, ::-1], yb[:, ::-1], yg[:, ::-1], ok[:, ::-1]
        x = np.where(ok[None], (x - self.mean) / self.sd, 0.0).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(yb)),
                torch.from_numpy(np.ascontiguousarray(yg)))
