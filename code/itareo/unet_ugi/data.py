"""Tiles, spatial folds, internal-validation blocks, the boundary band and the tile cache.

Conventions
  * image: 4 bands (R, G, B, NIR), reflectance x 10000 as uint16, `nodata` in any band = invalid pixel;
  * labels: uint8, 0 = no label, classes 1..K; the network sees classes 0..K-1 and IGNORE (255);
  * all rasters on one grid (same transform and shape); tiles of `tile` px at stride `stride` from the grid origin.
Folds: two lines in map coordinates (x_line, y_line) cut the grid into quadrants; a tile belongs to the quadrant of its
centre, with the rule left & south -> 1, right & south -> 2, left & north -> 3, right & north -> 4; tiles a line passes
through are dropped.
Internal validation (for checkpoint selection and normalisation, never the held-out fold): square blocks of
`block_m` aligned to the tile lattice, drawn at random among blocks of training-fold tiles; training pixels within
`buffer_m` of a validation block are ignored in the loss.
Boundary band: labelled pixels with a labelled pixel of another class among their 8 neighbours (1 px band); computed
with a 1 px margin around every tile, so it does not depend on the tiling.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy.ndimage import maximum_filter, minimum_filter

IGNORE = 255


def boundary_band(lab, nodata=0):
    """lab: 2-D labels with a 1 px margin all round (margin pixels may be nodata). Returns the band for the interior."""
    valid = lab != nodata
    mx = maximum_filter(np.where(valid, lab, 0), size=3, mode="constant", cval=0)
    mn = minimum_filter(np.where(valid, lab, 255), size=3, mode="constant", cval=255)
    band = valid & (mx != mn)
    return band[1:-1, 1:-1]


def read_padded(src, window, pad, fill, bands=None):
    w = Window(window.col_off - pad, window.row_off - pad, window.width + 2 * pad, window.height + 2 * pad)
    return src.read(bands, window=w, boundless=True, fill_value=fill)


def fold_of(cx, cy, x_line, y_line):
    left, south = cx <= x_line, cy <= y_line
    return np.where(left & south, 1, np.where(~left & south, 2, np.where(left & ~south, 3, 4)))


@dataclass
class Paths:
    image: str
    labels: str
    strips: str = None


def tile_index(paths: Paths, tile, stride, x_line, y_line, n_classes, min_share=0.0):
    """One row per tile position: labelled-and-valid share, fold, straddle flag, majority strip, class and band counts."""
    rows = []
    with rasterio.open(paths.image) as im, rasterio.open(paths.labels) as lb:
        assert im.transform == lb.transform and im.shape == lb.shape, "image and labels must share one grid"
        sp = rasterio.open(paths.strips) if paths.strips else None
        T = im.transform
        for r0 in range(0, im.height - tile + 1, stride):
            for c0 in range(0, im.width - tile + 1, stride):
                w = Window(c0, r0, tile, tile)
                labp = read_padded(lb, w, 1, 0, 1)
                lab = labp[1:-1, 1:-1]
                if not (lab > 0).any():
                    continue
                img_ok = (im.read(window=w) != im.nodata).all(0)
                ok = img_ok & (lab > 0)
                share = float(ok.mean())
                if share < min_share:
                    continue
                band = boundary_band(labp) & ok
                x0, y0 = T * (c0, r0)
                x1, y1 = T * (c0 + tile, r0 + tile)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                straddle = (min(x0, x1) < x_line < max(x0, x1)) or (min(y0, y1) < y_line < max(y0, y1))
                r = {"row0": r0, "col0": c0, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "share": round(share, 5),
                     "fold": int(fold_of(cx, cy, x_line, y_line)), "straddle": bool(straddle),
                     "band_px": int(band.sum()), "labelled_valid_px": int(ok.sum())}
                cnt = np.bincount(lab[ok], minlength=n_classes + 1)
                for k in range(1, n_classes + 1):
                    r[f"c{k}"] = int(cnt[k])
                if sp is not None:
                    s = sp.read(1, window=w)[ok]
                    ids, cn = np.unique(s[s > 0], return_counts=True)
                    r["strip"] = int(ids[cn.argmax()]) if len(ids) else 0
                    r["strip_share"] = round(float(cn.max() / ok.sum()), 4) if len(ids) else 0.0
                rows.append(r)
        if sp is not None:
            sp.close()
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({k: (v == "True" if v in ("True", "False") else float(v) if "." in v or "e" in v else int(v))
                    for k, v in r.items()})
    return out


def internal_blocks(tiles, held_out, block_m, frac, seed, res, n_classes, min_tiles=4, tries=200):
    """Validation blocks for one held-out fold: blocks of block_m metres from the grid origin (res = pixel size in
    metres; block_m should be a multiple of the tile size in metres so that blocks follow the tile lattice), with at
    least `min_tiles` training tiles; a fraction `frac` drawn at random (seed); redrawn (seed + 1, ...) until every class
    has labelled pixels in the validation tiles. Returns (list of block rectangles in map units, seed used)."""
    train = [t for t in tiles if t["fold"] != held_out]
    blocks = {}
    for t in train:
        key = (int(t["row0"] * res // block_m), int(t["col0"] * res // block_m))
        blocks.setdefault(key, []).append(t)
    keys = sorted(k for k, v in blocks.items() if len(v) >= min_tiles)
    n_val = max(1, int(round(frac * len(keys))))
    for k in range(tries):
        rng = np.random.default_rng(seed + k)
        pick = [keys[i] for i in sorted(rng.choice(len(keys), n_val, replace=False))]
        cnt = np.zeros(n_classes)
        for key in pick:
            for t in blocks[key]:
                cnt += [t[f"c{c}"] for c in range(1, n_classes + 1)]
        if (cnt > 0).all():
            rects = []
            for key in pick:
                ts = blocks[key]
                rects.append({"block": list(key), "x0": min(min(t["x0"], t["x1"]) for t in ts),
                              "x1": max(max(t["x0"], t["x1"]) for t in ts), "y0": min(min(t["y0"], t["y1"]) for t in ts),
                              "y1": max(max(t["y0"], t["y1"]) for t in ts), "tiles": len(ts)})
            return rects, seed + k
    raise RuntimeError("no draw with every class in the internal validation")


def in_rects(t, rects):
    cx, cy = (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2
    return any(r["x0"] <= cx <= r["x1"] and r["y0"] <= cy <= r["y1"] for r in rects)


def buffer_mask(t, rects, buffer_m, tile, res):
    """Boolean (tile, tile): pixels of training tile t within buffer_m of any validation rectangle."""
    xs = min(t["x0"], t["x1"]) + (np.arange(tile) + 0.5) * res
    ys = max(t["y0"], t["y1"]) - (np.arange(tile) + 0.5) * res
    m = np.zeros((tile, tile), bool)
    for r in rects:
        dx = np.maximum(np.maximum(r["x0"] - xs, xs - r["x1"]), 0)
        dy = np.maximum(np.maximum(r["y0"] - ys, ys - r["y1"]), 0)
        m |= (dy[:, None] ** 2 + dx[None, :] ** 2) <= buffer_m ** 2
    return m


def build_cache(paths: Paths, tiles, out_dir):
    """Tile arrays as .npy (memory-mappable): img (N, 4, T, T) uint16, lab (N, T, T) uint8 (0 = no label or invalid
    image), band (N, T, T) uint8, strip (N, T, T) uint8 when a strip raster is given. Row i = tiles[i]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(paths.image) as im, rasterio.open(paths.labels) as lb:
        tile =int(round(abs(tiles[0]["x1"] - tiles[0]["x0"]) / im.transform.a))
        N = len(tiles)
        img = np.lib.format.open_memmap(out / "img.npy", "w+", np.uint16, (N, im.count, tile, tile))
        lab = np.lib.format.open_memmap(out / "lab.npy", "w+", np.uint8, (N, tile, tile))
        band = np.lib.format.open_memmap(out / "band.npy", "w+", np.uint8, (N, tile, tile))
        sp = rasterio.open(paths.strips) if paths.strips else None
        strip = np.lib.format.open_memmap(out / "strip.npy", "w+", np.uint8, (N, tile, tile)) if sp else None
        for i, t in enumerate(tiles):
            w = Window(int(t["col0"]), int(t["row0"]), tile, tile)
            a = im.read(window=w)
            labp = read_padded(lb, w, 1, 0, 1)
            ok = (a != im.nodata).all(0)
            l_ = np.where(ok, labp[1:-1, 1:-1], 0).astype(np.uint8)
            img[i] = a
            lab[i] = l_
            band[i] = boundary_band(labp) & (l_ > 0)
            if sp:
                strip[i] = sp.read(1, window=w)
        for a in (img, lab, band) + ((strip,) if sp else ()):
            a.flush()
        if sp:
            sp.close()
        nodata = im.nodata
    json.dump({"n": N, "tile": tile, "nodata": nodata, "image": str(paths.image), "labels": str(paths.labels)},
              open(out / "cache.json", "w"), indent=1)


def band_stats(cache_dir, idx, nodata):
    """Per-band mean and SD (reflectance x 10000) over valid pixels of tiles idx."""
    img = np.load(Path(cache_dir) / "img.npy", mmap_mode="r")
    s = np.zeros(img.shape[1])
    ss = np.zeros(img.shape[1])
    n = 0
    for i in idx:
        a = np.asarray(img[i], np.float64)
        ok = (img[i] != nodata).all(0)
        v = a[:, ok]
        s += v.sum(1)
        ss += (v * v).sum(1)
        n += v.shape[1]
    m = s / n
    return m, np.sqrt(ss / n - m * m)


def class_counts(cache_dir, idx, n_classes, ignore_band=False, buffer_masks=None):
    """Labelled pixels per class over tiles idx, counting only the pixels that enter the loss: without the boundary
    band when ignore_band, and without the buffer pixels (buffer_masks: tile index -> boolean mask)."""
    lab = np.load(Path(cache_dir) / "lab.npy", mmap_mode="r")
    band = np.load(Path(cache_dir) / "band.npy", mmap_mode="r")
    buffer_masks = buffer_masks or {}
    cnt = np.zeros(n_classes, np.int64)
    present = np.zeros(n_classes, np.int64)
    for i in idx:
        lb = np.asarray(lab[i])
        use = lb > 0
        if ignore_band:
            use &= np.asarray(band[i]) == 0
        if i in buffer_masks:
            use &= ~buffer_masks[i]
        c = np.bincount(lb[use].ravel(), minlength=n_classes + 1)[1:]
        cnt += c
        present += (c > 0) * c.sum()
    return cnt, present


def median_frequency_weights(counts, present):
    """Median-frequency balancing (Eigen and Fergus 2015): freq_c = pixels of c / pixels of the tiles containing c;
    w_c = median(freq) / freq_c over the classes present; absent classes get weight 0."""
    counts, present = np.asarray(counts, float), np.asarray(present, float)
    ok = (counts > 0) & (present > 0)
    freq = np.where(ok, counts / np.maximum(present, 1), np.nan)
    med = np.nanmedian(freq)
    return np.where(ok, med / freq, 0.0)


class SeededSampler(torch.utils.data.Sampler):
    """Shuffled order plus one augmentation seed per sample, both drawn in the main process from `generator`, so that
    the augmentation of every sample depends on the seed only, not on the number of DataLoader workers. Yields
    (index, seed) pairs, which TileDataset accepts."""

    def __init__(self, n, generator):
        self.n, self.g = n, generator

    def __len__(self):
        return self.n

    def __iter__(self):
        perm = torch.randperm(self.n, generator=self.g).tolist()
        seeds = torch.randint(0, 2 ** 31 - 1, (self.n,), generator=self.g).tolist()
        return iter(zip(perm, seeds))


class TileDataset(torch.utils.data.Dataset):
    """Training or validation tiles from the cache. Returns x (4, T, T) float32 normalised, y (T, T) int64 with IGNORE
    for no label, invalid image, the boundary band (if ignore_band) and the buffer around validation blocks.
    With augmentation the item key is (position, seed) from SeededSampler; the seed alone drives the augmentation."""

    def __init__(self, cache_dir, idx, mean, sd, nodata, ignore_band, augment, buffer_masks=None, photometric=None):
        self.cache, self.idx = str(cache_dir), list(idx)
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.sd = np.asarray(sd, np.float32)[:, None, None]
        self.nodata, self.ignore_band, self.augment = nodata, ignore_band, augment
        self.buffer = buffer_masks or {}
        self.photo = photometric or {"global": [0.9, 1.1], "per_band": 0.03}
        self.arr = None

    def _open(self):
        c = Path(self.cache)
        self.arr = {k: np.load(c / f"{k}.npy", mmap_mode="r") for k in ("img", "lab", "band")}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        if self.arr is None:
            self._open()
        j, aug_seed = key if isinstance(key, tuple) else (key, None)
        i = self.idx[j]
        raw = np.asarray(self.arr["img"][i])
        ok = (raw != self.nodata).all(0)
        x = raw.astype(np.float32)
        lab = np.asarray(self.arr["lab"][i])
        y = np.where(lab > 0, lab.astype(np.int64) - 1, IGNORE)
        if self.ignore_band:
            y[np.asarray(self.arr["band"][i]) > 0] = IGNORE
        if i in self.buffer:
            y[self.buffer[i]] = IGNORE
        if self.augment:
            if aug_seed is None:
                raise ValueError("augmentation needs (index, seed) keys from SeededSampler")
            rng = np.random.default_rng(aug_seed)
            g = rng.uniform(*self.photo["global"]) * rng.uniform(1 - self.photo["per_band"], 1 + self.photo["per_band"],
                                                                 size=(x.shape[0], 1, 1))
            x = x * g.astype(np.float32)
            k = int(rng.integers(4))
            x, y, ok = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(ok, k)
            if rng.random() < 0.5:
                x, y, ok = x[:, :, ::-1], y[:, ::-1], ok[:, ::-1]
        x = np.where(ok[None], (x - self.mean) / self.sd, 0.0).astype(np.float32)
        y = np.where(ok, y, IGNORE)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
