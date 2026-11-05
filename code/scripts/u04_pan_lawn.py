"""Lawn homogeneity from the Pléiades Neo 2022 PAN (CLAUDE.md 13.1, 13.2): read from the delivered zips with GDAL /vsizip/
(read-only, nothing extracted), texture at the PAN resolution, one average warp to GRID_1M, only the 1 m layers stored.

  * PAN tiles: those intersecting the common extent (docs/evidence_phase2/texture_screen/pan_extraction_size.json). Tile
    geotransform from its TFW (the DIMAP convention, as A6). The tiles of a strip lie on one lattice and are read as one
    strip mosaic (no tile seams), translated by the strip's U0.3 registration displacement
    (derived/pneo2022_registered/registration.json, applied_displacements), so PAN follows the registered MS geometry.
  * Values: TOA reflectance = DN / 10000 (DIMAP GAIN 10000, BIAS 0); DN 0 (NODATA) and 32767 (SATURATED) are invalid.
  * Layers (windows of 3 and 9 PAN pixels, centred):
      var3, var9   local variance of reflectance, stored in (reflectance x 100)^2, i.e. (percent reflectance)^2; valid
                   where every pixel of the window is valid;
      hom3, hom9   GLCM homogeneity: reflectance quantised to 32 levels over [0, 0.4] (step 0.0125), pixel pairs at offsets
                   (0, 1) and (1, 0) whose first pixel lies in the window; homogeneity = mean over those pairs of
                   1 / (1 + (i - j)^2); valid where every one of those pairs has two valid pixels.
    A PAN pixel keeps the four layers only where all four are valid.
  * Strip rows are processed in bands of 2,048 (core) with an 8-row margin on each side (the layers of the margin rows are
    exact, computed with a further 5-row halo). Each band, core plus margins, is warped to GRID_1M by averaging
    (rasterio.warp.reproject, EPSG:4326 -> EPSG:32720, GDAL's approximate transformer with its default error threshold of
    0.125 source pixel, i.e. under 4 cm; layers with NaN nodata, and the 0/1 validity for the valid fraction), and a 1 m pixel is taken from the one band whose core holds the PAN row of its centre. Its whole footprint
    (under 2.5 PAN rows from the centre) lies inside that band's core plus margin, so every 1 m value is the area-weighted
    mean of the valid PAN pixels of its full footprint (located to the transformer's threshold), computed once. A 1 m pixel keeps a value where the valid fraction
    is >= 0.5.
  * Strips: each GRID_1M pixel takes the strip of the registered mosaic (pneo2022_strip_id_grid1m.tif).
  * Vegetated pixels only: pixels with registered Neo NDVI < 0.3 (derived/features/neo2022_indices_grid1m.tif) are nodata.
  * Output: derived/features/neo2022_pan_lawn_grid1m.tif, COG, 4 bands float16 (NBITS=16), nodata NaN, deflate; written
    through an in-memory copy (no temporary file).
  --selftest   layers against brute force; banded layers against whole-array layers; banded warp against a single warp
               (synthetic data) -> docs/evidence_phase2/texture_screen/pan_lawn_selftest.json
  --strip S    one strip on its GRID_1M bounding box -> derived/features/pan_lawn_strips/neo2022_pan_lawn_S.tif (COG) + log;
               the four strips run as parallel processes.
  --merge      the four strip layers placed on GRID_1M -> derived/features/neo2022_pan_lawn_grid1m.tif (+ _log.json).
  History: v1 single process (archive/2026-11-02/u04_pan_lawn_v1_single_process.py); v2 per tile with a 4-row halo and
  per-band accumulation (archive/2026-11-02/pan_lawn_v2_pair_mask_defect/: the homogeneity validity window was one pixel
  off its pairs, the halo one row short for hom9, and seam pixels mixed their parts by equal weight; outputs archived).
  v3 as run on 2 Nov 2026 also passed tolerance=0 to reproject; rasterio 1.4.4 ignores that keyword, so the outputs are
  those of the default transformer described above (argument removed on 3 Nov, no change to the results).
Run from work_ms/.
"""
import argparse
import glob
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import uniform_filter

WM = Path(__file__).resolve().parents[1]
ROOT = WM.parent
LIST = ROOT / "docs/evidence_phase2/texture_screen/pan_extraction_size.json"
REG = WM / "derived/pneo2022_registered/registration.json"
SID = WM / "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"
NDVI = WM / "derived/features/neo2022_indices_grid1m.tif"
OUT = WM / "derived/features/neo2022_pan_lawn_grid1m.tif"
SELFTEST = ROOT / "docs/evidence_phase2/texture_screen/pan_lawn_selftest.json"
CODE = {"1_2": 2, "1_3": 3, "1_5": 5, "1_6": 6}
LAT0 = -31.4
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))
M_PER_DEG_LAT = 110574.0
BAND, MARGIN, HALO, QSTEP, QLEV = 2048, 8, 5, 0.0125, 32
NAMES = ["var3 (percent reflectance)^2", "var9 (percent reflectance)^2", "GLCM homogeneity 3 x 3", "GLCM homogeneity 9 x 9"]

with rasterio.open(SID) as _s:
    G_T, G_W, G_H, G_CRS = _s.transform, _s.width, _s.height, _s.crs


def outer_window(xmin, ymin, xmax, ymax):
    """GRID_1M window covering the bounds: offsets floored, far edges ceiled."""
    w = from_bounds(xmin, ymin, xmax, ymax, transform=G_T)
    c0, r0 = math.floor(w.col_off), math.floor(w.row_off)
    c1, r1 = math.ceil(w.col_off + w.width), math.ceil(w.row_off + w.height)
    return Window(c0, r0, c1 - c0, r1 - r0)


def vsi(t):
    return f"/vsizip/{Path(t['zip']).resolve().as_posix()}/{t['member']}"


def tiles_of(strip):
    d = json.loads(LIST.read_text(encoding="utf-8"))["strips"][strip]
    z = [p for p in glob.glob(str(ROOT / "raw_data/pleiades_neo/*.zip")) if d["zip"] in p][0]
    names = zipfile.ZipFile(z).namelist()
    out = []
    for t in d["tiles"]:
        if not t["intersects_common_extent"]:
            continue
        mem = [n for n in names if n.endswith(t["member"])][0]
        tfw = zipfile.ZipFile(z).read(mem[:-4] + ".TFW").decode().split()
        a, dd, b, e, c, f = (float(v) for v in tfw)
        assert dd == 0 and b == 0
        out.append({"zip": z, "member": mem, "transform": Affine(a, 0.0, c - a / 2, 0.0, e, f - e / 2)})
    return out


def strip_lattice(tiles, shift):
    """Places the tiles on one strip lattice. Returns (strip transform with the registration shift, rows, cols)."""
    a, e = tiles[0]["transform"].a, tiles[0]["transform"].e
    c0 = min(t["transform"].c for t in tiles)
    f0 = max(t["transform"].f for t in tiles)
    rects = []
    for t in tiles:
        T = t["transform"]
        assert abs(T.a - a) < 1e-15 and abs(T.e - e) < 1e-15, "tiles of a strip must share the pixel size"
        co, ro = (T.c - c0) / a, (T.f - f0) / e
        assert abs(co - round(co)) < 1e-3 and abs(ro - round(ro)) < 1e-3, (t["member"], co, ro)
        with rasterio.open(vsi(t)) as s:
            t["W"], t["H"] = s.width, s.height
        t["col_off"], t["row_off"] = round(co), round(ro)
        rects.append((t["row_off"], t["row_off"] + t["H"], t["col_off"], t["col_off"] + t["W"]))
    for i, p in enumerate(rects):
        for q in rects[i + 1:]:
            assert p[1] <= q[0] or q[1] <= p[0] or p[3] <= q[2] or q[3] <= p[2], "tiles overlap"
    Hs = max(t["row_off"] + t["H"] for t in tiles)
    Ws = max(t["col_off"] + t["W"] for t in tiles)
    dx, dy = shift
    return Affine(a, 0.0, c0 - dx / M_PER_DEG_LON, 0.0, e, f0 - dy / M_PER_DEG_LAT), Hs, Ws


def tile_reader(tiles):
    """read_rows(ra, rb) -> reflectance (rb - ra, Ws) float32 of the strip mosaic, NaN where invalid or not covered."""
    srcs = [rasterio.open(vsi(t)) for t in tiles]
    Ws = max(t["col_off"] + t["W"] for t in tiles)

    def read_rows(ra, rb):
        x = np.full((rb - ra, Ws), np.nan, np.float32)
        for t, s in zip(tiles, srcs):
            a0, b0 = max(ra, t["row_off"]), min(rb, t["row_off"] + t["H"])
            if b0 <= a0:
                continue
            dn = s.read(1, window=Window(0, a0 - t["row_off"], t["W"], b0 - a0)).astype(np.float32)
            x[a0 - ra:b0 - ra, t["col_off"]:t["col_off"] + t["W"]] = np.where((dn > 0) & (dn < 32767), dn / 1e4, np.nan)
        return x
    return read_rows, srcs


def layers(x):
    """x: reflectance (rows, cols) float32 with NaN where invalid -> 4 layers (NaN where invalid, see the docstring)."""
    ok = np.isfinite(x)
    xf = np.where(ok, x, 0.0).astype(np.float32)
    q = np.clip(np.floor(xf / QSTEP), 0, QLEV - 1).astype(np.float32)
    hh = np.zeros(x.shape, np.float32)
    hv = np.zeros(x.shape, np.float32)
    okh = np.zeros(x.shape, np.float32)
    okv = np.zeros(x.shape, np.float32)
    hh[:, :-1] = 1.0 / (1.0 + (q[:, 1:] - q[:, :-1]) ** 2)       # pair (i, j)-(i, j + 1), stored at its first pixel
    hv[:-1, :] = 1.0 / (1.0 + (q[1:, :] - q[:-1, :]) ** 2)       # pair (i, j)-(i + 1, j)
    okh[:, :-1] = ok[:, :-1] & ok[:, 1:]
    okv[:-1, :] = ok[:-1, :] & ok[1:, :]
    okf = ok.astype(np.float32)
    out = []
    for w in (3, 9):
        full = uniform_filter(okf, w, mode="constant") > 1 - 1e-6
        m1 = uniform_filter(xf, w, mode="constant")
        m2 = uniform_filter(xf * xf, w, mode="constant")
        v = np.maximum(m2 - m1 * m1, 0) * 1e4
        out.append(np.where(full, v, np.nan))
    for w in (3, 9):
        full = (uniform_filter(okh, w, mode="constant") > 1 - 1e-6) & (uniform_filter(okv, w, mode="constant") > 1 - 1e-6)
        h = 0.5 * (uniform_filter(hh, w, mode="constant") + uniform_filter(hv, w, mode="constant"))
        out.append(np.where(full, h, np.nan))
    L = np.stack(out).astype(np.float32)
    L[:, ~np.isfinite(L).all(0)] = np.nan
    return L


def grid_layers(read_rows, Ts, Hs, Ws, bw, band=BAND, threads=8):
    """Banded computation and warp of one strip mosaic onto the GRID_1M window bw.
    Returns (val [4, h, w] NaN where unassigned, frac [h, w], times_assigned [h, w])."""
    to_ll = Transformer.from_crs(G_CRS, "EPSG:4326", always_xy=True)
    to_g = Transformer.from_crs("EPSG:4326", G_CRS, always_xy=True)
    h, w = int(bw.height), int(bw.width)
    val = np.full((4, h, w), np.nan, np.float32)
    frac = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.uint8)
    for r0 in range(0, Hs, band):
        r1 = min(r0 + band, Hs)
        sa, sb = max(r0 - MARGIN, 0), min(r1 + MARGIN, Hs)
        ra, rb = max(sa - HALO, 0), min(sb + HALO, Hs)
        L = layers(read_rows(ra, rb))[:, sa - ra:sb - ra]
        Tb = Ts * Affine.translation(0, sa)
        # GRID_1M window around the core rows' footprint, within bw
        lon = [Ts.c, Ts.c + Ws * Ts.a, Ts.c, Ts.c + Ws * Ts.a]
        lat = [Ts.f + r0 * Ts.e, Ts.f + r0 * Ts.e, Ts.f + r1 * Ts.e, Ts.f + r1 * Ts.e]
        X, Y = to_g.transform(lon, lat)
        win = outer_window(min(X), min(Y), max(X), max(Y))
        c0, rr0 = max(int(win.col_off) - 2, int(bw.col_off)), max(int(win.row_off) - 2, int(bw.row_off))
        c1 = min(int(win.col_off + win.width) + 2, int(bw.col_off + bw.width))
        rr1 = min(int(win.row_off + win.height) + 2, int(bw.row_off + bw.height))
        if c1 <= c0 or rr1 <= rr0:
            continue
        dT = G_T * Affine.translation(c0, rr0)
        dst = np.full((4, rr1 - rr0, c1 - c0), np.nan, np.float32)
        reproject(L, dst, src_transform=Tb, src_crs="EPSG:4326", dst_transform=dT, dst_crs=G_CRS,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan, num_threads=threads)
        fr = np.zeros((rr1 - rr0, c1 - c0), np.float32)
        reproject(np.isfinite(L[0]).astype(np.float32), fr, src_transform=Tb, src_crs="EPSG:4326", dst_transform=dT,
                  dst_crs=G_CRS, resampling=Resampling.average, num_threads=threads)
        # assign each 1 m pixel to the band whose core holds the PAN row (and column) of its centre
        cc, rr = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(rr0, rr1) + 0.5)
        lo, la = to_ll.transform(G_T.c + cc * G_T.a, G_T.f + rr * G_T.e)
        prow = (np.asarray(la) - Ts.f) / Ts.e
        pcol = (np.asarray(lo) - Ts.c) / Ts.a
        sel = (prow >= r0) & (prow < r1) & (pcol >= 0) & (pcol < Ws)
        sl = (slice(rr0 - int(bw.row_off), rr1 - int(bw.row_off)), slice(c0 - int(bw.col_off), c1 - int(bw.col_off)))
        v_sub, f_sub, n_sub = val[:, sl[0], sl[1]], frac[sl], count[sl]
        v_sub[:, sel] = dst[:, sel]
        f_sub[sel] = fr[sel]
        n_sub[sel] += 1
    return val, frac, count


def strip_bbox(Ts, Hs, Ws):
    to_g = Transformer.from_crs("EPSG:4326", G_CRS, always_xy=True)
    xs, ys = [], []
    for cc, rr in ((0, 0), (Ws, 0), (0, Hs), (Ws, Hs)):
        X, Y = to_g.transform(*(Ts * (cc, rr)))
        xs.append(X)
        ys.append(Y)
    w = outer_window(min(xs), min(ys), max(xs), max(ys))
    c0, r0 = max(int(w.col_off) - 2, 0), max(int(w.row_off) - 2, 0)
    c1, r1 = min(int(w.col_off + w.width) + 2, G_W), min(int(w.row_off + w.height) + 2, G_H)
    return Window(c0, r0, c1 - c0, r1 - r0)


def write_cog(path, arr, transform):
    free = shutil.disk_usage(path.parent).free
    assert free > 10e9, f"only {free / 1e9:.1f} GB free"
    prof = dict(driver="COG", width=arr.shape[2], height=arr.shape[1], count=4, dtype="float32", nodata=np.nan, crs=G_CRS,
                transform=transform, compress="DEFLATE", NBITS=16, BLOCKSIZE=512, BIGTIFF="YES")
    with rasterio.open(path, "w", **prof) as d:      # copy-only driver: in-memory dataset, no temporary file
        d.write(arr)
        for i, n in enumerate(NAMES, 1):
            d.set_band_description(i, n)
    with rasterio.open(path) as chk:
        return {"bytes": path.stat().st_size, "nbits": chk.tags(1, ns="IMAGE_STRUCTURE").get("NBITS"),
                "layout": chk.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"), "overviews": len(chk.overviews(1)),
                "free_bytes_before_write": free}


def run_strip(strip, out_dir):
    """One strip: its PAN mosaic on its GRID_1M bounding box; values kept where the registered mosaic uses this strip,
    the valid fraction is >= 0.5 and NDVI >= 0.3. Written as a COG on the bounding box."""
    t0 = time.time()
    reg = json.loads(REG.read_text(encoding="utf-8"))["applied_displacements"]
    tiles = tiles_of(strip)
    shift = (reg[strip]["dx_east_m"], reg[strip]["dy_north_m"])
    Ts, Hs, Ws = strip_lattice(tiles, shift)
    bw = strip_bbox(Ts, Hs, Ws)
    read_rows, srcs = tile_reader(tiles)
    val, frac, count = grid_layers(read_rows, Ts, Hs, Ws, bw)
    for s in srcs:
        s.close()
    assert count.max() <= 1, "a 1 m pixel was assigned by two bands"
    with rasterio.open(SID) as s_:
        sid = s_.read(1, window=bw)
    with rasterio.open(NDVI) as s_:
        nd = s_.read(1, window=bw)
        veg = (nd != s_.nodata) & (nd >= 3000)
    use = (sid == CODE[strip]) & (frac >= 0.5) & veg & np.isfinite(val).all(0)
    val = np.where(use[None], val, np.nan).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"neo2022_pan_lawn_{strip}.tif"
    chk = write_cog(path, val, G_T * Affine.translation(int(bw.col_off), int(bw.row_off)))
    info = {"strip": strip, "tiles": [t["member"] for t in tiles], "strip_lattice_rows_cols": [Hs, Ws],
            "pan_pixels": int(sum(t["H"] * t["W"] for t in tiles)), "shift_m_east_north": shift,
            "window_grid1m": [int(bw.col_off), int(bw.row_off), int(bw.width), int(bw.height)],
            "strip_vegetated_pixels": int(((sid == CODE[strip]) & veg).sum()), "pixels_with_values": int(use.sum()),
            "output": str(path.relative_to(WM)), **chk, "seconds": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / f"neo2022_pan_lawn_{strip}_log.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info), flush=True)


def merge(strip_dir, out_path, log_path):
    """The per-strip layers placed on GRID_1M (each pixel holds values from one strip only, by construction)."""
    t0 = time.time()
    final = np.full((4, G_H, G_W), np.nan, np.float32)
    logs = {}
    for strip in CODE:
        with rasterio.open(strip_dir / f"neo2022_pan_lawn_{strip}.tif") as s_:
            c0 = round((s_.transform.c - G_T.c) / G_T.a)
            r0 = round((s_.transform.f - G_T.f) / G_T.e)
            a = s_.read()
        sub = final[:, r0:r0 + a.shape[1], c0:c0 + a.shape[2]]
        m = np.isfinite(a[0])
        assert not np.isfinite(sub[0][m]).any(), f"strip {strip} overlaps another strip's pixels"
        sub[:, m] = a[:, m]
        logs[strip] = json.loads((strip_dir / f"neo2022_pan_lawn_{strip}_log.json").read_text(encoding="utf-8"))
    chk = write_cog(out_path, final, G_T)
    with rasterio.open(NDVI) as s_:
        nd = s_.read(1)
        ndv = s_.nodata
    veg = (nd != ndv) & (nd >= 3000)
    info = {"output": str(out_path.relative_to(WM)), **chk, "vegetated_pixels": int(veg.sum()),
            "pixels_with_values": int(np.isfinite(final[0]).sum()), "strips": logs, "seconds_merge": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(log_path).write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "strips"}, indent=1))


def brute_layers(x):
    """Direct per-pixel definition of the four layers (slow; self-test only)."""
    H, W = x.shape
    ok = np.isfinite(x)
    q = np.clip(np.floor(np.where(ok, x, 0) / QSTEP), 0, QLEV - 1)
    out = np.full((4, H, W), np.nan)
    for i in range(H):
        for j in range(W):
            for k, w in enumerate((3, 9)):
                r = w // 2
                if i - r < 0 or j - r < 0 or i + r >= H or j + r >= W:
                    continue
                win = x[i - r:i + r + 1, j - r:j + r + 1]
                if np.isfinite(win).all():
                    out[k, i, j] = win.astype(np.float64).var() * 1e4
                vals, good = [], True
                for a in range(i - r, i + r + 1):
                    for b in range(j - r, j + r + 1):
                        for a2, b2 in ((a, b + 1), (a + 1, b)):
                            if not (0 <= a < H and 0 <= b < W and 0 <= a2 < H and 0 <= b2 < W and ok[a, b] and ok[a2, b2]):
                                good = False
                                break
                            vals.append(1.0 / (1.0 + (q[a, b] - q[a2, b2]) ** 2))
                        if not good:
                            break
                    if not good:
                        break
                if good:
                    out[2 + k, i, j] = np.mean(vals)
    out[:, ~np.isfinite(out).all(0)] = np.nan
    return out


def selftest():
    rng = np.random.default_rng(0)
    res = {}
    # 1. layers against the direct definition, with invalid pixels inside and at the edges
    x = (0.05 + 0.25 * rng.random((48, 52))).astype(np.float32)
    x[20, 20] = np.nan
    x[5:7, 40] = np.nan
    x[30:33, 10:12] = np.nan
    x[:, 51] = np.nan
    a, b = layers(x), brute_layers(x)
    same_mask = bool((np.isfinite(a) == np.isfinite(b)).all())
    m = np.isfinite(a)
    res["layers_vs_direct"] = {"same_valid_mask": same_mask, "valid_pixels": int(m[0].sum()),
                               "max_abs_diff": [float(np.abs(a[k][m[k]] - b[k][m[k]]).max()) for k in range(4)],
                               "max_rel_diff_var": [float((np.abs(a[k][m[k]] - b[k][m[k]]) / np.maximum(b[k][m[k]], 1e-3)).max()) for k in (0, 1)]}
    # 2. banded layers (band 64, margin 8, halo 5) stitched against layers of the whole array
    big = np.clip(0.15 + 0.08 * rng.standard_normal((400, 90)), 0, 0.6).astype(np.float32)
    big[rng.random(big.shape) < 0.01] = np.nan
    whole = layers(big)
    st = np.full_like(whole, np.nan)
    for r0 in range(0, 400, 64):
        r1 = min(r0 + 64, 400)
        sa, sb = max(r0 - MARGIN, 0), min(r1 + MARGIN, 400)
        ra, rb = max(sa - HALO, 0), min(sb + HALO, 400)
        L = layers(big[ra:rb])[:, sa - ra:sb - ra]
        st[:, sa:sb][:, r0 - sa:r1 - sa] = L[:, r0 - sa:r1 - sa]
        assert np.array_equal(np.isfinite(L), np.isfinite(whole[:, sa:sb])), "margin rows must equal the whole-array layers"
        assert np.allclose(L, whole[:, sa:sb], equal_nan=True, atol=1e-6)
    res["banded_layers_vs_whole"] = {"identical_mask": bool(np.array_equal(np.isfinite(st), np.isfinite(whole))),
                                     "max_abs_diff": float(np.nanmax(np.abs(st - whole)))}
    # 3. banded warp (band 64) against one warp of the whole synthetic strip, on GRID_1M
    Hs, Ws = 900, 700
    lon0, lat0 = -64.20, -31.40
    Ts = Affine(2.7e-6, 0, lon0, 0, -2.7e-6, lat0)
    from scipy.ndimage import gaussian_filter
    field = (0.15 + 0.6 * gaussian_filter(rng.standard_normal((Hs, Ws)), 3)).astype(np.float32)
    field[rng.random(field.shape) < 0.002] = np.nan
    field[400:460, 300:380] = np.nan
    bw = strip_bbox(Ts, Hs, Ws)

    def rr(ra, rb):
        return field[ra:rb].copy()
    v1, f1, n1 = grid_layers(rr, Ts, Hs, Ws, bw, band=Hs, threads=2)
    v2, f2, n2 = grid_layers(rr, Ts, Hs, Ws, bw, band=64, threads=2)
    both = np.isfinite(v1) & np.isfinite(v2)
    keep = (f1 >= 0.5) & (f2 >= 0.5)        # the pixels that keep a value (valid fraction >= 0.5)
    kb = both & keep[None]
    res["banded_warp_vs_single"] = {
        "grid_pixels": int(bw.width * bw.height), "assigned_single": int((n1 == 1).sum()), "assigned_banded": int((n2 == 1).sum()),
        "max_times_assigned": int(n2.max()), "same_assignment": bool(np.array_equal(n1, n2)),
        "same_value_mask": bool(np.array_equal(np.isfinite(v1), np.isfinite(v2))),
        "same_kept_pixels": bool(np.array_equal(f1 >= 0.5, f2 >= 0.5)), "kept_pixels": int(keep.sum()),
        "max_rel_diff_kept": [float((np.abs(v1[k] - v2[k]) / np.maximum(np.abs(v1[k]), 1e-6))[kb[k]].max()) for k in range(4)],
        "max_abs_diff_frac": float(np.abs(f1 - f2).max()),
        "note": "GDAL's average warp differs by float rounding when the source window changes; tolerance is the float16 "
                "storage precision (relative 2^-11)"}
    # 4. the warp is an area-weighted mean: a constant field gives the constant, a step gives the overlap share
    const = np.full((Hs, Ws), 0.2, np.float32)
    vc, fc, _ = grid_layers(lambda a, b: const[a:b].copy(), Ts, Hs, Ws, bw, band=64, threads=2)
    inner = fc > 0.999
    res["constant_field"] = {"var3_max": float(np.nanmax(vc[0][inner])), "hom9_min": float(np.nanmin(vc[3][inner])),
                             "hom9_max": float(np.nanmax(vc[3][inner]))}
    ok = (res["layers_vs_direct"]["same_valid_mask"] and max(res["layers_vs_direct"]["max_abs_diff"][2:]) < 1e-5
          and max(res["layers_vs_direct"]["max_rel_diff_var"]) < 1e-3
          and res["banded_layers_vs_whole"]["identical_mask"] and res["banded_layers_vs_whole"]["max_abs_diff"] < 1e-6
          and res["banded_warp_vs_single"]["same_assignment"] and res["banded_warp_vs_single"]["max_times_assigned"] == 1
          and res["banded_warp_vs_single"]["same_value_mask"] and res["banded_warp_vs_single"]["same_kept_pixels"]
          and max(res["banded_warp_vs_single"]["max_rel_diff_kept"]) < 2 ** -11
          and res["banded_warp_vs_single"]["max_abs_diff_frac"] < 1e-3
          and res["constant_field"]["var3_max"] < 1e-6 and abs(res["constant_field"]["hom9_min"] - 1) < 1e-6)
    res["passed"] = bool(ok)
    SELFTEST.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--strip", choices=list(CODE))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-dir", default=str(WM / "derived/features/pan_lawn_strips"))
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.strip:
        run_strip(a.strip, Path(a.out_dir))
    if a.merge:
        merge(Path(a.out_dir), OUT, WM / "derived/features/neo2022_pan_lawn_log.json")
