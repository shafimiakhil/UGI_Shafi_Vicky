"""Stage A7 (2 Nov 2026): co-registration MEASURED (nothing corrected) by phase correlation on stable surfaces, for the
pairs of Shafi's instruction:
  P1  Pleiades 2017 (A5, GRID_1M)           vs Pleiades Neo Oct 2022 (A6, GRID_1M)               at 1 m
  P2  Pleiades Neo Oct 2022 (A6, 3 x 3 mean) vs PlanetScope 24 Apr 2022                            at 3 m
  P3  PlanetScope 24 Apr 2022                vs PlanetScope 19 Jul 2025 (Luigi's, on COMMON)        at 3 m
  P4  PlanetScope 24 Apr 2022 (area-weighted to the S2 lattice) vs Sentinel-2 2022Q2 SR            at 10 m
  P5  PlanetScope 24 Apr 2022 (averaged to the PRISMA grid) vs PRISMA 27 Apr 2022 L2D (from the he5) at 30 m
  plus P1g and P2g, the same with the Neo version built from the GeoTIFF tags (derived/pneo2022/v1_geotiff_tags), and
  P6, Neo strip against Neo strip in their overlaps (1_2/1_5, 1_2/1_3, 1_2/1_6, 1_3/1_6) under both georeferencing
  conventions: the offsets between strips at the mosaic seams, and a consistency check of the two conventions.
Second version, 2 Nov 2026, after the stage-A review (first and second runs in run1_strict/ and run2_first_version/).
Method:
  - one structure band per sensor: the mean of its blue, green and red reflectance (PRISMA: the VNIR bands nearest
    490, 560 and 665 nm); NDVI from its red and NIR bands;
  - P1-P5: the pair grid is tiled into square blocks of about 2 km (1 m: 2000 px; 3 m: 667 px; 10 m: 200 px; 30 m:
    67 px). P6: blocks of 1000 m on a 500 m stride, kept where >= 95 % of the block lies inside the MS-FS ROI masks of
    both strips (inside the extent the overlaps are 1.8-2.7 km wide, so a fixed 2 km tiling misses most of them;
    1_3/1_6 is an along-track overlap 1.8-1.9 km north-south);
  - a block is correlated when >= 90 % of its pixels are valid in both images and >= 20 % are stable: NDVI < 0.25 in
    both images (non-vegetated surfaces; vegetation changes between dates) and, for Neo, outside the cloud mask.
    Invalid pixels of each image are set to that image's mean (the two images' nodata patterns differ). Blocks are not
    masked inside (a mask shared by both images would bias the correlation toward zero shift);
  - strict subset (sensitivity check, flag `strict`): >= 99.9 % valid and >= 50 % stable (the first run of 2 Nov
    required every pixel valid, which isolated invalid pixels of the Pleiades 2017 product prevented in most blocks);
  - phase correlation (itareo.phasecorr): mean removed, 2-D Hann window, sub-pixel peak refined to 1/20 px by the
    matrix-multiply upsampled DFT (Guizar-Sicairos et al. 2008); quality = peak / highest competing peak on a
    frequency-tapered surface, which does not depend on the sub-pixel part of the shift; blocks with quality < 2 are
    kept in blocks.csv but not in the summaries;
  - a used block whose shift lies more than 10 pixels from the median shift of its pair is flagged `outlier` (a false
    correlation peak) and left out of the summaries; it stays in blocks.csv;
  - shift = displacement of the SECOND image relative to the FIRST, in metres east (dx) and north (dy): a feature at
    x in the first image lies at x + shift in the second. The sign is checked by the self-test (--selftest).
Summaries (also rebuilt from blocks.csv alone with --summarise):
  - summary.csv: per pair and criteria, median, MAD and 10th-90th percentiles of dx and dy, median shift length;
  - trend.csv: per pair, a linear trend of the block shifts in east and north position (least squares), its span over
    the blocks and the residual RMS: a non-uniform shift cannot be removed by one translation;
  - convention_test.csv: P1/P1g and P2/P2g, Neo's own displacement regressed on the block's fractional position in the
    FULL strip (u east, v south; from the DIMAP NCOLS, NROWS, ULXMAP, ULYMAP), with one intercept per strip, slope
    standard errors, and the slope each convention predicts (the GeoTIFF tags place a pixel at fraction u of the strip
    u x 1 pixel west and v x 1 pixel north of the DIMAP/TFW position);
  - neo_overlap_test.csv: P6 per strip pair and convention, and the paired difference GeoTIFF minus TFW per block
    against the difference the two conventions imply.
Outputs: docs/evidence_phase2/coregistration/{blocks.csv, summary.csv, trend.csv, convention_test.csv,
neo_overlap_test.csv, prisma_he5_vs_spectral.csv, selftest.json, run.json}; intermediate rasters in
work_ms/derived/coregistration/.
From work_ms/:  python -I scripts/a7_coregistration.py --selftest   then   python -I scripts/a7_coregistration.py
                python -I scripts/a7_coregistration.py --summarise   (summaries from blocks.csv only)
"""
import os

os.environ["GDAL_PAM_ENABLED"] = "NO"

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np                                   # noqa: E402
import rasterio                                      # noqa: E402
from affine import Affine                            # noqa: E402
from pyproj import Geod, Transformer                 # noqa: E402
from rasterio.warp import Resampling, reproject, transform_geom   # noqa: E402
from rasterio.windows import Window                  # noqa: E402
from shapely.geometry import box, mapping, shape     # noqa: E402
from shapely.ops import unary_union                  # noqa: E402

from itareo import EVIDENCE, PROJECT, W              # noqa: E402
from itareo.gee.common import GRID_S2_10M            # noqa: E402
from itareo.grid import COMMON, GRID_1M, Grid        # noqa: E402
from itareo.phasecorr import phase_corr              # noqa: E402
from itareo.phasecorr import selftest as phasecorr_selftest   # noqa: E402

OUT = EVIDENCE / "coregistration"
TMP = W / "derived" / "coregistration"
PLE = W / "derived/pleiades2017/pleiades2017_toa_hpf_grid1m.tif"
NEO_DIR = W / "derived/pneo2022"
NEO = NEO_DIR / "pneo2022_toa_grid1m.tif"
NEO_CLOUD = NEO_DIR / "pneo2022_cloud_grid1m.tif"
NEO_G = NEO_DIR / "v1_geotiff_tags/pneo2022_toa_grid1m.tif"
NEO_G_CLOUD = NEO_DIR / "v1_geotiff_tags/pneo2022_cloud_grid1m.tif"
PS22 = PROJECT / "raw_data/PRISMA/PSScene/20220424_135836_31_2446_3B_AnalyticMS_SR_8b_clip.tif"
PS25 = PROJECT / "work/raw_data/PLANET/20250719_8b_clip_aligned.tif"
S2Q2 = PROJECT / "gee_exports/2026-10-28_nodata/s2_sr_median_2022Q2.tif"
PRISMA_DIR = PROJECT / "raw_data/PRISMA/2PRS_L2D_STD_20220427143228_20220427143232_0001"
HE5 = PRISMA_DIR / "PRS_L2D_STD_20220427143228_20220427143232_0001.he5"
SPECTRAL = PRISMA_DIR / "PRS_L2D_STD_20220427143228_20220427143232_0001_SPECTRAL.tif"
NEO_STRIPS = ("1_2", "1_3", "1_5", "1_6")
NEO_STRIP_CODE = {2: "1_2", 3: "1_3", 5: "1_5", 6: "1_6"}
P6_PAIRS = (("1_2", "1_5"), ("1_2", "1_3"), ("1_2", "1_6"), ("1_3", "1_6"))
NDVI_STABLE = 0.25
VALID_MIN, STABLE_MIN, QUALITY_MIN = 0.9, 0.2, 2.0
STRICT_VALID_MIN, STRICT_STABLE_MIN = 0.999, 0.5
BLOCK_M = 2000.0
P6_BLOCK, P6_STRIDE, P6_FOOT_MIN = 1000, 500, 0.95
MIN_TREND_BLOCKS = 6
NUM = {"block_row", "block_col", "block_px", "res_m", "valid_frac", "invalid_px", "stable_frac", "dx_m", "dy_m", "peak",
       "quality", "neo_u_full", "neo_v_full", "neo_strip_share", "u_full_a", "v_full_a", "u_full_b", "v_full_b", "du", "dv",
       "implied_geotiff_minus_tfw_dx_m", "implied_geotiff_minus_tfw_dy_m"}


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    """itareo.phasecorr self-test (known shifts, sign, quality), written to selftest.json."""
    res = phasecorr_selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / "selftest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "unrelated_pairs")}, indent=1))
    assert res["passed"], "phase correlation self-test failed"
    return res


# ------------------------------------------------------------------------------------------------ loaders
def read_window(path, grid: Grid, r0, c0, h, w, bands, scale, nodata=None):
    """Bands of a raster aligned to `grid` (same CRS, res, lattice) over grid rows r0:r0+h, cols c0:c0+w, as float
    reflectance (value * scale); NaN where nodata or outside the file."""
    with rasterio.open(path) as s:
        t = s.transform
        co = (grid.left - t.c) / t.a
        ro = (t.f - grid.top) / -t.e
        assert abs(t.a - grid.res) < 1e-6 and abs(co - round(co)) < 1e-6 and abs(ro - round(ro)) < 1e-6, f"{path} off {grid}"
        win = Window(c0 + int(round(co)), r0 + int(round(ro)), w, h)
        nd = s.nodata if nodata is None else nodata
        a = s.read(bands, window=win, boundless=True, fill_value=nd if nd is not None else 0).astype(np.float64)
    a[a == nd] = np.nan
    return a * scale


def vis_ndvi(a, iv, ir, inir):
    vis = np.mean(a[iv], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (a[inir] - a[ir]) / (a[inir] + a[ir])
    return vis, ndvi


class Src:
    """One image of a pair on the pair grid: returns (vis, ndvi, extra_invalid) for a block."""

    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def __call__(self, r0, c0, h, w):
        return self.fn(r0, c0, h, w)


def src_ple():
    def f(r0, c0, h, w):
        a = read_window(PLE, GRID_1M, r0, c0, h, w, [1, 2, 3, 4], 1e-4, 65535)
        return (*vis_ndvi(a, [0, 1, 2], 0, 3), np.zeros((h, w), bool))
    return Src("pleiades2017_1m", f)


def src_neo(path, cloud, factor=1):
    def f(r0, c0, h, w):
        a = read_window(path, GRID_1M, r0 * factor, c0 * factor, h * factor, w * factor, [1, 2, 3, 4], 1e-4, 65535)
        cl = read_window(cloud, GRID_1M, r0 * factor, c0 * factor, h * factor, w * factor, [1], 1.0, 255)[0]
        if factor > 1:   # mean of factor x factor blocks; NaN if any member is NaN
            a = a.reshape(4, h, factor, w, factor).mean(axis=(2, 4))
            cl = cl.reshape(h, factor, w, factor).max(axis=(1, 3))
        return (*vis_ndvi(a, [0, 1, 2], 0, 3), ~(cl == 0))
    return Src(f"pneo2022_{'1m' if factor == 1 else '3m'}{'' if path == NEO else '_geotiff_tags'}", f)


def src_ps(path, grid, label):
    def f(r0, c0, h, w):
        a = read_window(path, grid, r0, c0, h, w, [2, 4, 6, 8], 1e-4, 0)
        return (*vis_ndvi(a, [0, 1, 2], 2, 3), np.zeros((h, w), bool))
    return Src(label, f)


def src_s2():
    def f(r0, c0, h, w):
        a = read_window(S2Q2, GRID_S2_10M, r0, c0, h, w, [1, 2, 3, 7], 1e-4, 65535)
        return (*vis_ndvi(a, [0, 1, 2], 2, 3), np.zeros((h, w), bool))
    return Src("s2_2022Q2_10m", f)


# ------------------------------------------------------------------------------------------------ PRISMA from the he5
def prisma_from_he5():
    """VNIR reflectance of 4 bands (nearest 490, 560, 665, 865 nm) from the he5 on a grid fitted to the he5 lat/lon
    of the pixel centres; written to TMP/prisma20220427_he5_b_g_r_nir.tif. Also compares the fitted grid and the
    values with the third-party _SPECTRAL.tif (prisma_he5_vs_spectral.csv)."""
    import re
    with rasterio.open(HE5) as h:
        tags = h.tags()
        sub = {re.sub(r".*://", "", s): s for s in h.subdatasets}
    cw = [float(v) for v in tags["List_Cw_Vnir"].split()]
    vmin, vmax = float(tags["L2ScaleVnirMin"]), float(tags["L2ScaleVnirMax"])
    cube_key = [k for k in sub if k.endswith("PRS_L2D_HCO/Data_Fields/VNIR_Cube")][0]
    lat_key = [k for k in sub if k.endswith("PRS_L2D_HCO/Geolocation_Fields/Latitude")][0]
    lon_key = [k for k in sub if k.endswith("PRS_L2D_HCO/Geolocation_Fields/Longitude")][0]
    with rasterio.open(sub[cube_key]) as s:            # GDAL view: bands = rows, height = spectral, width = cols
        cube = s.read()                                 # (rows, spectral, cols)
    with rasterio.open(sub[lat_key]) as s:
        lat = s.read(1)
    with rasterio.open(sub[lon_key]) as s:
        lon = s.read(1)
    rows, nb, cols = cube.shape
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{tags['Epsg_Code']}", always_xy=True).transform(lon, lat)
    rr, cc = np.mgrid[0:rows, 0:cols]
    A = np.column_stack([np.ones(rr.size), cc.ravel() + 0.5, rr.ravel() + 0.5])
    bx, resx = np.linalg.lstsq(A, x.ravel(), rcond=None)[:2]
    by, resy = np.linalg.lstsq(A, y.ravel(), rcond=None)[:2]
    tr = Affine(bx[1], bx[2], bx[0], by[1], by[2], by[0])
    fit_rms = math.sqrt((resx[0] + resy[0]) / rr.size) if len(resx) and len(resy) else None
    pick = {}
    for name, target in (("blue", 490.0), ("green", 560.0), ("red", 665.0), ("nir", 865.0)):
        valid = [i for i, c in enumerate(cw) if c > 0]
        i = min(valid, key=lambda k: abs(cw[k] - target))
        pick[name] = (i, cw[i])
    arr = np.stack([cube[:, pick[n][0], :].astype(np.float64) for n in ("blue", "green", "red", "nir")])
    dn0 = arr == 0
    arr = vmin + arr * (vmax - vmin) / 65535.0
    arr[dn0] = np.nan
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / "prisma20220427_he5_b_g_r_nir.tif"
    prof = {"driver": "GTiff", "width": cols, "height": rows, "count": 4, "dtype": "float32", "crs": f"EPSG:{tags['Epsg_Code']}",
            "transform": tr, "nodata": np.nan, "compress": "deflate"}
    with rasterio.open(out, "w", **prof) as d:
        d.write(arr.astype(np.float32))
        for k, n in enumerate(("blue", "green", "red", "nir")):
            d.set_band_description(k + 1, f"{n} {pick[n][1]:.2f} nm reflectance (he5 VNIR index {pick[n][0]})")
    with rasterio.open(SPECTRAL) as s:
        st = s.transform
        descr = s.descriptions
        bands = {}
        for n in ("blue", "green", "red", "nir"):
            wl = pick[n][1]
            j = min(range(len(descr)), key=lambda k: abs(float(re.search(r"\(([\d.]+) Nano", descr[k]).group(1)) - wl))
            bands[n] = (j + 1, s.read(j + 1).astype(np.float64))
    corners = {"UL": (0, 0), "UR": (0, cols), "LL": (rows, 0), "LR": (rows, cols), "centre": (rows / 2, cols / 2)}
    comp = []
    for k, (r, c) in corners.items():
        x1, y1 = tr * (c, r)
        x2, y2 = st * (c, r)
        comp.append({"item": f"corner {k} (pixel edge)", "he5_fit_x": round(x1, 2), "he5_fit_y": round(y1, 2),
                     "spectral_x": round(x2, 2), "spectral_y": round(y2, 2), "diff_m": round(math.hypot(x2 - x1, y2 - y1), 3)})
    comp.append({"item": "he5 grid fit rms (m)", "he5_fit_x": fit_rms})
    comp.append({"item": "he5 fitted pixel size x, y (m); rotation terms", "he5_fit_x": bx[1], "he5_fit_y": by[2],
                 "spectral_x": st.a, "spectral_y": st.e, "diff_m": f"rot {bx[2]:.3e} {by[1]:.3e}"})
    for n in ("blue", "green", "red", "nir"):
        j, sv = bands[n]
        hv = arr[["blue", "green", "red", "nir"].index(n)]
        ok = np.isfinite(hv) & (sv > 0)
        ratio = sv[ok] / hv[ok]
        comp.append({"item": f"{n}: SPECTRAL band {j} / he5 reflectance (min, median, max)",
                     "he5_fit_x": float(ratio.min()), "he5_fit_y": float(np.median(ratio)), "spectral_x": float(ratio.max()),
                     "diff_m": f"wavelength {pick[n][1]:.2f} nm"})
    with open(OUT / "prisma_he5_vs_spectral.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["item", "he5_fit_x", "he5_fit_y", "spectral_x", "spectral_y", "diff_m"])
        wr.writeheader()
        wr.writerows(comp)
    g = Grid(f"EPSG:{tags['Epsg_Code']}", tr.c, tr.f, tr.a, cols, rows)
    return out, tr, g, pick, comp


# ------------------------------------------------------------------------------------------------ Neo geometry
def neo_dimap_geometry():
    """Full-strip geometry of each Neo strip from its MS-FS DIMAP: ULXMAP, ULYMAP, XDIM, YDIM, NCOLS, NROWS, and one
    pixel in metres east-west and north-south at the strip centre."""
    geod = Geod(ellps="WGS84")
    out = {}
    for strip in NEO_STRIPS:
        xml = sorted((NEO_DIR / "extracted" / strip).glob("DIM_*MS-FS*.XML"))[0]
        r = ET.parse(xml).getroot()
        g = {"ulx": float(r.findtext(".//ULXMAP")), "uly": float(r.findtext(".//ULYMAP")),
             "xdim": float(r.findtext(".//XDIM")), "ydim": float(r.findtext(".//YDIM")),
             "ncols": int(r.findtext(".//Raster_Dimensions/NCOLS")), "nrows": int(r.findtext(".//Raster_Dimensions/NROWS"))}
        lonc = g["ulx"] + g["ncols"] * g["xdim"] / 2
        latc = g["uly"] - g["nrows"] * g["ydim"] / 2
        g["xdim_m"] = geod.inv(lonc, latc, lonc + g["xdim"], latc)[2]
        g["ydim_m"] = geod.inv(lonc, latc, lonc, latc + g["ydim"])[2]
        out[strip] = g
    return out


def full_uv(g, lon, lat):
    """Fractional position in the full strip: u east (0 at the west edge), v south (0 at the north edge)."""
    return (lon - g["ulx"]) / (g["ncols"] * g["xdim"]), (g["uly"] - lat) / (g["nrows"] * g["ydim"])


def neo_strip_position(geo, strip_grid1m, version):
    """For a block of GRID_1M (or 3 m): the dominant Neo strip and the block centre's position in the full strip."""
    to_ll = Transformer.from_crs(GRID_1M.crs, "EPSG:4326", always_xy=True)

    def extra(r0, c0, bs, factor):
        with rasterio.open(strip_grid1m) as s:
            sid = s.read(1, window=Window(c0 * factor, r0 * factor, bs * factor, bs * factor))
        vals, cnt = np.unique(sid[sid > 0], return_counts=True)
        if not vals.size:
            return {}
        strip = NEO_STRIP_CODE[int(vals[np.argmax(cnt)])]
        lon, lat = to_ll.transform(GRID_1M.left + (c0 + bs / 2) * factor, GRID_1M.top - (r0 + bs / 2) * factor)
        u, v = full_uv(geo[strip], lon, lat)
        return {"neo_version": version, "neo_strip": strip, "neo_u_full": round(u, 4), "neo_v_full": round(v, 4),
                "neo_strip_share": round(float(cnt.max() / sid.size), 3)}
    return extra


# ------------------------------------------------------------------------------------------------ pairs
def fill_invalid(a, valid):
    """Invalid pixels of one image set to the mean of its valid pixels (each image on its own: the two images' nodata
    patterns differ, so no shared structure is created)."""
    b = a.copy()
    b[~valid] = np.nanmean(a[valid])
    return b


def correlate(row, va, na, xa, vb, nb, xb, res, lowpass=None):
    """Validity, stability and phase correlation of one block; fills `row` and returns it. With lowpass (cycles per
    pixel, itareo.phasecorr) a second estimate weighted toward low frequencies is added as dx_lp_m, dy_lp_m (the one to
    use on bilinear-resampled copies; the unweighted dx_m, dy_m and the usage gate are unchanged)."""
    ok_a, ok_b = np.isfinite(va), np.isfinite(vb)
    valid = ok_a & ok_b
    vf = float(valid.mean())
    row.update(valid_frac=round(vf, 6), invalid_px=int((~valid).sum()))
    if vf < VALID_MIN:
        row.update(used=False, strict=False, note="valid share < 0.9")
        return row
    stable = valid & (na < NDVI_STABLE) & (nb < NDVI_STABLE) & ~xa & ~xb
    sf = float(stable.mean())
    row["stable_frac"] = round(sf, 4)
    if sf < STABLE_MIN:
        row.update(used=False, strict=False, note="stable share < 0.2")
        return row
    sy, sx, pk, q = phase_corr(fill_invalid(va, ok_a), fill_invalid(vb, ok_b))
    good = bool(q >= QUALITY_MIN)
    row.update(dx_m=round(sx * res, 3), dy_m=round(-sy * res, 3), peak=round(pk, 4), quality=round(q, 2), used=good,
               strict=bool(good and vf >= STRICT_VALID_MIN and sf >= STRICT_STABLE_MIN),
               note="" if good else "quality < 2")
    if lowpass:
        ly, lx, _, _ = phase_corr(fill_invalid(va, ok_a), fill_invalid(vb, ok_b), lowpass=lowpass)
        row.update(dx_lp_m=round(lx * res, 3), dy_lp_m=round(-ly * res, 3), lowpass=lowpass)
    return row


def run_pair(name, grid_res, H, Wd, src_a, src_b, rows_out, extra=None):
    """Blocks of about 2 km; one row per block (see the module docstring for the criteria)."""
    bs = int(round(BLOCK_M / grid_res))
    n_used = 0
    for r0 in range(0, H - bs + 1, bs):
        for c0 in range(0, Wd - bs + 1, bs):
            va, na, xa = src_a(r0, c0, bs, bs)
            vb, nb, xb = src_b(r0, c0, bs, bs)
            row = correlate({"pair": name, "block_row": r0, "block_col": c0, "block_px": bs, "res_m": grid_res},
                            va, na, xa, vb, nb, xb, grid_res)
            if extra and "dx_m" in row:
                row.update(extra(r0, c0, bs))
            n_used += bool(row.get("used"))
            rows_out.append(row)
    print(name, "blocks used", n_used, flush=True)


def neo_roi_footprints():
    """MS-FS ROI polygons of each Neo strip in GRID_1M coordinates (extracted MASKS GML, read from a copy in TMP)."""
    import pyogrio
    TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for strip in NEO_STRIPS:
        gml = sorted((NEO_DIR / "extracted" / strip).glob("ROI_*MS-FS*.GML"))[0]
        tmp = TMP / f"roi_{strip}.gml"
        shutil.copyfile(gml, tmp)
        df = pyogrio.read_dataframe(tmp, WRITE_GFS="NO")      # no .gfs schema file
        out[strip] = unary_union([shape(transform_geom(df.crs.to_string(), GRID_1M.crs, mapping(g), precision=-1))
                                  for g in df.geometry if g is not None])
    return out


def overlap_test(rows_out, geo):
    """P6: Neo strip against Neo strip in their overlaps under both conventions (blocks of P6_BLOCK m on a P6_STRIDE m
    stride inside both ROI masks). Each strip's VRT is warped (bilinear) onto the block of GRID_1M. One row per block
    tried; with the full-strip positions of the block centre and the GeoTIFF-minus-TFW difference the conventions
    imply for the shift of strip b relative to strip a: (u_a - u_b) x pixel east, -(v_a - v_b) x pixel north."""
    feet = neo_roi_footprints()
    to_ll = Transformer.from_crs(GRID_1M.crs, "EPSG:4326", always_xy=True)
    bs = P6_BLOCK
    n = 0
    for sa, sb in P6_PAIRS:
        inter = feet[sa].intersection(feet[sb])
        cands = []
        for r0 in range(0, GRID_1M.height - bs + 1, P6_STRIDE):
            for c0 in range(0, GRID_1M.width - bs + 1, P6_STRIDE):
                blk = box(GRID_1M.left + c0, GRID_1M.top - r0 - bs, GRID_1M.left + c0 + bs, GRID_1M.top - r0)
                if blk.intersects(inter) and blk.intersection(inter).area >= P6_FOOT_MIN * bs * bs:
                    cands.append((r0, c0))
        print("P6", sa, sb, "candidate blocks", len(cands), flush=True)
        for ver, base in (("tfw", NEO_DIR), ("geotiff", NEO_DIR / "v1_geotiff_tags")):
            for r0, c0 in cands:
                tr = Affine(1.0, 0, GRID_1M.left + c0, 0, -1.0, GRID_1M.top - r0)
                imgs = []
                for s_ in (sa, sb):
                    buf = np.full((4, bs, bs), np.nan, np.float32)
                    with rasterio.open(base / "vrt" / f"{s_}.vrt") as s:
                        reproject(rasterio.band(s, [1, 2, 3, 4]), buf, dst_transform=tr, dst_crs=GRID_1M.crs,
                                  resampling=Resampling.bilinear, src_nodata=0, dst_nodata=np.nan, num_threads=8)
                    imgs.append(buf.astype(np.float64) / 10000.0)
                va, na = vis_ndvi(imgs[0], [0, 1, 2], 0, 3)
                vb, nb = vis_ndvi(imgs[1], [0, 1, 2], 0, 3)
                va[~np.isfinite(imgs[0]).all(0)] = np.nan
                vb[~np.isfinite(imgs[1]).all(0)] = np.nan
                lon, lat = to_ll.transform(GRID_1M.left + c0 + bs / 2, GRID_1M.top - r0 - bs / 2)
                ua, va_ = full_uv(geo[sa], lon, lat)
                ub, vb_ = full_uv(geo[sb], lon, lat)
                xdim_m = (geo[sa]["xdim_m"] + geo[sb]["xdim_m"]) / 2
                ydim_m = (geo[sa]["ydim_m"] + geo[sb]["ydim_m"]) / 2
                row = {"pair": f"P6_neo_{sa}_vs_{sb}_{ver}", "convention": ver, "strip_a": sa, "strip_b": sb,
                       "block_row": r0, "block_col": c0, "block_px": bs, "res_m": 1.0,
                       "u_full_a": round(ua, 4), "v_full_a": round(va_, 4), "u_full_b": round(ub, 4), "v_full_b": round(vb_, 4),
                       "du": round(ua - ub, 4), "dv": round(va_ - vb_, 4),
                       "implied_geotiff_minus_tfw_dx_m": round((ua - ub) * xdim_m, 3),
                       "implied_geotiff_minus_tfw_dy_m": round(-(va_ - vb_) * ydim_m, 3)}
                no_x = np.zeros((bs, bs), bool)
                rows_out.append(correlate(row, va, na, no_x, vb, nb, no_x, 1.0))
                n += 1
    print("P6 overlap blocks tried", n, flush=True)


# ------------------------------------------------------------------------------------------------ summaries
OUTLIER_PX = 10.0          # a used block whose shift lies > 10 px from its pair's median is a false match


def flag_outliers(rows):
    """Per pair (5 or more used blocks): a used block whose shift differs from the median shift of the pair by more
    than OUTLIER_PX pixels is flagged outlier=True and not used (a false correlation peak: genuine blocks of every pair
    agree within about 6 px, the 30 m PRISMA trend included). Returns the flagged rows."""
    flagged = []
    for p in dict.fromkeys(r["pair"] for r in rows):
        u = [r for r in rows if r["pair"] == p and r.get("used")]
        if len(u) < 5:
            continue
        mx, my = float(np.median([r["dx_m"] for r in u])), float(np.median([r["dy_m"] for r in u]))
        for r in u:
            dev = math.hypot(r["dx_m"] - mx, r["dy_m"] - my) / r["res_m"]
            if dev > OUTLIER_PX:
                r.update(used=False, strict=False, outlier=True,
                         note=f"outlier: {dev:.1f} px from the median shift of the pair")
                flagged.append(r)
    return flagged


def _summary_row(p, rs, key, crit):
    u = [r for r in rs if r.get(key)]
    row = {"pair": p, "criteria": crit, "res_m": rs[0]["res_m"], "block_px": rs[0]["block_px"], "blocks_tried": len(rs),
           "blocks_correlated": sum(1 for r in rs if "dx_m" in r),
           "blocks_outlier": sum(1 for r in rs if r.get("outlier") is True),
           "blocks_used": len(u)}
    if u:
        dx = np.array([r["dx_m"] for r in u])
        dy = np.array([r["dy_m"] for r in u])
        mag = np.hypot(dx, dy)
        row.update(median_dx_m=round(float(np.median(dx)), 3), median_dy_m=round(float(np.median(dy)), 3),
                   mad_dx_m=round(float(np.median(np.abs(dx - np.median(dx)))), 3),
                   mad_dy_m=round(float(np.median(np.abs(dy - np.median(dy)))), 3),
                   p10_dx_m=round(float(np.percentile(dx, 10)), 3), p90_dx_m=round(float(np.percentile(dx, 90)), 3),
                   p10_dy_m=round(float(np.percentile(dy, 10)), 3), p90_dy_m=round(float(np.percentile(dy, 90)), 3),
                   median_shift_m=round(float(np.median(mag)), 3), median_shift_px=round(float(np.median(mag)) / rs[0]["res_m"], 3),
                   median_quality=round(float(np.median([r["quality"] for r in u])), 2))
    return row


def summarise(rows):
    out = []
    for p in dict.fromkeys(r["pair"] for r in rows):
        rs = [r for r in rows if r["pair"] == p]
        out.append(_summary_row(p, rs, "used", f"valid >= {VALID_MIN}, stable >= {STABLE_MIN}, quality >= {QUALITY_MIN}"))
        out.append(_summary_row(p, rs, "strict", f"strict: valid >= {STRICT_VALID_MIN}, stable >= {STRICT_STABLE_MIN}, "
                                                 f"quality >= {QUALITY_MIN}"))
    return out


def trend(rows):
    """Linear trend of the used-block shifts per pair: shift = a + b east_km + c north_km (least squares, centred on
    the used blocks' centroid), its span over the blocks and the residual RMS."""
    out = []
    for p in dict.fromkeys(r["pair"] for r in rows):
        u = [r for r in rows if r["pair"] == p and r.get("used")]
        row = {"pair": p, "blocks_used": len(u)}
        if len(u) >= MIN_TREND_BLOCKS:
            bs, res = u[0]["block_px"], u[0]["res_m"]
            x = np.array([(r["block_col"] + bs / 2) * res / 1000.0 for r in u])
            y = np.array([-(r["block_row"] + bs / 2) * res / 1000.0 for r in u])
            A = np.column_stack([np.ones_like(x), x - x.mean(), y - y.mean()])
            row.update(east_extent_km=round(float(np.ptp(x)), 2), north_extent_km=round(float(np.ptp(y)), 2))
            for comp in ("dx_m", "dy_m"):
                v = np.array([r[comp] for r in u])
                if np.ptp(x) > 0 and np.ptp(y) > 0:
                    c = np.linalg.lstsq(A, v, rcond=None)[0]
                else:                                  # one row or one column of blocks: trend along it only
                    Ab = A[:, [0, 1 if np.ptp(x) > 0 else 2]]
                    cb = np.linalg.lstsq(Ab, v, rcond=None)[0]
                    c = np.array([cb[0], cb[1], 0.0]) if np.ptp(x) > 0 else np.array([cb[0], 0.0, cb[1]])
                k = comp[:2]
                row.update({f"{k}_at_centroid_m": round(float(c[0]), 3), f"{k}_per_km_east_m": round(float(c[1]), 3),
                            f"{k}_per_km_north_m": round(float(c[2]), 3),
                            f"{k}_trend_span_m": round(float(np.ptp(A[:, 1:] @ c[1:])), 3),
                            f"{k}_residual_rms_m": round(float(np.sqrt(np.mean((v - A @ c) ** 2))), 3)})
        out.append(row)
    return out


def _fit_strip_intercepts(strips, pos, val):
    """val = a_strip + b pos; returns b, its standard error, the number of strips and the residual RMS."""
    names = sorted(set(strips))
    X = np.column_stack([np.array([s == n for s in strips], float) for n in names] + [np.asarray(pos, float)])
    y = np.asarray(val, float)
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - rank
    if dof <= 0 or rank < X.shape[1]:
        return None
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(coef[-1]), float(math.sqrt(cov[-1, -1])), len(names), float(math.sqrt(np.mean(resid ** 2)))


def convention_test(rows, geo):
    """Neo's own displacement (east, north) regressed on the full-strip position (u, v) with one intercept per strip,
    for P1/P1g and P2/P2g, with the slopes each convention predicts."""
    xdim_m = float(np.mean([g["xdim_m"] for g in geo.values()]))
    ydim_m = float(np.mean([g["ydim_m"] for g in geo.values()]))
    out = []
    for pair, ver in (("P1_pleiades2017_vs_pneo2022_1m", "tfw"), ("P1g_pleiades2017_vs_pneo2022_geotiff_tags_1m", "geotiff"),
                      ("P2_pneo2022_vs_planetscope20220424_3m", "tfw"),
                      ("P2g_pneo2022_geotiff_tags_vs_planetscope20220424_3m", "geotiff")):
        rs = [r for r in rows if r["pair"] == pair and r.get("used") and "neo_u_full" in r]
        row = {"pair": pair, "neo_version": ver, "blocks": len(rs)}
        if len(rs) >= 8:
            sgn = -1.0 if pair.startswith("P2") else 1.0          # P2 has Neo first: Neo's own displacement = -shift
            strips = [r["neo_strip"] for r in rs]
            for comp, poskey, unit in (("dx_m", "neo_u_full", "east_per_unit_u"), ("dy_m", "neo_v_full", "north_per_unit_v")):
                fit = _fit_strip_intercepts(strips, [r[poskey] for r in rs], [sgn * r[comp] for r in rs])
                if fit:
                    b, se, ns, rms = fit
                    row.update({f"slope_{unit}_m": round(b, 3), f"slope_{unit}_se_m": round(se, 3), "strips": ns,
                                f"residual_rms_{comp}": round(rms, 3)})
            row.update(expected_slope_east_per_unit_u_if_tfw_right=0.0 if ver == "tfw" else round(-xdim_m, 3),
                       expected_slope_east_per_unit_u_if_geotiff_right=round(xdim_m, 3) if ver == "tfw" else 0.0,
                       expected_slope_north_per_unit_v_if_tfw_right=0.0 if ver == "tfw" else round(ydim_m, 3),
                       expected_slope_north_per_unit_v_if_geotiff_right=round(-ydim_m, 3) if ver == "tfw" else 0.0)
        out.append(row)
    return out


def overlap_summary(rows):
    """P6 per strip pair and convention, and the paired GeoTIFF-minus-TFW difference against the implied one."""
    p6 = [r for r in rows if r["pair"].startswith("P6_")]
    out = []
    for sa, sb in P6_PAIRS:
        for ver in ("tfw", "geotiff"):
            rs = [r for r in p6 if r["strip_a"] == sa and r["strip_b"] == sb and r["convention"] == ver]
            u = [r for r in rs if r.get("used")]
            row = {"strip_a": sa, "strip_b": sb, "convention": ver, "blocks_tried": len(rs), "blocks_used": len(u)}
            if u:
                dx = np.array([r["dx_m"] for r in u])
                dy = np.array([r["dy_m"] for r in u])
                row.update(median_dx_m=round(float(np.median(dx)), 3), median_dy_m=round(float(np.median(dy)), 3),
                           p10_dx_m=round(float(np.percentile(dx, 10)), 3), p90_dx_m=round(float(np.percentile(dx, 90)), 3),
                           p10_dy_m=round(float(np.percentile(dy, 10)), 3), p90_dy_m=round(float(np.percentile(dy, 90)), 3),
                           du_range=[round(float(min(r["du"] for r in u)), 4), round(float(max(r["du"] for r in u)), 4)],
                           dv_range=[round(float(min(r["dv"] for r in u)), 4), round(float(max(r["dv"] for r in u)), 4)])
            out.append(row)
        key = lambda r: (r["block_row"], r["block_col"])
        t = {key(r): r for r in p6 if r["strip_a"] == sa and r["strip_b"] == sb and r["convention"] == "tfw" and r.get("used")}
        g = {key(r): r for r in p6 if r["strip_a"] == sa and r["strip_b"] == sb and r["convention"] == "geotiff" and r.get("used")}
        both = sorted(set(t) & set(g))
        row = {"strip_a": sa, "strip_b": sb, "convention": "geotiff minus tfw, paired blocks", "blocks_used": len(both)}
        if both:
            row.update(median_dx_m=round(float(np.median([g[k]["dx_m"] - t[k]["dx_m"] for k in both])), 3),
                       median_dy_m=round(float(np.median([g[k]["dy_m"] - t[k]["dy_m"] for k in both])), 3))
            if all("implied_geotiff_minus_tfw_dx_m" in t[k] for k in both):
                row.update(implied_median_dx_m=round(float(np.median([t[k]["implied_geotiff_minus_tfw_dx_m"] for k in both])), 3),
                           implied_median_dy_m=round(float(np.median([t[k]["implied_geotiff_minus_tfw_dy_m"] for k in both])), 3))
        out.append(row)
    return out


def _write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader()
        w.writerows(data)


def write_summaries(rows, geo):
    summ, tr, conv, p6 = summarise(rows), trend(rows), convention_test(rows, geo), overlap_summary(rows)
    _write_csv(OUT / "summary.csv", summ)
    _write_csv(OUT / "trend.csv", tr)
    _write_csv(OUT / "convention_test.csv", conv)
    _write_csv(OUT / "neo_overlap_test.csv", p6)
    return summ, tr, conv, p6


def load_blocks():
    rows = []
    with open(OUT / "blocks.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if v == "":
                    continue
                out[k] = (v == "True") if k in ("used", "strict", "outlier") else float(v) if k in NUM else v
            rows.append(out)
    return rows


def parameters():
    return {"ndvi_stable": NDVI_STABLE, "valid_min": VALID_MIN, "stable_min": STABLE_MIN, "quality_min": QUALITY_MIN,
            "strict_valid_min": STRICT_VALID_MIN, "strict_stable_min": STRICT_STABLE_MIN, "block_m": BLOCK_M,
            "p6_block_m": P6_BLOCK, "p6_stride_m": P6_STRIDE, "p6_min_share_inside_both_roi": P6_FOOT_MIN,
            "upsample": 20, "phase_correlation": "itareo.phasecorr", "outlier_px": OUTLIER_PX}


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    started = dt.datetime.now().isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    rows = []
    geo = neo_dimap_geometry()
    ext_t = neo_strip_position(geo, NEO_DIR / "pneo2022_strip_id_grid1m.tif", "tfw")
    ext_g = neo_strip_position(geo, NEO_DIR / "v1_geotiff_tags/pneo2022_strip_id_grid1m.tif", "geotiff")
    # P1 / P1g at 1 m
    run_pair("P1_pleiades2017_vs_pneo2022_1m", 1.0, GRID_1M.height, GRID_1M.width, src_ple(), src_neo(NEO, NEO_CLOUD),
             rows, lambda r0, c0, bs: ext_t(r0, c0, bs, 1))
    run_pair("P1g_pleiades2017_vs_pneo2022_geotiff_tags_1m", 1.0, GRID_1M.height, GRID_1M.width, src_ple(),
             src_neo(NEO_G, NEO_G_CLOUD), rows, lambda r0, c0, bs: ext_g(r0, c0, bs, 1))
    # P2 / P2g at 3 m
    run_pair("P2_pneo2022_vs_planetscope20220424_3m", 3.0, COMMON.height, COMMON.width, src_neo(NEO, NEO_CLOUD, 3),
             src_ps(PS22, COMMON, "planetscope20220424_3m"), rows, lambda r0, c0, bs: ext_t(r0, c0, bs, 3))
    run_pair("P2g_pneo2022_geotiff_tags_vs_planetscope20220424_3m", 3.0, COMMON.height, COMMON.width,
             src_neo(NEO_G, NEO_G_CLOUD, 3), src_ps(PS22, COMMON, "planetscope20220424_3m"), rows,
             lambda r0, c0, bs: ext_g(r0, c0, bs, 3))
    # P3 at 3 m
    run_pair("P3_planetscope20220424_vs_planetscope20250719_3m", 3.0, COMMON.height, COMMON.width,
             src_ps(PS22, COMMON, "planetscope20220424_3m"), src_ps(PS25, COMMON, "planetscope20250719_3m"), rows)
    # P4 at 10 m: PlanetScope 2022 area-weighted onto the S2 lattice
    from itareo.zonal import area_weighted_resample
    ps_s2 = TMP / "planetscope20220424_on_s2_10m.tif"
    area_weighted_resample(PS22, ps_s2, GRID_S2_10M, bands=[2, 4, 6, 8], min_valid=1.0)
    with rasterio.open(ps_s2) as s:
        nd = s.nodata

    def src_ps_s2(r0, c0, h, w):
        a = read_window(ps_s2, GRID_S2_10M, r0, c0, h, w, [1, 2, 3, 4], 1e-4, nd)
        return (*vis_ndvi(a, [0, 1, 2], 2, 3), np.zeros((h, w), bool))
    run_pair("P4_planetscope20220424_vs_s2_2022Q2_10m", 10.0, GRID_S2_10M.height, GRID_S2_10M.width,
             Src("ps_s2", src_ps_s2), src_s2(), rows)
    # P5 at 30 m: PRISMA from the he5; PlanetScope 2022 averaged onto the PRISMA grid
    pr_path, pr_tr, pr_grid, pick, comp = prisma_from_he5()
    ps_pr = TMP / "planetscope20220424_on_prisma_grid.tif"
    with rasterio.open(PS22) as s:
        src = s.read([2, 4, 6, 8]).astype(np.float32)
        src[src == 0] = np.nan
        dst = np.full((4, pr_grid.height, pr_grid.width), np.nan, np.float32)
        reproject(src, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=pr_tr, dst_crs=pr_grid.crs,
                  resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan, num_threads=8)
    with rasterio.open(ps_pr, "w", driver="GTiff", width=pr_grid.width, height=pr_grid.height, count=4, dtype="float32",
                       crs=pr_grid.crs, transform=pr_tr, nodata=np.nan, compress="deflate") as d:
        d.write(dst)

    def src_float(path, label, scale, iv, ir, inir):
        with rasterio.open(path) as s:
            arr = s.read().astype(np.float64) * scale

        def f(r0, c0, h, w):
            a = arr[:, r0:r0 + h, c0:c0 + w]
            return (*vis_ndvi(a, iv, ir, inir), np.zeros((h, w), bool))
        return Src(label, f)
    run_pair("P5_planetscope20220424_vs_prisma20220427_30m", pr_grid.res, pr_grid.height, pr_grid.width,
             src_float(ps_pr, "ps_on_prisma", 1e-4, [0, 1, 2], 2, 3), src_float(pr_path, "prisma_he5", 1.0, [0, 1, 2], 2, 3), rows)
    # P6: Neo strip against Neo strip in the overlaps, both conventions
    overlap_test(rows, geo)
    flagged = flag_outliers(rows)
    print("outlier blocks", len(flagged), flush=True)
    _write_csv(OUT / "blocks.csv", rows)
    summ, tr, conv, p6 = write_summaries(rows, geo)
    meta = {"started": started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1), "prisma_bands": {k: [v[0], v[1]] for k, v in pick.items()},
            "neo_full_strip_geometry": geo, "parameters": parameters(),
            "inputs": {"pleiades2017": str(PLE.relative_to(PROJECT)), "pneo2022": str(NEO.relative_to(PROJECT)),
                       "pneo2022_geotiff_tags": str(NEO_G.relative_to(PROJECT)), "planetscope2022": str(PS22.relative_to(PROJECT)),
                       "planetscope2025": str(PS25.relative_to(PROJECT)), "s2_2022Q2": str(S2Q2.relative_to(PROJECT)),
                       "prisma_he5": str(HE5.relative_to(PROJECT))}}
    (OUT / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    print(json.dumps(conv, indent=1))
    print(json.dumps(p6, indent=1))


def summarise_only():
    """Summaries from blocks.csv alone; the outlier rule is applied again and blocks.csv is rewritten with its flags
    (the measured columns are unchanged)."""
    rows = load_blocks()
    for r in rows:                               # flags re-derived from the stored measures and the current constants
        if "dx_m" in r:
            used = r["quality"] >= QUALITY_MIN
            r.update(used=used, outlier=False, note="" if used else "quality < 2",
                     strict=bool(used and r["valid_frac"] >= STRICT_VALID_MIN and r["stable_frac"] >= STRICT_STABLE_MIN))
    geo = neo_dimap_geometry()
    flagged = flag_outliers(rows)
    _write_csv(OUT / "blocks.csv", rows)
    write_summaries(rows, geo)
    meta = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    meta["summaries_rewritten_from_blocks_csv"] = dt.datetime.now().isoformat(timespec="seconds")
    meta["outlier_blocks"] = [{k: r[k] for k in ("pair", "block_row", "block_col", "dx_m", "dy_m", "note")} for r in flagged]
    meta["parameters"] = parameters()
    (OUT / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("summaries rewritten from", len(rows), "block rows;", len(flagged), "outlier blocks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.summarise:
        summarise_only()
    else:
        main()
