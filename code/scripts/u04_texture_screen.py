"""U0.4 texture screen (CLAUDE.md 13.1, 2 Nov 2026 night): which texture features separate the UGI inventory from other
green, measured on the training folds only, before any texture enters B2.

Stages (from work_ms/, PYTHONUTF8=1, CPU only; the script imports no torch and sets CUDA_VISIBLE_DEVICES=-1):
  python scripts/u04_texture_screen.py --scales   UGI scales and the texture windows       -> scales.json
  python scripts/u04_texture_screen.py --pan      Neo PAN extraction size (zip directories) -> pan_extraction_size.json
  python scripts/u04_texture_screen.py --screen   features, samples, AUROC, NDVI add-on      -> auroc.csv, ...
  python scripts/u04_texture_screen.py --readme   README.md from the files above
Every output goes to docs/evidence_phase2/texture_screen/. Inputs are read only; the inventory GeoPackage is read from
a copy in %TEMP% (the copy made by u01 is reused when its sha256 matches); the Neo zips are only listed.

Scales (--scales)
  * Narrow UGI width: 2 x area / perimeter of every CANTERO and FRANJA polygon of raw_data/22174_UrbanGreenInfras.gpkg
    in EPSG:32720 (exact duplicate geometries dropped as in u01: the typed copy kept). For a strip of width w and
    length L >> w, 2A/P = wL / (L + w) ~ w.
  * Crowns from the Meta/WRI 1 m CHM on a seeded sample of tiles (10 per fold, non-straddling, with UGI): (a) 8-connected
    components of CHM >= 3 m inside the tile, components touching the tile border dropped, equivalent diameter
    2 sqrt(A / pi); (b) tree tops = plateaus of local maxima (3 x 3) of the CHM smoothed by a Gaussian (sigma 1 px) with
    height >= 3 m; crown spacing = distance to the nearest other top (tops of a 32 px margin count as neighbours).
  * Windows: WINDOW_RULE below.

Screen (--screen)
  * Tiles: tiles.csv (512 px, GRID_1M), non-straddling, Neo available. Per fold and class, a seeded simple random sample
    of K_TILES tiles (positives: tiles with UGI; negatives: any tile). Then N_PER pixels per fold and class drawn
    uniformly from the pooled eligible pixels of those tiles (multivariate hypergeometric allocation, then uniform
    within the tile), so the pixel sample is self-weighting within the sampled tiles.
  * Positives: UGI = 1, edge band = 0, NDVI >= 0.3, valid clear Neo. Negatives: UGI = 0, edge = 0, NDVI >= 0.3, valid
    clear Neo (valid = TOA band 1 != 65535 and NDVI != -32768; clear = cloud flag 0).
  * Features are computed on the tile plus a margin (windows have context), at the sampled pixels only.
  * AUROC: for each held-out fold k, sklearn roc_auc_score on the pooled samples of the other three folds. Oriented
    AUROC = AUROC if the feature is higher in UGI on average over the four sets (sign +1), else 1 - AUROC.
  * Add-on check: inside each training-fold set, leave-one-fold-out over its three folds; models are logistic
    regressions on rank-transformed, cubic-spline-expanded inputs (QuantileTransformer -> SplineTransformer(5 knots)
    -> LogisticRegression); pooled out-of-fold AUROC of NDVI + feature against NDVI alone (and NDVI + CHM + feature
    against NDVI + CHM). The held-out fold k is never used.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GDAL_PAM_ENABLED"] = "NO"          # never write .aux.xml next to an input

import argparse                                      # noqa: E402
import datetime as dt                                # noqa: E402
import hashlib                                       # noqa: E402
import json                                          # noqa: E402
import math                                          # noqa: E402
import re                                            # noqa: E402
import shutil                                        # noqa: E402
import sys                                           # noqa: E402
import time                                          # noqa: E402
import xml.etree.ElementTree as ET                   # noqa: E402
import zipfile                                       # noqa: E402
from concurrent.futures import ProcessPoolExecutor   # noqa: E402
from pathlib import Path                             # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
PROJECT = WM.parent

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import rasterio                                      # noqa: E402
from rasterio.windows import Window                  # noqa: E402
from scipy import ndimage as ndi                     # noqa: E402

from itareo.grid import COMMON, GRID_1M              # noqa: E402

OUT = PROJECT / "docs" / "evidence_phase2" / "texture_screen"
NEO_TOA = WM / "derived/pneo2022_registered/pneo2022_toa_grid1m.tif"
NEO_IDX = WM / "derived/features/neo2022_indices_grid1m.tif"
NEO_CLOUD = WM / "derived/pneo2022_registered/pneo2022_cloud_grid1m.tif"
CHM = PROJECT / "gee_exports/2026-11-02_a8/canopy_height_meta_wri_1m.tif"
CSK_RAW = PROJECT / "work/raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI.tif"
UGI_BIN = WM / "derived/ugi_labels/ugi_binary_grid1m.tif"
UGI_EDGE = WM / "derived/ugi_labels/ugi_edge_grid1m.tif"
UGI_GROUP = WM / "derived/ugi_labels/ugi_group_grid1m.tif"
TILES = WM / "derived/ugi_tiles/tiles.csv"
GPKG = PROJECT / "raw_data/22174_UrbanGreenInfras.gpkg"
GPKG_SHA = "9e37d14c82e6829450112396f6080ec37ee7a252c3e3735f910555603f188be9"   # derived/ugi_labels/README.md
NEO_ZIPS = PROJECT / "raw_data/pleiades_neo"
STRIPS = ["1_2", "1_3", "1_5", "1_6"]
FOLD_X, FOLD_Y = 388282.5, 6525025.5                 # Luigi's quadrant lines (u08_tile_index.py)
GROUPS = {1: "road_greens", 2: "squares", 3: "parks_green_areas", 4: "water_corridors", 5: "rail_verges",
          6: "no_definido"}

SEED = 0
K_TILES = 30            # tiles per fold and class
N_PER = 20000           # pixels per fold and class
GREEN = 0.3             # NDVI threshold of the green mask and of the samples
TREE_H = 3              # m, tree mask on the CHM
LAGS = list(range(5, 16))
GLCM_L = 16             # grey levels
VALID_MIN = 0.5         # minimum valid share of a window
WORKERS = 6
WINDOW_RULE = ("w_s (element scale) = the smallest odd number of 1 m pixels >= the median CANTERO/FRANJA width "
               "(2A/P); w_l (configuration scale) = the smallest odd number of pixels >= 3 x the median crown spacing "
               "(nearest tree-top distance), i.e. a window about three crowns across. COSMO-SkyMed windows at 2.5 m: "
               "the odd number of pixels nearest to the 1 m window in metres, at least 3. Autocorrelation lags 5-15 m "
               "(CLAUDE.md 13.1) are kept and compared with the measured crown spacing.")


# ----------------------------------------------------------------------------------------------------------- helpers
def sha256(p, chunk=1 << 23):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "mean": round(float(x.mean()), 3), "p5": round(float(p[0]), 3),
            "q25": round(float(p[1]), 3), "median": round(float(p[2]), 3), "q75": round(float(p[3]), 3),
            "p95": round(float(p[4]), 3)}


def odd_ceil(x):
    n = int(math.ceil(x - 1e-9))
    return n if n % 2 == 1 else n + 1


def odd_nearest(x, minimum=3):
    n = int(round(x))
    if n % 2 == 0:
        n = n + 1 if (x - n) >= 0 else n - 1
    return max(minimum, n)


def fold_of(x, y):
    left, south = x < FOLD_X, y < FOLD_Y
    return 1 if left and south else 2 if south else 3 if left else 4


def read_pad(src, band, r0, c0, h, w, fill):
    """Window of (h, w) pixels at (r0, c0), padded with `fill` outside the raster."""
    H, W_ = src.height, src.width
    a0, b0, a1, b1 = max(r0, 0), max(c0, 0), min(r0 + h, H), min(c0 + w, W_)
    out = np.full((h, w), fill, dtype=np.dtype(src.dtypes[band - 1]))
    if a1 > a0 and b1 > b0:
        out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = src.read(band, window=Window(b0, a0, b1 - b0, a1 - a0))
    return out


def msum(x, w):
    """Moving-window sum over w x w pixels (zero outside the array); rounded to 1e-9 so that the float residue of the
    running sum does not break exact ties (e.g. zero edges in a window)."""
    return np.round(ndi.uniform_filter(x, size=w, mode="constant", cval=0.0) * float(w * w), 9)


def shifted(x, dy, dx, fill=0.0):
    """y[i, j] = x[i + dy, j + dx], `fill` where that falls outside."""
    y = np.full_like(x, fill)
    H, W_ = x.shape
    ys, ye = max(0, -dy), min(H, H - dy)
    xs, xe = max(0, -dx), min(W_, W_ - dx)
    if ye > ys and xe > xs:
        y[ys:ye, xs:xe] = x[ys + dy:ye + dy, xs + dx:xe + dx]
    return y


def gpkg_copy():
    cand = Path(os.environ.get("TEMP", ".")) / "u01_ugi" / GPKG.name
    if cand.exists() and sha256(cand) == GPKG_SHA:
        return cand
    tmp = Path(os.environ.get("TEMP", ".")) / "u04_texture" / GPKG.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not tmp.exists() or sha256(tmp) != GPKG_SHA:
        shutil.copyfile(GPKG, tmp)
    assert sha256(tmp) == GPKG_SHA
    return tmp


def load_tiles():
    t = pd.read_csv(TILES)
    t = t[(~t["straddle"].astype(bool)) & (t["avail_neo2022"] > 0)].copy()
    t["h"] = (t["y1"] - t["y0"]).round().astype(int)
    t["w"] = (t["x1"] - t["x0"]).round().astype(int)
    return t.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------ scales
def narrow_widths():
    import pyogrio
    src = gpkg_copy()
    df = pyogrio.read_dataframe(src).to_crs(GRID_1M.crs)
    df["_nodef"] = df["Tipo_evp_2"] == "NO DEFINIDO"
    df = df.sort_values(["_nodef", "ID_NUEVO"])
    wkb = df.geometry.to_wkb()
    df = df[~wkb.duplicated(keep="first").values]
    nar = df[df["Tipo_evp_2"].isin(["CANTERO", "FRANJA"])].copy()
    nar["area"] = nar.geometry.area
    nar["perim"] = nar.geometry.length
    nar["width"] = 2.0 * nar["area"] / nar["perim"]
    cen = nar.geometry.centroid
    nar["fold"] = [fold_of(p.x, p.y) for p in cen]
    out = {"source": str(GPKG.relative_to(PROJECT)), "source_sha256": GPKG_SHA, "read_from": str(src),
           "polygons_after_duplicates": int(len(df)), "crs": GRID_1M.crs,
           "all_narrow": stats(nar["width"]),
           "by_type": {t: stats(g["width"]) for t, g in nar.groupby("Tipo_evp_2")},
           "by_fold_centroid": {int(f): stats(g["width"]) for f, g in nar.groupby("fold")},
           "area_m2_narrow": stats(nar["area"])}
    return out


def crown_tile(args):
    r0, c0, h, w = args
    M = 32
    with rasterio.open(CHM) as s:
        chm = read_pad(s, 1, r0 - M, c0 - M, h + 2 * M, w + 2 * M, 255)
    with rasterio.open(UGI_BIN) as s:
        ugi = read_pad(s, 1, r0 - M, c0 - M, h + 2 * M, w + 2 * M, 0)
    vc = chm != 255
    core = np.zeros(chm.shape, bool)
    core[M:M + h, M:M + w] = True
    T = (chm >= TREE_H) & vc & core
    lab, n = ndi.label(T, structure=np.ones((3, 3), bool))
    comps = []
    if n:
        area = np.bincount(lab.ravel())
        inside = np.bincount(lab.ravel(), weights=(ugi == 1).ravel().astype(float))
        border = np.zeros(chm.shape, bool)
        border[M, M:M + w] = border[M + h - 1, M:M + w] = True
        border[M:M + h, M] = border[M:M + h, M + w - 1] = True
        touch = np.unique(lab[border & T])
        keep = np.ones(n + 1, bool)
        keep[0] = False
        keep[touch] = False
        for i in np.nonzero(keep)[0]:
            comps.append((float(area[i]), float(inside[i] / area[i])))
    f = np.where(vc, chm.astype(float), 0.0)
    sm = ndi.gaussian_filter(f, 1.0)
    pk = (sm == ndi.maximum_filter(sm, size=3)) & (sm >= TREE_H) & vc
    pl, npk = ndi.label(pk, structure=np.ones((3, 3), bool))
    tops = []
    if npk > 1:
        rows, cols = np.indices(pk.shape)
        cnt = np.bincount(pl.ravel())[1:]
        cy = np.bincount(pl.ravel(), weights=rows.ravel())[1:] / cnt
        cx = np.bincount(pl.ravel(), weights=cols.ravel())[1:] / cnt
        from scipy.spatial import cKDTree
        kd = cKDTree(np.c_[cy, cx])
        incore = (cy >= M) & (cy < M + h) & (cx >= M) & (cx < M + w)
        d, _ = kd.query(np.c_[cy[incore], cx[incore]], k=2)
        u = ugi[np.rint(cy[incore]).astype(int), np.rint(cx[incore]).astype(int)] == 1
        tops = list(zip(d[:, 1].tolist(), u.tolist()))
    return comps, tops


def run_scales():
    t0, c0_ = time.time(), time.process_time()
    OUT.mkdir(parents=True, exist_ok=True)
    widths = narrow_widths()
    tiles = load_tiles()
    cand = tiles[(tiles["ugi_px"] > 0) & (tiles["avail_meta1m"] >= 0.9)]
    rng = np.random.default_rng(SEED)
    pick = []
    for f in (1, 2, 3, 4):
        g = cand[cand["fold"] == f]
        pick += g.iloc[np.sort(rng.choice(len(g), size=min(10, len(g)), replace=False))].index.tolist()
    sel = tiles.loc[pick]
    comps, tops = [], []
    for r in sel.itertuples():
        c, t = crown_tile((int(r.row0), int(r.col0), int(r.h), int(r.w)))
        comps += c
        tops += t
    ca = np.array([c[0] for c in comps])
    cu = np.array([c[1] for c in comps])
    deq = 2.0 * np.sqrt(ca / np.pi)
    ta = np.array([t[0] for t in tops])
    tu = np.array([t[1] for t in tops], bool)
    crowns = {"tiles": [[int(r.row0), int(r.col0), int(r.fold)] for r in sel.itertuples()],
              "chm": str(CHM.relative_to(PROJECT)), "chm_sha256": sha256(CHM), "tree_threshold_m": TREE_H,
              "components_equivalent_diameter_m": {"all": stats(deq), "inside_ugi_majority": stats(deq[cu >= 0.5]),
                                                   "outside_ugi": stats(deq[cu < 0.5]),
                                                   "all_area_ge_4m2": stats(deq[ca >= 4])},
              "tree_top_spacing_m": {"all": stats(ta), "inside_ugi": stats(ta[tu]), "outside_ugi": stats(ta[~tu])}}
    med_w = widths["all_narrow"]["median"]
    med_sp = crowns["tree_top_spacing_m"]["all"]["median"]
    w_s = odd_ceil(med_w)
    w_l = odd_ceil(3.0 * med_sp)
    win = {"rule": WINDOW_RULE, "inputs": {"median_narrow_width_m": med_w, "median_crown_spacing_m": med_sp},
           "w_s_px_1m": w_s, "w_l_px_1m": w_l,
           "csk_px_2p5m": {"w_s": odd_nearest(w_s / 2.5), "w_l": odd_nearest(w_l / 2.5)},
           "acf_lags_m": LAGS,
           "crown_spacing_inside_lag_range": bool(LAGS[0] <= med_sp <= LAGS[-1])}
    out = {"finished": dt.datetime.now().isoformat(timespec="seconds"), "seed": SEED,
           "narrow_ugi_width_m": widths, "crowns": crowns, "windows": win,
           "seconds_wall": round(time.time() - t0, 1), "seconds_cpu_main": round(time.process_time() - c0_, 1)}
    (OUT / "scales.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"widths": widths["all_narrow"], "by_type": widths["by_type"],
                      "crowns": crowns["components_equivalent_diameter_m"], "tops": crowns["tree_top_spacing_m"],
                      "windows": win}, indent=1))


# --------------------------------------------------------------------------------------------------------------- PAN
def common_4326():
    import shapely
    from rasterio.warp import transform_geom
    from shapely.geometry import box, mapping, shape
    b = shapely.segmentize(box(*COMMON.bounds), 100.0)
    return shape(transform_geom(COMMON.crs, "EPSG:4326", mapping(b), precision=-1))


def run_pan():
    from shapely.geometry import box
    t0 = time.time()
    ext = common_4326()
    res = {"finished": None, "common_extent_epsg32720": list(COMMON.bounds), "strips": {}}
    tot = {"tiles": 0, "tif_bytes": 0, "tif_compressed": 0, "tfw_bytes": 0, "tfw_compressed": 0,
           "dimap_bytes": 0, "dimap_compressed": 0}
    for strip in STRIPS:
        z = sorted(NEO_ZIPS.glob(f"*WO_000047718_{strip}_SAL*.zip"))
        assert len(z) == 1, (strip, z)
        zp = z[0]
        with zipfile.ZipFile(zp) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            dim = [n for n in infos if re.search(r"/IMG_\d+_PNEO\d_PAN/DIM_.*\.XML$", n)]
            assert len(dim) == 1, dim
            r = ET.fromstring(zf.read(dim[0]))
            ncols = int(r.findtext(".//Raster_Dimensions/NCOLS"))
            nrows = int(r.findtext(".//Raster_Dimensions/NROWS"))
            size = r.find(".//Regular_Tiling/NTILES_SIZE")
            tc, tr_ = int(size.get("ncols")), int(size.get("nrows"))
            ulx, uly = float(r.findtext(".//ULXMAP")), float(r.findtext(".//ULYMAP"))
            xdim, ydim = float(r.findtext(".//XDIM")), float(r.findtext(".//YDIM"))
            tiles = sorted(n for n in infos if re.search(r"/IMG_\d+_PNEO\d_PAN/IMG_.*_P_R\d+C\d+\.TIF$", n))
            rows, take = [], []
            for n in tiles:
                m = re.search(r"_R(\d+)C(\d+)\.TIF$", n)
                R, C = int(m.group(1)), int(m.group(2))
                tfw = n[:-4] + ".TFW"
                a, d, b_, e, c, f = (float(v) for v in zf.read(tfw).decode().split())
                x0, y0 = c - a / 2.0, f - e / 2.0          # world file: centre of the upper-left pixel
                w = min(tc, ncols - (C - 1) * tc)
                h = min(tr_, nrows - (R - 1) * tr_)
                # consistency with the DIMAP origin and tiling
                dx0 = (x0 - (ulx + (C - 1) * tc * xdim)) / xdim
                dy0 = (y0 - (uly - (R - 1) * tr_ * ydim)) / ydim
                fp = box(x0, y0 + h * e, x0 + w * a, y0)
                inter = fp.intersects(ext)
                rec = {"member": n.split("/")[-1], "R": R, "C": C, "cols": w, "rows": h,
                       "tif_bytes": infos[n].file_size, "tif_compressed": infos[n].compress_size,
                       "tfw_bytes": infos[tfw].file_size, "tfw_compressed": infos[tfw].compress_size,
                       "origin_vs_dimap_px": [round(dx0, 6), round(dy0, 6)], "intersects_common_extent": bool(inter),
                       "share_of_tile_in_extent": round(fp.intersection(ext).area / fp.area, 4)}
                rows.append(rec)
                if inter:
                    take.append(rec)
            dimap_other = [n for n in infos if "/IMG_01_PNEO" in n and "_PAN/" in n and
                           (n.endswith(".XML") or "/MASKS/" in n)]
        s = {"zip": zp.name, "pan_dimap": dim[0].split("/")[-1], "ncols": ncols, "nrows": nrows,
             "tile_size": [tc, tr_], "xdim_deg": xdim, "pan_tiles_total": len(tiles),
             "pan_tiles_intersecting": len(take),
             "tif_bytes": sum(t["tif_bytes"] for t in take), "tif_compressed": sum(t["tif_compressed"] for t in take),
             "tfw_bytes": sum(t["tfw_bytes"] for t in take), "tfw_compressed": sum(t["tfw_compressed"] for t in take),
             "dimap_xml_and_masks_bytes": sum(infos[n].file_size for n in dimap_other),
             "dimap_xml_and_masks_compressed": sum(infos[n].compress_size for n in dimap_other),
             "all_pan_tiles_bytes": sum(t["tif_bytes"] for t in rows),
             "all_pan_tiles_compressed": sum(t["tif_compressed"] for t in rows),
             "tiles": rows}
        s["extract_bytes"] = s["tif_bytes"] + s["tfw_bytes"] + s["dimap_xml_and_masks_bytes"]
        s["extract_compressed"] = s["tif_compressed"] + s["tfw_compressed"] + s["dimap_xml_and_masks_compressed"]
        s["extract_GB"] = round(s["extract_bytes"] / 1e9, 3)
        s["extract_compressed_GB"] = round(s["extract_compressed"] / 1e9, 3)
        # strip 1_2 was delivered both zipped and unpacked: compare the unpacked PAN tiles with the zip members
        unpacked = sorted(NEO_ZIPS.glob(f"*WO_000047718_{strip}_SAL*"))
        unpacked = [p for p in unpacked if p.is_dir()]
        if unpacked:
            pdir = list(unpacked[0].glob("IMG_*_PAN"))
            on_disk = {}
            for t in take:
                p = pdir[0] / t["member"] if pdir else None
                on_disk[t["member"]] = (p is not None and p.exists() and p.stat().st_size == t["tif_bytes"])
            s["unpacked_folder"] = str(unpacked[0].relative_to(PROJECT))
            s["intersecting_tiles_already_unpacked_same_size"] = sum(on_disk.values())
        tot["tiles"] += s["pan_tiles_intersecting"]
        for k in ("tif_bytes", "tif_compressed", "tfw_bytes", "tfw_compressed"):
            tot[k] += s[k]
        tot["dimap_bytes"] += s["dimap_xml_and_masks_bytes"]
        tot["dimap_compressed"] += s["dimap_xml_and_masks_compressed"]
        res["strips"][strip] = s
        print(strip, s["pan_tiles_intersecting"], "/", s["pan_tiles_total"], "tiles", s["extract_GB"], "GB",
              s["extract_compressed_GB"], "GB compressed", flush=True)
    tot["extract_bytes"] = tot["tif_bytes"] + tot["tfw_bytes"] + tot["dimap_bytes"]
    tot["extract_compressed"] = tot["tif_compressed"] + tot["tfw_compressed"] + tot["dimap_compressed"]
    tot["extract_GB"] = round(tot["extract_bytes"] / 1e9, 3)
    tot["extract_compressed_GB"] = round(tot["extract_compressed"] / 1e9, 3)
    res["total"] = tot
    # strips whose intersecting PAN tiles are all already unpacked (same size) need no extraction
    need = [s for s, d in res["strips"].items()
            if d.get("intersecting_tiles_already_unpacked_same_size", 0) < d["pan_tiles_intersecting"]]
    res["total_still_to_extract"] = {
        "strips": need, "tiles": sum(res["strips"][s]["pan_tiles_intersecting"] for s in need),
        "extract_GB": round(sum(res["strips"][s]["extract_bytes"] for s in need) / 1e9, 3),
        "extract_compressed_GB": round(sum(res["strips"][s]["extract_compressed"] for s in need) / 1e9, 3)}
    res["method"] = ("Zip directories listed with zipfile (read-only; nothing extracted). PAN image tiles "
                     "(IMG_*_P_R<r>C<c>.TIF) whose footprint intersects the common extent: footprint from the tile's "
                     "world file (.TFW, centre of the upper-left pixel) and its size from the PAN DIMAP tiling "
                     "(NTILES_SIZE, NCOLS, NROWS); common extent COMMON.bounds (EPSG:32720) densified every 100 m and "
                     "transformed to EPSG:4326. Sizes: file_size (uncompressed) and compress_size (in the zip) of the "
                     "tiles, their TFW and the PAN DIMAP XML files and MASKS of the strip. GB = 1e9 bytes.")
    res["seconds"] = round(time.time() - t0, 1)
    res["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / "pan_extraction_size.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))


# ------------------------------------------------------------------------------------------------------------ screen
def eligibility(r0, c0, h, w):
    with rasterio.open(UGI_BIN) as s:
        ugi = s.read(1, window=Window(c0, r0, w, h))
    with rasterio.open(UGI_EDGE) as s:
        edge = s.read(1, window=Window(c0, r0, w, h))
    with rasterio.open(NEO_IDX) as s:
        nd = s.read(1, window=Window(c0, r0, w, h))
    with rasterio.open(NEO_CLOUD) as s:
        cl = s.read(1, window=Window(c0, r0, w, h))
    with rasterio.open(NEO_TOA) as s:
        t1 = s.read(1, window=Window(c0, r0, w, h))
    ok = (nd != -32768) & (t1 != 65535) & (cl == 0) & (nd >= int(GREEN * 10000)) & (edge == 0)
    return ok & (ugi == 1), ok & (ugi == 0), nd


def csk_index(rows_1m, cols_1m, csk_tr):
    x = GRID_1M.left + cols_1m + 0.5
    y = GRID_1M.top - rows_1m - 0.5
    return (np.floor((csk_tr.f - y) / -csk_tr.e).astype(int), np.floor((x - csk_tr.c) / csk_tr.a).astype(int))


def pass_a(task):
    c0_ = time.process_time()
    r0, c0, h, w = task
    pos, neg, nd = eligibility(r0, c0, h, w)
    rng = np.random.default_rng([SEED, r0, c0, 1])
    v = nd != -32768
    ndv = nd[v].astype(float) / 1e4
    ndv = rng.choice(ndv, size=min(2000, ndv.size), replace=False) if ndv.size else ndv
    with rasterio.open(CSK_RAW) as s:
        tr = s.transform
        rr, cc = csk_index(np.array([r0, r0 + h - 1]), np.array([c0, c0 + w - 1]), tr)
        dn = read_pad(s, 1, int(rr[0]), int(cc[0]), int(rr[1] - rr[0] + 1), int(cc[1] - cc[0] + 1), 0)
    dbv = 20.0 * np.log10(dn[dn > 0].astype(float))
    dbv = rng.choice(dbv, size=min(2000, dbv.size), replace=False) if dbv.size else dbv
    return {"tile": (r0, c0), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "ndvi": ndv, "csk_db": dbv,
            "cpu": time.process_time() - c0_}


def structure_feats(img, v, windows, prefix):
    f = np.where(v, img, np.nanmean(img[v]) if v.any() else 0.0)
    gx = ndi.sobel(f, axis=1) / 8.0
    gy = ndi.sobel(f, axis=0) / 8.0
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy
    a, b, c = (ndi.gaussian_filter(j, 1.5) for j in (jxx, jyy, jxy))
    th2 = np.arctan2(2.0 * c, a - b)
    flat = (a + b) < 1e-10
    u = np.where(flat, 0.0, np.cos(th2))
    s = np.where(flat, 0.0, np.sin(th2))
    out = {}
    for name, w in windows.items():
        A, B, C = (np.round(ndi.uniform_filter(j, w, mode="nearest"), 12) for j in (jxx, jyy, jxy))
        out[f"{prefix}_coherence_{name}"] = np.round(np.sqrt((A - B) ** 2 + 4.0 * C ** 2) / (A + B + 1e-6), 9)
        out[f"{prefix}_orientcons_{name}"] = np.round(np.hypot(ndi.uniform_filter(u, w, mode="nearest"),
                                                               ndi.uniform_filter(s, w, mode="nearest")), 9)
    return out


def spatial_stats(x, v, mu, sd, lo, hi, windows, prefix):
    """Local Moran's I, Geary's c, Getis-Ord Gi* (global mean and SD) and moving-window GLCM variance and correlation
    (16 levels on [lo, hi], distance 1, 4 directions averaged, symmetric) from moving-window moments."""
    vf = v.astype(float)
    z = np.where(v, (x - mu) / sd, 0.0)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, w in windows.items():
            N = msum(vf, w)
            S1, S2 = msum(z, w), msum(z * z, w)
            nn = N - vf
            m1, m2 = (S1 - z) / nn, (S2 - z * z) / nn
            ok = v & (N / (w * w) >= VALID_MIN) & (nn > 0.5)
            out[f"{prefix}_moran_{name}"] = np.where(ok, z * m1, np.nan)
            out[f"{prefix}_geary_{name}"] = np.where(ok, z * z - 2 * z * m1 + m2, np.nan)
            out[f"{prefix}_gistar_{name}"] = np.where(ok, S1 / np.sqrt(N), np.nan)
        q = np.where(v, np.clip(np.floor((x - lo) / (hi - lo) * GLCM_L), 0, GLCM_L - 1), 0.0)
        acc = {name: [np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)] for name in windows}
        for dy, dx in ((0, 1), (1, 1), (1, 0), (1, -1)):
            b = shifted(q, dy, dx)
            pv = vf * shifted(vf, dy, dx)
            for name, w in windows.items():
                N = msum(pv, w)
                Ea, Eb = msum(q * pv, w) / N, msum(b * pv, w) / N
                Eaa, Ebb, Eab = msum(q * q * pv, w) / N, msum(b * b * pv, w) / N, msum(q * b * pv, w) / N
                m = (Ea + Eb) / 2.0
                var = (Eaa + Ebb) / 2.0 - m * m
                corr = np.where(var > 1e-9, (Eab - m * m) / var, 1.0)
                acc[name][0] += var
                acc[name][1] += corr
                acc[name][2] += (N / (w * w) >= VALID_MIN)
        for name, w in windows.items():
            ok = v & (acc[name][2] == 4)
            out[f"{prefix}_glcmvar_{name}"] = np.where(ok, acc[name][0] / 4.0, np.nan)
            out[f"{prefix}_glcmcorr_{name}"] = np.where(ok, acc[name][1] / 4.0, np.nan)
    return out


def pass_b(task):
    c0_ = time.process_time()
    r0, c0, h, w, n_pos, n_neg, win, cwin, k = task
    M = max(win.values()) // 2 + LAGS[-1] + 4
    pos, neg, _ = eligibility(r0, c0, h, w)
    rng = np.random.default_rng([SEED, r0, c0, 2])
    pick = {}
    for cls, m, n in ((1, pos, n_pos), (0, neg, n_neg)):
        idx = np.flatnonzero(m.ravel())
        pick[cls] = np.sort(rng.choice(idx, size=n, replace=False)) if n > 0 else np.zeros(0, int)
    H, W_ = h + 2 * M, w + 2 * M
    with rasterio.open(NEO_IDX) as s:
        ndi_ = read_pad(s, 1, r0 - M, c0 - M, H, W_, -32768)
    with rasterio.open(CHM) as s:
        chm = read_pad(s, 1, r0 - M, c0 - M, H, W_, 255)
    with rasterio.open(UGI_GROUP) as s:
        grp = s.read(1, window=Window(c0, r0, w, h))
    v = ndi_ != -32768
    nd = np.where(v, ndi_.astype(float) / 1e4, np.nan)
    vf = v.astype(float)
    feats = {"ndvi": nd}
    vc = chm != 255
    chf = np.where(vc, chm.astype(float), np.nan)
    feats["chm"] = chf
    with np.errstate(divide="ignore", invalid="ignore"):
        # linearity
        feats.update(structure_feats(nd, v, win, "ndvi"))
        feats.update(structure_feats(chf, vc, win, "chm"))
        # configuration of green patches
        G = v & (nd >= GREEN)
        Gf = G.astype(float)
        E = np.zeros_like(Gf)
        E[:, :-1] += ((G[:, :-1] != G[:, 1:]) & v[:, :-1] & v[:, 1:])
        E[:-1, :] += ((G[:-1, :] != G[1:, :]) & v[:-1, :] & v[1:, :])
        lab, npatch = ndi.label(G, structure=np.ones((3, 3), bool))
        area = np.bincount(lab.ravel()).astype(float)
        area[0] = 0.0
        own = area[lab]
        seed = np.zeros_like(Gf)
        if npatch:
            rows, cols = np.indices(G.shape)
            cy = np.bincount(lab.ravel(), weights=rows.ravel())[1:] / area[1:]
            cx = np.bincount(lab.ravel(), weights=cols.ravel())[1:] / area[1:]
            np.add.at(seed, (np.rint(cy).astype(int), np.rint(cx).astype(int)), 1.0)
        feats["green_patcharea"] = np.where(G, np.log10(np.maximum(own, 1.0)), np.nan)
        for name, ww in win.items():
            N = msum(vf, ww)
            ok = N / (ww * ww) >= VALID_MIN
            feats[f"green_frac_{name}"] = np.where(ok, msum(Gf, ww) / N, np.nan)
            feats[f"green_edgedens_{name}"] = np.where(ok, msum(E, ww) / N, np.nan)
            feats[f"green_patchdens_{name}"] = np.where(ok, msum(seed, ww) / N * 1e4, np.nan)
            sg = msum(Gf, ww)
            feats[f"green_awmps_{name}"] = np.where(ok & (sg > 0.5), np.log10(msum(Gf * own, ww) / sg), np.nan)
        # regularity of tree planting
        Tf = ((chm >= TREE_H) & vc).astype(float)
        vcf = vc.astype(float)
        for name, ww in win.items():
            Nv = msum(vcf, ww)
            ok = Nv / (ww * ww) >= VALID_MIN
            feats[f"tree_frac_{name}"] = np.where(ok, msum(Tf, ww) / Nv, np.nan)
            rs = {}
            for L in LAGS:
                h1, h2 = L // 2, L - L // 2
                for ax, (sy, sx) in enumerate(((1, 0), (0, 1))):
                    a = shifted(Tf, -h1 * sy, -h1 * sx)
                    b = shifted(Tf, h2 * sy, h2 * sx)
                    pv = shifted(vcf, -h1 * sy, -h1 * sx) * shifted(vcf, h2 * sy, h2 * sx)
                    N = msum(pv, ww)
                    Ea, Eb, Eab = msum(a * pv, ww) / N, msum(b * pv, ww) / N, msum(a * b * pv, ww) / N
                    den = np.sqrt(np.clip(Ea * (1 - Ea), 0, None) * np.clip(Eb * (1 - Eb), 0, None))
                    r = np.where(den > 1e-9, (Eab - Ea * Eb) / den, 0.0)
                    rs[(L, ax)] = np.where(ok & (N / (ww * ww) >= VALID_MIN), r, np.nan)
            for L in (5, 10, 15):
                feats[f"tree_acf{L}_{name}"] = (rs[(L, 0)] + rs[(L, 1)]) / 2.0
            st = np.stack(list(rs.values()))
            feats[f"tree_acfpeak_{name}"] = np.max(st, axis=0)
            feats[f"tree_acfcontrast_{name}"] = np.max(st, axis=0) - np.min(st, axis=0)
        # spatial-statistics family on NDVI
        feats.update(spatial_stats(nd, v, k["ndvi_mu"], k["ndvi_sd"], k["ndvi_lo"], k["ndvi_hi"], win, "ndvi"))
    # sampled pixels (padded coordinates)
    out = {"tile": (r0, c0)}
    csk_vals = {}
    with rasterio.open(CSK_RAW) as s:
        tr = s.transform
        Mc = max(cwin.values()) // 2 + 3
        rr, cc = csk_index(np.array([r0, r0 + h - 1]), np.array([c0, c0 + w - 1]), tr)
        cr0, cc0 = int(rr[0]) - Mc, int(cc[0]) - Mc
        ch, cw = int(rr[1] - rr[0] + 1) + 2 * Mc, int(cc[1] - cc[0] + 1) + 2 * Mc
        dn = read_pad(s, 1, cr0, cc0, ch, cw, 0).astype(float)
    vk = dn > 0
    vkf = vk.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = np.where(vk, 20.0 * np.log10(np.where(vk, dn, 1.0)), np.nan)
        cfe = {"csk_db": db}
        cv_sub = {}
        coreK = np.zeros(vk.shape, bool)
        coreK[Mc:ch - Mc, Mc:cw - Mc] = True
        for name, ww in cwin.items():
            N = msum(vkf, ww)
            m = msum(dn, ww) / N
            m2 = msum(dn * dn, ww) / N
            cv = np.sqrt(np.clip(m2 - m * m, 0, None)) / m
            cv = np.where(vk & (N / (ww * ww) >= VALID_MIN), cv, np.nan)
            cfe[f"csk_cv_{name}"] = cv
            cvs = cv[coreK & np.isfinite(cv)]
            r_ = np.random.default_rng([SEED, r0, c0, 3])
            cv_sub[name] = r_.choice(cvs, size=min(1000, cvs.size), replace=False) if cvs.size else cvs
        cfe.update(spatial_stats(db, vk, k["csk_mu"], k["csk_sd"], k["csk_lo"], k["csk_hi"], cwin, "csk"))
    for cls in (1, 0):
        idx = pick[cls]
        rr_, cc_ = np.divmod(idx, w)
        d = {f: a[rr_ + M, cc_ + M] for f, a in feats.items()}
        kr, kc = csk_index(rr_ + r0, cc_ + c0, tr)
        kr, kc = kr - cr0, kc - cc0
        inside = (kr >= 0) & (kr < ch) & (kc >= 0) & (kc < cw)
        for f, a in cfe.items():
            vals = np.full(idx.size, np.nan)
            vals[inside] = a[kr[inside], kc[inside]]
            d[f] = vals
        d["row"], d["col"] = rr_ + r0, cc_ + c0
        d["group"] = grp[rr_, cc_]
        out[cls] = d
    out["cv_sub"] = cv_sub
    out["cpu"] = time.process_time() - c0_
    return out


FAMILIES = [("ndvi", "reference"), ("chm", "reference"), ("csk_db", "reference_radar"),
            ("ndvi_coherence", "linearity"), ("ndvi_orientcons", "linearity"), ("chm_coherence", "linearity"),
            ("chm_orientcons", "linearity"), ("green_", "configuration"), ("tree_frac", "configuration"),
            ("tree_acf", "regularity"), ("ndvi_moran", "spatial_stats_ndvi"), ("ndvi_geary", "spatial_stats_ndvi"),
            ("ndvi_gistar", "spatial_stats_ndvi"), ("ndvi_glcm", "spatial_stats_ndvi"),
            ("csk_cv", "radar_heterogeneity"), ("csk_specdiv", "radar_heterogeneity"),
            ("csk_moran", "spatial_stats_csk"), ("csk_geary", "spatial_stats_csk"),
            ("csk_gistar", "spatial_stats_csk"), ("csk_glcm", "spatial_stats_csk")]


def describe(feat, win, cwin):
    refs = {p: f for p, f in FAMILIES if f.startswith("reference")}
    fam = refs.get(feat) or next(f for p, f in FAMILIES if not f.startswith("reference") and feat.startswith(p))
    name = feat.rsplit("_", 1)[-1]
    if name in win:
        csk = feat.startswith("csk_")
        px = cwin[name] if csk else win[name]
        return fam, name, px * (2.5 if csk else 1.0), px, "CSK 2.5 m" if csk else "GRID_1M"
    if feat == "csk_db":
        return fam, "", 2.5, 1, "CSK 2.5 m"
    return fam, "", 1.0, 1, "GRID_1M"


def addon_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer
    return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="uniform", random_state=SEED),
                         SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=1000))


def inner_auc(X, y, fold, held):
    """Pooled out-of-fold AUROC over the three training folds (held-out fold `held` excluded)."""
    from sklearn.metrics import roc_auc_score
    tr = fold != held
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    p = np.empty(yt.size)
    for j in np.unique(ft):
        m = ft == j
        mdl = addon_model().fit(Xt[~m], yt[~m])
        p[m] = mdl.predict_proba(Xt[m])[:, 1]
    return roc_auc_score(yt, p)


_DATA = {}


def _init(data):
    _DATA.update(data)


def finite_key(feat):
    X = _DATA["X"]
    fin = np.isfinite(X[feat]) & np.isfinite(X["ndvi"]) & np.isfinite(X["chm"])
    return fin, hashlib.sha1(np.packbits(fin).tobytes()).hexdigest()


def addon_task(job):
    """job = ("base", feat): NDVI and NDVI + CHM models on the rows where feat, NDVI and CHM are finite;
    job = ("aug", feat): the same models plus feat."""
    c0_ = time.process_time()
    kind, feat = job
    y, fold = _DATA["y"], _DATA["fold"]
    x, ndv, chm = _DATA["X"][feat], _DATA["X"]["ndvi"], _DATA["X"]["chm"]
    fin, _ = finite_key(feat)
    res = {}
    for k in (1, 2, 3, 4):
        for tag, cols in (("ndvi", [ndv]), ("ndvi_chm", [ndv, chm])):
            cols = cols + ([x] if kind == "aug" else [])
            res[(tag, k)] = inner_auc(np.c_[tuple(c[fin] for c in cols)], y[fin], fold[fin], k)
    return job, res, time.process_time() - c0_


def run_screen():
    from sklearn.metrics import roc_auc_score
    t_wall, t_cpu = time.time(), time.process_time()
    started = dt.datetime.now().isoformat(timespec="seconds")
    sc = json.loads((OUT / "scales.json").read_text(encoding="utf-8"))["windows"]
    win = {f"{sc['w_s_px_1m']}m": sc["w_s_px_1m"], f"{sc['w_l_px_1m']}m": sc["w_l_px_1m"]}
    cwin = {f"{sc['w_s_px_1m']}m": sc["csk_px_2p5m"]["w_s"], f"{sc['w_l_px_1m']}m": sc["csk_px_2p5m"]["w_l"]}
    tiles = load_tiles()
    rng = np.random.default_rng(SEED)
    sel = {}
    for f in (1, 2, 3, 4):
        g = tiles[tiles["fold"] == f]
        gp = g[g["ugi_px"] > 0]
        sel[(f, 1)] = gp.iloc[np.sort(rng.choice(len(gp), size=min(K_TILES, len(gp)), replace=False))].index.tolist()
        sel[(f, 0)] = g.iloc[np.sort(rng.choice(len(g), size=min(K_TILES, len(g)), replace=False))].index.tolist()
    uniq = sorted(set(i for v in sel.values() for i in v))
    tasks = [(int(tiles.at[i, "row0"]), int(tiles.at[i, "col0"]), int(tiles.at[i, "h"]), int(tiles.at[i, "w"]))
             for i in uniq]
    cpu_workers = 0.0
    with ProcessPoolExecutor(WORKERS) as ex:
        A = {i: r for i, r in zip(uniq, ex.map(pass_a, tasks))}
    cpu_workers += sum(r["cpu"] for r in A.values())
    print("pass A", len(uniq), "tiles", round(time.time() - t_wall, 1), "s", flush=True)
    nd_all = np.concatenate([r["ndvi"] for r in A.values()])
    db_all = np.concatenate([r["csk_db"] for r in A.values()])
    k = {"ndvi_mu": float(nd_all.mean()), "ndvi_sd": float(nd_all.std()),
         "ndvi_lo": float(np.percentile(nd_all, 1)), "ndvi_hi": float(np.percentile(nd_all, 99)),
         "csk_mu": float(db_all.mean()), "csk_sd": float(db_all.std()),
         "csk_lo": float(np.percentile(db_all, 1)), "csk_hi": float(np.percentile(db_all, 99)),
         "n_ndvi_values": int(nd_all.size), "n_csk_values": int(db_all.size)}
    # allocation: N_PER per fold and class, uniform over the pooled eligible pixels of the sampled tiles
    alloc = {i: [0, 0] for i in uniq}
    pools = {}
    for (f, cls), idx in sel.items():
        cnt = np.array([A[i]["n_pos" if cls == 1 else "n_neg"] for i in idx])
        pools[f"fold{f}_{'pos' if cls == 1 else 'neg'}"] = {"tiles": len(idx), "eligible_pixels": int(cnt.sum()),
                                                            "tiles_with_eligible": int((cnt > 0).sum())}
        n = min(N_PER, int(cnt.sum()))
        draw = rng.multivariate_hypergeometric(cnt, n)
        for i, d in zip(idx, draw):
            alloc[i][cls] += int(d)
    tasks_b = [(int(tiles.at[i, "row0"]), int(tiles.at[i, "col0"]), int(tiles.at[i, "h"]), int(tiles.at[i, "w"]),
                alloc[i][1], alloc[i][0], win, cwin, k) for i in uniq if sum(alloc[i]) > 0]
    fold_of_tile = {(int(tiles.at[i, "row0"]), int(tiles.at[i, "col0"])): int(tiles.at[i, "fold"]) for i in uniq}
    t_b = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        B = list(ex.map(pass_b, tasks_b))
    cpu_workers += sum(r["cpu"] for r in B)
    print("pass B", len(tasks_b), "tiles", round(time.time() - t_b, 1), "s", flush=True)
    frames = []
    for r in B:
        for cls in (1, 0):
            d = pd.DataFrame(r[cls])
            if d.empty:
                continue
            d.insert(0, "cls", cls)
            d.insert(0, "fold", fold_of_tile[r["tile"]])
            d.insert(2, "tile_row0", r["tile"][0])
            d.insert(3, "tile_col0", r["tile"][1])
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cv_h = {}
    for name in cwin:
        allcv = np.concatenate([r["cv_sub"][name] for r in B])
        cv_h[name] = float(np.percentile(allcv, 5))
        df[f"csk_specdiv_{name}"] = df[f"csk_cv_{name}"] - cv_h[name]
    k["csk_cv_homogeneous_p5"] = cv_h
    meta_cols = ["fold", "cls", "tile_row0", "tile_col0", "row", "col", "group"]
    feats = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feats]
    df.to_csv(OUT / "samples.csv", index=False, float_format="%.6g")
    y = df["cls"].to_numpy()
    fold = df["fold"].to_numpy()
    # univariate AUROC per training-fold set
    rows = []
    for f in feats:
        x = df[f].to_numpy(float)
        fin = np.isfinite(x)
        raw = []
        npos, nneg = [], []
        for kk in (1, 2, 3, 4):
            m = fin & (fold != kk)
            raw.append(roc_auc_score(y[m], x[m]))
            npos.append(int((y[m] == 1).sum()))
            nneg.append(int((y[m] == 0).sum()))
        raw = np.array(raw)
        sign = 1 if raw.mean() >= 0.5 else -1
        ori = raw if sign > 0 else 1 - raw
        fam, wname, wm, wpx, grid = describe(f, win, cwin)
        rows.append({"feature": f, "family": fam, "window": wname, "window_m": wm, "window_px": wpx, "grid": grid,
                     **{f"auc_wo_fold{kk}": round(float(a), 4) for kk, a in zip((1, 2, 3, 4), ori)},
                     "auc_mean": round(float(ori.mean()), 4), "auc_min": round(float(ori.min()), 4), "sign": sign,
                     "sign_consistent": bool(np.all((raw - 0.5) * sign > 0)),
                     "n_pos_min": min(npos), "n_neg_min": min(nneg), "nan_share": round(float(1 - fin.mean()), 4)})
    res = pd.DataFrame(rows)
    print("univariate AUROC done", round(time.time() - t_wall, 1), "s", flush=True)
    # add-on check over NDVI and over NDVI + CHM
    t_c = time.time()
    data = {"y": y, "fold": fold, "X": {f: df[f].to_numpy(float) for f in feats}}
    todo = [f for f in feats if f != "ndvi"]          # chm included: its gain over NDVI is a reference
    _init(data)
    keys = {f: finite_key(f)[1] for f in todo}
    rep = {}
    for f in todo:
        rep.setdefault(keys[f], f)
    jobs = [("base", f) for f in rep.values()] + [("aug", f) for f in todo]
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(data,)) as ex:
        AD = list(ex.map(addon_task, jobs))
    cpu_workers += sum(c for _, _, c in AD)
    print("add-on check", len(todo), "features,", len(rep), "row sets", round(time.time() - t_c, 1), "s", flush=True)
    base = {keys[job[1]]: r for job, r, _ in AD if job[0] == "base"}
    add = {}
    for job, r, _ in AD:
        if job[0] != "aug":
            continue
        feat = job[1]
        b = base[keys[feat]]
        d = {}
        for tag in ("ndvi", "ndvi_chm"):
            dl = [r[(tag, kk)] - b[(tag, kk)] for kk in (1, 2, 3, 4)]
            for kk, v in zip((1, 2, 3, 4), dl):
                d[f"dauc_{tag}_wo_fold{kk}"] = round(float(v), 4)
            d[f"dauc_{tag}_min"] = round(float(min(dl)), 4)
            d[f"dauc_{tag}_mean"] = round(float(np.mean(dl)), 4)
            d[f"auc_{tag}_base_mean"] = round(float(np.mean([b[(tag, kk)] for kk in (1, 2, 3, 4)])), 4)
            d[f"auc_{tag}_aug_mean"] = round(float(np.mean([r[(tag, kk)] for kk in (1, 2, 3, 4)])), 4)
        if feat == "chm":                              # NDVI + CHM + CHM against NDVI + CHM: not meaningful
            d = {kk: (np.nan if "ndvi_chm" in kk else v) for kk, v in d.items()}
        add[feat] = d
    addf = pd.DataFrame.from_dict(add, orient="index")
    res = res.merge(addf, left_on="feature", right_index=True, how="left")
    res["separation_pass"] = res["sign_consistent"] & (res["auc_min"] >= 0.60)
    res["addon_pass"] = res["dauc_ndvi_min"] >= 0.02
    res["keep_proposed"] = res["separation_pass"] & res["addon_pass"] & ~res["family"].str.startswith("reference")
    res["note"] = ""
    for name in cwin:
        res.loc[res["feature"] == f"csk_specdiv_{name}", "note"] = (
            f"= csk_cv_{name} - {cv_h[name]:.4f} (scene constant): same ranking and AUROC as csk_cv_{name}")
    for name in win:
        res.loc[res["feature"] == f"ndvi_gistar_{name}", "note"] = "monotone in the local mean NDVI of the window"
    res = res.sort_values(["keep_proposed", "auc_min"], ascending=[False, False])
    res.to_csv(OUT / "auroc.csv", index=False)
    # AUROC by UGI group (positives of one group against all negatives), oriented by the overall sign
    grp_rows = []
    sgn = dict(zip(res["feature"], res["sign"]))
    for f in feats:
        x = df[f].to_numpy(float)
        fin = np.isfinite(x)
        for gcode, gname in GROUPS.items():
            vals = []
            for kk in (1, 2, 3, 4):
                m = fin & (fold != kk) & ((y == 0) | (df["group"].to_numpy() == gcode))
                if (y[m] == 1).sum() < 50:
                    continue
                a = roc_auc_score(y[m], x[m])
                vals.append(a if sgn[f] > 0 else 1 - a)
            if vals:
                grp_rows.append({"feature": f, "group": gname, "auc_mean": round(float(np.mean(vals)), 4),
                                 "auc_min": round(float(np.min(vals)), 4), "fold_sets": len(vals),
                                 "n_pos_total": int(((y == 1) & fin & (df["group"].to_numpy() == gcode)).sum())})
    pd.DataFrame(grp_rows).to_csv(OUT / "auroc_by_group.csv", index=False)
    # redundancy among kept features and with the references (Spearman, all samples)
    keep = res.loc[res["keep_proposed"], "feature"].tolist()
    cols = ["ndvi", "chm"] + keep
    sub = df[cols].sample(n=min(40000, len(df)), random_state=SEED)
    rho = sub.rank().corr(method="pearson")
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({"feature_a": a, "feature_b": b, "spearman": round(float(rho.at[a, b]), 4)})
    pd.DataFrame(pairs).sort_values("spearman", key=np.abs, ascending=False).to_csv(OUT / "kept_spearman.csv",
                                                                                     index=False)
    log = {"started": started, "finished": dt.datetime.now().isoformat(timespec="seconds"),
           "command": "python scripts/u04_texture_screen.py --screen (from work_ms/, PYTHONUTF8=1, CUDA_VISIBLE_DEVICES=-1)",
           "seed": SEED, "k_tiles_per_fold_class": K_TILES, "n_per_fold_class": N_PER, "workers": WORKERS,
           "windows_1m_px": win, "windows_csk_px": cwin, "lags_m": LAGS, "glcm_levels": GLCM_L, "green_ndvi": GREEN,
           "tree_height_m": TREE_H, "valid_min": VALID_MIN, "constants": k, "pools": pools,
           "tiles_processed": len(tasks_b), "tiles_sampled_unique": len(uniq),
           "samples_by_fold_class": {f"fold{f}_{'pos' if c == 1 else 'neg'}": int(((fold == f) & (y == c)).sum())
                                     for f in (1, 2, 3, 4) for c in (1, 0)},
           "positives_by_group": {GROUPS[g]: int(((y == 1) & (df["group"] == g)).sum()) for g in GROUPS},
           "features": len(feats), "kept_proposed": keep,
           "keep_rule": "sign consistent in the four training-fold sets and oriented AUROC >= 0.60 in each "
                        "(|AUROC - 0.5| >= 0.10), and min over the four sets of the inner-CV AUROC gain of NDVI + feature "
                        "over NDVI alone >= 0.02; references excluded",
           "inputs": {"neo_toa": str(NEO_TOA.relative_to(PROJECT)), "neo_indices": str(NEO_IDX.relative_to(PROJECT)),
                      "neo_cloud": str(NEO_CLOUD.relative_to(PROJECT)), "chm": str(CHM.relative_to(PROJECT)),
                      "csk_raw": str(CSK_RAW.relative_to(PROJECT)), "ugi_binary": str(UGI_BIN.relative_to(PROJECT)),
                      "ugi_edge": str(UGI_EDGE.relative_to(PROJECT)), "ugi_group": str(UGI_GROUP.relative_to(PROJECT)),
                      "tiles": str(TILES.relative_to(PROJECT)),
                      "sha256": {"chm": sha256(CHM), "ugi_binary": sha256(UGI_BIN), "ugi_edge": sha256(UGI_EDGE),
                                 "ugi_group": sha256(UGI_GROUP), "tiles": sha256(TILES)},
                      "sha256_recorded_elsewhere": {
                          "neo_indices": "derived/features/features_log.json (neo.sha256)",
                          "neo_toa": "derived/features/features_log.json (neo.source_sha256)",
                          "csk_raw": "derived/features/features_log.json (csk.source_sha256)"}},
           "seconds_wall": round(time.time() - t_wall, 1),
           "seconds_cpu_main": round(time.process_time() - t_cpu, 1),
           "seconds_cpu_workers": round(cpu_workers, 1)}
    log["seconds_cpu_total"] = round(log["seconds_cpu_main"] + log["seconds_cpu_workers"], 1)
    (OUT / "run_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(json.dumps({kk: log[kk] for kk in ("samples_by_fold_class", "kept_proposed", "seconds_wall",
                                             "seconds_cpu_total")}, indent=1))


# ------------------------------------------------------------------------------------------------------------ README
def run_readme():
    sc = json.loads((OUT / "scales.json").read_text(encoding="utf-8"))
    pan = json.loads((OUT / "pan_extraction_size.json").read_text(encoding="utf-8"))
    log = json.loads((OUT / "run_log.json").read_text(encoding="utf-8"))
    res = pd.read_csv(OUT / "auroc.csv")
    grp = pd.read_csv(OUT / "auroc_by_group.csv")
    rho = pd.read_csv(OUT / "kept_spearman.csv")
    w = sc["windows"]
    nw = sc["narrow_ugi_width_m"]
    cr = sc["crowns"]
    L = []
    L.append("# Texture screen for UGI segmentation (U0.4, CLAUDE.md 13.1)\n")
    L.append(f"`python scripts/u04_texture_screen.py --scales | --pan | --screen | --readme` (from `work_ms/`, "
             f"`PYTHONUTF8=1`, CPU only), 2 Nov 2026; screen finished {log['finished']} (`run_log.json`). "
             "Question: which texture features separate inventory UGI from other green on the training folds. "
             "Every number below is read from the files of this folder.\n")
    L.append("## UGI scales and windows (`scales.json`)\n")
    a = nw["all_narrow"]
    L.append(f"- **Narrow UGI width** (2A/P, CANTERO and FRANJA, EPSG:32720, {a['n']} polygons): median "
             f"{a['median']} m, IQR {a['q25']}-{a['q75']} m, p5-p95 {a['p5']}-{a['p95']} m. By type: " +
             "; ".join(f"{t} {s['n']} polygons, median {s['median']} m (IQR {s['q25']}-{s['q75']})"
                       for t, s in nw["by_type"].items()) + ". By fold (centroid): " +
             ", ".join(f"fold {f} {s['median']} m" for f, s in nw["by_fold_centroid"].items()) + ".")
    c = cr["components_equivalent_diameter_m"]
    t = cr["tree_top_spacing_m"]
    L.append(f"- **Crowns** (Meta/WRI CHM >= {cr['tree_threshold_m']} m, {len(cr['tiles'])} tiles, 10 per fold): "
             f"connected components, equivalent diameter median {c['all']['median']} m (IQR {c['all']['q25']}-"
             f"{c['all']['q75']}, n {c['all']['n']}; inside UGI {c['inside_ugi_majority']['median']} m, n "
             f"{c['inside_ugi_majority']['n']}); components merge touching crowns, so this is an upper-side "
             f"estimate. Tree-top spacing (nearest other top) median {t['all']['median']} m (IQR {t['all']['q25']}-"
             f"{t['all']['q75']}, n {t['all']['n']}; inside UGI {t['inside_ugi']['median']} m).")
    L.append(f"- **Rule.** {w['rule']}")
    L.append(f"- **Windows.** w_s = {w['w_s_px_1m']} px (1 m), w_l = {w['w_l_px_1m']} px (1 m); COSMO-SkyMed "
             f"{w['csk_px_2p5m']['w_s']} and {w['csk_px_2p5m']['w_l']} px of 2.5 m "
             f"({w['csk_px_2p5m']['w_s'] * 2.5} and {w['csk_px_2p5m']['w_l'] * 2.5} m). Crown spacing inside the "
             f"5-15 m lag range: {w['crown_spacing_inside_lag_range']}.\n")
    L.append("## Method (`run_log.json`)\n")
    sb = log["samples_by_fold_class"]
    L.append(f"- **Tiles:** non-straddling 512 px tiles of `derived/ugi_tiles/tiles.csv`; per fold and class a seeded "
             f"random sample of {log['k_tiles_per_fold_class']} tiles (positives from tiles with UGI, negatives from "
             f"any tile); {log['tiles_processed']} distinct tiles processed. Features computed on the tile plus a "
             "margin, read at the sampled pixels only.")
    L.append(f"- **Pixels:** {log['n_per_fold_class']} per fold and class, uniform over the pooled eligible pixels "
             "of the sampled tiles. Positives UGI = 1, edge band = 0, NDVI >= 0.3, valid clear Neo; negatives "
             "UGI = 0, edge = 0, NDVI >= 0.3, valid clear Neo. Drawn: " +
             ", ".join(f"{kk} {v}" for kk, v in sb.items()) + ". Positives by group: " +
             ", ".join(f"{kk} {v}" for kk, v in log["positives_by_group"].items()) + ".")
    L.append("- **Features** (window names give the 1 m window; COSMO-SkyMed features use the 2.5 m window in "
             "`window_px`):")
    L.append("  - references: Neo NDVI, Meta CHM, CSK 20 log10 DN at the pixel;")
    L.append("  - linearity: structure-tensor coherence (Sobel gradients, tensor averaged over the window) and "
             "orientation consistency (resultant length of the doubled fine-scale orientation, sigma 1.5 px) on NDVI "
             "and on the CHM;")
    L.append("  - configuration (green mask NDVI >= 0.3): green fraction, edge density (m of green/non-green edge per "
             "m2), patch density (8-connected patches per ha, by centroid), area-weighted mean patch size (log10 m2), "
             "own patch size (log10 m2); tree fraction (CHM >= 3 m);")
    L.append("  - regularity: correlation of the tree mask with itself at lags 5, 10, 15 m (mean of rows and columns), "
             "the peak over lags 5-15 m and both axes, and peak minus trough;")
    L.append("  - spatial statistics on NDVI and on CSK dB: local Moran's I, Geary's c, Getis-Ord Gi* (global mean "
             "and SD from the sampled tiles), GLCM variance and correlation (16 levels between the 1st and 99th "
             "percentiles, distance 1, four directions averaged, from moving-window moments);")
    L.append("  - COSMO-SkyMed heterogeneity: local coefficient of variation of the amplitude (DN) and speckle "
             "divergence = CV minus the CV of homogeneous areas (5th percentile of the local CV in the sampled tiles: "
             + ", ".join(f"{kk} {v:.4f}" for kk, v in log["constants"]["csk_cv_homogeneous_p5"].items()) +
             "); a scene constant, so its AUROC equals the CV's.")
    L.append("- **AUROC:** for each held-out fold k, `roc_auc_score` on the pooled samples of the other three folds "
             "(columns `auc_wo_foldk`). Oriented: sign +1 if higher in UGI on average over the four sets, and "
             "1 - AUROC otherwise. The held-out fold is never used.")
    L.append("- **Add-on check:** inside each training-fold set, leave-one-fold-out over its three folds; rank-"
             "transformed, cubic-spline-expanded logistic regression; pooled out-of-fold AUROC of NDVI + feature minus "
             "NDVI alone (`dauc_ndvi_*`) and of NDVI + CHM + feature minus NDVI + CHM (`dauc_ndvi_chm_*`).\n")
    L.append("## Results (`auroc.csv`, oriented AUROC)\n")
    ref = res[res["family"].str.startswith("reference")]
    top = res[~res["family"].str.startswith("reference")].sort_values("auc_min", ascending=False).head(20)
    cols = ["feature", "family", "window_px", "grid", "auc_wo_fold1", "auc_wo_fold2", "auc_wo_fold3", "auc_wo_fold4",
            "auc_mean", "auc_min", "sign", "dauc_ndvi_min", "dauc_ndvi_chm_min", "keep_proposed"]

    def table(d):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in d.itertuples():
            out.append("| " + " | ".join(str(getattr(r, cc)) for cc in cols) + " |")
        return out
    L.append("References:\n")
    L += table(ref)
    L.append("\nTop 20 candidates by the minimum oriented AUROC over the four training-fold sets:\n")
    L += table(top)
    L.append(f"\nAll {len(res)} features are in `auroc.csv`; AUROC by UGI group in `auroc_by_group.csv`.\n")
    L.append("## Proposed keep list (proposal for Shafi, not a decision)\n")
    L.append(f"- **Rule:** {log['keep_rule']}.")
    kept = res[res["keep_proposed"]]
    if len(kept):
        L.append(f"- **Passing ({len(kept)}):** " + ", ".join(
            f"`{r.feature}` (AUROC min {r.auc_min}, dAUROC over NDVI min {r.dauc_ndvi_min}, over NDVI + CHM min "
            f"{r.dauc_ndvi_chm_min})" for r in kept.itertuples()) + ".")
    else:
        L.append("- **Passing:** none.")
    if len(kept):
        wins = sorted(set(kept["window"].astype(str)))
        L.append(f"- **Windows of the passing features:** {', '.join(wins)}; no feature at the other window passes.")
        strict = kept[kept["dauc_ndvi_chm_min"] >= 0.02]
        L.append("- **Stricter variant** (also a gain >= 0.02 over NDVI + CHM in all four sets, since the CHM enters "
                 "the model anyway): " + (", ".join(f"`{r.feature}`" for r in strict.itertuples()) if len(strict)
                                          else "none") + ".")
    nonmono = res[~res["separation_pass"] & (res["dauc_ndvi_min"] >= 0.05) & ~res["family"].str.startswith("reference")]
    if len(nonmono):
        L.append("- **Not caught by the rule:** features that fail the monotone AUROC test but add >= 0.05 over NDVI "
                 "in all four sets (a non-monotone effect, or information conditional on NDVI): " + ", ".join(
                     f"`{r.feature}` (AUROC min {r.auc_min}, dAUROC over NDVI min {r.dauc_ndvi_min}, over NDVI + CHM "
                     f"min {r.dauc_ndvi_chm_min})" for r in nonmono.itertuples()) + ". Candidates for B2 if Shafi "
                 "prefers a model-based criterion.")
    neg = res[(res["dauc_ndvi_min"] < -0.02)]
    if len(neg):
        L.append(f"- **Negative gains:** {len(neg)} features lower the inner-CV AUROC by more than 0.02 in at least one "
                 "set (relation learned on two folds does not hold on the third), e.g. " +
                 ", ".join(f"`{r.feature}` {r.dauc_ndvi_min}" for r in neg.sort_values("dauc_ndvi_min").head(4)
                           .itertuples()) + ".")
    hi = rho[(rho["spearman"].abs() >= 0.8)]
    L.append(f"- **Redundancy** (`kept_spearman.csv`, Spearman on {min(40000, sum(sb.values()))} samples): "
             f"{len(hi)} pair{'' if len(hi) == 1 else 's'} among NDVI, CHM and the kept features "
             f"{'has' if len(hi) == 1 else 'have'} |rho| >= 0.8" +
             (": " + "; ".join(f"{r.feature_a} / {r.feature_b} {r.spearman}" for r in hi.head(12).itertuples())
              if len(hi) else "") + ".")
    gk = grp[grp["feature"].isin(kept["feature"])]
    if len(gk):
        piv = gk.pivot(index="feature", columns="group", values="auc_mean")
        L.append("- **By group** (oriented AUROC, mean over fold sets, one group's positives against all negatives):\n")
        gc = list(piv.columns)
        L.append("| feature | " + " | ".join(gc) + " |")
        L.append("|---|" + "---|" * len(gc))
        for f_, r in piv.iterrows():
            L.append(f"| {f_} | " + " | ".join(str(r[g]) for g in gc) + " |")
    L.append("")
    L.append("## Neo PAN extraction size (`pan_extraction_size.json`; nothing extracted)\n")
    for s_, d in pan["strips"].items():
        extra = ""
        if "intersecting_tiles_already_unpacked_same_size" in d:
            extra = (f"; {d['intersecting_tiles_already_unpacked_same_size']} of these tiles already exist unpacked "
                     f"with the same size in `{d['unpacked_folder']}` (delivered)")
        L.append(f"- Strip {s_}: {d['pan_tiles_intersecting']} of {d['pan_tiles_total']} PAN tiles intersect the "
                 f"common extent; {d['extract_GB']} GB uncompressed, {d['extract_compressed_GB']} GB in the zip "
                 f"(tiles + TFW + PAN DIMAP XML and masks){extra}.")
    tt = pan["total"]
    L.append(f"- Total: {tt['tiles']} tiles, {tt['extract_GB']} GB uncompressed, {tt['extract_compressed_GB']} GB "
             "compressed.")
    ts = pan.get("total_still_to_extract")
    if ts:
        L.append(f"- Still to extract (strips whose intersecting PAN tiles are not already unpacked: "
                 f"{', '.join(ts['strips'])}): {ts['tiles']} tiles, {ts['extract_GB']} GB uncompressed, "
                 f"{ts['extract_compressed_GB']} GB compressed.")
    L.append("- Extraction needs Shafi's GO (CLAUDE.md 13.1); lawn homogeneity at 0.3 m is not screened.\n")
    L.append("## Limitations\n")
    L.append(f"- Cluster sample: pixels come from {log['k_tiles_per_fold_class']} tiles per fold and class, so "
             "neighbouring samples are correlated; AUROCs are point estimates without confidence intervals.")
    pg = log["positives_by_group"]
    L.append(f"- The pixel sample is area-weighted: water corridors give {pg['water_corridors']} of "
             f"{sum(pg.values())} positives and road greens {pg['road_greens']}, so the pooled AUROC mostly reflects "
             "large riverside green; `auroc_by_group.csv` shows each group separately.")
    L.append("- The monotone AUROC criterion misses non-monotone separators; the spline add-on check catches some "
             "(see 'Not caught by the rule').")
    L.append("- Gi* is monotone in the local mean of the window (CSK Gi* = local mean of 20 log10 DN, lower in UGI).")
    L.append("- Epochs differ: Neo October 2022, Meta CHM (image dates vary), COSMO-SkyMed July 2025. A UGI built or "
             "cleared between dates weakens the CSK and CHM features; no change mask is applied here.")
    L.append("- COSMO-SkyMed is uncalibrated DN (20 log10 DN); only relative statistics are used. Pixels outside its "
             "footprint are NaN and excluded per feature (`nan_share`).")
    L.append("- Green patches are labelled on the tile plus margin, so patches larger than the read window are "
             "truncated (patch size and area-weighted mean patch size are underestimated for large parks).")
    L.append("- Global constants for Moran's I, Gi* and GLCM quantisation come from the sampled tiles of all four "
             "folds (no labels involved).")
    L.append("- Windows are set from scales measured on all four folds (inventory widths and CHM); the per-fold "
             "widths in `scales.json` show how much they vary.")
    L.append("- The add-on check measures gain over NDVI and NDVI + CHM only, not over the other texture features or "
             "the multispectral bands; redundancy among kept features is shown in `kept_spearman.csv`.")
    L.append("- Lawn homogeneity at 0.3 m (Neo PAN) is not screened (PAN not extracted).\n")
    L.append("## Runtime\n")
    L.append(f"- Screen: {log['seconds_wall']} s wall clock, {log['seconds_cpu_total']} s CPU "
             f"({log['workers']} worker processes; main {log['seconds_cpu_main']} s, workers "
             f"{log['seconds_cpu_workers']} s). Scales: {sc['seconds_wall']} s. PAN listing: {pan['seconds']} s. A first "
             "--screen attempt was stopped during pass B to fix the feature-family labels; its CPU time is not in "
             "`run_log.json`.")
    L.append("\n## Files\n")
    L.append("`scales.json`, `pan_extraction_size.json`, `samples.csv` (every sampled pixel with fold, class, tile, "
             "GRID_1M row and column, UGI group and all features), `auroc.csv`, `auroc_by_group.csv`, "
             "`kept_spearman.csv`, `run_log.json`, this README; script `work_ms/scripts/u04_texture_screen.py`.")
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("README written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", action="store_true")
    ap.add_argument("--pan", action="store_true")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--readme", action="store_true")
    a = ap.parse_args()
    if a.scales:
        run_scales()
    if a.pan:
        run_pan()
    if a.screen:
        run_screen()
    if a.readme:
        run_readme()
