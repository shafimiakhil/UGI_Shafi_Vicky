"""Zonal join (CLAUDE.md Section 10 point a: every output joined to UGI polygon ids, census-tract ids and the 452
thermal cells). Statistics of raster bands per polygon id for rasters on any project grid (0.5 m, 3 m, 10 m, 25 m,
30 m, 1 km), and the area-weighted 10 m -> 30 m resampling of Sentinel-2.

How a polygon meets a raster cell:
  - A fine lattice is laid over the raster: the raster's own grid when its pixel is <= 3 m, otherwise the raster grid
    refined by an integer factor to the largest of 3, 2.5, 2, 1, 0.5 m that divides the pixel (30 m -> 3 m; 10 m, 25 m
    and 1 km -> 2.5 m). Every fine pixel whose centre lies in a polygon belongs to it (rasterio.features.rasterize);
    polygons that overlap are rasterised in separate passes, so each polygon gets its whole area and overlapping area
    counts for both. The weight of a (cell, polygon) pair is its fine pixels / factor^2: the overlap area in cell
    units, with the precision of the fine pixel.
  - Polygons with fewer than `exact_below` fine pixels (default 25, i.e. smaller than about 225 m^2 at 3 m) get exact
    weights instead: the area of the polygon intersected with each raster cell, divided by the cell area. So small
    polygons are area-weighted on every grid, including the 0.5 m and 3 m grids, and no polygon that touches the
    raster is lost.
  - Nodata as in itareo.grid: the raster's nodata tag, or an explicit value when the file has none (NaN is always
    invalid). Nodata cells count in cells_total, never in a statistic; valid zeros are valid.
Per id: polygon_area_m2, inside_raster_frac (share of the polygon inside the raster extent), exact_weights, cells_total
and cells_valid (summed weights in raster-cell units, over the part inside the raster), area_valid_m2, valid_frac
(= cells_valid / cells_total), mean, min, max, pNN (weighted percentile, inverted CDF: the smallest value whose
cumulative valid weight reaches NN % of the valid weight), frac_gt_T and frac_ge_T (valid weight above / at or above
T, divided by the valid weight; written as frac_gt:T on the command line). Ids with no weight (empty geometry, or
outside the raster) keep cells_total 0 and empty statistics. With several bands the columns are prefixed with the
band description (or b<index>) and "__"; the weights are computed once.
Memory: the (cell, polygon) pairs stay in memory, and the raster is read on the window spanned by the pairs; a 0.5 m
raster with large polygons makes one pair per pixel, so aggregate 0.5 m class maps with itareo.aggregate first.
Geometries: Polygon and MultiPolygon only (a GeometryCollection keeps its polygonal parts; other types are refused);
invalid geometries are repaired with shapely.make_valid(method="structure") and counted.

Sentinel-2 10 m -> 30 m (point a): area_weighted_resample() gives each 30 m cell the exact area-weighted mean of the
10 m pixels overlapping it (separable 1-D overlaps along x and y, as sparse matrices), over valid pixels only. It
writes <dst>_valid_frac.tif = the valid area fraction of the cell, taken as the minimum over the bands when their masks
differ (area outside the source counts as invalid); each band's cells below min_valid, or without any valid pixel, are
nodata. The source nodata tag is kept (float32 output). A JSON summary is written next to every output.

From work_ms/:
  python -m itareo.zonal --raster <tif> --polygons <shp|gpkg> --id <field> --out <csv>
         [--band 1 [2 ...] | --all-bands] [--nodata V] [--stats mean min max p10 p50 p90 frac_gt:0.5] [--fine-res 3]
  python -m itareo.zonal --s2-to-30m <src.tif> --out <dst.tif> [--min-valid 0.5]
  python -m itareo.zonal --selftest
"""
import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy import sparse
from shapely.geometry import MultiPolygon, Polygon, box

from itareo import EVIDENCE, W
from itareo.grid import COARSE_30M, COMMON, Grid, grid_of

FINE_CANDIDATES_M = (3.0, 2.5, 2.0, 1.0, 0.5)
DEFAULT_STATS = ("mean", "min", "max", "p10", "p50", "p90")
EXACT_BELOW = 25        # fine pixels: smaller polygons get exact intersection weights
_STAT_RE = re.compile(r"^(mean|min|max|p(\d+(?:\.\d+)?)|(frac_g[te]):(-?\d+(?:\.\d+)?))$")


# ----------------------------------------------------------------------------------------------- polygons and weights
def fine_factor(res, fine_res=None):
    """Integer refinement factor of a raster grid: 1 for pixels <= 3 m, else the largest of FINE_CANDIDATES_M that
    divides the pixel; or res / fine_res when fine_res is given. Tolerance 1e-6 relative, as Grid.offset_in."""
    if fine_res is None:
        if res <= 3.0 * (1 + 1e-6):
            return 1
        for c in FINE_CANDIDATES_M:
            f = res / c
            if abs(f - round(f)) < 1e-6 * f:
                return int(round(f))
        raise ValueError(f"no fine lattice in {FINE_CANDIDATES_M} divides a {res} m pixel; pass fine_res")
    f = res / fine_res
    if abs(f - round(f)) > 1e-6 * f or round(f) < 1:
        raise ValueError(f"fine_res {fine_res} does not divide the {res} m pixel")
    return int(round(f))


def polygonal(geom):
    """The polygonal parts of a geometry as a Polygon or MultiPolygon (empty Polygon when there is none)."""
    parts = []
    stack = [geom]
    while stack:
        g = stack.pop()
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon":
            parts.append(g)
        elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
            stack.extend(g.geoms)
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def load_polygons(polygons, id_field, crs):
    """GeoDataFrame in `crs` with unique, non-null ids, Polygon/MultiPolygon geometries only, invalid ones repaired
    with make_valid(method="structure") and reduced to their polygonal parts. Returns (gdf, number repaired)."""
    import geopandas as gpd
    gdf = polygons if hasattr(polygons, "geometry") else gpd.read_file(polygons)
    if id_field not in gdf.columns:
        raise ValueError(f"id field {id_field!r} not in {list(gdf.columns)}")
    ids = gdf[id_field]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"{id_field}: ids must be unique and non-null "
                         f"({int(ids.isna().sum())} null, {int(ids.duplicated().sum())} duplicated)")
    if gdf.crs is None:
        raise ValueError("polygon layer without CRS")
    types = gdf.geometry.geom_type
    bad_type = ~types.isin(["Polygon", "MultiPolygon", "GeometryCollection"]) & gdf.geometry.notna()
    if bad_type.any():
        raise ValueError(f"{int(bad_type.sum())} geometries are not polygons ({sorted(set(types[bad_type]))}); "
                         "a polygon layer is required")
    gdf = gdf.to_crs(crs).reset_index(drop=True)
    bad = ~gdf.geometry.is_valid | (types == "GeometryCollection").values
    if bad.any():
        gdf.loc[bad, "geometry"] = [polygonal(shapely.make_valid(g, method="structure", keep_collapsed=False))
                                    for g in gdf.loc[bad, "geometry"]]
    return gdf, int(bad.sum())


def grid_cells(grid: Grid, id_field="cell_id"):
    """The cells of `grid` as a polygon layer with ids row * width + col (stand-in for a thermal-cell layer)."""
    import geopandas as gpd
    rows, cols = np.divmod(np.arange(grid.width * grid.height), grid.width)
    x0 = grid.left + cols * grid.res
    y1 = grid.top - rows * grid.res
    geoms = shapely.box(x0, y1 - grid.res, x0 + grid.res, y1)
    return gpd.GeoDataFrame({id_field: np.arange(grid.width * grid.height), "row": rows, "col": cols},
                            geometry=list(geoms), crs=grid.crs)


def overlap_passes(gdf):
    """Groups of polygons without any area overlap inside a group (greedy, in layer order): each group is rasterised
    on its own, so overlapping polygons all keep their whole area."""
    n = len(gdf)
    geoms = gdf.geometry.values
    left, right = gdf.sindex.query(geoms, predicate="intersects")
    conflicts = [set() for _ in range(n)]
    for i, j in zip(left, right):
        if i < j and shapely.intersection(geoms[i], geoms[j]).area > 0:
            conflicts[i].add(j)
            conflicts[j].add(i)
    passes, members = [], []
    for i in range(n):
        for k, taken in enumerate(passes):
            if not (conflicts[i] & taken):
                taken.add(i)
                members[k].append(i)
                break
        else:
            passes.append({i})
            members.append([i])
    return members


@dataclass
class Weights:
    rows: np.ndarray      # raster cell row of each (cell, polygon) pair
    cols: np.ndarray      # raster cell column
    idx: np.ndarray       # polygon position in the layer (0-based)
    w: np.ndarray         # weight in cell units
    grid: Grid
    factor: int
    fine_res: float
    n_polygons: int
    n_passes: int
    exact_idx: np.ndarray  # polygon positions weighted by exact intersection areas


def exact_weights(grid: Grid, geom):
    """(rows, cols, w) of one polygon by exact intersection with the raster cells its bounds touch."""
    l, b, r, t = geom.bounds
    c0, c1 = max(0, int(math.floor((l - grid.left) / grid.res))), min(grid.width - 1, int(math.floor((r - grid.left) / grid.res)))
    r0, r1 = max(0, int(math.floor((grid.top - t) / grid.res))), min(grid.height - 1, int(math.floor((grid.top - b) / grid.res)))
    if c1 < c0 or r1 < r0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    boxes = shapely.box(grid.left + cc * grid.res, grid.top - (rr + 1) * grid.res,
                        grid.left + (cc + 1) * grid.res, grid.top - rr * grid.res)
    w = shapely.area(shapely.intersection(boxes, geom)) / grid.res ** 2
    keep = w > 0
    return rr[keep].astype(np.int64), cc[keep].astype(np.int64), w[keep]


def cell_weights(grid: Grid, gdf, fine_res=None, max_fine_pixels=2e7, exact_below=EXACT_BELOW) -> Weights:
    """(cell, polygon) weights of a polygon layer on `grid`. The layer must be in grid.crs (load_polygons)."""
    if gdf.crs is None or not gdf.crs.equals(grid.crs):
        raise ValueError(f"polygon layer CRS {gdf.crs} is not the grid CRS {grid.crs}; use load_polygons")
    f = fine_factor(grid.res, fine_res)
    fres = grid.res / f
    Wd, n = grid.width, len(gdf)
    fine_w = Wd * f
    rows_per_block = max(1, int(max_fine_pixels // max(1, fine_w * f)))
    sidx = gdf.sindex
    geoms = gdf.geometry.values
    passes = overlap_passes(gdf)
    out = {"r": [], "c": [], "i": [], "w": []}
    for r0 in range(0, grid.height, rows_per_block):
        nr = min(rows_per_block, grid.height - r0)
        top = grid.top - r0 * grid.res
        window_box = box(grid.left, top - nr * grid.res, grid.left + Wd * grid.res, top)
        cand = set(sidx.query(window_box, predicate="intersects").tolist())
        if not cand:
            continue
        for members in passes:
            sel = sorted(cand.intersection(members))
            if not sel:
                continue
            burned = features.rasterize([(geoms[i], i + 1) for i in sel], out_shape=(nr * f, fine_w),
                                        transform=from_origin(grid.left, top, fres, fres), fill=0, all_touched=False,
                                        dtype="int32")
            fr, fc = np.nonzero(burned)
            if fr.size == 0:
                continue
            ids = burned[fr, fc].astype(np.int64)
            cell = (fr // f).astype(np.int64) * Wd + (fc // f)
            uk, cnt = np.unique(cell * (n + 1) + ids, return_counts=True)
            cell_u, id_u = uk // (n + 1), uk % (n + 1)
            out["r"].append(cell_u // Wd + r0)
            out["c"].append(cell_u % Wd)
            out["i"].append(id_u - 1)
            out["w"].append(cnt / float(f * f))
    cat = {k: (np.concatenate(v) if v else np.zeros(0, np.float64 if k == "w" else np.int64)) for k, v in out.items()}
    rows, cols, idx, w = cat["r"].astype(np.int64), cat["c"].astype(np.int64), cat["i"].astype(np.int64), cat["w"].astype(np.float64)
    # small polygons: exact intersection areas instead of fine-pixel counts
    exact = np.zeros(0, np.int64)
    if exact_below > 0:
        fine_count = np.bincount(idx, weights=w, minlength=n) * f * f
        gl, gb, gr, gt = grid.bounds
        extent = box(gl, gb, gr, gt)
        small = [i for i in range(n) if fine_count[i] < exact_below and not geoms[i].is_empty
                 and geoms[i].area > 0 and geoms[i].intersects(extent)]
        if small:
            keep = ~np.isin(idx, small)
            parts = [(rows[keep], cols[keep], idx[keep], w[keep])]
            for i in small:
                er, ec, ew = exact_weights(grid, geoms[i])
                parts.append((er, ec, np.full(er.size, i, np.int64), ew))
            rows, cols, idx, w = (np.concatenate([p[k] for p in parts]) for k in range(4))
            exact = np.array(small, np.int64)
    return Weights(rows, cols, idx, w, grid, f, fres, n, len(passes), exact)


# ----------------------------------------------------------------------------------------------- statistics
def parse_stat(s):
    """mean | min | max | pNN | frac_gt:T | frac_ge:T -> (kind, argument, column name)."""
    m = _STAT_RE.match(s)
    if not m:
        raise ValueError(f"unknown statistic {s!r}; use mean, min, max, pNN, frac_gt:T or frac_ge:T")
    if m.group(2) is not None:
        q = float(m.group(2))
        if q > 100:
            raise ValueError(f"percentile out of range: {s}")
        return "p", q, s
    if m.group(3):
        return m.group(3), float(m.group(4)), f"{m.group(3)}_{m.group(4)}"
    return s, None, s


def weighted_percentile_by_group(idx, v, w, qs, n):
    """Inverted-CDF weighted percentiles per group: for each q, the smallest value whose cumulative weight (values
    sorted ascending within the group) reaches q/100 of the group's weight. NaN for empty groups."""
    order = np.lexsort((v, idx))
    idx, v, w = idx[order], v[order], w[order]
    starts = np.searchsorted(idx, np.arange(n), "left")
    ends = np.searchsorted(idx, np.arange(n), "right")
    out = np.full((len(qs), n), np.nan)
    for g in range(n):
        a, b = starts[g], ends[g]
        if b <= a:
            continue
        cw = np.cumsum(w[a:b])
        tot = cw[-1]
        for k, q in enumerate(qs):
            pos = np.searchsorted(cw, q / 100.0 * tot - 1e-12 * tot, "left")
            out[k, g] = v[a + min(pos, b - a - 1)]
    return out


def stats_from_weights(values, nodata, wts: Weights, stats=DEFAULT_STATS, row_off=0, col_off=0):
    """Per-polygon statistics of a 2-D array from precomputed weights. `values` is the raster band, or its window
    starting at (row_off, col_off) of the weights' grid. Returns a DataFrame indexed by polygon position."""
    if values.ndim != 2:
        raise ValueError(f"values must be a 2-D band, got shape {values.shape}")
    rows, cols = wts.rows - row_off, wts.cols - col_off
    if rows.size and (rows.min() < 0 or cols.min() < 0 or rows.max() >= values.shape[0] or cols.max() >= values.shape[1]):
        raise ValueError(f"values of shape {values.shape} at offset ({row_off}, {col_off}) do not cover the weights' "
                         f"grid {wts.grid.width} x {wts.grid.height}")
    n = wts.n_polygons
    v = values[rows, cols].astype(np.float64)
    valid = np.isfinite(v)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        valid &= v != float(nodata)
    w = wts.w
    wv = np.where(valid, w, 0.0)
    vv = np.where(valid, v, 0.0)
    tot = np.bincount(wts.idx, weights=w, minlength=n)
    cv = np.bincount(wts.idx, weights=wv, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = {"cells_total": tot, "cells_valid": cv, "area_valid_m2": cv * wts.grid.res ** 2,
               "valid_frac": np.where(tot > 0, cv / tot, np.nan)}
        parsed = [parse_stat(s) for s in stats]
        qs = [q for k, q, _ in parsed if k == "p"]
        pct = weighted_percentile_by_group(wts.idx[valid], v[valid], w[valid], qs, n) if qs else None
        qi = 0
        for kind, arg, col in parsed:
            if kind == "mean":
                res[col] = np.where(cv > 0, np.bincount(wts.idx, weights=wv * vv, minlength=n) / cv, np.nan)
            elif kind in ("min", "max"):
                acc = np.full(n, np.inf if kind == "min" else -np.inf)
                (np.minimum if kind == "min" else np.maximum).at(acc, wts.idx[valid], v[valid])
                res[col] = np.where(np.isfinite(acc), acc, np.nan)
            elif kind == "p":
                res[col] = pct[qi]
                qi += 1
            else:
                above = (v > arg) if kind == "frac_gt" else (v >= arg)
                res[col] = np.where(cv > 0, np.bincount(wts.idx, weights=wv * above, minlength=n) / cv, np.nan)
    return pd.DataFrame(res)


def _check_raster_crs(src, path):
    if src.crs is None:
        raise ValueError(f"{path} has no CRS")
    if src.crs.to_epsg() is None:
        raise ValueError(f"{path}: CRS has no EPSG code; warp it onto a project grid first (itareo.grid.warp_to_grid)")
    if src.crs.is_geographic:
        raise ValueError(f"{path} is in geographic coordinates ({src.crs}); areas need a projected grid, warp it first")


def read_window(src, band, wts: Weights):
    """The band on the window spanned by the weights' pairs. Returns (values, row_off, col_off)."""
    if wts.rows.size == 0:
        return np.zeros((0, 0), np.float32), 0, 0
    r0, r1 = int(wts.rows.min()), int(wts.rows.max()) + 1
    c0, c1 = int(wts.cols.min()), int(wts.cols.max()) + 1
    return src.read(band, window=Window(c0, r0, c1 - c0, r1 - r0)), r0, c0


def _versions():
    import geopandas
    return {"rasterio": rasterio.__version__, "geopandas": geopandas.__version__, "shapely": shapely.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def zonal_stats(raster, polygons, id_field, stats=DEFAULT_STATS, band=1, nodata=None, fine_res=None, out_csv=None,
                exact_below=EXACT_BELOW):
    """Statistics of `raster` per polygon id. `band`: an index, a list of indices, or None for all bands (columns
    prefixed with the band description or b<index> when more than one). nodata=None takes the file tag; a file without
    a tag needs an explicit value. Returns the DataFrame (id first); with out_csv also writes the CSV and a JSON of the
    run next to it."""
    for s in stats:
        parse_stat(s)
    with rasterio.open(raster) as src:
        _check_raster_crs(src, raster)
        grid = grid_of(raster)
        tag = src.nodata
        if nodata is None:
            if tag is None:
                raise ValueError(f"{raster} has no nodata tag; pass nodata explicitly")
            nodata = tag
        bands = list(range(1, src.count + 1)) if band is None else ([band] if isinstance(band, int) else list(band))
        gdf, repaired = load_polygons(polygons, id_field, grid.crs)
        wts = cell_weights(grid, gdf, fine_res, exact_below=exact_below)
        frames = []
        for b in bands:
            values, r0, c0 = read_window(src, b, wts)
            d = stats_from_weights(values, nodata, wts, stats, r0, c0)
            if len(bands) > 1:
                d = d.add_prefix(f"{src.descriptions[b - 1] or f'b{b}'}__")
            frames.append(d)
    gl, gb, gr, gt = grid.bounds
    extent = box(gl, gb, gr, gt)
    area = gdf.geometry.area.values
    inside = shapely.area(shapely.intersection(gdf.geometry.values, extent))
    head = pd.DataFrame({id_field: gdf[id_field].values, "polygon_area_m2": area,
                         "inside_raster_frac": np.where(area > 0, inside / np.where(area > 0, area, 1), np.nan),
                         "exact_weights": np.isin(np.arange(len(gdf)), wts.exact_idx)})
    df = pd.concat([head] + frames, axis=1)
    first = frames[0].columns[0] if len(bands) == 1 else frames[0].columns[0]
    df.attrs.update({"raster": str(raster), "bands": bands, "nodata": nodata, "stats": list(stats),
                     "factor": wts.factor, "fine_res_m": wts.fine_res, "exact_below_fine_pixels": exact_below,
                     "rasterize_passes": wts.n_passes, "polygons": len(gdf), "polygons_repaired": repaired,
                     "polygons_exact_weights": int(wts.exact_idx.size),
                     "ids_without_weight": int((df[first] == 0).sum()), "pairs": int(wts.w.size)})
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        stat_cols = [c for c in df.columns if c not in head.columns and not c.endswith(("cells_total", "cells_valid", "area_valid_m2"))]
        df_out = df.copy()
        df_out[stat_cols] = df_out[stat_cols].round(6)
        df_out.to_csv(out_csv, index=False)
        out_csv.with_suffix(".json").write_text(json.dumps({**df.attrs, "polygons_file": str(polygons) if not hasattr(polygons, "geometry") else "GeoDataFrame",
                                                            "id_field": id_field, "csv": str(out_csv), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                                            "versions": _versions()}, indent=1, default=str), encoding="utf-8")
    return df


# ----------------------------------------------------------------------------------------------- 10 m -> 30 m
def overlap_matrix(src_edges, dst_edges):
    """Sparse [n_dst, n_src] matrix of overlap lengths / dst length between two increasing edge arrays."""
    rows, cols, vals = [], [], []
    for I in range(len(dst_edges) - 1):
        lo, hi = dst_edges[I], dst_edges[I + 1]
        i0 = max(0, np.searchsorted(src_edges, lo, "right") - 1)
        i1 = min(len(src_edges) - 1, np.searchsorted(src_edges, hi, "left"))
        for i in range(i0, i1):
            ov = min(hi, src_edges[i + 1]) - max(lo, src_edges[i])
            if ov > 1e-9:
                rows.append(I)
                cols.append(i)
                vals.append(ov / (hi - lo))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(dst_edges) - 1, len(src_edges) - 1))


def overlap_matrices(src: Grid, dst: Grid):
    """(Wy, Wx): dst-cell x src-pixel overlap fractions along y (top-down) and x; Wy @ A @ Wx.T is the area-weighted
    sum over an array A on `src`, in units of the dst cell area."""
    if src.crs != dst.crs:
        raise ValueError(f"CRS differ: {src.crs} vs {dst.crs}")
    sx = src.left + np.arange(src.width + 1) * src.res
    dx = dst.left + np.arange(dst.width + 1) * dst.res
    sy = -(src.top - np.arange(src.height + 1) * src.res)      # negated so that edges increase
    dy = -(dst.top - np.arange(dst.height + 1) * dst.res)
    return overlap_matrix(sy, dy), overlap_matrix(sx, dx)


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def area_weighted_resample(src_path, dst_path, dst_grid: Grid = COARSE_30M, bands=None, min_valid=0.5, nodata=None):
    """Area-weighted mean of a raster onto `dst_grid` (default the 30 m fusion grid), valid pixels only. Writes
    dst_path (float32, nodata kept from the source tag unless given), <dst>_valid_frac.tif (valid area fraction of
    each cell, minimum over the bands; area outside the source counts as invalid) and <dst>.json. Cells whose valid
    fraction is below min_valid, or without any valid pixel, are nodata. Returns the summary dict."""
    dst_path = Path(dst_path)
    vf_path = dst_path.with_name(dst_path.stem + "_valid_frac.tif")
    with rasterio.open(src_path) as src:
        _check_raster_crs(src, src_path)
        src_grid = grid_of(src_path)
        Wy, Wx = overlap_matrices(src_grid, dst_grid)
        tag = src.nodata
        if nodata is None:
            if tag is None:
                raise ValueError(f"{src_path} has no nodata tag; pass nodata explicitly")
            nodata = tag
        bands = list(bands or range(1, src.count + 1))
        out = np.full((len(bands), dst_grid.height, dst_grid.width), nodata, np.float32)
        vf = None
        for k, b in enumerate(bands):
            a = src.read(b).astype(np.float64)
            valid = np.isfinite(a) if _is_nan(nodata) else (np.isfinite(a) & (a != float(nodata)))
            num = (Wx @ (Wy @ np.where(valid, a, 0.0)).T).T
            den = (Wx @ (Wy @ valid.astype(np.float64)).T).T
            vf = den.astype(np.float32) if vf is None else np.minimum(vf, den.astype(np.float32))
            ok = (den > 0) & (den >= min_valid - 1e-12)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k][ok] = (num[ok] / den[ok]).astype(np.float32)
        descr = [src.descriptions[b - 1] for b in bands]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **dst_grid.profile(count=len(bands), dtype="float32", nodata=nodata)) as d:
        d.write(out)
        for i, n in enumerate(descr):
            if n:
                d.set_band_description(i + 1, n)
    with rasterio.open(vf_path, "w", **dst_grid.profile(count=1, dtype="float32")) as d:
        d.write(vf[None])
    nd_count = [int((np.isnan(out[k]) if _is_nan(nodata) else (out[k] == nodata)).sum()) for k in range(len(bands))]
    summary = {"src": str(src_path), "dst": str(dst_path), "valid_frac": str(vf_path), "dst_grid": dst_grid.__dict__,
               "src_grid": src_grid.__dict__, "bands": bands, "band_names": descr, "min_valid": min_valid,
               "nodata": nodata, "cells_nodata_per_band": nd_count, "cells_total": int(dst_grid.width * dst_grid.height),
               "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "versions": _versions()}
    dst_path.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def brute_force_cell(values, valid, src: Grid, dst: Grid, I, J):
    """Area-weighted mean of one dst cell by explicit loops over the overlapping src pixels (independent check)."""
    lo_x, hi_x = dst.left + J * dst.res, dst.left + (J + 1) * dst.res
    hi_y, lo_y = dst.top - I * dst.res, dst.top - (I + 1) * dst.res
    num = den = 0.0
    for i in range(src.height):
        y1, y0 = src.top - i * src.res, src.top - (i + 1) * src.res
        oy = min(hi_y, y1) - max(lo_y, y0)
        if oy <= 0:
            continue
        for j in range(src.width):
            x0, x1 = src.left + j * src.res, src.left + (j + 1) * src.res
            ox = min(hi_x, x1) - max(lo_x, x0)
            if ox <= 0 or not valid[i, j]:
                continue
            num += ox * oy * values[i, j]
            den += ox * oy
    area = dst.res ** 2
    return (num / den if den > 0 else np.nan), den / area


# ----------------------------------------------------------------------------------------------- self-test
def selftest(out=None):
    """Synthetic checks, all against independent brute-force computations:
    1. weights of a 30 m raster (fine lattice 3 m) and of a 3 m raster (factor 1) against fine-pixel centres tested
       with shapely contains_xy, polygon by polygon (overlapping and nested polygons keep their whole area);
       the same with the layer cut into 2-row blocks (multi-block path);
    2. every statistic against numpy on the expanded fine-pixel values, with a nodata block and NaNs;
    3. small polygons (below exact_below fine pixels): weights equal exact shapely intersection areas;
    4. a spiked (invalid) ring and a GeometryCollection: repaired to their polygonal part, no line pixels counted;
    5. the same polygons given in EPSG:4326 give the same weights; a layer in the wrong CRS is refused;
    6. error cases: duplicate ids, raster without nodata tag, a line geometry, a values array of the wrong shape;
    7. area_weighted_resample of a synthetic 10 m raster on the Sentinel-2 lattice onto a 30 m window of COARSE_30M
       (offset lattices) against brute_force_cell on every cell, with nodata holes and the min_valid rule; min_valid 0
       never writes NaN; a NaN nodata tag is counted;
    8. for information, fine-pixel weights against exact shapely intersection areas."""
    import geopandas as gpd
    from shapely import contains_xy
    from shapely.affinity import rotate
    from shapely.geometry import GeometryCollection, LineString
    from itareo.gee.common import GRID_S2_10M
    rng = np.random.default_rng(1)
    res = {}
    out = Path(out or W / "runs_itareo" / "zonal_selftest")
    out.mkdir(parents=True, exist_ok=True)

    # --- 1-2: 30 m raster, 12 x 10 cells, origin COMMON
    g30 = Grid(COMMON.crs, COMMON.left, COMMON.top, 30.0, 12, 10)
    vals = rng.uniform(0, 1, (g30.height, g30.width)).astype(np.float32)
    vals[2:4, 3:6] = -9999.0                                  # nodata block
    vals[7, 1] = np.nan                                       # NaN is invalid too
    L, T = g30.left, g30.top
    polys = [Polygon([(L + 20, T - 20), (L + 160, T - 35), (L + 150, T - 130), (L + 25, T - 110)]),   # several cells
             box(L + 200, T - 80, L + 212, T - 70),                                                  # inside one cell, 16 fine px
             rotate(box(L + 100, T - 250, L + 230, T - 170), 30),                                    # rotated, overlaps 6
             box(L + 300, T - 290, L + 400, T - 200),                                                # partly outside
             box(L + 50.1, T - 150.1, L + 50.2, T - 150.0),                                          # 0.01 m2, no fine pixel
             box(L + 150, T - 230, L + 260, T - 150),                                                # overlaps 3
             box(L + 185, T - 215, L + 215, T - 185),                                                # nested inside 6 and 3
             Polygon([(L + 10, T - 290), (L + 60, T - 290), (L + 110, T - 290)])]                    # collinear: empty after repair
    gdf = gpd.GeoDataFrame({"zid": [11, 12, 13, 14, 15, 16, 17, 18]}, geometry=polys, crs=g30.crs)
    gdf, repaired = load_polygons(gdf, "zid", g30.crs)
    res["repaired_in_layer"] = repaired                        # the collinear ring
    res["empty_after_repair"] = int(gdf.geometry.is_empty.sum())

    def brute_weights(grid, gdf):
        f = fine_factor(grid.res)
        fres = grid.res / f
        xs = grid.left + (np.arange(grid.width * f) + 0.5) * fres
        ys = grid.top - (np.arange(grid.height * f) + 0.5) * fres
        X, Y = np.meshgrid(xs, ys)
        frames, masks = [], []
        for i, geom in enumerate(gdf.geometry.values):                 # polygon by polygon: overlaps count for both
            m = np.zeros(X.shape, bool) if geom.is_empty else contains_xy(geom, X, Y)
            masks.append(m)
            rr, cc = np.nonzero(m)
            frames.append(pd.DataFrame({"r": rr // f, "c": cc // f, "i": i}))
        d = pd.concat(frames).groupby(["r", "c", "i"]).size().reset_index(name="n")
        d["w"] = d["n"] / (f * f)
        return d.sort_values(["r", "c", "i"]).reset_index(drop=True), masks, f

    def as_frame(wts):
        return pd.DataFrame({"r": wts.rows, "c": wts.cols, "i": wts.idx, "w": wts.w}).sort_values(["r", "c", "i"]).reset_index(drop=True)

    def same(a, b):
        return bool(len(a) == len(b) and np.array_equal(a[["r", "c", "i"]].values, b[["r", "c", "i"]].values)
                    and np.allclose(a["w"].values, b["w"].values))

    wts0 = cell_weights(g30, gdf, exact_below=0)                        # pure fine-pixel weights
    bw, masks, f = brute_weights(g30, gdf)
    res["30m_factor"] = wts0.factor
    res["30m_passes"] = wts0.n_passes
    res["30m_pairs"] = int(len(wts0.w))
    res["30m_weights_equal_bruteforce"] = same(as_frame(wts0), bw)
    res["30m_total_weight_equals_fine_pixels"] = bool(np.isclose(wts0.w.sum() * f * f, sum(m.sum() for m in masks)))
    res["30m_nested_polygon_keeps_whole_area"] = bool(np.isclose(wts0.w[wts0.idx == 6].sum() * 900, 900.0, rtol=0.05))
    res["30m_overlapping_polygons_keep_whole_area"] = bool(np.isclose(wts0.w[wts0.idx == 5].sum() * 900, polys[5].area, rtol=0.03))
    wblk = cell_weights(g30, gdf, exact_below=0, max_fine_pixels=g30.width * f * f * 2)   # 2 raster rows per block
    res["30m_weights_same_with_2_row_blocks"] = same(as_frame(wblk), as_frame(wts0))
    # statistics against the expanded fine pixels
    stats = ("mean", "min", "max", "p10", "p50", "p90", "frac_gt:0.5", "frac_ge:0.25")
    cols = [parse_stat(s)[2] for s in stats]
    df = stats_from_weights(vals, -9999.0, wts0, stats)
    fine_vals = np.kron(vals.astype(np.float64), np.ones((f, f)))
    checks = []
    for i in range(len(gdf)):
        sel = masks[i]
        tot = sel.sum() / (f * f)
        v = fine_vals[sel]
        v = v[np.isfinite(v) & (v != -9999.0)]
        row = df.iloc[i]
        checks.append(np.isclose(row["cells_total"], tot))
        checks.append(np.isclose(row["cells_valid"], len(v) / (f * f)))
        if len(v):
            vs = np.sort(v)
            exp = {"mean": v.mean(), "min": v.min(), "max": v.max(), "frac_gt_0.5": (v > 0.5).mean(),
                   "frac_ge_0.25": (v >= 0.25).mean()}
            for q in (10, 50, 90):
                k = int(np.searchsorted(np.arange(1, len(vs) + 1) / len(vs), q / 100 - 1e-12, "left"))
                exp[f"p{q}"] = vs[min(k, len(vs) - 1)]
            checks += [np.isclose(row[c], exp[c]) for c in cols]
        else:
            checks += [bool(np.isnan(row[c])) for c in cols]
    res["30m_stats_equal_bruteforce"] = bool(all(checks))
    res["30m_ids_without_weight_pure_fine"] = int((df["cells_total"] == 0).sum())      # the 0.01 m2 box and the empty ring
    # --- 3: small polygons get exact weights
    wts = cell_weights(g30, gdf)
    res["30m_exact_weighted_polygons"] = sorted(int(i) for i in wts.exact_idx)        # 12 (16 fine px) and 15 (0.01 m2)
    ok_exact = []
    for i in wts.exact_idx:
        sel = wts.idx == i
        for r, c, w in zip(wts.rows[sel], wts.cols[sel], wts.w[sel]):
            cell = box(L + c * 30, T - (r + 1) * 30, L + (c + 1) * 30, T - r * 30)
            ok_exact.append(np.isclose(w, cell.intersection(gdf.geometry.iloc[i]).area / 900.0))
        ok_exact.append(np.isclose(wts.w[sel].sum() * 900.0, gdf.geometry.iloc[i].area))
    res["30m_exact_weights_equal_intersection_areas"] = bool(ok_exact and all(ok_exact))
    res["30m_tiny_polygon_now_has_weight"] = bool(wts.w[wts.idx == 4].sum() > 0)
    df_x = stats_from_weights(vals, -9999.0, wts, stats)
    res["30m_ids_without_weight_default"] = int((df_x["cells_total"] == 0).sum())       # only the empty ring
    # --- 3 m raster, factor 1
    g3 = Grid(COMMON.crs, COMMON.left, COMMON.top, 3.0, 120, 100)
    w3 = cell_weights(g3, gdf, exact_below=0)
    b3, _, f3 = brute_weights(g3, gdf)
    res["3m_factor"] = w3.factor
    res["3m_weights_equal_bruteforce"] = bool(f3 == 1 and same(as_frame(w3), b3) and np.all(w3.w == 1.0))
    w3x = cell_weights(g3, gdf)
    res["3m_exact_weighted_polygons"] = sorted(int(i) for i in w3x.exact_idx)
    res["3m_exact_tiny_polygon_weight_equals_area"] = bool(np.isclose(w3x.w[w3x.idx == 4].sum() * 9.0, polys[4].area))
    # --- 4: spiked ring and GeometryCollection
    spike = Polygon([(L + 300, T - 20), (L + 400, T - 20), (L + 400, T - 120), (L + 400, T - 420), (L + 400, T - 120), (L + 300, T - 120)])
    gc = GeometryCollection([box(L + 500, T - 100, L + 560, T - 130), LineString([(L + 500, T - 100), (L + 700, T - 100)])])
    g_sp, rep_sp = load_polygons(gpd.GeoDataFrame({"zid": [1, 2]}, geometry=[spike, gc], crs=g30.crs), "zid", g30.crs)
    res["spike_repaired_and_gc_reduced"] = rep_sp == 2 and all(t in ("Polygon", "MultiPolygon") for t in g_sp.geometry.geom_type)
    w_sp = cell_weights(g30, g_sp, exact_below=0)
    b_sp, _, _ = brute_weights(g30, g_sp)                       # contains_xy on the polygonal parts: no line pixels
    res["spike_weights_equal_bruteforce_on_polygonal_part"] = same(as_frame(w_sp), b_sp)
    res["spike_area_m2_after_repair"] = [float(g.area) for g in g_sp.geometry]      # 10000 and 1800
    # --- 5: CRS round trip and wrong CRS
    g4326, _ = load_polygons(gdf.to_crs(4326), "zid", g30.crs)
    res["weights_same_after_crs_roundtrip"] = same(as_frame(cell_weights(g30, g4326)), as_frame(wts))
    try:
        cell_weights(g30, gdf.to_crs(4326))
        res["error_wrong_crs_layer"] = False
    except ValueError:
        res["error_wrong_crs_layer"] = True
    # --- 6: errors
    tif = out / "vals30.tif"
    with rasterio.open(tif, "w", **g30.profile(dtype="float32", nodata=-9999.0)) as d:
        d.write(vals[None])
    tif_notag = out / "vals30_notag.tif"
    with rasterio.open(tif_notag, "w", **g30.profile(dtype="float32")) as d:
        d.write(vals[None])
    dup = gdf.copy()
    dup.loc[1, "zid"] = 11
    line_layer = gpd.GeoDataFrame({"zid": [1]}, geometry=[LineString([(L, T), (L + 100, T - 100)])], crs=g30.crs)
    for key, fn in (("error_duplicate_ids", lambda: zonal_stats(tif, dup, "zid")),
                    ("error_no_nodata_tag", lambda: zonal_stats(tif_notag, gdf, "zid")),
                    ("error_line_geometry", lambda: zonal_stats(tif, line_layer, "zid")),
                    ("error_values_wrong_shape", lambda: stats_from_weights(vals[:5], -9999.0, wts)),
                    ("error_bad_stat", lambda: parse_stat("p1x"))):
        try:
            fn()
            res[key] = False
        except ValueError:
            res[key] = True
    full = zonal_stats(tif, gdf, "zid", stats=stats, out_csv=out / "zonal_selftest.csv")
    res["zonal_stats_matches_stats_from_weights"] = bool(np.allclose(full[list(df_x.columns)].values, df_x.values, equal_nan=True))
    res["zonal_stats_area_valid_m2_ok"] = bool(np.allclose(full["area_valid_m2"], full["cells_valid"] * 900.0))
    res["zonal_stats_inside_raster_frac_partly_outside"] = float(full["inside_raster_frac"].iloc[3])   # box 300..400 x 200..290 vs grid 360 x 300
    res["zonal_stats_json_written"] = (out / "zonal_selftest.json").exists()
    csv_back = pd.read_csv(out / "zonal_selftest.csv")
    res["csv_counts_exact"] = bool(np.allclose(csv_back["cells_total"], full["cells_total"]) and np.allclose(csv_back["cells_valid"], full["cells_valid"]))
    two = zonal_stats(out / "vals30.tif", gdf, "zid", stats=("mean",), band=None)
    res["zonal_stats_all_bands_single_band_file"] = bool(np.allclose(two["mean"], full["mean"], equal_nan=True))
    # --- 7: 10 m -> 30 m area weighting
    s2 = GRID_S2_10M.subgrid((387000.0, 6524000.0, 388000.0, 6525000.0))          # 10 m window, 100 x 100
    d30 = COARSE_30M.subgrid((387100.0, 6524100.0, 387900.0, 6524900.0))           # 30 m window inside it
    a = rng.uniform(100, 5000, (2, s2.height, s2.width)).astype(np.float32)
    a[:, 40:52, 40:55] = 65535                                                     # nodata hole, both bands
    a[1, 10:12, :] = 65535                                                         # extra hole in band 2
    src = out / "s2_synth_10m.tif"
    with rasterio.open(src, "w", **s2.profile(count=2, dtype="float32", nodata=65535.0)) as d:
        d.write(a)
        d.set_band_description(1, "B4")
        d.set_band_description(2, "B8")
    summ = area_weighted_resample(src, out / "s2_synth_30m.tif", d30, min_valid=0.5)
    with rasterio.open(out / "s2_synth_30m.tif") as d, rasterio.open(out / "s2_synth_30m_valid_frac.tif") as v:
        got, vfr, descr = d.read(), v.read(1), d.descriptions
    maxdiff, maxrel, nd_ok, vf_ok, below = 0.0, 0.0, True, True, 0
    for I in range(d30.height):
        for J in range(d30.width):
            dens = []
            for k in range(2):
                valid = a[k] != 65535
                m, den = brute_force_cell(a[k].astype(np.float64), valid, s2, d30, I, J)
                dens.append(den)
                if den < 0.5 - 1e-12:
                    nd_ok &= got[k, I, J] == 65535
                    below += 1
                else:
                    maxdiff = max(maxdiff, abs(got[k, I, J] - m))
                    maxrel = max(maxrel, abs(got[k, I, J] - m) / abs(m))
            vf_ok &= np.isclose(vfr[I, J], min(dens))
    res["s2_30m_max_abs_diff_vs_bruteforce"] = float(maxdiff)          # float32 output vs float64 brute force
    res["s2_30m_max_rel_diff_vs_bruteforce"] = float(maxrel)
    res["s2_30m_nodata_below_min_valid_ok"] = bool(nd_ok)
    res["s2_30m_cells_below_min_valid_checked"] = below
    res["s2_30m_valid_frac_equals_bruteforce_min_over_bands"] = bool(vf_ok)
    res["s2_30m_band_names_kept"] = list(descr) == ["B4", "B8"]
    res["s2_30m_json_written"] = (out / "s2_synth_30m.json").exists()
    Wy, Wx = overlap_matrices(s2, d30)
    res["s2_30m_weights_sum_to_1_inside_source"] = bool(np.allclose((Wx @ (Wy @ np.ones((s2.height, s2.width))).T).T, 1.0))
    res["s2_30m_lattices_offset_m"] = [d30.left - s2.left, s2.top - d30.top]
    res["s2_30m_summary"] = summ
    # min_valid = 0: a cell with no valid pixel stays nodata, never NaN; the hole 40:52 x 40:55 covers whole 30 m cells
    s0 = area_weighted_resample(src, out / "s2_synth_30m_minvalid0.tif", d30, min_valid=0.0)
    with rasterio.open(out / "s2_synth_30m_minvalid0.tif") as d:
        g0 = d.read()
    res["s2_30m_min_valid_0_no_nan"] = bool(not np.isnan(g0).any() and s0["cells_nodata_per_band"][0] > 0)
    # NaN nodata tag: counted
    srcn = out / "s2_synth_10m_nan.tif"
    an = np.where(a == 65535, np.nan, a).astype(np.float32)
    with rasterio.open(srcn, "w", **s2.profile(count=2, dtype="float32", nodata=float("nan"))) as d:
        d.write(an)
    sn = area_weighted_resample(srcn, out / "s2_synth_30m_nan.tif", d30, min_valid=0.5)
    res["s2_30m_nan_nodata_counted"] = bool(sn["cells_nodata_per_band"] == summ["cells_nodata_per_band"])
    # --- 8: fine-pixel weights vs exact intersection areas (information only)
    rel = []
    ours = as_frame(wts0)
    for _, r in ours.iterrows():
        cell = box(g30.left + r.c * 30, g30.top - (r.r + 1) * 30, g30.left + (r.c + 1) * 30, g30.top - r.r * 30)
        exact = cell.intersection(gdf.geometry.iloc[int(r.i)]).area / 900.0
        if exact > 0:
            rel.append(abs(r.w - exact))
    res["30m_weight_abs_error_vs_exact_area_max"] = float(max(rel)) if rel else None
    res["30m_weight_abs_error_vs_exact_area_mean"] = float(np.mean(rel)) if rel else None
    res["30m_pairs_compared_to_exact_area"] = len(rel)
    d = EVIDENCE / "zonal"
    d.mkdir(parents=True, exist_ok=True)
    (d / "selftest_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))
    musts = [k for k, v in res.items() if isinstance(v, bool) and not v]
    assert not musts, f"failed: {musts}"
    assert res["s2_30m_max_rel_diff_vs_bruteforce"] < 1e-6
    assert res["30m_ids_without_weight_default"] == 1 and res["empty_after_repair"] == 1
    assert res["30m_exact_weighted_polygons"] == [1, 4] and res["3m_exact_weighted_polygons"] == [1, 4]   # 120 m2 = 13 px at 3 m
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster")
    ap.add_argument("--band", type=int, nargs="+", default=[1])
    ap.add_argument("--all-bands", action="store_true")
    ap.add_argument("--nodata", type=float)
    ap.add_argument("--polygons")
    ap.add_argument("--id")
    ap.add_argument("--stats", nargs="+", default=list(DEFAULT_STATS))
    ap.add_argument("--fine-res", type=float)
    ap.add_argument("--exact-below", type=int, default=EXACT_BELOW, help="fine pixels below which a polygon gets exact weights")
    ap.add_argument("--s2-to-30m", metavar="SRC")
    ap.add_argument("--min-valid", type=float, default=0.5)
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.s2_to_30m:
        if not a.out:
            raise SystemExit("--out <dst.tif> is required")
        print(json.dumps(area_weighted_resample(a.s2_to_30m, a.out, min_valid=a.min_valid), indent=1, default=str))
        return
    if not (a.raster and a.polygons and a.id and a.out):
        raise SystemExit("--raster, --polygons, --id and --out are required (or --selftest / --s2-to-30m)")
    try:
        for s in a.stats:
            parse_stat(s)
    except ValueError as e:
        raise SystemExit(str(e))
    band = None if a.all_bands else (a.band[0] if len(a.band) == 1 else a.band)
    df = zonal_stats(a.raster, a.polygons, a.id, a.stats, band, a.nodata, a.fine_res, a.out, a.exact_below)
    print(json.dumps(df.attrs, indent=1, default=str))
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
