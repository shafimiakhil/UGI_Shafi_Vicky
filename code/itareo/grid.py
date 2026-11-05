"""The common grid of the project and warping onto it, reusing Luigi's alignment logic.

COMMON is the 3 m PlanetScope grid of Luigi's aligned rasters: EPSG:32720, extent 375327, 6511452, 399624, 6535833,
8099 x 8127 px (work/runs/alignment_check/run_log.md). Coarser grids keep the same top-left corner:
  - COMMON.coarser(k), k integer: pixel k x 3 m, size floor(8099 / k) x floor(8127 / k), the floor rule of Luigi's
    build_grid (preprocessing/align_sar_hs_ms_common_grid.py l.58-69). Nested in the 3 m grid: every coarse pixel is
    exactly k x k fine pixels. k = 10 gives 30 m, 809 x 812 px, identical to Luigi's aligned PRISMA grid.
  - COMMON.same_origin(res) for any resolution, e.g. 10 m (2429 x 2438 px). 10 m is NOT an integer multiple of 3 m,
    so a 10 m grid is not nested in the 3 m grid (pixel edges coincide only every 30 m). Use it for products whose
    native grid is 10 m (Sentinel-2), and warp to COMMON or COMMON.coarser(10) before stacking with other sensors.
  - COMMON.finer(6) = GRID_05M, the 0.5 m grid for Pleiades (D1): 48594 x 48762 px, same extent; every 3 m pixel is
    exactly 6 x 6 pixels of it. The native Sentinel-2 10 m lattice (itareo.gee.common.GRID_S2_10M, origin 375320,
    6535840) is also exact on it: each 10 m cell is 20 x 20 pixels of 0.5 m, starting at 0.5 m pixel (-14, -14)
    (offset_in). Rasters for D1 live on windows of GRID_05M (subgrid, snapped to 6 px so that 3 m cells nest).
  - COMMON.finer(3) = GRID_1M, the 1 m grid of D1 since 2 Nov 2026 (CLAUDE.md Section 11.3 a, the GEOBIA label
    resolution): 24297 x 24381 px, same extent. It nests in the 3 m grid (factor 3), in the Sentinel-2 10 m lattice
    (factor 10, origin offset 7 m = 7 px), in the 25 m GEDI grid (COMMON.same_origin(25), factor 25), in the 30 m
    fusion grid (factor 30) and in a 1 km grid from the COMMON origin (factor 1000). The 0.5 m grid is used only if a
    step needs it.
warp_to_grid() calls Luigi's reproject_to_grid (same file, l.24-55) unchanged: per band rasterio.warp.reproject,
output metadata copied from the source, nodata set on the output.

From work_ms/:  python -m itareo.grid --selftest   (checks the constants against the aligned rasters; GRID_1M nesting)
                python -m itareo.grid --check <raster> [...] [--out <csv>]   (incoming data: lattice and extent report)
"""
import argparse
import math
from dataclasses import dataclass

import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling

from preprocessing.align_sar_hs_ms_common_grid import reproject_to_grid  # Luigi's, unchanged

from itareo import PROJECT


@dataclass(frozen=True)
class Grid:
    crs: str
    left: float
    top: float
    res: float
    width: int
    height: int

    @property
    def transform(self):
        return from_origin(self.left, self.top, self.res, self.res)

    @property
    def bounds(self):
        return (self.left, self.top - self.height * self.res, self.left + self.width * self.res, self.top)

    def coarser(self, factor: int) -> "Grid":
        """Nested coarser grid: pixel factor x res, same top-left corner, floor size (Luigi's build_grid rule)."""
        if int(factor) != factor or factor < 1:
            raise ValueError("factor must be a positive integer; use same_origin() for other resolutions")
        factor = int(factor)
        return Grid(self.crs, self.left, self.top, self.res * factor, self.width // factor, self.height // factor)

    def same_origin(self, res: float) -> "Grid":
        """Grid with any resolution and the same top-left corner, covering at most the same extent (floor).
        Not nested in self unless res / self.res is an integer."""
        ext_w, ext_h = self.width * self.res, self.height * self.res
        return Grid(self.crs, self.left, self.top, float(res), int(math.floor(ext_w / res + 1e-9)),
                    int(math.floor(ext_h / res + 1e-9)))

    def finer(self, factor: int) -> "Grid":
        """Nested finer grid: pixel res / factor, same top-left corner and extent (COMMON.finer(6) = 0.5 m)."""
        if int(factor) != factor or factor < 1:
            raise ValueError("factor must be a positive integer")
        factor = int(factor)
        return Grid(self.crs, self.left, self.top, self.res / factor, self.width * factor, self.height * factor)

    def subgrid(self, bounds, snap: int = 1) -> "Grid":
        """Window of this grid covering bounds = (left, bottom, right, top) in map units, snapped outward to multiples
        of `snap` pixels counted from this grid's origin and clipped to its extent. Same lattice as self."""
        l, b, r, t = bounds
        c0 = max(0, int(math.floor((l - self.left) / self.res / snap + 1e-9)) * snap)
        r0 = max(0, int(math.floor((self.top - t) / self.res / snap + 1e-9)) * snap)
        c1 = min(self.width, int(math.ceil((r - self.left) / self.res / snap - 1e-9)) * snap)
        r1 = min(self.height, int(math.ceil((self.top - b) / self.res / snap - 1e-9)) * snap)
        if c1 <= c0 or r1 <= r0:
            raise ValueError(f"bounds {bounds} do not overlap the grid")
        return Grid(self.crs, self.left + c0 * self.res, self.top - r0 * self.res, self.res, c1 - c0, r1 - r0)

    def offset_in(self, fine: "Grid"):
        """(factor, col_off, row_off) when every pixel of self is exactly factor x factor pixels of `fine`, the
        first one starting at fine pixel (col_off, row_off) (may be negative: self may extend beyond fine);
        None when the lattices do not nest (different CRS, non-integer ratio or offset)."""
        if self.crs != fine.crs:
            return None
        vals = (self.res / fine.res, (self.left - fine.left) / fine.res, (fine.top - self.top) / fine.res)
        if any(abs(v - round(v)) > 1e-6 for v in vals) or round(vals[0]) < 1:
            return None
        return tuple(int(round(v)) for v in vals)

    def is_nested_in(self, fine: "Grid") -> bool:
        r = self.res / fine.res
        return (self.crs == fine.crs and abs(r - round(r)) < 1e-9 and self.left == fine.left and self.top == fine.top)

    def profile(self, count=1, dtype="float32", nodata=None, **kw):
        p = {"driver": "GTiff", "crs": self.crs, "transform": self.transform, "width": self.width,
             "height": self.height, "count": count, "dtype": dtype, "compress": "deflate"}
        if nodata is not None:
            p["nodata"] = nodata
        p.update(kw)
        return p


COMMON = Grid(crs="EPSG:32720", left=375327.0, top=6535833.0, res=3.0, width=8099, height=8127)
COARSE_30M = COMMON.coarser(10)       # 809 x 812, = Luigi's PRISMA aligned grid
SAME_ORIGIN_10M = COMMON.same_origin(10.0)   # 2429 x 2438, not nested in COMMON
GRID_05M = COMMON.finer(6)            # 48594 x 48762, 0.5 m (used only if a step needs it)
GRID_1M = COMMON.finer(3)             # 24297 x 24381, 1 m, D1 since 2 Nov 2026 (CLAUDE.md Section 11.3 a)


def grid_of(path) -> Grid:
    with rasterio.open(path) as src:
        t = src.transform
        if abs(t.a + t.e) > 1e-9 or t.b != 0 or t.d != 0:
            raise ValueError(f"{path}: non-square or rotated pixels {t}")
        return Grid(f"EPSG:{src.crs.to_epsg()}", t.c, t.f, t.a, src.width, src.height)


def warp_to_grid(src_path, dst_path, grid: Grid = COMMON, resampling=Resampling.bilinear, nodata=None):
    """Warp any raster onto `grid` with Luigi's reproject_to_grid, which takes the source nodata from the file tag
    (rasterio.band) and writes `nodata` on the output. nodata=None keeps the source tag (e.g. the Earth Engine
    exports' -9999 / -32768 / 255), so masked pixels never mix into valid values and valid zeros stay valid; a source
    without a tag needs an explicit nodata. Resampling: Luigi used nearest for SAR and bilinear for PlanetScope and
    PRISMA (align_sar_hs_ms_common_grid.py l.133-164); use average to aggregate onto a coarser nested grid, and
    nearest for class maps, sparse data (GEDI) and count bands."""
    with rasterio.open(src_path) as s:
        src_nd = s.nodata
    if nodata is None:
        if src_nd is None:
            raise ValueError(f"{src_path} has no nodata tag; pass nodata explicitly")
        nodata = src_nd
    reproject_to_grid(str(src_path), str(dst_path), rasterio.crs.CRS.from_string(grid.crs), grid.transform,
                      grid.width, grid.height, resampling=resampling, nodata=nodata)
    return dst_path


def check(paths, out_csv=None):
    """Grid report for incoming rasters (docs/FRIDAY_PLAN.md, grid checks): CRS, pixel size, size, bands, dtype, nodata;
    for each project grid R (GRID_05M, GRID_1M, COMMON, GRID_S2_10M, COARSE_30M) the nesting of the raster in R
    (raster.offset_in(R): factor, column and row offset in R pixels) and of R in the raster (R.offset_in(raster)),
    None when the lattices do not nest; whether a 0.5 m raster's offset is a multiple of 6 (3 m cells nest without
    a shift); and the fraction of the common extent covered."""
    import csv
    from itareo.gee.common import GRID_S2_10M
    refs = {"GRID_05M": GRID_05M, "GRID_1M": GRID_1M, "COMMON": COMMON, "GRID_S2_10M": GRID_S2_10M,
            "COARSE_30M": COARSE_30M}
    rows = []
    for p in paths:
        with rasterio.open(p) as s:
            r = {"file": str(p), "crs": str(s.crs), "res": s.res, "width": s.width, "height": s.height,
                 "count": s.count, "dtype": s.dtypes[0], "nodata": s.nodata, "bounds": tuple(s.bounds)}
        try:
            g = grid_of(p)
        except (ValueError, TypeError, AttributeError) as e:
            r["note"] = f"no usable grid: {e}"
            rows.append(r)
            continue
        for name, ref in refs.items():
            r[f"in_{name}"] = g.offset_in(ref)
            r[f"{name}_in_raster"] = ref.offset_in(g)
        o = r["in_GRID_05M"]
        r["on_05m_lattice_snapped_to_3m"] = bool(o and o[0] == 1 and o[1] % 6 == 0 and o[2] % 6 == 0)
        l, b, rr, t = g.bounds
        L, B, R, T = COMMON.bounds
        inter = max(0.0, min(rr, R) - max(l, L)) * max(0.0, min(t, T) - max(b, B))
        r["common_extent_covered"] = round(inter / ((R - L) * (T - B)), 6) if g.crs == COMMON.crs else None
        rows.append(r)
        print(r)
    if out_csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def selftest():
    gm = selftest_1m()          # first: its JSON is written (with passed true or false) whatever fails later
    aligned = {
        "PLANET": PROJECT / "work/raw_data/PLANET/20250719_8b_clip_aligned.tif",
        "CSK": PROJECT / "work/raw_data/CSK/CSKS2_GTC_B_HI_04_HH_RA_SF_20250728103626_20250728103634.S01.SBI_aligned.tif",
        "PRISMA": PROJECT / "work/raw_data/PRISMA/prisma_sr_vnir_swir_28_03_2021_40_30_20_georref_230_bands_aligned.tif",
    }
    out = {"COMMON.bounds": COMMON.bounds, "COMMON == PLANET aligned": grid_of(aligned["PLANET"]) == COMMON,
           "COMMON == CSK aligned": grid_of(aligned["CSK"]) == COMMON,
           "coarser(10) == PRISMA aligned": grid_of(aligned["PRISMA"]) == COARSE_30M,
           "COARSE_30M": COARSE_30M, "SAME_ORIGIN_10M": SAME_ORIGIN_10M,
           "10 m nested in 3 m": SAME_ORIGIN_10M.is_nested_in(COMMON), "30 m nested in 3 m": COARSE_30M.is_nested_in(COMMON)}
    from itareo.gee.common import GRID_S2_10M
    out.update({"GRID_05M": GRID_05M, "3 m in 0.5 m": COMMON.offset_in(GRID_05M),
                "30 m in 0.5 m": COARSE_30M.offset_in(GRID_05M), "S2 10 m in 0.5 m": GRID_S2_10M.offset_in(GRID_05M),
                "S2 10 m in 3 m": GRID_S2_10M.offset_in(COMMON)})
    for k, v in out.items():
        print(f"{k}: {v}")
    assert COMMON.bounds == (375327.0, 6511452.0, 399624.0, 6535833.0)
    assert out["COMMON == PLANET aligned"] and out["COMMON == CSK aligned"] and out["coarser(10) == PRISMA aligned"]
    assert GRID_05M.bounds == COMMON.bounds and (GRID_05M.width, GRID_05M.height) == (48594, 48762)
    assert out["3 m in 0.5 m"] == (6, 0, 0) and out["30 m in 0.5 m"] == (60, 0, 0)
    assert out["S2 10 m in 0.5 m"] == (20, -14, -14) and out["S2 10 m in 3 m"] is None
    sg = GRID_05M.subgrid((386281.0, 6523027.0, 390281.0, 6527027.0), snap=6)
    assert sg.offset_in(GRID_05M)[0] == 1 and COMMON.offset_in(sg) is not None and sg.bounds[0] <= 386281.0
    out.update(gm)
    return out


def selftest_1m():
    """GRID_1M (CLAUDE.md Section 11.3 a): its nesting in COMMON, the Sentinel-2 10 m lattice, the 25 m GEDI grid,
    the 30 m fusion grid and a 1 km grid, checked with offset_in and, independently, by pixel-edge arithmetic (every
    edge of the coarse grid falls on a 1 m edge and lies inside a 1 m pixel count that is an integer). Writes
    docs/evidence_phase2/grid_1m/selftest_summary.json."""
    import json
    from itareo import EVIDENCE
    from itareo.gee.common import GRID_25M, GRID_S2_10M
    km = Grid(COMMON.crs, COMMON.left, COMMON.top, 1000.0, 25, 25)
    coarse = {"COMMON 3 m": COMMON, "GRID_S2_10M": GRID_S2_10M, "GRID_25M (GEDI)": GRID_25M,
              "COARSE_30M": COARSE_30M, "1 km from the COMMON origin": km}
    expected = {"COMMON 3 m": (3, 0, 0), "GRID_S2_10M": (10, -7, -7), "GRID_25M (GEDI)": (25, 0, 0),
                "COARSE_30M": (30, 0, 0), "1 km from the COMMON origin": (1000, 0, 0)}
    res = {"GRID_1M": GRID_1M.__dict__, "GRID_1M.bounds": GRID_1M.bounds,
           "GRID_1M == COMMON.finer(3)": GRID_1M == COMMON.finer(3),
           "GRID_1M bounds == COMMON bounds": GRID_1M.bounds == COMMON.bounds}
    for name, g in coarse.items():
        o = g.offset_in(GRID_1M)
        # independent check: coarse edges in 1 m units are integers, and the coarse pixel is an integer number of 1 m px
        edges_x = [(g.left + i * g.res - GRID_1M.left) / GRID_1M.res for i in range(g.width + 1)]
        edges_y = [(GRID_1M.top - (g.top - j * g.res)) / GRID_1M.res for j in range(g.height + 1)]
        integer_edges = all(abs(e - round(e)) < 1e-9 for e in edges_x + edges_y)
        res[name] = {"offset_in_GRID_1M": o, "expected": expected[name], "edges_on_1m_lattice": integer_edges,
                     "coarse_extent_inside_GRID_1M": (g.bounds[0] >= GRID_1M.bounds[0] and g.bounds[1] >= GRID_1M.bounds[1]
                                                      and g.bounds[2] <= GRID_1M.bounds[2] and g.bounds[3] <= GRID_1M.bounds[3])}
    res["0.5 m in 1 m (GRID_1M in GRID_05M)"] = GRID_1M.offset_in(GRID_05M)
    res["Pleiades 2017 PAN lattice note"] = "GRID_1M is EPSG:32720; the Pleiades 2017 products are EPSG:32721, so they are warped once (CLAUDE.md 11.3 b)"
    # every check is evaluated and recorded first; the JSON is written on every run with the result, then a failure
    # raises (a failed run therefore leaves "passed": false, never a stale passing file)
    import datetime as dt
    checks = {"GRID_1M == COMMON.finer(3)": bool(res["GRID_1M == COMMON.finer(3)"]),
              "GRID_1M bounds == COMMON bounds": bool(res["GRID_1M bounds == COMMON bounds"]),
              "size 24297 x 24381 at 1 m": (GRID_1M.width, GRID_1M.height, GRID_1M.res) == (24297, 24381, 1.0),
              "GRID_1M in GRID_05M == (2, 0, 0)": res["0.5 m in 1 m (GRID_1M in GRID_05M)"] == (2, 0, 0)}
    for name in coarse:
        checks[f"{name}: offset as expected"] = tuple(res[name]["offset_in_GRID_1M"] or ()) == expected[name]
        checks[f"{name}: edges on the 1 m lattice"] = bool(res[name]["edges_on_1m_lattice"])
    res["checks"] = checks
    res["passed"] = all(checks.values())
    res["checked_at"] = dt.datetime.now().isoformat(timespec="seconds")
    d = EVIDENCE / "grid_1m"
    d.mkdir(parents=True, exist_ok=True)
    (d / "selftest_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for k, v in res.items():
        print(f"{k}: {v}")
    assert res["passed"], [k for k, v in checks.items() if not v]
    return {"GRID_1M": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.check:
        check(a.check, a.out)
