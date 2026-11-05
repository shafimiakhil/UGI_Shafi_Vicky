"""Shared settings for the Earth Engine exports (Phase 2). Shafi authenticates and runs the exports
(itareo/gee/RUN_SHEET.md). Reduced plan of 27 Oct 2026 (CLAUDE.md Section 10, point f): only what Dr. Marinelli lacks.

- GEE project id: PROJECT below (set to Shafi's project on 27 Oct 2026).
- Every export covers the common extent (EPSG:32720, 375327, 6511452, 399624, 6535833; itareo.grid.COMMON):
    - 3 m  = COMMON itself (8099 x 8127; Meta canopy height mean and max);
    - 10 m = GRID_S2_10M, the native Sentinel-2 / UTM 10 m lattice snapped outward around COMMON (origin 375320,
      6535840; 2431 x 2439), so Earth Engine copies native S2 pixels without resampling (S1 resampled bilinearly);
    - 25 m = COMMON.same_origin(25) (971 x 975; GEDI, sparse: warp it with nearest);
    - 30 m = COMMON.coarser(10) (Landsat module, kept but not in the plan).
  The 10 m and 25 m grids are not nested in the 3 m grid: warp them locally with itareo.grid.warp_to_grid.
- Nodata: every export is written with formatOptions noData = NODATA[dtype] (int16 -32768, uint16 65535, uint8 255,
  float32 -9999). itareo.grid.warp_to_grid reads that tag, so masked pixels never mix into valid ones.
  The exports of 28 Oct 2026 nevertheless came out with masked pixels = 0 (restored by fix_nodata.py); since then
  run_exports burns NODATA[dtype] into masked pixels before export (with_nodata).
- Periods: S1 monthly Jan-Dec 2022 and Oct 2024-Sep 2025; S2 quarterly 2022 and Oct 2024-Sep 2025, plus Jan-Mar 2017
  (L1C TOA); GEDI monthly Apr 2019 - Dec 2024 without the instrument's storage gap. Added 28 Oct 2026: S1 descending
  orbit 112 only, Mar-Sep 2025 (s1o112). Added 29 Oct 2026: S2 SR Jan-Mar 2017 and the single scene of 12 Feb 2017
  (s2sr2017); GEDI Jan-Jul 2025 (gedi2025). Added 2 Nov 2026 (stage A8, CLAUDE.md 11.3 g): GBA building polygons and
  their presence and height on GRID_1M (gba); Open Buildings 2.5D Temporal 2017, 2022, 2023 at 3 m (openbuildings);
  Meta canopy height at 1 m on GRID_1M (meta1m); ETH canopy height 2020 and SD on the 10 m lattice (eth); S1 orbit 112
  monthly Jun-Oct 2026 (s1o112_2026) and S2 SR medians on the NISAR seasons of CLAUDE.md 11.4, winter 17 Jun - 31 Aug,
  September and October 2026 (s2_2026), to pair with NISAR. Added 4 Nov 2026: S1 orbit 112 monthly Jan-Dec 2021,
  Sentinel-1B (s1o112_2021).
- 1 m = GRID_1M = itareo.grid.GRID_1M (COMMON.finer(3), 24297 x 24381).
- Exports go to Google Drive, folder DRIVE_FOLDER, as GeoTIFF (fileNamePrefix = task name); table exports as GeoJSON.
"""
import calendar

from itareo.grid import COMMON, GRID_1M, Grid

PROJECT = "ee-shafiullahmiakhil01"
DRIVE_FOLDER = "itareo_cordoba_gee"
MONTHS_2022 = [(2022, m) for m in range(1, 13)]
MONTHS_2425 = [(2024, m) for m in (10, 11, 12)] + [(2025, m) for m in range(1, 10)]
QUARTERS = {  # name -> (start "YYYY-MM-DD", end exclusive)
    "2022Q1": ("2022-01-01", "2022-04-01"), "2022Q2": ("2022-04-01", "2022-07-01"),
    "2022Q3": ("2022-07-01", "2022-10-01"), "2022Q4": ("2022-10-01", "2023-01-01"),
    "2024Q4": ("2024-10-01", "2025-01-01"), "2025Q1": ("2025-01-01", "2025-04-01"),
    "2025Q2": ("2025-04-01", "2025-07-01"), "2025Q3": ("2025-07-01", "2025-10-01"),
}
GRID_3M = COMMON
GRID_30M = COMMON.coarser(10)
GRID_S2_10M = Grid("EPSG:32720", 375320.0, 6535840.0, 10.0, 2431, 2439)   # native 10 m lattice, contains COMMON
GRID_25M = COMMON.same_origin(25.0)
BYTES = {"uint8": 1, "int16": 2, "uint16": 2, "float32": 4}
NODATA = {"float32": -9999.0, "int16": -32768, "uint16": 65535, "uint8": 255}
CAST = {"float32": "toFloat", "int16": "toInt16", "uint16": "toUint16", "uint8": "toUint8"}
# legacy periods, used only by modules kept but not in the plan (landsat_lst, s2_monthly)
PERIODS = {"P2425": MONTHS_2425, "P2021": [(2021, m) for m in range(1, 13)]}
BUDGET_MB = 7000   # 3000 approved by Shafi on 27 Oct 2026; 3500 on 29 Oct (s2sr2017, gedi2025); 7000 on 2 Nov 2026 for the
                   # stage-A8 products of CLAUDE.md 11.3 g (Shafi: "raise the gate as needed") (uncompressed total)


def with_nodata(img, dtype):
    """Masked pixels set to NODATA[dtype], then cast back to dtype: formatOptions noData alone left them at 0."""
    return getattr(img.unmask(NODATA[dtype], False), CAST[dtype])()


def month_range(year, month):
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year + (month == 12):04d}-{month % 12 + 1:02d}-01", last


def ee_init():
    if PROJECT.startswith("["):
        raise SystemExit("Set PROJECT in itareo/gee/common.py to your GEE project id first")
    try:
        import ee  # only for real exports
    except ImportError:
        raise SystemExit("earthengine-api is not installed in this environment: "
                         "pip install earthengine-api, then earthengine authenticate")
    ee.Initialize(project=PROJECT)
    return ee


def region(ee, grid: Grid):
    l, b, r, t = grid.bounds
    return ee.Geometry.Rectangle([l, b, r, t], proj=grid.crs, geodesic=False)


def export_kwargs(ee, grid: Grid, dtype):
    """Pin the output to `grid` (CRS, affine transform, extent) and tag masked pixels with NODATA[dtype]."""
    return {"crs": grid.crs, "crsTransform": [grid.res, 0, grid.left, 0, -grid.res, grid.top],
            "region": region(ee, grid), "maxPixels": 1e10, "fileFormat": "GeoTIFF", "folder": DRIVE_FOLDER,
            "formatOptions": {"noData": NODATA[dtype]}}


def task(name, build, grid: Grid, bands, dtype, note="", count=None):
    """A deferred export: build(ee) returns the ee.Image; count(ee) (optional) returns the number of source images.
    Nothing touches Earth Engine until run_exports --check or --start."""
    size = grid.width * grid.height * len(bands) * BYTES[dtype]
    return {"name": name, "build": build, "count": count, "grid": grid, "bands": bands, "dtype": dtype,
            "bytes_uncompressed": size, "note": note}


def table_task(name, build, est_bytes, note="", count=None, grid: Grid = None):
    """A deferred table (vector) export to Drive as GeoJSON: build(ee) returns the ee.FeatureCollection, count(ee) its
    number of features. est_bytes is an estimate (vector size is not known before the export). grid = the extent
    the features are clipped to (for the plan table only)."""
    return {"name": name, "build": build, "count": count, "grid": grid or COMMON, "bands": ["features"],
            "dtype": "vector", "bytes_uncompressed": int(est_bytes), "note": note, "kind": "table"}


def table_export_kwargs():
    return {"folder": DRIVE_FOLDER, "fileFormat": "GeoJSON"}
