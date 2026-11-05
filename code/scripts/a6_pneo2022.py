"""Stage A6 (2 Nov 2026): Pleiades Neo, Oct 2022, MS-FS (6 bands) of strips 1_2, 1_3, 1_5, 1_6 (order WO_000047718)
on GRID_1M by one bilinear warp, without pansharpening (CLAUDE.md Section 11.3 b).

Inputs (read-only): the four zips in raw_data/pleiades_neo/. Only the MS-FS RGB and NED tiles whose footprint
intersects the common extent are extracted (36 tiles, 10.05 GB; the whole MS-FS set of the four strips is 18.3 GB),
with the MS-FS DIMAP XML and the MASKS GML, into work_ms/derived/pneo2022/extracted/<strip>/ (file names kept; the
folders are flattened because the member paths exceed the Windows path limit; each file's member path is in
extract_log.json); the zips are not modified. No PAN tile is extracted.
Product facts (DIMAP): radiometric processing REFLECTANCE, Band_Reflectance GAIN 10000, BIAS 0 (reflectance =
DN / 10000), NODATA 0, SATURATED 32767; EPSG:4326, 1.08e-5 deg pixels; RGB tiles hold R, G, B, NED tiles NIR, RE, DB
(Raster_Display; checked below against the pixel values).
Method:
  1. per strip, a 6-band VRT (R, G, B from the RGB tiles, NIR, RE, DB from the NED tiles) on the strip's own lattice;
  2. per strip, one bilinear warp to GRID_1M (EPSG:32720), source nodata 0; saturated source pixels (32767) are
     counted first and the warp refuses to run if there is any;
  3. mosaic on GRID_1M, pixel by pixel, without further resampling: the valid, cloud-free strip with the lowest
     incidence angle (1_6 4.9 deg, 1_5 15.8, 1_3 18.5, 1_2 19.4); where only cloudy pixels exist (strip 1_2, CLD
     mask), the cloudy pixel is kept and flagged. Each output pixel comes from one strip and one resampling.
Outputs on GRID_1M in work_ms/derived/pneo2022/:
  pneo2022_toa_grid1m.tif       6 bands uint16 TOA reflectance x 10000 (R, G, B, NIR, RE, DB), nodata 65535
  pneo2022_strip_id_grid1m.tif  uint8, the strip number (2, 3, 5, 6 for 1_2, 1_3, 1_5, 1_6), 0 none
  pneo2022_date_grid1m.tif      uint32 acquisition date YYYYMMDD, 0 none
  pneo2022_cloud_grid1m.tif     uint8, 1 = inside the CLD mask of the strip used, 0 = clear, 255 = no data
  vrt/<strip>.vrt, extracted/, extract_log.json, warp_log.json (README.md written after the run)
  With --convention geotiff the warp outputs, its VRTs and its warp_log.json go to v1_geotiff_tags/.
Georeferencing check (--georef-check): (1) the DIMAP Dataset_Extent vertices carry ROW and COL, and the DIMAP Raster_CRS
declares PIXEL_CRS_TYPE CELL, PIXEL_ORIENTATION UL, PIXEL_ORIGIN 1, so a vertex is the upper-left corner of pixel
(ROW, COL); for every vertex and every extracted tile, that corner as predicted by the tile's world file (.TFW) and by
the tile's GeoTIFF tags is compared with the vertex (the DIMAP geoposition row is a consistency check: it holds by
construction). (2) PAN against MS-FS of the same strip, from the PAN DIMAP read inside the zip: same origin, pixel
ratio and size ratio, and how far their east and south edges part under each convention (the GeoTIFF tags scale each
product by (N - 1) / N of its own N). Outputs georef_metadata_check.csv and .json.

From work_ms/:  python -I scripts/a6_pneo2022.py --extract   then   python -I scripts/a6_pneo2022.py --warp
                python -I scripts/a6_pneo2022.py --georef-check
"""
import os

os.environ["GDAL_PAM_ENABLED"] = "NO"

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np                                   # noqa: E402
import rasterio                                      # noqa: E402
from rasterio import features                        # noqa: E402
from rasterio.transform import from_origin           # noqa: E402
from rasterio.warp import Resampling, reproject, transform_geom   # noqa: E402
from rasterio.windows import Window                  # noqa: E402
from shapely.geometry import box, mapping, shape     # noqa: E402

from itareo import PROJECT, W                        # noqa: E402
from itareo.grid import COMMON, GRID_1M              # noqa: E402

RAW = PROJECT / "raw_data" / "pleiades_neo"
OUT = W / "derived" / "pneo2022"
EXT = OUT / "extracted"
CONVENTION = "tfw"                                   # set by --convention
RUN = OUT                                            # outputs of the warp; OUT / "v1_geotiff_tags" for the other convention
STRIPS = {"1_6": 6, "1_5": 5, "1_3": 3, "1_2": 2}          # priority order: lowest incidence first (DIMAP centre)
BANDS = ["R", "G", "B", "NIR", "RE", "DB"]
ND_DN, SAT_DN, ND16 = 0, 32767, 65535
GTIFF = dict(compress="zstd", zstd_level=9, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES",
             num_threads="ALL_CPUS")


def zip_of(strip):
    z = sorted(RAW.glob(f"*WO_000047718_{strip}_SAL*.zip"))
    assert len(z) == 1, (strip, z)
    return z[0]


def common_4326():
    return shape(transform_geom(COMMON.crs, "EPSG:4326", mapping(box(*COMMON.bounds)), precision=-1))


def tile_bounds(transform, w, h):
    x0, y0 = transform.c, transform.f
    return box(x0, y0 + h * transform.e, x0 + w * transform.a, y0)


def plan_extract(strip):
    """Members to extract: MS-FS RGB/NED tiles (+ TFW) intersecting the common extent, the MS-FS DIM XML, MASKS GML."""
    zp = zip_of(strip)
    ext = common_4326()
    take, skipped = [], []
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for n in names:
            if "/IMG_01_PNEO" not in n or "MS-FS" not in n:
                continue
            if n.endswith(".TIF") and re.search(r"_(RGB|NED)_R\d+C\d+\.TIF$", n):
                with rasterio.open(f"/vsizip/{zp.as_posix()}/{n}") as s:
                    inter = tile_bounds(s.transform, s.width, s.height).intersects(ext)
                (take if inter else skipped).append(n)
                if inter and n[:-4] + ".TFW" in names:
                    take.append(n[:-4] + ".TFW")
            elif re.search(r"/DIM_.*\.XML$", n) or "/MASKS/" in n:
                take.append(n)
    return zp, take, skipped


def extract(strip):
    zp, take, skipped = plan_extract(strip)
    dst_root = EXT / strip
    names = [Path(n).name for n in take]
    assert len(names) == len(set(names)), f"file names repeat in {strip}: the flat layout would overwrite"
    out = []
    with zipfile.ZipFile(zp) as z:
        for n in take:
            dst = dst_root / Path(n).name          # flat: the member paths exceed the Windows path limit
            dst.parent.mkdir(parents=True, exist_ok=True)
            info = z.getinfo(n)
            with z.open(n) as src, open(dst, "wb") as fh:      # zipfile checks the CRC at the end of the read
                shutil.copyfileobj(src, fh, 1 << 24)
            out.append({"member": n, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"})
    return {"strip": strip, "zip": zp.name, "extracted": out, "skipped_tiles_outside_extent": skipped,
            "extracted_bytes": sum(o["bytes"] for o in out)}


def dimap_ms(strip):
    zp = zip_of(strip)
    xml = sorted((EXT / strip).rglob("DIM_*MS-FS*.XML"))[0]
    r = ET.parse(xml).getroot()
    cen = [lv for lv in r.iter("Located_Geometric_Values") if lv.findtext("LOCATION_TYPE") == "Center"][0]
    return {"xml": str(xml.relative_to(OUT)), "date": r.findtext(".//IMAGING_DATE"), "time": r.findtext(".//IMAGING_TIME"),
            "incidence": float(cen.findtext(".//INCIDENCE_ANGLE")), "sun_elevation": float(cen.findtext(".//SUN_ELEVATION")),
            "sun_azimuth": float(cen.findtext(".//SUN_AZIMUTH")), "cloud_pct": float(r.findtext(".//CLOUD_COVERAGE")),
            "reflectance_gain": {e.findtext("BAND_ID"): float(e.findtext("GAIN")) for e in r.iter("Band_Reflectance")},
            "reflectance_bias": {e.findtext("BAND_ID"): float(e.findtext("BIAS")) for e in r.iter("Band_Reflectance")},
            "special": {e.findtext("SPECIAL_VALUE_TEXT"): int(e.findtext("SPECIAL_VALUE_COUNT")) for e in r.iter("Special_Value")}}


def tile_transform(p, src):
    """Tile geotransform. CONVENTION "tfw": the world file (= DIMAP ULXMAP/XDIM, pixel size exactly 1.08e-5 deg;
    the world file gives the centre of the upper-left pixel); "geotiff": the GeoTIFF tags, whose pixel size is
    XDIM * (NCOLS - 1) / NCOLS of the strip, so the two agree at the strip origin and differ by 1 pixel at its far
    east and south edges (found by the verification of 2 Nov 2026)."""
    if CONVENTION == "geotiff":
        return src.transform
    a, d, b, e, c, f = (float(v) for v in p.with_suffix(".TFW").read_text().split())
    assert d == 0 and b == 0, f"rotated world file {p}"
    from affine import Affine
    return Affine(a, 0.0, c - a / 2.0, 0.0, e, f - e / 2.0)


def build_vrt(strip):
    """6-band VRT of the extracted tiles; returns the path and the strip lattice."""
    zp = zip_of(strip)
    tiles = {"RGB": sorted((EXT / strip).rglob("*_RGB_R*C*.TIF")), "NED": sorted((EXT / strip).rglob("*_NED_R*C*.TIF"))}
    info = {}
    for kind, paths in tiles.items():
        for p in paths:
            with rasterio.open(p) as s:
                info[p] = (tile_transform(p, s), s.width, s.height, s.crs, s.count, s.dtypes[0])
    t0 = next(iter(info.values()))[0]
    dx, dy = t0.a, t0.e
    x0 = min(v[0].c for v in info.values())
    y0 = max(v[0].f for v in info.values())
    W_, H_ = 0, 0
    pos = {}
    for p, (t, w, h, crs, cnt, dtp) in info.items():
        assert abs(t.a - dx) < 1e-15 and abs(t.e - dy) < 1e-15 and cnt == 3 and dtp == "uint16"
        ox, oy = (t.c - x0) / dx, (t.f - y0) / dy
        assert abs(ox - round(ox)) < 1e-6 and abs(oy - round(oy)) < 1e-6, f"tile off the strip lattice: {p}"
        pos[p] = (int(round(ox)), int(round(oy)), w, h)
        W_, H_ = max(W_, int(round(ox)) + w), max(H_, int(round(oy)) + h)
    crs = next(iter(info.values()))[3]
    vrt = RUN / "vrt" / f"{strip}.vrt"
    vrt.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'<VRTDataset rasterXSize="{W_}" rasterYSize="{H_}">', f"  <SRS>{crs.to_wkt()}</SRS>",
             f"  <GeoTransform>{x0!r}, {dx!r}, 0, {y0!r}, 0, {dy!r}</GeoTransform>"]
    for b, name in enumerate(BANDS, start=1):
        kind, sb = ("RGB", b) if b <= 3 else ("NED", b - 3)
        lines += [f'  <VRTRasterBand dataType="UInt16" band="{b}">', f"    <NoDataValue>{ND_DN}</NoDataValue>",
                  f"    <Description>{name}</Description>"]
        for p in tiles[kind]:
            ox, oy, w, h = pos[p]
            rel = os.path.relpath(p, vrt.parent).replace("\\", "/")
            lines += ["    <SimpleSource>", f'      <SourceFilename relativeToVRT="1">{rel}</SourceFilename>',
                      f"      <SourceBand>{sb}</SourceBand>", f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                      f'      <DstRect xOff="{ox}" yOff="{oy}" xSize="{w}" ySize="{h}"/>', "    </SimpleSource>"]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    vrt.write_text("\n".join(lines), encoding="utf-8")
    return vrt, {"x0": x0, "y0": y0, "dx": dx, "dy": dy, "width": W_, "height": H_, "tiles": len(info)}


def cloud_polygons(strip, lattice):
    """CLD mask polygons of the strip in GRID_1M coordinates, and one record per polygon (bounds in EPSG:32720, area,
    whether it intersects the common extent and the strip's extracted footprint). A GML that exists but cannot be read
    stops the run: an empty list would pass cloudy pixels as clear. The GML stores posList in lat-lon order; GDAL's GML
    driver swaps the axes (EPSG:4326 lat-lon), which the footprint check makes visible."""
    gml = sorted((EXT / strip).rglob("CLD_*MS-FS*.GML"))
    if not gml:
        return [], []
    import pyogrio
    tmp = Path(os.environ.get("TEMP", ".")) / f"pneo_cld_{strip}.gml"     # a copy: GDAL writes a .gfs next to a GML
    shutil.copyfile(gml[0], tmp)
    df = pyogrio.read_dataframe(tmp, WRITE_GFS="NO")     # raises on a read error; no .gfs
    if df.empty:
        return [], []
    foot = box(lattice["x0"], lattice["y0"] + lattice["height"] * lattice["dy"],
               lattice["x0"] + lattice["width"] * lattice["dx"], lattice["y0"])
    ext = box(*GRID_1M.bounds)
    geoms, info = [], []
    for g in df.geometry:
        if g is None:
            continue
        gg = transform_geom(df.crs.to_string(), GRID_1M.crs, mapping(g), precision=-1)
        sg = shape(gg)
        geoms.append(gg)
        info.append({"bounds_epsg32720": [round(v, 1) for v in sg.bounds], "area_km2": round(sg.area / 1e6, 4),
                     "intersects_common_extent": bool(sg.intersects(ext)),
                     "intersects_extracted_footprint": bool(g.intersects(foot)),
                     "lon_lat_centroid": [round(g.centroid.x, 5), round(g.centroid.y, 5)]})
    return geoms, info


def band_check(vrt):
    """Pixel-value check of the band layout on a central window of the strip: correlations and means."""
    with rasterio.open(vrt) as s:
        h, w = s.height, s.width
        win = Window(w // 2 - 1000, h // 2 - 1000, 2000, 2000)
        a = s.read(window=win).astype(np.float64)
    ok = (a > 0).all(0) & (a < SAT_DN).all(0)
    x = a[:, ok] / 10000.0
    c = np.corrcoef(x)
    ndvi = (x[3] - x[0]) / (x[3] + x[0])
    return {"means": dict(zip(BANDS, np.round(x.mean(1), 4).tolist())),
            "corr_DB_vs_B": round(float(c[5, 2]), 4), "corr_DB_vs_R": round(float(c[5, 0]), 4),
            "corr_NIR_vs_RE": round(float(c[3, 4]), 4), "corr_NIR_vs_R": round(float(c[3, 0]), 4),
            "ndvi_p5_p50_p95": np.round(np.percentile(ndvi, [5, 50, 95]), 3).tolist(), "pixels": int(ok.sum())}


def run_extract():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(extract, list(STRIPS)))
    log = {"started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"), "extract": res,
           "extracted_bytes_total": sum(r["extracted_bytes"] for r in res), "seconds": round(time.time() - t0, 1),
           "zips_sha256": "docs/inventory/marinelli_2026-10-30/files.csv"}
    (OUT / "extract_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("extracted", log["extracted_bytes_total"] / 1e9, "GB in", log["seconds"], "s", flush=True)


def count_saturated():
    """Saturated (32767) values in the extracted tiles; the warp assumes there are none (checked before it runs)."""
    total = {}
    for strip in STRIPS:
        zp = zip_of(strip)
        n = 0
        for p in sorted((EXT / strip).rglob("*_R*C*.TIF")):
            with rasterio.open(p) as s:
                for r0 in range(0, s.height, 2048):
                    a = s.read(window=Window(0, r0, s.width, min(2048, s.height - r0)))
                    n += int((a == SAT_DN).sum())
        total[strip] = n
    return total


def run_warp():
    t0 = time.time()
    log = {"started": dt.datetime.now().isoformat(timespec="seconds"), "strips": {}}
    sat = count_saturated()
    log["saturated_values_in_extracted_tiles"] = sat
    if any(sat.values()):
        raise SystemExit(f"saturated values present {sat}: mask them before a bilinear warp")
    vrts, clouds = {}, {}
    for strip in STRIPS:
        meta = dimap_ms(strip)
        vrt, lat = build_vrt(strip)
        vrts[strip] = vrt
        clouds[strip], cinfo = cloud_polygons(strip, lat)
        log["strips"][strip] = {"dimap": meta, "vrt": str(vrt.relative_to(OUT)), "convention": CONVENTION, "lattice": lat,
                                "cloud_polygons": len(clouds[strip]), "cloud_polygon_records": cinfo,
                                "band_check": band_check(vrt)}
        print(strip, meta["date"], meta["incidence"], lat, len(clouds[strip]), "cloud polygons", flush=True)
    order = sorted(STRIPS, key=lambda s: log["strips"][s]["dimap"]["incidence"])
    log["priority_order"] = order
    dates = {s: int(log["strips"][s]["dimap"]["date"].replace("-", "")) for s in STRIPS}
    RUN.mkdir(parents=True, exist_ok=True)
    paths = {"toa": RUN / "pneo2022_toa_grid1m.tif", "sid": RUN / "pneo2022_strip_id_grid1m.tif",
             "date": RUN / "pneo2022_date_grid1m.tif", "cloud": RUN / "pneo2022_cloud_grid1m.tif"}
    profs = {"toa": GRID_1M.profile(count=6, dtype="uint16", nodata=ND16, predictor=2, **GTIFF),
             "sid": GRID_1M.profile(count=1, dtype="uint8", nodata=0, **GTIFF),
             "date": GRID_1M.profile(count=1, dtype="uint32", nodata=0, **GTIFF),
             "cloud": GRID_1M.profile(count=1, dtype="uint8", nodata=255, **GTIFF)}
    cnt = {"valid": 0, "cloudy_kept": 0, "cloudy_replaced": 0, **{f"from_{s}": 0 for s in STRIPS}}
    srcs = {s: rasterio.open(v) for s, v in vrts.items()}
    dsts = {k: rasterio.open(p, "w", **profs[k]) for k, p in paths.items()}
    for i, b in enumerate(BANDS):
        dsts["toa"].set_band_description(i + 1, f"TOA reflectance x 10000, {b}")
    dsts["sid"].set_band_description(1, "strip: 2 = 1_2 (23 Oct), 3 = 1_3 (23 Oct), 5 = 1_5 (28 Oct), 6 = 1_6 (28 Oct)")
    dsts["date"].set_band_description(1, "acquisition date YYYYMMDD")
    dsts["cloud"].set_band_description(1, "1 = inside the DIMAP CLD mask of the strip used, 0 = clear")
    try:
        for r0 in range(0, GRID_1M.height, 2048):
            nr = min(2048, GRID_1M.height - r0)
            tr = from_origin(GRID_1M.left, GRID_1M.top - r0, 1.0, 1.0)
            shape_ = (nr, GRID_1M.width)
            out = np.full((6,) + shape_, ND16, np.uint16)
            sid = np.zeros(shape_, np.uint8)
            date = np.zeros(shape_, np.uint32)
            cloud = np.full(shape_, 255, np.uint8)
            filled = np.zeros(shape_, bool)
            pending = []                                   # cloudy candidates, used only where nothing clear exists
            for strip in order:
                buf = np.full((6,) + shape_, np.nan, np.float32)
                reproject(rasterio.band(srcs[strip], [1, 2, 3, 4, 5, 6]), buf, dst_transform=tr, dst_crs=GRID_1M.crs,
                          resampling=Resampling.bilinear, src_nodata=ND_DN, dst_nodata=np.nan, num_threads=8)
                valid = np.isfinite(buf).all(0)
                if not valid.any():
                    continue
                cl = np.zeros(shape_, bool)
                if clouds[strip]:
                    cl = features.rasterize([(g, 1) for g in clouds[strip]], out_shape=shape_, transform=tr, fill=0,
                                            dtype="uint8").astype(bool)
                take = valid & ~cl & ~filled
                vals = np.clip(np.rint(buf), 0, ND16 - 1).astype(np.uint16)
                out[:, take] = vals[:, take]
                sid[take], date[take], cloud[take] = STRIPS[strip], dates[strip], 0
                cnt[f"from_{strip}"] += int(take.sum())
                filled |= take
                if cl.any():
                    pending.append((strip, valid & cl, vals))
                del buf
            for strip, cand, vals in pending:                 # cloudy pixels only where no clear strip exists
                take = cand & ~filled
                out[:, take] = vals[:, take]
                sid[take], date[take], cloud[take] = STRIPS[strip], dates[strip], 1
                cnt["cloudy_kept"] += int(take.sum())
                cnt["cloudy_replaced"] += int((cand & filled).sum())
                cnt[f"from_{strip}"] += int(take.sum())
                filled |= take
            cnt["valid"] += int(filled.sum())
            win = Window(0, r0, GRID_1M.width, nr)
            dsts["toa"].write(out, window=win)
            dsts["sid"].write(sid, 1, window=win)
            dsts["date"].write(date, 1, window=win)
            dsts["cloud"].write(cloud, 1, window=win)
            print("rows", r0, "valid so far", cnt["valid"], round(time.time() - t0), "s", flush=True)
    finally:
        for d in dsts.values():
            d.close()
        for s in srcs.values():
            s.close()
    total = GRID_1M.width * GRID_1M.height
    log.update({"counts": cnt, "valid_share_of_common_extent": cnt["valid"] / total,
                "cloudy_kept_share": cnt["cloudy_kept"] / total, "seconds": round(time.time() - t0, 1),
                "finished": dt.datetime.now().isoformat(timespec="seconds")})
    h = {}
    for k, p in paths.items():
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 23), b""):
                sha.update(b)
        h[p.name] = sha.hexdigest()
    log["outputs_sha256"] = h
    (RUN / "warp_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "strips"}, indent=1, default=str))


def pan_ms_consistency(strip, ms):
    """PAN against MS-FS of one strip from their DIMAP files (the PAN DIMAP is read from the zip, nothing extracted):
    edges of the two products under the DIMAP/TFW pixel and under the GeoTIFF-tag pixel XDIM (N - 1) / N."""
    with zipfile.ZipFile(zip_of(strip)) as z:
        name = [n for n in z.namelist() if re.search(r"/IMG_\d+_PNEO\d_PAN/DIM_.*_PAN_.*\.XML$", n)][0]
        r = ET.fromstring(z.read(name))
    pan = {"ulx": float(r.findtext(".//ULXMAP")), "uly": float(r.findtext(".//ULYMAP")),
           "xdim": float(r.findtext(".//XDIM")), "ydim": float(r.findtext(".//YDIM")),
           "ncols": int(r.findtext(".//Raster_Dimensions/NCOLS")), "nrows": int(r.findtext(".//Raster_Dimensions/NROWS")),
           "pixel_orientation": r.findtext(".//PIXEL_ORIENTATION")}
    east = lambda g, k: g["ulx"] + g["ncols"] * g["xdim"] * k(g["ncols"])
    south = lambda g, k: g["uly"] - g["nrows"] * g["ydim"] * k(g["nrows"])
    tfw, gtf = (lambda n: 1.0), (lambda n: (n - 1) / n)
    px = pan["xdim"]
    return {"pan_dimap": name.split("/")[-1], "pan": pan,
            "same_origin": pan["ulx"] == ms["ulx"] and pan["uly"] == ms["uly"],
            "ms_over_pan_pixel": [ms["xdim"] / pan["xdim"], ms["ydim"] / pan["ydim"]],
            "pan_over_ms_size": [pan["ncols"] / ms["ncols"], pan["nrows"] / ms["nrows"]],
            "east_edge_pan_minus_ms_pan_px": {"dimap_tfw": round((east(pan, tfw) - east(ms, tfw)) / px, 6),
                                              "geotiff_tags": round((east(pan, gtf) - east(ms, gtf)) / px, 6)},
            "south_edge_pan_minus_ms_pan_px": {"dimap_tfw": round((south(pan, tfw) - south(ms, tfw)) / px, 6),
                                               "geotiff_tags": round((south(pan, gtf) - south(ms, gtf)) / px, 6)}}


def georef_check():
    """Vendor-metadata test of the two georeferencing conventions (see the module docstring)."""
    import csv
    from affine import Affine
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    rows, summary = [], {}
    for strip in STRIPS:
        xml = sorted((EXT / strip).rglob("DIM_*MS-FS*.XML"))[0]
        r = ET.parse(xml).getroot()
        ulx, uly = float(r.findtext(".//ULXMAP")), float(r.findtext(".//ULYMAP"))
        xdim, ydim = float(r.findtext(".//XDIM")), float(r.findtext(".//YDIM"))
        ncols, nrows = int(r.findtext(".//Raster_Dimensions/NCOLS")), int(r.findtext(".//Raster_Dimensions/NROWS"))
        size = r.find(".//Regular_Tiling/NTILES_SIZE")
        tc, tr_ = int(size.get("ncols")), int(size.get("nrows"))
        verts = [(float(v.findtext("LON")), float(v.findtext("LAT")), int(v.findtext("COL")), int(v.findtext("ROW")))
                 for v in r.find(".//Dataset_Extent").iter("Vertex")]
        tiles = sorted((EXT / strip).rglob("*_R*C*.TIF"))
        worst = {"dimap": 0.0, "tfw": 0.0, "geotiff": 0.0}
        for lon, lat, col, row in verts:
            cand = {"dimap": (None, Affine(xdim, 0, ulx, 0, -ydim, uly), 0, 0)}
            for p in tiles:
                m = re.search(r"_R(\d+)C(\d+)\.TIF$", p.name)
                R, C = int(m.group(1)), int(m.group(2))
                a_, d_, b_, e_, c_, f_ = (float(v) for v in p.with_suffix(".TFW").read_text().split())
                with rasterio.open(p) as s:
                    g = s.transform
                cand[f"tfw:{p.name}"] = (p.name, Affine(a_, 0, c_ - a_ / 2.0, 0, e_, f_ - e_ / 2.0), (C - 1) * tc, (R - 1) * tr_)
                cand[f"geotiff:{p.name}"] = (p.name, g, (C - 1) * tc, (R - 1) * tr_)
            for key, (tile, aff, c0, r0) in cand.items():
                x, y = aff * (col - 1 - c0, row - 1 - r0)        # upper-left corner of the vertex pixel
                _, _, de = geod.inv(lon, lat, x, lat)
                _, _, dn = geod.inv(lon, lat, lon, y)
                kind = key.split(":")[0]
                rows.append({"strip": strip, "vertex_col": col, "vertex_row": row, "vertex_lon": lon, "vertex_lat": lat,
                             "model": kind, "tile": tile or "", "pred_lon": round(x, 10), "pred_lat": round(y, 10),
                             "diff_east_m": round(math.copysign(de, x - lon), 3),
                             "diff_north_m": round(math.copysign(dn, y - lat), 3)})
                worst[kind] = max(worst[kind], de, dn)
        summary[strip] = {"pixel_crs": {k: r.findtext(f".//{k}") for k in ("PIXEL_CRS_TYPE", "PIXEL_ORIENTATION", "PIXEL_ORIGIN")},
                          "pan_ms_consistency": pan_ms_consistency(strip, {"ulx": ulx, "uly": uly, "xdim": xdim, "ydim": ydim,
                                                                        "ncols": ncols, "nrows": nrows}),
                          "ncols": ncols, "nrows": nrows, "xdim_deg": xdim, "vertices": len(verts), "tiles": len(tiles),
                          "max_abs_diff_m": {k: round(v, 3) for k, v in worst.items()},
                          "one_pixel_m_east_west_north_south": [round(geod.inv(ulx, uly, ulx + xdim, uly)[2], 3),
                                                                round(geod.inv(ulx, uly, ulx, uly - ydim)[2], 3)]}
    with open(OUT / "georef_metadata_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = {"finished": dt.datetime.now().isoformat(timespec="seconds"), "strips": summary,
           "reading": "max_abs_diff_m: largest east-west or north-south distance between a DIMAP Dataset_Extent vertex "
                      "and the upper-left corner of its pixel (ROW, COL) predicted by each model, over all vertices and "
                      "all extracted tiles (each tile's affine extended to the vertex). A vertex is the upper-left corner "
                      "of its pixel because the DIMAP declares PIXEL_CRS_TYPE CELL, PIXEL_ORIENTATION UL, PIXEL_ORIGIN 1 "
                      "(pixel_crs); the dimap row holds by construction and is a consistency check. "
                      "pan_ms_consistency: PAN and MS-FS share the origin and nest 4:1; their east and south edges "
                      "coincide under the DIMAP/TFW pixel and part by the given PAN pixels under the GeoTIFF tags"}
    (OUT / "georef_metadata_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--georef-check", action="store_true")
    ap.add_argument("--convention", choices=["tfw", "geotiff"], default="tfw")
    a = ap.parse_args()
    CONVENTION = a.convention
    RUN = OUT if CONVENTION == "tfw" else OUT / "v1_geotiff_tags"
    if a.plan:
        for s in STRIPS:
            zp, take, skipped = plan_extract(s)
            print(s, zp.name, len([t for t in take if t.endswith('.TIF')]), "tiles to extract,", len(skipped), "outside")
    if a.extract:
        run_extract()
    if a.warp:
        run_warp()
    if a.georef_check:
        georef_check()
