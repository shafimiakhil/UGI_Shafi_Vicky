"""Stage A4 (2 Nov 2026): the GEOBIA map (Rodriguez Gonzalez et al. 2023) warped once from EPSG:32721 to GRID_1M by
nearest neighbour (CLAUDE.md Section 11.3 b).

Input (read-only): raw_data/Clasification_CBA_GEOBIA/clasification_GEOBIA_CBA_1m.tif (1 m, EPSG:32721, uint8, codes 1-7,
no nodata tag). Output in work_ms/derived/geobia2017/:
  geobia2017_grid1m.tif   uint8 on GRID_1M (EPSG:32720, 24297 x 24381, origin 375327, 6535833); codes 1-7 as in the
                          source; 0 = nodata, outside the source footprint (the source holds no 0, checked below)
  class_counts.csv        pixel counts and shares per code: source (whole map), source pixels whose centre lies in the
                          common extent, warped map
  shadow_test.csv         the code-7 shadow test on the source and on the warped map: for the pixels of each code, the
                          share of building pixels (code 4) 10 m toward the sun azimuth and 10 m away from it, with the
                          sun azimuth converted to grid azimuth in each CRS (whole map and a 3 x 3 km downtown window)
  geobia2017_downtown.jpg, geobia2017_suburban.jpg  1.5 x 1.5 km windows, legend only
  summary.json            run record; also the share of code 7 in the 3 x 3 km downtown window against the whole map,
                          and with the window centre moved 500 m in the four directions (downtown_window_3km)
Codes (CLAUDE.md 11.2): 1 Trees, 2 Low vegetation, 3 Water, 4 Buildings, 5 Asphalt, 6 Bare soil, 7 Shadows (code 7
inferred, to be confirmed by Dr. Marinelli).

From work_ms/:  python -I scripts/a4_geobia_warp.py
"""
import os

os.environ["GDAL_PAM_ENABLED"] = "NO"

import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np                                  # noqa: E402
import rasterio                                     # noqa: E402
from pyproj import Geod, Transformer                # noqa: E402
from rasterio.warp import Resampling, reproject     # noqa: E402

from itareo import PROJECT, W                       # noqa: E402
from itareo.grid import COMMON, GRID_1M             # noqa: E402

SRC = PROJECT / "raw_data/Clasification_CBA_GEOBIA/clasification_GEOBIA_CBA_1m.tif"
OUT = W / "derived" / "geobia2017"
NAMES = {1: "Trees", 2: "Low vegetation", 3: "Water", 4: "Buildings", 5: "Asphalt", 6: "Bare soil", 7: "Shadows"}
COLORS = {0: "#ffffff", 1: "#1b7837", 2: "#a6dba0", 3: "#2166ac", 4: "#d6604d", 5: "#7f7f7f", 6: "#dfc27d", 7: "#2d004b"}
SUN_AZIMUTH_DEG = 67.1          # Pleiades 1A, 12 Feb 2017 (CLAUDE.md 11.2), clockwise from true north
DOWNTOWN = (-64.1836, -31.4167)  # Plaza San Martin (lon, lat)
SUBURBAN = (-64.2300, -31.3800)  # residential north-west of the centre (lon, lat)
NODATA = 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def grid_azimuth(crs, lon, lat, true_az, dist=1000.0):
    """Grid bearing (deg clockwise from grid north) of the geodesic leaving (lon, lat) at true azimuth true_az."""
    lon2, lat2, _ = Geod(ellps="WGS84").fwd(lon, lat, true_az, dist)
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x1, y1 = t.transform(lon, lat)
    x2, y2 = t.transform(lon2, lat2)
    return float(np.degrees(np.arctan2(x2 - x1, y2 - y1)) % 360)


def shadow_test(arr, res, az_grid, nodata=None, dist_m=10.0, win=None):
    """Per code: share of code-4 pixels at dist_m toward az_grid and away from it (nearest pixel), over neighbours
    inside the array and not nodata. win = (r0, r1, c0, c1) restricts the code pixels (not the neighbours) to a
    window."""
    dc = int(round(dist_m * np.sin(np.radians(az_grid)) / res))
    dr = int(round(-dist_m * np.cos(np.radians(az_grid)) / res))
    H, Wd = arr.shape
    out = {}
    for code in range(1, 8):
        if win is None:
            rr, cc = np.nonzero(arr == code)
        else:
            rr, cc = np.nonzero(arr[win[0]:win[1], win[2]:win[3]] == code)
            rr, cc = rr + win[0], cc + win[2]
        row = {"pixels": int(rr.size), "offset_px_col_row_toward_sun": [dc, dr]}
        for name, sgn in (("toward_sun", 1), ("away_from_sun", -1)):
            r2, c2 = rr + sgn * dr, cc + sgn * dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < Wd)
            nb = arr[r2[ok], c2[ok]]
            if nodata is not None:
                nb = nb[nb != nodata]
            row[f"building_share_{name}"] = round(float((nb == 4).mean()), 4) if nb.size else None
            row[f"neighbours_{name}"] = int(nb.size)
        out[code] = row
    return out


def window(transform, crs, lonlat, half_m):
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*lonlat)
    col, row = ~transform * (x, y)
    h = int(round(half_m / abs(transform.a)))
    return int(round(row)) - h, int(round(row)) + h, int(round(col)) - h, int(round(col)) + h, (x, y)


def counts(a):
    return np.bincount(a.ravel(), minlength=256)[:8].astype(np.int64)


def save_jpg(arr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap([COLORS[i] for i in range(8)])
    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=150)
    ax.imshow(arr, cmap=cmap, vmin=-0.5, vmax=7.5, interpolation="nearest")
    ax.set_axis_off()
    present = [c for c in range(1, 8) if (arr == c).any()] + ([0] if (arr == 0).any() else [])
    ax.legend(handles=[Patch(color=COLORS[c], label=f"{c} {NAMES.get(c, 'nodata')}") for c in present],
              loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path, format="jpg", bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"source": str(SRC.relative_to(PROJECT)), "source_sha256": sha256(SRC), "target_grid": GRID_1M.__dict__,
            "resampling": "nearest", "nodata_out": NODATA, "started": dt.datetime.now().isoformat(timespec="seconds")}
    with rasterio.open(SRC) as s:
        src = s.read(1)
        s_tr, s_crs, s_nod = s.transform, s.crs, s.nodata
        summ["source_header"] = {"crs": s_crs.to_string(), "transform": list(s_tr)[:6], "width": s.width,
                                 "height": s.height, "dtype": s.dtypes[0], "nodata_tag": s_nod}
    c_src = counts(src)
    summ["source_values_outside_1_7"] = int(src.size - c_src[1:8].sum())
    assert c_src[0] == 0, "the source holds 0: choose another nodata code"
    # source pixels whose centre lies in the common extent (row by row)
    tr = Transformer.from_crs(s_crs, COMMON.crs, always_xy=True)
    l, b, r, t = COMMON.bounds
    c_in = np.zeros(8, np.int64)
    cols = np.arange(src.shape[1]) + 0.5
    for i0 in range(0, src.shape[0], 512):
        rows = np.arange(i0, min(i0 + 512, src.shape[0])) + 0.5
        C, R = np.meshgrid(cols, rows)
        X, Y = s_tr * (C, R)
        x2, y2 = tr.transform(X, Y)
        inside = (x2 >= l) & (x2 < r) & (y2 > b) & (y2 <= t)
        c_in += np.bincount(src[i0:i0 + rows.size][inside], minlength=8)[:8]
    # the warp
    dst = np.zeros((GRID_1M.height, GRID_1M.width), np.uint8)
    reproject(src, dst, src_transform=s_tr, src_crs=s_crs, dst_transform=GRID_1M.transform, dst_crs=GRID_1M.crs,
              resampling=Resampling.nearest, src_nodata=NODATA, dst_nodata=NODATA, num_threads=8)
    c_dst = counts(dst)
    prof = GRID_1M.profile(count=1, dtype="uint8", nodata=NODATA, tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(OUT / "geobia2017_grid1m.tif", "w", **prof) as d:
        d.write(dst, 1)
        d.set_band_description(1, "GEOBIA 2017 class (1 Trees, 2 Low veg, 3 Water, 4 Buildings, 5 Asphalt, 6 Bare soil, 7 Shadows; 0 nodata)")
    with open(OUT / "class_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "class", "source_pixels", "source_share_pct", "source_in_common_extent_pixels",
                    "source_in_common_extent_share_pct", "grid1m_pixels", "grid1m_share_pct_of_valid",
                    "grid1m_minus_source_in_extent_pixels", "relative_difference_pct"])
        for code in range(0, 8):
            sp = c_src[code]
            si = c_in[code]
            gp = c_dst[code]
            w.writerow([code, NAMES.get(code, "nodata (outside the source footprint)"), sp,
                        round(100 * sp / c_src[1:].sum(), 4) if code else "",
                        si, round(100 * si / c_in[1:].sum(), 4) if code else "",
                        gp, round(100 * gp / c_dst[1:].sum(), 4) if code else "",
                        int(gp) - int(si), round(100 * (int(gp) - int(si)) / si, 4) if code and si else ""])
    summ["counts"] = {"source": c_src.tolist(), "source_in_common_extent": c_in.tolist(), "grid1m": c_dst.tolist()}
    summ["grid1m_valid_share_of_common_extent"] = round(float(c_dst[1:].sum() / dst.size), 6)

    # shadow test, source and warped
    lon0, lat0 = DOWNTOWN
    az_src = grid_azimuth(s_crs.to_string(), lon0, lat0, SUN_AZIMUTH_DEG)
    az_dst = grid_azimuth(GRID_1M.crs, lon0, lat0, SUN_AZIMUTH_DEG)
    st_src = shadow_test(src, abs(s_tr.a), az_src)
    st_dst = shadow_test(dst, GRID_1M.res, az_dst, nodata=NODATA)
    w_src = window(s_tr, s_crs.to_string(), DOWNTOWN, 1500.0)[:4]
    w_dst = window(GRID_1M.transform, GRID_1M.crs, DOWNTOWN, 1500.0)[:4]
    st_src_dt = shadow_test(src, abs(s_tr.a), az_src, win=w_src)
    st_dst_dt = shadow_test(dst, GRID_1M.res, az_dst, nodata=NODATA, win=w_dst)
    win = {}
    for name, a, trf, crs in (("source", src, s_tr, s_crs.to_string()), ("grid1m", dst, GRID_1M.transform, GRID_1M.crs)):
        r0, r1, c0, c1, xy = window(trf, crs, DOWNTOWN, 1500.0)
        sub = a[max(r0, 0):r1, max(c0, 0):c1]
        v = sub[sub != NODATA]
        whole = a[a != NODATA]
        win[name] = {"centre_xy": [round(xy[0], 1), round(xy[1], 1)], "rows": [r0, r1], "cols": [c0, c1],
                     "code7_share_window_pct": round(100 * float((v == 7).mean()), 2),
                     "code7_share_whole_pct": round(100 * float((whole == 7).mean()), 2)}
        # sensitivity: centre shifted by 500 m in the four directions
        sens = []
        for dx, dy in ((500, 0), (-500, 0), (0, 500), (0, -500)):
            cc0, rr0 = c0 + int(dx / abs(trf.a)), r0 - int(dy / abs(trf.a))
            s2 = a[max(rr0, 0):rr0 + (r1 - r0), max(cc0, 0):cc0 + (c1 - c0)]
            v2 = s2[s2 != NODATA]
            sens.append(round(100 * float((v2 == 7).mean()), 2))
        win[name]["code7_share_window_shifted_500m_pct"] = sens
    with open(OUT / "shadow_test.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["map", "area", "crs", "sun_true_azimuth_deg", "sun_grid_azimuth_deg", "code", "class", "pixels",
                    "offset_col_row_px", "building_share_toward_sun", "building_share_away_from_sun",
                    "neighbours_toward", "neighbours_away"])
        for name, area, st, crs, az in (("source", "whole map", st_src, s_crs.to_string(), az_src),
                                        ("grid1m", "whole map", st_dst, GRID_1M.crs, az_dst),
                                        ("source", "downtown 3 x 3 km", st_src_dt, s_crs.to_string(), az_src),
                                        ("grid1m", "downtown 3 x 3 km", st_dst_dt, GRID_1M.crs, az_dst)):
            for code, r in st.items():
                w.writerow([name, area, crs, SUN_AZIMUTH_DEG, round(az, 3), code, NAMES[code], r["pixels"],
                            "|".join(map(str, r["offset_px_col_row_toward_sun"])), r["building_share_toward_sun"],
                            r["building_share_away_from_sun"], r["neighbours_toward_sun"], r["neighbours_away_from_sun"]])
    summ["shadow_test"] = {"source": st_src, "grid1m": st_dst, "source_downtown": st_src_dt, "grid1m_downtown": st_dst_dt,
                           "sun_grid_azimuth": {"source": az_src, "grid1m": az_dst},
                           "downtown_windows_rows_cols": {"source": w_src, "grid1m": w_dst}}
    summ["downtown_window_3km"] = win

    # JPGs on the warped map, 1.5 x 1.5 km
    for name, ll in (("downtown", DOWNTOWN), ("suburban", SUBURBAN)):
        r0, r1, c0, c1, xy = window(GRID_1M.transform, GRID_1M.crs, ll, 750.0)
        save_jpg(dst[r0:r1, c0:c1], OUT / f"geobia2017_{name}.jpg")
        summ[f"jpg_{name}"] = {"lonlat": ll, "centre_xy_epsg32720": [round(xy[0], 1), round(xy[1], 1)],
                               "rows": [r0, r1], "cols": [c0, c1]}
    summ["output_sha256"] = sha256(OUT / "geobia2017_grid1m.tif")
    summ["seconds"] = round(time.time() - t0, 1)
    summ["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("shadow_test",)}, indent=1, default=str))
    print(json.dumps(summ["shadow_test"], indent=1, default=str))


if __name__ == "__main__":
    main()
