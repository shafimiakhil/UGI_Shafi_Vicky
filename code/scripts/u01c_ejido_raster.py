"""Rule 9 (CLAUDE.md 13.5): the ejido municipal on GRID_1M, and the facts of 13.5 measured on it.

  * raw_data/ejido-municipal_iAEC1kw/EjidoMunicipal.shp, read only: one polygon in Argentina zone 4 on WGS84, its vertices
    reprojected to EPSG:32720 (both projections are Transverse Mercator on the meridian 63 W);
  * one rasterisation on GRID_1M, pixel-centre rule (all_touched=False), as the UGI labels: derived/ugi_labels/
    ejido_grid1m.tif, uint8, 1 inside, 0 outside;
  * derived/ugi_labels/ejido_log.json: source hashes, the polygon area in both CRSs, the raster's pixel count and each fact
    of 13.5 beside the value measured here; derived/ugi_labels/README_ejido.md. Exit 1 if a measured fact differs from 13.5
    at the precision 13.5 gives. Refuses to overwrite.
Run from work_ms/: python scripts/u01c_ejido_raster.py
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
import numpy as np                      # noqa: E402
import pyogrio                          # noqa: E402
import rasterio                         # noqa: E402
import shapely                          # noqa: E402
from rasterio import features           # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.grid import GRID_1M         # noqa: E402
from itareo.safe_write import refuse_existing  # noqa: E402

SRC = WM.parent / "raw_data/ejido-municipal_iAEC1kw/EjidoMunicipal.shp"
LAB = WM / "derived/ugi_labels"
OUT = LAB / "ejido_grid1m.tif"
LOG = LAB / "ejido_log.json"
README = LAB / "README_ejido.md"
MAPS = WM / "runs_itareo/ugi/maps"
EXPECTED = {"grid_share_outside_pct": 2.81, "inventory_px_outside": 263, "stage_m_mapped_share_outside_pct": 0.86,
            "unmatched_components": 4312, "unmatched_centroid_outside": 138, "unmatched_area_share_outside_pct": 2.4,
            "unmatched_over_1ha": 90, "unmatched_over_1ha_centroid_outside": 2}
EXPECTED_SCORES = {"inside": {"precision": 0.642, "recall": 0.492, "f1": 0.557, "iou": 0.386},
                   "all": {"precision": 0.636, "recall": 0.492, "f1": 0.555, "iou": 0.384}}
ZOOM_PX = 1500


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    return h.hexdigest()


def scores(tp, fp, fn):
    p, r = tp / (tp + fp), tp / (tp + fn)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": 2 * p * r / (p + r),
            "iou": tp / (tp + fp + fn)}


def main():
    t0 = time.time()
    refuse_existing(OUT, LOG, README)
    with rasterio.open(LAB / "ugi_binary_grid1m.tif") as s:
        H, W, T, crs = s.height, s.width, s.transform, s.crs
        ref = s.read(1) > 0
    if (W, H) != (GRID_1M.width, GRID_1M.height) or T != GRID_1M.transform:
        raise ValueError("ugi_binary_grid1m.tif is not on GRID_1M")
    gdf = pyogrio.read_dataframe(SRC)
    if len(gdf) != 1:
        raise ValueError(f"{len(gdf)} features in the ejido layer, expected one")
    src_crs = gdf.crs.to_wkt()
    area_src = float(gdf.geometry.area.sum())
    g = gdf.to_crs(32720)
    geom = g.geometry.iloc[0]
    area_utm = float(geom.area)
    b = geom.bounds
    x0, x1, y1, y0 = T.c, T.c + W * T.a, T.f, T.f + H * T.e
    inside_grid = bool(b[0] >= x0 and b[2] <= x1 and b[1] >= y0 and b[3] <= y1)
    mask = features.rasterize([(geom, 1)], out_shape=(H, W), transform=T, fill=0, dtype="uint8", all_touched=False)
    n_in = int(mask.sum())
    prof = dict(driver="GTiff", dtype="uint8", count=1, width=W, height=H, crs=crs, transform=T, compress="deflate", tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(OUT, "w", **prof) as d:
        d.write(mask, 1)
        d.set_band_description(1, "1 = inside the ejido municipal, 0 = outside (no reference outside it)")
    with rasterio.open(OUT) as s:                                   # the raster as stored
        m = s.read(1) == 1
    if int(m.sum()) != n_in:
        raise ValueError("the stored raster differs from the rasterised mask")
    inv_out = int((ref & ~m).sum())
    with rasterio.open(MAPS / "ugi_a1_e30_vote.tif") as s:
        vote = s.read(1)
    mapped = vote == 1
    evaluated = vote != 255
    mapped_out = int((mapped & ~m).sum())
    full = scores(*(int(x.sum()) for x in (mapped & ref, mapped & ~ref & evaluated, ~mapped & ref & evaluated)))
    ins = scores(*(int(x.sum()) for x in (mapped & ref & m, mapped & ~ref & evaluated & m, ~mapped & ref & evaluated & m)))
    comps = list(csv.DictReader(open(MAPS / "components.csv")))
    xy = np.array([[float(c["x"]), float(c["y"])] for c in comps])
    area = np.array([int(c["area_m2"]) for c in comps], float)
    in_poly = shapely.contains_xy(geom, xy[:, 0], xy[:, 1])
    in_raster = m[[int(c["row"]) for c in comps], [int(c["col"]) for c in comps]]
    big = area > 10000
    zooms = {z["zoom"]: z for z in csv.DictReader(open(MAPS / "zooms.csv"))}
    z4 = zooms["4"]
    r4, c4 = int(z4["row0"]), int(z4["col0"])
    z4_in = bool(m[r4:r4 + ZOOM_PX, c4:c4 + ZOOM_PX].all())
    measured = {"grid_share_outside_pct": 100 * (1 - n_in / (H * W)), "inventory_px_outside": inv_out,
                "stage_m_mapped_share_outside_pct": 100 * mapped_out / int(mapped.sum()), "unmatched_components": len(comps),
                "unmatched_centroid_outside": int((~in_poly).sum()),
                "unmatched_area_share_outside_pct": 100 * float(area[~in_poly].sum() / area.sum()),
                "unmatched_over_1ha": int(big.sum()), "unmatched_over_1ha_centroid_outside": int((big & ~in_poly).sum())}
    digits = {"grid_share_outside_pct": 2, "stage_m_mapped_share_outside_pct": 2, "unmatched_area_share_outside_pct": 1}
    match = {k: round(measured[k], digits.get(k, 0)) == v for k, v in EXPECTED.items()}
    score_match = {a: {k: round(s_[k], 3) == v for k, v in EXPECTED_SCORES[a].items()} for a, s_ in (("inside", ins), ("all", full))}
    ok = (inside_grid and z4_in and all(match.values()) and all(v for d in score_match.values() for v in d.values()))
    log = {"source": str(SRC.relative_to(WM.parent)), "source_sha256": {p.name: sha256(p) for p in sorted(SRC.parent.glob("EjidoMunicipal.*"))},
           "source_crs": src_crs, "polygon_area_km2_source_crs": round(area_src / 1e6, 4), "polygon_area_km2_epsg32720": round(area_utm / 1e6, 4),
           "raster": str(OUT.relative_to(WM)), "raster_sha256": sha256(OUT), "pixels_inside": n_in, "pixels_grid": H * W,
           "raster_minus_polygon_m2": round(n_in - area_utm, 1), "ejido_wholly_inside_grid": inside_grid, "bounds_epsg32720": list(b),
           "zoom4_wholly_inside": z4_in, "measured": measured, "expected_13_5": EXPECTED, "match": match,
           "unmatched_centroid_outside_by_raster": int((~in_raster).sum()),
           "stage_m_vote_scores": {"inside": ins, "all": full}, "expected_scores_13_5": EXPECTED_SCORES, "score_match": score_match,
           "all_match": bool(ok), "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(LOG, "w"), indent=1, default=float)
    with open(README, "w", encoding="utf-8") as f:
        f.write("# Ejido municipal on GRID_1M (rule 9, CLAUDE.md 13.5)\n\n`python scripts/u01c_ejido_raster.py` (from `work_ms/`).\n\n")
        f.write(f"- **Source:** `{log['source']}` (one polygon, Argentina zone 4 on WGS84; {log['polygon_area_km2_source_crs']} km2 in its CRS, "
                f"{log['polygon_area_km2_epsg32720']} km2 in EPSG:32720). Hashes in `ejido_log.json`.\n")
        f.write("- **Raster:** `ejido_grid1m.tif`, uint8, 1 inside, 0 outside; one rasterisation on GRID_1M, pixel-centre rule, as the UGI "
                f"labels; {n_in:,} pixels inside; sha256 {log['raster_sha256']}.\n")
        f.write("- **Meaning (Dr. Marinelli):** inside the ejido everything outside the inventory is non-UGI; outside it there is no reference.\n")
        f.write(f"- **Facts of 13.5 measured here** (`ejido_log.json`): all match = {log['all_match']}.\n")
    print(json.dumps({k: log[k] for k in ("polygon_area_km2_source_crs", "polygon_area_km2_epsg32720", "pixels_inside", "raster_minus_polygon_m2",
                                          "ejido_wholly_inside_grid", "zoom4_wholly_inside", "measured", "match",
                                          "unmatched_centroid_outside_by_raster", "stage_m_vote_scores", "score_match", "all_match", "seconds")},
                     indent=1, default=float))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
