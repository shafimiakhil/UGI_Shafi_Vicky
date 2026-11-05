"""U0.1 (CLAUDE.md 13): UGI label rasters on GRID_1M from the final inventory raw_data/22174_UrbanGreenInfras.gpkg.

  * read from a copy in TEMP (read-only on raw_data); polygons reprojected EPSG:22174 -> EPSG:32720 (vector, exact);
  * exact duplicate geometries: one copy kept, preferring a defined type over NO DEFINIDO, then the smaller ID_NUEVO;
    listed in duplicates.csv (a first version kept the smaller id only: archive/2026-11-02/ugi_labels_v1/);
  * overlap rule "smaller polygon wins": every overlapping pair (intersection area > 0) is listed with both ids, types,
    groups, areas, the overlap area and the winner (overlap_pairs.csv); in the rasters the polygons are burnt from the
    largest to the smallest, so where they overlap the pixel takes the smaller polygon;
  * one rasterisation on GRID_1M, pixel-centre rule (rasterio, all_touched=False);
  * outputs (uint8 unless stated), 0 = not UGI:
      ugi_id_grid1m.tif      uint16, ID_NUEVO of the winning polygon
      ugi_binary_grid1m.tif  1 = UGI
      ugi_group_grid1m.tif   1 Road greens, 2 Squares, 3 Parks and green areas, 4 Water corridors, 5 Rail verges,
                             6 NO DEFINIDO (UGI, no group) [type groups proposed in CLAUDE.md 13, pending Dr. Marinelli]
      ugi_edge_grid1m.tif    1 = within 1 px of a UGI / non-UGI boundary (a pixel whose 8-neighbourhood holds both)
  * polygons.csv: per polygon id, type, group, area (EPSG:32720) and pixels after the overlap rule.
Run from work_ms/: python scripts/u01_ugi_labels.py
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
import numpy as np                      # noqa: E402
import pyogrio                          # noqa: E402
import rasterio                         # noqa: E402
from rasterio import features           # noqa: E402
from scipy.ndimage import maximum_filter, minimum_filter  # noqa: E402
from shapely import STRtree             # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.grid import GRID_1M         # noqa: E402

SRC = WM.parent / "raw_data/22174_UrbanGreenInfras.gpkg"
OUT = WM / "derived/ugi_labels"
GROUPS = {"CANTERO": 1, "FRANJA": 1, "ROTONDA": 1, "CIRCUNVALACION": 1, "PLAZA": 2, "PLAZOLETA": 2, "PARQUE": 3,
          "AREA VERDE": 3, "CIUDAD UNIV": 3, "RIO": 4, "CANAL": 4, "CANADA": 4, "LineaFerrea": 5, "NO DEFINIDO": 6}
GROUP_NAMES = {1: "Road greens", 2: "Squares", 3: "Parks and green areas", 4: "Water corridors", 5: "Rail verges",
               6: "NO DEFINIDO"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 24):
            h.update(c)
    return h.hexdigest()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get("TEMP", ".")) / "u01_ugi" / SRC.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, tmp)
    df = pyogrio.read_dataframe(tmp).to_crs(GRID_1M.crs)
    df["group"] = df["Tipo_evp_2"].map(GROUPS)
    assert df["group"].notna().all(), set(df.loc[df["group"].isna(), "Tipo_evp_2"])
    df["area_m2"] = df.geometry.area
    df = df.sort_values("ID_NUEVO").reset_index(drop=True)
    # exact duplicates (same WKB): keep a defined type over NO DEFINIDO, then the smaller ID_NUEVO
    by_wkb = {}
    for i, g in enumerate(df.geometry):
        by_wkb.setdefault(g.wkb, []).append(i)
    dup_rows, keep = [], []
    for idx in by_wkb.values():
        best = sorted(idx, key=lambda i: (df.at[i, "Tipo_evp_2"] == "NO DEFINIDO", df.at[i, "ID_NUEVO"]))
        keep.append(best[0])
        for i in best[1:]:
            dup_rows.append({"kept_id": int(df.at[best[0], "ID_NUEVO"]), "dropped_id": int(df.at[i, "ID_NUEVO"]),
                             "kept_type": df.at[best[0], "Tipo_evp_2"], "dropped_type": df.at[i, "Tipo_evp_2"],
                             "area_m2": round(float(df.at[i, "area_m2"]), 2)})
    df = df.loc[sorted(keep)].reset_index(drop=True)
    if dup_rows:
        write_csv(OUT / "duplicates.csv", dup_rows)
    # overlapping pairs
    geoms = list(df.geometry)
    tree = STRtree(geoms)
    pairs = []
    for i, j in zip(*tree.query(geoms, predicate="intersects")):
        if i >= j:
            continue
        inter = geoms[i].intersection(geoms[j]).area
        if inter <= 0:
            continue
        a, b = df.loc[i], df.loc[j]
        win = a if a.area_m2 <= b.area_m2 else b
        pairs.append({"id_a": int(a.ID_NUEVO), "type_a": a.Tipo_evp_2, "group_a": int(a.group), "area_a_m2": round(a.area_m2, 2),
                      "id_b": int(b.ID_NUEVO), "type_b": b.Tipo_evp_2, "group_b": int(b.group), "area_b_m2": round(b.area_m2, 2),
                      "overlap_m2": round(inter, 6), "overlap_share_of_smaller": round(inter / min(a.area_m2, b.area_m2), 4),
                      "winner_id": int(win.ID_NUEVO), "groups_differ": bool(a.group != b.group)})
    pairs.sort(key=lambda r: -r["overlap_m2"])
    if pairs:
        write_csv(OUT / "overlap_pairs.csv", pairs)
    # rasterise: largest first, smaller overwrite
    order = df.sort_values("area_m2", ascending=False)
    prof = GRID_1M.profile(count=1, dtype="uint16", nodata=None, compress="deflate", tiled=True, blockxsize=512,
                           blockysize=512)
    ids = features.rasterize(((g, int(i)) for g, i in zip(order.geometry, order.ID_NUEVO)),
                             out_shape=(GRID_1M.height, GRID_1M.width), transform=GRID_1M.transform, fill=0,
                             dtype="uint16", all_touched=False)
    lut = np.zeros(int(df.ID_NUEVO.max()) + 1, np.uint8)
    lut[df.ID_NUEVO.astype(int).values] = df.group.astype(int).values
    grp = lut[ids]
    binary = (ids > 0).astype(np.uint8)
    edge = ((maximum_filter(binary, 3, mode="nearest") != minimum_filter(binary, 3, mode="nearest"))).astype(np.uint8)
    outs = {"ugi_id_grid1m.tif": (ids, "uint16", "ID_NUEVO of the UGI polygon (smaller polygon wins overlaps); 0 = not UGI"),
            "ugi_binary_grid1m.tif": (binary, "uint8", "1 = UGI (final inventory, 3,169 polygons less duplicates), 0 = not UGI"),
            "ugi_group_grid1m.tif": (grp, "uint8", "; ".join(f"{k} {v}" for k, v in GROUP_NAMES.items()) + "; 0 not UGI (proposed groups)"),
            "ugi_edge_grid1m.tif": (edge, "uint8", "1 = within 1 px of a UGI / non-UGI boundary (8-neighbourhood)")}
    sums = {}
    for name, (arr, dt, desc) in outs.items():
        with rasterio.open(OUT / name, "w", **dict(prof, dtype=dt)) as d:
            d.write(arr.astype(dt), 1)
            d.set_band_description(1, desc)
        sums[name] = sha256(OUT / name)
    px = np.bincount(ids.ravel(), minlength=int(df.ID_NUEVO.max()) + 1)
    poly = []
    for _, r in df.iterrows():
        poly.append({"ID_NUEVO": int(r.ID_NUEVO), "Tipo_evp_2": r.Tipo_evp_2, "group": int(r.group),
                     "area_m2": round(float(r.area_m2), 2), "pixels": int(px[int(r.ID_NUEVO)])})
    write_csv(OUT / "polygons.csv", poly)
    gp = np.bincount(grp.ravel(), minlength=7)
    summary = {"source": str(SRC.relative_to(WM.parent)), "source_sha256": sha256(SRC),
               "polygons_read": int(len(df) + len(dup_rows)), "duplicates_dropped": len(dup_rows),
               "polygons_used": int(len(df)), "vector_area_km2_sum": round(float(df.area_m2.sum()) / 1e6, 4),
               "overlapping_pairs": len(pairs), "overlapping_pairs_groups_differ": sum(p["groups_differ"] for p in pairs),
               "overlap_area_m2_total": round(sum(p["overlap_m2"] for p in pairs), 1),
               "overlapping_pairs_by_overlap_area": {f">= {t} m2": sum(p["overlap_m2"] >= t for p in pairs)
                                                      for t in (0.01, 1, 10)},
               "ugi_pixels": int(binary.sum()), "ugi_share_of_extent": round(float(binary.mean()), 5),
               "edge_pixels": int(edge.sum()), "edge_share_of_ugi_pixels_inside": round(float((edge & binary).sum() / binary.sum()), 4),
               "group_pixels": {GROUP_NAMES[k]: int(gp[k]) for k in GROUP_NAMES},
               "polygons_with_zero_pixels": [p["ID_NUEVO"] for p in poly if p["pixels"] == 0],
               "rule": "pixel centre; polygons burnt largest first so the smaller polygon wins overlaps",
               "outputs_sha256": sums, "seconds": round(time.time() - t0, 1),
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
