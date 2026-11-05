"""Polygon attributes for rule 7 (CLAUDE.md 13.4): one row per rasterised inventory polygon of
derived/ugi_labels/polygons.csv (3,167 after the two exact duplicates were dropped).
  * type `Tipo_evp_2` and group (polygons.csv); source `Fte_Princi` from raw_data/22174_UrbanGreenInfras.gpkg (read-only;
    null -> 'none'); core = source IV_Muni_catastrada_2022; circunvalacion = type CIRCUNVALACION;
  * width_m = twice the maximum of the inside Euclidean distance transform of the polygon's pixels on GRID_1M (scipy:
    distance from a pixel centre to the nearest outside pixel centre, so a 1 px strip has width 2 m);
  * crosses_fold_line: pixels on both sides of x_line 388282.5 or y_line 6525025.5 (pixel centres);
  * row and column bounds, pixel count on the raster (equal to polygons.csv pixels).
Output: derived/ugi_labels/polygon_attributes.csv (+ _log.json). Run from work_ms/.
"""
import csv
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage

WM = Path(__file__).resolve().parents[1]
LAB = WM / "derived/ugi_labels"
GPKG = WM.parent / "raw_data/22174_UrbanGreenInfras.gpkg"
X_LINE, Y_LINE = 388282.5, 6525025.5
CORE = "IV_Muni_catastrada_2022"


def main():
    t0 = time.time()
    polys = {int(r["ID_NUEVO"]): r for r in csv.DictReader(open(LAB / "polygons.csv", encoding="utf-8"))}
    g = gpd.read_file(GPKG, layer="22174_urbangreeninfras", ignore_geometry=True)
    src = {int(i): (s if isinstance(s, str) and s else "none") for i, s in zip(g["ID_NUEVO"], g["Fte_Princi"])}
    with rasterio.open(LAB / "ugi_id_grid1m.tif") as s:
        ids = s.read(1)
        T = s.transform
    c_split = int(np.floor((X_LINE - T.c) / T.a - 0.5)) + 1        # columns < c_split have centres <= x_line
    r_split = int(np.ceil((T.f - Y_LINE) / -T.e - 0.5))           # rows >= r_split have centres <= y_line
    assert T.c + (c_split - 0.5) * T.a <= X_LINE < T.c + (c_split + 0.5) * T.a
    assert T.f + (r_split + 0.5) * T.e <= Y_LINE < T.f + (r_split - 0.5) * T.e
    sl = ndimage.find_objects(ids)
    rows = []
    for pid, r in polys.items():
        s_ = sl[pid - 1] if pid - 1 < len(sl) else None
        if s_ is None:
            continue
        rs, cs = s_
        crop = np.pad(ids[rs, cs] == pid, 1)
        w = 2 * float(ndimage.distance_transform_edt(crop).max())
        r0, r1, c0, c1 = rs.start, rs.stop - 1, cs.start, cs.stop - 1
        rows.append({"id": pid, "Tipo_evp_2": r["Tipo_evp_2"], "group": int(r["group"]), "source": src.get(pid, "none"),
                     "core": src.get(pid) == CORE, "circunvalacion": r["Tipo_evp_2"] == "CIRCUNVALACION",
                     "area_m2": float(r["area_m2"]), "pixels": int(crop.sum()), "width_m": round(w, 3),
                     "crosses_fold_line": bool((c0 < c_split <= c1) or (r0 < r_split <= r1)),
                     "row_min": r0, "row_max": r1, "col_min": c0, "col_max": c1})
        assert rows[-1]["pixels"] == int(r["pixels"]), pid
    with open(LAB / "polygon_attributes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import collections
    log = {"polygons": len(rows), "core": sum(r["core"] for r in rows), "circunvalacion": sum(r["circunvalacion"] for r in rows),
           "crossing_fold_lines": sum(r["crosses_fold_line"] for r in rows),
           "sources": dict(collections.Counter(r["source"] for r in rows)), "split_col": c_split, "split_row": r_split,
           "width_m_quantiles_5_25_50_75_95": np.percentile([r["width_m"] for r in rows], [5, 25, 50, 75, 95]).round(2).tolist(),
           "gpkg": str(GPKG.relative_to(WM.parent)), "seconds": round(time.time() - t0, 1),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(log, open(LAB / "polygon_attributes_log.json", "w"), indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
