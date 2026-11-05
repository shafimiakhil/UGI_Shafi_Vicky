"""The centre of the city as the test (CLAUDE.md 13.7, step 1): the candidates, tabulated for Shafi and Prof. Gamba to choose. Reads files only; refuses existing outputs.

Candidates, each limited to the ejido, rasterised on GRID_1M (pixel centre inside), derived/ugi_centre/<name>_grid1m.tif, uint8, 1 inside:
  ring    the area enclosed by the ring road: the union of the 37 CIRCUNVALACION polygons is buffered until it encloses a hole, the smallest distance in DISTANCES that does; the largest hole is buffered back by the same distance and the polygons are taken out, so no CIRCUNVALACION pixel lies inside
  sq3km, sq5km, sq8km   squares centred on the fold crossing, 388,282.5 E / 6,525,025.5 N
Per candidate: area, inventory polygons (>= 50 % of their pixels inside, the scorers' rule) and area by group, UGI pixel share, tiles (cached tiles and all tiles, wholly inside and touching), pixels per fold quadrant, 1 km blocks; the Stage M2 scores inside it (rules 4, 7, 9); the seed SD of ugi_r1_none seeds 0-2, rescored inside it (itareo.ugi_seg.rescore_centre, itareo.ugi_seg.summarize_r).
The historic centre: Plaza San Martin and the Cathedral, coordinates as printed by Wikipedia (fetched 4 Nov 2026), with their offset from the fold crossing and the inventory polygon at that point.

Steps (from work_ms/, CPU):
  python scripts/centre_candidates.py --regions     derived/ugi_centre/ (rasters, regions.json, geometry log, README)
  python -m itareo.ugi_seg.rescore_centre --regions ring sq3km sq5km sq8km --stage-m2 --runs runs_itareo/ugi/ugi_r1_none --workers 6
  python scripts/centre_candidates.py --summaries   runs_itareo/ugi/summary_r/summary_centre_<name>.json
  python scripts/centre_candidates.py --table       docs/evidence_phase2/ugi_centre/ (candidates.csv, by_group.csv, candidates_map.jpg, historic_centre.jpg, README.md)
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("GDAL_PAM_ENABLED", "NO")
import numpy as np                     # noqa: E402
import rasterio                        # noqa: E402
from pyproj import Transformer         # noqa: E402
from rasterio import features          # noqa: E402
from rasterio.windows import Window    # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

from itareo.grid import GRID_1M        # noqa: E402
from itareo.safe_write import refuse_existing  # noqa: E402

GPKG = WM.parent / "raw_data/22174_UrbanGreenInfras.gpkg"
LAYER = "22174_urbangreeninfras"
DER = WM / "derived/ugi_centre"
DOC = WM.parent / "docs/evidence_phase2/ugi_centre"
LAB = WM / "derived/ugi_labels"
SUM = WM / "runs_itareo/ugi/summary_r"
M2 = WM / "runs_itareo/ugi/maps_m2"
IMAGE = WM / "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif"
XL, YL = 388282.5, 6525025.5
SQUARES = {"sq3km": 3000, "sq5km": 5000, "sq8km": 8000}
NAMES = ["ring", *SQUARES]
DISTANCES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
MIN_HOLE_KM2 = 20
GROUPS = {"1": "Road greens", "2": "Squares", "3": "Parks and green areas", "4": "Water corridors", "5": "Rail verges", "6": "NO DEFINIDO"}
HISTORIC = {"Plaza San Martin": (-31.41677, -64.18356, "https://en.wikipedia.org/wiki/Plaza_San_Mart%C3%ADn,_C%C3%B3rdoba"), "Cathedral": (-31.41667, -64.18472, "https://en.wikipedia.org/wiki/Cathedral_of_C%C3%B3rdoba,_Argentina")}
BLOCK = 1000


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(2 ** 22), b""):
            h.update(b)
    return h.hexdigest()


def inventory_ring():
    """The CIRCUNVALACION polygons, in EPSG:32720, from a copy of the delivered file in TEMP (GDAL may touch a GeoPackage it opens)."""
    import geopandas as gpd
    tmp = Path(tempfile.gettempdir()) / "centre_candidates"
    tmp.mkdir(exist_ok=True)
    copy = tmp / GPKG.name
    if not copy.exists() or sha256(copy) != sha256(GPKG):
        shutil.copy2(GPKG, copy)
    g = gpd.read_file(copy, layer=LAYER).to_crs(32720)
    ids = {int(r["id"]) for r in csv.DictReader(open(LAB / "polygon_attributes.csv", encoding="utf-8")) if r["Tipo_evp_2"] == "CIRCUNVALACION"}
    g = g[g["ID_NUEVO"].astype(int).isin(ids)]
    assert len(g) == len(ids) == 37, (len(g), len(ids))
    return g


def regions():
    """Rasters and regions.json."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    t0 = time.time()
    refuse_existing(DER)
    DER.mkdir(parents=True)
    g = inventory_ring()
    U = unary_union([x.buffer(0) for x in g.geometry])
    trials = []
    chosen = None
    for d in DISTANCES:
        B = U.buffer(d)
        holes = [Polygon(r) for p in (B.geoms if B.geom_type == "MultiPolygon" else [B]) for r in p.interiors]
        big = max(holes, key=lambda h: h.area) if holes else None
        parts = len(B.geoms) if B.geom_type == "MultiPolygon" else 1
        trials.append({"distance_m": d, "parts": parts, "largest_hole_km2": round(big.area / 1e6, 3) if big is not None else 0.0})
        print(trials[-1], flush=True)
        if chosen is None and big is not None and big.area >= MIN_HOLE_KM2 * 1e6:
            chosen = (d, big)
    if chosen is None:
        raise SystemExit(f"the ring does not close within {DISTANCES[-1]} m: {trials}")
    d, hole = chosen
    ring_area = hole.buffer(d).difference(U)
    if ring_area.geom_type == "MultiPolygon":
        ring_area = max(ring_area.geoms, key=lambda p: p.area)
    T = rasterio.Affine(GRID_1M.res, 0, GRID_1M.left, 0, -GRID_1M.res, GRID_1M.top)
    H, W = GRID_1M.height, GRID_1M.width
    with rasterio.open(LAB / "ejido_grid1m.tif") as s:
        ej = s.read(1) == 1
        crs = s.crs
    m = {"ring": features.rasterize([ring_area], out_shape=(H, W), transform=T, fill=0, default_value=1, dtype="uint8") == 1}
    xs = T.c + (np.arange(W) + 0.5) * T.a
    ys = T.f + (np.arange(H) + 0.5) * T.e
    for n, side in SQUARES.items():
        m[n] = (np.abs(ys - YL) <= side / 2)[:, None] & (np.abs(xs - XL) <= side / 2)[None, :]
    prof = dict(driver="GTiff", dtype="uint8", count=1, width=W, height=H, crs=crs, transform=T, nodata=0, compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    info = {}
    for n in NAMES:
        ej_share = float((m[n] & ej).sum() / m[n].sum())
        a = m[n] & ej
        rr, cc = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
        with rasterio.open(DER / f"{n}_grid1m.tif", "w", **prof) as dst:
            dst.write(a.astype(np.uint8), 1)
            dst.write_colormap(1, {0: (0, 0, 0, 0), 1: (43, 108, 212, 255)})
            dst.set_band_description(1, f"centre candidate {n}, inside the ejido: 1")
        info[n] = {"row0": int(rr[0]), "col0": int(cc[0]), "rows": int(rr[-1] - rr[0] + 1), "cols": int(cc[-1] - cc[0] + 1), "pixels": int(a.sum()), "share_inside_ejido": round(ej_share, 5)}
    json.dump(info, open(DER / "regions.json", "w"), indent=1)
    log = {"ring_polygons": len(g), "ring_polygons_km2": round(U.area / 1e6, 3), "closure_trials": trials, "closure_distance_m": d, "min_hole_km2": MIN_HOLE_KM2, "ring_area_km2_before_ejido": round(ring_area.area / 1e6, 3), "source_sha256": sha256(GPKG), "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(DER / "regions_log.json", "w"), indent=1)
    import geopandas as gpd
    gpd.GeoDataFrame({"name": ["ring", "ring_closed_hole"]}, geometry=[ring_area, hole], crs=32720).to_file(DER / "ring_area.gpkg", driver="GPKG")
    (DER / "README.md").write_text(
        "# Centre candidates on GRID_1M (CLAUDE.md 13.7)\n\n`python scripts/centre_candidates.py --regions` (from `work_ms/`), " + time.strftime("%Y-%m-%d %H:%M") + ".\n\n"
        f"- **Source:** `raw_data/22174_UrbanGreenInfras.gpkg` (read from a copy in TEMP), sha256 {log['source_sha256']}; `derived/ugi_labels/ejido_grid1m.tif`.\n"
        f"- **`ring_grid1m.tif`:** the union of the 37 CIRCUNVALACION polygons, buffered by {d} m, the smallest of {DISTANCES} that encloses a hole of at least {MIN_HOLE_KM2} km2 (`regions_log.json`); the largest hole, buffered back by {d} m, without the polygons, largest part, pixel centres inside, inside the ejido. Geometry: `ring_area.gpkg`.\n"
        f"- **`sq3km_grid1m.tif`, `sq5km_grid1m.tif`, `sq8km_grid1m.tif`:** squares centred on the fold crossing {XL} E, {YL} N, inside the ejido.\n- **`regions.json`:** bounding box (row0, col0, rows, cols), pixels, share of the candidate inside the ejido.\n", encoding="utf-8")
    print(json.dumps({"log": log, "regions": info}, indent=1))


def summaries():
    for n in NAMES:
        subprocess.run([sys.executable, "-W", "ignore", "-m", "itareo.ugi_seg.summarize_r", "--arms", "ugi_r1_none", "--seeds", "0", "1", "2", "--sub", f"centre_{n}", "--out", str(SUM.relative_to(WM)), "--tag", f"centre_{n}"], check=True, stdout=subprocess.DEVNULL)
        print("summary", n, flush=True)


def describe():
    """Area, polygons by group, shares, tiles, quadrants and blocks per candidate."""
    with rasterio.open(LAB / "ugi_id_grid1m.tif") as s:
        ids = s.read(1).astype(np.int32)
    with rasterio.open(LAB / "ugi_binary_grid1m.tif") as s:
        ref = s.read(1) == 1
    polys = list(csv.DictReader(open(LAB / "polygons.csv", encoding="utf-8")))
    total = {int(p["ID_NUEVO"]): int(p["pixels"]) for p in polys}
    grp = {int(p["ID_NUEVO"]): p["group"] for p in polys}
    typ = {int(p["ID_NUEVO"]): p["Tipo_evp_2"] for p in polys}
    def read_tiles(p):
        return list(csv.DictReader(open(p)))
    cached = read_tiles(WM / "runs_itareo/ugi/cache/neo2022_registered_v3/tiles.csv")
    alltiles = read_tiles(WM / "derived/ugi_tiles/tiles.csv")
    H, W = ids.shape
    xs = GRID_1M.left + (np.arange(W) + 0.5) * GRID_1M.res
    ys = GRID_1M.top - (np.arange(H) + 0.5) * GRID_1M.res
    west, south = xs <= XL, ys <= YL
    out = {}
    for n in NAMES:
        with rasterio.open(DER / f"{n}_grid1m.tif") as s:
            m = s.read(1) == 1
        c = np.bincount(ids[m].ravel())
        inside = {i: int(c[i]) for i in np.nonzero(c)[0] if i > 0}
        counted = [i for i, v in inside.items() if v >= 0.5 * total[i]]
        by = {}
        for g, name in GROUPS.items():
            by[name] = {"polygons": sum(grp[i] == g for i in counted), "km2_inside": round(sum(v for i, v in inside.items() if grp[i] == g) / 1e6, 3)}
        nc = sum(v for i, v in inside.items() if typ[i] == "CIRCUNVALACION")
        ii = np.pad(np.cumsum(np.cumsum(m.astype(np.int64), 0), 1), ((1, 0), (1, 0)))

        def tiles(rows):
            fully = touch = 0
            for t in rows:
                a, b = int(t["row0"]), int(t["col0"])
                v = ii[a + 512, b + 512] - ii[a, b + 512] - ii[a + 512, b] + ii[a, b]
                fully += int(v == 512 * 512)
                touch += int(v > 0)
            return fully, touch
        fc, tc = tiles(cached)
        fa, ta = tiles(alltiles)
        rr, cc = np.nonzero(m)
        blocks = np.unique((rr // BLOCK) * 100000 + cc // BLOCK)
        quad = {"fold 1 SW": int(m[south][:, west].sum()), "fold 2 SE": int(m[south][:, ~west].sum()), "fold 3 NW": int(m[~south][:, west].sum()), "fold 4 NE": int(m[~south][:, ~west].sum())}
        out[n] = {"area_km2": round(m.sum() / 1e6, 3), "ugi_pixels": int((ref & m).sum()), "ugi_share": round(float((ref & m).sum() / m.sum()), 4), "polygons_at_least_half_inside": len(counted), "polygons_touching": len(inside), "inventory_km2_inside": round(sum(inside.values()) / 1e6, 3), "circunvalacion_km2_inside": round(nc / 1e6, 3),
                  "by_group": by, "cached_tiles_wholly_inside": fc, "cached_tiles_touching": tc, "all_tiles_wholly_inside": fa, "all_tiles_touching": ta, "blocks_1km": int(blocks.size), "pixels_per_quadrant": quad}
    return out, ids


def historic(ids):
    to = Transformer.from_crs("EPSG:4326", "EPSG:32720", always_xy=True)
    out = {}
    for n, (lat, lon, url) in HISTORIC.items():
        x, y = to.transform(lon, lat)
        r, c = int((GRID_1M.top - y) // GRID_1M.res), int((x - GRID_1M.left) // GRID_1M.res)
        pid = int(ids[r, c])
        out[n] = {"lat": lat, "lon": lon, "source": url, "x": round(x, 1), "y": round(y, 1), "offset_from_crossing_m_east": round(x - XL, 1), "offset_from_crossing_m_north": round(y - YL, 1), "distance_from_crossing_m": round(float(np.hypot(x - XL, y - YL)), 1), "row": r, "col": c, "inventory_polygon_at_the_point": pid or None}
    return out


def fmt(v, n=4):
    return "" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def table():
    refuse_existing(DOC)
    DOC.mkdir(parents=True)
    desc, ids = describe()
    hist = historic(ids)
    rows, by_group = [], []
    for n in NAMES:
        d = desc[n]
        m, m7 = json.load(open(M2 / f"centre_{n}/metrics.json")), json.load(open(M2 / f"centre_{n}/metrics_rule7.json"))
        s = json.load(open(SUM / f"summary_centre_{n}.json"))
        a = s["arms"]["ugi_r1_none"]
        r4 = m["rule4"]["all"].get("veg", {})
        nc, al = m7["pixel"]["no_circunvalacion"], m7["pixel"]["all"]
        row = {"candidate": n, "area_km2": d["area_km2"], "ugi_share": d["ugi_share"], "ugi_pixels": d["ugi_pixels"], "polygons_at_least_half_inside": d["polygons_at_least_half_inside"], "inventory_km2_inside": d["inventory_km2_inside"], "circunvalacion_km2_inside": d["circunvalacion_km2_inside"],
               "cached_tiles_wholly_inside": d["cached_tiles_wholly_inside"], "cached_tiles_touching": d["cached_tiles_touching"], "blocks_1km": d["blocks_1km"], "share_fold1": round(d["pixels_per_quadrant"]["fold 1 SW"] / (d["area_km2"] * 1e6), 3), "share_fold2": round(d["pixels_per_quadrant"]["fold 2 SE"] / (d["area_km2"] * 1e6), 3),
               "share_fold3": round(d["pixels_per_quadrant"]["fold 3 NW"] / (d["area_km2"] * 1e6), 3), "share_fold4": round(d["pixels_per_quadrant"]["fold 4 NE"] / (d["area_km2"] * 1e6), 3),
               "m2_ap": al["ap"], "m2_ap_nocirc": nc["ap"], "m2_auroc": al["auroc"], "m2_auroc_nocirc": nc["auroc"], "m2_iou": al["iou"], "m2_iou_nocirc": nc["iou"], "m2_f1": al["f1"], "m2_f1_nocirc": nc["f1"], "m2_ap_vegetated": r4.get("ap"), "m2_auroc_vegetated": r4.get("auroc"), "ndvi_alone_ap_vegetated": r4.get("ndvi_ap"), "ndvi_alone_auroc_vegetated": r4.get("ndvi_auroc"),
               "m2_object_recall_strict": m["object_recall_strict"], "m2_object_recall_tolerant": m["object_recall_tolerant"], "m2_precision_by_area_strict": m7["object_precision"]["strict"]["by_area"], "m2_precision_by_area_tolerant": m7["object_precision"]["tolerant"]["by_area"], "m2_boundary_f": m["boundary"]["f"], "m2_block_mae_pp": m["blocks_1km"].get("mae_pp"),
               "seeds_ap_mean": a["ap"], "seeds_ap_sd": a["ap_sd_over_seeds"], "seeds_ap_nocirc_mean": a["ap_nocirc"], "seeds_ap_nocirc_sd": a["ap_nocirc_sd_over_seeds"], "ap_threshold_3sd": round(3 * a["ap_nocirc_sd_over_seeds"], 5), "seeds_ap_nocirc_ci95_block_low": a["ap_nocirc_ci95_block"][0], "seeds_ap_nocirc_ci95_block_high": a["ap_nocirc_ci95_block"][1],
               "seeds_iou_mean": a["iou"], "seeds_iou_sd": a["iou_sd_over_seeds"], "seeds_auroc_mean": a["auroc"], "seeds_auroc_sd": a["auroc_sd_over_seeds"], "seeds_recall_tolerant_mean": a["object_recall_tolerant"], "seeds_recall_tolerant_sd": a["object_recall_tolerant_sd_over_seeds"], "blocks_resampled": s["blocks_resampled"]}
        rows.append(row)
        for g, v in d["by_group"].items():
            by_group.append({"candidate": n, "group": g, **v})
    ej = json.load(open(SUM / "summary_baseline_ejido.json"))["arms"]["ugi_r1_none"]
    with open(DOC / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(DOC / "by_group.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(by_group[0]))
        w.writeheader()
        w.writerows(by_group)
    json.dump({"describe": desc, "historic_centre": hist}, open(DOC / "candidates_detail.json", "w"), indent=1)
    figures(hist, ids)
    log = json.load(open(DER / "regions_log.json"))
    L = ["# The centre of the city: candidates (CLAUDE.md 13.7, step 1)", "", "`python scripts/centre_candidates.py` (from `work_ms/`; docstring for the steps). Inside the ejido; every candidate is limited to it. Choice: Shafi and Prof. Gamba.", ""]
    L += ["| | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    def line(label, key, n=4):
        L.append(f"| {label} | " + " | ".join(fmt(r[key], n) for r in rows) + " |")
    for label, key, n in (("Area, km2", "area_km2", 2), ("UGI pixel share", "ugi_share", 4), ("Inventory polygons (>= 50 % inside)", "polygons_at_least_half_inside", 0), ("Inventory area inside, km2", "inventory_km2_inside", 2), ("CIRCUNVALACION inside, km2", "circunvalacion_km2_inside", 2), ("Cached tiles wholly inside", "cached_tiles_wholly_inside", 0), ("Cached tiles touching", "cached_tiles_touching", 0), ("1 km blocks", "blocks_1km", 0),
                          ("M2 AP", "m2_ap", 4), ("M2 AP, no C", "m2_ap_nocirc", 4), ("M2 AUROC, no C", "m2_auroc_nocirc", 4), ("M2 IoU, no C", "m2_iou_nocirc", 4), ("M2 F1, no C", "m2_f1_nocirc", 4), ("M2 AP, vegetated pixels", "m2_ap_vegetated", 4), ("NDVI alone, AP, vegetated", "ndvi_alone_ap_vegetated", 4), ("M2 AUROC, vegetated pixels", "m2_auroc_vegetated", 4), ("NDVI alone, AUROC, vegetated", "ndvi_alone_auroc_vegetated", 4),
                          ("M2 object recall, tolerant", "m2_object_recall_tolerant", 4), ("M2 object recall, strict", "m2_object_recall_strict", 4), ("M2 precision by area, tolerant", "m2_precision_by_area_tolerant", 4), ("M2 block MAE, pp", "m2_block_mae_pp", 2),
                          ("Seeds 0-2: AP no C, mean", "seeds_ap_nocirc_mean", 4), ("Seeds 0-2: AP no C, SD", "seeds_ap_nocirc_sd", 4), ("3 x SD (the AP threshold of 13.7)", "ap_threshold_3sd", 4), ("95 % block interval low, AP no C", "seeds_ap_nocirc_ci95_block_low", 4), ("95 % block interval high, AP no C", "seeds_ap_nocirc_ci95_block_high", 4), ("Seeds 0-2: IoU, mean", "seeds_iou_mean", 4), ("Seeds 0-2: IoU, SD", "seeds_iou_sd", 4), ("Seeds 0-2: tolerant recall, mean", "seeds_recall_tolerant_mean", 4), ("Seeds 0-2: tolerant recall, SD", "seeds_recall_tolerant_sd", 4)):
        line(label, key, n)
    L += ["", f"Whole ejido, for comparison (`runs_itareo/ugi/summary_r/summary_baseline_ejido.json`): AP no C {ej['ap_nocirc']:.4f}, SD {ej['ap_nocirc_sd_over_seeds']:.4f}; IoU {ej['iou']:.4f}, SD {ej['iou_sd_over_seeds']:.4f}; tolerant recall {ej['object_recall_tolerant']:.4f}, SD {ej['object_recall_tolerant_sd_over_seeds']:.4f}.", ""]
    L += ["## By group (`by_group.csv`)", "", "| group | " + " | ".join(f"{n}: polygons, km2" for n in NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    for g in GROUPS.values():
        L.append(f"| {g} | " + " | ".join(f"{desc[n]['by_group'][g]['polygons']}, {desc[n]['by_group'][g]['km2_inside']:.2f}" for n in NAMES) + " |")
    p, c = hist["Plaza San Martin"], hist["Cathedral"]
    L += ["", "## Geometry", "", f"- **The ring (`derived/ugi_centre/`).** The 37 polygons close at a buffer of {log['closure_distance_m']} m (`regions_log.json`). Inside it there is no CIRCUNVALACION polygon, so AP and AP no C coincide.",
          f"- **The historic centre lies {p['distance_from_crossing_m']:.0f} m from the fold crossing:** Plaza San Martin at {p['offset_from_crossing_m_east']:+.0f} m east, {p['offset_from_crossing_m_north']:+.0f} m north; the Cathedral at {c['offset_from_crossing_m_east']:+.0f}, {c['offset_from_crossing_m_north']:+.0f}. Coordinates as printed by Wikipedia ({p['source']}, {c['source']}), checked on `historic_centre.jpg`. Inventory polygon at the plaza's point: {p['inventory_polygon_at_the_point']}.",
          "- **All four folds meet at the crossing,** so a square on it takes a quarter from each quadrant, and its pixels were held out by four different models. Tiles straddling a fold line never trained any model and are not in the cache; `sq3km` has the largest share on them: 31 % of its pixels (13 of the 49 tiles it touches), against 19 % for `sq5km`, 12 % for `sq8km` and 9 % for `ring`; those pixels are still scored.",
          "- **Figures:** `candidates_map.jpg`, `historic_centre.jpg`.", "", "## Seeds", "", "Each of `ugi_r1_none`'s 12 runs is rescored inside each candidate (`runs_itareo/ugi/ugi_r1_none/fold*_seed*/centre_<name>/`), and pooled per seed by `itareo.ugi_seg.summarize_r` (`runs_itareo/ugi/summary_r/summary_centre_<name>.json`). M2 is the ensemble of the three seeds (`runs_itareo/ugi/maps_m2/centre_<name>/`): its AP and AUROC are above the seeds' mean in all four candidates. Its IoU and object recall come from the vote map and can fall on either side (`sq8km` IoU and `sq5km` tolerant recall are lower). In `sq3km` M2 counts 62 polygons and the seeds 61: polygon 3226 lies across folds 1 and 2 and reaches 50 % inside the square only when they are pooled."]
    (DOC / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def figures(hist, ids):
    """candidates_map.jpg: the candidates on the inventory, axes in UTM 20S km. historic_centre.jpg: Neo v3 around the plaza."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    f = 8
    H, W = ids.shape
    h8, w8 = H // f, W // f
    with rasterio.open(LAB / "ejido_grid1m.tif") as s:
        ej = s.read(1)[:h8 * f:f, :w8 * f:f] == 1
    attrs = {int(r["id"]): r["Tipo_evp_2"] for r in csv.DictReader(open(LAB / "polygon_attributes.csv", encoding="utf-8"))}
    look = np.zeros(int(ids.max()) + 1, np.uint8)
    for i, t in attrs.items():
        if i < look.size:
            look[i] = 2 if t == "CIRCUNVALACION" else 1
    cat4 = look[ids][:h8 * f, :w8 * f].reshape(h8, f, w8, f).max(axis=(1, 3))        # a cell shows the inventory if any pixel of it is inventory
    img = np.full(ej.shape + (3,), 255, np.uint8)
    img[ej] = (235, 235, 235)
    img[cat4 == 1] = (25, 169, 121)
    img[cat4 == 2] = (232, 116, 59)
    x0, x1, y0, y1 = GRID_1M.left, GRID_1M.left + W, GRID_1M.top - H, GRID_1M.top
    fig, ax = plt.subplots(figsize=(8.4, 8.2), dpi=170)
    ax.imshow(img, extent=(x0 / 1000, (x0 + w8 * f) / 1000, (y1 - h8 * f) / 1000, y1 / 1000), interpolation="nearest")
    styles = {"ring": ("-", 2.0, "Area enclosed by the ring road"), "sq3km": (":", 1.8, "Square, 3 km"), "sq5km": ("-.", 1.8, "Square, 5 km"), "sq8km": ("--", 1.8, "Square, 8 km")}
    handles = [Patch(color=(25 / 255, 169 / 255, 121 / 255), label="Inventory"), Patch(color=(232 / 255, 116 / 255, 59 / 255), label="Inventory, CIRCUNVALACION"), Patch(color=(235 / 255, 235 / 255, 235 / 255), label="Ejido")]
    for n, (ls, lw, lab) in styles.items():
        with rasterio.open(DER / f"{n}_grid1m.tif") as s:
            m = s.read(1)[::f, ::f].astype(float)
        ax.contour((GRID_1M.left + (np.arange(m.shape[1]) + 0.5) * f) / 1000, (GRID_1M.top - (np.arange(m.shape[0]) + 0.5) * f) / 1000, m, levels=[0.5], colors="#1a1a1a", linestyles=ls, linewidths=lw)
        handles.append(Line2D([0], [0], color="#1a1a1a", ls=ls, lw=lw, label=lab))
    ax.axvline(XL / 1000, color="#2b6cd4", lw=0.9, ls=(0, (6, 3)))
    ax.axhline(YL / 1000, color="#2b6cd4", lw=0.9, ls=(0, (6, 3)))
    handles.append(Line2D([0], [0], color="#2b6cd4", lw=0.9, ls=(0, (6, 3)), label="Fold lines"))
    p = hist["Plaza San Martin"]
    ax.plot(p["x"] / 1000, p["y"] / 1000, marker="*", ms=13, mfc="#ffffff", mec="#1a1a1a", ls="none")
    handles.append(Line2D([0], [0], marker="*", ms=11, mfc="#ffffff", mec="#1a1a1a", ls="none", label="Plaza San Martin"))
    ax.set_xlabel("Easting, UTM 20S (km)")
    ax.set_ylabel("Northing, UTM 20S (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8, frameon=False)
    fig.savefig(DOC / "candidates_map.jpg", bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)
    half = 300
    with rasterio.open(IMAGE) as s:
        a = s.read([1, 2, 3], window=Window(p["col"] - half, p["row"] - half, 2 * half, 2 * half)).astype(np.float32)
    out = np.zeros(a.shape[1:] + (3,), np.float32)
    for k in range(3):
        lo, hi = np.percentile(a[k], [2, 98])
        out[..., k] = np.clip((a[k] - lo) / (hi - lo), 0, 1)
    fig, ax = plt.subplots(figsize=(6, 6.3), dpi=170)
    ext = ((p["col"] - half + GRID_1M.left) / 1000, (p["col"] + half + GRID_1M.left) / 1000, (GRID_1M.top - p["row"] - half) / 1000, (GRID_1M.top - p["row"] + half) / 1000)
    ax.imshow(out, extent=ext, interpolation="nearest")
    hs = []
    for (n, v), mk in zip(hist.items(), ("*", "P")):
        ax.plot(v["x"] / 1000, v["y"] / 1000, marker=mk, ms=11, mfc="#ffffff", mec="#1a1a1a", ls="none")
        hs.append(Line2D([0], [0], marker=mk, ms=10, mfc="#ffffff", mec="#1a1a1a", ls="none", label=f"{n}, coordinates as printed by Wikipedia"))
    pid = p["inventory_polygon_at_the_point"]
    if pid:
        sub = ids[p["row"] - half:p["row"] + half, p["col"] - half:p["col"] + half] == pid
        ax.contour(np.linspace(ext[0], ext[1], sub.shape[1]), np.linspace(ext[3], ext[2], sub.shape[0]), sub.astype(float), levels=[0.5], colors="#ffffff", linewidths=1.2)
        hs.append(Line2D([0], [0], color="#7a7a7a", lw=1.2, label=f"Inventory polygon {pid}, outline in white"))
    ax.set_xlabel("Easting, UTM 20S (km)")
    ax.set_ylabel("Northing, UTM 20S (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8, frameon=False)
    fig.savefig(DOC / "historic_centre.jpg", bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


def foldc_preview(buffer_m=512):
    """What fold C would leave to train on, for the choice of the centre: docs/evidence_phase2/ugi_centre/foldc_preview.csv and a section of the README."""
    from scipy import ndimage
    out_csv = DOC / "foldc_preview.csv"
    refuse_existing(out_csv)
    tiles = list(csv.DictReader(open(WM / "runs_itareo/ugi/cache/neo2022_registered_v3/tiles.csv")))
    all_info = {(t["row0"], t["col0"]): t for t in csv.DictReader(open(WM / "derived/ugi_tiles/tiles.csv"))}
    groups = ("px_road_greens", "px_squares", "px_parks_green_areas", "px_water_corridors", "px_rail_verges", "px_no_definido")
    with rasterio.open(LAB / "ugi_binary_grid1m.tif") as s:
        ref = s.read(1) == 1
    rows = []
    for n in NAMES:
        with rasterio.open(DER / f"{n}_grid1m.tif") as s:
            m = s.read(1) == 1
        rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        a0, a1 = max(rr[0] - buffer_m - 4, 0), min(rr[-1] + buffer_m + 5, m.shape[0])
        c0, c1 = max(cc[0] - buffer_m - 4, 0), min(cc[-1] + buffer_m + 5, m.shape[1])
        b = np.zeros(m.shape, bool)
        b[a0:a1, c0:c1] = ndimage.distance_transform_edt(~m[a0:a1, c0:c1]) <= buffer_m
        ii = np.pad(np.cumsum(np.cumsum(b.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
        left, ugi = [], 0
        for t in tiles:
            r, c = int(t["row0"]), int(t["col0"])
            if ii[r + 512, c + 512] - ii[r, c + 512] - ii[r + 512, c] + ii[r, c] == 0:
                left.append(all_info[(t["row0"], t["col0"])])
                ugi += int(float(all_info[(t["row0"], t["col0"])]["ugi_px"]))
        row = {"candidate": n, "buffer_m": buffer_m, "cached_tiles": len(tiles), "tiles_left_to_train": len(left), "tiles_held_out": len(tiles) - len(left), "ugi_km2_in_tiles_left": round(ugi / 1e6, 3), "ugi_km2_scored": round((ref & m).sum() / 1e6, 3)}
        for g in groups:
            row[f"{g}_km2_left"] = round(sum(float(t[g]) for t in left) / 1e6, 3)
        rows.append(row)
        print(row, flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p = DOC / "README.md"
    s = p.read_text(encoding="utf-8")
    add = ["", f"## What fold C would leave to train on (`foldc_preview.csv`, {buffer_m} m buffer)", "", "| | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
    for label, key in (("Cached tiles left to train", "tiles_left_to_train"), ("UGI in them, km2", "ugi_km2_in_tiles_left"), ("UGI scored, km2", "ugi_km2_scored"), ("Road greens left, km2", "px_road_greens_km2_left"), ("Squares left, km2", "px_squares_km2_left"), ("Parks left, km2", "px_parks_green_areas_km2_left"), ("Water corridors left, km2", "px_water_corridors_km2_left"), ("Rail verges left, km2", "px_rail_verges_km2_left")):
        add.append(f"| {label} | " + " | ".join(str(r[key]) for r in rows) + " |")
    add += ["", "With `ring` held out, every training tile lies outside the ring road, in the peri-urban part of the city. With a square, the urban fabric around it stays in training."]
    p.write_text(s + "\n".join(add) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldc-preview", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--summaries", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.regions:
        regions()
    if a.summaries:
        summaries()
    if a.table:
        table()
    if a.foldc_preview:
        foldc_preview()
