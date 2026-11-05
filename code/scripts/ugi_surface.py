"""Inventory surface composition (CLAUDE.md 13.5, night CPU task 2). Descriptive only: no label changes, no model.

Every inventory pixel (derived/ugi_labels/ugi_id_grid1m.tif > 0) gets one class from the 2022 observations available at that
pixel, tested in this order:
  1 water             Neo v3 NDWI > 0.1 and NDVI < 0.1
  2 never vegetated   Neo v3 NDVI < 0.2, NDVI < 0.2 on every available PlanetScope 2022 date, and the maximum of the available
                      Sentinel-2 2022 quarterly NDVI < 0.3
  3 always vegetated  NDVI >= 0.3 in Neo, on every available PlanetScope date and in every available Sentinel-2 quarter
  4 seasonal          the rest
Observations, read at the pixel without interpolation:
  * Neo v3: derived/features/neo2022_v3_indices_grid1m.tif, NDVI and NDWI x 10000, on GRID_1M;
  * PlanetScope 24 Apr, 10 Sep, 28 Oct 2022: derived/planetscope2022/ps<date>_sr_norm_s2_3m.tif, NDVI from bands 6 (red) and
    8 (NIR), a band value of 0 = not available. The 3 m pixel that holds the 1 m pixel's centre after the date's measured
    displacement (runs_itareo/ugi/b4_ps_offsets.json; content at (x, y) in Neo geometry lies at (x + dx, y + dy) in the file,
    as in itareo.ugi_seg.ps_stage). Thresholds evaluated exactly on the integer band values;
  * Sentinel-2: derived/features/s2_2022_indices_10m.tif, bands 1, 4, 7, 10 (NDVI 2022 Q1-Q4 x 10000), the 10 m lattice cell
    that holds the 1 m pixel (GRID_1M nests in it with an origin offset of 7 m).
A pixel without any PlanetScope date or any Sentinel-2 quarter is classified from the rest; counts in surface_log.json.
Outputs in docs/evidence_phase2/ugi_surface/ (the script refuses an existing folder): surface_class_grid1m.tif (uint8, nodata 0
outside the inventory, embedded colour table, internal overviews), surface_by_polygon.csv, surface_by_type.csv, surface_log.json,
README.md. A random sample of 1,000 inventory pixels (seed 20261103) is recomputed by a separate scalar path from the
stored rasters and compared with the stored class raster; exit 1 on any difference.
Run from work_ms/: python scripts/ugi_surface.py      (--readme-only: README.md again from the stored tables, after
scripts/ugi_surface_checks.py; refuses an existing README)
"""
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
import numpy as np                      # noqa: E402
import rasterio                         # noqa: E402
from rasterio.enums import Resampling   # noqa: E402
from rasterio.windows import Window     # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.grid import GRID_1M         # noqa: E402

OUT = WM.parent / "docs/evidence_phase2/ugi_surface"
IDS = WM / "derived/ugi_labels/ugi_id_grid1m.tif"
ATTRS = WM / "derived/ugi_labels/polygon_attributes.csv"
NEO = WM / "derived/features/neo2022_v3_indices_grid1m.tif"
S2 = WM / "derived/features/s2_2022_indices_10m.tif"
PS = {d: WM / f"derived/planetscope2022/ps{d}_sr_norm_s2_3m.tif" for d in ("20220424", "20220910", "20221028")}
OFFSETS = WM / "runs_itareo/ugi/b4_ps_offsets.json"
NEO_NDVI, NEO_NDWI = 1, 3
PS_RED, PS_NIR = 6, 8
S2_NDVI = [1, 4, 7, 10]
CLASSES = {1: "water", 2: "never_vegetated", 3: "always_vegetated", 4: "seasonal"}
COLOURS = {1: (0x2a, 0x78, 0xd6, 255), 2: (0xeb, 0x68, 0x34, 255), 3: (0x1b, 0xaf, 0x7a, 255), 4: (0xed, 0xa1, 0x00, 255)}
GROUPS = {1: "Road greens", 2: "Squares", 3: "Parks and green areas", 4: "Water corridors", 5: "Rail verges", 6: "NO DEFINIDO"}
CHUNK = 2048                            # rows, a multiple of the 512 px tiles
NID = 4000
SAMPLE, SEED = 1000, 20261103


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(2 ** 22), b""):
            h.update(b)
    return h.hexdigest()


def ps_flags(path):
    """(available, NDVI < 0.2, NDVI >= 0.3) of one date at 3 m, exact on the integer band values."""
    with rasterio.open(path) as s:
        if s.transform != GRID_1M.transform * rasterio.Affine.scale(3) or s.nodata != 0:
            raise ValueError(f"{path}: not on the 3 m grid nested in GRID_1M, or nodata is not 0")
        red, nir = s.read(PS_RED).astype(np.int32), s.read(PS_NIR).astype(np.int32)
    avail = (red > 0) & (nir > 0)
    return avail, 10 * (nir - red) < 2 * (nir + red), 10 * (nir - red) >= 3 * (nir + red)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=False)
    off = json.load(open(OFFSETS))["offsets_relative_to_registered_neo2022"]
    flags = {}
    for d, p in PS.items():
        flags[d] = ps_flags(p)
        print("read", p.name, round(time.time() - t0), "s", flush=True)
    with rasterio.open(S2) as s:
        if (s.transform.c, s.transform.f, s.transform.a) != (375320.0, 6535840.0, 10.0):
            raise ValueError("Sentinel-2 raster is not on the expected lattice")
        q = s.read(S2_NDVI)
        s2_nd = s.nodata
        s2_h, s2_w = s.height, s.width
    q_ok = q != s2_nd
    q_max = np.where(q_ok, q, -32768).max(0)
    q_min = np.where(q_ok, q, 32767).min(0)
    q_n = q_ok.sum(0)
    del q, q_ok
    counts = np.zeros((NID, 5), np.int64)
    ps_sum = np.zeros(NID, np.int64)
    n_dates = np.zeros(4, np.int64)
    per_date = {d: 0 for d in PS}
    n_quarters = np.zeros(5, np.int64)
    neo_missing = 0
    rng = np.random.default_rng(SEED)
    sample = []
    p_keep = SAMPLE / 32_610_831
    prof = dict(driver="GTiff", dtype="uint8", count=1, width=GRID_1M.width, height=GRID_1M.height, crs=GRID_1M.crs,
                transform=GRID_1M.transform, nodata=0, compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                BIGTIFF="IF_SAFER")
    H, W = GRID_1M.height, GRID_1M.width
    ps_h, ps_w = flags["20220424"][0].shape
    with rasterio.open(IDS) as sid, rasterio.open(NEO) as sn, rasterio.open(OUT / "surface_class_grid1m.tif", "w", **prof) as dst:
        if sid.transform != GRID_1M.transform or sn.transform != GRID_1M.transform:
            raise ValueError("inputs are not on GRID_1M")
        nd = sn.nodata
        for a in range(0, H, CHUNK):
            win = Window(0, a, W, min(CHUNK, H - a))
            ids = sid.read(1, window=win)
            vi = ids > 0
            out = np.zeros(ids.shape, np.uint8)
            if vi.any():
                rr, cc = np.nonzero(vi)
                rows = rr + a
                ii = ids[vi].astype(np.int64)
                ndvi = sn.read(NEO_NDVI, window=win)[vi]
                ndwi = sn.read(NEO_NDWI, window=win)[vi]
                neo_ok = (ndvi != nd) & (ndwi != nd)
                neo_missing += int((~neo_ok).sum())
                all_lt = np.ones(len(ii), bool)
                all_ge = np.ones(len(ii), bool)
                nd_avail = np.zeros(len(ii), np.int64)
                for d, (av, lt, ge) in flags.items():
                    dx, dy = off[d]["dx_east_m"], off[d]["dy_north_m"]
                    pr = np.floor((rows + 0.5 - dy) / 3).astype(np.int64)
                    pc = np.floor((cc + 0.5 + dx) / 3).astype(np.int64)
                    ins = (pr >= 0) & (pr < ps_h) & (pc >= 0) & (pc < ps_w)
                    have = np.zeros(len(ii), bool)
                    have[ins] = av[pr[ins], pc[ins]]
                    is_lt, is_ge = np.ones(len(ii), bool), np.ones(len(ii), bool)      # true where not available
                    is_lt[have] = lt[pr[have], pc[have]]
                    is_ge[have] = ge[pr[have], pc[have]]
                    all_lt &= is_lt
                    all_ge &= is_ge
                    nd_avail += have
                    per_date[d] += int(have.sum())
                sr, sc = (rows + 7) // 10, (cc + 7) // 10
                if sr.max() >= s2_h or sc.max() >= s2_w:
                    raise ValueError("an inventory pixel lies outside the Sentinel-2 lattice raster")
                never_s2, always_s2 = q_max[sr, sc] < 3000, q_min[sr, sc] >= 3000
                water = neo_ok & (ndwi > 1000) & (ndvi < 1000)
                never = neo_ok & (ndvi < 2000) & all_lt & never_s2
                always = neo_ok & (ndvi >= 3000) & all_ge & always_s2
                cls = np.full(len(ii), 4, np.uint8)
                cls[always] = 3
                cls[never] = 2
                cls[water] = 1
                out[vi] = cls
                np.add.at(counts, (ii, cls), 1)
                ps_sum += np.bincount(ii, weights=nd_avail, minlength=NID).astype(np.int64)
                n_dates += np.bincount(nd_avail, minlength=4)
                n_quarters += np.bincount(q_n[sr, sc], minlength=5)
                keep = rng.random(len(ii)) < p_keep
                sample += [(int(r), int(c), int(k)) for r, c, k in zip(rows[keep], cc[keep], cls[keep])]
            dst.write(out, 1, window=win)
            print("rows", a + win.height, "of", H, round(time.time() - t0), "s", flush=True)
        dst.write_colormap(1, {0: (0, 0, 0, 0), **COLOURS})
        dst.set_band_description(1, "1 water, 2 never vegetated, 3 always vegetated, 4 seasonal; 0 = outside the inventory")
        dst.update_tags(**{f"CLASS_{k}": v for k, v in CLASSES.items()})
        dst.build_overviews([2, 4, 8, 16], Resampling.mode)
    # ---- tables ------------------------------------------------------------------------------------------------------
    attrs = {int(r["id"]): r for r in csv.DictReader(open(ATTRS, encoding="utf-8"))}
    poly_rows, type_acc = [], {}
    for pid in np.nonzero(counts.sum(1))[0]:
        pid = int(pid)
        c, n = counts[pid], int(counts[pid].sum())
        a_ = attrs[pid]
        if int(a_["pixels"]) != n:
            raise ValueError(f"polygon {pid}: {n} classified pixels against {a_['pixels']} in polygon_attributes.csv")
        poly_rows.append({"id": pid, "Tipo_evp_2": a_["Tipo_evp_2"], "group": GROUPS[int(a_["group"])], "pixels": n,
                          **{f"share_{v}": round(int(c[k]) / n, 5) for k, v in CLASSES.items()},
                          "planetscope_dates_mean": round(int(ps_sum[pid]) / n, 3)})
        t = type_acc.setdefault(a_["Tipo_evp_2"], {"polygons": 0, "pixels": np.zeros(5, np.int64)})
        t["polygons"] += 1
        t["pixels"] += c
    type_rows = []
    for name in sorted(type_acc, key=lambda k: -type_acc[k]["pixels"].sum()) + ["ALL"]:
        c = counts.sum(0) if name == "ALL" else type_acc[name]["pixels"]
        n = int(c.sum())
        type_rows.append({"Tipo_evp_2": name, "polygons": len(poly_rows) if name == "ALL" else type_acc[name]["polygons"], "area_km2": round(n / 1e6, 4),
                          **{f"share_{v}": round(int(c[k]) / n, 5) for k, v in CLASSES.items()}})
    for name, rows in (("surface_by_polygon.csv", poly_rows), ("surface_by_type.csv", type_rows)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # ---- independent check on a random sample ---------------------------------------------------------------------------
    sample = sample[:SAMPLE]
    wrong = check_sample(sample, off)
    n_all = int(counts.sum())
    log = {"inputs_sha256": {p.name: sha256(p) for p in [IDS, NEO, S2, ATTRS, OFFSETS, *PS.values()]},
           "inventory_pixels": n_all, "class_pixels": {v: int(counts.sum(0)[k]) for k, v in CLASSES.items()},
           "pixels_without_neo_index": neo_missing, "planetscope_share_of_inventory_pixels_by_date": {d: round(v / n_all, 5) for d, v in per_date.items()},
           "planetscope_dates_available_pixels": {str(k): int(n_dates[k]) for k in range(4)},
           "sentinel2_quarters_available_pixels": {str(k): int(n_quarters[k]) for k in range(5)},
           "planetscope_offsets_applied": {d: [off[d]["dx_east_m"], off[d]["dy_north_m"]] for d in PS},
           "sample_check": {"seed": SEED, "pixels": len(sample), "different": len(wrong), "examples": wrong[:10]},
           "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(OUT / "surface_log.json", "w"), indent=1)
    readme(log, type_rows, n_all)
    print(json.dumps({k: log[k] for k in ("inventory_pixels", "class_pixels", "pixels_without_neo_index", "planetscope_share_of_inventory_pixels_by_date",
                                          "planetscope_dates_available_pixels", "sentinel2_quarters_available_pixels", "sample_check", "seconds")}, indent=1))
    print(json.dumps(type_rows, indent=0))
    raise SystemExit(0 if not wrong and neo_missing == 0 else 1)


def check_sample(sample, off):
    """The class of each sampled pixel recomputed one by one from coordinates, with float NDVI, against the stored class."""
    wrong = []
    with rasterio.open(OUT / "surface_class_grid1m.tif") as c, rasterio.open(NEO) as n, rasterio.open(S2) as s:
        ps = {d: rasterio.open(p) for d, p in PS.items()}

        def one(ds, band, row, col):
            return ds.read(band, window=Window(col, row, 1, 1))[0, 0]
        for r, k, stored in sample:
            x, y = rasterio.transform.xy(c.transform, r, k)                    # pixel centre
            ndvi, ndwi = one(n, NEO_NDVI, r, k) / 1e4, one(n, NEO_NDWI, r, k) / 1e4
            ps_v = []
            for d, ds in ps.items():
                i, j = ds.index(x + off[d]["dx_east_m"], y + off[d]["dy_north_m"], op=math.floor)
                if 0 <= i < ds.height and 0 <= j < ds.width:
                    red, nir = float(one(ds, PS_RED, i, j)), float(one(ds, PS_NIR, i, j))
                    if red > 0 and nir > 0:
                        ps_v.append((nir - red) / (nir + red))
            i, j = s.index(x, y, op=math.floor)
            qv = [float(one(s, b, i, j)) / 1e4 for b in S2_NDVI if one(s, b, i, j) != s.nodata]
            if ndwi > 0.1 and ndvi < 0.1:
                want = 1
            elif ndvi < 0.2 and all(v < 0.2 for v in ps_v) and (not qv or max(qv) < 0.3):
                want = 2
            elif ndvi >= 0.3 and all(v >= 0.3 for v in ps_v) and all(v >= 0.3 for v in qv):
                want = 3
            else:
                want = 4
            got = int(one(c, 1, r, k))
            if got != want or got != stored:
                wrong.append({"row": r, "col": k, "stored": got, "in_memory": stored, "recomputed": want})
        for ds in ps.values():
            ds.close()
    return wrong


def readme(log, type_rows, n_all):
    with open(OUT / "README.md", "w", encoding="utf-8") as f:
        f.write("# Inventory surface composition (CLAUDE.md 13.5)\n\n")
        f.write("Descriptive only: no label changes, no model. `python scripts/ugi_surface.py` (from `work_ms/`).\n\n")
        f.write("Every inventory pixel gets one class from the 2022 observations available at that pixel, tested in this order:\n\n")
        f.write("| code | class | rule |\n|---|---|---|\n")
        f.write("| 1 | water | Neo v3 NDWI > 0.1 and NDVI < 0.1 |\n")
        f.write("| 2 | never vegetated | Neo v3 NDVI < 0.2, NDVI < 0.2 on every available PlanetScope 2022 date, and maximum Sentinel-2 2022 quarterly NDVI < 0.3 |\n")
        f.write("| 3 | always vegetated | NDVI >= 0.3 in Neo, on every available PlanetScope date and in every available Sentinel-2 quarter |\n")
        f.write("| 4 | seasonal | the rest |\n\n")
        f.write("**Buildings cannot be separated without a 2022 building layer.** A roof, a road or bare soil reaches never vegetated only where the 3 m and 10 m pixels "
                "that hold it also show no vegetation; otherwise it falls in seasonal.")
        geo_path, an_path, s2_path = OUT / "surface_vs_geobia2017.csv", OUT / "surface_seasonal_anatomy.json", OUT / "surface_s2_displacement.json"
        if geo_path.exists():
            geo = {r["geobia2017_class"]: r for r in csv.DictReader(open(geo_path, encoding="utf-8")) if r["pixels_with"] == "all inventory pixels"}
            f.write(" Against the GEOBIA 2017 map (Rodriguez Gonzalez et al. 2023; a classification of 2017, a cross-check only; `surface_vs_geobia2017.csv`), shares never vegetated / seasonal: " +
                    ", ".join(f"{g} {100 * float(geo[g]['share_never_vegetated']):.0f} / {100 * float(geo[g]['share_seasonal']):.0f} %" for g in ("Buildings", "Asphalt", "Bare soil")) + ".")
        f.write("\n\n")
        f.write("**Seasonal is the remainder of the rule, not a phenology class.**")
        if an_path.exists():
            an = json.load(open(an_path))
            f.write(f" {100 * an['class4_share_cross_one_sensor']:.0f} % of its pixels hold an NDVI below 0.2 and one at or above 0.3 within one sensor (`surface_seasonal_anatomy.json`).")
        f.write("\n\n")
        if s2_path.exists():
            s2d = json.load(open(s2_path))
            e = s2d["effect_of_reading_s2_displaced"]
            a_, b_ = e["shares_nominal"], e["shares_displaced"]
            f.write(f"**Sentinel-2 is read at its lattice cell, without a displacement,** while PlanetScope is read after its measured one; CLAUDE.md 13.5 asks for neither. Measured against Neo v3, "
                    f"over the four quarters: {s2d['dx_east_m_all']:+.2f} m east, {s2d['dy_north_m_all']:+.2f} m north. Read at that position, {100 * e['share_changing_class']:.2f} % of the inventory pixels would change class "
                    f"(always vegetated {100 * a_['always_vegetated']:.2f} to {100 * b_['always_vegetated']:.2f} %, never vegetated {100 * a_['never_vegetated']:.2f} to {100 * b_['never_vegetated']:.2f} %, "
                    f"seasonal {100 * a_['seasonal']:.2f} to {100 * b_['seasonal']:.2f} %; `surface_s2_displacement.json`). Not applied.\n\n")
        f.write("## Inputs\n\n")
        f.write("- Neo v3: `work_ms/derived/features/neo2022_v3_indices_grid1m.tif` (NDVI, NDWI).\n")
        f.write("- PlanetScope 24 Apr, 10 Sep, 28 Oct 2022: `work_ms/derived/planetscope2022/ps<date>_sr_norm_s2_3m.tif`, NDVI from bands 6 and 8; the 3 m pixel "
                "holding the 1 m pixel's centre after the measured displacement of the date (`work_ms/runs_itareo/ugi/b4_ps_offsets.json`). No interpolation.\n")
        f.write("- Sentinel-2 2022 quarters: `work_ms/derived/features/s2_2022_indices_10m.tif`, bands 1, 4, 7, 10; the 10 m lattice cell holding the 1 m pixel. No interpolation.\n\n")
        f.write(f"## PlanetScope coverage of the {n_all:,} inventory pixels\n\n")
        f.write("| date | share of inventory pixels available |\n|---|---|\n")
        for d, v in log["planetscope_share_of_inventory_pixels_by_date"].items():
            f.write(f"| {d} | {v:.3f} |\n")
        f.write("\n| dates available | pixels | share |\n|---|---|---|\n")
        for k, v in log["planetscope_dates_available_pixels"].items():
            f.write(f"| {k} | {v:,} | {v / n_all:.4f} |\n")
        f.write("\nWhere fewer dates are available the rules test fewer observations, so \"always\" and \"never\" are weaker there. "
                "Pixels with 0 to 4 Sentinel-2 quarters are counted in `surface_log.json`.\n\n")
        f.write("## Area shares by type (`surface_by_type.csv`)\n\n")
        f.write("| type | polygons | km2 | water | never vegetated | always vegetated | seasonal |\n|---|---|---|---|---|---|---|\n")
        for r in type_rows:
            f.write(f"| {r['Tipo_evp_2']} | {r['polygons']} | {r['area_km2']:.3f} | {r['share_water']:.3f} | {r['share_never_vegetated']:.3f} | "
                    f"{r['share_always_vegetated']:.3f} | {r['share_seasonal']:.3f} |\n")
        f.write("\n## Files\n\n")
        f.write("- `surface_class_grid1m.tif`: uint8 on GRID_1M, codes above, nodata 0 outside the inventory, embedded colour table, internal overviews (mode).\n")
        f.write("- `surface_by_polygon.csv`: per polygon, type, group (as proposed in CLAUDE.md 13), pixels, area shares, mean number of PlanetScope dates.\n")
        f.write("- `surface_by_type.csv`: per type and overall.\n")
        s = log["sample_check"]
        f.write(f"- `surface_log.json`: input hashes, counts; a random sample of {s['pixels']} inventory pixels (seed {s['seed']}) recomputed one by one from "
                f"the stored rasters differs in {s['different']}.\n")
        if s2_path.exists():
            c = json.load(open(s2_path))["stored_raster_recomputed"]
            f.write(f"- `surface_vs_geobia2017.csv`, `surface_seasonal_anatomy.json`, `surface_s2_displacement.json`: `python scripts/ugi_surface_checks.py`, read-only on the classes; "
                    f"its second pass over all {c['pixels']:,} inventory pixels differs from the stored raster in {c['different']}.\n")


def readme_only():
    """README.md again from the stored tables, after scripts/ugi_surface_checks.py; refuses an existing README."""
    if (OUT / "README.md").exists():
        raise FileExistsError(f"refusing to overwrite: {OUT / 'README.md'}")
    log = json.load(open(OUT / "surface_log.json"))
    rows = list(csv.DictReader(open(OUT / "surface_by_type.csv", encoding="utf-8")))
    for r in rows:
        r.update({k: float(v) for k, v in r.items() if k.startswith("share_") or k == "area_km2"})
    readme(log, rows, log["inventory_pixels"])


if __name__ == "__main__":
    if "--readme-only" in sys.argv:
        readme_only()
    else:
        main()
