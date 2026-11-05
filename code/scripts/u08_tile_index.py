"""U0.8 (CLAUDE.md 13): tile index for UGI segmentation on GRID_1M.

512 x 512 px tiles at stride 512 from the GRID_1M origin (as D1 L1a). Per tile:
  * UGI: share of UGI pixels (derived/ugi_labels/ugi_binary_grid1m.tif), edge-band pixels, pixels per type group
    (ugi_group_grid1m.tif), number of distinct polygons (ugi_id_grid1m.tif);
  * fold by the tile centre with Luigi's quadrant lines (x 388282.5, y 6525025.5; left & south 1, right & south 2,
    left & north 3, right & north 4) and the straddle flag (a fold line crosses the tile);
  * sensor availability, as the valid share of the tile footprint:
      neo2022     derived/pneo2022_registered/pneo2022_toa_grid1m.tif (all 6 bands valid)
      ple2017     derived/pleiades2017/pleiades2017_toa_hpf_grid1m_harmonised.tif
      ps20220424  raw_data/PRISMA/PSScene/..._8b_clip.tif (3 m, COMMON lattice; nodata 0)
      s2_2022     derived/features/s2_2022_indices_10m.tif (band 1)
      csk2025     derived/features/csk_20250728_localstats_2p5m.tif (band 1, native 2.5 m grid, read over the tile)
      meta1m      gee_exports/2026-11-02_a8/canopy_height_meta_wri_1m.tif (nodata 255)
      gba         gee_exports/2026-11-02_a8/gba_presence_height_grid1m.tif (band 1, nodata 255)
      ob2022      gee_exports/2026-11-02_a8/openbuildings_temporal_2022_3m.tif (band 1, nodata 65535)
      prisma2022  share of the tile inside the corrected footprint of derived/prisma2022/...gcps.tif (GCP polygon)
      nisar2026   not available yet (U0.7 waits for the Earthdata login): -1
Outputs: derived/ugi_tiles/tiles.csv and summary.json. Run from work_ms/.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, box

WM = Path(__file__).resolve().parents[1]
ROOT = WM.parent
sys.path.insert(0, str(WM))
from itareo.grid import GRID_1M   # noqa: E402

LAB = WM / "derived/ugi_labels"
OUT = WM / "derived/ugi_tiles"
T, XL, YL = 512, 388282.5, 6525025.5
GROUPS = {1: "road_greens", 2: "squares", 3: "parks_green_areas", 4: "water_corridors", 5: "rail_verges", 6: "no_definido"}
SENSORS = {
    "neo2022": (WM / "derived/pneo2022_registered/pneo2022_toa_grid1m.tif", None),
    "ple2017": (WM / "derived/pleiades2017/pleiades2017_toa_hpf_grid1m_harmonised.tif", None),
    "ps20220424": (ROOT / "raw_data/PRISMA/PSScene/20220424_135836_31_2446_3B_AnalyticMS_SR_8b_clip.tif", 0),
    "s2_2022": (WM / "derived/features/s2_2022_indices_10m.tif", None),
    "csk2025": (WM / "derived/features/csk_20250728_localstats_2p5m.tif", None),
    "meta1m": (ROOT / "gee_exports/2026-11-02_a8/canopy_height_meta_wri_1m.tif", None),
    "gba": (ROOT / "gee_exports/2026-11-02_a8/gba_presence_height_grid1m.tif", None),
    "ob2022": (ROOT / "gee_exports/2026-11-02_a8/openbuildings_temporal_2022_3m.tif", None),
}
PRISMA = WM / "derived/prisma2022/prisma20220427_l2d_reflectance_x10000_gcps.tif"


def valid_share(src, bounds, nodata_override, all_bands):
    w = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
    if w.width <= 0 or w.height <= 0:
        return 0.0
    bands = list(range(1, src.count + 1)) if all_bands else [1]
    a = src.read(bands, window=w, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
    nd = nodata_override if nodata_override is not None else src.nodata
    if nd is None:
        ok = np.ones(a.shape[1:], bool)
    elif isinstance(nd, float) and np.isnan(nd):
        ok = np.isfinite(a).all(0)
    else:
        ok = (a != nd).all(0)
    return round(float(ok.mean()), 4)


def prisma_footprint():
    with rasterio.open(PRISMA) as s:
        g = {(round(p.row), round(p.col)): (p.x, p.y) for p in s.gcps[0]}
        rows, cols = s.height, s.width
    edge = [g[k] for k in sorted(g) if k[0] == 0] + [g[k] for k in sorted(g) if k[1] == cols][1:] + \
           [g[k] for k in sorted(g, reverse=True) if k[0] == rows][1:] + [g[k] for k in sorted(g, reverse=True) if k[1] == 0][1:]
    return Polygon(edge).buffer(0)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pf = prisma_footprint()
    srcs = {k: rasterio.open(p) for k, (p, _) in SENSORS.items()}
    lb = {k: rasterio.open(LAB / f"ugi_{k}_grid1m.tif") for k in ("binary", "edge", "group", "id")}
    rows = []
    for r0 in range(0, GRID_1M.height - T + 1, T):
        for c0 in range(0, GRID_1M.width - T + 1, T):
            win = Window(c0, r0, T, T)
            x0, y1 = GRID_1M.left + c0, GRID_1M.top - r0
            x1, y0 = x0 + T, y1 - T
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            left, south = cx <= XL, cy <= YL
            b = lb["binary"].read(1, window=win)
            g = lb["group"].read(1, window=win)
            ids = lb["id"].read(1, window=win)
            gc = np.bincount(g.ravel(), minlength=7)
            r = {"row0": r0, "col0": c0, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                 "fold": int(1 if left and south else 2 if south else 3 if left else 4),
                 "straddle": bool(x0 < XL < x1 or y0 < YL < y1),
                 "ugi_share": round(float(b.mean()), 5), "ugi_px": int(b.sum()), "ugi_edge_px": int(lb["edge"].read(1, window=win).sum()),
                 "ugi_polygons": int(len(np.unique(ids[ids > 0])))}
            for k, n in GROUPS.items():
                r[f"px_{n}"] = int(gc[k])
            for k, s in srcs.items():
                r[f"avail_{k}"] = valid_share(s, (x0, y0, x1, y1), SENSORS[k][1], k in ("neo2022", "ple2017"))
            r["avail_prisma2022"] = round(pf.intersection(box(x0, y0, x1, y1)).area / (T * T), 4)
            r["avail_nisar2026"] = -1
            rows.append(r)
    for s in list(srcs.values()) + list(lb.values()):
        s.close()
    with open(OUT / "tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keep = [r for r in rows if not r["straddle"]]
    summ = {"tiles": len(rows), "straddling": len(rows) - len(keep),
            "tiles_with_ugi": sum(r["ugi_share"] > 0 for r in keep),
            "tiles_with_ugi_by_fold": {f: sum(r["ugi_share"] > 0 and r["fold"] == f for r in keep) for f in (1, 2, 3, 4)},
            "ugi_px_by_fold": {f: int(sum(r["ugi_px"] for r in keep if r["fold"] == f)) for f in (1, 2, 3, 4)},
            "group_px_by_fold": {f: {n: sum(r[f"px_{n}"] for r in keep if r["fold"] == f) for n in GROUPS.values()}
                                 for f in (1, 2, 3, 4)},
            "tiles_full_availability_share_ge_0.9": {k: sum(r[f"avail_{k}"] >= 0.9 for r in keep)
                                                     for k in list(SENSORS) + ["prisma2022"]},
            "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
