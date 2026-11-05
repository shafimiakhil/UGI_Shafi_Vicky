"""E3 (CLAUDE.md 13.7): what is left of the Sentinel-2 displacement against Neo v3 after the one constant that E3 removes, -1.42 E / +3.98 N m (docs/evidence_phase2/ugi_surface/surface_s2_displacement.json), per 1 km block, as scripts/ps_residuals.py does for PlanetScope. Measurement only.

Each quarterly median s2_sr_median_2022Q<k> on its 10 m lattice against Neo v3 averaged on the same 10 m cells (a cell of the lattice nested in GRID_1M covers rows and columns with the same (i + 7) // 10), blocks of 1 km (100 cells) on a 1 km stride, phase correlation with the criteria of scripts/a7_coregistration.correlate, low-pass estimate beside it.
Per quarter and per fold quadrant: the median and spread, the residual after the constant, the adjacent-block difference (if the block values were noise, its robust SD / sqrt 2 would equal the spread), a plane per axis and the residual after it.
Outputs in docs/evidence_phase2/ugi_seg/e3_s2_residuals/ (refuses an existing folder): summary.json, blocks_<quarter>.csv. Run from work_ms/ (CPU): python scripts/e3_s2_residuals.py
--from-blocks writes summary_v2.json from the block files, measuring nothing again. In summary.json the plane of Q2 is spoiled by two gross low-pass estimates (the usage gate looks at the unweighted quality only), and comparing the plane with the constant mixes each quarter's own intercept with a slope. Version 2 fits the plane without blocks more than 100 m (10 px) from the median, says how many, and gives the residual after the quarter's own median beside it. The spread and the adjacent-block difference are multiples of the 0.5 m estimator step, so they are no noise test here.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import a7_coregistration as A7  # noqa: E402

NEO = WM / "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif"
S2 = WM.parent / "gee_exports/2026-10-28_nodata/s2_sr_median_{}.tif"
OUT = WM.parent / "docs/evidence_phase2/ugi_seg/e3_s2_residuals"
XL, YL = 388282.5, 6525025.5
CONST = (-1.42, 3.98)
BS = 100          # cells of 10 m
ORIGIN = (375320.0, 6535840.0)


def robust_sd(v):
    return float(np.subtract(*np.percentile(v, [75, 25])) / 1.349)


def measure(q):
    rows = []
    with rasterio.open(NEO) as neo, rasterio.open(str(S2).format(q)) as src:
        assert (src.transform.c, src.transform.f) == ORIGIN and src.transform.a == 10
        names = list(src.descriptions)
        bands = [names.index(b) + 1 for b in ("B2", "B3", "B4", "B8")]
        for r0 in range(1, src.height - BS, BS):             # row 1 of the lattice starts at row 3 of GRID_1M; Neo needs whole cells
            for c0 in range(1, src.width - BS, BS):
                a = src.read(bands, window=Window(c0, r0, BS, BS)).astype(np.float64)
                a[:, (a == src.nodata).any(0)] = np.nan
                if np.isfinite(a[0]).mean() < 0.9:
                    continue
                g_r, g_c = r0 * 10 - 7, c0 * 10 - 7
                if g_r < 0 or g_c < 0 or g_r + BS * 10 > neo.height or g_c + BS * 10 > neo.width:
                    continue
                n = neo.read([1, 2, 3, 4], window=Window(g_c, g_r, BS * 10, BS * 10)).astype(np.float64)
                n[:, (n == neo.nodata).any(0)] = np.nan
                n = n.reshape(4, BS, 10, BS, 10).mean(axis=(2, 4))
                if np.isfinite(n[0]).mean() < 0.9:
                    continue
                xc, yc = ORIGIN[0] + 10 * (c0 + BS / 2), ORIGIN[1] - 10 * (r0 + BS / 2)
                fold = (1 if xc <= XL else 2) if yc <= YL else (3 if xc <= XL else 4)
                rows.append(A7.correlate({"r0": r0, "c0": c0, "x": xc, "y": yc, "fold": fold}, *A7.vis_ndvi(n * 1e-4, [0, 1, 2], 0, 3), np.zeros((BS, BS), bool), *A7.vis_ndvi(a * 1e-4, [0, 1, 2], 2, 3), np.zeros((BS, BS), bool), 10.0, lowpass=0.1))
    return rows


def summary(rows):
    u = [r for r in rows if r.get("used")]
    dx, dy = np.array([r["dx_m"] for r in u]), np.array([r["dy_m"] for r in u])
    lx, ly = np.array([r["dx_lp_m"] for r in u]), np.array([r["dy_lp_m"] for r in u])
    fd, x, y = np.array([r["fold"] for r in u]), np.array([r["x"] for r in u]), np.array([r["y"] for r in u])
    res, resl = np.hypot(dx - CONST[0], dy - CONST[1]), np.hypot(lx - CONST[0], ly - CONST[1])
    idx = {(r["r0"], r["c0"]): i for i, r in enumerate(u)}
    de, dn = [], []
    for (r0, c0), i in idx.items():
        for k in ((r0 + BS, c0), (r0, c0 + BS)):
            j = idx.get(k)
            if j is not None:
                de.append(lx[i] - lx[j])
                dn.append(ly[i] - ly[j])
    ok = np.hypot(lx - np.median(lx), ly - np.median(ly)) <= 100
    A = np.c_[np.ones(len(u)), (x - XL) / 1000, (y - YL) / 1000]
    pe, pn = np.linalg.lstsq(A[ok], lx[ok], rcond=None)[0], np.linalg.lstsq(A[ok], ly[ok], rcond=None)[0]
    rp = np.hypot(lx - A @ pe, ly - A @ pn)[ok]
    rm = np.hypot(lx - np.median(lx), ly - np.median(ly))[ok]
    out = {"blocks_measured": len(rows), "blocks_used": len(u), "unweighted_median_m_east_north": [round(float(np.median(dx)), 3), round(float(np.median(dy)), 3)], "lowpass_median_m_east_north": [round(float(np.median(lx)), 3), round(float(np.median(ly)), 3)],
           "residual_after_constant_per_block_m_unweighted": {"median": round(float(np.median(res)), 2), "p90": round(float(np.percentile(res, 90)), 2)}, "residual_after_constant_per_block_m_lowpass": {"median": round(float(np.median(resl)), 2), "p90": round(float(np.percentile(resl, 90)), 2), "share_over_4m": round(float((resl > 4).mean()), 3), "share_over_8m": round(float((resl > 8).mean()), 3)},
           "spread_over_blocks_robust_sd_m_east_north_lowpass": [round(robust_sd(lx), 2), round(robust_sd(ly), 2)], "adjacent_block_difference_robust_sd_over_sqrt2_m_east_north": [round(robust_sd(np.array(de)) / np.sqrt(2), 2), round(robust_sd(np.array(dn)) / np.sqrt(2), 2)],
           "plane_lowpass_m_and_m_per_km_east_and_north": {"east": np.round(pe, 3).tolist(), "north": np.round(pn, 3).tolist(), "reference_point_utm": [XL, YL]}, "residual_after_plane_per_block_m": {"median": round(float(np.median(rp)), 2), "p90": round(float(np.percentile(rp, 90)), 2)}, "blocks_left_out_of_the_plane_over_100m_from_the_median": int((~ok).sum()), "residual_after_the_quarters_own_median_per_block_m": {"median": round(float(np.median(rm)), 2), "p90": round(float(np.percentile(rm, 90)), 2)}, "per_fold_quadrant": {}}
    for f in (1, 2, 3, 4):
        m = fd == f
        if m.any():
            out["per_fold_quadrant"][str(f)] = {"blocks": int(m.sum()), "lowpass_median_m_east_north": [round(float(np.median(lx[m])), 2), round(float(np.median(ly[m])), 2)], "residual_after_constant_m_east_north": [round(float(np.median(lx[m]) - CONST[0]), 2), round(float(np.median(ly[m]) - CONST[1]), 2)]}
    return out


def from_blocks():
    from itareo.safe_write import refuse_existing
    refuse_existing(OUT / "summary_v2.json")
    res = {"what": "summary.json recomputed from the block files; see the docstring", "constant_removed_m_east_north": list(CONST), "quarters": {}}
    for q in ("2022Q1", "2022Q2", "2022Q3", "2022Q4"):
        rows = []
        for r in csv.DictReader(open(OUT / f"blocks_{q}.csv")):
            d = {"used": r["used"] == "True", "r0": int(r["r0"]), "c0": int(r["c0"]), "x": float(r["x"]), "y": float(r["y"]), "fold": int(r["fold"])}
            if d["used"]:
                d.update({k: float(r[k]) for k in ("dx_m", "dy_m", "dx_lp_m", "dy_lp_m")})
            rows.append(d)
        res["quarters"][q] = summary(rows)
        print(q, json.dumps(res["quarters"][q]), flush=True)
    json.dump(res, open(OUT / "summary_v2.json", "w"), indent=1)


def main():
    if "--from-blocks" in sys.argv:
        from_blocks()
        return
    OUT.mkdir(parents=True, exist_ok=False)
    res = {"what": "displacement of each Sentinel-2 2022 quarter relative to Neo v3 (m east, m north), per 1 km block; E3 removes one constant", "constant_removed_m_east_north": list(CONST), "quarters": {}}
    for q in ("2022Q1", "2022Q2", "2022Q3", "2022Q4"):
        rows = measure(q)
        with open(OUT / f"blocks_{q}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        res["quarters"][q] = summary(rows)
        print(q, json.dumps(res["quarters"][q]), flush=True)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
