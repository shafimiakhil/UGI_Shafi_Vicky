"""What is left of the PlanetScope displacement after the one constant per date that E1 removes (itareo.ugi_seg.ps_stage, CLAUDE.md 13.2 and 13.6). Measurement only; E1's inputs stay as they are.

Each normalised PlanetScope mosaic at 3 m against Neo v3 averaged to 3 m, on independent blocks of 1 km (333 px) on a 1 km stride: phase correlation with the criteria of scripts/a7_coregistration.correlate, low-pass estimate beside it.
Per fold quadrant, and per block: the median and spread, the residual after the constant, the adjacent-block difference (if the block values were noise, its robust SD / sqrt 2 would equal the spread over blocks), a plane per axis and the residual after it.
Outputs in docs/evidence_phase2/ugi_seg/e1_ps_residuals/ (refuses an existing folder): summary.json, blocks_<date>.csv.
Run from work_ms/ (CPU): python scripts/ps_residuals.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import a7_coregistration as A7  # noqa: E402

from itareo.grid import COMMON  # noqa: E402
from itareo.ugi_seg.ps_stage import DATES, PS, load_offsets  # noqa: E402

NEO = WM / "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif"
OUT = WM.parent / "docs/evidence_phase2/ugi_seg/e1_ps_residuals"
XL, YL = 388282.5, 6525025.5
BS = 333


def robust_sd(v):
    return float(np.subtract(*np.percentile(v, [75, 25])) / 1.349)


def measure(d, off):
    t0 = time.time()
    rows = []
    with rasterio.open(NEO) as neo, rasterio.open(PS.format(d)) as src:
        for r0 in range(0, COMMON.height - BS, BS):
            for c0 in range(0, COMMON.width - BS, BS):
                a = src.read([2, 4, 6, 8], window=Window(c0, r0, BS, BS)).astype(np.float64)
                a[:, (a == 0).any(0)] = np.nan
                if np.isfinite(a[0]).mean() < 0.9:
                    continue
                n = neo.read([1, 2, 3, 4], window=Window(c0 * 3, r0 * 3, BS * 3, BS * 3), boundless=True, fill_value=65535).astype(np.float64)
                n[:, (n == 65535).any(0)] = np.nan
                n = n.reshape(4, BS, 3, BS, 3).mean(axis=(2, 4))
                if np.isfinite(n[0]).mean() < 0.9:
                    continue
                xc, yc = COMMON.left + 3 * (c0 + BS / 2), COMMON.top - 3 * (r0 + BS / 2)
                fold = (1 if xc <= XL else 2) if yc <= YL else (3 if xc <= XL else 4)
                r = A7.correlate({"r0": r0, "c0": c0, "x": xc, "y": yc, "fold": fold}, *A7.vis_ndvi(n * 1e-4, [0, 1, 2], 0, 3), np.zeros((BS, BS), bool), *A7.vis_ndvi(a * 1e-4, [0, 1, 2], 2, 3), np.zeros((BS, BS), bool), 3.0, lowpass=0.1)
                rows.append(r)
    u = [r for r in rows if r.get("used")]
    with open(OUT / f"blocks_{d}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    cx, cy = off["dx_east_m"], off["dy_north_m"]
    dx, dy = np.array([r["dx_m"] for r in u]), np.array([r["dy_m"] for r in u])
    lx, ly = np.array([r["dx_lp_m"] for r in u]), np.array([r["dy_lp_m"] for r in u])
    fd = np.array([r["fold"] for r in u])
    x, y = np.array([r["x"] for r in u]), np.array([r["y"] for r in u])
    res = np.hypot(dx - cx, dy - cy)
    idx = {(r["r0"], r["c0"]): i for i, r in enumerate(u)}
    de, dn = [], []
    for (r0, c0), i in idx.items():
        for k in ((r0 + BS, c0), (r0, c0 + BS)):
            j = idx.get(k)
            if j is not None:
                de.append(dx[i] - dx[j])
                dn.append(dy[i] - dy[j])
    A = np.c_[np.ones(len(u)), (x - XL) / 1000, (y - YL) / 1000]
    pe, pn = np.linalg.lstsq(A, dx, rcond=None)[0], np.linalg.lstsq(A, dy, rcond=None)[0]
    rp = np.hypot(dx - A @ pe, dy - A @ pn)
    out = {"constant_removed_m_east_north": [cx, cy], "blocks_measured": len(rows), "blocks_used": len(u), "median_m_east_north": [float(np.median(dx)), float(np.median(dy))], "lowpass_median_m_east_north": [round(float(np.median(lx)), 3), round(float(np.median(ly)), 3)],
           "per_fold_quadrant": {}, "residual_after_constant_per_block_m": {"median": round(float(np.median(res)), 2), "p90": round(float(np.percentile(res, 90)), 2), "share_over_2m": round(float((res > 2).mean()), 3), "share_over_4m": round(float((res > 4).mean()), 3)},
           "spread_over_blocks_robust_sd_m_east_north": [round(robust_sd(dx), 2), round(robust_sd(dy), 2)], "adjacent_block_difference_robust_sd_over_sqrt2_m_east_north": [round(robust_sd(np.array(de)) / np.sqrt(2), 2), round(robust_sd(np.array(dn)) / np.sqrt(2), 2)],
           "plane_m_and_m_per_km_east_and_north": {"east": np.round(pe, 3).tolist(), "north": np.round(pn, 3).tolist(), "reference_point_utm": [XL, YL]}, "east_predicted_at_west_and_east_block_m": [round(float(pe[0] + pe[1] * (x.min() - XL) / 1000), 2), round(float(pe[0] + pe[1] * (x.max() - XL) / 1000), 2)],
           "residual_after_plane_per_block_m": {"median": round(float(np.median(rp)), 2), "p90": round(float(np.percentile(rp, 90)), 2)}, "seconds": round(time.time() - t0, 1)}
    for f in (1, 2, 3, 4):
        m = fd == f
        if m.any():
            out["per_fold_quadrant"][str(f)] = {"blocks": int(m.sum()), "median_m_east": float(np.median(dx[m])), "median_m_north": float(np.median(dy[m])), "residual_after_constant_m_east": round(float(np.median(dx[m]) - cx), 2), "residual_after_constant_m_north": round(float(np.median(dy[m]) - cy), 2)}
    print(d, json.dumps(out), flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    off = load_offsets()
    res = {"what": "displacement of each PlanetScope mosaic relative to Neo v3 (m east, m north), per 1 km block; E1 removes one constant per date", "dates": {d: measure(d, off[d]) for d in DATES}}
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
