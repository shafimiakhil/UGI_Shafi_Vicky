"""E1b (CLAUDE.md 13.7): the checks of the plane per date, before E1b trains. Reads only; refuses an existing output folder.

  ramp      on a synthetic date whose content is a known linear ramp displaced by the plane, stage_tile_plane must return the ramp at the 4 m cell centres: mean signed error under 0.02 m, largest under 0.5 m, the staircase of averaging 3 m pixels into 4 m cells. This proves the sign of the constant in plane_transform and catches a slope term only if it moves the tile mean by 0.02 m (slope x about 264 m from the window corner): on these planes c1e, c1n and a flipped c2e, not an omitted c2e or c2n. A wrong constant inside plane_value cancels here; the blocks catch it. The other terms were checked separately in the review of 4 Nov (algebra, and one term at a time at +-30 m/km). The first run (e1b_ps_plane/) asked for a largest error under 0.05 m, which that staircase cannot meet.
  constant  with slopes of zero, stage_tile_plane must equal stage_tile.
  blocks    PlanetScope warped to 4 m on 1,024 m windows on a 1,024 m stride against Neo v3 averaged to 4 m, phase correlation with the criteria of scripts/a7_coregistration.correlate (low-pass beside it): the displacement per block left by the constant and by the plane, per date and per fold quadrant.
Criterion, set before the run: for every date, the median residual per block (low-pass) with the plane is at most that with the constant, and it is below 1.5 m. Outputs in docs/evidence_phase2/ugi_seg/<out>/: residuals.json, blocks_<date>.csv. Exit 1 if it fails.
Run from work_ms/ (CPU): python scripts/e1b_plane_check.py [--out-name e1b_ps_plane_run2]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import a7_coregistration as A7  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.ugi_seg.ps_stage import DATES, GRID1M_ORIGIN, PS, XL, YL, load_offsets, load_planes, plane_value, stage_tile, stage_tile_plane  # noqa: E402

OUT = WM.parent / "docs/evidence_phase2/ugi_seg/e1b_ps_plane"
NEO = WM / "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif"
SIZE = 1024


def ramp_test(planes):
    """Content g(x, y) = 1000 + 0.2 (x - x00) + 0.1 (y00 - y) at the corrected position. A source pixel at raw position q holds g(q - D(q)); after the warp a 4 m cell centred on c must hold g(c)."""
    rows = []
    with rasterio.open(PS.format(DATES[0])) as s:
        crs, T = s.crs, s.transform
    x00, y00 = T.c, T.f
    for d, p in planes.items():
        for row0, col0 in ((4096, 4096), (10240, 12288), (16384, 20480)):
            x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
            c0, r0 = int((x0 - T.c) // 3) - 20, int((T.f - y0) // 3) - 20
            n = 240
            tr = T * Affine.translation(c0, r0)
            xs = tr.c + (np.arange(n) + 0.5) * 3
            ys = tr.f - (np.arange(n) + 0.5) * 3
            X, Y = np.meshgrid(xs, ys)
            de, dn = plane_value(p, X, Y)
            val = 1000 + 0.2 * (X - de - x00) + 0.1 * (y00 - (Y - dn))
            with MemoryFile() as mf:
                with mf.open(driver="GTiff", width=n, height=n, count=8, dtype="float32", crs=crs, transform=tr, nodata=0) as dst:
                    for b in range(1, 9):
                        dst.write(val.astype(np.float32), b)
                with mf.open() as src:
                    got = stage_tile_plane(src, row0, col0, p)[0]
            cx = x0 + (np.arange(128) + 0.5) * 4
            cy = y0 - (np.arange(128) + 0.5) * 4
            CX, CY = np.meshgrid(cx, cy)
            want = 1000 + 0.2 * (CX - x00) + 0.1 * (y00 - CY)
            ok = np.isfinite(got)
            rows.append({"date": d, "row0": row0, "col0": col0, "valid_cells": int(ok.sum()), "plane_at_the_tile_m_east_north": [round(float(v), 3) for v in plane_value(p, x0 + 256, y0 - 256)], "mean_signed_error_m": float((got - want)[ok].mean() / 0.2), "max_abs_error_m": float(np.abs(got - want)[ok].max() / 0.2)})
    return rows


def constant_test():
    off = load_offsets()
    rows = []
    for d in DATES:
        p = {"e": [off[d]["dx_east_m"], 0.0, 0.0], "n": [off[d]["dy_north_m"], 0.0, 0.0]}
        with rasterio.open(PS.format(d)) as s:
            for row0, col0 in ((10240, 12288), (12288, 8192)):
                a, b = stage_tile(s, row0, col0, off[d]["dx_east_m"], off[d]["dy_north_m"]), stage_tile_plane(s, row0, col0, p)
                both = np.isfinite(a) & np.isfinite(b)
                rows.append({"date": d, "row0": row0, "col0": col0, "same_nan_mask": bool(np.array_equal(np.isfinite(a), np.isfinite(b))), "max_rel_diff": float((np.abs(a - b)[both] / np.maximum(np.abs(a[both]), 1)).max()) if both.any() else None})
    return rows


def blocks(d, plane, off):
    rows = []
    with rasterio.open(NEO) as neo, rasterio.open(PS.format(d)) as src:
        for row0 in range(0, 24381 - SIZE, SIZE):
            for col0 in range(0, 24297 - SIZE, SIZE):
                n = neo.read([1, 2, 3, 4], window=Window(col0, row0, SIZE, SIZE)).astype(np.float64)
                n[:, (n == 65535).any(0)] = np.nan
                n = n.reshape(4, SIZE // 4, 4, SIZE // 4, 4).mean(axis=(2, 4))
                if np.isfinite(n[0]).mean() < 0.9:
                    continue
                xc, yc = GRID1M_ORIGIN[0] + col0 + SIZE / 2, GRID1M_ORIGIN[1] - row0 - SIZE / 2
                fold = (1 if xc <= XL else 2) if yc <= YL else (3 if xc <= XL else 4)
                rec = {"row0": row0, "col0": col0, "x": xc, "y": yc, "fold": fold}
                for tag, a in (("plane", stage_tile_plane(src, row0, col0, plane, SIZE)), ("constant", constant_window(src, row0, col0, off))):
                    a = a[[1, 3, 5, 7]].astype(np.float64)
                    if np.isfinite(a[0]).mean() < 0.9:
                        continue
                    r = A7.correlate({}, *A7.vis_ndvi(n * 1e-4, [0, 1, 2], 0, 3), np.zeros(n.shape[1:], bool), *A7.vis_ndvi(a * 1e-4, [0, 1, 2], 2, 3), np.zeros(n.shape[1:], bool), 4.0, lowpass=0.1)
                    if r.get("used"):
                        rec.update({f"{tag}_dx_m": r["dx_m"], f"{tag}_dy_m": r["dy_m"], f"{tag}_dx_lp_m": r["dx_lp_m"], f"{tag}_dy_lp_m": r["dy_lp_m"]})
                rows.append(rec)
    return rows


def constant_window(src, row0, col0, off):
    """stage_tile on a window of SIZE: the plane path with slopes of zero (equal to stage_tile, constant_test)."""
    return stage_tile_plane(src, row0, col0, {"e": [off["dx_east_m"], 0.0, 0.0], "n": [off["dy_north_m"], 0.0, 0.0]}, SIZE)


def summary(rows):
    out = {}
    for tag in ("constant", "plane"):
        u = [r for r in rows if f"{tag}_dx_lp_m" in r]
        lx, ly, dx, dy = (np.array([r[f"{tag}_{k}"] for r in u]) for k in ("dx_lp_m", "dy_lp_m", "dx_m", "dy_m"))
        res, resu = np.hypot(lx, ly), np.hypot(dx, dy)
        fold = np.array([r["fold"] for r in u])
        out[tag] = {"blocks": len(u), "lowpass_median_m_east_north": [round(float(np.median(lx)), 3), round(float(np.median(ly)), 3)], "lowpass_residual_per_block_m": {"median": round(float(np.median(res)), 2), "p90": round(float(np.percentile(res, 90)), 2), "share_over_2m": round(float((res > 2).mean()), 3), "share_over_4m": round(float((res > 4).mean()), 3)},
                    "unweighted_residual_per_block_m": {"median": round(float(np.median(resu)), 2), "p90": round(float(np.percentile(resu, 90)), 2)}, "per_fold_quadrant_lowpass_median_m_east_north": {str(f): [round(float(np.median(lx[fold == f])), 2), round(float(np.median(ly[fold == f])), 2)] for f in (1, 2, 3, 4) if (fold == f).any()}}
    return out


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="e1b_ps_plane")
    OUT = OUT.parent / ap.parse_args().out_name
    refuse_existing(OUT)
    OUT.mkdir(parents=True)
    planes, off = load_planes(), load_offsets()
    res = {"planes": planes, "ramp": ramp_test(planes), "constant": constant_test(), "dates": {}}
    # Averaging 3 m pixels into 4 m cells leaves a staircase on a ramp, of up to about 0.3 m here, so the maximum cannot be small; a wrong sign of the constant would shift the mean by twice the plane; a wrong slope term by slope x about 264 m.
    res["ramp_passed"] = all(r["valid_cells"] > 0.9 * 128 * 128 and abs(r["mean_signed_error_m"]) < 0.02 and r["max_abs_error_m"] < 0.5 for r in res["ramp"])
    res["constant_passed"] = all(r["same_nan_mask"] and (r["max_rel_diff"] or 0) < 1e-5 for r in res["constant"])
    print(json.dumps({k: res[k] for k in ("ramp", "constant", "ramp_passed", "constant_passed")}, indent=0), flush=True)
    for d in DATES:
        rows = blocks(d, planes[d], off[d])
        with open(OUT / f"blocks_{d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        res["dates"][d] = summary(rows)
        print(d, json.dumps(res["dates"][d]), flush=True)
    res["criterion"] = "for every date, the low-pass median residual per block with the plane is at most that with the constant and below 1.5 m; set before the run"
    res["blocks_passed"] = all(v["plane"]["lowpass_residual_per_block_m"]["median"] <= v["constant"]["lowpass_residual_per_block_m"]["median"] and v["plane"]["lowpass_residual_per_block_m"]["median"] < 1.5 for v in res["dates"].values())
    res["passed"] = bool(res["ramp_passed"] and res["constant_passed"] and res["blocks_passed"])
    json.dump(res, open(OUT / "residuals.json", "w"), indent=1)
    print("passed", res["passed"], flush=True)
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
