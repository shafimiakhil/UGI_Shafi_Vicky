"""PlanetScope 2022 at the 4 m stage grid for B4 (CLAUDE.md 13.1, 13.2): data-loader component, no training.

Each date's normalised mosaic (derived/planetscope2022/ps2022MMDD_sr_norm_s2_3m.tif, 3 m on the COMMON lattice, SR x 10000,
nodata 0) is resampled once, by area averaging (rasterio.warp.reproject, Resampling.average), to the 4 m stage grid of a
512 m tile: 128 x 128 px whose origin is the tile's GRID_1M corner. The measured displacement of the date relative to the
registered Neo 2022 (m east, m north) is removed inside that same resampling by shifting the source transform by minus
the displacement, so the date's median displacement is removed. One constant per date: a residual that varies over the scene
remains (per block and per fold quadrant, docs/evidence_phase2/ugi_seg/e1_ps_residuals/summary.json). Offsets: runs_itareo/ugi/b4_ps_offsets.json
(written by write_offsets() from derived/planetscope2022/u06_log.json, coregistration, vs_neo2022_registered).

E1b (CLAUDE.md 13.7): a plane per date instead of the constant, D(x, y) = c0 + c1 x' + c2 y' per axis, x' and y' in km from the fold crossing, removed inside the same single resampling by an affine source transform (stage_tile_plane). Fitted by write_planes()
on the used 1 km blocks of docs/evidence_phase2/ugi_seg/e1_ps_residuals/ (low-pass estimate, the standard for resampled copies since 13.4; least squares, then one refit without the blocks more than 3 m off on either axis). No labels enter it. Planes: runs_itareo/ugi/b4_ps_planes.json.

  python -m itareo.ugi_seg.ps_stage --write-planes       the planes file
  python -m itareo.ugi_seg.ps_stage --write-offsets      the offsets file
  python -m itareo.ugi_seg.ps_stage --check [--tiles N]  phase correlation of the stage-grid PlanetScope (with and without
                                                         the offsets) against Neo averaged to 4 m on N random tiles
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

DATES = ["20220424", "20220910", "20221028"]
PS = "derived/planetscope2022/ps{}_sr_norm_s2_3m.tif"
OFFSETS = "runs_itareo/ugi/b4_ps_offsets.json"
PLANES = "runs_itareo/ugi/b4_ps_planes.json"
RESIDUAL_BLOCKS = "../docs/evidence_phase2/ugi_seg/e1_ps_residuals/blocks_{}.csv"
XL, YL = 388282.5, 6525025.5
GROSS_M = 3.0
GRID1M_ORIGIN = (375327.0, 6535833.0)
STAGE, T = 4.0, 512


def write_offsets():
    log = json.loads(Path("derived/planetscope2022/u06_log.json").read_text(encoding="utf-8"))["coregistration"]
    off = {d: {"dx_east_m": log[d]["vs_neo2022_registered"]["dx_east_m_median"],
               "dy_north_m": log[d]["vs_neo2022_registered"]["dy_north_m_median"],
               "blocks": log[d]["vs_neo2022_registered"]["blocks"]} for d in DATES}
    Path(OFFSETS).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"offsets_relative_to_registered_neo2022": off, "source": "derived/planetscope2022/u06_log.json",
               "use": "content displacement of the date relative to the Neo; removed by shifting the source transform "
                      "by (-dx, -dy) inside the single resampling to the 4 m stage grid"}, open(OFFSETS, "w"), indent=1)
    return off


def load_offsets():
    return json.load(open(OFFSETS))["offsets_relative_to_registered_neo2022"]


def stage_tile(src, row0, col0, dx=0.0, dy=0.0):
    """src: open rasterio dataset of one date. Returns (8, 128, 128) float32 SR (NaN where invalid) on the tile's 4 m
    stage grid, with the content shifted by (-dx, -dy) m."""
    n = int(T / STAGE)
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    dst_t = Affine(STAGE, 0, x0, 0, -STAGE, y0)
    st = src.transform
    c = int((x0 - st.c) // 3) - 2
    r = int((st.f - y0) // 3) - 2
    w = Window(c, r, int(T / 3) + 6, int(T / 3) + 6)
    a = src.read(window=w, boundless=True, fill_value=0).astype(np.float32)
    a[:, (a == 0).any(0)] = np.nan
    src_t = src.window_transform(w)
    src_t = Affine(src_t.a, src_t.b, src_t.c - dx, src_t.d, src_t.e, src_t.f - dy)
    out = np.full((a.shape[0], n, n), np.nan, np.float32)
    reproject(a, out, src_transform=src_t, src_crs=src.crs, dst_transform=dst_t, dst_crs=src.crs,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def write_planes():
    import csv
    from itareo.safe_write import refuse_existing
    refuse_existing(PLANES)
    out = {}
    for d in DATES:
        rows = [r for r in csv.DictReader(open(RESIDUAL_BLOCKS.format(d))) if r.get("used") == "True"]
        x, y = np.array([float(r["x"]) for r in rows]), np.array([float(r["y"]) for r in rows])
        D = np.column_stack([[float(r["dx_lp_m"]) for r in rows], [float(r["dy_lp_m"]) for r in rows]])
        A = np.c_[np.ones(len(rows)), (x - XL) / 1000, (y - YL) / 1000]
        c = np.linalg.lstsq(A, D, rcond=None)[0]
        keep = (np.abs(D - A @ c) <= GROSS_M).all(1)
        c = np.linalg.lstsq(A[keep], D[keep], rcond=None)[0]
        out[d] = {"e": [float(v) for v in c[:, 0]], "n": [float(v) for v in c[:, 1]], "blocks": len(rows), "blocks_refit": int(keep.sum())}
    json.dump({"planes": out, "model": "displacement of the date relative to Neo v3, m: c0 + c1 x' + c2 y', x' = (x - 388282.5) / 1 km, y' = (y - 6525025.5) / 1 km", "source": RESIDUAL_BLOCKS, "estimate": "low-pass phase correlation, used blocks", "gross_m": GROSS_M}, open(PLANES, "w"), indent=1)
    return out


def load_planes():
    return json.load(open(PLANES))["planes"]


def plane_value(p, x, y):
    return (p["e"][0] + p["e"][1] * (x - XL) / 1000 + p["e"][2] * (y - YL) / 1000, p["n"][0] + p["n"][1] * (x - XL) / 1000 + p["n"][2] * (y - YL) / 1000)


def plane_transform(st, p):
    """Source transform st (north up) with the plane removed: x' = x - De(x, y), y' = y - Dn(x, y), an affine map."""
    be, ce, bn, cn = p["e"][1] / 1000, p["e"][2] / 1000, p["n"][1] / 1000, p["n"][2] / 1000
    de, dn = plane_value(p, st.c, st.f)
    return Affine(st.a * (1 - be), -st.e * ce, st.c - de, -st.a * bn, st.e * (1 - cn), st.f - dn)


def stage_tile_plane(src, row0, col0, p, size=T):
    """As stage_tile, with the plane p removed. Returns (bands, size / 4, size / 4) float32 SR (NaN where invalid). The source window follows the displacement at the window's centre."""
    n = int(size / STAGE)
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    dst_t = Affine(STAGE, 0, x0, 0, -STAGE, y0)
    st = src.transform
    de, dn = plane_value(p, x0 + size / 2, y0 - size / 2)
    c = int((x0 + de - st.c) // 3) - 2
    r = int((st.f - y0 - dn) // 3) - 2
    w = Window(c, r, int(size / 3) + 6, int(size / 3) + 6)
    a = src.read(window=w, boundless=True, fill_value=0).astype(np.float32)
    a[:, (a == 0).any(0)] = np.nan
    out = np.full((a.shape[0], n, n), np.nan, np.float32)
    reproject(a, out, src_transform=plane_transform(src.window_transform(w), p), src_crs=src.crs, dst_transform=dst_t, dst_crs=src.crs, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def check(n_tiles=60, seed=0):
    import sys
    sys.path.insert(0, "scripts")
    from itareo.phasecorr import phase_corr
    from itareo.unet_ugi.data import read_csv
    off = load_offsets()
    tiles = [t for t in read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv") if not t["straddle"] and t["avail_ps_all3"] >= 0.95]
    rng = np.random.default_rng(seed)
    pick = [tiles[i] for i in rng.choice(len(tiles), min(n_tiles, len(tiles)), replace=False)]
    res = {}
    with rasterio.open("derived/pneo2022_registered/pneo2022_toa_grid1m.tif") as neo:
        for d in DATES:
            with rasterio.open(PS.format(d)) as src:
                rows = {"with_offset": [], "without_offset": []}
                for t in pick:
                    a = neo.read([1, 2, 3], window=Window(t["col0"], t["row0"], T, T)).astype(np.float32)
                    a[:, (a == 65535).any(0)] = np.nan
                    nv = a.reshape(3, 128, 4, 128, 4).mean(axis=(2, 4)).mean(0)
                    for key, (dx, dy) in (("with_offset", (off[d]["dx_east_m"], off[d]["dy_north_m"])), ("without_offset", (0.0, 0.0))):
                        p = stage_tile(src, t["row0"], t["col0"], dx, dy)
                        pv = p[[1, 3, 5]].mean(0)                     # blue, green, red
                        ok = np.isfinite(nv) & np.isfinite(pv)
                        if ok.mean() < 0.95:
                            continue
                        sy, sx, pk, q = phase_corr(np.where(ok, nv, np.nanmean(nv)), np.where(ok, pv, np.nanmean(pv)))
                        if q >= 2:
                            rows[key].append((sx * STAGE, -sy * STAGE))
                res[d] = {k: {"tiles": len(v), "dx_east_m_median": round(float(np.median([x[0] for x in v])), 3) if v else None,
                              "dy_north_m_median": round(float(np.median([x[1] for x in v])), 3) if v else None}
                          for k, v in rows.items()}
                res[d]["offset_applied_m"] = [off[d]["dx_east_m"], off[d]["dy_north_m"]]
    out = {"check": res, "note": "displacement of the stage-grid PlanetScope content relative to Neo averaged to 4 m "
                                 "(mean of R, G, B against mean of blue, green, red), phase correlation, quality >= 2"}
    json.dump(out, open(Path(OFFSETS).with_name("b4_ps_stage_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def _overlap(dst_edges, src_edges):
    lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1])
    hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:])
    return np.clip(hi - lo, 0, None)


def _reference(src, row0, col0, dx, dy):
    """Independent stage-grid value: separable overlap-length weights of the shifted 3 m pixels on the 4 m pixels, invalid
    3 m pixels excluded (float64)."""
    st = src.transform
    x0, y0 = GRID1M_ORIGIN[0] + col0, GRID1M_ORIGIN[1] - row0
    c, r, n = int((x0 - st.c) // 3) - 6, int((st.f - y0) // 3) - 6, 190
    a = src.read(window=Window(c, r, n, n), boundless=True, fill_value=0).astype(np.float64)
    v = ~(a == 0).any(0)
    a = np.where(v, a, 0)
    xs = st.c + 3 * (c + np.arange(n + 1)) - dx
    ys = st.f - 3 * (r + np.arange(n + 1)) - dy
    m = int(T / STAGE)
    Wx = _overlap(x0 + STAGE * np.arange(m + 1), xs)
    Wy = _overlap(-(y0 - STAGE * np.arange(m + 1)), -ys)
    num = np.einsum("ir,brc,jc->bij", Wy, a, Wx)
    den = Wy @ v.astype(float) @ Wx.T
    out = num / np.where(den > 1e-9, den, 1)
    out[:, den <= 1e-9] = np.nan
    return out


def selftest(seed=1):
    """stage_tile against _reference with the real offsets, on 3 fully and 3 partly valid B4 tiles per date (adapted from
    the code review of 2 Nov 2026) -> runs_itareo/ugi/b4_ps_stage_selftest.json"""
    from itareo.unet_ugi.data import read_csv
    off = load_offsets()
    tiles = [t for t in read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv") if not t["straddle"]]
    full = [t for t in tiles if t["avail_ps_all3"] >= 0.99]
    part = [t for t in tiles if 0.3 < t["avail_ps_all3"] < 0.9]
    rng = np.random.default_rng(seed)
    sel = [full[i] for i in rng.choice(len(full), 3, replace=False)] + [part[i] for i in rng.choice(len(part), 3, replace=False)]
    rows = []
    for d in DATES:
        dx, dy = off[d]["dx_east_m"], off[d]["dy_north_m"]
        with rasterio.open(PS.format(d)) as src:
            for t in sel:
                p = stage_tile(src, t["row0"], t["col0"], dx, dy).astype(np.float64)
                q = _reference(src, t["row0"], t["col0"], dx, dy)
                both = np.isfinite(p) & np.isfinite(q)
                rel = np.abs(p - q)[both] / np.maximum(np.abs(q[both]), 1)
                rows.append({"date": d, "row0": t["row0"], "col0": t["col0"], "avail_ps_all3": t["avail_ps_all3"],
                             "valid_cells": int(both[0].sum()),
                             "same_nan_mask": bool(np.array_equal(np.isfinite(p[0]), np.isfinite(q[0]))),
                             "max_rel_diff": float(rel.max()) if rel.size else None})
    ok = all(r["same_nan_mask"] and (r["max_rel_diff"] or 0) < 1e-6 for r in rows)
    out = {"passed": ok, "tolerance_rel": 1e-6, "rows": rows,
           "note": "stage_tile (rasterio average warp, float32) against an independent separable overlap-weighted average"}
    json.dump(out, open(Path(OFFSETS).with_name("b4_ps_stage_selftest.json"), "w"), indent=1)
    print(json.dumps({"passed": ok, "max_rel_diff": max(r["max_rel_diff"] or 0 for r in rows),
                      "tiles": len(rows)}, indent=1))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-offsets", action="store_true")
    ap.add_argument("--write-planes", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tiles", type=int, default=60)
    a = ap.parse_args()
    if a.write_planes:
        print(json.dumps(write_planes(), indent=1))
    if a.write_offsets:
        print(json.dumps(write_offsets(), indent=1))
    if a.check:
        check(a.tiles)
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
