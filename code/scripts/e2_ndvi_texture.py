"""E2 (CLAUDE.md 13.6): ndvi_orientcons_41m and ndvi_glcmcorr_41m on every pixel of GRID_1M, from the Neo v3 NDVI (derived/features/neo2022_v3_indices_grid1m.tif, band 1).

The features are the functions of the texture screen, unchanged (scripts/u04_texture_screen.py: structure_feats, spatial_stats; window 41 m, scales.json), computed in blocks of 4,096 px with a 64 px margin, wider than
any filter's reach (window 20 + Gaussian 6 + Sobel 1 + pair 1 px). At the grid edge the GLCM windows use what exists. The orientation layer follows scipy's edge rules (reflect in Sobel and Gaussian, nearest in the 41 px window), so within 26 px of the grid edge it differs from the screen, whose read_pad padded with invalid pixels; no inventory pixel lies in that zone.

The GLCM needs a quantisation range. The screen took its range from sampled tiles of all folds (run_log.json: -0.0146, 0.7987). Section 0 wants statistics from training folds only, so here the range is the 1st and 99th
percentile of NDVI over the internal training tiles of each fold, and the correlation is computed four times: bands 2-5, one per held-out fold. No label enters any of this.

Output: derived/features/neo2022_v3_ndvi_texture_grid1m.tif, COG, float32 NBITS=16, nodata NaN, deflate; band 1 ndvi_orientcons_41m, bands 2-5 ndvi_glcmcorr_41m_fold1..4; ..._log.json with the ranges, the CPU time and a check: the same functions on
windows placed differently from the blocks, read back from the file, within float16 precision. Exit 1 if the check fails, or if the CPU time exceeds the 1 h of 13.6 (whether the GLCM stays is then a later decision: kept, derived/features/README.md; the file stays).
Refuses existing outputs. Run from work_ms/.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import u04_texture_screen as S   # noqa: E402       (sets one thread per process and no GPU)

import numpy as np               # noqa: E402
import rasterio                  # noqa: E402
from rasterio.windows import Window  # noqa: E402

NDVI = WM / "derived/features/neo2022_v3_indices_grid1m.tif"
OUT = WM / "derived/features/neo2022_v3_ndvi_texture_grid1m.tif"
LOG = WM / "derived/features/neo2022_v3_ndvi_texture_log.json"
CFG = WM / "itareo/configs/ugi_e2_tex.yaml"
W41 = {"41m": 41}
BLOCK, M = 4096, 64
NAMES = ["ndvi_orientcons_41m"] + [f"ndvi_glcmcorr_41m_fold{f}" for f in (1, 2, 3, 4)]


def features(r0, r1, c0, c1, ranges, H, W):
    """The five layers of the rows and columns [r0, r1), [c0, c1): float32."""
    ra, rb, ca, cb = max(r0 - M, 0), min(r1 + M, H), max(c0 - M, 0), min(c1 + M, W)
    with rasterio.open(NDVI) as s:
        nd = s.read(1, window=Window(ca, ra, cb - ca, rb - ra))
        v = nd != s.nodata
    x = np.where(v, nd.astype(float) / 1e4, np.nan)
    out = np.empty((5, r1 - r0, c1 - c0), np.float32)
    core = (slice(r0 - ra, r1 - ra), slice(c0 - ca, c1 - ca))
    out[0] = S.structure_feats(x, v, W41, "ndvi")["ndvi_orientcons_41m"][core]
    for k, (lo, hi) in enumerate(ranges):
        out[1 + k] = S.spatial_stats(x, v, 0.0, 1.0, lo, hi, W41, "ndvi")["ndvi_glcmcorr_41m"][core]
    return out


def task(t):
    c = time.process_time()
    r0, r1, c0, c1, ranges, H, W = t
    return r0, r1, c0, c1, features(r0, r1, c0, c1, ranges, H, W), time.process_time() - c


def fold_ranges():
    """1st and 99th percentile of NDVI over the internal training tiles of each fold."""
    import yaml

    from itareo.ugi_seg.train import split
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    with rasterio.open(NDVI) as s:
        a = s.read(1)
        nod = s.nodata
    out = {}
    for f in (1, 2, 3, 4):
        _, tr, _, _, _ = split(cfg, f)
        v = np.concatenate([a[t["row0"]:t["row0"] + 512, t["col0"]:t["col0"] + 512].ravel() for t in tr])
        v = v[v != nod]
        lo, hi = np.percentile(v, [1, 99]) / 1e4
        out[f] = {"lo": float(lo), "hi": float(hi), "tiles": len(tr), "pixels": int(v.size)}
        print("fold", f, out[f], flush=True)
    return out


def main():
    S_T0 = time.time()
    from itareo.safe_write import refuse_existing
    refuse_existing(OUT, LOG)
    r = fold_ranges()
    ranges = [(r[f]["lo"], r[f]["hi"]) for f in (1, 2, 3, 4)]
    with rasterio.open(NDVI) as s:
        H, W, T, crs = s.height, s.width, s.transform, s.crs
    tasks = [(a, min(a + BLOCK, H), b, min(b + BLOCK, W), ranges, H, W) for a in range(0, H, BLOCK) for b in range(0, W, BLOCK)]
    arr = np.full((5, H, W), np.nan, np.float32)
    cpu = 0.0
    with ProcessPoolExecutor(8) as ex:
        for n, (a, a1, b, b1, f, c) in enumerate(ex.map(task, tasks, chunksize=1), 1):
            arr[:, a:a1, b:b1] = f
            cpu += c
            if n % 6 == 0:
                print(n, "of", len(tasks), round(time.time() - S_T0), "s", flush=True)
    print("computed", round(time.time() - S_T0), "s, cpu", round(cpu), "s", flush=True)
    prof = dict(driver="COG", width=W, height=H, count=5, dtype="float32", nodata=np.nan, crs=crs, transform=T, compress="DEFLATE", NBITS=16, BLOCKSIZE=512, BIGTIFF="YES")
    with rasterio.open(OUT, "w", **prof) as d:
        d.write(arr)
        for i, n in enumerate(NAMES, 1):
            d.set_band_description(i, n)
    shares = {n: float(np.isfinite(arr[i]).mean()) for i, n in enumerate(NAMES)}
    stats = {n: [round(float(v), 4) for v in np.nanpercentile(arr[i][::16, ::16], [1, 25, 50, 75, 99])] for i, n in enumerate(NAMES)}
    del arr
    # the check: windows that do not start on the blocks, computed anew, against the file as stored
    rng = np.random.default_rng(0)
    rows = []
    with rasterio.open(OUT) as s:
        for _ in range(12):
            a, b = int(rng.integers(0, H - 512)), int(rng.integers(0, W - 512))
            f = features(a, a + 512, b, b + 512, ranges, H, W)
            g = s.read(window=Window(b, a, 512, 512))
            ok = np.isfinite(f) & np.isfinite(g)
            rows.append({"row0": a, "col0": b, "same_valid": bool((np.isfinite(f) == np.isfinite(g)).all()), "max_abs_diff": float(np.abs(f[ok] - g[ok]).max())})
    passed = all(x["same_valid"] and x["max_abs_diff"] < 2e-3 for x in rows)
    log = {"output": str(OUT.relative_to(WM)), "bands": NAMES, "bytes": OUT.stat().st_size, "ranges_per_held_out_fold": r, "screen_range_all_folds": [-0.0146, 0.7987], "valid_share": shares, "percentiles_1_25_50_75_99": stats, "blocks": len(tasks),
           "cpu_seconds_of_blocks": round(cpu, 1), "wall_seconds": round(time.time() - S_T0, 1), "within_1h_of_cpu": cpu <= 3600, "check_windows": rows, "check_passed": passed, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "check_windows"}, indent=1))
    raise SystemExit(0 if passed and cpu <= 3600 else 1)


if __name__ == "__main__":
    main()
