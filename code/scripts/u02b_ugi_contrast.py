"""U0.2 redone as a contrast test (Shafi's instruction of 2 Nov 2026, night): mean NDVI inside the UGI polygons minus the
mean NDVI of a 3 m outer ring, with the polygons moved by -4..+4 m in 0.25 m steps, per Neo strip.

NDVI: derived/features/neo2022_indices_grid1m.tif (registered Neo 2022). Masks: derived/ugi_labels/ugi_binary_grid1m.tif
(inside) and the pixels outside the polygons within 3 m of them (Euclidean distance <= 3 px; ring). Moving the polygons by
(e east, s south) reads the NDVI at p + (e, s) for every mask pixel p; fractional offsets by bilinear interpolation of
the NDVI (16 quarter-pixel copies), integer offsets by index arithmetic. Windows: the 1.5 km tiles of GRID_1M with >= 90 %
valid NDVI, >= 95 % from one strip and >= 1 % UGI (as U0.2). Per strip the sums of NDVI and the pixel counts of both
masks are pooled over windows, so each strip has one contrast surface C(e, s). The peak is refined by a parabola in e
and in s. A positive peak (e, s) means the image vegetation lies e m east and s m south of the inventory polygons.
Decision rule (Shafi): correct only if a clear common peak of at least 0.5 m appears.
Outputs: docs/evidence_phase2/ugi_alignment/contrast_surface_<strip>.csv, contrast_summary.json. Run from work_ms/.
--ndvi, --sid, --out run it on another mosaic (v3: CLAUDE.md 13.3 F1) without touching the v1 outputs.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, shift as nd_shift

WM = Path(__file__).resolve().parents[1]
NDVI = WM / "derived/features/neo2022_indices_grid1m.tif"
SID = WM / "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"
UGI = WM / "derived/ugi_labels/ugi_binary_grid1m.tif"
OUT = WM.parent / "docs/evidence_phase2/ugi_alignment"
WIN, PAD, MAXS = 1500, 6, 4
STRIPS = {2: "1_2", 3: "1_3", 5: "1_5", 6: "1_6"}
STEPS = np.arange(-4.0, 4.0001, 0.25)


def main():
    t0 = time.time()
    n = len(STEPS)
    acc = {s: {"in_sum": np.zeros((n, n)), "in_n": np.zeros((n, n)), "ring_sum": np.zeros((n, n)),
               "ring_n": np.zeros((n, n)), "windows": 0} for s in STRIPS.values()}
    with rasterio.open(NDVI) as nv, rasterio.open(SID) as sp, rasterio.open(UGI) as ug:
        for r0 in range(0, nv.height - WIN + 1, WIN):
            for c0 in range(0, nv.width - WIN + 1, WIN):
                w = Window(c0, r0, WIN, WIN)
                wp = Window(c0 - PAD, r0 - PAD, WIN + 2 * PAD, WIN + 2 * PAD)
                a = nv.read(1, window=wp, boundless=True, fill_value=nv.nodata).astype(np.float64)
                ok = a != nv.nodata
                if ok[PAD:-PAD, PAD:-PAD].mean() < 0.9:
                    continue
                sid = sp.read(1, window=w)
                ids, cnt = np.unique(sid[sid > 0], return_counts=True)
                if not len(ids) or cnt.max() / sid.size < 0.95:
                    continue
                m = ug.read(1, window=wp, boundless=True, fill_value=0).astype(bool)
                if m[PAD:-PAD, PAD:-PAD].mean() < 0.01:
                    continue
                strip = STRIPS[int(ids[cnt.argmax()])]
                ring = (~m) & (distance_transform_edt(~m) <= 3.0)
                inner = np.zeros_like(m)
                inner[PAD:-PAD, PAD:-PAD] = True                 # masks counted only inside the window itself
                ndvi = np.where(ok, a / 1e4, np.nan)
                fill = np.nan_to_num(ndvi, nan=np.nanmean(ndvi))
                wvalid = ok.astype(np.float64)
                Hp, Wp = ndvi.shape
                idx_in = np.flatnonzero(m & inner)
                idx_ring = np.flatnonzero(ring & inner)
                for fy in (0.0, 0.25, 0.5, 0.75):
                    for fx in (0.0, 0.25, 0.5, 0.75):
                        # value at (r + fy, c + fx): shift the image by (-fy, -fx)
                        v = nd_shift(fill, (-fy, -fx), order=1, mode="nearest").ravel()
                        q = nd_shift(wvalid, (-fy, -fx), order=1, mode="nearest").ravel() > 0.999
                        for iy in range(-MAXS - 1, MAXS + 1):
                            sy = iy + fy
                            if sy < -MAXS - 1e-9 or sy > MAXS + 1e-9:
                                continue
                            ky = int(round((sy + MAXS) / 0.25))
                            for ix in range(-MAXS - 1, MAXS + 1):
                                sx = ix + fx
                                if sx < -MAXS - 1e-9 or sx > MAXS + 1e-9:
                                    continue
                                kx = int(round((sx + MAXS) / 0.25))
                                off = iy * Wp + ix
                                for key, idx in (("in", idx_in), ("ring", idx_ring)):
                                    j = idx + off
                                    good = q[j]
                                    acc[strip][f"{key}_sum"][ky, kx] += v[j][good].sum()
                                    acc[strip][f"{key}_n"][ky, kx] += good.sum()
                acc[strip]["windows"] += 1
                print(strip, r0, c0, round(time.time() - t0), "s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    for s, d in acc.items():
        if d["windows"] == 0:
            summ[s] = {"windows": 0}
            continue
        C = d["in_sum"] / np.maximum(d["in_n"], 1) - d["ring_sum"] / np.maximum(d["ring_n"], 1)
        with open(OUT / f"contrast_surface_{s}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["south_m \\ east_m"] + [f"{v:.2f}" for v in STEPS])
            for k, sy in enumerate(STEPS):
                w.writerow([f"{sy:.2f}"] + [f"{v:.5f}" for v in C[k]])
        ky, kx = np.unravel_index(np.argmax(C), C.shape)
        edge = ky in (0, n - 1) or kx in (0, n - 1)

        def para(a, b, c):
            dd = a - 2 * b + c
            return 0.0 if dd >= 0 else 0.5 * (a - c) / dd * 0.25
        e = STEPS[kx] + (0 if edge else para(C[ky, kx - 1], C[ky, kx], C[ky, kx + 1]))
        so = STEPS[ky] + (0 if edge else para(C[ky - 1, kx], C[ky, kx], C[ky + 1, kx]))
        k0 = int(round(MAXS / 0.25))
        summ[s] = {"windows": d["windows"], "inside_px": int(d["in_n"][k0, k0]), "ring_px": int(d["ring_n"][k0, k0]),
                   "contrast_at_zero": round(float(C[k0, k0]), 5), "contrast_at_peak": round(float(C[ky, kx]), 5),
                   "peak_east_m": round(float(e), 3), "peak_south_m": round(float(so), 3), "peak_at_edge": bool(edge),
                   "gain_over_zero": round(float(C[ky, kx] - C[k0, k0]), 5)}
    strips_ok = [s for s, v in summ.items() if v.get("windows", 0) >= 10]
    peaks = np.array([[summ[s]["peak_east_m"], summ[s]["peak_south_m"]] for s in strips_ok])
    mags = np.hypot(peaks[:, 0], peaks[:, 1]) if len(peaks) else np.array([])
    common = bool(len(peaks) >= 2 and (mags >= 0.5).all() and np.ptp(peaks[:, 0]) <= 0.5 and np.ptp(peaks[:, 1]) <= 0.5
                  and not any(summ[s]["peak_at_edge"] for s in strips_ok))
    res = {"per_strip": summ, "strips_with_10_or_more_windows": strips_ok,
           "rule": "correct only if every strip with >= 10 windows peaks >= 0.5 m from zero, the peaks agree within "
                   "0.5 m in each axis, and no peak lies on the +-4 m edge",
           "common_peak_found": common, "seconds": round(time.time() - t0),
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(res, open(OUT / "contrast_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndvi", help="NDVI raster (default: the v1 index raster)")
    ap.add_argument("--sid", help="strip-id raster of the same mosaic")
    ap.add_argument("--out", help="output folder (default docs/evidence_phase2/ugi_alignment)")
    a = ap.parse_args()
    if a.ndvi:
        NDVI = Path(a.ndvi).resolve()
    if a.sid:
        SID = Path(a.sid).resolve()
    if a.out:
        OUT = Path(a.out).resolve()
    main()
