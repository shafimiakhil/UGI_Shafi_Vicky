"""B1b and B1c: harmonised 1 m Pléiades 2017 product and the nominal-lattice fusion (CLAUDE.md 12.3 a and c).

B1b (variant corrected, the A5 products in derived/pleiades2017/):
  --estimate   per-band gains for the west strip from the two fused 0.5 m strips where both are valid in all bands
               (the strip overlap), on stable pixels: NDVI < 0.25 in both strips (not vegetation) and brightness (mean
               of the 4 bands) above the 20th and below the 99.9th percentile of the overlap in both strips (not shadow,
               not saturated or cloud). Gain = ratio of means east / west. Read on a 2 m subsample (every 4th 0.5 m pixel).
               Reported: gains; the same on the north and south halves of the overlap, each applied to the other half;
               the residual east / (gain x west) per brightness decile of the stable pixels and on all overlap pixels.
  --mosaic     A5 mosaic and GRID_1M warp with the west strip multiplied by the gains: *_harmonised.tif next to the
               unharmonised products, which are kept (scripts/a5_pleiades2017.py run_mosaic, same seam rule).
  --seam       radiometric step across the mosaic seam, corrected for the land-cover difference between its sides
               (seam(): same-strip ratio across the seam), unharmonised and with the gains.
B1c (variant nominal, derived/pleiades2017_nominal/):
  --nominal-strips   A5 strips with the PAN-MS offset set to zero (scripts/a5_pleiades2017.py NOMINAL), everything else
                     identical; then --mosaic --variant nominal --gains-from corrected applies the gains of the corrected
                     products, so that the two variants differ only in geometry. --estimate --variant nominal logs the
                     gains the nominal strips would give, for comparison.

Run from work_ms/: python scripts/b1b_harmonise.py --estimate | --mosaic | --seam [--variant nominal --gains-from corrected]
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5_pleiades2017 as A5                       # noqa: E402

DIRS = {"corrected": A5.W / "derived" / "pleiades2017", "nominal": A5.W / "derived" / "pleiades2017_nominal"}
ND = A5.ND16
SUB = 4
BAND = ["R", "G", "B", "NIR"]


def ndvi(a):
    r, n = a[0].astype(np.float64), a[3].astype(np.float64)
    return (n - r) / np.maximum(n + r, 1.0)


def overlap_samples(out):
    srcs = {n: rasterio.open(out / f"pleiades2017_{n}_toa_hpf_05m.tif") for n in ("west", "east")}
    b = [s.bounds for s in srcs.values()]
    ob = (max(x.left for x in b), max(x.bottom for x in b), min(x.right for x in b), min(x.top for x in b))
    wins = {n: from_bounds(*ob, transform=s.transform).round_offsets().round_lengths() for n, s in srcs.items()}
    H, Wd = int(wins["west"].height), int(wins["west"].width)
    assert (H, Wd) == (int(wins["east"].height), int(wins["east"].width))
    vals = {"west": [], "east": []}
    rows = []
    for r0 in range(0, H, 2048):
        nr = min(2048, H - r0)
        blk = {n: srcs[n].read(window=Window(wins[n].col_off, wins[n].row_off + r0, Wd, nr))[:, ::SUB, ::SUB]
               for n in srcs}
        ok = (blk["west"] != ND).all(0) & (blk["east"] != ND).all(0)
        for n in srcs:
            vals[n].append(blk[n][:, ok])
        rows.append((np.nonzero(ok)[0] * SUB + r0).astype(np.int32))
    for s in srcs.values():
        s.close()
    return ({n: np.concatenate(v, axis=1) for n, v in vals.items()}, np.concatenate(rows), H,
            {"overlap_bounds_epsg32721": ob, "overlap_px_05m": [H, Wd]})


def estimate(variant):
    t0 = time.time()
    out = DIRS[variant]
    v, rows, H, info = overlap_samples(out)
    w, e = v["west"].astype(np.float64), v["east"].astype(np.float64)
    bw, be = w.mean(0), e.mean(0)
    lo, hi = np.percentile(np.concatenate([bw, be]), [20, 99.9])
    stable = (ndvi(w) < 0.25) & (ndvi(e) < 0.25) & (bw > lo) & (be > lo) & (bw < hi) & (be < hi)

    def gains(m):
        return e[:, m].sum(1) / w[:, m].sum(1)

    g = gains(stable)
    north = rows < H / 2
    gn, gs = gains(stable & north), gains(stable & ~north)
    split = {"gains_north_half": gn.round(4).tolist(), "gains_south_half": gs.round(4).tolist(),
             "residual_south_with_north_gains": (e[:, stable & ~north].sum(1) / (gn[:, None] * w[:, stable & ~north]).sum(1)).round(4).tolist(),
             "residual_north_with_south_gains": (e[:, stable & north].sum(1) / (gs[:, None] * w[:, stable & north]).sum(1)).round(4).tolist()}
    wg = g[:, None] * w
    bright = 0.5 * (wg.mean(0) + be)
    edges = np.percentile(bright[stable], np.arange(0, 101, 10))
    dec = []
    for k in range(10):
        m = stable & (bright >= edges[k]) & (bright <= edges[k + 1])
        dec.append({"decile": k + 1, "brightness_range": [round(edges[k] / 1e4, 4), round(edges[k + 1] / 1e4, 4)],
                    "pixels": int(m.sum()),
                    "ratio_before": (e[:, m].sum(1) / w[:, m].sum(1)).round(4).tolist(),
                    "ratio_after": (e[:, m].sum(1) / wg[:, m].sum(1)).round(4).tolist()})
    allpx = np.ones(stable.shape, bool)
    res = {"variant": variant, "started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
           "subsample": f"every {SUB}th 0.5 m pixel in rows and columns (2 m)", **info,
           "pixels_both_valid": int(stable.size), "pixels_stable": int(stable.sum()),
           "stable_rule": "NDVI < 0.25 in both strips; brightness (mean of 4 bands) in (P20, P99.9) of the overlap in both",
           "brightness_p20_p999": [round(lo / 1e4, 4), round(hi / 1e4, 4)],
           "gains_west": g.round(5).tolist(), "bands": BAND,
           "ratio_of_means_before_stable": g.round(4).tolist(),
           "ratio_of_means_before_all": (e.sum(1) / w.sum(1)).round(4).tolist(),
           "ratio_of_means_after_all": (e.sum(1) / wg.sum(1)).round(4).tolist(),
           "median_pixel_ratio_stable": np.median(e[:, stable] / np.maximum(w[:, stable], 1), axis=1).round(4).tolist(),
           "split_half": split, "residual_by_brightness_decile_stable": dec,
           "seconds": round(time.time() - t0, 1)}
    del allpx
    (out / "harmonisation_gains.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pixels_both_valid", "pixels_stable", "gains_west",
                                          "ratio_of_means_before_all", "ratio_of_means_after_all", "split_half")}, indent=1))


def seam(variant):
    """Radiometric step across the mosaic seam, corrected for land cover. On the 0.5 m mosaic lattice (strip id), every
    8th row: at each west-to-east change of strip id, the stable pixels (NDVI < 0.25) 3-21 m left and 4-22 m right of
    it, read from both fused strips. land cover = right / left within one strip; seam = east right / west left;
    step = seam / land cover (mean of the two strips), unharmonised and with the west gains. Rows need >= 10 stable
    pixels on each side in both strips. (A first version compared the two sides on GRID_1M without the land-cover term;
    it is archived in archive/2026-11-02/b1b_seam_metric_v1/.)"""
    out = DIRS[variant]
    log = json.loads((out / "mosaic_log.json").read_text(encoding="utf-8")) if (out / "mosaic_log.json").exists() else         json.loads((out / "mosaic_log_harmonised.json").read_text(encoding="utf-8"))
    offs, Wm, Hm = log["strip_offsets_px"], log["mosaic_grid"]["width"], log["mosaic_grid"]["height"]
    hlog = out / "mosaic_log_harmonised.json"          # the gains applied in the harmonised mosaic
    g = np.asarray(json.loads(hlog.read_text(encoding="utf-8"))["west_gains"]) if hlog.exists() else None
    sid_path = out / "pleiades2017_strip_id_05m_epsg32721.tif"
    if not sid_path.exists():
        sid_path = out / "pleiades2017_strip_id_05m_epsg32721_harmonised.tif"
    acc = {k: np.zeros(4) for k in ("ww_l", "ww_r", "ee_l", "ee_r")}
    n = 0
    with rasterio.open(sid_path) as sp:
        srcs = {k: rasterio.open(out / f"pleiades2017_{k}_toa_hpf_05m.tif") for k in ("west", "east")}
        for r0 in range(0, Hm, 2048):
            nr = min(2048, Hm - r0)
            sid = sp.read(1, window=Window(0, r0, Wm, nr))[::8]
            arr = {}
            for k, s in srcs.items():
                c, r = offs[k]
                arr[k] = s.read(window=Window(-c, r0 - r, Wm, nr), boundless=True, fill_value=ND)[:, ::8].astype(np.float64)
            ok = {k: (a != ND).all(0) & (ndvi(a) < 0.25) for k, a in arr.items()}
            for i in range(sid.shape[0]):
                for c in np.nonzero((sid[i, :-1] == 1) & (sid[i, 1:] == 2))[0]:
                    L, R = slice(max(c - 42, 0), max(c - 6, 0)), slice(c + 8, c + 44)
                    m = {k: (ok[k][i, L], ok[k][i, R]) for k in arr}
                    if min(x.sum() for k in m for x in m[k]) < 10:
                        continue
                    acc["ww_l"] += arr["west"][:, i, L][:, m["west"][0]].mean(1)
                    acc["ww_r"] += arr["west"][:, i, R][:, m["west"][1]].mean(1)
                    acc["ee_l"] += arr["east"][:, i, L][:, m["east"][0]].mean(1)
                    acc["ee_r"] += arr["east"][:, i, R][:, m["east"][1]].mean(1)
                    n += 1
        for s in srcs.values():
            s.close()
    lw, le = acc["ww_r"] / acc["ww_l"], acc["ee_r"] / acc["ee_l"]
    lc = (lw + le) / 2
    seam_u = acc["ee_r"] / acc["ww_l"]
    res = {"variant": variant, "rows_used": n, "landcover_right_over_left_west": lw.round(4).tolist(),
           "landcover_right_over_left_east": le.round(4).tolist(),
           "seam_east_right_over_west_left_unharmonised": seam_u.round(4).tolist(),
           "step_unharmonised": (seam_u / lc).round(4).tolist()}
    if g is not None:
        res["west_gains"] = g.round(5).tolist()
        res["step_harmonised"] = (seam_u / g / lc).round(4).tolist()
    (out / "harmonisation_seam_step.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(DIRS), default="corrected")
    ap.add_argument("--gains-from", choices=list(DIRS))
    ap.add_argument("--nominal-strips", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--mosaic", action="store_true")
    ap.add_argument("--seam", action="store_true")
    a = ap.parse_args()
    A5.OUT = DIRS[a.variant]
    A5.OUT.mkdir(parents=True, exist_ok=True)
    if a.nominal_strips:
        assert a.variant == "nominal"
        A5.NOMINAL = True
        A5.run_strip("west")
        A5.run_strip("east")
    if a.estimate:
        estimate(a.variant)
    if a.mosaic:
        src = DIRS[a.gains_from or a.variant] / "harmonisation_gains.json"
        g = json.loads(src.read_text(encoding="utf-8"))["gains_west"]
        print("gains from", src, g, flush=True)
        A5.run_mosaic(gains=g, suffix="_harmonised")
    if a.seam:
        seam(a.variant)
