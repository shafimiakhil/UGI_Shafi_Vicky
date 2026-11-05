"""E2 (CLAUDE.md 13.6): the PAN lawn layers on the v3 mosaic. The layers of scripts/u04_pan_lawn.py were finished on 2 Nov, before v3 (3 Nov), and strip 1_3 carries the translation of v1.
Strip 1_3 is therefore recomputed with the correction of v3; the other strips are kept, because v3 equals v1 wherever neither mosaic takes strip 1_3 (derived/pneo2022_registered_v3/compare_v1.json).

  * Strip 1_3: same layers, bands, margins and average warp as u04_pan_lawn, from the raw strip lattice. The quadratic correction of v3 (scripts/u03_neo_register.quad13_model on the low-pass pair table; same
    coefficients as derived/pneo2022_registered_v3/README.md, asserted) is carried into each band's warp by a GCP list (GDAL's polynomial transformer, order 2 for >= 10 GCPs), as the MS GCP VRT of v3 does. A 1 m pixel
    is taken from the band whose core holds the raw PAN row of its centre, found by inverting the correction to first order. Values are kept where the v3 strip id is 1_3, the valid fraction is >= 0.5 and v3 NDVI >= 0.3.
  * Merge: strips 1_2, 1_5, 1_6 from derived/features/pan_lawn_strips/ limited to the pixels whose v3 strip id is theirs, strip 1_3 from the new layer. A pixel that left strip 1_3 for another strip in v3 has no value
    (not recomputed for so few pixels; counted in the log).

  --selftest  (a) the GCP warp against the translated warp of u04_pan_lawn on synthetic data, when the correction is a pure translation; (b) on real data, PAN reflectance warped by this code against Neo v3 (R, G, B, NIR mean), low-pass phase
              correlation on 512 m blocks of strip 1_3: corrected, raw lattice and v1 translation, beside strip 1_2 with its v1 translation (the PAN-MS offset of a strip that needs no recomputation). Criterion, set before the run: the corrected median
              lies within 0.25 m of the strip 1_2 median on both axes. -> docs/evidence_phase2/ugi_seg/selftest_e2/pan_lawn_v3_selftest.json; exit 1 if it fails.
  --strip13   derived/features/pan_lawn_strips_v3/neo2022_pan_lawn_1_3.tif (COG) + log.
  --merge     derived/features/neo2022_v3_pan_lawn_grid1m.tif (COG, same four bands) + neo2022_v3_pan_lawn_log.json.
Refuses existing outputs. Run from work_ms/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import a7_coregistration as A7        # noqa: E402
import u03_neo_register as U          # noqa: E402
import u04_pan_lawn as P              # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402

G_T, G_W, G_H, G_CRS = P.G_T, P.G_W, P.G_H, P.G_CRS
V3 = WM / "derived/pneo2022_registered_v3"
SID_V1 = WM / "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"
SID_V3 = V3 / "pneo2022_strip_id_grid1m.tif"
IMAGE_V3 = V3 / "pneo2022_toa_grid1m.tif"
NDVI_V3 = WM / "derived/features/neo2022_v3_indices_grid1m.tif"
STRIPS_V1 = WM / "derived/features/pan_lawn_strips"
STRIPS_V3 = WM / "derived/features/pan_lawn_strips_v3"
OUT = WM / "derived/features/neo2022_v3_pan_lawn_grid1m.tif"
LOG = WM / "derived/features/neo2022_v3_pan_lawn_log.json"
SELFTEST = WM.parent / "docs/evidence_phase2/ugi_seg/selftest_e2/pan_lawn_v3_selftest.json"
COEF_DX, COEF_DY = [-0.878545, -0.021918, 0.0405, -0.007042], [2.06064, -0.108798, 0.02138, -0.007867]


def v3_correction():
    corrected, info = U.quad13_model(U.lowpass_table())
    assert info["coef_dx_m"] == COEF_DX and info["coef_dy_m"] == COEF_DY, info
    return corrected, info


def band_gcps(Ts, sa, sb, Ws, corrected):
    """GCPs on the band's corners and every 500 rows x 1,500 columns: raw PAN pixel -> corrected lon, lat."""
    cols = np.unique(np.r_[np.arange(0, Ws, 1500), Ws]).astype(float)
    lines = np.unique(np.r_[np.arange(sa, sb, 500), sb]).astype(float)
    C, L = np.meshgrid(cols, lines)
    X, Y = corrected((Ts.c + C * Ts.a).ravel(), (Ts.f + L * Ts.e).ravel())
    return [GroundControlPoint(row=float(l - sa), col=float(c), x=float(x), y=float(y)) for c, l, x, y in zip(C.ravel(), L.ravel(), X, Y)]


def bbox_corrected(Ts, Hs, Ws, corrected):
    to_g = Transformer.from_crs("EPSG:4326", G_CRS, always_xy=True)
    lon, lat = zip(*(Ts * (c, r) for c, r in ((0, 0), (Ws, 0), (0, Hs), (Ws, Hs))))
    X, Y = to_g.transform(*corrected(np.array(lon), np.array(lat)))
    w = P.outer_window(min(X), min(Y), max(X), max(Y))
    c0, r0 = max(int(w.col_off) - 2, 0), max(int(w.row_off) - 2, 0)
    c1, r1 = min(int(w.col_off + w.width) + 2, G_W), min(int(w.row_off + w.height) + 2, G_H)
    return Window(c0, r0, c1 - c0, r1 - r0)


def grid_layers_gcp(read_rows, Ts, Hs, Ws, bw, corrected, layer_fn=P.layers, band=P.BAND, threads=8, only=None):
    """u04_pan_lawn.grid_layers with the correction carried by GCPs. Ts is the raw strip lattice. only: band starts to process (a test). Returns (val, frac, times_assigned)."""
    to_ll = Transformer.from_crs(G_CRS, "EPSG:4326", always_xy=True)
    to_g = Transformer.from_crs("EPSG:4326", G_CRS, always_xy=True)
    h, w = int(bw.height), int(bw.width)
    val = np.full((4, h, w), np.nan, np.float32)
    frac = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.uint8)
    for r0 in range(0, Hs, band):
        if only is not None and r0 not in only:
            continue
        r1 = min(r0 + band, Hs)
        sa, sb = max(r0 - P.MARGIN, 0), min(r1 + P.MARGIN, Hs)
        ra, rb = max(sa - P.HALO, 0), min(sb + P.HALO, Hs)
        L = layer_fn(read_rows(ra, rb))[:, sa - ra:sb - ra]
        gcps = band_gcps(Ts, sa, sb, Ws, corrected)
        lon = np.array([Ts.c, Ts.c + Ws * Ts.a, Ts.c, Ts.c + Ws * Ts.a])
        lat = np.array([Ts.f + r0 * Ts.e, Ts.f + r0 * Ts.e, Ts.f + r1 * Ts.e, Ts.f + r1 * Ts.e])
        X, Y = to_g.transform(*corrected(lon, lat))
        win = P.outer_window(min(X), min(Y), max(X), max(Y))
        c0, rr0 = max(int(win.col_off) - 2, int(bw.col_off)), max(int(win.row_off) - 2, int(bw.row_off))
        c1 = min(int(win.col_off + win.width) + 2, int(bw.col_off + bw.width))
        rr1 = min(int(win.row_off + win.height) + 2, int(bw.row_off + bw.height))
        if c1 <= c0 or rr1 <= rr0:
            continue
        dT = G_T * Affine.translation(c0, rr0)
        dst = np.full((4, rr1 - rr0, c1 - c0), np.nan, np.float32)
        reproject(L, dst, gcps=gcps, src_crs="EPSG:4326", dst_transform=dT, dst_crs=G_CRS, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan, num_threads=threads)
        fr = np.zeros((rr1 - rr0, c1 - c0), np.float32)
        reproject(np.isfinite(L[0]).astype(np.float32), fr, gcps=gcps, src_crs="EPSG:4326", dst_transform=dT, dst_crs=G_CRS, resampling=Resampling.average, num_threads=threads)
        cc, rr = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(rr0, rr1) + 0.5)
        lo, la = to_ll.transform(G_T.c + cc * G_T.a, G_T.f + rr * G_T.e)
        lo, la = np.asarray(lo), np.asarray(la)
        lo_c, la_c = corrected(lo.ravel(), la.ravel())
        raw_lon, raw_lat = lo - (np.asarray(lo_c).reshape(lo.shape) - lo), la - (np.asarray(la_c).reshape(la.shape) - la)     # the correction varies by < 1 mm over its own size
        prow, pcol = (raw_lat - Ts.f) / Ts.e, (raw_lon - Ts.c) / Ts.a
        sel = (prow >= r0) & (prow < r1) & (pcol >= 0) & (pcol < Ws)
        sl = (slice(rr0 - int(bw.row_off), rr1 - int(bw.row_off)), slice(c0 - int(bw.col_off), c1 - int(bw.col_off)))
        v_sub, f_sub, n_sub = val[:, sl[0], sl[1]], frac[sl], count[sl]
        v_sub[:, sel] = dst[:, sel]
        f_sub[sel] = fr[sel]
        n_sub[sel] += 1
    return val, frac, count


def translation(dx, dy):
    return lambda lon, lat: (np.asarray(lon, float) - dx / P.M_PER_DEG_LON, np.asarray(lat, float) - dy / P.M_PER_DEG_LAT)


def reflectance_layers(x):
    return np.stack([x] * 4)


def neo_mean(win):
    with rasterio.open(IMAGE_V3) as s:
        a = s.read([1, 2, 3, 4], window=win).astype(np.float32)
        bad = (a == s.nodata).any(0)
    m = a.mean(0)
    m[bad] = np.nan
    return m


def block_offsets(pan, frac, bw, sid, code, block=512, stride=1024):
    """Displacement of PAN relative to Neo v3 (m east, m north; low-pass and unweighted) on the blocks wholly inside the strip."""
    out = []
    H, W = pan.shape
    for r in range(0, H - block + 1, block):
        for c in range(0, W - block + 1, stride):
            ok = frac[r:r + block, c:c + block] >= 0.5
            if ok.mean() < 0.9 or (sid[r:r + block, c:c + block] == code).mean() < 0.9:
                continue
            b = pan[r:r + block, c:c + block]
            a = neo_mean(Window(int(bw.col_off) + c, int(bw.row_off) + r, block, block))
            oka, okb = np.isfinite(a), np.isfinite(b)
            if oka.mean() < 0.9 or okb.mean() < 0.9:
                continue
            A, B = A7.fill_invalid(a, oka), A7.fill_invalid(b, okb)
            sy, sx, _, q = A7.phase_corr(A, B)
            ly, lx, _, _ = A7.phase_corr(A, B, lowpass=0.1)
            if q >= A7.QUALITY_MIN:
                out.append((round(float(lx), 3), round(float(-ly), 3), round(float(sx), 3), round(float(-sy), 3)))
    return out


def case(strip, corrected, code, step):
    tiles = P.tiles_of(strip)
    Ts, Hs, Ws = P.strip_lattice(tiles, (0.0, 0.0))
    bw = bbox_corrected(Ts, Hs, Ws, corrected)
    read_rows, srcs = P.tile_reader(tiles)
    only = set(list(range(0, Hs, P.BAND))[::step])
    val, frac, n = grid_layers_gcp(read_rows, Ts, Hs, Ws, bw, corrected, reflectance_layers, only=only)
    for s in srcs:
        s.close()
    with rasterio.open(SID_V3) as s:
        sid = s.read(1, window=bw)
    got = block_offsets(val[0], frac, bw, sid, code)
    v = np.array(got) if got else np.zeros((0, 4))
    return {"blocks": len(got), "lowpass_median_m_east_north": [round(float(np.median(v[:, 0])), 3), round(float(np.median(v[:, 1])), 3)] if got else None,
            "lowpass_iqr_east": np.percentile(v[:, 0], [25, 75]).round(3).tolist() if got else None, "lowpass_iqr_north": np.percentile(v[:, 1], [25, 75]).round(3).tolist() if got else None,
            "unweighted_median_m_east_north": [round(float(np.median(v[:, 2])), 3), round(float(np.median(v[:, 3])), 3)] if got else None}


def selftest(path=SELFTEST, step_corrected=4):
    refuse_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}
    # ---- (a) a pure translation by GCPs against the translated lattice of u04_pan_lawn --------------------------------------------------
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    Hs, Ws = 900, 700
    Ts = Affine(2.7e-6, 0, -64.20, 0, -2.7e-6, -31.40)
    field = (0.15 + 0.6 * gaussian_filter(rng.standard_normal((Hs, Ws)), 3)).astype(np.float32)
    field[rng.random(field.shape) < 0.002] = np.nan
    field[400:460, 300:380] = np.nan
    dx, dy = -0.25, -2.0
    fn = translation(dx, dy)
    Tt = Affine(Ts.a, 0, Ts.c - dx / P.M_PER_DEG_LON, 0, Ts.e, Ts.f - dy / P.M_PER_DEG_LAT)
    bw = bbox_corrected(Ts, Hs, Ws, fn)

    def rr(a, b):
        return field[a:b].copy()
    v1, f1, n1 = P.grid_layers(rr, Tt, Hs, Ws, bw, band=64, threads=2)
    v2, f2, n2 = grid_layers_gcp(rr, Ts, Hs, Ws, bw, fn, band=64, threads=2)
    keep = (f1 >= 0.5) & (f2 >= 0.5)
    both = np.isfinite(v1) & np.isfinite(v2) & keep[None]
    res["a_gcp_against_translation"] = {"pixels_kept": int(keep.sum()), "same_assignment": bool(np.array_equal(n1, n2)), "max_times_assigned": int(n2.max()), "same_kept_pixels_share": float((f1 >= 0.5) .__eq__(f2 >= 0.5).mean()),
                                        "max_rel_diff_kept": [float((np.abs(v1[k] - v2[k]) / np.maximum(np.abs(v1[k]), 1e-6))[both[k]].max()) for k in range(4)], "max_abs_diff_frac": float(np.abs(f1 - f2).max())}
    a = res["a_gcp_against_translation"]
    a["passed"] = bool(a["same_assignment"] and a["max_times_assigned"] == 1 and a["same_kept_pixels_share"] > 0.9995 and max(a["max_rel_diff_kept"]) < 0.01)
    # ---- (b) real data ------------------------------------------------------------------------------------------------------------------
    reg = json.loads((WM / "derived/pneo2022_registered/registration.json").read_text(encoding="utf-8"))["applied_displacements"]
    corrected, info = v3_correction()
    res["correction"] = info
    res["b_strip_1_3_corrected"] = case("1_3", corrected, P.CODE["1_3"], step_corrected)
    print("corrected", res["b_strip_1_3_corrected"], round(time.time() - t0), "s", flush=True)
    res["b_strip_1_3_raw_lattice"] = case("1_3", translation(0.0, 0.0), P.CODE["1_3"], 4)
    print("raw", res["b_strip_1_3_raw_lattice"], round(time.time() - t0), "s", flush=True)
    t = reg["1_3"]
    res["b_strip_1_3_v1_translation"] = case("1_3", translation(t["dx_east_m"], t["dy_north_m"]), P.CODE["1_3"], 4)
    print("v1", res["b_strip_1_3_v1_translation"], round(time.time() - t0), "s", flush=True)
    t = reg["1_2"]
    res["b_strip_1_2_v1_translation"] = case("1_2", translation(t["dx_east_m"], t["dy_north_m"]), P.CODE["1_2"], 6)
    print("1_2", res["b_strip_1_2_v1_translation"], round(time.time() - t0), "s", flush=True)
    c, r = res["b_strip_1_3_corrected"], res["b_strip_1_2_v1_translation"]
    res["b_passed"] = bool(c["blocks"] >= 10 and r["blocks"] >= 10 and all(abs(c["lowpass_median_m_east_north"][k] - r["lowpass_median_m_east_north"][k]) <= 0.25 for k in (0, 1)))
    res["criterion"] = "corrected strip 1_3 within 0.25 m of strip 1_2 on both axes (low-pass medians), at least 10 blocks each; set before the run"
    res["passed"] = bool(a["passed"] and res["b_passed"])
    res["seconds"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "correction"}, indent=1))
    return res["passed"]


def strip13():
    t0 = time.time()
    path = STRIPS_V3 / "neo2022_pan_lawn_1_3.tif"
    refuse_existing(path)
    STRIPS_V3.mkdir(parents=True, exist_ok=True)
    corrected, info = v3_correction()
    tiles = P.tiles_of("1_3")
    Ts, Hs, Ws = P.strip_lattice(tiles, (0.0, 0.0))
    bw = bbox_corrected(Ts, Hs, Ws, corrected)
    read_rows, srcs = P.tile_reader(tiles)
    val, frac, count = grid_layers_gcp(read_rows, Ts, Hs, Ws, bw, corrected)
    for s in srcs:
        s.close()
    assert count.max() <= 1, "a 1 m pixel was assigned by two bands"
    with rasterio.open(SID_V3) as s:
        sid = s.read(1, window=bw)
    with rasterio.open(NDVI_V3) as s:
        nd = s.read(1, window=bw)
        veg = (nd != s.nodata) & (nd >= 3000)
    use = (sid == P.CODE["1_3"]) & (frac >= 0.5) & veg & np.isfinite(val).all(0)
    val = np.where(use[None], val, np.nan).astype(np.float32)
    chk = P.write_cog(path, val, G_T * Affine.translation(int(bw.col_off), int(bw.row_off)))
    log = {"strip": "1_3", "correction": info, "tiles": [t["member"] for t in tiles], "strip_lattice_rows_cols": [Hs, Ws], "window_grid1m": [int(bw.col_off), int(bw.row_off), int(bw.width), int(bw.height)],
           "strip_vegetated_pixels": int(((sid == P.CODE["1_3"]) & veg).sum()), "pixels_with_values": int(use.sum()), "output": str(path.relative_to(WM)), **chk, "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (STRIPS_V3 / "neo2022_pan_lawn_1_3_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k not in ("tiles", "correction")}, indent=1), flush=True)


def merge():
    t0 = time.time()
    refuse_existing(OUT, LOG)
    with rasterio.open(SID_V3) as s:
        sid3 = s.read(1)
    with rasterio.open(SID_V1) as s:
        sid1 = s.read(1)
    with rasterio.open(NDVI_V3) as s:
        nd = s.read(1)
        veg = (nd != s.nodata) & (nd >= 3000)
    del nd
    final = np.full((4, G_H, G_W), np.nan, np.float32)
    per = {}
    for strip, code in P.CODE.items():
        with rasterio.open((STRIPS_V3 if strip == "1_3" else STRIPS_V1) / f"neo2022_pan_lawn_{strip}.tif") as s:
            c0, r0 = round((s.transform.c - G_T.c) / G_T.a), round((s.transform.f - G_T.f) / G_T.e)
            a = s.read()
        sub = final[:, r0:r0 + a.shape[1], c0:c0 + a.shape[2]]
        m = np.isfinite(a[0]) & (sid3[r0:r0 + a.shape[1], c0:c0 + a.shape[2]] == code)
        assert not np.isfinite(sub[0][m]).any(), f"strip {strip} overlaps another strip's pixels"
        sub[:, m] = a[:, m]
        per[strip] = {"pixels_with_values": int(m.sum()), "vegetated_pixels_v3": int(((sid3 == code) & veg).sum())}
    has = np.isfinite(final[0])
    moved = sid1 != sid3
    log = {"output": str(OUT.relative_to(WM)), "strips": per, "vegetated_pixels_v3": int(veg.sum()), "pixels_with_values": int(has.sum()), "values_outside_vegetated_pixels": int((has & ~veg).sum()),
           "pixels_that_changed_strip_in_v3": int(moved.sum()), "of_them_vegetated": int((moved & veg).sum()), "of_them_vegetated_without_value": int((moved & veg & ~has).sum()), "of_them_taken_by_1_3": int((moved & (sid3 == P.CODE["1_3"])).sum())}
    log.update(P.write_cog(OUT, final, G_T))
    log.update({"seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    LOG.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-out", help="another output path for a repeat of the self-test; the earlier record stays")
    ap.add_argument("--step-corrected", type=int, default=4, help="every n-th band of strip 1_3 for the corrected case (the first run, n = 4, gave 9 blocks, one short of its criterion)")
    ap.add_argument("--strip13", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest(Path(a.selftest_out) if a.selftest_out else SELFTEST, a.step_corrected) else 1)
    if a.strip13:
        strip13()
    if a.merge:
        merge()
