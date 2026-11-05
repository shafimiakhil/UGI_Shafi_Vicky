"""U0.6 (CLAUDE.md 13, 13.1): PlanetScope 10 Sep 2022 (satellite 2483) and 28 Oct 2022 (241a), three frames each, from
raw_data/planetscope/cordoba_psscene_analytic_8b_sr_udm2/PSScene/ (read-only). 24 Apr 2022 is read only from
raw_data/PRISMA/PSScene/ (the copy in raw_data/planetscope/ is not ingested).

  --mosaic     one mosaic per date on the COMMON 3 m lattice (every frame lies on it, offset 0: no resampling); per pixel
               the first frame in time order that is valid in all 8 bands and clear in UDM2 band 1; output 8 bands SR
               x 10000 (nodata 0) + a frame-index layer.  -> derived/planetscope2022/ps2022MMDD_sr_3m.tif, _frame_3m.tif
  --coreg      displacement of each date mosaic against the registered Neo 2022 (mean of 3 x 3 px of GRID_1M) and
               against PlanetScope 24 Apr 2022, on 1 km blocks (333 px) on a 500 m stride; mean of blue, green, red;
               phase correlation with the A7 criteria (valid >= 0.9, stable >= 0.2, quality >= 2).
  --normalise  relative normalisation to the Sentinel-2 SR 2022 quarter median of the date (10 Sep -> Q3, 28 Oct -> Q4):
               per band with an S2 counterpart (blue B2, green B3, red B4, red edge B5, NIR B8A), a linear fit
               S2 = a + b PS on the 30 m lattice that nests both (origin GRID_1M + 3 m: 10 x 10 PS px, 3 x 3 S2 px), cells
               fully valid and clear in both, ordinary least squares with 3-sigma clipping (5 passes); applied to the
               3 m pixels. Coastal blue, green I and yellow have no S2 counterpart and are kept as delivered.
               -> derived/planetscope2022/ps2022MMDD_sr_norm_s2_3m.tif (uint16 x 10000, nodata 0)
  --normalise_apr  the same normalisation for 24 Apr 2022 (raw_data/PRISMA/PSScene/, placed on COMMON by its integer
               offset, no UDM2) against S2 2022 Q2 -> ps20220424_sr_norm_s2_3m.tif, for B4.
  --coverage   valid-and-clear share of the common extent and of the UGI area (UGI 1 m pixels whose 3 m cell is valid).
Run from work_ms/. Log: derived/planetscope2022/u06_log.json.
"""
import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
ROOT = WM.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(WM))
import a7_coregistration as A7          # noqa: E402
from itareo.grid import COMMON           # noqa: E402

SRC = ROOT / "raw_data/planetscope/cordoba_psscene_analytic_8b_sr_udm2/PSScene"
APR = ROOT / "raw_data/PRISMA/PSScene/20220424_135836_31_2446_3B_AnalyticMS_SR_8b_clip.tif"
NEO = WM / "derived/pneo2022_registered/pneo2022_toa_grid1m.tif"
UGI = WM / "derived/ugi_labels/ugi_binary_grid1m.tif"
OUT = WM / "derived/planetscope2022"
DATES = {"20220910": "2022Q3", "20221028": "2022Q4"}
S2_MAP = {2: 1, 4: 2, 6: 3, 7: 4, 8: 8}          # PS band -> S2 band index in the 10-band export (B2 B3 B4 B5 ... B8A=8)
GTIFF = dict(compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 24):
            h.update(c)
    return h.hexdigest()


def log_update(key, val):
    p = OUT / "u06_log.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[key] = val
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def frames(date):
    return sorted(glob.glob(str(SRC / f"{date}_*_3B_AnalyticMS_SR_8b_clip.tif")))


def mosaic():
    OUT.mkdir(parents=True, exist_ok=True)
    for date in DATES:
        fr = frames(date)
        assert len(fr) == 3, fr
        out = np.zeros((8, COMMON.height, COMMON.width), np.uint16)
        fid = np.zeros((COMMON.height, COMMON.width), np.uint8)
        info = []
        for k, p in enumerate(fr, 1):
            udm = p.replace("_AnalyticMS_SR_8b_clip.tif", "_udm2_clip.tif")
            with rasterio.open(p) as s, rasterio.open(udm) as u:
                assert s.crs.to_epsg() == 32720 and s.res == (3.0, 3.0)
                co = (s.transform.c - COMMON.left) / 3.0
                ro = (COMMON.top - s.transform.f) / 3.0
                assert co == int(co) and ro == int(ro), "frame off the COMMON lattice"
                co, ro = int(co), int(ro)
                a = s.read()
                clear = u.read(1) == 1
            valid = (a != 0).all(0) & clear
            r0, c0 = max(ro, 0), max(co, 0)
            r1, c1 = min(ro + a.shape[1], COMMON.height), min(co + a.shape[2], COMMON.width)
            sub = valid[r0 - ro:r1 - ro, c0 - co:c1 - co] & (fid[r0:r1, c0:c1] == 0)
            out[:, r0:r1, c0:c1][:, sub] = a[:, r0 - ro:r1 - ro, c0 - co:c1 - co][:, sub]
            fid[r0:r1, c0:c1][sub] = k
            info.append({"frame": Path(p).name, "sha256_sr": sha256(p), "sha256_udm2": sha256(udm),
                         "pixels_in_frame": int(a.shape[1] * a.shape[2]), "valid_clear": int(valid.sum()),
                         "taken": int(sub.sum())})
        prof = COMMON.profile(count=8, dtype="uint16", nodata=0, **GTIFF)
        names = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
        p_out = OUT / f"ps{date}_sr_3m.tif"
        with rasterio.open(p_out, "w", **prof) as d:
            d.write(out)
            for i, n in enumerate(names, 1):
                d.set_band_description(i, f"{n} SR x 10000 (mosaic of 3 frames, UDM2 clear)")
        with rasterio.open(OUT / f"ps{date}_frame_3m.tif", "w", **COMMON.profile(count=1, dtype="uint8", nodata=0, **GTIFF)) as d:
            d.write(fid, 1)
        log_update(f"mosaic_{date}", {"frames": info, "output": str(p_out.relative_to(WM)), "sha256": sha256(p_out),
                                      "valid_share_of_extent": round(float((fid > 0).mean()), 5),
                                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(date, "valid share", round(float((fid > 0).mean()), 4), flush=True)


def blocks(src_a, src_b, label):
    """1 km blocks (333 px) on a 500 m stride of COMMON; src_*(r0, c0, n) -> (vis, ndvi, extra) arrays."""
    rows = []
    bs, st = 333, 167
    for r0 in range(0, COMMON.height - bs, st):
        for c0 in range(0, COMMON.width - bs, st):
            a = src_a(r0, c0, bs)
            b = src_b(r0, c0, bs)
            if np.isfinite(a[0]).mean() < 0.9 or np.isfinite(b[0]).mean() < 0.9:
                continue
            rows.append(A7.correlate({"pair": label, "block_row": r0, "block_col": c0}, *a, *b, 3.0))
    u = [r for r in rows if r.get("used")]
    if not u:
        return {"blocks": 0}
    dx, dy = np.array([r["dx_m"] for r in u]), np.array([r["dy_m"] for r in u])
    return {"blocks": len(u), "dx_east_m_median": round(float(np.median(dx)), 3), "dy_north_m_median": round(float(np.median(dy)), 3),
            "dx_iqr": [round(float(v), 3) for v in np.percentile(dx, [25, 75])],
            "dy_iqr": [round(float(v), 3) for v in np.percentile(dy, [25, 75])]}


def ps_src(path, nodata=0):
    def f(r0, c0, n):
        a = A7.read_window(path, COMMON, r0, c0, n, n, [2, 4, 6, 8], 1e-4, nodata)
        return (*A7.vis_ndvi(a, [0, 1, 2], 2, 3), np.zeros((n, n), bool))
    return f


def neo3_src(r0, c0, n):
    with rasterio.open(NEO) as s:
        a = s.read([1, 2, 3, 4], window=Window(c0 * 3, r0 * 3, n * 3, n * 3), boundless=True, fill_value=65535).astype(np.float64)
    a[:, (a == 65535).any(0)] = np.nan
    a = a.reshape(4, n, 3, n, 3).mean(axis=(2, 4)) * 1e-4
    return (*A7.vis_ndvi(a, [0, 1, 2], 0, 3), np.zeros((n, n), bool))


def coreg():
    res = {}
    for date in DATES:
        p = OUT / f"ps{date}_sr_3m.tif"
        res[date] = {"vs_neo2022_registered": blocks(neo3_src, ps_src(p), f"neo->ps{date}"),
                     "vs_ps20220424": blocks(ps_src(APR), ps_src(p), f"ps20220424->ps{date}")}
        print(date, json.dumps(res[date]), flush=True)
    res["convention"] = "displacement of the date's content relative to the reference, m east and m north"
    log_update("coregistration", res)


def coreg_apr():
    """Displacement of PlanetScope 24 Apr 2022 (raw_data/PRISMA/PSScene/, read on COMMON by its integer offset) relative to
    the registered Neo, as coreg() measures the other dates. First computed on 2 Nov 2026 by an inline command and
    appended to u06_log.json; this function reproduces it from a script and records whether it matches."""
    d = json.loads((OUT / "u06_log.json").read_text(encoding="utf-8"))
    before = d["coregistration"].get("20220424", {}).get("vs_neo2022_registered")
    r = blocks(neo3_src, ps_src(APR), "neo->ps20220424")
    r["reproduces_previous_entry"] = before is not None and all(r.get(k) == before.get(k) for k in before)
    d["coregistration"]["20220424"] = {"vs_neo2022_registered": r, "source": "u06_planetscope.coreg_apr()"}
    (OUT / "u06_log.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({"before": before, "now": r}, indent=1))


def load_on_common(date):
    """8-band SR of a date on the full COMMON grid (24 Apr: placed by its integer lattice offset, no resampling)."""
    if date != "20220424":
        with rasterio.open(OUT / f"ps{date}_sr_3m.tif") as s:
            return s.read().astype(np.float64), s.profile
    with rasterio.open(APR) as s:
        a = s.read()
        co, ro = int((s.transform.c - COMMON.left) / 3), int((COMMON.top - s.transform.f) / 3)
    full = np.zeros((8, COMMON.height, COMMON.width), np.float64)
    full[:, ro:ro + a.shape[1], co:co + a.shape[2]] = a[:, :COMMON.height - ro, :COMMON.width - co]
    return full, COMMON.profile(count=8, dtype="uint16", nodata=0, **GTIFF)


def normalise(dates=None):
    res = {}
    for date, q in (dates or DATES).items():
        s2p = ROOT / f"gee_exports/2026-10-28_nodata/s2_sr_median_{q}.tif"
        ps, prof = load_on_common(date)
        with rasterio.open(s2p) as s2:
            S2 = s2.read().astype(np.float64)
            s2nd = s2.nodata
        # 30 m lattice from GRID_1M + 3 m: PS px from 1 (10 x 10), S2 px from 1 (3 x 3)
        H30 = min((ps.shape[1] - 1) // 10, (S2.shape[1] - 1) // 3)
        W30 = min((ps.shape[2] - 1) // 10, (S2.shape[2] - 1) // 3)
        pv = ps[:, 1:1 + H30 * 10, 1:1 + W30 * 10]
        okp = (pv != 0).all(0).reshape(H30, 10, W30, 10).all(axis=(1, 3))
        pm = pv.reshape(8, H30, 10, W30, 10).mean(axis=(2, 4))
        sv = S2[:, 1:1 + H30 * 3, 1:1 + W30 * 3]
        oks = (sv != s2nd).all(0).reshape(H30, 3, W30, 3).all(axis=(1, 3))
        sm = sv.reshape(S2.shape[0], H30, 3, W30, 3).mean(axis=(2, 4))
        ok = okp & oks
        fits = {}
        out = ps.copy()
        valid3 = (ps != 0).all(0)
        for b_ps, b_s2 in S2_MAP.items():
            x, y = pm[b_ps - 1][ok], sm[b_s2 - 1][ok]
            keep = np.ones(x.size, bool)
            for _ in range(5):
                bb, aa = np.polyfit(x[keep], y[keep], 1)
                r = y - (aa + bb * x)
                sd = r[keep].std()
                nk = np.abs(r) <= 3 * sd
                if (nk == keep).all():
                    break
                keep = nk
            fits[b_ps] = {"s2_band": b_s2, "gain": round(float(bb), 5), "offset": round(float(aa), 2), "cells": int(keep.sum()),
                          "r2": round(float(1 - np.var(y[keep] - (aa + bb * x[keep])) / np.var(y[keep])), 4)}
            out[b_ps - 1] = np.where(valid3, np.clip(np.rint(aa + bb * ps[b_ps - 1]), 1, 65534), 0)
        p_out = OUT / f"ps{date}_sr_norm_s2_3m.tif"
        with rasterio.open(p_out, "w", **prof) as d:
            d.write(out.astype(np.uint16))
            for i in range(1, 9):
                d.set_band_description(i, f"band {i} SR x 10000" + (f", normalised to S2 {q} band index {S2_MAP[i]}" if i in S2_MAP else ", as delivered"))
        res[date] = {"s2": str(s2p.relative_to(ROOT)), "cells_30m": int(ok.sum()), "fits": fits,
                     "output": str(p_out.relative_to(WM)), "sha256": sha256(p_out)}
        print(date, json.dumps(fits), flush=True)
    log_update("normalisation" if dates is None else "normalisation_" + "_".join(dates), res)


def coverage():
    res = {}
    with rasterio.open(UGI) as u:
        ugi = u.read(1).astype(bool)
    for date in ["20220424"] + list(DATES):
        if date == "20220424":
            with rasterio.open(APR) as s:
                a = s.read(1)
                co, ro = int((s.transform.c - COMMON.left) / 3), int((COMMON.top - s.transform.f) / 3)
            v3 = np.zeros((COMMON.height, COMMON.width), bool)
            v3[ro:ro + a.shape[0], co:co + a.shape[1]] = (a != 0)[:COMMON.height - ro, :COMMON.width - co]
        else:
            with rasterio.open(OUT / f"ps{date}_frame_3m.tif") as s:
                v3 = s.read(1) > 0
        v1 = np.repeat(np.repeat(v3, 3, 0), 3, 1)[:ugi.shape[0], :ugi.shape[1]]
        res[date] = {"valid_clear_share_of_extent": round(float(v3.mean()), 5),
                     "ugi_share_covered": round(float(v1[ugi].mean()), 5)}
    log_update("coverage", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("mosaic", "coreg", "coreg_apr", "normalise", "coverage", "normalise_apr"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.mosaic:
        mosaic()
    if a.coreg:
        coreg()
    if a.coreg_apr:
        coreg_apr()
    if a.normalise:
        normalise()
    if a.coverage:
        coverage()
    if a.normalise_apr:
        normalise({"20220424": "2022Q2"})
