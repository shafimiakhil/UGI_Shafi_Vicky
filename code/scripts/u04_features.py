"""U0.4 (CLAUDE.md 13): feature layers on native grids (no resampling here). Texture is computed in the data loader.

  --s2    Sentinel-2 SR quarterly medians 2022 (gee_exports/2026-10-28_nodata/s2_sr_median_2022Q{1-4}.tif, 10 m lattice):
          per quarter NDVI = (B8 - B4)/(B8 + B4), NDRE = (B8A - B5)/(B8A + B5), NDMI = (B8A - B11)/(B8A + B11); dry-season
          NDVI minimum = min(Q2, Q3) (Apr-Sep, the dry season of Córdoba); NDVI amplitude = max - min over Q1-Q4.
          -> derived/features/s2_2022_indices_10m.tif, int16 x 10000, nodata -32768 (14 bands, described)
  --s2-2021  the same, same band order, for the 2021 quarterly medians (gee_exports/2026-11-04_s2o112_2021/s2_sr_median_2021Q{1-4}.tif; "o112" in the folder name is a misnomer, CLAUDE.md 13.8)
          -> derived/features/s2_2021_indices_10m.tif. Refuses an existing output.
  --csk   COSMO-SkyMed 28 Jul 2025 GTC HH (work/raw_data/CSK, the extracted copy of Luigi's zip; native 2.5 m,
          EPSG:32720): uncalibrated log amplitude 20 log10(DN) dB (DN 0 = nodata) and its local mean and standard
          deviation in 5 x 5 and 11 x 11 windows (12.5 m, 27.5 m; valid pixels only).
          -> derived/features/csk_20250728_localstats_2p5m.tif, float32, nodata NaN (5 bands). Relative values: no
          calibration constant is applied, so only differences and local statistics are meaningful.
  --neo   Pléiades Neo 2022 registered mosaic (derived/pneo2022_registered/pneo2022_toa_grid1m.tif, GRID_1M):
          NDVI = (NIR - R)/(NIR + R), NDRE = (NIR - RE)/(NIR + RE), NDWI = (G - NIR)/(G + NIR).
          -> derived/features/neo2022_indices_grid1m.tif, int16 x 10000, nodata -32768 (3 bands)
Each output gets a README entry in derived/features/README.md (written by hand from features_log.json).
Run from work_ms/.
"""
import argparse
import glob
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

WM = Path(__file__).resolve().parents[1]
ROOT = WM.parent
OUT = WM / "derived/features"
ND16 = -32768


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 24):
            h.update(c)
    return h.hexdigest()


def nd(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def to_i16(x):
    return np.where(np.isfinite(x), np.clip(np.rint(x * 10000), -10000, 10000), ND16).astype(np.int16)


def log_entry(key, entry):
    p = OUT / "features_log.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[key] = entry
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")


def s2(year=2022, folder="2026-10-28_nodata", key="s2"):
    t0 = time.time()
    names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    q = {}
    srcs = {}
    for k in (1, 2, 3, 4):
        p = ROOT / f"gee_exports/{folder}/s2_sr_median_{year}Q{k}.tif"
        with rasterio.open(p) as s:
            assert list(s.descriptions) == names
            a = s.read().astype(np.float64)
            prof = s.profile
            bad = (a == s.nodata).any(0)
        a[:, bad] = np.nan
        b = dict(zip(names, a))
        q[k] = {"NDVI": nd(b["B8"], b["B4"]), "NDRE": nd(b["B8A"], b["B5"]), "NDMI": nd(b["B8A"], b["B11"])}
        srcs[p.name] = sha256(p)
    layers, desc = [], []
    for k in (1, 2, 3, 4):
        for n in ("NDVI", "NDRE", "NDMI"):
            layers.append(q[k][n])
            desc.append(f"{n} {year}Q{k} x 10000")
    nd_all = np.stack([q[k]["NDVI"] for k in (1, 2, 3, 4)])
    layers.append(np.fmin(q[2]["NDVI"], q[3]["NDVI"]))
    desc.append(f"NDVI dry-season minimum min(Q2, Q3) {year} x 10000")
    with np.errstate(invalid="ignore"):
        amp = np.nanmax(nd_all, 0) - np.nanmin(nd_all, 0)
    amp[np.isnan(nd_all).any(0)] = np.nan
    layers.append(amp)
    desc.append(f"NDVI amplitude max - min over {year} Q1-Q4 x 10000")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"s2_{year}_indices_10m.tif"
    if year != 2022 and out.exists():
        raise SystemExit(f"{out} exists")
    prof.update(count=len(layers), dtype="int16", nodata=ND16, compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **prof) as d:
        d.write(np.stack([to_i16(x) for x in layers]))
        for i, t in enumerate(desc, 1):
            d.set_band_description(i, t)
    valid = [float(np.isfinite(x).mean()) for x in layers]
    log_entry(key, {"output": str(out.relative_to(WM)), "sha256": sha256(out), "sources_sha256": srcs, "bands": desc,
                     "valid_share_per_band": [round(v, 4) for v in valid], "seconds": round(time.time() - t0, 1),
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("s2", out, [round(v, 4) for v in valid])


def csk():
    t0 = time.time()
    src = Path(glob.glob(str(ROOT / "work/raw_data/CSK/*SBI.tif"))[0])
    with rasterio.open(src) as s:
        dn = s.read(1).astype(np.float64)
        prof = s.profile
    ok = dn > 0
    db = np.full(dn.shape, np.nan)
    db[ok] = 20 * np.log10(dn[ok])
    layers, desc = [db], ["log amplitude 20 log10(DN) dB, uncalibrated"]
    w = ok.astype(np.float64)
    x = np.where(ok, db, 0.0)
    for k in (5, 11):
        n = uniform_filter(w, k, mode="constant")
        m1 = uniform_filter(x, k, mode="constant")
        m2 = uniform_filter(x * x, k, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = m1 / n
            sd = np.sqrt(np.maximum(m2 / n - mean * mean, 0))
        enough = n * k * k >= 0.5 * k * k
        mean[~(enough & ok)] = np.nan
        sd[~(enough & ok)] = np.nan
        layers += [mean, sd]
        desc += [f"local mean of dB, {k} x {k} px ({k * 2.5:g} m), valid pixels >= 50 %",
                 f"local SD of dB, {k} x {k} px ({k * 2.5:g} m), valid pixels >= 50 %"]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "csk_20250728_localstats_2p5m.tif"
    prof.update(count=len(layers), dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="YES")
    with rasterio.open(out, "w", **prof) as d:
        d.write(np.stack(layers).astype(np.float32))
        for i, t in enumerate(desc, 1):
            d.set_band_description(i, t)
    log_entry("csk", {"output": str(out.relative_to(WM)), "sha256": sha256(out),
                      "source": str(src.relative_to(ROOT)), "source_sha256": sha256(src), "bands": desc,
                      "valid_share": round(float(ok.mean()), 4), "grid": {"crs": str(prof["crs"]),
                      "transform": list(prof["transform"])[:6], "shape": [prof["height"], prof["width"]]},
                      "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("csk", out, round(float(ok.mean()), 4))


def neo(src=None, out=None, key="neo"):
    """Neo indices of a registered mosaic; v1 by default; --neo-v3 for the v3 mosaic (CLAUDE.md 13.3 F2, a new file)."""
    t0 = time.time()
    src = src or WM / "derived/pneo2022_registered/pneo2022_toa_grid1m.tif"
    OUT.mkdir(parents=True, exist_ok=True)
    out = out or OUT / "neo2022_indices_grid1m.tif"
    assert not out.exists(), f"{out} exists: never overwritten"
    with rasterio.open(src) as s:
        prof = s.profile
        prof.update(count=3, dtype="int16", nodata=ND16, predictor=2)
        with rasterio.open(out, "w", **prof) as d:
            for i, t in enumerate(["NDVI (NIR - R)/(NIR + R) x 10000", "NDRE (NIR - RE)/(NIR + RE) x 10000",
                                   "NDWI (G - NIR)/(G + NIR) x 10000"], 1):
                d.set_band_description(i, t)
            for r0 in range(0, s.height, 2048):
                w = Window(0, r0, s.width, min(2048, s.height - r0))
                a = s.read([1, 2, 4, 5], window=w).astype(np.float64)      # R, G, NIR, RE
                bad = (a == s.nodata).any(0)
                a[:, bad] = np.nan
                r, g, nir, re = a
                d.write(np.stack([to_i16(nd(nir, r)), to_i16(nd(nir, re)), to_i16(nd(g, nir))]), window=w)
    log_entry(key, {"output": str(out.relative_to(WM)), "sha256": sha256(out), "source": str(src.relative_to(WM)),
                      "source_sha256": sha256(src), "seconds": round(time.time() - t0, 1),
                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("neo", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2", action="store_true")
    ap.add_argument("--s2-2021", action="store_true")
    ap.add_argument("--csk", action="store_true")
    ap.add_argument("--neo", action="store_true")
    ap.add_argument("--neo-v3", action="store_true")
    a = ap.parse_args()
    if a.s2:
        s2()
    if a.s2_2021:
        s2(2021, "2026-11-04_s2o112_2021", "s2_2021")
    if a.csk:
        csk()
    if a.neo:
        neo()
    if a.neo_v3:
        neo(WM / "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif", OUT / "neo2022_v3_indices_grid1m.tif", "neo_v3")
