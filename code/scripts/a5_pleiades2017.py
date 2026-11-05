"""Stage A5 (2 Nov 2026): Pleiades 1A, 12 Feb 2017 -> TOA reflectance, HPF pansharpening at 0.5 m in native
EPSG:32721, mosaic of the two strips, one warp to GRID_1M by averaging (CLAUDE.md Section 11.3 b).
Second version, 2 Nov 2026, after the stage-A review: the PAN-to-MS offset is measured and removed before the fusion
(first-run outputs in archive/2026-11-02/pleiades2017_run1_without_pan_ms_registration/).

Inputs (read-only): raw_data/Pleiades/cba_FCGC600488610 (west strip) and cba_TPP1600676611 (east strip), primary ortho
DIMAP products, PAN 0.5 m (tiles R1C1, R2C1, read through the DIM_*.XML) and MS 2 m (R1C1.TIF), 12-bit DN, radiometric
processing BASIC, EPSG:32721. Special values from the DIMAP: NODATA 0, SATURATED 4095; both are masks here.

1. TOA reflectance (DIMAP: L = DN / GAIN + BIAS, W/m2/sr/um; Band_Solar_Irradiance E, W/m2/um):
   rho = pi * L * d^2 / (E * cos(theta_s)), theta_s = 90 deg - sun elevation at the product centre, d = Earth-Sun
   distance in AU on 12 Feb 2017 (d = 1 - 0.01672 cos(0.9856 (DOY - 4)), DOY 43).
   MS file band order R, G, B, NIR = DIMAP bands B2, B1, B0, B3. GDAL's DIMAP driver labels file band 1 with B0's
   calibration; that labelling is not used. The order is checked against Sentinel-2 (band_order_check_<strip>.csv).
2. PAN-to-MS registration. The geotransforms nest each 2 m MS pixel on 4 x 4 PAN pixels, but the image content of the
   two products is offset by about 3 PAN px in both directions (stage-A review, 2 Nov 2026). The offset is measured on
   an 8 x 8 grid of 256 x 256 MS-px windows: PAN TOA averaged over each 4 x 4 block against a synthetic PAN (least-
   squares combination of the 4 MS bands, refitted once on the aligned MS), phase correlation refined to 1/100 MS px
   (itareo.phasecorr); windows with >= 95 % valid pixels and quality >= 3. The median over the windows is removed by
   sampling the MS at the shifted positions during the bilinear upsampling; the PAN is not resampled, so the fused
   product keeps the PAN geometry. The residual offset is measured again through the same upsampling code on the same
   windows and must be below 0.25 PAN px (median, both axes), with the phase correlation weighted toward low
   frequencies: linear interpolation at a fractional position pulls high frequencies toward the nearest sample (an
   over-shift for the fractions 0.68-0.92 used here), which biases an unweighted estimate (A5-path simulation in
   selftest.json). Tables: pan_ms_registration_<strip>.csv; --registration-check recomputes
   both residual estimates after the run (pan_ms_registration_check.json, pan_ms_registration_residual_<strip>.csv).
3. HPF pansharpening as GRASS i.fusion.hpf (Gangkofner et al. 2008) with its defaults, read from the addon source on
   2 Nov 2026 (constants.py, high_pass_filter.py, i.fusion.hpf.py, branch grass8):
   - resolution ratio 2 m / 0.5 m = 4 -> kernel 9 x 9, all cells -1, centre 80 (center=low, the default), divisor 1;
   - MS upsampled bilinearly to 0.5 m (r.resamp.interp method=bilinear in GRASS); here a separable bilinear at the
     registered positions (step 2);
   - fused_k = MS_up_k + HPF * w_k, w_k = SD(MS_k) / SD(HPF) * 0.50 (modulation=mid, the default, for the 9 x 9
     kernel); population standard deviations over the valid pixels;
   - no second pass (only for ratio > 5.5);
   - default output scaling: quantiles 10, 50, 90 of MS_k and of fused_k; out = (F - q50_F) * s + q50_MS, with
     s = (q50_MS - q10_MS) / (q50_F - q10_F) below q50_F and (q90_MS - q50_MS) / (q90_F - q50_F) above; then clipped to
     the MS band's min and max. Quantiles of fused_k from its histogram at 1e-4 reflectance (the output quantisation);
     quantiles of MS_k with numpy (linear interpolation).
   Masks: a fused pixel is valid when its PAN pixel is valid (not 0, not 4095) and the MS pixel nearest to its sample
   position is valid in that band. Departures from GRASS at invalid pixels (GRASS returns NULL): the HPF next to
   invalid PAN pixels uses the mean of the valid pixels in the 9 x 9 window, and the bilinear weights are renormalised
   over the valid MS neighbours (as GDAL does at nodata). Both are logged: SD(HPF) and w are also given for the GRASS
   null rule (only pixels whose 9 x 9 window is fully valid). Reflectance is computed first; the fusion runs on TOA
   reflectance.
4. Spectral fidelity (Wald consistency): the fused 0.5 m bands degraded to 2 m by the mean of each 4 x 4 block (all 16
   valid) against the MS TOA reflectance sampled at the same registered positions (bilinear, all 4 neighbours valid):
   per-band Pearson r, bias, RMSE, and ERGAS = 100 * (0.5/2) * sqrt(mean_k (RMSE_k / mean_k)^2); before and after the
   output scaling, and for the upsampled MS alone (no HPF) on the same pixels (bilinear-only baseline).
5. Mosaic at 0.5 m in EPSG:32721 on the common lattice of the two strips (the MS and PAN lattices of both strips are
   the same 0.5 m lattice): the east strip (incidence 15.2 deg at the centre) is used inside its footprint shrunk by
   8 MS px (16 m) of its MS footprint, the west strip elsewhere where it is valid, so the seam runs inside the overlap,
   away from the east strip's edge pixels (the fused data are in the PAN geometry, so the distance from their edge
   varies around 16 m); strip-id layer 1 = west, 2 = east, 0 = none. The radiometric step
   between the strips is measured on the overlap (means and quantiles per band) on the scaled HPF products and on the
   unscaled MS TOA products, and logged; it is not corrected.
6. One warp of the mosaic to GRID_1M (EPSG:32720) by averaging (GDAL average), each band with its own nodata mask
   (UNIFIED_SRC_NODATA=NO: a pixel invalid in one band only must not enter that band's average), 4 bands uint16
   reflectance x 10000, nodata 65535; strip id by nearest.
Outputs: work_ms/derived/pleiades2017/ (README.md written after the run).

From work_ms/:  python -I scripts/a5_pleiades2017.py --selftest
                python -I scripts/a5_pleiades2017.py --strip west      (and --strip east, in parallel)
                python -I scripts/a5_pleiades2017.py --mosaic          (after both strips)
"""
import os

os.environ["GDAL_PAM_ENABLED"] = "NO"

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np                                   # noqa: E402
import rasterio                                      # noqa: E402
from rasterio.transform import from_origin           # noqa: E402
from rasterio.warp import Resampling, reproject      # noqa: E402
from rasterio.windows import Window                  # noqa: E402
from scipy.ndimage import binary_erosion, binary_fill_holes, uniform_filter   # noqa: E402

from itareo import PROJECT, W                        # noqa: E402
from itareo.grid import GRID_1M, Grid                # noqa: E402
from itareo.phasecorr import phase_corr              # noqa: E402

RAW = PROJECT / "raw_data" / "Pleiades"
OUT = W / "derived" / "pleiades2017"
NOMINAL = False      # B1c ablation (CLAUDE.md 12.3 c): PAN-MS offset measured and logged but set to zero; OUT is changed
S2_TOA = PROJECT / "gee_exports/2026-10-28_nodata/s2_toa_median_2017Q1.tif"
STRIPS = {
    "west": {"dir": "cba_FCGC600488610", "id": 1,
             "ms": "IMG_PHR1A_MS_002/DIM_PHR1A_MS_201702121419408_ORT_2397688101-002.XML",
             "ms_tif": "IMG_PHR1A_MS_002/IMG_PHR1A_MS_201702121419408_ORT_2397688101-002_R1C1.TIF",
             "pan": "IMG_PHR1A_P_001/DIM_PHR1A_P_201702121419408_ORT_2397688101-001.XML"},
    "east": {"dir": "cba_TPP1600676611", "id": 2,
             "ms": "IMG_PHR1A_MS_002/DIM_PHR1A_MS_201702121419273_ORT_2397689101-002.XML",
             "ms_tif": "IMG_PHR1A_MS_002/IMG_PHR1A_MS_201702121419273_ORT_2397689101-002_R1C1.TIF",
             "pan": "IMG_PHR1A_P_001/DIM_PHR1A_P_201702121419273_ORT_2397689101-001.XML"},
}
MS_FILE_BANDS = ["B2", "B1", "B0", "B3"]            # file bands 1-4 = R, G, B, NIR
BAND_NAMES = ["R (B2)", "G (B1)", "B (B0)", "NIR (B3)"]
DN_NODATA, DN_SAT = 0, 4095
ND16 = 65535
KERNEL, CENTRE, MODULATION = 9, 80, 0.50             # GRASS i.fusion.hpf defaults for ratio 3.5-5.5
BLOCK_MS_ROWS = 512
PAD = 3                                              # MS border for the shifted bilinear (|offset| < 2 MS px)
REG_GRID, REG_WIN, REG_UPSAMPLE = 8, 256, 100        # registration: 8 x 8 windows of 256 x 256 MS px, 1/100 px
REG_MIN_VALID, REG_MIN_QUALITY, REG_MAX_RESIDUAL_PAN_PX = 0.95, 3.0, 0.25
SEAM_MARGIN_MS = 8                                   # the seam lies 8 MS px (16 m) inside the east strip's footprint
GTIFF = dict(driver="GTiff", compress="zstd", zstd_level=9, predictor=2, tiled=True, blockxsize=512, blockysize=512,
             BIGTIFF="YES", num_threads="ALL_CPUS")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def dimap(path):
    """Calibration and geometry from a DIMAP v2 XML."""
    r = ET.parse(path).getroot()
    gain = {e.findtext("BAND_ID"): float(e.findtext("GAIN")) for e in r.iter("Band_Radiance")}
    bias = {e.findtext("BAND_ID"): float(e.findtext("BIAS")) for e in r.iter("Band_Radiance")}
    irr = {e.findtext("BAND_ID"): float(e.findtext("VALUE")) for e in r.iter("Band_Solar_Irradiance")}
    centre = [lv for lv in r.iter("Located_Geometric_Values") if lv.findtext("LOCATION_TYPE") == "Center"][0]
    special = {e.findtext("SPECIAL_VALUE_TEXT"): int(e.findtext("SPECIAL_VALUE_COUNT")) for e in r.iter("Special_Value")}
    return {"gain": gain, "bias": bias, "irradiance": irr, "date": r.findtext(".//IMAGING_DATE"),
            "time": r.findtext(".//IMAGING_TIME"), "sun_elevation": float(centre.findtext(".//SUN_ELEVATION")),
            "sun_azimuth": float(centre.findtext(".//SUN_AZIMUTH")),
            "incidence": float(centre.findtext(".//INCIDENCE_ANGLE")), "special": special,
            "radiometric_processing": r.findtext(".//RADIOMETRIC_PROCESSING")}


def earth_sun_distance(date):
    doy = dt.date.fromisoformat(date).timetuple().tm_yday
    return 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4))), doy


def refl_factor(meta, band, d):
    """rho = DN * factor + offset."""
    c = math.pi * d * d / (meta["irradiance"][band] * math.cos(math.radians(90.0 - meta["sun_elevation"])))
    return c / meta["gain"][band], c * meta["bias"][band]


def grid_of_ds(ds):
    t = ds.transform
    return Grid(f"EPSG:{ds.crs.to_epsg()}", t.c, t.f, t.a, ds.width, ds.height)


def to_u16(a):
    """Reflectance (NaN = invalid) -> uint16 x 10000 with nodata 65535; negatives and > 6.5534 clipped."""
    out = np.full(a.shape, ND16, np.uint16)
    ok = np.isfinite(a)
    out[ok] = np.clip(np.rint(a[ok] * 10000.0), 0, ND16 - 1).astype(np.uint16)
    return out


def _sub_positions(d):
    """For the 4 sub-pixels s of an MS pixel: floor and fraction of the sample position (s + 0.5)/4 - 0.5 + d, and the
    nearest MS pixel, in MS px relative to that MS pixel."""
    out = []
    for s in range(4):
        t = (s + 0.5) / 4.0 - 0.5 + d
        k = math.floor(t)
        out.append((s, k, t - k, math.floor(t + 0.5)))
    return out


def upsample4_bilinear(a, dy=0.0, dx=0.0, pad=PAD):
    """Bilinear upsampling by 4 onto the nested 0.5 m lattice. Output pixel (y, x) samples the MS at row
    (y + 0.5)/4 - 0.5 + dy and column (x + 0.5)/4 - 0.5 + dx (MS pixel-centre indices of the unpadded image), so
    dy = dx = 0 is the plain nested upsampling. NaN-aware: weights renormalised over the valid neighbours, as GDAL does
    at nodata. a: (h + 2 pad, w + 2 pad) with a NaN border of `pad` px. Returns (4h, 4w)."""
    v = np.isfinite(a)
    x = np.where(v, a, 0.0)
    vf = v.astype(np.float64)

    def pass1d(x, vf, axis, d):
        n = x.shape[axis] - 2 * pad
        shape = list(x.shape)
        shape[axis] = 4 * n
        ox, ov = np.empty(shape), np.empty(shape)
        for s, k, f, _ in _sub_positions(d):
            assert -pad <= k and k + 1 <= pad, "offset too large for the padding"
            sl = lambda o: tuple(slice(pad + o, pad + o + n) if ax == axis else slice(None) for ax in range(2))
            idx = tuple(slice(s, None, 4) if ax == axis else slice(None) for ax in range(2))
            ox[idx] = (1.0 - f) * x[sl(k)] + f * x[sl(k + 1)]
            ov[idx] = (1.0 - f) * vf[sl(k)] + f * vf[sl(k + 1)]
        return ox, ov

    x1, v1 = pass1d(x, vf, 0, dy)                  # rows: (4h, w + 2 pad)
    x2, v2 = pass1d(x1, v1, 1, dx)                 # cols: (4h, 4w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v2 > 1e-12, x2 / v2, np.nan)


def nearest4(valid, dy=0.0, dx=0.0, pad=PAD):
    """Validity of the MS pixel nearest to each sample position of upsample4_bilinear. valid: padded bool array."""
    h, w = valid.shape[0] - 2 * pad, valid.shape[1] - 2 * pad
    rows = np.empty((4 * h, w + 2 * pad), bool)
    for s, _, _, kn in _sub_positions(dy):
        rows[s::4] = valid[pad + kn:pad + kn + h]
    out = np.empty((4 * h, 4 * w), bool)
    for s, _, _, kn in _sub_positions(dx):
        out[:, s::4] = rows[:, pad + kn:pad + kn + w]
    return out


def shift_bilinear(img, dy, dx, pad=PAD):
    """out(y, x) = img(y + dy, x + dx) by bilinear interpolation at the same resolution; NaN where any of the 4
    neighbours is invalid or outside the image."""
    h, w = img.shape
    ky, kx = math.floor(dy), math.floor(dx)
    fy, fx = dy - ky, dx - kx
    assert -pad <= ky and ky + 1 <= pad and -pad <= kx and kx + 1 <= pad
    p = np.full((h + 2 * pad, w + 2 * pad), np.nan, np.float64)
    p[pad:-pad, pad:-pad] = img
    s = lambda o, n: slice(pad + o, pad + o + n)
    return ((1 - fy) * ((1 - fx) * p[s(ky, h), s(kx, w)] + fx * p[s(ky, h), s(kx + 1, w)])
            + fy * ((1 - fx) * p[s(ky + 1, h), s(kx, w)] + fx * p[s(ky + 1, h), s(kx + 1, w)]))


class Strip:
    def __init__(self, name):
        self.name = name
        s = STRIPS[name]
        self.dir = RAW / s["dir"]
        self.ms_meta = dimap(self.dir / s["ms"])
        self.pan_meta = dimap(self.dir / s["pan"])
        self.d, self.doy = earth_sun_distance(self.ms_meta["date"])
        self.ms_path = self.dir / s["ms_tif"]
        self.pan_path = self.dir / s["pan"]
        with rasterio.open(self.ms_path) as m, rasterio.open(self.pan_path) as p:
            self.msg, self.pang = grid_of_ds(m), grid_of_ds(p)
            self.ms_profile = m.profile
        g = self.msg
        self.fg = Grid(g.crs, g.left, g.top, 0.5, 4 * g.width, 4 * g.height)      # fused grid, nested in MS
        off = self.fg.offset_in(self.pang)
        assert off is not None and off[0] == 1, f"MS and PAN lattices do not nest: {off}"
        self.pan_col_off, self.pan_row_off = off[1], off[2]                        # PAN pixel of fused (0, 0)
        assert self.msg.offset_in(self.pang) == (4, off[1], off[2]), "each MS pixel must be 4 x 4 PAN pixels"

    def ms_reflectance(self):
        with rasterio.open(self.ms_path) as m:
            dn = m.read()
        refl = np.empty(dn.shape, np.float32)
        stats = {}
        for i, b in enumerate(MS_FILE_BANDS):
            f, o = refl_factor(self.ms_meta, b, self.d)
            bad = (dn[i] == DN_NODATA) | (dn[i] == DN_SAT)
            refl[i] = np.where(bad, np.nan, dn[i] * f + o)
            stats[b] = {"nodata_px": int((dn[i] == DN_NODATA).sum()), "saturated_px": int((dn[i] == DN_SAT).sum()),
                        "factor": f, "offset": o}
        return refl, stats

    def pan_block(self, r0, r1, c0, c1):
        """PAN TOA reflectance on fused-grid rows r0:r1, cols c0:c1 (may extend outside: NaN)."""
        with rasterio.open(self.pan_path) as p:
            win = Window(c0 + self.pan_col_off, r0 + self.pan_row_off, c1 - c0, r1 - r0)
            dn = p.read(1, window=win, boundless=True, fill_value=DN_NODATA)
        f, o = refl_factor(self.pan_meta, "P", self.d)
        bad = (dn == DN_NODATA) | (dn == DN_SAT)
        return np.where(bad, np.nan, dn * f + o), int((dn == DN_SAT).sum())


def hpf(pan):
    """GRASS kernel 9 x 9, -1 everywhere, centre 80, divisor 1 = 81 * (P - box mean); box mean over valid pixels.
    Returns the HPF and a flag for pixels whose 9 x 9 window is fully valid (where GRASS would give a value)."""
    v = np.isfinite(pan)
    p0 = np.where(v, pan, 0.0).astype(np.float64)
    s = uniform_filter(p0, KERNEL, mode="constant", cval=0.0)
    n = uniform_filter(v.astype(np.float64), KERNEL, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = KERNEL * KERNEL * (p0 - s / n)
    h[~v | (n <= 0)] = np.nan
    return h, n > 1.0 - 1e-9


# ------------------------------------------------------------------------------------------------ registration
def _fill(x):
    return np.where(np.isfinite(x), x, np.nanmean(x))


def _synth_fit(p2, bands):
    ok = np.isfinite(p2) & np.all(np.isfinite(bands), 0)
    A = np.column_stack([np.ones(int(ok.sum()))] + [b[ok] for b in bands])
    coef = np.linalg.lstsq(A, p2[ok], rcond=None)[0]
    return coef, float(1.0 - np.var(p2[ok] - A @ coef) / np.var(p2[ok]))


def registration_windows(st):
    H, Wd = st.msg.height, st.msg.width
    return [(int(round(REG_WIN + i * (H - 3 * REG_WIN) / (REG_GRID - 1))),
             int(round(REG_WIN + j * (Wd - 3 * REG_WIN) / (REG_GRID - 1))))
            for i in range(REG_GRID) for j in range(REG_GRID)]


def pan_2m(st, m0, c0, n):
    """PAN TOA averaged over the 4 x 4 PAN pixels of each MS pixel of the window (NaN if any of the 16 is invalid)."""
    pan, _ = st.pan_block(4 * m0, 4 * (m0 + n), 4 * c0, 4 * (c0 + n))
    return pan.reshape(n, 4, n, 4).mean(axis=(1, 3))


def measure_registration(st, refl):
    """Per window: shift (sy, sx) of the PAN content relative to the MS content in MS px (PAN(y, x) ~ MS(y - sy,
    x - sx) on the nominal lattice), from a synthetic PAN refitted once on the aligned MS."""
    rows = []
    for m0, c0 in registration_windows(st):
        ms = refl[:, m0:m0 + REG_WIN, c0:c0 + REG_WIN].astype(np.float64)
        p2 = pan_2m(st, m0, c0, REG_WIN)
        valid = np.isfinite(p2) & np.all(np.isfinite(ms), 0)
        row = {"ms_row": m0, "ms_col": c0, "valid_frac": round(float(valid.mean()), 4)}
        if valid.mean() < REG_MIN_VALID:
            row.update(used=False, note="valid share < 0.95")
            rows.append(row)
            continue
        coef, r2n = _synth_fit(p2, ms)
        syn = coef[0] + np.tensordot(coef[1:], np.nan_to_num(ms), 1)
        syn[~np.all(np.isfinite(ms), 0)] = np.nan
        sy, sx, _, q1 = phase_corr(_fill(syn), _fill(p2), upsample=REG_UPSAMPLE)
        if max(abs(sy), abs(sx)) > 2.0 or q1 < REG_MIN_QUALITY:
            row.update(used=False, note=f"first estimate ({sy:.2f}, {sx:.2f}) MS px, quality {q1:.1f}: not used")
            rows.append(row)
            continue
        aligned = np.stack([shift_bilinear(ms[k], -sy, -sx) for k in range(4)])
        coef2, r2a = _synth_fit(p2, aligned)
        syn2 = coef2[0] + np.tensordot(coef2[1:], np.nan_to_num(ms), 1)
        syn2[~np.all(np.isfinite(ms), 0)] = np.nan
        sy, sx, pk, q = phase_corr(_fill(syn2), _fill(p2), upsample=REG_UPSAMPLE)
        row.update(sy_ms_px=round(sy, 3), sx_ms_px=round(sx, 3), sy_pan_px=round(4 * sy, 3), sx_pan_px=round(4 * sx, 3),
                   peak=round(pk, 4), quality=round(q, 2), r2_synthetic_pan_nominal=round(r2n, 4),
                   r2_synthetic_pan_aligned=round(r2a, 4), used=bool(q >= REG_MIN_QUALITY),
                   note="" if q >= REG_MIN_QUALITY else "quality < 3")
        rows.append(row)
    return rows


REG_RESIDUAL_LOWPASS = 0.1                          # cycles per MS px (itareo.phasecorr lowpass)


def residual_registration(st, msp, rows, dy, dx):
    """Residual shift after the correction, through the production code path: the MS of each used window upsampled
    by upsample4_bilinear at the registered positions, averaged back over 4 x 4, against the PAN 4 x 4 means.
    Two estimates: unweighted phase correlation, and weighted toward low frequencies (lowpass 0.1 cycles per MS px).
    Linear interpolation at a fractional position pulls high frequencies toward the nearest sample, so the unweighted
    estimate of an exactly corrected image is not zero (A5-path simulation in selftest.json); the weighted one is the
    test."""
    for row in rows:
        if not row.get("used"):
            continue
        m0, c0 = row["ms_row"], row["ms_col"]
        up = np.stack([upsample4_bilinear(msp[k, m0:m0 + REG_WIN + 2 * PAD, c0:c0 + REG_WIN + 2 * PAD].astype(np.float64),
                                          dy, dx) for k in range(4)])
        u2 = up.reshape(4, REG_WIN, 4, REG_WIN, 4).mean(axis=(2, 4))
        p2 = pan_2m(st, m0, c0, REG_WIN)
        coef, r2 = _synth_fit(p2, u2)
        syn = coef[0] + np.tensordot(coef[1:], np.nan_to_num(u2), 1)
        syn[~np.all(np.isfinite(u2), 0)] = np.nan
        ry, rx, _, rq = phase_corr(_fill(syn), _fill(p2), upsample=REG_UPSAMPLE)
        ly, lx, _, _ = phase_corr(_fill(syn), _fill(p2), upsample=REG_UPSAMPLE, lowpass=REG_RESIDUAL_LOWPASS)
        row.update(residual_sy_pan_px=round(4 * ry, 3), residual_sx_pan_px=round(4 * rx, 3),
                   residual_lowfreq_sy_pan_px=round(4 * ly, 3), residual_lowfreq_sx_pan_px=round(4 * lx, 3),
                   residual_quality=round(rq, 2), r2_synthetic_pan_corrected=round(r2, 4))


def registration_check():
    """After the strip runs: the residual of the applied correction (both estimates) on the registration windows of
    each strip, recomputed with the offsets logged in strip_<strip>_log.json; writes pan_ms_registration_check.json
    and pan_ms_registration_residual_<strip>.csv. Read-only on the fused products."""
    out = {"started": dt.datetime.now().isoformat(timespec="seconds"), "lowpass_cycles_per_ms_px": REG_RESIDUAL_LOWPASS,
           "max_allowed_pan_px": REG_MAX_RESIDUAL_PAN_PX, "strips": {}}
    for name in STRIPS:
        lg = json.loads((OUT / f"strip_{name}_log.json").read_text(encoding="utf-8"))
        dy = lg["registration"]["applied_ms_sample_offset_ms_px"]["rows"]
        dx = lg["registration"]["applied_ms_sample_offset_ms_px"]["cols"]
        st = Strip(name)
        refl, _ = st.ms_reflectance()
        msp = np.full((4, st.msg.height + 2 * PAD, st.msg.width + 2 * PAD), np.nan, np.float32)
        msp[:, PAD:-PAD, PAD:-PAD] = refl
        with open(OUT / f"pan_ms_registration_{name}.csv", encoding="utf-8") as fh:
            rows = [{"ms_row": int(r["ms_row"]), "ms_col": int(r["ms_col"]), "used": r["used"] == "True"}
                    for r in csv.DictReader(fh)]
        residual_registration(st, msp, rows, dy, dx)
        used = [r for r in rows if r["used"]]
        with open(OUT / f"pan_ms_registration_residual_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in used for k in r)))
            wr.writeheader()
            wr.writerows(used)
        res = {"applied_ms_sample_offset_ms_px": [dy, dx], "windows": len(used),
               "residual_unweighted_pan_px": {"sy": _stats([r["residual_sy_pan_px"] for r in used]),
                                              "sx": _stats([r["residual_sx_pan_px"] for r in used])},
               "residual_lowfreq_pan_px": {"sy": _stats([r["residual_lowfreq_sy_pan_px"] for r in used]),
                                           "sx": _stats([r["residual_lowfreq_sx_pan_px"] for r in used])}}
        lf = res["residual_lowfreq_pan_px"]
        res["passed"] = bool(abs(lf["sy"]["median"]) < REG_MAX_RESIDUAL_PAN_PX
                             and abs(lf["sx"]["median"]) < REG_MAX_RESIDUAL_PAN_PX)
        out["strips"][name] = res
        print(name, json.dumps(res), flush=True)
    out["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / "pan_ms_registration_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    assert all(v["passed"] for v in out["strips"].values()), "residual PAN-to-MS offset too large"


def _stats(v):
    v = np.asarray(v, float)
    return {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
            "p90": round(float(np.percentile(v, 90)), 3), "mad": round(float(np.median(np.abs(v - np.median(v)))), 3)}


def _plane(rows, key):
    x = np.array([r["ms_col"] for r in rows], float)
    y = np.array([r["ms_row"] for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    A = np.column_stack([np.ones_like(x), (x - x.mean()) / 1000.0, (y - y.mean()) / 1000.0])
    c = np.linalg.lstsq(A, v, rcond=None)[0]
    return {"centre": round(float(c[0]), 3), "per_1000_ms_cols": round(float(c[1]), 4),
            "per_1000_ms_rows": round(float(c[2]), 4), "span_over_windows": round(float(np.ptp(A[:, 1:] @ c[1:])), 3),
            "residual_rms": round(float(np.sqrt(np.mean((v - A @ c) ** 2))), 3)}


# ------------------------------------------------------------------------------------------------ checks
def band_order_check(st, refl):
    """Correlations of normalised differences of file bands 1-3 with Sentinel-2 TOA (2017Q1) for every permutation."""
    win_ms = Window(st.msg.width // 2 - 2000, st.msg.height // 2 - 2000, 4000, 4000)
    a = refl[:, win_ms.row_off:win_ms.row_off + 4000, win_ms.col_off:win_ms.col_off + 4000].astype(np.float64)
    tr = from_origin(st.msg.left + win_ms.col_off * 2.0, st.msg.top - win_ms.row_off * 2.0, 2.0, 2.0)
    s2 = {}
    with rasterio.open(S2_TOA) as t:
        idx = {d: i + 1 for i, d in enumerate(t.descriptions)}
        for b in ("B2", "B3", "B4", "B8"):
            dst = np.full(a.shape[1:], np.nan)
            reproject(t.read(idx[b]).astype(np.float64), dst, src_transform=t.transform, src_crs=t.crs,
                      dst_transform=tr, dst_crs=st.msg.crs, resampling=Resampling.bilinear, src_nodata=ND16,
                      dst_nodata=np.nan)
            s2[b] = dst / 10000.0
    f = [uniform_filter(np.nan_to_num(x), 5) for x in a]
    ok = np.all(np.isfinite(a), 0) & np.all([np.isfinite(v) for v in s2.values()], 0)
    rows = []
    for perm in itertools.permutations(("B2", "B3", "B4")):
        sc = []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            nd_f = (f[i] - f[j]) / (f[i] + f[j])
            nd_s = (s2[perm[i]] - s2[perm[j]]) / (s2[perm[i]] + s2[perm[j]])
            sc.append(np.corrcoef(nd_f[ok], nd_s[ok])[0, 1])
        rows.append({"file_bands_1_2_3_as_S2": "|".join(perm), "mean_r_normalised_differences": round(float(np.mean(sc)), 4)})
    rows.append({"file_bands_1_2_3_as_S2": "band 4 vs S2 B8 (r)",
                 "mean_r_normalised_differences": round(float(np.corrcoef(f[3][ok], s2["B8"][ok])[0, 1]), 4)})
    for i, b in enumerate(("R", "G", "B")):
        s2b = {"R": "B4", "G": "B3", "B": "B2"}[b]
        rows.append({"file_bands_1_2_3_as_S2": f"mean TOA file band {i + 1} vs S2 {s2b}",
                     "mean_r_normalised_differences": f"{np.mean(a[i][ok]):.4f} vs {np.mean(s2[s2b][ok]):.4f}"})
    return rows


def fidelity_acc():
    return {k: np.zeros(4) for k in ("n", "sx", "sy", "sxx", "syy", "sxy", "sdd")}


def fidelity_add(acc, fused, ms):
    """fused (4, 4h, 4w) at 0.5 m, ms (4, h, w): 4 x 4 means where all 16 valid, against ms."""
    k, H, Wd = fused.shape
    blk = fused.reshape(k, H // 4, 4, Wd // 4, 4)
    full = np.isfinite(blk).all(axis=(2, 4))
    if not full.any():
        return
    mean = np.nanmean(np.where(np.isfinite(blk), blk, np.nan), axis=(2, 4))
    for i in range(k):
        ok = full[i] & np.isfinite(ms[i])
        if not ok.any():
            continue
        x, y = mean[i][ok].astype(np.float64), ms[i][ok].astype(np.float64)
        acc["n"][i] += x.size
        acc["sx"][i] += x.sum()
        acc["sy"][i] += y.sum()
        acc["sxx"][i] += (x * x).sum()
        acc["syy"][i] += (y * y).sum()
        acc["sxy"][i] += (x * y).sum()
        acc["sdd"][i] += ((x - y) ** 2).sum()


def fidelity_report(acc):
    out = {}
    terms = []
    for i, b in enumerate(BAND_NAMES):
        n = acc["n"][i]
        mx, my = acc["sx"][i] / n, acc["sy"][i] / n
        cov = acc["sxy"][i] / n - mx * my
        vx, vy = acc["sxx"][i] / n - mx * mx, acc["syy"][i] / n - my * my
        rmse = math.sqrt(acc["sdd"][i] / n)
        out[b] = {"pixels_2m": int(n), "pearson_r": round(cov / math.sqrt(vx * vy), 5), "bias": round(mx - my, 6),
                  "rmse": round(rmse, 6), "mean_ms": round(my, 6)}
        terms.append((rmse / my) ** 2)
    out["ERGAS"] = round(100 * (0.5 / 2.0) * math.sqrt(np.mean(terms)), 4)
    return out


def quantiles_from_hist(hist, offset, qs):
    c = np.cumsum(hist)
    n = c[-1]
    return [float((np.searchsorted(c, q * n, "left") - offset) / 10000.0) for q in qs]


# ------------------------------------------------------------------------------------------------ strip
def run_strip(name):
    t0 = time.time()
    st = Strip(name)
    OUT.mkdir(parents=True, exist_ok=True)
    log = {"strip": name, "dir": st.dir.name, "script": "scripts/a5_pleiades2017.py (second version, with PAN-to-MS "
           "registration)", "started": dt.datetime.now().isoformat(timespec="seconds"),
           "earth_sun_distance_au": st.d, "doy": st.doy, "ms_meta": st.ms_meta, "pan_meta": st.pan_meta,
           "ms_grid": st.msg.__dict__, "pan_grid": st.pang.__dict__, "fused_grid": st.fg.__dict__,
           "pan_offset_of_fused_origin_px": [st.pan_col_off, st.pan_row_off],
           "ms_file_band_order": MS_FILE_BANDS, "kernel": KERNEL, "centre": CENTRE, "modulation": MODULATION}
    log["sources_sha256"] = {str(p.relative_to(PROJECT)): sha256(p) for p in
                             sorted(st.dir.glob("IMG_PHR1A_*/IMG_*.TIF")) + sorted(st.dir.glob("IMG_PHR1A_*/DIM_*.XML"))}
    refl, mstats = st.ms_reflectance()
    log["ms_dn_stats"] = mstats
    # band order check against Sentinel-2
    rows = band_order_check(st, refl)
    with open(OUT / f"band_order_check_{name}.csv", "w", encoding="utf-8") as fh:
        fh.write("test,value\n" + "".join(f"{r['file_bands_1_2_3_as_S2']},{r['mean_r_normalised_differences']}\n" for r in rows))
    log["band_order_check"] = rows
    # MS TOA product (nominal geometry, as delivered)
    prof = dict(GTIFF, count=4, dtype="uint16", nodata=ND16, crs=st.msg.crs, transform=st.msg.transform,
                width=st.msg.width, height=st.msg.height)
    with rasterio.open(OUT / f"pleiades2017_{name}_ms_toa_2m.tif", "w", **prof) as d:
        d.write(np.stack([to_u16(b) for b in refl]))
        for i, n in enumerate(BAND_NAMES):
            d.set_band_description(i + 1, f"TOA reflectance x 10000, {n}")
    ms_sd = [float(np.nanstd(b, dtype=np.float64)) for b in refl]
    ms_q = [list(map(float, np.nanquantile(b, [0.1, 0.5, 0.9]))) for b in refl]
    ms_min = [float(np.nanmin(b)) for b in refl]
    ms_max = [float(np.nanmax(b)) for b in refl]
    log.update({"ms_sd": ms_sd, "ms_q10_50_90": ms_q, "ms_min": ms_min, "ms_max": ms_max})

    # PAN-to-MS registration
    reg = measure_registration(st, refl)
    used = [r for r in reg if r.get("used")]
    assert len(used) >= 10, f"only {len(used)} registration windows usable"
    sy = float(np.median([r["sy_ms_px"] for r in used]))
    sx = float(np.median([r["sx_ms_px"] for r in used]))
    DY, DX = (0.0, 0.0) if NOMINAL else (-sy, -sx)   # sample the MS at the nominal position + (DY, DX) MS px
    log["variant"] = "nominal lattice (PAN-MS offset set to zero)" if NOMINAL else "PAN-MS registration applied"
    msp = np.full((4, st.msg.height + 2 * PAD, st.msg.width + 2 * PAD), np.nan, np.float32)
    msp[:, PAD:-PAD, PAD:-PAD] = refl
    msvp = np.zeros(msp.shape, bool)
    msvp[:, PAD:-PAD, PAD:-PAD] = np.isfinite(refl)
    residual_registration(st, msp, reg, DY, DX)
    with open(OUT / f"pan_ms_registration_{name}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in reg for k in r)))
        wr.writeheader()
        wr.writerows(reg)
    res_y = _stats([r["residual_lowfreq_sy_pan_px"] for r in used])
    res_x = _stats([r["residual_lowfreq_sx_pan_px"] for r in used])
    log["registration"] = {
        "windows": len(reg), "windows_used": len(used),
        "shift_of_pan_relative_to_ms_pan_px": {"sy": _stats([r["sy_pan_px"] for r in used]),
                                               "sx": _stats([r["sx_pan_px"] for r in used])},
        "plane_fit_pan_px": {"sy": _plane(used, "sy_pan_px"), "sx": _plane(used, "sx_pan_px")},
        "applied_ms_sample_offset_ms_px": {"rows": DY, "cols": DX},
        "applied_offset_m": {"east": round(DX * 2.0, 3), "north": round(-DY * 2.0, 3),
                             "meaning": "the MS content of a ground point lies this far from the PAN content of the "
                                        "same point on the nominal lattice; removed by resampling the MS"},
        "residual_after_correction_lowfreq_pan_px": {"sy": res_y, "sx": res_x},
        "residual_after_correction_unweighted_pan_px": {"sy": _stats([r["residual_sy_pan_px"] for r in used]),
                                                        "sx": _stats([r["residual_sx_pan_px"] for r in used])},
        "r2_synthetic_pan": {"nominal": _stats([r["r2_synthetic_pan_nominal"] for r in used]),
                             "corrected": _stats([r["r2_synthetic_pan_corrected"] for r in used])},
        "table": f"pan_ms_registration_{name}.csv"}
    assert NOMINAL or (abs(res_y["median"]) < REG_MAX_RESIDUAL_PAN_PX and abs(res_x["median"]) < REG_MAX_RESIDUAL_PAN_PX), \
        f"residual PAN-to-MS offset too large: {res_y} {res_x}"
    print(name, "registration: PAN rel. MS (PAN px) sy", log["registration"]["shift_of_pan_relative_to_ms_pan_px"]["sy"],
          "sx", log["registration"]["shift_of_pan_relative_to_ms_pan_px"]["sx"], "residual", res_y, res_x, flush=True)

    # fidelity reference: the MS at the registered positions (2 m)
    ms_ref = np.stack([shift_bilinear(refl[k].astype(np.float64), DY, DX).astype(np.float32) for k in range(4)])
    H, Wd = st.fg.height, st.fg.width
    blocks = [(m0, min(m0 + BLOCK_MS_ROWS, st.msg.height)) for m0 in range(0, st.msg.height, BLOCK_MS_ROWS)]
    half = KERNEL // 2

    def fused_block(m0, m1, w, with_up=False):
        r0, r1 = 4 * m0, 4 * m1
        pan, nsat = st.pan_block(r0 - half, r1 + half, -half, Wd + half)
        h, full = hpf(pan)
        h, full = h[half:-half, half:-half], full[half:-half, half:-half]
        panv = np.isfinite(pan[half:-half, half:-half])
        F = np.empty((4, r1 - r0, Wd), np.float64)
        U = np.empty((4, r1 - r0, Wd), np.float64) if with_up else None
        for k in range(4):
            up = upsample4_bilinear(msp[k, m0:m1 + 2 * PAD, :].astype(np.float64), DY, DX)
            cont = nearest4(msvp[k, m0:m1 + 2 * PAD, :], DY, DX)
            bad = ~(panv & cont & np.isfinite(up) & np.isfinite(h))
            f = up + (w[k] * h if w is not None else 0.0)
            f[bad] = np.nan
            F[k] = f
            if with_up:
                up[bad] = np.nan
                U[k] = up
        return F, U, h, full, nsat

    # pass 1: SD of the HPF (all valid pixels; and only full 9 x 9 windows, the GRASS null rule)
    n = s = ss = 0.0
    ng = sg = ssg = 0.0
    sat_pan = 0
    for m0, m1 in blocks:
        _, _, h, full, nsat = fused_block(m0, m1, None)
        fin = np.isfinite(h)
        hv = h[fin]
        n += hv.size
        s += hv.sum()
        ss += (hv * hv).sum()
        hg = h[fin & full]
        ng += hg.size
        sg += hg.sum()
        ssg += (hg * hg).sum()
        sat_pan += nsat
    hpf_sd = math.sqrt(ss / n - (s / n) ** 2)
    hpf_sd_g = math.sqrt(ssg / ng - (sg / ng) ** 2)
    w = [ms_sd[k] / hpf_sd * MODULATION for k in range(4)]
    log.update({"hpf_sd": hpf_sd, "hpf_pixels": int(n), "weights_w": w, "pan_saturated_px_in_fused_grid_rows": sat_pan,
                "grass_null_rule": {"hpf_sd_full_windows_only": hpf_sd_g, "hpf_pixels_full_windows": int(ng),
                                    "weights_w": [ms_sd[k] / hpf_sd_g * MODULATION for k in range(4)],
                                    "w_ratio_ours_over_grass": hpf_sd_g / hpf_sd}})
    pprof = dict(GTIFF, count=1, dtype="uint16", nodata=ND16, crs=st.pang.crs, transform=st.pang.transform,
                 width=st.pang.width, height=st.pang.height)
    f_pan, o_pan = refl_factor(st.pan_meta, "P", st.d)
    pan_counts = {"nodata": 0, "saturated": 0}
    with rasterio.open(st.pan_path) as p, rasterio.open(OUT / f"pleiades2017_{name}_pan_toa_05m.tif", "w", **pprof) as d:
        d.set_band_description(1, "TOA reflectance x 10000, PAN")
        for r0 in range(0, st.pang.height, 2048):
            nr = min(2048, st.pang.height - r0)
            dn = p.read(1, window=Window(0, r0, st.pang.width, nr))
            bad = (dn == DN_NODATA) | (dn == DN_SAT)
            pan_counts["nodata"] += int((dn == DN_NODATA).sum())
            pan_counts["saturated"] += int((dn == DN_SAT).sum())
            d.write(to_u16(np.where(bad, np.nan, dn * f_pan + o_pan)), 1, window=Window(0, r0, st.pang.width, nr))
    log["pan_dn_counts"] = pan_counts
    print(name, "pass 1 done", round(time.time() - t0), "s; SD(HPF)", hpf_sd, "w", w, flush=True)

    # pass 2: histograms of the unscaled fusion (quantiles); fidelity before scaling and of the upsampling alone
    OFF = 20000
    hists = [np.zeros(OFF + 70000, np.int64) for _ in range(4)]
    acc0, accb = fidelity_acc(), fidelity_acc()
    for m0, m1 in blocks:
        F, U, _, _, _ = fused_block(m0, m1, w, with_up=True)
        for k in range(4):
            v = F[k][np.isfinite(F[k])]
            iv = np.clip(np.rint(v * 10000).astype(np.int64) + OFF, 0, hists[k].size - 1)
            hists[k] += np.bincount(iv, minlength=hists[k].size)
        fidelity_add(acc0, F, ms_ref[:, m0:m1, :])
        fidelity_add(accb, U, ms_ref[:, m0:m1, :])
    f_q = [quantiles_from_hist(hists[k], OFF, [0.1, 0.5, 0.9]) for k in range(4)]
    log["fused_q10_50_90_before_scaling"] = f_q
    log["fidelity_before_scaling"] = fidelity_report(acc0)
    log["fidelity_bilinear_only"] = fidelity_report(accb)
    log["fidelity_reference"] = ("MS TOA sampled bilinearly at the registered positions (all 4 neighbours valid); "
                                 "fused or upsampled bands averaged over 4 x 4 (all 16 valid); same pixels for all three")
    print(name, "pass 2 done", round(time.time() - t0), "s", log["fidelity_before_scaling"]["ERGAS"],
          "bilinear only", log["fidelity_bilinear_only"]["ERGAS"], flush=True)

    # pass 3: scaled output, written on the fused grid; fidelity after scaling
    scale = []
    for k in range(4):
        qm, qf = ms_q[k], f_q[k]
        lo = (qm[1] - qm[0]) / (qf[1] - qf[0]) if qf[1] != qf[0] and qm[1] != qm[0] else 1.0
        hi = (qm[2] - qm[1]) / (qf[2] - qf[1]) if qf[2] != qf[1] and qm[2] != qm[1] else 1.0
        scale.append((lo, hi))
    log["quantile_scaling_s_low_high"] = scale
    acc1 = fidelity_acc()
    fprof = dict(GTIFF, count=4, dtype="uint16", nodata=ND16, crs=st.fg.crs, transform=st.fg.transform, width=Wd, height=H)
    clipped = np.zeros(4, np.int64)
    with rasterio.open(OUT / f"pleiades2017_{name}_toa_hpf_05m.tif", "w", **fprof) as d:
        for i, n_ in enumerate(BAND_NAMES):
            d.set_band_description(i + 1, f"TOA reflectance x 10000, HPF-pansharpened (PAN geometry), {n_}")
        for m0, m1 in blocks:
            F, _, _, _, _ = fused_block(m0, m1, w)
            for k in range(4):
                f = F[k]
                lo, hi = scale[k]
                q50f, q50m = f_q[k][1], ms_q[k][1]
                g = np.where(f < q50f, (f - q50f) * lo, (f - q50f) * hi) + q50m
                before = np.isfinite(g) & ((g < ms_min[k]) | (g > ms_max[k]))
                clipped[k] += int(before.sum())
                F[k] = np.clip(g, ms_min[k], ms_max[k])
            fidelity_add(acc1, F, ms_ref[:, m0:m1, :])
            d.write(np.stack([to_u16(F[k]) for k in range(4)]), window=Window(0, 4 * m0, Wd, 4 * (m1 - m0)))
    log["clipped_to_ms_min_max_px"] = clipped.tolist()
    log["fidelity_after_scaling"] = fidelity_report(acc1)
    log["seconds"] = round(time.time() - t0, 1)
    log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / f"strip_{name}_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(name, "done", log["seconds"], "s", json.dumps(log["fidelity_after_scaling"]), flush=True)


# ------------------------------------------------------------------------------------------------ mosaic
def east_core():
    """East strip footprint (all 4 MS bands valid, holes filled) shrunk by SEAM_MARGIN_MS MS px, on the east MS grid."""
    with rasterio.open(OUT / "pleiades2017_east_ms_toa_2m.tif") as s:
        v = (s.read() != ND16).all(0)
        g = grid_of_ds(s)
    core = binary_erosion(binary_fill_holes(v), structure=np.ones((3, 3), bool), iterations=SEAM_MARGIN_MS)
    return core, g


def ms_toa_step(offs):
    """Radiometric step between the strips on their unscaled MS TOA products (2 m), where both are valid in all bands:
    the DN-level step, before the per-strip HPF weights and quantile scaling."""
    arr, org = {}, {}
    for n in ("west", "east"):
        with rasterio.open(OUT / f"pleiades2017_{n}_ms_toa_2m.tif") as s:
            arr[n] = s.read()
        c, r = offs[n]
        assert c % 4 == 0 and r % 4 == 0, "MS lattices must coincide at 2 m"
        org[n] = (c // 4, r // 4)
    W2 = max(org[n][0] + arr[n].shape[2] for n in arr)
    H2 = max(org[n][1] + arr[n].shape[1] for n in arr)
    full = {}
    for n in arr:
        f = np.full((4, H2, W2), ND16, np.uint16)
        c, r = org[n]
        f[:, r:r + arr[n].shape[1], c:c + arr[n].shape[2]] = arr[n]
        full[n] = f
    both = (full["west"] != ND16).all(0) & (full["east"] != ND16).all(0)
    out = {"pixels_2m": int(both.sum()), "per_band": {}}
    for k, b in enumerate(BAND_NAMES):
        w, e = full["west"][k][both].astype(np.float64) / 1e4, full["east"][k][both].astype(np.float64) / 1e4
        qw, qe = np.quantile(w, [0.1, 0.5, 0.9]), np.quantile(e, [0.1, 0.5, 0.9])
        out["per_band"][b] = {"mean_west": round(float(w.mean()), 5), "mean_east": round(float(e.mean()), 5),
                              "mean_ratio_west_over_east": round(float(w.mean() / e.mean()), 4),
                              "q10_q50_q90_west": [round(float(x), 4) for x in qw],
                              "q10_q50_q90_east": [round(float(x), 4) for x in qe],
                              "quantile_ratio_west_over_east": [round(float(x / y), 4) for x, y in zip(qw, qe)]}
    return out


def run_mosaic(gains=None, suffix=""):
    """Mosaic of the two fused strips and one warp to GRID_1M. gains: optional per-band multipliers of the west strip
    (B1b harmonisation, CLAUDE.md 12.3 a); outputs then carry `suffix` and the unharmonised products are kept."""
    t0 = time.time()
    srcs = {n: rasterio.open(OUT / f"pleiades2017_{n}_toa_hpf_05m.tif") for n in ("west", "east")}
    grids = {n: grid_of_ds(ds) for n, ds in srcs.items()}
    left = min(g.left for g in grids.values())
    top = max(g.top for g in grids.values())
    right = max(g.left + g.width * g.res for g in grids.values())
    bottom = min(g.top - g.height * g.res for g in grids.values())
    mg = Grid(grids["west"].crs, left, top, 0.5, int(round((right - left) / 0.5)), int(round((top - bottom) / 0.5)))
    offs = {}
    for n, g in grids.items():
        o = g.offset_in(mg)
        assert o is not None and o[0] == 1, f"{n} not on the mosaic lattice"
        offs[n] = (o[1], o[2])
    core, cg = east_core()
    ce, re_ = offs["east"]
    assert cg.offset_in(grids["east"]) == (4, 0, 0), "east MS grid must underlie the east fused grid"
    log = {"mosaic_grid": mg.__dict__, "strip_offsets_px": offs, "seam_margin_ms_px": SEAM_MARGIN_MS,
           "rule": "east where all 4 east bands are valid and the pixel lies inside the east footprint shrunk by "
                   f"{SEAM_MARGIN_MS} MS px (or the west strip is not valid there); else west where all 4 west bands "
                   "are valid; else per band the valid one (east first)",
           "started": dt.datetime.now().isoformat(timespec="seconds"),
           "west_gains": None if gains is None else [float(g) for g in gains], "suffix": suffix}
    prof = dict(GTIFF, count=4, dtype="uint16", nodata=ND16, crs=mg.crs, transform=mg.transform, width=mg.width, height=mg.height)
    iprof = dict(GTIFF, count=1, dtype="uint8", nodata=0, crs=mg.crs, transform=mg.transform, width=mg.width, height=mg.height)
    iprof.pop("predictor")
    cnt = {"east": 0, "west": 0, "both_valid": 0, "both_valid_given_to_west_near_east_edge": 0, "partial": 0, "none": 0}
    hist = {n: np.zeros((4, 65536), np.int64) for n in ("west", "east")}
    sums = {n: np.zeros(4) for n in ("west", "east")}
    mpath = OUT / f"pleiades2017_toa_hpf_05m_mosaic_epsg32721{suffix}.tif"
    ipath = OUT / f"pleiades2017_strip_id_05m_epsg32721{suffix}.tif"
    cols = np.arange(mg.width) - ce
    ms_c = cols // 4
    in_c = (cols >= 0) & (ms_c < core.shape[1])
    with rasterio.open(mpath, "w", **prof) as d, rasterio.open(ipath, "w", **iprof) as di:
        for i, n_ in enumerate(BAND_NAMES):
            d.set_band_description(i + 1, f"TOA reflectance x 10000, HPF-pansharpened, {n_}")
        di.set_band_description(1, "strip id: 1 west cba_FCGC600488610 (incidence 18.7), 2 east cba_TPP1600676611 (15.2), 0 none")
        for r0 in range(0, mg.height, 2048):
            nr = min(2048, mg.height - r0)
            arr = {}
            for n, ds in srcs.items():
                c, r = offs[n]
                arr[n] = ds.read(window=Window(-c, r0 - r, mg.width, nr), boundless=True, fill_value=ND16)
            if gains is not None:
                v = arr["west"] != ND16
                g = np.asarray(gains, np.float64)[:, None, None]
                arr["west"] = np.where(v, np.clip(np.rint(arr["west"] * g), 0, ND16 - 1), ND16).astype(np.uint16)
            rows_ = np.arange(r0, r0 + nr) - re_
            ms_r = rows_ // 4
            in_r = (rows_ >= 0) & (ms_r < core.shape[0])
            cblk = core[np.clip(ms_r, 0, core.shape[0] - 1)][:, np.clip(ms_c, 0, core.shape[1] - 1)]
            cblk &= in_r[:, None] & in_c[None, :]
            ev, wv = arr["east"] != ND16, arr["west"] != ND16
            e_all, w_all = ev.all(0), wv.all(0)
            both = e_all & w_all
            take_e = e_all & (cblk | ~w_all)
            take_w = ~take_e & w_all
            out = np.where(take_e[None], arr["east"], np.where(take_w[None], arr["west"],
                                                               np.where(ev, arr["east"], arr["west"])))
            sid = np.where(take_e, 2, np.where(take_w, 1, np.where(ev.any(0), 2, np.where(wv.any(0), 1, 0)))).astype(np.uint8)
            cnt["east"] += int(take_e.sum())
            cnt["west"] += int(take_w.sum())
            cnt["both_valid"] += int(both.sum())
            cnt["both_valid_given_to_west_near_east_edge"] += int((both & ~cblk).sum())
            cnt["partial"] += int((~e_all & ~w_all & (ev.any(0) | wv.any(0))).sum())
            cnt["none"] += int((~ev.any(0) & ~wv.any(0)).sum())
            if both.any():
                for n in ("west", "east"):
                    for k in range(4):
                        v = arr[n][k][both]
                        hist[n][k] += np.bincount(v, minlength=65536)
                        sums[n][k] += float(v.sum(dtype=np.float64))
            d.write(out, window=Window(0, r0, mg.width, nr))
            di.write(sid, 1, window=Window(0, r0, mg.width, nr))
    for ds in srcs.values():
        ds.close()
    log["pixel_counts_05m"] = cnt
    nb = cnt["both_valid"]
    rad = {}
    for k, b in enumerate(BAND_NAMES):
        q = {}
        for n in ("west", "east"):
            c = np.cumsum(hist[n][k])
            q[n] = [round(float(np.searchsorted(c, f * c[-1], "left")) / 10000.0, 4) for f in (0.1, 0.5, 0.9)]
        mw, me = sums["west"][k] / nb / 10000.0, sums["east"][k] / nb / 10000.0
        rad[b] = {"mean_west": round(mw, 5), "mean_east": round(me, 5), "mean_ratio_west_over_east": round(mw / me, 4),
                  "q10_q50_q90_west": q["west"], "q10_q50_q90_east": q["east"],
                  "quantile_ratio_west_over_east": [round(a / b_, 4) for a, b_ in zip(q["west"], q["east"])]}
    log["overlap_radiometry"] = {"pixels": nb, "per_band": rad,
                                 "note": "both strips valid in all bands, scaled HPF products, " +
                                         ("not corrected" if gains is None else "west multiplied by west_gains")}
    log["overlap_radiometry_ms_toa_2m"] = ms_toa_step(offs)
    print("mosaic done", round(time.time() - t0), "s", cnt, flush=True)

    # one warp to GRID_1M: average for reflectance, nearest for the strip id
    gprof = GRID_1M.profile(count=4, dtype="uint16", nodata=ND16, compress="zstd", zstd_level=9, predictor=2,
                            tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES", num_threads="ALL_CPUS")
    sprof = GRID_1M.profile(count=1, dtype="uint8", nodata=0, compress="zstd", tiled=True, blockxsize=512,
                            blockysize=512, num_threads="ALL_CPUS")
    gpath, spath = OUT / f"pleiades2017_toa_hpf_grid1m{suffix}.tif", OUT / f"pleiades2017_strip_id_grid1m{suffix}.tif"
    valid1m = 0
    with rasterio.open(mpath) as src, rasterio.open(ipath) as isrc, rasterio.open(gpath, "w", **gprof) as d, \
            rasterio.open(spath, "w", **sprof) as ds_:
        for i, n_ in enumerate(BAND_NAMES):
            d.set_band_description(i + 1, f"TOA reflectance x 10000, HPF-pansharpened, averaged to 1 m, {n_}")
        ds_.set_band_description(1, "strip id (nearest): 1 west, 2 east, 0 none")
        for r0 in range(0, GRID_1M.height, 1024):
            nr = min(1024, GRID_1M.height - r0)
            tr = from_origin(GRID_1M.left, GRID_1M.top - r0 * GRID_1M.res, GRID_1M.res, GRID_1M.res)
            buf = np.full((4, nr, GRID_1M.width), np.nan, np.float32)
            reproject(rasterio.band(src, [1, 2, 3, 4]), buf, dst_transform=tr, dst_crs=GRID_1M.crs,
                      resampling=Resampling.average, src_nodata=ND16, dst_nodata=np.nan, num_threads=8,
                      UNIFIED_SRC_NODATA="NO")          # per-band nodata masks (second review, 2 Nov 2026)
            out = np.where(np.isfinite(buf), np.clip(np.rint(buf), 0, ND16 - 1), ND16).astype(np.uint16)
            valid1m += int(np.isfinite(buf).all(0).sum())
            d.write(out, window=Window(0, r0, GRID_1M.width, nr))
            sid = np.zeros((nr, GRID_1M.width), np.uint8)
            reproject(rasterio.band(isrc, 1), sid, dst_transform=tr, dst_crs=GRID_1M.crs, resampling=Resampling.nearest,
                      src_nodata=0, dst_nodata=0, num_threads=8)
            ds_.write(sid, 1, window=Window(0, r0, GRID_1M.width, nr))
    log["grid1m_valid_px_all_bands"] = valid1m
    log["grid1m_valid_share_of_common_extent"] = valid1m / (GRID_1M.width * GRID_1M.height)
    log["outputs_sha256"] = {p.name: sha256(p) for p in (mpath, ipath, gpath, spath)}
    log["seconds"] = round(time.time() - t0, 1)
    log["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (OUT / f"mosaic_log{suffix}.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print("warp done", log["seconds"], "s; valid 1 m share", log["grid1m_valid_share_of_common_extent"], flush=True)


# ------------------------------------------------------------------------------------------------ selftest
def selftest():
    """upsample4_bilinear and shift_bilinear against GDAL bilinear (reproject) on a random 20 x 30 image with a NaN
    hole, for no offset and three fractional offsets (GDAL: source origin moved by the offset); compared where both are
    finite, 3 MS px from the border; plus nearest4 against a direct rounding. Writes selftest.json in OUT."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, (20, 30))
    a[8:10, 12:14] = np.nan
    pad = np.full((20 + 2 * PAD, 30 + 2 * PAD), np.nan)
    pad[PAD:-PAD, PAD:-PAD] = a
    vpad = np.isfinite(pad)
    cases = []
    for dy, dx in ((0.0, 0.0), (0.9, 0.75), (-0.9, -0.65), (0.25, -1.5)):
        src_tr = from_origin(-2.0 * dx, 2.0 * dy, 2.0, 2.0)
        ours = upsample4_bilinear(pad, dy, dx)
        ref = np.full((80, 120), np.nan)
        reproject(a, ref, src_transform=src_tr, src_crs="EPSG:32721", dst_transform=from_origin(0, 0, 0.5, 0.5),
                  dst_crs="EPSG:32721", resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        inner = (slice(12, -12), slice(12, -12))
        o, r = ours[inner], ref[inner]
        both = np.isfinite(o) & np.isfinite(r)
        ours2 = shift_bilinear(a, dy, dx)
        ref2 = np.full((20, 30), np.nan)
        reproject(a, ref2, src_transform=src_tr, src_crs="EPSG:32721", dst_transform=from_origin(0, 0, 2.0, 2.0),
                  dst_crs="EPSG:32721", resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        in2 = (slice(3, -3), slice(3, -3))
        both2 = np.isfinite(ours2[in2]) & np.isfinite(ref2[in2])
        yy, xx = np.mgrid[0:80, 0:120]
        ny = np.floor((yy + 0.5) / 4.0 - 0.5 + dy + 0.5).astype(int)
        nx = np.floor((xx + 0.5) / 4.0 - 0.5 + dx + 0.5).astype(int)
        direct = vpad[np.clip(ny + PAD, 0, vpad.shape[0] - 1), np.clip(nx + PAD, 0, vpad.shape[1] - 1)]
        near = nearest4(vpad, dy, dx)[inner]
        cases.append({"dy": dy, "dx": dx, "upsample_max_abs_diff": float(np.max(np.abs(o[both] - r[both]))),
                      "upsample_pixels_compared": int(both.sum()),
                      "upsample_finite_only_ours": int((np.isfinite(o) & ~np.isfinite(r)).sum()),
                      "upsample_finite_only_ours_outside_invalid_ms_cells": int((np.isfinite(o) & ~np.isfinite(r)
                                                                                 & near).sum()),
                      "upsample_finite_only_gdal": int((~np.isfinite(o) & np.isfinite(r)).sum()),
                      "shift2m_max_abs_diff": float(np.max(np.abs(ours2[in2][both2] - ref2[in2][both2]))),
                      "shift2m_pixels_compared": int(both2.sum()),
                      "nearest4_mismatches": int((nearest4(vpad, dy, dx) != direct).sum())})
    # A5 path: an image shifted by a known fractional offset, corrected exactly through upsample4_bilinear and the 4 x 4
    # mean; the residual estimated unweighted and weighted toward low frequencies (the registration test)
    from scipy.ndimage import gaussian_filter
    sim = []
    n, rng2 = 256, np.random.default_rng(5)
    for dy, dx in ((0.88, 0.73), (0.92, 0.68)):
        base = gaussian_filter(rng2.normal(size=(n + 2 * PAD, n + 2 * PAD)), 1.0)
        ky = np.fft.fftfreq(n + 2 * PAD)[:, None]
        kx = np.fft.fftfreq(n + 2 * PAD)[None, :]
        pan2 = np.fft.ifft2(np.fft.fft2(base) * np.exp(2j * np.pi * (ky * dy + kx * dx))).real[PAD:-PAD, PAD:-PAD]
        u2 = upsample4_bilinear(base, dy, dx).reshape(n, 4, n, 4).mean(axis=(1, 3))
        r0 = phase_corr(u2, pan2, upsample=REG_UPSAMPLE)
        r1 = phase_corr(u2, pan2, upsample=REG_UPSAMPLE, lowpass=REG_RESIDUAL_LOWPASS)
        sim.append({"offset_ms_px": [dy, dx], "residual_unweighted_pan_px": [round(4 * r0[0], 3), round(4 * r0[1], 3)],
                    "residual_lowfreq_pan_px": [round(4 * r1[0], 3), round(4 * r1[1], 3)]})
    res = {"cases": cases, "max_abs_diff": max(max(c["upsample_max_abs_diff"], c["shift2m_max_abs_diff"]) for c in cases),
           "a5_path_exact_correction": sim,
           "nearest4_mismatches": sum(c["nearest4_mismatches"] for c in cases),
           "finished": dt.datetime.now().isoformat(timespec="seconds")}
    res["passed"] = bool(res["max_abs_diff"] < 1e-12 and res["nearest4_mismatches"] == 0
                         and all(max(abs(v) for v in x["residual_lowfreq_pan_px"]) < 0.1 for x in sim)
                         and all(c["upsample_finite_only_ours_outside_invalid_ms_cells"] == 0
                                 and c["upsample_finite_only_gdal"] == 0 for c in cases))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selftest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    assert res["passed"], "a5 selftest failed"
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", choices=list(STRIPS))
    ap.add_argument("--mosaic", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--registration-check", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.strip:
        run_strip(a.strip)
    if a.mosaic:
        run_mosaic()
    if a.registration_check:
        registration_check()
