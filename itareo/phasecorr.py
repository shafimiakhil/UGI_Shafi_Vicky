"""Sub-pixel phase correlation shared by stage A5 (PAN-to-MS registration of Pleiades 2017) and stage A7
(co-registration between layers). Added 2 Nov 2026 after the stage-A review.

phase_corr(a, b) returns the shift of image b relative to image a, in pixels: b(y, x) ~ a(y - sy, x - sx), so a
feature at column x in a lies at column x + sx in b (positive sx = moved toward +x, positive sy = moved toward +rows).

Method:
  - mean removed, 2-D Hann window, normalised cross-power spectrum (phase only);
  - integer peak of its inverse FFT, then sub-pixel refinement to 1/upsample px by the matrix-multiply upsampled DFT
    around that peak (Guizar-Sicairos, Thurman and Fienup 2008, Opt. Lett. 33, 156);
  - peak = height of the refined peak (inverse-DFT normalisation, 1 for identical images);
  - quality = peak / highest competing value, both read on the correlation surface of the cross-power spectrum
    tapered by a Hann window in frequency. The taper turns the sub-pixel peak into a smooth blob, so quality does not
    depend on the fractional part of the shift (on the untapered surface the Dirichlet sidelobes of a half-pixel shift
    cap the ratio near 5). Competing values are taken outside a (2 excl + 1)^2 box around the peak. On unrelated
    textured images quality is about 1.0-1.7 (selftest), so quality >= 2 marks a clear match.

From work_ms/:  python -I -m itareo.phasecorr --selftest
"""
import json
import math

import numpy as np

UPSAMPLE = 20
EXCL = 5


def upsampled_dft(data, size, usfac, offsets):
    """Inverse-DFT values of `data` (frequency domain) on a size x size grid upsampled by usfac, starting at offsets
    (in upsampled pixels), without normalisation."""
    for (n, off) in list(zip(data.shape, offsets))[::-1]:
        kernel = np.exp(-2j * np.pi * (np.arange(size) - off)[:, None] * np.fft.fftfreq(n, usfac)[None, :])
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def phase_corr(a, b, upsample=UPSAMPLE, excl=EXCL, lowpass=None):
    """Shift (sy, sx) of b relative to a in pixels, peak height and quality (see the module docstring).
    a, b: 2-D float arrays of the same shape, finite everywhere.
    lowpass: if given (cycles per pixel), the whitened cross-power spectrum is weighted by a Gaussian of that standard
    deviation in frequency, so the estimate follows the low-frequency content. Used to check a resampled image:
    linear interpolation at a fractional position pulls the high frequencies toward the nearest sample (an under-shift
    for fractions below 0.5, an over-shift above), which biases an unweighted estimate (selftest)."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    cps = fb * np.conj(fa)
    cps /= np.abs(cps) + 1e-12
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    if lowpass:
        cps = cps * np.exp(-0.5 * (fy ** 2 + fx ** 2) / lowpass ** 2)
    r = np.fft.ifft2(cps).real
    py, px = np.unravel_index(np.argmax(r), r.shape)
    rt = np.fft.ifft2(cps * ((0.5 + 0.5 * np.cos(2 * np.pi * fy)) * (0.5 + 0.5 * np.cos(2 * np.pi * fx)))).real
    ty, tx = np.unravel_index(np.argmax(rt), rt.shape)
    keep = np.ones(rt.shape, bool)
    ys = (ty + np.arange(-excl, excl + 1)) % h
    xs = (tx + np.arange(-excl, excl + 1)) % w
    keep[np.ix_(ys, xs)] = False
    second = rt[keep].max() if keep.any() else 0.0
    quality = float(rt[ty, tx] / second) if second > 0 else float("inf")
    py = py - h if py > h // 2 else py
    px = px - w if px > w // 2 else px
    size = int(math.ceil(upsample * 1.5))
    shift0 = size // 2
    up = np.conj(upsampled_dft(np.conj(cps), size, upsample, (shift0 - py * upsample, shift0 - px * upsample))).real
    uy, ux = np.unravel_index(np.argmax(up), up.shape)
    sy = py + (uy - shift0) / upsample
    sx = px + (ux - shift0) / upsample
    return float(sy), float(sx), float(up[uy, ux] / (h * w)), quality


def selftest():
    """Known shifts applied in the Fourier domain to textured random images (several sizes and noise levels):
    recovered shift and sign, quality across fractional shifts, and quality on unrelated image pairs."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    cases = []
    for size, noise in ((512, 0.05), (200, 0.3), (67, 0.3)):
        base = gaussian_filter(rng.normal(size=(size, size)), 2.0) + 0.3 * rng.normal(size=(size, size))
        ky = np.fft.fftfreq(size)[:, None]
        kx = np.fft.fftfreq(size)[None, :]
        for sy, sx in ((0.0, 0.0), (3.0, -7.0), (0.4, 0.25), (-1.3, 2.7), (12.6, -0.5), (0.33, -0.77), (3.5, -7.5)):
            b = np.fft.ifft2(np.fft.fft2(base) * np.exp(-2j * np.pi * (ky * sy + kx * sx))).real
            b += noise * rng.normal(size=b.shape)
            gy, gx, pk, q = phase_corr(base, b)
            cases.append({"size": size, "noise": noise, "true_sy": sy, "true_sx": sx, "est_sy": round(gy, 3),
                          "est_sx": round(gx, 3), "err_px": round(math.hypot(gy - sy, gx - sx), 3),
                          "peak": round(pk, 3), "quality": round(q, 2)})
    unrelated = []
    for size in (67, 200, 667):
        for _ in range(10):
            a = gaussian_filter(rng.normal(size=(size, size)), 2.0) + 0.3 * rng.normal(size=(size, size))
            b = gaussian_filter(rng.normal(size=(size, size)), 2.0) + 0.3 * rng.normal(size=(size, size))
            unrelated.append(round(phase_corr(a, b)[3], 3))
    # bilinear resampling: an exact fractional correction leaves an apparent residual in the unweighted estimate;
    # the low-frequency weighting (lowpass=0.1) removes it
    resampled = []
    n, pad = 256, 3
    for sy, sx in ((-0.88, -0.73), (-0.5, -0.25), (-0.1, -0.6)):
        base = gaussian_filter(rng.normal(size=(n + 2 * pad, n + 2 * pad)), 1.0)
        ky = np.fft.fftfreq(n + 2 * pad)[:, None]
        kx = np.fft.fftfreq(n + 2 * pad)[None, :]
        b = np.fft.ifft2(np.fft.fft2(base) * np.exp(-2j * np.pi * (ky * sy + kx * sx))).real[pad:-pad, pad:-pad]
        fy, fx = -sy - math.floor(-sy), -sx - math.floor(-sx)
        ky0, kx0 = math.floor(-sy), math.floor(-sx)
        s = lambda o: slice(pad + o, pad + o + n)
        corr = ((1 - fy) * ((1 - fx) * base[s(ky0), s(kx0)] + fx * base[s(ky0), s(kx0 + 1)])
                + fy * ((1 - fx) * base[s(ky0 + 1), s(kx0)] + fx * base[s(ky0 + 1), s(kx0 + 1)]))
        r0 = phase_corr(corr, b)
        r1 = phase_corr(corr, b, lowpass=0.1)
        resampled.append({"true_sy": sy, "true_sx": sx, "residual_unweighted_px": [round(r0[0], 3), round(r0[1], 3)],
                          "residual_lowpass_0.1_px": [round(r1[0], 3), round(r1[1], 3)]})
    big = [c for c in cases if c["size"] >= 200]
    res = {"cases": cases, "max_err_px_size_ge_200": max(c["err_px"] for c in big),
           "bilinear_correction_residuals": resampled,
           "max_residual_lowpass_px": max(max(abs(v) for v in r["residual_lowpass_0.1_px"]) for r in resampled),
           "max_err_px_size_67": max(c["err_px"] for c in cases if c["size"] == 67),
           "min_quality_known_shift": min(c["quality"] for c in cases),
           "max_quality_unrelated_pairs": max(unrelated), "unrelated_pairs": unrelated,
           "convention": "b(y, x) = a(y - sy, x - sx): positive sx = second image moved toward +columns (east on a "
                         "north-up grid); positive sy = moved toward +rows (south)"}
    res["passed"] = bool(res["max_err_px_size_ge_200"] < 0.06 and res["max_err_px_size_67"] < 0.2
                         and res["min_quality_known_shift"] > 2.0 and res["max_quality_unrelated_pairs"] < 2.0
                         and res["max_residual_lowpass_px"] < 0.02)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        out = selftest()
        print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))
        assert out["passed"], "phasecorr selftest failed"
