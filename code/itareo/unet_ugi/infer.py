"""Sliding-window inference with 50 % overlap and raised-cosine blending (CLAUDE.md 12.3 h).

Windows of n x n px start every n/2 px from (region origin - n/2), so every pixel of the region lies in exactly four
windows; the 2-D weight sin^2(pi (i + 0.5) / n) x sin^2(pi (j + 0.5) / n) of these four sums to 1, and the blended
softmax is a probability. The region is processed in bands of rows with the same window lattice, so the result does
not depend on the band height. Pixels outside the raster are read as nodata (input 0 after normalisation).
"""
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from .data import boundary_band, read_padded
from .metrics import NBINS, confusion, prob_bins


def raised_cosine(n):
    t = (np.arange(n) + 0.5) / n
    w = np.sin(np.pi * t) ** 2
    return np.outer(w, w).astype(np.float32)


@torch.no_grad()
def predict_bands(model, image_path, region, mean, sd, n=512, band_rows=2048, batch=16, device="cuda", amp=True):
    """Yields (row0, probs [K, rows, cols] float32 numpy) for region = (row0, col0, rows, cols) in grid pixels."""
    model.eval()
    h = n // 2
    assert band_rows % h == 0
    r_lo, c_lo, R, Cn = region
    wgt = torch.from_numpy(raised_cosine(n)).to(device)
    mean_t = torch.as_tensor(np.asarray(mean, np.float32))[:, None, None]
    sd_t = torch.as_tensor(np.asarray(sd, np.float32))[:, None, None]
    with rasterio.open(image_path) as src:
        nodata = src.nodata
        col_starts = list(range(-h, Cn, h))
        for b0 in range(0, R, band_rows):
            b1 = min(b0 + band_rows, R)
            rows_tot = (b1 - b0) + 2 * h
            cols_tot = Cn + 2 * h + (n if (Cn % h) else 0)
            raw = src.read(window=Window(c_lo - h, r_lo + b0 - h, cols_tot, rows_tot + h), boundless=True,
                           fill_value=nodata)
            ok = torch.from_numpy((raw != nodata).all(0))
            x = torch.where(ok[None], (torch.from_numpy(raw.astype(np.float32)) - mean_t) / sd_t, torch.zeros(()))
            acc = None
            wsum = torch.zeros((rows_tot + h, cols_tot), device=device)
            row_starts = list(range(b0 - h, b1, h))
            jobs = [(rs, cs) for rs in row_starts for cs in col_starts]
            for k in range(0, len(jobs), batch):
                part = jobs[k:k + batch]
                xb = torch.stack([x[:, rs - (b0 - h):rs - (b0 - h) + n, cs + h:cs + h + n] for rs, cs in part]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
                    o = model(xb)
                p = torch.softmax(o.float(), 1) * wgt
                if acc is None:
                    acc = torch.zeros((p.shape[1], rows_tot + h, cols_tot), device=device)
                for (rs, cs), pi in zip(part, p):
                    r = rs - (b0 - h)
                    acc[:, r:r + n, cs + h:cs + h + n] += pi
                    wsum[r:r + n, cs + h:cs + h + n] += wgt
            out = acc[:, h:h + (b1 - b0), h:h + Cn] / wsum[h:h + (b1 - b0), h:h + Cn].clamp(min=1e-12)
            yield b0, out.cpu().numpy()


def evaluate_region(model, image_path, labels_path, region, mean, sd, n_cls, eval_mask_fn=None, strips_path=None,
                    device="cuda", n=512, band_rows=2048):
    """Confusion matrices over a region against the labels (classes 1..K -> 0..K-1; 0 = no label): 'full' = labelled
    and valid pixels, 'eroded' = the same without the 1 px boundary band; per strip id when strips_path is given.
    eval_mask_fn(row0, col0, rows, cols) -> bool array restricts the pixels (e.g. the held-out fold).
    Also returns, for both pixel sets, histograms of each class probability over the pixels of that class (positives)
    and of the other classes (negatives), for one-vs-rest AUROC and AP (metrics.auroc_ap_scores)."""
    r_lo, c_lo, R, Cn = region
    out = {"full": np.zeros((n_cls, n_cls), np.int64), "eroded": np.zeros((n_cls, n_cls), np.int64)}
    hist = {k: {"pos": np.zeros((n_cls, NBINS), np.int64), "neg": np.zeros((n_cls, NBINS), np.int64)}
            for k in out}
    per_strip = {}
    with rasterio.open(labels_path) as lb, rasterio.open(image_path) as im:
        sp = rasterio.open(strips_path) if strips_path else None
        for b0, prob in predict_bands(model, image_path, region, mean, sd, n=n, band_rows=band_rows, device=device):
            rows = prob.shape[1]
            w = Window(c_lo, r_lo + b0, Cn, rows)
            labp = read_padded(lb, w, 1, 0, 1)
            lab = labp[1:-1, 1:-1]
            ok = (lab > 0) & (im.read(window=w, boundless=True, fill_value=im.nodata) != im.nodata).all(0)
            if eval_mask_fn is not None:
                ok &= eval_mask_fn(r_lo + b0, c_lo, rows, Cn)
            band = boundary_band(labp)
            pred = prob.argmax(0)
            ref = lab.astype(np.int64) - 1
            for key, m in (("full", ok), ("eroded", ok & ~band)):
                out[key] += confusion(ref, pred, n_cls, m)
                rm = ref[m]
                for c in range(n_cls):
                    h = np.bincount(rm * NBINS + prob_bins(prob[c][m]), minlength=n_cls * NBINS).reshape(n_cls, NBINS)
                    hist[key]["pos"][c] += h[c]
                    hist[key]["neg"][c] += h.sum(0) - h[c]
            if sp is not None:
                sid = sp.read(1, window=w, boundless=True, fill_value=0)
                for s in np.unique(sid[ok]):
                    d = per_strip.setdefault(int(s), {"full": np.zeros((n_cls, n_cls), np.int64),
                                                      "eroded": np.zeros((n_cls, n_cls), np.int64)})
                    d["full"] += confusion(ref, pred, n_cls, ok & (sid == s))
                    d["eroded"] += confusion(ref, pred, n_cls, ok & ~band & (sid == s))
        if sp is not None:
            sp.close()
    return out, per_strip, hist


def predict_to_raster(model, image_path, out_path, region, mean, sd, n_cls, device="cuda", n=512, band_rows=2048):
    """Writes argmax (band 1, classes 1..K) and class probabilities x 250 (bands 2..K+1, 0..250) as uint8 for the
    region on the image grid; nodata = 255 in every band where the input is invalid (a probability of 0 stays valid)."""
    r_lo, c_lo, R, Cn = region
    with rasterio.open(image_path) as im:
        prof = dict(driver="GTiff", count=n_cls + 1, dtype="uint8", nodata=255, crs=im.crs, width=Cn, height=R,
                    transform=im.window_transform(Window(c_lo, r_lo, Cn, R)), compress="zstd", tiled=True,
                    blockxsize=512, blockysize=512, BIGTIFF="YES")
        with rasterio.open(out_path, "w", **prof) as d:
            d.set_band_description(1, f"argmax class 1..{n_cls} (255 = invalid input)")
            for k in range(n_cls):
                d.set_band_description(k + 2, f"probability of class {k + 1} x 250")
            for b0, prob in predict_bands(model, image_path, region, mean, sd, n=n, band_rows=band_rows, device=device):
                rows = prob.shape[1]
                w = Window(0, b0, Cn, rows)
                ok = (im.read(window=Window(c_lo, r_lo + b0, Cn, rows), boundless=True, fill_value=im.nodata)
                      != im.nodata).all(0)
                arg = np.where(ok, prob.argmax(0) + 1, 255).astype(np.uint8)
                d.write(arg, 1, window=w)
                d.write(np.where(ok[None], np.rint(prob * 250), 255).astype(np.uint8), list(range(2, n_cls + 2)),
                        window=w)
