"""Held-out evaluation of a UGI model (design note, Evaluation): pixel, object, type and boundary level.

Inference: windows of n px every n/2 px with raised-cosine weights (as itareo.unet_ugi.infer); the blended outputs are
p = sigmoid(channel 0) for UGI and softmax(channels 1-5) for the groups. Pixels: valid Neo inside the held-out quadrant.
  * pixel: at the threshold fixed on internal validation, IoU, F1, precision, recall; AUROC and AP from histograms of
    p (itareo.unet_ugi.metrics.auroc_ap_from_hist, logit bins); on the full reference and without the 1 m edge band;
  * object: inventory polygons with >= 50 % of their pixels inside the evaluated pixels; a polygon is detected when
    >= 50 % of its evaluated pixels are predicted UGI; recall overall, by group and by size (< 500, 500-2,000,
    2,000-10,000, > 10,000 m2, from derived/ugi_labels/polygons.csv);
  * type: over detected polygons with a defined group, the predicted group is the argmax of the mean group
    probabilities over the polygon's pixels; accuracy and macro F1 over the groups present;
  * boundary: F-score within 2 m between the boundary pixels of the predicted and of the reference UGI mask
    (pixels with a 4-neighbour of the other value, inside the evaluated area), from Euclidean distance transforms.
"""
import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

from itareo.unet_ugi.infer import raised_cosine
from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins

SIZE_EDGES = [0, 500, 2000, 10000, np.inf]
SIZE_NAMES = ["<500", "500-2000", "2000-10000", ">10000"]
GROUP_NAMES = ["Road greens", "Squares", "Parks and green areas", "Water corridors", "Rail verges"]


@torch.no_grad()
def predict_region(model, image, region, mean, sd, n=512, band_rows=2048, batch=16, device="cuda", extra=None):
    """Yields (row0 within region, p_ugi [rows, cols], p_grp [5, rows, cols]) for region = (row0, col0, rows, cols). extra(row0, col0), if
    given, returns the second input of a fusion model for the window whose corner is (row0, col0) of the grid."""
    model.eval()
    h = n // 2
    r_lo, c_lo, R, Cn = region
    wgt = torch.from_numpy(raised_cosine(n)).to(device)
    mean_t = torch.as_tensor(np.asarray(mean, np.float32))[:, None, None]
    sd_t = torch.as_tensor(np.asarray(sd, np.float32))[:, None, None]
    with rasterio.open(image) as src:
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
            acc = torch.zeros((6, rows_tot + h, cols_tot), device=device)
            wsum = torch.zeros((rows_tot + h, cols_tot), device=device)
            jobs = [(rs, cs) for rs in range(b0 - h, b1, h) for cs in col_starts]
            for k in range(0, len(jobs), batch):
                part = jobs[k:k + batch]
                xb = torch.stack([x[:, rs - (b0 - h):rs - (b0 - h) + n, cs + h:cs + h + n] for rs, cs in part]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    o = (model(xb) if extra is None else model(xb, torch.stack([extra(r_lo + rs, c_lo + cs) for rs, cs in part]).to(device))).float()
                p = torch.cat([torch.sigmoid(o[:, :1]), torch.softmax(o[:, 1:], 1)], 1) * wgt
                for (rs, cs), pi in zip(part, p):
                    r = rs - (b0 - h)
                    acc[:, r:r + n, cs + h:cs + h + n] += pi
                    wsum[r:r + n, cs + h:cs + h + n] += wgt
            out = (acc[:, h:h + (b1 - b0), h:h + Cn] / wsum[h:h + (b1 - b0), h:h + Cn].clamp(min=1e-12)).cpu().numpy()
            yield b0, out[0], out[1:]


def boundary(mask, valid):
    b = np.zeros_like(mask)
    for ax in (0, 1):
        for d in (1, -1):
            sh = np.roll(mask, d, axis=ax)
            vv = np.roll(valid, d, axis=ax)
            b |= (sh != mask) & vv
    b[0, :] = b[-1, :] = False
    b[:, 0] = b[:, -1] = False
    return b & valid


def boundary_f(pred, ref, valid, tol=2.0):
    bp, br = boundary(pred, valid), boundary(ref, valid)
    if not bp.any() or not br.any():
        return {"boundary_precision": None, "boundary_recall": None, "boundary_f": None}
    d_to_ref = distance_transform_edt(~br)
    d_to_pred = distance_transform_edt(~bp)
    p = float((d_to_ref[bp] <= tol).mean())
    r = float((d_to_pred[br] <= tol).mean())
    return {"boundary_precision": round(p, 5), "boundary_recall": round(r, 5),
            "boundary_f": round(2 * p * r / (p + r), 5) if p + r else 0.0, "tolerance_m": tol}


def pixel_scores(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"iou": round(tp / (tp + fp + fn), 5) if tp + fp + fn else None, "f1": round(2 * p * r / (p + r), 5) if p + r else 0.0,
            "precision": round(p, 5), "recall": round(r, 5), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def evaluate(model, image, labels_dir, region, mask_fn, mean, sd, threshold, polygons_csv, device="cuda", n=512,
             band_rows=2048):
    r_lo, c_lo, R, Cn = region
    lab = Path(labels_dir)
    pred = np.zeros((R, Cn), bool)
    refm = np.zeros((R, Cn), bool)
    valid = np.zeros((R, Cn), bool)
    conf = {"full": np.zeros(4, np.int64), "no_edge": np.zeros(4, np.int64)}
    hist = {k: {"pos": np.zeros(NBINS, np.int64), "neg": np.zeros(NBINS, np.int64)} for k in conf}
    nid = 4000
    poly_px = np.zeros(nid, np.int64)
    poly_pos = np.zeros(nid, np.int64)
    poly_g = np.zeros((nid, 5))
    with rasterio.open(image) as im, rasterio.open(lab / "ugi_binary_grid1m.tif") as ub, \
            rasterio.open(lab / "ugi_edge_grid1m.tif") as ue, rasterio.open(lab / "ugi_id_grid1m.tif") as ui:
        for b0, pu, pg in predict_region(model, image, region, mean, sd, n=n, band_rows=band_rows, device=device):
            rows = pu.shape[0]
            w = Window(c_lo, r_lo + b0, Cn, rows)
            ok = (im.read(window=w) != im.nodata).all(0) & mask_fn(r_lo + b0, c_lo, rows, Cn)
            ref = ub.read(1, window=w).astype(bool)
            edge = ue.read(1, window=w).astype(bool)
            ids = ui.read(1, window=w).astype(np.int64)
            pp = pu >= threshold
            pred[b0:b0 + rows], refm[b0:b0 + rows], valid[b0:b0 + rows] = pp & ok, ref & ok, ok
            for key, m in (("full", ok), ("no_edge", ok & ~edge)):
                t, pr = ref[m], pp[m]
                conf[key] += [np.sum(t & pr), np.sum(~t & pr), np.sum(t & ~pr), np.sum(~t & ~pr)]
                bins = prob_bins(pu[m])
                hist[key]["pos"] += np.bincount(bins[t], minlength=NBINS)
                hist[key]["neg"] += np.bincount(bins[~t], minlength=NBINS)
            sel = ok & (ids > 0)
            ii = ids[sel]
            poly_px += np.bincount(ii, minlength=nid)
            poly_pos += np.bincount(ii, weights=pp[sel], minlength=nid).astype(np.int64)
            for g in range(5):
                poly_g[:, g] += np.bincount(ii, weights=pg[g][sel], minlength=nid)
    res = {}
    for key in conf:
        s = pixel_scores(*conf[key])
        a, ap = auroc_ap_from_hist(hist[key]["pos"], hist[key]["neg"])
        s.update({"auroc": None if a is None else round(a, 5), "ap": None if ap is None else round(ap, 5)})
        res[f"pixel_{key}"] = s
    res.update(boundary_f(pred, refm, valid))
    # objects
    polys = {int(r["ID_NUEVO"]): r for r in csv.DictReader(open(polygons_csv, encoding="utf-8"))}
    rows_obj = []
    for pid, r in polys.items():
        total = int(r["pixels"])
        if total == 0 or poly_px[pid] < 0.5 * total:
            continue
        cov = poly_pos[pid] / poly_px[pid]
        grp = int(r["group"])
        rows_obj.append({"id": pid, "group": grp, "area_m2": float(r["area_m2"]), "coverage": round(float(cov), 4),
                         "detected": bool(cov >= 0.5),
                         "pred_group": int(np.argmax(poly_g[pid]) + 1) if poly_px[pid] else 0})
    det = np.array([o["detected"] for o in rows_obj])
    res["objects_evaluated"] = len(rows_obj)
    res["object_recall"] = round(float(det.mean()), 5) if len(det) else None
    res["object_recall_by_group"] = {}
    for g, name in enumerate(GROUP_NAMES + ["NO DEFINIDO"], start=1):
        d = [o["detected"] for o in rows_obj if o["group"] == g]
        res["object_recall_by_group"][name] = {"n": len(d), "recall": round(float(np.mean(d)), 5) if d else None}
    res["object_recall_by_size"] = {}
    for k, name in enumerate(SIZE_NAMES):
        d = [o["detected"] for o in rows_obj if SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1]]
        res["object_recall_by_size"][name] = {"n": len(d), "recall": round(float(np.mean(d)), 5) if d else None}
    typed = [o for o in rows_obj if o["detected"] and 1 <= o["group"] <= 5]
    if typed:
        t = np.array([o["group"] for o in typed])
        p = np.array([o["pred_group"] for o in typed])
        f1s = []
        for g in sorted(set(t)):
            tp = np.sum((t == g) & (p == g))
            pr = tp / max(np.sum(p == g), 1)
            rc = tp / max(np.sum(t == g), 1)
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        res["type_polygons"] = len(typed)
        res["type_accuracy"] = round(float((t == p).mean()), 5)
        res["type_macro_f1"] = round(float(np.mean(f1s)), 5)
    return res, rows_obj
