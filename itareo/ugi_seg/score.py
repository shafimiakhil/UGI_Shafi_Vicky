"""Scoring of UGI probability maps (CLAUDE.md 13.3, rules 1, 2 and 4); reads rasters, never infers.

score_map(prob, pred, r0, c0, paths) scores one window of GRID_1M (a held-out quadrant, or the whole grid for an
out-of-fold ensemble). prob: P(UGI), NaN where not evaluated; pred: the binary map (a run's threshold, or a vote).
  * pixel: IoU, F1, precision, recall on all evaluated pixels and without the 1 m edge band; AUROC and AP from histograms
    of logit(p) (itareo.unet_ugi.metrics);
  * rule 4: AUROC and AP on vegetated pixels (NDVI >= 0.3 of `paths["ndvi"]`, the index raster of the mosaic the run
    used) and on the other pixels, beside NDVI alone as the score on the same pixels, overall and per Neo strip
    (`paths["sid"]`); NDVI-alone AUROC/AP from integer NDVI x 10000 (20,001 bins, ties within a value);
  * boundary F within 2 m (evaluate.boundary; counts returned so that folds can be pooled);
  * objects (inventory polygons with >= 50 % of their pixels evaluated): strict (>= 50 % of the evaluated pixels
    predicted) and tolerant (>= 50 % of them within 1 px, 8-neighbourhood, of a predicted pixel); edge share (evaluated
    pixels in the edge band), median NDVI, majority strip, centroid; type from summed group probabilities if given;
  * object precision: 8-connected predicted components >= 50 m2, matched when >= 50 % of their area lies inside the
    inventory dilated by 1 px (8-neighbourhood);
  * 1 km blocks of GRID_1M (origin at the grid origin; 25 x 25 blocks): valid, inventory and predicted pixels, the
    confusion counts and coarse histograms (1,000 logit bins) of all and of vegetated pixels, for the block bootstrap
    of rule 2; polygons and components carry the block of their centroid.
"""
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins

from .evaluate import GROUP_NAMES, SIZE_EDGES, SIZE_NAMES, boundary, pixel_scores

NDVI_VEG = 3000                 # NDVI x 10000
STRIPS = {2: "1_2", 3: "1_3", 5: "1_5", 6: "1_6"}
BLOCK = 1000
CBINS = 1000                    # coarse logit bins for the block bootstrap (NBINS // CBINS fine bins each)
MIN_COMPONENT = 50              # m2 = px at 1 m
EIGHT = np.ones((3, 3), bool)
NID = 4000
GROUPS_ALL = GROUP_NAMES + ["NO DEFINIDO"]


def read_win(path, r0, c0, H, W, band=1):
    with rasterio.open(path) as s:
        return s.read(band, window=Window(c0, r0, W, H)), s.nodata


def grid_shape(path):
    with rasterio.open(path) as s:
        return s.height, s.width, s.transform


def _ap(h):
    a, p = auroc_ap_from_hist(h[0], h[1])
    return (None if a is None else round(a, 5)), (None if p is None else round(p, 5))


def block_stats(blocks, min_valid=500_000):
    """Pearson r, OLS slope (mapped on inventory), bias and MAE (percentage points) over blocks with >= min_valid
    evaluated pixels (half a 1 km block)."""
    b = [x for x in blocks if x["valid"] >= min_valid]
    if len(b) < 3:
        return {"blocks": len(b)}
    inv = np.array([x["inventory_share"] for x in b]) * 100
    mp = np.array([x["mapped_share"] for x in b]) * 100
    r = float(np.corrcoef(inv, mp)[0, 1])
    slope, icpt = np.polyfit(inv, mp, 1)
    return {"blocks": len(b), "min_valid_px": min_valid, "pearson_r": round(r, 4), "slope": round(float(slope), 4),
            "intercept_pp": round(float(icpt), 4), "bias_pp": round(float(np.mean(mp - inv)), 4),
            "mae_pp": round(float(np.mean(np.abs(mp - inv))), 4), "inventory_share_mean_pp": round(float(inv.mean()), 4),
            "mapped_share_mean_pp": round(float(mp.mean()), 4)}


def score_map(prob, pred, r0, c0, paths, group_sums=None, rows_chunk=2048):
    """prob (H, W) float32 NaN where not evaluated; pred (H, W) bool. paths: labels, polygons, ndvi, sid.
    Returns (res, objects, unmatched_components, blocks, boot)."""
    H, W = prob.shape
    lab = Path(paths["labels"])
    GH, GW, GT = grid_shape(lab / "ugi_binary_grid1m.tif")
    nbc = math.ceil(GW / BLOCK)
    nblk = math.ceil(GH / BLOCK) * nbc
    valid = np.isfinite(prob)
    pred = pred & valid
    ref = read_win(lab / "ugi_binary_grid1m.tif", r0, c0, H, W)[0].astype(bool)
    edge = read_win(lab / "ugi_edge_grid1m.tif", r0, c0, H, W)[0] > 0
    ids = read_win(lab / "ugi_id_grid1m.tif", r0, c0, H, W)[0].astype(np.int32)
    nd, ndn = read_win(paths["ndvi"], r0, c0, H, W)
    sid = read_win(paths["sid"], r0, c0, H, W)[0]
    ndok = nd != ndn
    # ---- pixel histograms, rule 4 and blocks, by row chunks -------------------------------------------------------
    conf = {"full": np.zeros(4, np.int64), "no_edge": np.zeros(4, np.int64)}
    hist = {k: np.zeros((2, NBINS), np.int64) for k in ("full", "no_edge")}
    subsets = ["all"] + list(STRIPS.values())
    r4 = {(v, s): {"m": np.zeros((2, NBINS), np.int64), "n": np.zeros((2, 20001), np.int64)}
          for v in ("veg", "other") for s in subsets}
    nodata_ndvi = 0
    bk = {k: np.zeros(nblk, np.int64) for k in ("valid", "inv", "pred", "tp", "fp", "fn", "veg")}
    bh = np.zeros((2, nblk, CBINS), np.int64)           # coarse hist of all valid pixels per block (pos, neg)
    bhv = np.zeros((2, nblk, CBINS), np.int64)          # the same, vegetated pixels
    sid_code = {c: s for c, s in STRIPS.items()}
    for a in range(0, H, rows_chunk):
        b = min(a + rows_chunk, H)
        v = valid[a:b]
        if not v.any():
            continue
        p, t, pr, e = prob[a:b][v], ref[a:b][v], pred[a:b][v], edge[a:b][v]
        bins = prob_bins(p)
        for key, m in (("full", np.ones_like(t)), ("no_edge", ~e)):
            tt, pp = t[m], pr[m]
            conf[key] += [np.sum(tt & pp), np.sum(~tt & pp), np.sum(tt & ~pp), np.sum(~tt & ~pp)]
            hist[key][0] += np.bincount(bins[m][tt], minlength=NBINS)
            hist[key][1] += np.bincount(bins[m][~tt], minlength=NBINS)
        n_ok, n_val = ndok[a:b][v], nd[a:b][v].astype(np.int64)
        nodata_ndvi += int((~n_ok).sum())
        veg = n_ok & (n_val >= NDVI_VEG)
        oth = n_ok & (n_val < NDVI_VEG)
        nb = np.clip(n_val + 10000, 0, 20000)
        sv = sid[a:b][v]
        for vname, vm in (("veg", veg), ("other", oth)):
            for s in subsets:
                m = vm if s == "all" else vm & (sv == [c for c, x in sid_code.items() if x == s][0])
                if not m.any():
                    continue
                tt = t[m]
                r4[(vname, s)]["m"][0] += np.bincount(bins[m][tt], minlength=NBINS)
                r4[(vname, s)]["m"][1] += np.bincount(bins[m][~tt], minlength=NBINS)
                r4[(vname, s)]["n"][0] += np.bincount(nb[m][tt], minlength=20001)
                r4[(vname, s)]["n"][1] += np.bincount(nb[m][~tt], minlength=20001)
        rr, cc = np.nonzero(v)
        blk = ((r0 + a + rr) // BLOCK) * nbc + (c0 + cc) // BLOCK
        cb = bins // (NBINS // CBINS)
        bk["valid"] += np.bincount(blk, minlength=nblk)
        bk["inv"] += np.bincount(blk[t], minlength=nblk)
        bk["pred"] += np.bincount(blk[pr], minlength=nblk)
        bk["tp"] += np.bincount(blk[t & pr], minlength=nblk)
        bk["fp"] += np.bincount(blk[~t & pr], minlength=nblk)
        bk["fn"] += np.bincount(blk[t & ~pr], minlength=nblk)
        bk["veg"] += np.bincount(blk[veg], minlength=nblk)
        flat = blk * CBINS + cb
        bh[0] += np.bincount(flat[t], minlength=nblk * CBINS).reshape(nblk, CBINS)
        bh[1] += np.bincount(flat[~t], minlength=nblk * CBINS).reshape(nblk, CBINS)
        bhv[0] += np.bincount(flat[veg & t], minlength=nblk * CBINS).reshape(nblk, CBINS)
        bhv[1] += np.bincount(flat[veg & ~t], minlength=nblk * CBINS).reshape(nblk, CBINS)
    res = {}
    for key in conf:
        s = pixel_scores(*conf[key])
        s["auroc"], s["ap"] = _ap(hist[key])
        res[f"pixel_{key}"] = s
    rule4 = {}
    for (vname, s), h in r4.items():
        P, N = int(h["m"][0].sum()), int(h["m"][1].sum())
        if P + N == 0:
            continue
        ma, mp_ = _ap(h["m"])
        na, np_ = _ap(h["n"])
        rule4.setdefault(s, {})[vname] = {"pixels": P + N, "ugi_pixels": P, "prevalence": round(P / (P + N), 5),
                                          "auroc": ma, "ap": mp_, "ndvi_auroc": na, "ndvi_ap": np_}
    res["rule4"] = rule4
    res["rule4_note"] = (f"vegetated = NDVI >= {NDVI_VEG / 1e4} of {paths['ndvi']}; NDVI alone scores the same pixels; "
                         f"evaluated pixels without NDVI: {nodata_ndvi}")
    # ---- boundary ------------------------------------------------------------------------------------------------
    bp, br = boundary(pred, valid), boundary(ref & valid, valid)
    nbp, nbr = int(bp.sum()), int(br.sum())
    if nbp and nbr:
        okp = int((distance_transform_edt(~br)[bp] <= 2.0).sum())
        okr = int((distance_transform_edt(~bp)[br] <= 2.0).sum())
    else:
        okp = okr = 0
    del bp, br
    P_, R_ = (okp / nbp if nbp else None), (okr / nbr if nbr else None)
    res["boundary"] = {"pred_boundary_px": nbp, "pred_boundary_within_2m": okp, "ref_boundary_px": nbr,
                       "ref_boundary_within_2m": okr, "precision": None if P_ is None else round(P_, 5),
                       "recall": None if R_ is None else round(R_, 5),
                       "f": round(2 * P_ * R_ / (P_ + R_), 5) if P_ and R_ else None, "tolerance_m": 2.0}
    # ---- objects ---------------------------------------------------------------------------------------------------
    dil = ndimage.binary_dilation(pred, structure=EIGHT)
    vi = valid & (ids > 0)
    iv = ids[vi]
    ev = np.bincount(iv, minlength=NID)
    strict = np.bincount(ids[vi & pred], minlength=NID)
    toler = np.bincount(ids[vi & dil], minlength=NID)
    epx = np.bincount(ids[vi & edge], minlength=NID)
    rr, cc = np.nonzero(vi)
    rsum = np.bincount(iv, weights=rr + r0, minlength=NID)
    csum = np.bincount(iv, weights=cc + c0, minlength=NID)
    del rr, cc
    pstrip = np.bincount(iv * 8 + sid[vi].astype(np.int64), minlength=NID * 8).reshape(NID, 8)
    m_nd = vi & ndok
    order = np.argsort(ids[m_nd], kind="stable")
    sid_sorted, nd_sorted = ids[m_nd][order], nd[m_nd][order]
    starts = np.searchsorted(sid_sorted, np.arange(NID + 1))
    del order
    import csv as _csv
    polys = {int(r["ID_NUEVO"]): r for r in _csv.DictReader(open(paths["polygons"], encoding="utf-8"))}
    objs = []
    for pid, r in polys.items():
        total = int(r["pixels"])
        if total == 0 or ev[pid] < 0.5 * total:
            continue
        seg = nd_sorted[starts[pid]:starts[pid + 1]]
        cy, cx = rsum[pid] / ev[pid], csum[pid] / ev[pid]
        o = {"id": pid, "group": int(r["group"]), "area_m2": float(r["area_m2"]), "pixels_evaluated": int(ev[pid]),
             "coverage_strict": round(float(strict[pid] / ev[pid]), 4), "coverage_tolerant": round(float(toler[pid] / ev[pid]), 4),
             "found_strict": bool(strict[pid] >= 0.5 * ev[pid]), "found_tolerant": bool(toler[pid] >= 0.5 * ev[pid]),
             "edge_share": round(float(epx[pid] / ev[pid]), 4),
             "median_ndvi": round(float(np.median(seg)) / 1e4, 4) if seg.size else None,
             "strip": STRIPS.get(int(np.argmax(pstrip[pid])), "none"),
             "x": round(float(GT.c + (cx + 0.5) * GT.a), 1), "y": round(float(GT.f + (cy + 0.5) * GT.e), 1),
             "block": int((int(cy) // BLOCK) * nbc + int(cx) // BLOCK)}
        if group_sums is not None and pid in group_sums:
            o["pred_group"] = int(np.argmax(group_sums[pid]) + 1)
        objs.append(o)
    del nd_sorted, sid_sorted

    def recall(rows, key):
        d = [o[key] for o in rows]
        return round(float(np.mean(d)), 5) if d else None
    for key, name in (("found_strict", "strict"), ("found_tolerant", "tolerant")):
        res[f"object_recall_{name}"] = recall(objs, key)
        res[f"object_recall_{name}_by_group"] = {n: {"n": sum(o["group"] == g for o in objs),
                                                     "recall": recall([o for o in objs if o["group"] == g], key)}
                                                 for g, n in enumerate(GROUPS_ALL, 1)}
        res[f"object_recall_{name}_by_size"] = {n: {"n": sum(SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1] for o in objs),
                                                    "recall": recall([o for o in objs if SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1]], key)}
                                                for k, n in enumerate(SIZE_NAMES)}
    res["objects_evaluated"] = len(objs)
    typed = [o for o in objs if o["found_strict"] and 1 <= o["group"] <= 5 and "pred_group" in o]
    if typed:
        t = np.array([o["group"] for o in typed])
        p = np.array([o["pred_group"] for o in typed])
        f1s = []
        for g in sorted(set(t)):
            tp = np.sum((t == g) & (p == g))
            pr_, rc_ = tp / max(np.sum(p == g), 1), tp / max(np.sum(t == g), 1)
            f1s.append(2 * pr_ * rc_ / (pr_ + rc_) if pr_ + rc_ else 0.0)
        res.update({"type_polygons": len(typed), "type_accuracy": round(float((t == p).mean()), 5),
                    "type_macro_f1": round(float(np.mean(f1s)), 5)})
    # ---- components and object precision ----------------------------------------------------------------------------
    inv_d = ndimage.binary_dilation(ref, structure=EIGHT)
    comp, ncomp = ndimage.label(pred, structure=EIGHT)
    area = np.bincount(comp.ravel(), minlength=ncomp + 1)
    inside = np.bincount(comp[inv_d & pred], minlength=ncomp + 1)
    psum = np.bincount(comp.ravel(), weights=np.nan_to_num(prob).ravel(), minlength=ncomp + 1)
    rr, cc = np.nonzero(pred)
    lab_p = comp[rr, cc]
    rs = np.bincount(lab_p, weights=rr + r0, minlength=ncomp + 1)
    cs = np.bincount(lab_p, weights=cc + c0, minlength=ncomp + 1)
    del comp, rr, cc, lab_p, inv_d
    keep = np.nonzero(area[1:] >= MIN_COMPONENT)[0] + 1
    matched = inside[keep] >= 0.5 * area[keep]
    unmatched = []
    for k, ok in zip(keep, matched):
        if ok:
            continue
        cy, cx = rs[k] / area[k], cs[k] / area[k]
        srow, scol = int(round(cy - r0)), int(round(cx - c0))
        srow, scol = min(max(srow, 0), H - 1), min(max(scol, 0), W - 1)
        unmatched.append({"component": int(k), "area_m2": int(area[k]), "mean_p": round(float(psum[k] / area[k]), 4),
                          "x": round(float(GT.c + (cx + 0.5) * GT.a), 1), "y": round(float(GT.f + (cy + 0.5) * GT.e), 1),
                          "row": int(cy), "col": int(cx), "strip_at_centroid": STRIPS.get(int(sid[srow, scol]), "none"),
                          "block": int((int(cy) // BLOCK) * nbc + int(cx) // BLOCK)})
    res["components"] = {"min_area_m2": MIN_COMPONENT, "components": int(len(keep)), "matched": int(matched.sum()),
                         "object_precision": round(float(matched.mean()), 5) if len(keep) else None,
                         "unmatched_area_m2": int(sum(u["area_m2"] for u in unmatched)),
                         "matched_area_m2": int(area[keep][matched].sum())}
    # ---- blocks -----------------------------------------------------------------------------------------------------
    blocks = []
    for k in np.nonzero(bk["valid"])[0]:
        brow, bcol = divmod(int(k), nbc)
        blocks.append({"block": int(k), "block_row": brow, "block_col": bcol, "x_min": GT.c + bcol * BLOCK,
                       "y_max": GT.f - brow * BLOCK, "valid": int(bk["valid"][k]), "inventory_px": int(bk["inv"][k]),
                       "predicted_px": int(bk["pred"][k]), "tp": int(bk["tp"][k]), "fp": int(bk["fp"][k]),
                       "fn": int(bk["fn"][k]), "vegetated_px": int(bk["veg"][k]),
                       "inventory_share": round(bk["inv"][k] / bk["valid"][k], 6),
                       "mapped_share": round(bk["pred"][k] / bk["valid"][k], 6)})
    res["blocks_1km"] = block_stats(blocks)
    boot = {"block_hist_pos": bh[0], "block_hist_neg": bh[1], "block_hist_veg_pos": bhv[0], "block_hist_veg_neg": bhv[1],
            "hist_full_pos": hist["full"][0], "hist_full_neg": hist["full"][1],
            "hist_veg_pos": r4[("veg", "all")]["m"][0], "hist_veg_neg": r4[("veg", "all")]["m"][1],
            "hist_veg_ndvi_pos": r4[("veg", "all")]["n"][0], "hist_veg_ndvi_neg": r4[("veg", "all")]["n"][1],
            **{f"block_{k}": v for k, v in bk.items()}}
    return res, objs, unmatched, blocks, boot
