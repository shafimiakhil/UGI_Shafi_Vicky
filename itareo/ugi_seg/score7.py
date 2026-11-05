"""Rule 7 scoring (CLAUDE.md 13.4): evaluation only, from rasters (prob_oof.tif of a run, or the Stage M ensemble).

score7(prob, pred, r0, c0, paths, folds) for a window of GRID_1M (prob NaN where not evaluated; pred the binary map):
  * pixel AUROC / AP / IoU / F1 on subsets: all; CIRCUNVALACION excluded (its polygons' pixels left out); by source,
    IV_Muni_catastrada_2022 ('core') against the rest (positives of one source against the background, the other source's
    pixels left out); by type group (positives of the group against the background, other groups left out); away from
    the fold lines (held-out pixels within 256 m of x_line or y_line left out);
  * objects (polygons with >= 50 % of their pixels evaluated): strict and tolerant recall (as score.py) and area-weighted
    coverage, overall, without CIRCUNVALACION, by source, by group, by width class (< 3, 3-6, 6-12, 12-25, > 25 m; width =
    2 x max inside distance transform, derived/ugi_labels/polygon_attributes.csv), with and without polygons crossing a
    fold line;
  * object precision of 8-connected predicted components >= 50 m2: strict (>= 50 % of the area inside the inventory) and
    tolerant (inside the inventory dilated by 1 px, 8-neighbourhood), by count and by area, by component size class;
  * one-to-one matching (greedy by IoU, IoU >= 0.5) between components and polygons: precision, recall, F1
    (supplementary, never for selection), and per component the number of polygons with >= 50 % of their evaluated
    pixels inside it;
  * centreline completeness for CANTERO, FRANJA, CANAL, CANADA, RIO, LineaFerrea: skeleton length (skimage skeletonize
    of the polygon's pixels, 8-connected pixel count) within 2 m of a predicted pixel over the total;
  * calibration of prob: 10 equal bins, mean p against observed frequency, expected calibration error;
  * 1 km blocks: signed and absolute error of the mapped share in percentage points by inventory-share bin, strip,
    dominant group and dominant source (rows for blocks_1km_errors.csv);
  * boot: per-block confusion and coarse logit histograms without CIRCUNVALACION, and its full-resolution histogram.
  * for every pixel subset also the rule-4 scores: AUROC and AP on vegetated pixels (NDVI >= 0.3 of paths["ndvi"]) and
    on the others, beside NDVI alone on the same pixels; boundary F (2 m), block r / slope / bias / MAE, and object
    precision also without CIRCUNVALACION (pixels of its polygons left out; components with >= 50 % of their area inside
    CIRCUNVALACION polygons left out). Blocks without inventory pixels are labelled 'no inventory' (a null source is
    'none'). Version 2 of 3 Nov 2026 after the code review (v1 in archive/2026-11-03/score7_v1/).
Returns (res, objects, components, block_rows, boot).
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins

from .evaluate import GROUP_NAMES, SIZE_EDGES, SIZE_NAMES, boundary_f, pixel_scores
from .score import BLOCK, CBINS, EIGHT, MIN_COMPONENT, NDVI_VEG, STRIPS, block_stats, grid_shape, read_win

GROUPS_ALL = GROUP_NAMES + ["NO DEFINIDO"]
WIDTH_EDGES = [0, 3, 6, 12, 25, np.inf]
WIDTH_NAMES = ["<3", "3-6", "6-12", "12-25", ">25"]
COMP_EDGES = [50, 200, 500, 2000, 10000, np.inf]
COMP_NAMES = ["50-200", "200-500", "500-2000", "2000-10000", ">10000"]
CENTRELINE_TYPES = ["CANTERO", "FRANJA", "CANAL", "CANADA", "RIO", "LineaFerrea"]
SHARE_EDGES = [0, 0.005, 0.02, 0.05, 0.10, 0.20, 1.01]
SHARE_NAMES = ["<0.5 %", "0.5-2 %", "2-5 %", "5-10 %", "10-20 %", ">=20 %"]
FOLD_MARGIN = 256.0
NID = 4000


def load_attrs(path="derived/ugi_labels/polygon_attributes.csv"):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    a = {int(r["id"]): r for r in rows}
    sources = sorted({r["source"] for r in rows})
    look = {"core": np.zeros(NID, bool), "circ": np.zeros(NID, bool), "src": np.full(NID, -1, np.int64),
            "width": np.full(NID, np.nan), "cross": np.zeros(NID, bool), "type": np.full(NID, "", object)}
    for i, r in a.items():
        look["core"][i] = r["core"] == "True"
        look["circ"][i] = r["circunvalacion"] == "True"
        look["src"][i] = sources.index(r["source"])
        look["width"][i] = float(r["width_m"])
        look["cross"][i] = r["crosses_fold_line"] == "True"
        look["type"][i] = r["Tipo_evp_2"]
    return a, look, sources


def _scores(h, conf):
    s = pixel_scores(*conf)
    au, ap = auroc_ap_from_hist(h[0], h[1])
    return {"auroc": None if au is None else round(au, 5), "ap": None if ap is None else round(ap, 5), "iou": s["iou"],
            "f1": s["f1"], "precision": s["precision"], "recall": s["recall"], "ugi_pixels": int(h[0].sum()),
            "pixels": int(h.sum())}


def score7(prob, pred, r0, c0, paths, folds, attrs=None):
    H, W = prob.shape
    lab = Path(paths["labels"])
    GH, GW, GT = grid_shape(lab / "ugi_binary_grid1m.tif")
    nbc = math.ceil(GW / BLOCK)
    nblk = math.ceil(GH / BLOCK) * nbc
    A, look, sources = attrs or load_attrs()
    valid = np.isfinite(prob)
    pred = pred & valid
    ref = read_win(lab / "ugi_binary_grid1m.tif", r0, c0, H, W)[0].astype(bool)
    grp = read_win(lab / "ugi_group_grid1m.tif", r0, c0, H, W)[0]
    ids = read_win(lab / "ugi_id_grid1m.tif", r0, c0, H, W)[0].astype(np.int64)
    sid = read_win(paths["sid"], r0, c0, H, W)[0]
    nd, ndn = read_win(paths["ndvi"], r0, c0, H, W)
    # distance of pixel centres to the fold lines
    xs = GT.c + (np.arange(c0, c0 + W) + 0.5) * GT.a
    ys = GT.f + (np.arange(r0, r0 + H) + 0.5) * GT.e
    near_x = np.abs(xs - folds["x_line"]) < FOLD_MARGIN
    near_y = np.abs(ys - folds["y_line"]) < FOLD_MARGIN
    # ---- pixel subsets, by row chunks ---------------------------------------------------------------------------
    keys = ["all", "no_circunvalacion", "core", "rest", "away_from_fold_lines"] + [f"group_{g}" for g in GROUPS_ALL]
    hist = {k: np.zeros((2, NBINS), np.int64) for k in keys}
    conf = {k: np.zeros(4, np.int64) for k in keys}
    cal = {"n": np.zeros(10, np.int64), "p": np.zeros(10), "pos": np.zeros(10, np.int64)}
    bnc = {k: np.zeros(nblk, np.int64) for k in ("valid", "inv", "pred", "tp", "fp", "fn")}
    bh = np.zeros((2, nblk, CBINS), np.int64)
    blk_grp = np.zeros((nblk, 7), np.int64)
    blk_src = np.zeros((nblk, len(sources)), np.int64)
    blk_sid = np.zeros((nblk, 8), np.int64)
    r4 = {(k, v): {"m": np.zeros((2, NBINS), np.int64), "n": np.zeros((2, 20001), np.int64)} for k in keys for v in ("veg", "other")}
    for a in range(0, H, 2048):
        b = min(a + 2048, H)
        v = valid[a:b]
        if not v.any():
            continue
        p, t, pr = prob[a:b][v], ref[a:b][v], pred[a:b][v]
        ii, gg, ss = ids[a:b][v], grp[a:b][v], sid[a:b][v]
        rr, cc = np.nonzero(v)
        fold_near = near_y[a:b][rr] | near_x[cc]
        bins = prob_bins(p)
        circ, core = look["circ"][ii] & t, look["core"][ii] & t
        rest = t & ~core
        masks = {"all": np.ones_like(t), "no_circunvalacion": ~circ, "core": ~rest, "rest": ~core,
                 "away_from_fold_lines": ~fold_near}
        for g, name in enumerate(GROUPS_ALL, 1):
            masks[f"group_{name}"] = ~t | (gg == g)
        n_val = nd[a:b][v].astype(np.int64)
        n_ok = nd[a:b][v] != ndn
        vegm, othm = n_ok & (n_val >= NDVI_VEG), n_ok & (n_val < NDVI_VEG)
        nbin = np.clip(n_val + 10000, 0, 20000)
        for k, m in masks.items():
            tt, pp, bb = t[m], pr[m], bins[m]
            hist[k][0] += np.bincount(bb[tt], minlength=NBINS)
            hist[k][1] += np.bincount(bb[~tt], minlength=NBINS)
            conf[k] += [np.sum(tt & pp), np.sum(~tt & pp), np.sum(tt & ~pp), np.sum(~tt & ~pp)]
            for vname, vm in (("veg", vegm), ("other", othm)):
                mm = m & vm
                if not mm.any():
                    continue
                t2, b2, n2 = t[mm], bins[mm], nbin[mm]
                r4[(k, vname)]["m"][0] += np.bincount(b2[t2], minlength=NBINS)
                r4[(k, vname)]["m"][1] += np.bincount(b2[~t2], minlength=NBINS)
                r4[(k, vname)]["n"][0] += np.bincount(n2[t2], minlength=20001)
                r4[(k, vname)]["n"][1] += np.bincount(n2[~t2], minlength=20001)
        cb = np.minimum((p * 10).astype(np.int64), 9)
        cal["n"] += np.bincount(cb, minlength=10)
        cal["p"] += np.bincount(cb, weights=p, minlength=10)
        cal["pos"] += np.bincount(cb[t], minlength=10)
        blk = ((r0 + a + rr) // BLOCK) * nbc + (c0 + cc) // BLOCK
        nc = ~circ
        for k, m in (("valid", nc), ("inv", t & nc), ("pred", pr & nc), ("tp", t & pr & nc), ("fp", ~t & pr & nc),
                     ("fn", t & ~pr & nc)):
            bnc[k] += np.bincount(blk[m], minlength=nblk)
        flat = blk * CBINS + bins // (NBINS // CBINS)
        bh[0] += np.bincount(flat[t & nc], minlength=nblk * CBINS).reshape(nblk, CBINS)
        bh[1] += np.bincount(flat[~t & nc], minlength=nblk * CBINS).reshape(nblk, CBINS)
        blk_grp += np.bincount(blk[t] * 7 + gg[t].astype(np.int64), minlength=nblk * 7).reshape(nblk, 7)
        ns = len(sources)
        blk_src += np.bincount(blk[t] * ns + look["src"][ii[t]], minlength=nblk * ns).reshape(nblk, ns)
        blk_sid += np.bincount(blk * 8 + ss.astype(np.int64), minlength=nblk * 8).reshape(nblk, 8)
    res = {"pixel": {k: _scores(hist[k], conf[k]) for k in keys}}
    res["rule4"] = {}
    for (k, vname), h in r4.items():
        if h["m"].sum() == 0:
            continue
        ma, mp_ = auroc_ap_from_hist(h["m"][0], h["m"][1])
        na, np_ = auroc_ap_from_hist(h["n"][0], h["n"][1])
        P_, N_ = int(h["m"][0].sum()), int(h["m"][1].sum())
        rnd = lambda x: None if x is None else round(x, 5)
        res["rule4"].setdefault(k, {})[vname] = {"pixels": P_ + N_, "prevalence": round(P_ / (P_ + N_), 5), "auroc": rnd(ma),
                                                 "ap": rnd(mp_), "ndvi_auroc": rnd(na), "ndvi_ap": rnd(np_)}
    circ_px = look["circ"][ids] & ref
    res["boundary"] = {"all": boundary_f(pred, ref & valid, valid),
                       "no_circunvalacion": boundary_f(pred & ~circ_px, ref & valid & ~circ_px, valid & ~circ_px)}
    res["pixel_note"] = ("no_circunvalacion: CIRCUNVALACION polygons' pixels left out; core / rest: positives of one "
                         "source against the background, the other left out; group_*: positives of the group against the "
                         f"background, other groups left out; away_from_fold_lines: pixels within {FOLD_MARGIN:.0f} m of a "
                         "fold line left out")
    if folds.get("centre"):
        res["pixel_note"] += "; fold C: x_line and y_line (also objects fold_crossing, crosses_fold_line) are the quadrant lines, not limits of this fold"
    n, conf_b = cal["n"], np.where(cal["n"] > 0, cal["p"] / np.maximum(cal["n"], 1), np.nan)
    freq = np.where(n > 0, cal["pos"] / np.maximum(n, 1), np.nan)
    ece = float(np.nansum(n / n.sum() * np.abs(freq - conf_b)))
    res["calibration"] = {"bins": [{"from": k / 10, "to": (k + 1) / 10, "pixels": int(n[k]),
                                    "mean_p": None if not n[k] else round(float(conf_b[k]), 5),
                                    "observed_frequency": None if not n[k] else round(float(freq[k]), 5)} for k in range(10)],
                          "ece": round(ece, 5)}
    # ---- objects ---------------------------------------------------------------------------------------------------
    dil = ndimage.binary_dilation(pred, structure=EIGHT)
    vi = valid & (ids > 0)
    iv = ids[vi]
    ev = np.bincount(iv, minlength=NID)
    st = np.bincount(ids[vi & pred], minlength=NID)
    to = np.bincount(ids[vi & dil], minlength=NID)
    rr_, cc_ = np.nonzero(vi)
    rsum = np.bincount(iv, weights=rr_ + r0, minlength=NID)
    csum = np.bincount(iv, weights=cc_ + c0, minlength=NID)
    del rr_, cc_
    objs = []
    for pid, r in A.items():
        total = int(r["pixels"])
        if total == 0 or ev[pid] < 0.5 * total:
            continue
        cy, cx = rsum[pid] / ev[pid], csum[pid] / ev[pid]
        objs.append({"id": pid, "Tipo_evp_2": r["Tipo_evp_2"], "group": int(r["group"]), "source": r["source"],
                     "core": r["core"] == "True", "circunvalacion": r["circunvalacion"] == "True",
                     "width_m": float(r["width_m"]), "crosses_fold_line": r["crosses_fold_line"] == "True",
                     "area_m2": float(r["area_m2"]), "pixels_evaluated": int(ev[pid]),
                     "coverage_strict": round(float(st[pid] / ev[pid]), 4), "coverage_tolerant": round(float(to[pid] / ev[pid]), 4),
                     "found_strict": bool(st[pid] >= 0.5 * ev[pid]), "found_tolerant": bool(to[pid] >= 0.5 * ev[pid]),
                     "block": int((int(cy) // BLOCK) * nbc + int(cx) // BLOCK)})

    def osum(rows):
        if not rows:
            return {"n": 0}
        ev_ = np.array([o["pixels_evaluated"] for o in rows], float)
        cov = np.array([o["coverage_strict"] for o in rows])
        return {"n": len(rows), "recall_strict": round(float(np.mean([o["found_strict"] for o in rows])), 5),
                "recall_tolerant": round(float(np.mean([o["found_tolerant"] for o in rows])), 5),
                "area_weighted_coverage": round(float(np.sum(cov * ev_) / ev_.sum()), 5)}
    res["objects"] = {"all": osum(objs), "no_circunvalacion": osum([o for o in objs if not o["circunvalacion"]]),
                      "core": osum([o for o in objs if o["core"]]), "rest": osum([o for o in objs if not o["core"]]),
                      "without_fold_crossing": osum([o for o in objs if not o["crosses_fold_line"]]),
                      "fold_crossing": osum([o for o in objs if o["crosses_fold_line"]]),
                      "by_group": {n: osum([o for o in objs if o["group"] == g]) for g, n in enumerate(GROUPS_ALL, 1)},
                      "by_width": {n: osum([o for o in objs if WIDTH_EDGES[k] <= o["width_m"] < WIDTH_EDGES[k + 1]])
                                   for k, n in enumerate(WIDTH_NAMES)},
                      "under_6m_wide": osum([o for o in objs if o["width_m"] < 6]),
                      "by_size": {n: osum([o for o in objs if SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1]])
                                  for k, n in enumerate(SIZE_NAMES)}}
    # ---- components: precision, one-to-one matching ------------------------------------------------------------------
    comp, ncomp = ndimage.label(pred, structure=EIGHT)
    area = np.bincount(comp.ravel(), minlength=ncomp + 1)
    inv_d = ndimage.binary_dilation(ref, structure=EIGHT)
    in_strict = np.bincount(comp[ref & pred], minlength=ncomp + 1)
    in_tol = np.bincount(comp[inv_d & pred], minlength=ncomp + 1)
    psum = np.bincount(comp.ravel(), weights=np.nan_to_num(prob).ravel(), minlength=ncomp + 1)
    keep = np.nonzero(area[1:] >= MIN_COMPONENT)[0] + 1
    kept = np.zeros(ncomp + 1, bool)
    kept[keep] = True
    ms, mt = in_strict[keep] >= 0.5 * area[keep], in_tol[keep] >= 0.5 * area[keep]
    ak = area[keep].astype(float)

    def prec(m, sel=slice(None)):
        a_, m_ = ak[sel], m[sel]
        return {"components": int(len(a_)), "by_count": round(float(m_.mean()), 5) if len(a_) else None,
                "by_area": round(float(a_[m_].sum() / a_.sum()), 5) if len(a_) else None}
    res["object_precision"] = {"strict": prec(ms), "tolerant": prec(mt), "by_size": {}}
    in_circ = np.bincount(comp[circ_px & pred], minlength=ncomp + 1)[keep]
    nc_sel = in_circ < 0.5 * area[keep]
    res["object_precision"]["no_circunvalacion"] = {"strict": prec(ms, nc_sel), "tolerant": prec(mt, nc_sel),
                                                    "note": "components with >= 50 % of their area inside CIRCUNVALACION "
                                                            "polygons left out"}
    for k, n in enumerate(COMP_NAMES):
        sel = (ak >= COMP_EDGES[k]) & (ak < COMP_EDGES[k + 1])
        res["object_precision"]["by_size"][n] = {"strict": prec(ms, sel), "tolerant": prec(mt, sel)}
    # pairs (polygon, component) with their intersection
    m = vi & pred & kept[comp]
    key = ids[m] * (ncomp + 1) + comp[m]
    uk, cnt = np.unique(key, return_counts=True)
    pp_, cc2 = uk // (ncomp + 1), uk % (ncomp + 1)
    evald = {o["id"] for o in objs}
    sel = np.array([p in evald for p in pp_], bool) if len(pp_) else np.zeros(0, bool)
    pp_, cc2, cnt = pp_[sel], cc2[sel], cnt[sel]
    iou = cnt / (ev[pp_] + area[cc2] - cnt)
    order = np.argsort(-iou, kind="stable")
    used_p, used_c, matches = set(), set(), 0
    for k in order:
        if iou[k] < 0.5:
            break
        if pp_[k] in used_p or cc2[k] in used_c:
            continue
        used_p.add(int(pp_[k]))
        used_c.add(int(cc2[k]))
        matches += 1
    P_, R_ = matches / max(len(keep), 1), matches / max(len(objs), 1)
    covered = np.bincount(cc2[cnt >= 0.5 * ev[pp_]], minlength=ncomp + 1)[keep]
    res["one_to_one"] = {"matches": matches, "components": int(len(keep)), "polygons": len(objs),
                         "precision": round(P_, 5), "recall": round(R_, 5),
                         "f1": round(2 * P_ * R_ / (P_ + R_), 5) if P_ + R_ else 0.0,
                         "polygons_covered_per_component": {"0": int(np.sum(covered == 0)), "1": int(np.sum(covered == 1)),
                                                            "2": int(np.sum(covered == 2)),
                                                            "3-5": int(np.sum((covered >= 3) & (covered <= 5))),
                                                            ">5": int(np.sum(covered > 5))},
                         "note": "supplementary; never used for selection (the inventory splits contiguous green)"}
    comps = [{"component": int(k), "area_m2": int(area[k]), "mean_p": round(float(psum[k] / area[k]), 4),
              "matched_strict": bool(a), "matched_tolerant": bool(b), "polygons_covered": int(c)}
             for k, a, b, c in zip(keep, ms, mt, covered)]
    del comp, inv_d
    # ---- centreline completeness -------------------------------------------------------------------------------------
    from skimage.morphology import skeletonize
    sl = ndimage.find_objects(ids.astype(np.int32))
    cl = {t: [0, 0] for t in CENTRELINE_TYPES}
    for o in objs:
        t = o["Tipo_evp_2"]
        if t not in cl or o["id"] - 1 >= len(sl) or sl[o["id"] - 1] is None:
            continue
        rs, cs = sl[o["id"] - 1]
        R0, R1 = max(rs.start - 3, 0), min(rs.stop + 3, H)
        C0, C1 = max(cs.start - 3, 0), min(cs.stop + 3, W)
        poly = (ids[R0:R1, C0:C1] == o["id"]) & valid[R0:R1, C0:C1]
        sk = skeletonize(poly)
        if not sk.any():
            continue
        near = distance_transform_edt(~pred[R0:R1, C0:C1]) <= 2.0
        cl[t][0] += int(sk.sum())
        cl[t][1] += int((sk & near).sum())
    res["centreline_completeness"] = {t: {"skeleton_px": v[0], "share_within_2m": round(v[1] / v[0], 5) if v[0] else None}
                                      for t, v in cl.items()}
    tot = sum(v[0] for v in cl.values())
    res["centreline_completeness"]["all_types"] = {"skeleton_px": tot,
                                                   "share_within_2m": round(sum(v[1] for v in cl.values()) / tot, 5) if tot else None}
    # ---- block errors --------------------------------------------------------------------------------------------------
    brows = []
    tv = np.zeros(nblk, np.int64)
    inv_all = np.zeros(nblk, np.int64)
    pr_all = np.zeros(nblk, np.int64)
    for a in range(0, H, 2048):
        b = min(a + 2048, H)
        v = valid[a:b]
        rr, cc = np.nonzero(v)
        blk = ((r0 + a + rr) // BLOCK) * nbc + (c0 + cc) // BLOCK
        tv += np.bincount(blk, minlength=nblk)
        inv_all += np.bincount(blk[ref[a:b][v]], minlength=nblk)
        pr_all += np.bincount(blk[pred[a:b][v]], minlength=nblk)
    for k in np.nonzero(tv >= 500_000)[0]:
        sh_i, sh_m = inv_all[k] / tv[k], pr_all[k] / tv[k]
        dg = int(np.argmax(blk_grp[k][1:]) + 1) if blk_grp[k][1:].any() else 0
        ds = sources[int(np.argmax(blk_src[k]))] if blk_src[k].any() else "no inventory"
        brows.append({"block": int(k), "valid": int(tv[k]), "inventory_share": round(float(sh_i), 6),
                      "mapped_share": round(float(sh_m), 6), "error_pp": round(float(sh_m - sh_i) * 100, 4),
                      "abs_error_pp": round(abs(float(sh_m - sh_i)) * 100, 4),
                      "share_bin": SHARE_NAMES[int(np.searchsorted(SHARE_EDGES, sh_i, side="right") - 1)],
                      "strip": STRIPS.get(int(np.argmax(blk_sid[k])), "none"),
                      "dominant_group": GROUPS_ALL[dg - 1] if dg else "no inventory", "dominant_source": ds})

    def berr(key):
        out = {}
        for v in sorted({r[key] for r in brows}):
            rs = [r for r in brows if r[key] == v]
            out[v] = {"blocks": len(rs), "mean_error_pp": round(float(np.mean([r["error_pp"] for r in rs])), 4),
                      "mean_abs_error_pp": round(float(np.mean([r["abs_error_pp"] for r in rs])), 4)}
        return out
    res["block_errors"] = {k: berr(k) for k in ("share_bin", "strip", "dominant_group", "dominant_source")}
    res["blocks_1km_no_circunvalacion"] = block_stats(
        [{"valid": int(bnc["valid"][k]), "inventory_share": bnc["inv"][k] / bnc["valid"][k],
          "mapped_share": bnc["pred"][k] / bnc["valid"][k]} for k in np.nonzero(bnc["valid"])[0]])
    res["blocks_1km_all"] = block_stats([{"valid": r["valid"], "inventory_share": r["inventory_share"],
                                          "mapped_share": r["mapped_share"]} for r in brows])
    boot = {"block_hist_nocirc_pos": bh[0], "block_hist_nocirc_neg": bh[1],
            "hist_nocirc_pos": hist["no_circunvalacion"][0], "hist_nocirc_neg": hist["no_circunvalacion"][1],
            **{f"block_nocirc_{k}": v for k, v in bnc.items()}}
    return res, objs, comps, brows, boot
