"""Was registration the cause of E1's result? (CLAUDE.md 13.7: "if E1 reads no gain, E1b tells whether registration was the cause.") A check from the stored files; reads files only; refuses an existing output. The pre-registered reading is in ../reading.md and does not depend on this.

Runs, seed 0 unless stated: ugi_r1_none seeds 0, 1, 2 (b0, b1, b2), ugi_e1_ps (e1), ugi_e1b_ps_plane (e1b), ugi_e2_tex (e2).
  * combos.csv: pooled AP without CIRCUNVALACION and AUROC of the 81 baseline realisations (one seed per fold), against E1, E1b and E2, inside sq8km, in the ejido, and in the ejido outside sq8km; the SD over the 81, as computed and corrected for three values per fold; the share of pairs of the 81 that differ by at least E1b minus E1.
  * fold_ap.csv: AP per fold and run, with the runs' thresholds.
  * scores_by_run.csv: per run and region, pooled AP and AUROC; their mean over folds; IoU and F1 at each fold's own threshold; and pooled AP and AUROC after shifting each fold's logits so that its internal-validation threshold falls on logit 0. The last two are description only: four separately trained models have their own score scales, and pooling ranks across them.
  * e1b_vs_e1.csv: E1b minus E1, and the pairs of baseline seeds, with the paired block bootstrap of summarize_r (1 km blocks, 2,000 resamples, seed 0).
  * fit.csv: per fold and run, best epoch, internal-validation AP, mean training loss of epochs 51-60.
  * blocks.csv, block_tests.json: per 1 km block with at least 5,000 inventory pixels inside the ejido, the size of the plane's correction relative to the constant (mean over the dates with data in the block, weighted by availability), the measured residual with the constant and with the plane (used blocks of ../../e1_ps_residuals/, low-pass), and the change in block AP and AUROC. Rank correlations within fold (ranks taken per fold, by the majority fold of the block), with a permutation p-value (10,000 permutations within fold, seed 0), and the same statistic for the pairs of baseline seeds as a placebo. If registration mattered, blocks with a larger correction should gain more from E1 to E1b, and blocks with a larger residual should gain less against the baseline.

From work_ms/: python scripts/e1b_registration_check.py --out ../docs/evidence_phase2/ugi_seg/e1b/registration_check
"""
import argparse
import csv
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import e2_centre_check as ec  # noqa: E402
from itareo.ugi_seg.ps_stage import DATES, load_offsets, load_planes, plane_value  # noqa: E402
from itareo.unet_ugi.metrics import LOGIT_MAX, NBINS, auroc_ap_from_hist  # noqa: E402

ec.RUNS["e1b"] = ("ugi_e1b_ps_plane", 0)
RUNS = ec.RUNS
FOLDS = ec.FOLDS
ORDER = ("b0", "b1", "b2", "e1", "e1b", "e2")
WIDTH = 2 * LOGIT_MAX / NBINS
GH, GW = 24381, 24297
X0, Y0 = 375327.0, 6535833.0
MIN_INV = 5000
NPERM = 10000


def threshold(key, fold):
    return json.load(open(ec.run_dir(key, fold) / "prob_oof.json"))["threshold"]


def shift(h, d):
    """Move the histogram by d bins; mass pushed past an end stays in the end bin."""
    out = np.zeros_like(h)
    if d >= 0:
        out[d:] = h[:NBINS - d]
        out[-1] += h[NBINS - d:].sum()
    else:
        out[:NBINS + d] = h[-d:]
        out[0] += h[:-d].sum()
    return out


def region_hists(sub, outside=None):
    H = {}
    for k in RUNS:
        for f in FOLDS:
            p, n = ec.hist_fine(k, f, sub)
            if outside:
                po, no = ec.hist_fine(k, f, outside)
                p, n = p - po, n - no
                assert p.min() >= 0 and n.min() >= 0
            H[k, f] = (p, n)
    return H


def rank_within(x, fold):
    r = np.zeros(len(x))
    for f in FOLDS:
        m = fold == f
        if m.sum() > 1:
            r[m] = (np.argsort(np.argsort(x[m])) + 0.5) / m.sum()
    return r


def spearman_within(x, y, fold, rng):
    """Pearson on ranks taken per fold; two-sided permutation p, permuting within fold."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, fold = x[ok], y[ok], fold[ok]
    rx, ry = rank_within(x, fold), rank_within(y, fold)
    r = float(np.corrcoef(rx, ry)[0, 1])
    idx = {f: np.nonzero(fold == f)[0] for f in FOLDS}
    hits = 0
    for _ in range(NPERM):
        p = ry.copy()
        for f, i in idx.items():
            p[i] = ry[rng.permutation(i)]
        if abs(np.corrcoef(rx, p)[0, 1]) >= abs(r):
            hits += 1
    return {"r": round(r, 3), "p_permutation": round((hits + 1) / (NPERM + 1), 4), "blocks": int(ok.sum())}


def write_csv(path, rows):
    with open(path, "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    # 1. the 81 realisations, per-fold values, pooled against mean and threshold-aligned
    combos, folds, scores = [], [], []
    for label, sub, outside in (("sq8km", "centre_sq8km", None), ("ejido", "ejido", None), ("ejido outside sq8km", "ejido", "centre_sq8km")):
        H = region_hists(sub, outside)
        res = {}
        for pick in product(("b0", "b1", "b2"), repeat=4):
            res[pick] = auroc_ap_from_hist(sum(H[pick[i], FOLDS[i]][0] for i in range(4)), sum(H[pick[i], FOLDS[i]][1] for i in range(4)))
        au81, ap81 = np.array([v[0] for v in res.values()]), np.array([v[1] for v in res.values()])
        row = {"region": label, "ap_min": round(float(ap81.min()), 4), "ap_median": round(float(np.median(ap81)), 4), "ap_max": round(float(ap81.max()), 4), "auroc_min": round(float(au81.min()), 4), "auroc_median": round(float(np.median(au81)), 4), "auroc_max": round(float(au81.max()), 4)}
        # SD of the pooled score with one independent seed per fold. Three values per fold give a plug-in variance about 2/3 of the true one, hence the factor sqrt(3/2).
        row.update({"ap_sd_81": round(float(ap81.std()), 4), "ap_sd_81_corrected": round(float(ap81.std() * math.sqrt(1.5)), 4), "auroc_sd_81_corrected": round(float(au81.std() * math.sqrt(1.5)), 4)})
        gap = abs(auroc_ap_from_hist(sum(H["e1b", f][0] for f in FOLDS), sum(H["e1b", f][1] for f in FOLDS))[1] - auroc_ap_from_hist(sum(H["e1", f][0] for f in FOLDS), sum(H["e1", f][1] for f in FOLDS))[1])
        row.update({"e1b_minus_e1_abs": round(gap, 4), "share_of_pairs_of_81_differing_at_least_that": round(float((np.abs(ap81[:, None] - ap81[None, :])[np.triu_indices(len(ap81), 1)] >= gap).mean()), 4)})
        for k in ("e1", "e1b", "e2"):
            au, ap_ = auroc_ap_from_hist(sum(H[k, f][0] for f in FOLDS), sum(H[k, f][1] for f in FOLDS))
            row.update({f"{k}_ap": round(ap_, 4), f"{k}_ap_share_of_81_below": round(float((ap81 < ap_).mean()), 4), f"{k}_auroc": round(au, 4), f"{k}_auroc_share_of_81_below": round(float((au81 < au).mean()), 4)})
        combos.append(row)
        print(row)
        pf = {(k, f): auroc_ap_from_hist(*H[k, f]) for k in ORDER for f in FOLDS}
        for f in FOLDS:
            folds.append({"region": label, "fold": f, **{k: round(pf[k, f][1], 5) for k in ORDER}, **{f"thr_{k}": threshold(k, f) for k in ORDER}})
        for k in ORDER:
            au, ap_ = auroc_ap_from_hist(sum(H[k, f][0] for f in FOLDS), sum(H[k, f][1] for f in FOLDS))
            pos_al, neg_al = np.zeros(NBINS), np.zeros(NBINS)
            for f in FOLDS:
                t = threshold(k, f)
                d = -int(round(math.log(t / (1 - t)) / WIDTH))
                pos_al += shift(H[k, f][0], d)
                neg_al += shift(H[k, f][1], d)
            au_al, ap_al = auroc_ap_from_hist(pos_al, neg_al)
            row = {"region": label, "run": k, "ap_pooled": round(ap_, 4), "ap_mean_of_folds": round(float(np.mean([pf[k, f][1] for f in FOLDS])), 4), "ap_threshold_aligned": round(ap_al, 4), "auroc_pooled": round(au, 4), "auroc_mean_of_folds": round(float(np.mean([pf[k, f][0] for f in FOLDS])), 4), "auroc_threshold_aligned": round(au_al, 4)}
            if not outside:
                t_ = {q: sum(int(np.load(ec.run_dir(k, f) / sub / "boot.npz")[f"block_{q}"].sum()) for f in FOLDS) for q in ("tp", "fp", "fn")}
                row.update({"iou": round(t_["tp"] / (t_["tp"] + t_["fp"] + t_["fn"]), 4), "f1": round(2 * t_["tp"] / (2 * t_["tp"] + t_["fp"] + t_["fn"]), 4)})
            else:
                row.update({"iou": "", "f1": ""})
            scores.append(row)
            print(row)
    write_csv(out / "combos.csv", combos)
    write_csv(out / "fold_ap.csv", folds)
    write_csv(out / "scores_by_run.csv", scores)

    # 2. E1b minus E1, and the seed pairs
    rows = []
    for label, sub in (("sq8km", "centre_sq8km"), ("ejido", "ejido")):
        B = {k: ec.pooled_blocks(k, sub) for k in ("b0", "b1", "b2", "e1", "e1b")}
        for x, y in (("e1", "e1b"), ("b0", "e1"), ("b0", "e1b"), ("b0", "b1"), ("b0", "b2"), ("b1", "b2")):
            d, lo, hi, nb = ec.paired(B[x], B[y])
            rows.append({"region": label, "pair": f"{y} - {x}", "ap_difference": round(d, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "excludes_zero": lo > 0 or hi < 0, "blocks": nb})
            print(rows[-1])
    write_csv(out / "e1b_vs_e1.csv", rows)

    # 3. fit
    rows = []
    for f in FOLDS:
        for k in ORDER:
            e = list(csv.DictReader(open(ec.run_dir(k, f) / "epochs.csv")))
            v = np.array([float(x["val_ap"]) for x in e])
            i = int(np.argmax(v))
            loss = np.array([float(x["train_loss"]) for x in e])
            rows.append({"fold": f, "run": k, "best_epoch": int(e[i]["epoch"]), "internal_ap_at_best": float(v[i]), "train_loss_mean_epochs_51_60": round(float(loss[50:60].mean()), 4)})
    write_csv(out / "fit.csv", rows)
    for f in FOLDS:
        print(f, [(r["run"], round(r["internal_ap_at_best"], 4), r["train_loss_mean_epochs_51_60"]) for r in rows if r["fold"] == f])

    # 4. per block
    nbc, nbr = math.ceil(GW / 1000), math.ceil(GH / 1000)
    nb = nbc * nbr
    planes, const = load_planes(), load_offsets()
    # availability per date and block, from the tile table (tiles assigned by their centre)
    avail = {d: np.zeros(nb) for d in DATES}
    count = np.zeros(nb)
    for t in csv.DictReader(open("derived/ugi_tiles/tiles_b4_planetscope.csv")):
        r, c = (int(t["row0"]) + 256) // 1000, (int(t["col0"]) + 256) // 1000
        b = r * nbc + c
        count[b] += 1
        for d in DATES:
            avail[d][b] += float(t[f"avail_ps{d}"])
    for d in DATES:
        avail[d] = avail[d] / np.maximum(count, 1)
    # the measured displacement on the used blocks of the residual analysis
    meas = {d: {} for d in DATES}
    for d in DATES:
        for r in csv.DictReader(open(f"../docs/evidence_phase2/ugi_seg/e1_ps_residuals/blocks_{d}.csv")):
            if r["used"] == "True" and r["dx_lp_m"] != "":
                x, y = float(r["x"]), float(r["y"])
                meas[d][int((Y0 - y) // 1000) * nbc + int((x - X0) // 1000)] = (float(r["dx_lp_m"]), float(r["dy_lp_m"]))
    centre_x = X0 + 1000 * (np.arange(nb) % nbc) + 500
    centre_y = Y0 - 1000 * (np.arange(nb) // nbc) - 500
    corr = np.zeros(nb)
    res_const, res_plane = np.full(nb, np.nan), np.full(nb, np.nan)
    for b in range(nb):
        w = np.array([avail[d][b] for d in DATES])
        if w.sum() > 0:
            v = []
            for d in DATES:
                de, dn = plane_value(planes[d], centre_x[b], centre_y[b])
                v.append(math.hypot(de - const[d]["dx_east_m"], dn - const[d]["dy_north_m"]))
            corr[b] = float((w * np.array(v)).sum() / w.sum())
        else:
            corr[b] = np.nan
        rc, rp = [], []
        for d in DATES:
            if b in meas[d]:
                de, dn = plane_value(planes[d], centre_x[b], centre_y[b])
                mx, my = meas[d][b]
                rc.append(math.hypot(mx - const[d]["dx_east_m"], my - const[d]["dy_north_m"]))
                rp.append(math.hypot(mx - de, my - dn))
        if rc:
            res_const[b], res_plane[b] = float(np.mean(rc)), float(np.mean(rp))

    def block_scores(k, sub):
        pos, neg, valid_by_fold = None, None, []
        for f in FOLDS:
            z = np.load(ec.run_dir(k, f) / sub / "boot_rule7.npz")
            pos = z["block_hist_nocirc_pos"].astype(np.float64) if pos is None else pos + z["block_hist_nocirc_pos"]
            neg = z["block_hist_nocirc_neg"].astype(np.float64) if neg is None else neg + z["block_hist_nocirc_neg"]
            valid_by_fold.append(np.load(ec.run_dir(k, f) / sub / "boot.npz")["block_valid"])
        out_ap, out_au = np.full(nb, np.nan), np.full(nb, np.nan)
        for b in range(nb):
            au, ap_ = auroc_ap_from_hist(pos[b], neg[b])
            if ap_ is not None:
                out_ap[b], out_au[b] = ap_, au
        return out_ap, out_au, pos.sum(1), np.argmax(np.stack(valid_by_fold), 0) + 1

    S = {k: block_scores(k, "ejido") for k in ("b0", "b1", "b2", "e1", "e1b")}
    inv, fold_b = S["b0"][2], S["b0"][3]
    keep = inv >= MIN_INV
    rows = []
    for b in np.nonzero(keep)[0]:
        rows.append({"block": int(b), "fold": int(fold_b[b]), "x": centre_x[b], "y": centre_y[b], "inventory_px": int(inv[b]), "correction_m": round(corr[b], 3), "residual_constant_m": round(res_const[b], 3), "residual_plane_m": round(res_plane[b], 3), **{f"ap_{k}": round(S[k][0][b], 4) for k in S}, **{f"auroc_{k}": round(S[k][1][b], 4) for k in S}})
    write_csv(out / "blocks.csv", rows)
    idx = np.nonzero(keep)[0]
    x_corr, x_rc, x_rp, f_ = corr[idx], res_const[idx], res_plane[idx], fold_b[idx]
    d = {k: {"ap": S[k][0][idx], "auroc": S[k][1][idx]} for k in S}
    rng = np.random.default_rng(0)
    tests = {"blocks_with_at_least_5000_inventory_px": int(keep.sum()), "note": "within-fold rank correlation; p from 10,000 permutations within fold"}
    tests["change_e1_to_e1b_against_correction_size"] = {m: spearman_within(x_corr, d["e1b"][m] - d["e1"][m], f_, rng) for m in ("ap", "auroc")}
    tests["change_e1_to_e1b_against_measured_improvement"] = {m: spearman_within(x_rc - x_rp, d["e1b"][m] - d["e1"][m], f_, rng) for m in ("ap", "auroc")}
    tests["gain_of_e1_over_b0_against_residual_with_constant"] = {m: spearman_within(x_rc, d["e1"][m] - d["b0"][m], f_, rng) for m in ("ap", "auroc")}
    tests["gain_of_e1b_over_b0_against_residual_with_plane"] = {m: spearman_within(x_rp, d["e1b"][m] - d["b0"][m], f_, rng) for m in ("ap", "auroc")}
    tests["placebo_seed_pairs_against_correction_size"] = {f"{y} - {x}": {m: spearman_within(x_corr, d[y][m] - d[x][m], f_, rng) for m in ("ap", "auroc")} for x, y in (("b0", "b1"), ("b0", "b2"), ("b1", "b2"))}
    tests["block_change_summary"] = {f"{y} - {x}": {"ap_mean": round(float(np.nanmean(d[y]["ap"] - d[x]["ap"])), 4), "ap_sd": round(float(np.nanstd(d[y]["ap"] - d[x]["ap"], ddof=1)), 4)} for x, y in (("e1", "e1b"), ("b0", "e1"), ("b0", "e1b"), ("b0", "b1"), ("b0", "b2"), ("b1", "b2"))}
    tests["correction_m"] = {"median": round(float(np.nanmedian(x_corr)), 3), "p90": round(float(np.nanpercentile(x_corr, 90)), 3), "max": round(float(np.nanmax(x_corr)), 3)}
    tests["residual_m_on_measured_blocks"] = {"with_constant_median": round(float(np.nanmedian(x_rc)), 3), "with_plane_median": round(float(np.nanmedian(x_rp)), 3), "blocks": int(np.isfinite(x_rc).sum())}
    with open(out / "block_tests.json", "x", encoding="utf-8") as f:
        json.dump(tests, f, indent=1)
    print(json.dumps(tests, indent=1))
    print("done")


if __name__ == "__main__":
    main()
