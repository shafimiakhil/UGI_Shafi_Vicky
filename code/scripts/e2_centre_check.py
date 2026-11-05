"""Why E2 falls inside the centre: a check from the stored files, without the scoring code (CLAUDE.md 13.7: E2 keeps its ejido reading; the centre is for information). Reads files only; refuses an existing output.

Runs: ugi_r1_none seeds 0-2 (b0, b1, b2), ugi_e1_ps seed 0 (e1), ugi_e2_tex seed 0 (e2).
  * combos.csv: AP without CIRCUNVALACION and AUROC of the 81 baseline realisations (each fold trains alone, so one seed per fold gives 3^4 pooled scores), against E1 and E2, inside each candidate, in the ejido, and in the ejido outside sq8km (histograms of the ejido minus those of sq8km).
  * fold_ap.csv: the same AP per fold and run.
  * fit.csv: per fold and run, the best epoch and internal-validation AP (argmax of epochs.csv) and the mean training loss of epochs 51-60.
  * precision_recall.csv: TP, FP, FN, precision and recall of each run at its own thresholds, per candidate (the runs' boot.npz).
  * e2_vs_seeds.csv and ../../ugi_centre/seed_pairs.csv: AP difference with the paired block bootstrap (1 km blocks, 2,000 resamples, seed 0, coarse histograms shifted by fine minus coarse, as summarize_r): E2 and E1 against each baseline seed, and the pairs of baseline seeds.
  * polygons_sq8km.csv, by_size_sq8km.csv: from prob_oof.tif and the label rasters inside sq8km: true-positive pixels per polygon and run, net change by polygon size, AP without the five largest polygons. Exits 1 if the counts differ from the runs' boot.npz.

From work_ms/: python scripts/e2_centre_check.py --out ../docs/evidence_phase2/ugi_seg/e2/centre_fall
"""
import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

ROOT = Path("runs_itareo/ugi")
BASE, E1, E2 = "ugi_r1_none", "ugi_e1_ps", "ugi_e2_tex"
RUNS = {"b0": (BASE, 0), "b1": (BASE, 1), "b2": (BASE, 2), "e1": (E1, 0), "e2": (E2, 0)}
FOLDS = (1, 2, 3, 4)
NAMES = ["ring", "sq3km", "sq5km", "sq8km"]
NBOOT = 2000


def run_dir(key, fold):
    arm, seed = RUNS[key]
    return ROOT / arm / f"fold{fold}_seed{seed}"


def hist_fine(key, fold, sub):
    z = np.load(run_dir(key, fold) / sub / "boot_rule7.npz")
    return z["hist_nocirc_pos"].astype(np.float64), z["hist_nocirc_neg"].astype(np.float64)


def combos(sub, outside=None):
    """{key or (seed per fold): (auroc, ap)}: pooled over folds. outside: subtract this region's histograms."""
    H = {}
    for k in RUNS:
        for f in FOLDS:
            p, n = hist_fine(k, f, sub)
            if outside:
                po, no = hist_fine(k, f, outside)
                p, n = p - po, n - no
                assert p.min() >= 0 and n.min() >= 0
            H[k, f] = (p, n)
    out = {}
    for pick in product(("b0", "b1", "b2"), repeat=4):
        p = sum(H[pick[i], FOLDS[i]][0] for i in range(4))
        n = sum(H[pick[i], FOLDS[i]][1] for i in range(4))
        out[pick] = auroc_ap_from_hist(p, n)
    for k in ("e1", "e2"):
        out[k] = auroc_ap_from_hist(sum(H[k, f][0] for f in FOLDS), sum(H[k, f][1] for f in FOLDS))
    per_fold = {(k, f): auroc_ap_from_hist(*H[k, f])[1] for k in RUNS for f in FOLDS}
    return out, per_fold


def ap_vec(pos, neg):
    pos, neg = pos[:, ::-1], neg[:, ::-1]
    tp, fp = np.cumsum(pos, 1), np.cumsum(neg, 1)
    return np.sum(pos / tp[:, -1:] * (tp / np.maximum(tp + fp, 1)), 1)


def pooled_blocks(key, sub):
    pos, neg, fp_, fn_, valid = None, None, None, None, None
    for f in FOLDS:
        z = np.load(run_dir(key, f) / sub / "boot_rule7.npz")
        b = np.load(run_dir(key, f) / sub / "boot.npz")
        pos = z["block_hist_nocirc_pos"].astype(np.float64) if pos is None else pos + z["block_hist_nocirc_pos"]
        neg = z["block_hist_nocirc_neg"].astype(np.float64) if neg is None else neg + z["block_hist_nocirc_neg"]
        fp_ = z["hist_nocirc_pos"].astype(np.float64) if fp_ is None else fp_ + z["hist_nocirc_pos"]
        fn_ = z["hist_nocirc_neg"].astype(np.float64) if fn_ is None else fn_ + z["hist_nocirc_neg"]
        valid = b["block_valid"].astype(np.int64) if valid is None else valid + b["block_valid"]
    ap = auroc_ap_from_hist(fp_, fn_)[1]
    offset = ap - ap_vec(pos.sum(0, keepdims=True), neg.sum(0, keepdims=True))[0]
    return {"pos": pos, "neg": neg, "ap": ap, "offset": offset, "valid": valid}


def paired(a, b):
    """b minus a, paired, the resampling of summarize_r."""
    nb = a["valid"].shape[0]
    used = np.nonzero(a["valid"] > 0)[0]
    rng = np.random.default_rng(0)
    W = np.zeros((NBOOT, nb))
    for i in range(NBOOT):
        W[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)
    d = (ap_vec(W @ b["pos"], W @ b["neg"]) + b["offset"]) - (ap_vec(W @ a["pos"], W @ a["neg"]) + a["offset"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return b["ap"] - a["ap"], float(lo), float(hi), len(used)


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
    pairs_file = Path("../docs/evidence_phase2/ugi_centre/seed_pairs.csv")
    if pairs_file.exists():
        raise SystemExit(f"refused: {pairs_file} exists")
    out.mkdir(parents=True, exist_ok=False)

    # 1. the 81 realisations
    rows, fold_rows = [], []
    for label, sub, outside in [(n, f"centre_{n}", None) for n in NAMES] + [("ejido", "ejido", None), ("ejido outside sq8km", "ejido", "centre_sq8km")]:
        res, pf = combos(sub, outside)
        ap81 = np.array([v[1] for k, v in res.items() if isinstance(k, tuple)])
        au81 = np.array([v[0] for k, v in res.items() if isinstance(k, tuple)])
        row = {"region": label, "combinations": len(ap81), "ap_min": round(ap81.min(), 4), "ap_median": round(float(np.median(ap81)), 4), "ap_max": round(ap81.max(), 4)}
        for k in ("e1", "e2"):
            row[f"{k}_ap"] = round(res[k][1], 4)
            row[f"{k}_ap_share_of_combinations_below"] = round(float((ap81 < res[k][1]).mean()), 4)
        row.update({"auroc_min": round(au81.min(), 4), "auroc_median": round(float(np.median(au81)), 4), "auroc_max": round(au81.max(), 4), "e2_auroc": round(res["e2"][0], 4), "e2_auroc_share_of_combinations_below": round(float((au81 < res["e2"][0]).mean()), 4)})
        rows.append(row)
        for f in FOLDS:
            seeds = [pf["b0", f], pf["b1", f], pf["b2", f]]
            fold_rows.append({"region": label, "fold": f, "b0": round(pf["b0", f], 5), "b1": round(pf["b1", f], 5), "b2": round(pf["b2", f], 5), "e1": round(pf["e1", f], 5), "e2": round(pf["e2", f], 5), "e2_minus_b0": round(pf["e2", f] - pf["b0", f], 5), "e2_minus_lowest_seed": round(pf["e2", f] - min(seeds), 5)})
    write_csv(out / "combos.csv", rows)
    write_csv(out / "fold_ap.csv", fold_rows)
    for r in rows:
        print(r)

    # 2. fit
    rows = []
    for f in FOLDS:
        for k in RUNS:
            e = list(csv.DictReader(open(run_dir(k, f) / "epochs.csv")))
            v = np.array([float(x["val_ap"]) for x in e])
            i = int(np.argmax(v))
            loss = np.array([float(x["train_loss"]) for x in e])
            rows.append({"fold": f, "run": k, "epochs": len(e), "best_epoch": int(e[i]["epoch"]), "internal_ap_at_best": v[i], "threshold": float(e[i]["val_threshold"]), "train_loss_mean_epochs_51_60": round(float(loss[50:60].mean()), 4)})
    write_csv(out / "fit.csv", rows)
    for r in rows:
        print(r)

    # 3. precision and recall at the runs' own thresholds
    rows = []
    for n in NAMES:
        for k in RUNS:
            t = {q: sum(int(np.load(run_dir(k, f) / f"centre_{n}" / "boot.npz")[f"block_{q}"].sum()) for f in FOLDS) for q in ("tp", "fp", "fn")}
            rows.append({"candidate": n, "run": k, **t, "precision": round(t["tp"] / (t["tp"] + t["fp"]), 4), "recall": round(t["tp"] / (t["tp"] + t["fn"]), 4)})
    write_csv(out / "precision_recall.csv", rows)
    for r in rows:
        if r["candidate"] == "sq8km":
            print(r)

    # 4. differences with the block bootstrap
    rows, pair_rows = [], []
    for label, sub in [(n, f"centre_{n}") for n in NAMES] + [("ejido", "ejido")]:
        B = {k: pooled_blocks(k, sub) for k in RUNS}
        for x, y in (("b0", "b1"), ("b0", "b2"), ("b1", "b2")):
            d, lo, hi, nb = paired(B[x], B[y])
            pair_rows.append({"region": label, "pair": f"{y} - {x}", "ap_difference": round(d, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "excludes_zero": lo > 0 or hi < 0, "blocks": nb})
        for x, y in (("b0", "e2"), ("b1", "e2"), ("b2", "e2"), ("b0", "e1"), ("b1", "e1"), ("b2", "e1")):
            d, lo, hi, nb = paired(B[x], B[y])
            rows.append({"region": label, "pair": f"{y} - {x}", "ap_difference": round(d, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "excludes_zero": lo > 0 or hi < 0, "blocks": nb})
    write_csv(out / "e2_vs_seeds.csv", rows)
    write_csv(pairs_file, pair_rows)
    for r in pair_rows + rows:
        print(r)

    # 5. polygons inside sq8km, from the rasters
    with rasterio.open("derived/ugi_centre/sq8km_grid1m.tif") as s:
        M = s.read(1) > 0
    rr, cc = np.nonzero(M.any(1))[0], np.nonzero(M.any(0))[0]
    R0, R1, C0, C1 = rr[0], rr[-1] + 1, cc[0], cc[-1] + 1
    M = M[R0:R1, C0:C1]
    win = Window(int(C0), int(R0), int(C1 - C0), int(R1 - R0))
    with rasterio.open("derived/ugi_labels/ugi_id_grid1m.tif") as s:
        ID = s.read(1, window=win).astype(np.int64)
    with rasterio.open("derived/ugi_labels/ugi_binary_grid1m.tif") as s:
        Y = s.read(1, window=win)
    assert ((ID > 0) == (Y == 1)).all(), "id and binary rasters disagree"
    px = np.bincount(ID[M], minlength=ID.max() + 1)
    px[0] = 0
    big = np.argsort(px)[::-1][:5]
    print("five largest polygons inside sq8km:", big.tolist(), px[big].tolist(), "share", round(px[big].sum() / px.sum(), 4))
    Big = np.isin(ID, big)
    TP = {}
    hist = {}
    for k in RUNS:
        TP[k] = np.zeros_like(px)
        pos, neg, pos_small = np.zeros(65536), np.zeros(65536), np.zeros(65536)
        for f in FOLDS:
            d = run_dir(k, f)
            meta = json.load(open(d / "prob_oof.json"))
            r0, c0, h, w = meta["region_rows_cols"]
            ra, rb, ca, cb = max(r0, R0), min(r0 + h, R1), max(c0, C0), min(c0 + w, C1)
            with rasterio.open(d / "prob_oof.tif") as s:
                p = s.read(1, window=Window(int(ca - c0), int(ra - r0), int(cb - ca), int(rb - ra)))
            sl = (slice(ra - R0, rb - R0), slice(ca - C0, cb - C0))
            m = M[sl] & np.isfinite(p)
            y, i, b = Y[sl] == 1, ID[sl], Big[sl]
            pred = p >= meta["threshold"]
            tp, fp, fn = int((m & pred & y).sum()), int((m & pred & ~y).sum()), int((m & ~pred & y).sum())
            z = np.load(d / "centre_sq8km" / "boot.npz")
            if (tp, fp, fn) != tuple(int(z[f"block_{q}"].sum()) for q in ("tp", "fp", "fn")):
                raise SystemExit(f"counts differ from boot.npz: {k} fold {f}: {tp, fp, fn}")
            TP[k] += np.bincount(i[m & pred & y], minlength=px.size)
            code = p.astype(np.float16).view(np.uint16)                 # exact: the stored values are half-floats; positive codes sort as the values
            pos += np.bincount(code[m & y], minlength=65536)
            neg += np.bincount(code[m & ~y], minlength=65536)
            pos_small += np.bincount(code[m & y & ~b], minlength=65536)
        hist[k] = (pos, neg, pos_small)
        print(k, "AP from the rasters", round(auroc_ap_from_hist(pos, neg)[1], 5), "without the five largest polygons", round(auroc_ap_from_hist(pos_small, neg)[1], 5))

    attr = {}
    for f in FOLDS:
        for o in csv.DictReader(open(run_dir("b0", f) / "centre_sq8km" / "objects_rule7.csv")):
            attr[int(o["id"])] = (o["Tipo_evp_2"], o["group"])
    rows = [{"id": int(i), "type": attr.get(int(i), ("", ""))[0], "group": attr.get(int(i), ("", ""))[1], "pixels_inside_sq8km": int(px[i]), **{f"tp_{k}": int(TP[k][i]) for k in RUNS}, **{f"coverage_{k}": round(TP[k][i] / px[i], 4) for k in RUNS}} for i in np.argsort(px)[::-1] if px[i] > 0]
    write_csv(out / "polygons_sq8km.csv", rows)
    size = [("<10,000 px", 0, 10000), ("10,000-100,000 px", 10000, 100000), (">100,000 px", 100000, 10**12)]
    rows = []
    for name, lo, hi in size:
        s_ = (px >= lo) & (px < hi) & (px > 0)
        row = {"size_inside_sq8km": name, "polygons": int(s_.sum()), "pixels": int(px[s_].sum())}
        for k in RUNS:
            row[f"tp_{k}"] = int(TP[k][s_].sum())
        for k in ("b1", "b2", "e1", "e2"):
            row[f"tp_{k}_minus_b0"] = row[f"tp_{k}"] - row["tp_b0"]
        rows.append(row)
    rows.append({"size_inside_sq8km": "AP, all pixels", "polygons": "", "pixels": "", **{f"tp_{k}": round(auroc_ap_from_hist(hist[k][0], hist[k][1])[1], 4) for k in RUNS}})
    rows.append({"size_inside_sq8km": "AP, without the five largest polygons", "polygons": "", "pixels": "", **{f"tp_{k}": round(auroc_ap_from_hist(hist[k][2], hist[k][1])[1], 4) for k in RUNS}})
    with open(out / "by_size_sq8km.csv", "x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print("done")


if __name__ == "__main__":
    main()
