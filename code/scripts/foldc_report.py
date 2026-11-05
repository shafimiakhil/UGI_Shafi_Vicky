"""Fold C report (CLAUDE.md 13.7 step 3 and 13.8): the three seeds with their own seed SD, the quadrant out-of-fold scores of ugi_r1_none inside the same centre beside them, and the mean per-fold AP. Reads stored files only; refuses existing outputs.

Fold C is one model: its pooled AP and its per-model AP are the same number. The four-quadrant column pools four models with their own score scales. So beside it stands the mean of the four per-fold AP, and, on the same four quarter-squares (the fold lines inside the centre), the AP of the fold C model on each quarter and their mean, computed from prob_oof.tif and ugi_binary_grid1m.tif with the histograms of the scoring (the sum of the four quarters reproduces the stored AP of the seed; the run stops if it does not). AP depends on prevalence, so the mean over quarters, not the whole-square AP of fold C, is what compares with the quadrants' mean per-fold AP. Pooled AP is additive over positive pixels, so the mean weighted by each quarter's positives (ap_weighted_by_positives) stands beside the plain mean: fold 1's quarter holds 43.2 % of the positives (docs/evidence_phase2/ugi_seg/e1b/reading.md, Independent review).
Seed SD of fold C: the sample SD over its three seeds, with two degrees of freedom. It is "the same way" as the 81-based SD of 13.8: one fold gives 3 realisations, and the plug-in SD times sqrt(3/2) is the sample SD. Threshold for a later input in fold C: three times it.

Outputs in <out>: seeds.csv, quarters.csv, README.md (written last). From work_ms/: python scripts/foldc_report.py --out ../docs/evidence_phase2/ugi_seg/foldc
The paragraphs on the seed SD, the footing of the comparison, object recall and the caveat were corrected in foldc/README.md after the review of 13.8; this script still prints the earlier text, so a rerun into a new folder gives that text with its numbers.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins  # noqa: E402

ROOT = Path("runs_itareo/ugi")
NAME = "sq8km"
SUB = f"centre_{NAME}"
C, Q = "ugi_foldc_sq8km", "ugi_r1_none"
FOLDS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)
XL, YL = 388282.5, 6525025.5
REF = 3 * 0.0180


def scores(runs):
    """Pooled scores of a list of run folders, inside the centre, from the fine histograms and the block counts."""
    hp, hn = np.zeros(NBINS), np.zeros(NBINS)
    t = np.zeros(3)
    fs, ft = [], []
    for r in runs:
        d = r / SUB
        z = np.load(d / "boot_rule7.npz")
        hp, hn = hp + z["hist_nocirc_pos"], hn + z["hist_nocirc_neg"]
        b = np.load(d / "boot.npz")
        t += [b[f"block_{k}"].sum() for k in ("tp", "fp", "fn")]
        for o in csv.DictReader(open(d / "objects.csv")):
            fs.append(o["found_strict"] == "True")
            ft.append(o["found_tolerant"] == "True")
    au, ap = auroc_ap_from_hist(hp, hn)
    return {"ap": ap, "auroc": au, "iou": t[0] / t.sum(), "f1": 2 * t[0] / (2 * t[0] + t[1] + t[2]), "precision": t[0] / (t[0] + t[1]), "recall": t[0] / (t[0] + t[2]), "object_recall_tolerant": float(np.mean(ft)), "object_recall_strict": float(np.mean(fs)), "objects": len(fs)}


def quarters_of_foldc(seed):
    """AP of the fold C model on each quarter-square; quarter = the fold whose quadrant holds the pixel (1 SW, 2 SE, 3 NW, 4 NE)."""
    r = ROOT / C / f"fold5_seed{seed}"
    meta = json.load(open(r / "prob_oof.json"))
    r0, c0, H, W_ = meta["region_rows_cols"]
    with rasterio.open(r / "prob_oof.tif") as s:
        p = s.read(1)
        T = s.transform
    with rasterio.open("derived/ugi_labels/ugi_binary_grid1m.tif") as s:
        y = s.read(1, window=Window(c0, r0, W_, H)) > 0
    with rasterio.open(f"derived/ugi_centre/{NAME}_grid1m.tif") as s:
        inside = s.read(1, window=Window(c0, r0, W_, H)) == 1
    left = T.c + (np.arange(W_) + 0.5) * T.a <= XL
    south = T.f + (np.arange(H) + 0.5) * T.e <= YL
    ok = np.isfinite(p) & inside
    out, tot_p, tot_n = {}, np.zeros(NBINS), np.zeros(NBINS)
    for f, (mr, mc) in {1: (south, left), 2: (south, ~left), 3: (~south, left), 4: (~south, ~left)}.items():
        m = ok & mr[:, None] & mc[None, :]
        b = prob_bins(p[m])
        yy = y[m]
        hp, hn = np.bincount(b[yy], minlength=NBINS), np.bincount(b[~yy], minlength=NBINS)
        tot_p, tot_n = tot_p + hp, tot_n + hn
        out[f] = {"ap": auroc_ap_from_hist(hp, hn)[1], "auroc": auroc_ap_from_hist(hp, hn)[0], "ugi_share": float(yy.mean()), "pixels": int(m.sum()), "positives": int(yy.sum())}
    z = np.load(r / SUB / "boot_rule7.npz")
    stored = auroc_ap_from_hist(z["hist_nocirc_pos"], z["hist_nocirc_neg"])[1]
    whole = auroc_ap_from_hist(tot_p, tot_n)[1]
    if abs(stored - whole) > 1e-4:
        raise SystemExit(f"seed {seed}: AP over the four quarters {whole:.5f}, stored {stored:.5f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out / "seeds.csv", out / "quarters.csv")
    rows, qrows = [], []
    fc, fq = {}, {}
    for s in SEEDS:
        fc[s] = scores([ROOT / C / f"fold5_seed{s}"])
        fq[s] = scores([ROOT / Q / f"fold{f}_seed{s}" for f in FOLDS])
        q = quarters_of_foldc(s)
        fc[s]["ap_mean_of_quarters"] = float(np.mean([q[f]["ap"] for f in FOLDS]))
        w_pos = np.array([q[f]["positives"] for f in FOLDS], float)
        fc[s]["ap_weighted_by_positives"] = float(np.sum(w_pos * [q[f]["ap"] for f in FOLDS]) / w_pos.sum())
        per = {}
        for f in FOLDS:
            z = np.load(ROOT / Q / f"fold{f}_seed{s}" / SUB / "boot_rule7.npz")
            per[f] = auroc_ap_from_hist(z["hist_nocirc_pos"], z["hist_nocirc_neg"])[1]
            qrows.append({"seed": s, "quarter_fold": f, "ap_fold_c": round(q[f]["ap"], 5), "ap_quadrant": round(per[f], 5), "difference": round(q[f]["ap"] - per[f], 5), "auroc_fold_c": round(q[f]["auroc"], 5), "ugi_share": round(q[f]["ugi_share"], 5), "pixels": q[f]["pixels"]})
        fq[s]["ap_mean_of_quarters"] = float(np.mean(list(per.values())))
        fq[s]["ap_weighted_by_positives"] = float(np.sum(w_pos * [per[f] for f in FOLDS]) / w_pos.sum())
        for name, d in (("fold C", fc[s]), ("four quadrants", fq[s])):
            rows.append({"protocol": name, "seed": s, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "seeds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "seed", "ap", "ap_mean_of_quarters", "ap_weighted_by_positives", "auroc", "iou", "f1", "precision", "recall", "object_recall_tolerant", "object_recall_strict", "objects"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "quarters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(qrows[0]))
        w.writeheader()
        w.writerows(qrows)
    keys = (("AP (pooled for the quadrants)", "ap"), ("AP, mean over the four quarters", "ap_mean_of_quarters"), ("AP, quarters weighted by their positives", "ap_weighted_by_positives"), ("AUROC", "auroc"), ("IoU", "iou"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"), ("Object recall, tolerant", "object_recall_tolerant"), ("Object recall, strict", "object_recall_strict"))
    S = json.load(open(ROOT / "summary_r" / "summary_foldc_sq8km.json"))["arms"][C]
    thr = [json.load(open(ROOT / C / f"fold5_seed{s}" / "prob_oof.json"))["threshold"] for s in SEEDS]
    best = []
    for s in SEEDS:
        ep = list(csv.DictReader(open(ROOT / C / f"fold5_seed{s}" / "epochs.csv")))
        b = max(ep, key=lambda e: float(e["val_ap"]))
        best.append((int(b["epoch"]), float(b["val_ap"])))
    sd_ap = float(np.std([fc[s]["ap"] for s in SEEDS], ddof=1))
    L = [f"# Fold C: the centre {NAME} held out, with a 512 m buffer (CLAUDE.md 13.7 step 3; 13.8)", "",
         f"`ugi_r1_none`, seeds 0-2, config `itareo/configs/ugi_foldc_{NAME}.yaml`, `runs_itareo/ugi/{C}/`, summary `runs_itareo/ugi/summary_r/summary_foldc_{NAME}.json`. Beside it the same three seeds in the four-quadrant protocol, scored inside the same centre. No CIRCUNVALACION polygon lies inside {NAME}, so every score equals its no-C form. Every value: `seeds.csv`, `quarters.csv`.", "",
         "| score | fold C, seed 0 | seed 1 | seed 2 | mean | SD | quadrants, seed 0 | seed 1 | seed 2 | mean | SD |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, k in keys:
        v, w_ = [fc[s][k] for s in SEEDS], [fq[s][k] for s in SEEDS]
        L.append(f"| {lab} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {np.mean(v):.4f} | {np.std(v, ddof=1):.4f} | " + " | ".join(f"{x:.4f}" for x in w_) + f" | {np.mean(w_):.4f} | {np.std(w_, ddof=1):.4f} |")
    ci = S.get("ap_nocirc_ci95_block")
    L += ["", f"**Seed SD of fold C:** AP {sd_ap:.4f}, the sample SD over its three seeds (2 degrees of freedom; for one fold this is what the 81-based estimate of 13.8 reduces to). A later input in fold C would be read against three times it, {3 * sd_ap:.4f}, recomputed from these seeds as 13.8 says; with two degrees of freedom that number is itself uncertain. The 81-based SD of the quadrants inside {NAME} is 0.0180 (threshold {REF:.4f}).", "",
          f"**Block interval of the mean AP:** {ci[0]:.4f} to {ci[1]:.4f} (`summary_foldc_{NAME}.json`): it holds no training-seed variance.", "",
          f"**On the footing of the comparison.** Whole-square AP of fold C {np.mean([fc[s]['ap'] for s in SEEDS]):.4f} against pooled quadrants {np.mean([fq[s]['ap'] for s in SEEDS]):.4f}: one model against four, so pooling costs the right-hand number and not the left. On the same four quarter-squares, the mean AP is {np.mean([fc[s]['ap_mean_of_quarters'] for s in SEEDS]):.4f} for fold C and {np.mean([fq[s]['ap_mean_of_quarters'] for s in SEEDS]):.4f} for the quadrants, and weighted by each quarter's positives {np.mean([fc[s]['ap_weighted_by_positives'] for s in SEEDS]):.4f} and {np.mean([fq[s]['ap_weighted_by_positives'] for s in SEEDS]):.4f}: these are the like-for-like pairs (`quarters.csv`). Holding the centre out costs about {np.mean([fq[s]['ap_mean_of_quarters'] - fc[s]['ap_mean_of_quarters'] for s in SEEDS]):.3f} of AP on that footing, three of the four quarters by seed, with the seeds of the quadrants spread over {max(fq[s]['ap_mean_of_quarters'] for s in SEEDS) - min(fq[s]['ap_mean_of_quarters'] for s in SEEDS):.3f}.", "",
          f"**Thresholds and checkpoints.** Thresholds from internal validation, by seed: {', '.join(str(t) for t in thr)}. Best epoch and internal-validation AP, by seed: " + "; ".join(f"{e}, {v:.4f}" for e, v in best) + ".", "",
          f"Object recall is on {fc[0]['objects']} polygons in fold C and {fq[0]['objects']} in the four quadrants: a polygon split by a quadrant line is scored per quadrant on the right and whole in fold C.", "",
          "Caveat of 13.7: the centre is enclosed by training data on all sides, hence the buffer; fewer positives widen the intervals. Pixels outside the ejido stay in the loss, as in the sensor series (13.6); tiles straddling a quadrant line stay out, because the cache does not hold them."]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
