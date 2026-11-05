"""Two seeds of a fusion experiment against the same two seeds of ugi_r1_none (CLAUDE.md 13.8). Reads stored files only; refuses an existing output folder.

--mode reading: an undecided input. 13.8: "the reading uses the mean of the two seeds against the threshold divided by the square root of two". Gain: the mean over seeds 0 and 1 of arm minus baseline (the paired difference of itareo.ugi_seg.summarize_r --seeds 0 1). Inside the centre the test is the threshold and the folds, the interval is reported only; in the ejido the threshold, the interval and the folds, as for one seed. Positive folds: the mean of the two seeds' per-fold differences. Thresholds: 0.0540 / sqrt 2 = 0.0382 inside sq8km, 0.0372 / sqrt 2 = 0.0263 in the ejido.
--mode size: an input that stands as read (E1b, E2). 13.8: "so that their falls are sized with two seeds". The same table, no verdict.

Summaries needed: summary_<tag>_s01_ejido.json and summary_<tag>_s01_centre_sq8km.json in --sum (scripts/night_post_13_8.py writes them).
Outputs in <out>: two_seeds.md, two_seeds.csv, two_seeds.json. From work_ms/: python scripts/exp_two_seeds.py --arm ugi_e3_s2 --tag e3 --mode reading --out ../docs/evidence_phase2/ugi_seg/e3/two_seeds
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402

from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

BASE = "ugi_r1_none"
FOLDS = (1, 2, 3, 4)
SEEDS = (0, 1)
REGIONS = (("centre_sq8km", 0.0540, False), ("ejido", 0.0372, True))


def fold_scores(arm, seed, sub):
    """Per fold (AUROC, AP) without CIRCUNVALACION, and pooled, from the fine histograms."""
    per, hp, hn = {}, 0, 0
    for f in FOLDS:
        z = np.load(Path("runs_itareo/ugi") / arm / f"fold{f}_seed{seed}" / sub / "boot_rule7.npz")
        p, n = z["hist_nocirc_pos"].astype(np.float64), z["hist_nocirc_neg"].astype(np.float64)
        per[f] = auroc_ap_from_hist(p, n)
        hp, hn = hp + p, hn + n
    return per, auroc_ap_from_hist(hp, hn)


def region_table(arm, tag, sub, thr1, interval_is_a_test, mode, S):
    summ = json.load(open(S / f"summary_{tag}_s01_{sub}.json"))
    d = summ["paired_differences"][f"{arm} - {BASE}"]["ap_nocirc"]
    gain, ci = d["point"], d["ci95_block"]
    by_seed, folds = {}, {f: [] for f in FOLDS}
    for s in SEEDS:
        pa, ga = fold_scores(arm, s, sub)
        pb, gb = fold_scores(BASE, s, sub)
        by_seed[s] = {"ap_baseline": gb[1], "ap_arm": ga[1], "difference": ga[1] - gb[1], "auroc_baseline": gb[0], "auroc_arm": ga[0], "per_fold_difference": {f: pa[f][1] - pb[f][1] for f in FOLDS}}
        for f in FOLDS:
            folds[f].append(pa[f][1] - pb[f][1])
    mean_fold = {f: float(np.mean(folds[f])) for f in FOLDS}
    n_pos = sum(v > 0 for v in mean_fold.values())
    thr = thr1 / np.sqrt(2)
    tests = {f"gain_at_least_{thr:.4f}": gain >= thr, "positive_in_at_least_3_of_4_folds": n_pos >= 3}
    if interval_is_a_test:
        tests["interval_above_zero"] = ci[0] > 0
    return {"region": sub, "gain_mean_of_two_seeds": gain, "ci95_block": ci, "threshold_one_seed": thr1, "threshold_two_seeds": round(float(thr), 4), "by_seed": by_seed, "mean_per_fold_difference_by_fold": mean_fold, "mean_per_fold_difference": float(np.mean(list(mean_fold.values()))), "sign_disagreement": bool((np.mean(list(mean_fold.values())) > 0) != (gain > 0)), "folds_positive": n_pos, "tests": {k: bool(v) for k, v in tests.items()},
            "status": ("the input helps" if all(tests.values()) else "no gain") if mode == "reading" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["reading", "size"], required=True)
    ap.add_argument("--sum", default="runs_itareo/ugi/summary_r")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    S = Path(a.sum)
    res = {"arm": a.arm, "tag": a.tag, "mode": a.mode, "rule": "13.8", "regions": {}}
    L = [f"# {a.arm} against {BASE}, seeds 0 and 1 (CLAUDE.md 13.8)", ""]
    rows = []
    for sub, thr1, it in REGIONS:
        r = region_table(a.arm, a.tag, sub, thr1, it, a.mode, S)
        res["regions"][sub] = r
        head = {"centre_sq8km": "Centre sq8km, primary", "ejido": "Ejido, secondary"}[sub]
        if a.mode == "reading":
            L += [f"**Reading ({head}, two seeds, 13.8): {r['status']}.**", ""]
        else:
            L += [f"**{head}.** " + ("It stands as read; this sizes the change." if a.mode == "size" else ""), ""]
        L += [f"Gain in AP without CIRCUNVALACION, mean of the two seeds: {r['gain_mean_of_two_seeds']:+.4f}" + (f", against the threshold {r['threshold_two_seeds']:.4f} ({thr1:.4f} / sqrt 2)" if a.mode == "reading" else "") + f". Block interval{' (reported only)' if not it else ''}: [{r['ci95_block'][0]:+.4f}, {r['ci95_block'][1]:+.4f}] (`{S}/summary_{a.tag}_s01_{sub}.json`). Mean over the folds of the two seeds' differences: {r['mean_per_fold_difference']:+.4f}; by fold: " + ", ".join(f"{v:+.4f}" for v in r["mean_per_fold_difference_by_fold"].values()) + f"; positive in {r['folds_positive']} of 4." + (" The pooled gain and the mean over folds disagree in sign; pooled AP decides (13.8)." if r["sign_disagreement"] else ""), ""]
        if a.mode == "reading":
            L += ["Tests: " + "; ".join(f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" for k, v in r["tests"].items()) + ".", ""]
        L += ["| seed | AP baseline | AP arm | difference | AUROC baseline | AUROC arm | fold differences |", "|---|---|---|---|---|---|---|"]
        for s, v in r["by_seed"].items():
            L.append(f"| {s} | {v['ap_baseline']:.4f} | {v['ap_arm']:.4f} | {v['difference']:+.4f} | {v['auroc_baseline']:.4f} | {v['auroc_arm']:.4f} | " + ", ".join(f"{x:+.4f}" for x in v["per_fold_difference"].values()) + " |")
            rows.append({"region": sub, "seed": s, "ap_baseline": round(v["ap_baseline"], 5), "ap_arm": round(v["ap_arm"], 5), "difference": round(v["difference"], 5), "auroc_baseline": round(v["auroc_baseline"], 5), "auroc_arm": round(v["auroc_arm"], 5), **{f"fold{f}_difference": round(x, 5) for f, x in v["per_fold_difference"].items()}})
        L.append("")
    with open(out / "two_seeds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(res, open(out / "two_seeds.json", "w"), indent=1, default=float)
    (out / "two_seeds.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
