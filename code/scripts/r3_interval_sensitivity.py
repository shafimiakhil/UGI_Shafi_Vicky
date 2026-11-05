"""How firm the A2 against A1 reading is (CLAUDE.md 13.5; descriptive only, the pre-registered reading of summary_r3_ejido.json stands).
From the rule-7 files, with itareo.ugi_seg.summarize_r's pooling and block bootstrap (2,000 resamples of the 1 km blocks, paired, AP from
the coarse block histograms shifted by each arm's fine-minus-coarse offset). AP with CIRCUNVALACION excluded:
  * the interval of A2 seed 0 minus A1 seed 0 under the resampling seed of the summary (0) and five others, inside the ejido;
  * the same difference against A1 seeds 1 and 2 (A2 has one seed; A1 has three);
  * per fold, the held-out AP of each and the difference;
  * Monte Carlo: inside the ejido and on the full grid, 400 other resampling seeds (6-405) at 2,000 resamples each, the upper limit's
    mean, SD, range and the share of seeds whose interval includes zero; one long run of 50,000 resamples.
The independent check of the morning report found the Monte Carlo spread to be as large as the upper limit's distance from zero.
Writes runs_itareo/ugi/summary_r/r3_interval_sensitivity.json (refuses an existing file).
Run from work_ms/: python scripts/r3_interval_sensitivity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.ugi_seg import summarize_r as sr  # noqa: E402
from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

OUT = Path("runs_itareo/ugi/summary_r/r3_interval_sensitivity.json")
A1, A2 = "ugi_r1_none", "ugi_r3_a2_none"
FOLDS = (1, 2, 3, 4)
MANY = range(6, 406)
LONG_SEED, LONG_RUNS = 987654, 25


def weights(pb, seed_or_rng, n=sr.NBOOT):
    nb = pb["blk"]["valid"].shape[0]
    used = np.nonzero(pb["blk"]["valid"] > 0)[0]
    rng = seed_or_rng if isinstance(seed_or_rng, np.random.Generator) else np.random.default_rng(seed_or_rng)
    W = np.zeros((n, nb))
    for i in range(n):
        W[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)
    return W


def ap_vec(pos, neg):
    """AP per row of (B, bins) histograms: itareo.unet_ugi.metrics.auroc_ap_from_hist, vectorised."""
    pos, neg = pos[:, ::-1], neg[:, ::-1]
    tp, fp = np.cumsum(pos, 1), np.cumsum(neg, 1)
    return np.sum(pos / tp[:, -1:] * (tp / np.maximum(tp + fp, 1)), 1)


def ap_nc(data, W):
    h = data[1]["bhn"].astype(float)
    return ap_vec(W @ h[0], W @ h[1]) + data[0]["ap_nocirc_coarse_offset"]


def ci(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(lo), 7), round(float(hi), 7)]


def main():
    refuse_existing(OUT)
    sr.SUB[0] = "ejido"
    a2 = sr.pooled_arm_seed([sr.load(A2, f, 0) for f in FOLDS])
    a1 = {s: sr.pooled_arm_seed([sr.load(A1, f, s) for f in FOLDS]) for s in (0, 1, 2)}
    W = weights(a1[0][1], 0, 5)
    hb = a1[0][1]["bhn"].astype(float)
    check = max(abs(ap_vec((W @ hb[0])[i:i + 1], (W @ hb[1])[i:i + 1])[0] - auroc_ap_from_hist((W @ hb[0])[i], (W @ hb[1])[i])[1]) for i in range(5))
    if check > 1e-9:
        raise SystemExit(f"vectorised AP differs from the project's by {check}")
    res = {"metric": "AP with CIRCUNVALACION excluded", "a2": A2, "a1": A1, "resamples": sr.NBOOT,
           "pooled_ap_nocirc_inside_ejido": {"a2_seed0": round(a2[0]["ap_nocirc"], 7), **{f"a1_seed{s}": round(a1[s][0]["ap_nocirc"], 7) for s in a1}}}
    res["a2_seed0_minus_a1_seed0_by_resampling_seed"] = {str(rs): ci(ap_nc(a2, W_) - ap_nc(a1[0], W_)) for rs, W_ in ((r, weights(a1[0][1], r)) for r in range(6))}
    W0 = weights(a1[0][1], 0)
    res["a2_seed0_minus_a1_by_a1_seed"] = {str(s): {"point": round(a2[0]["ap_nocirc"] - a1[s][0]["ap_nocirc"], 7), "ci95_block": ci(ap_nc(a2, W0) - ap_nc(a1[s], W0))} for s in (0, 1, 2)}
    res["per_fold_seed0"] = {}
    for f in FOLDS:
        x = json.load(open(f"runs_itareo/ugi/{A2}/fold{f}_seed0/ejido/metrics_rule7.json"))["pixel"]["no_circunvalacion"]["ap"]
        y = json.load(open(f"runs_itareo/ugi/{A1}/fold{f}_seed0/ejido/metrics_rule7.json"))["pixel"]["no_circunvalacion"]["ap"]
        res["per_fold_seed0"][str(f)] = {"a2": x, "a1": y, "difference": round(x - y, 5)}
    res["many_seeds"], res["long_run"] = {}, {}
    for name, sub in (("inside_ejido", "ejido"), ("full_grid", "")):
        sr.SUB[0] = sub
        p2 = sr.pooled_arm_seed([sr.load(A2, f, 0) for f in FOLDS])
        p1 = sr.pooled_arm_seed([sr.load(A1, f, 0) for f in FOLDS])
        ups = []
        for rs in MANY:
            W_ = weights(p1[1], rs)
            ups.append(ci(ap_nc(p2, W_) - ap_nc(p1, W_))[1])
        ups = np.array(ups)
        res["many_seeds"][name] = {"seeds": len(ups), "seed_range": [MANY[0], MANY[-1]], "resamples_each": sr.NBOOT, "upper_limit_mean": round(float(ups.mean()), 6), "upper_limit_sd": round(float(ups.std(ddof=1)), 6),
                                   "upper_limit_min": round(float(ups.min()), 6), "upper_limit_max": round(float(ups.max()), 6), "share_including_zero": round(float((ups >= 0).mean()), 4)}
        rng = np.random.default_rng(LONG_SEED)
        parts = []
        for _ in range(LONG_RUNS):
            W_ = weights(p1[1], rng)
            parts.append(ap_nc(p2, W_) - ap_nc(p1, W_))
        d = np.concatenate(parts)
        res["long_run"][name] = {"resamples": int(d.size), "seed": LONG_SEED, "ci95": ci(d), "share_of_resamples_at_or_above_zero": round(float((d >= 0).mean()), 4)}
        print(name, res["many_seeds"][name], res["long_run"][name], flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
