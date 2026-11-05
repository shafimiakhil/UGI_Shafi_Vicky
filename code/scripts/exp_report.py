"""The pre-registered reading of an experiment against ugi_r1_none, seed 0 (CLAUDE.md 13.6). Reads files only; refuses an existing output folder.

The input helps when the gain in AP with CIRCUNVALACION excluded, inside the ejido, is at least 0.010 (about three times the seed SD of 0.003), its 95 % block-bootstrap
interval lies above zero, and it is positive in at least 3 of 4 folds. Otherwise "no gain". The reading comes from summary_<tag>_ejido.json, as for the other summaries;
the same interval with 50,000 resamples is shown beside it, because a limit near zero depends on the resampling seed. E1 also on the B4 tile set
(summary_<tag>_ejido_b4.json). With both tile sets the rule of 13.7 decides: both pass, "the input helps"; B4 alone, "helps where it exists"; all tiles alone, "no gain, to be understood"; neither, "no gain". The tables: paired scores, among them AUROC and F1 (Section 0), per-fold values and signs,
object recall by group, size and width, area-weighted coverage of parks and water corridors, precision by area inside the ejido and on the full grid.
With --centre <name> (from E1b on, 13.7 step 2) the centre is the primary region: the same three tests inside it (summary_<tag>_centre_<name>.json), with the AP threshold reset to three times the seed SD measured inside it (summary_centre_<name>.json), and the ejido reading above as the secondary one.

--rule 13.8 (the default; E3 on, ruled by Shafi before any E3 result existed; E1, E1b and E2 stand as read and --rule 13.7 reproduces them):
  * threshold: three times the SD of the 81 one-seed-per-fold baseline realisations, corrected by sqrt(3/2): ejido 0.0372 (SD 0.0124), centre sq8km 0.0540 (SD 0.0180), as written in 13.8 and checked against the fine histograms here. Another pixel set (the B4 tiles) gets its own SD, computed the same way;
  * inside the centre the seed threshold and the folds are the test of "the input helps"; the block interval is reported only, because the block bootstrap does not cover seed noise there. In the ejido all three tests stand;
  * "undecided" (13.8), computed in each region as written, while only the centre triggers the seed (Shafi, 4 Nov): positive in at least 3 of 4 folds, 95 % block interval above zero, gain below the threshold. It decides no pass: an undecided input gets seed 1 and the reading then uses the mean of the two seeds against the threshold divided by sqrt 2 (scripts/exp_two_seeds.py). Anything else below the threshold is "no gain";
  * the mean per-fold AP difference, with the per-fold values, stands beside the pooled gain, and a disagreement in sign is stated; the pooled gain decides.
reading.json holds the same reading for the queue (scripts/queue_plan_13_8.py).

Outputs in <out>: reading.md, reading.json (13.8), paired_table.csv, per_fold.csv, objects_by_class.csv.
Run from work_ms/: python scripts/exp_report.py --arm ugi_e1_ps --tag e1 --out ../docs/evidence_phase2/ugi_seg/e1 [--sum runs_itareo/ugi/summary_r] [--b4]
"""
import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.ugi_seg import summarize_r as sr  # noqa: E402
from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

BASE = "ugi_r1_none"
FOLDS = (1, 2, 3, 4)
RULE_13_8 = {"ejido": (0.0124, 0.0372), "centre_sq8km": (0.0180, 0.0540)}      # (SD, threshold) as written in CLAUDE.md 13.8; checked against the histograms by threshold_13_8
GROUPS = {"1": "Road greens", "2": "Squares", "3": "Parks and green areas", "4": "Water corridors", "5": "Rail verges", "6": "NO DEFINIDO"}
SIZE = [("<500", 0, 500), ("500-2000", 500, 2000), ("2000-10000", 2000, 10000), (">10000", 10000, np.inf)]
WIDTH = [("<3", 0, 3), ("3-6", 3, 6), ("6-12", 6, 12), ("12-25", 12, 25), (">25", 25, np.inf)]
ROWS = [("AP, CIRCUNVALACION excluded", "ap_nocirc"), ("AP", "ap"), ("AP, vegetated pixels", "ap_vegetated"), ("AUROC, CIRCUNVALACION excluded", "auroc_nocirc"), ("AUROC", "auroc"), ("IoU, CIRCUNVALACION excluded", "iou_nocirc"), ("IoU", "iou"),
        ("F1, CIRCUNVALACION excluded", "f1_nocirc"), ("F1", "f1"), ("Object recall, tolerant", "object_recall_tolerant"), ("Object recall, strict", "object_recall_strict"), ("Object recall, tolerant, CIRCUNVALACION excluded", "object_recall_tolerant_nocirc"),
        ("Object recall, tolerant, under 6 m wide", "recall_under_6m_tolerant"), ("Object recall, strict, under 6 m wide", "recall_under_6m_strict")]


def run_dir(arm, fold, sub):
    return Path("runs_itareo/ugi") / arm / f"fold{fold}_seed0" / sub


def per_fold(arm, sub):
    """AP without CIRCUNVALACION per fold, from the fine histograms of boot_rule7.npz: unrounded (metrics_rule7.json holds the same values at 5 decimals)."""
    out = {}
    for f in FOLDS:
        z = np.load(run_dir(arm, f, sub) / "boot_rule7.npz")
        out[f] = auroc_ap_from_hist(z["hist_nocirc_pos"], z["hist_nocirc_neg"])[1]
    return out


def long_run(arm, sub, n=50000):
    """The paired interval of AP no C with n resamples, the summary's estimator."""
    sr.SUB[0] = sub

    def ap_vec(pos, neg):
        pos, neg = pos[:, ::-1], neg[:, ::-1]
        tp, fp = np.cumsum(pos, 1), np.cumsum(neg, 1)
        return np.sum(pos / tp[:, -1:] * (tp / np.maximum(tp + fp, 1)), 1)
    a = sr.pooled_arm_seed([sr.load(BASE, f, 0) for f in FOLDS])
    b = sr.pooled_arm_seed([sr.load(arm, f, 0) for f in FOLDS])
    nb = a[1]["blk"]["valid"].shape[0]
    used = np.nonzero(a[1]["blk"]["valid"] > 0)[0]
    rng = np.random.default_rng(987654)
    ha, hb = a[1]["bhn"].astype(float), b[1]["bhn"].astype(float)
    parts = []
    for _ in range(n // 2000):
        W = np.zeros((2000, nb))
        for i in range(2000):
            W[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)
        parts.append((ap_vec(W @ hb[0], W @ hb[1]) + b[0]["ap_nocirc_coarse_offset"]) - (ap_vec(W @ ha[0], W @ ha[1]) + a[0]["ap_nocirc_coarse_offset"]))
    d = np.concatenate(parts)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"resamples": int(d.size), "ci95": [round(float(lo), 5), round(float(hi), 5)]}


def auroc_f1(arm, sub):
    """AUROC and F1 (Section 0), all pixels and CIRCUNVALACION excluded, baseline and arm, from the files the scoring leaves. Same interval as summarize_r: 2,000 resamples of the 1 km blocks, seed 0, paired, coarse histograms shifted by the fine-minus-coarse value."""
    sr.SUB[0] = sub
    runs = {x: [sr.load(x, f, 0) for f in FOLDS] for x in (BASE, arm)}
    nb = runs[BASE][0]["boot"]["block_valid"].shape[0]
    used = np.nonzero(sum(r["boot"]["block_valid"] for r in runs[BASE]) > 0)[0]
    rng = np.random.default_rng(0)
    W = np.zeros((sr.NBOOT, nb))
    for i in range(sr.NBOOT):
        W[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)
    val, boot = {}, {}
    for x, rs in runs.items():
        val[x], boot[x] = {}, {}
        for suffix, part, fine, coarse, cnt in (("", "boot", "hist_full", "block_hist", "block"), ("_nocirc", "boot7", "hist_nocirc", "block_hist_nocirc", "block_nocirc")):
            h = sum(np.stack([r[part][f"{fine}_pos"], r[part][f"{fine}_neg"]]).astype(np.float64) for r in rs)
            c = sum(np.stack([r[part][f"{coarse}_pos"], r[part][f"{coarse}_neg"]]).astype(np.float64) for r in rs)
            t, p, n = (sum(r[part][f"{cnt}_{k}"].astype(np.float64) for r in rs) for k in ("tp", "fp", "fn"))
            au = auroc_ap_from_hist(h[0], h[1])[0]
            val[x]["auroc" + suffix] = au
            val[x]["f1" + suffix] = 2 * t.sum() / (2 * t.sum() + p.sum() + n.sum())
            pos, neg = W @ c[0], W @ c[1]
            boot[x]["auroc" + suffix] = np.array([auroc_ap_from_hist(pos[i], neg[i])[0] for i in range(W.shape[0])]) + au - auroc_ap_from_hist(c[0].sum(0), c[1].sum(0))[0]
            tb, pb, nb_ = W @ t, W @ p, W @ n
            boot[x]["f1" + suffix] = 2 * tb / np.maximum(2 * tb + pb + nb_, 1)
    out = {}
    for k in ("auroc", "auroc_nocirc", "f1", "f1_nocirc"):
        lo, hi = np.percentile(boot[arm][k] - boot[BASE][k], [2.5, 97.5])
        out[k] = {"base": val[BASE][k], "arm": val[arm][k], "point": val[arm][k] - val[BASE][k], "ci95_block": [float(lo), float(hi)]}
    return out


def score_cell(summary, extra, arm, k):
    """(baseline, arm, difference, interval) of score k, from the summary or, for AUROC and F1, from auroc_f1."""
    if k in extra:
        e = extra[k]
        return e["base"], e["arm"], e["point"], e["ci95_block"]
    d = summary["paired_differences"][f"{arm} - {BASE}"][k]
    return summary["arms"][BASE][k], summary["arms"][arm][k], d["point"], d["ci95_block"]


def sd_81(sub):
    """The seed SD of the pooled AP without CIRCUNVALACION: the SD of the 81 one-seed-per-fold baseline realisations, times sqrt(3/2) (three values per fold give a plug-in variance of about 2/3 of the true one); docs/evidence_phase2/ugi_seg/e1b/registration_check/README.md."""
    H = {}
    for f, s in product(FOLDS, (0, 1, 2)):
        z = np.load(Path("runs_itareo/ugi") / BASE / f"fold{f}_seed{s}" / sub / "boot_rule7.npz")
        H[f, s] = (z["hist_nocirc_pos"].astype(np.float64), z["hist_nocirc_neg"].astype(np.float64))
    ap = [auroc_ap_from_hist(sum(H[f, s][0] for f, s in zip(FOLDS, pick)), sum(H[f, s][1] for f, s in zip(FOLDS, pick)))[1] for pick in product((0, 1, 2), repeat=4)]
    return float(np.std(ap) * np.sqrt(1.5))


def threshold_13_8(sub):
    """(SD, threshold) of a pixel set: the values written in 13.8 where it names them, checked against the histograms; otherwise computed the same way."""
    sd = sd_81(sub)
    if sub in RULE_13_8:
        if round(sd, 4) != RULE_13_8[sub][0]:
            raise SystemExit(f"the SD of {sub} from the histograms is {sd:.5f}, 13.8 says {RULE_13_8[sub][0]}")
        return RULE_13_8[sub]
    return round(sd, 4), round(3 * round(sd, 4), 4)


def reading(summary, arm, sub, threshold=0.010, interval_is_a_test=True):
    d = summary["paired_differences"][f"{arm} - {BASE}"]["ap_nocirc"]
    fb, fa = per_fold(BASE, sub), per_fold(arm, sub)
    diffs = {f: fa[f] - fb[f] for f in FOLDS}
    n_pos = sum(v > 0 for v in diffs.values())
    gain, ci = d["point"], d["ci95_block"]
    thr = f"{threshold:.3f}" if threshold == 0.010 else f"{threshold:.4f}"
    tests = {f"gain_at_least_{thr}": gain >= threshold, "interval_above_zero": ci[0] > 0, "positive_in_at_least_3_of_4_folds": n_pos >= 3}
    if not interval_is_a_test:
        del tests["interval_above_zero"]
    helps = all(tests.values())
    mean_fold = float(np.mean(list(diffs.values())))
    undecided = not helps and n_pos >= 3 and ci[0] > 0 and gain < threshold         # 13.8
    return {"gain": gain, "ci95_block": ci, "per_fold_base": fb, "per_fold_arm": fa, "per_fold_difference": diffs, "folds_positive": n_pos, "tests": tests, "helps": helps, "undecided": undecided,
            "status": "the input helps" if helps else "undecided" if undecided else "no gain", "threshold": threshold, "mean_per_fold_difference": mean_fold, "sign_disagreement": bool((mean_fold > 0) != (gain > 0))}


def beside_pooled(rd):
    """13.8: the mean per-fold AP difference beside the pooled gain; a disagreement in sign is stated."""
    return (f"Mean per-fold AP difference: {rd['mean_per_fold_difference']:+.4f}; " + ("its sign DISAGREES with the pooled gain, which decides (13.8)." if rd["sign_disagreement"] else "same sign as the pooled gain, which decides (13.8)."))


def for_json(rd, **extra):
    return {"status": rd["status"], "gain": float(rd["gain"]), "ci95_block": [float(v) for v in rd["ci95_block"]], "threshold": rd["threshold"], "folds_positive": int(rd["folds_positive"]), "per_fold_difference": {str(k): float(v) for k, v in rd["per_fold_difference"].items()},
            "mean_per_fold_difference": rd["mean_per_fold_difference"], "sign_disagreement": rd["sign_disagreement"], "tests": {k: bool(v) for k, v in rd["tests"].items()}, **extra}


def recalls(arm, sub):
    objs, comps = [], []
    for f in FOLDS:
        d = run_dir(arm, f, sub)
        objs += list(csv.DictReader(open(d / "objects_rule7.csv")))
        comps += list(csv.DictReader(open(d / "components_rule7.csv")))

    def summ(rows):
        if not rows:
            return {"n": 0}
        ev = np.array([float(o["pixels_evaluated"]) for o in rows])
        cov = np.array([float(o["coverage_strict"]) for o in rows])
        return {"n": len(rows), "recall_strict": float(np.mean([o["found_strict"] == "True" for o in rows])), "recall_tolerant": float(np.mean([o["found_tolerant"] == "True" for o in rows])),
                "area_weighted_coverage": float((cov * ev).sum() / ev.sum())}
    out = {"all": summ(objs)}
    for g, name in GROUPS.items():
        out[f"group: {name}"] = summ([o for o in objs if o["group"] == g])
    for name, lo, hi in SIZE:
        out[f"size: {name} m2"] = summ([o for o in objs if lo <= float(o["area_m2"]) < hi])
    for name, lo, hi in WIDTH:
        out[f"width: {name} m"] = summ([o for o in objs if lo <= float(o["width_m"]) < hi])
    area = np.array([float(c["area_m2"]) for c in comps])
    prec = {k: float(area[[c[f"matched_{k}"] == "True" for c in comps]].sum() / area.sum()) for k in ("strict", "tolerant")}
    return out, prec, len(comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sum", default="runs_itareo/ugi/summary_r")
    ap.add_argument("--b4", action="store_true")
    ap.add_argument("--centre", help="the chosen centre: the primary region from E1b on (13.7)")
    ap.add_argument("--rule", choices=["13.7", "13.8"], default="13.8", help="13.8 from E3 on; 13.7 reproduces the readings of E1 and E1b, which stand as read")
    a = ap.parse_args()
    new_rule = a.rule == "13.8"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    S = Path(a.sum)
    se, sf = json.load(open(S / f"summary_{a.tag}_ejido.json")), json.load(open(S / f"summary_{a.tag}.json"))
    sd_e, thr_e = threshold_13_8("ejido") if new_rule else (None, 0.010)
    r = reading(se, a.arm, "ejido", thr_e)
    lr = long_run(a.arm, "ejido")
    sb, rb, lrb = None, None, None
    if a.b4:
        sb = json.load(open(S / f"summary_{a.tag}_ejido_b4.json"))
        sd_b, thr_b = threshold_13_8("ejido_b4") if new_rule else (None, 0.010)
        rb = reading(sb, a.arm, "ejido_b4", thr_b)
        lrb = long_run(a.arm, "ejido_b4")
    rc = None
    if a.centre:
        sub_c = f"centre_{a.centre}"
        sc = json.load(open(S / f"summary_{a.tag}_{sub_c}.json"))
        if new_rule:
            sd_c, thr_c = threshold_13_8(sub_c)
            rc = reading(sc, a.arm, sub_c, thr_c, interval_is_a_test=False)
        else:
            sd_c = json.load(open(S / f"summary_{sub_c}.json"))["arms"][BASE]["ap_nocirc_sd_over_seeds"]
            thr_c = 3 * sd_c
            rc = reading(sc, a.arm, sub_c, thr_c)
        lrc = long_run(a.arm, sub_c)
        xc = auroc_f1(a.arm, sub_c)
    xe, xf = auroc_f1(a.arm, "ejido"), auroc_f1(a.arm, "")
    xb = auroc_f1(a.arm, "ejido_b4") if sb else None
    rows = []
    for name, k in ROWS:
        x, y, dp, dc = score_cell(se, xe, a.arm, k)
        _, _, fp, fc = score_cell(sf, xf, a.arm, k)
        row = {"score": name, "baseline_inside_ejido": round(x, 4), "arm_inside_ejido": round(y, 4), "difference_inside_ejido": round(dp, 4),
               "ci95_block_inside_ejido": f"[{dc[0]:+.4f}, {dc[1]:+.4f}]", "difference_full_grid": round(fp, 4), "ci95_block_full_grid": f"[{fc[0]:+.4f}, {fc[1]:+.4f}]"}
        if sb:
            _, _, gp, gc = score_cell(sb, xb, a.arm, k)
            row.update({"difference_b4_tiles": round(gp, 4), "ci95_block_b4_tiles": f"[{gc[0]:+.4f}, {gc[1]:+.4f}]"})
        rows.append(row)
    with open(out / "paired_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "per_fold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tile_set", "fold", "ap_nocirc_baseline", "ap_nocirc_arm", "difference"])
        for name, rd in ((f"centre {a.centre}", rc), ("all tiles", r), ("B4 tiles", rb)):
            if rd:
                for k in FOLDS:
                    w.writerow([name, k, round(rd["per_fold_base"][k], 5), round(rd["per_fold_arm"][k], 5), round(rd["per_fold_difference"][k], 5)])
    ob, pb, nb = recalls(BASE, "ejido")
    oa, pa, na = recalls(a.arm, "ejido")
    with open(out / "objects_by_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "polygons", "recall_tolerant_baseline", "recall_tolerant_arm", "recall_strict_baseline", "recall_strict_arm", "coverage_baseline", "coverage_arm"])
        for k in ob:
            if ob[k]["n"]:
                w.writerow([k, ob[k]["n"], round(ob[k]["recall_tolerant"], 4), round(oa[k]["recall_tolerant"], 4), round(ob[k]["recall_strict"], 4), round(oa[k]["recall_strict"], 4), round(ob[k]["area_weighted_coverage"], 4), round(oa[k]["area_weighted_coverage"], 4)])
    ob_f, pb_f, nb_f = recalls(BASE, "")
    oa_f, pa_f, na_f = recalls(a.arm, "")
    class_gap = max(abs(t1[k][m] - t2[k][m]) for t1, t2 in ((ob, ob_f), (oa, oa_f)) for k in t1 if t1[k]["n"] and t2[k]["n"] for m in ("recall_strict", "recall_tolerant", "area_weighted_coverage"))
    verdict = ("the input helps" if r["helps"] else "no gain") if not new_rule else r["status"]
    headline = f"**Reading: {verdict}.**"
    if rb:                                                    # the rule of 13.7, ruled before any E1 score was reported
        word = {True: "passes", False: "does not pass"}
        if r["helps"] and rb["helps"]:
            verdict = "the input helps"
        elif rb["helps"]:
            verdict = "helps where it exists"
        elif r["helps"]:
            verdict = "no gain, to be understood"
        elif new_rule and "undecided" in (r["status"], rb["status"]):
            verdict = "undecided"
        else:
            verdict = "no gain"
        headline = f"**Reading (rule of 13.7): {verdict}.** All tiles: {word[r['helps']]} the 13.6 rule. B4 tile set: {word[rb['helps']]} it."
        if new_rule:
            headline = f"**Reading (rule of 13.7 on the tests of 13.8): {verdict}.** All tiles: {r['status']}. B4 tile set: {rb['status']}."
    caveat = []
    if a.arm == "ugi_e1b_ps_plane":
        res = json.load(open(Path("../docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json")))["dates"]
        m_ = "lowpass_residual_per_block_m"
        caveat = ["", "**Registration (13.7).** E1b removes a plane per date. Residual per 1,024 m block, low-pass, median and 90th percentile, with the plane (with the constant, on the same blocks): "
                  + ", ".join(f"{d} {v['plane'][m_]['median']} and {v['plane'][m_]['p90']} m ({v['constant'][m_]['median']} and {v['constant'][m_]['p90']} m)" for d, v in res.items()) + " (`docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json`)."]
    elif rb and a.arm == "ugi_e1_ps":
        res = json.load(open(Path("../docs/evidence_phase2/ugi_seg/e1_ps_residuals/summary.json")))["dates"]
        caveat = ["", "**Registration caveat (13.7).** E1 removes one constant per date, which removes the median only. Residual per 1 km block, median and 90th percentile: "
                  + ", ".join(f"{d} {res[d]['residual_after_constant_per_block_m']['median']} and {res[d]['residual_after_constant_per_block_m']['p90']} m" for d in res)
                  + f". A plane per axis would cut {next(iter(res))} to {res[next(iter(res))]['residual_after_plane_per_block_m']['median']} m (`docs/evidence_phase2/ugi_seg/e1_ps_residuals/summary.json`). E1b tests that."]
    elif a.arm == "ugi_e1c_ps1m":
        res = json.load(open(Path("../docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json")))["dates"]
        m_ = "lowpass_residual_per_block_m"
        caveat = ["", "**What E1c holds and its registration (13.8).** The three PlanetScope dates with the plane per date of E1b removed inside one bilinear resampling to 1 m, into the stem as zero-initialised conv1 channels with one availability channel per date (the mechanism of E2). Residual per 1,024 m block with the plane, low-pass, median and 90th percentile, measured for E1b at 4 m: "
                  + ", ".join(f"{d} {v['plane'][m_]['median']} and {v['plane'][m_]['p90']} m" for d, v in res.items()) + " (`docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json`)."]
    elif a.arm == "ugi_e3_s2":
        res = json.load(open(Path("../docs/evidence_phase2/ugi_seg/e3_s2_residuals/summary.json")))
        m_ = "residual_after_constant_per_block_m_lowpass"
        caveat = ["", "**What E3 holds and its registration (13.7).** The 14 indices of `derived/features/s2_2022_indices_10m.tif` (NDVI, NDRE, NDMI of each 2022 quarter, dry-season minimum, amplitude), not the bands, as `itareo/configs/ugi_e3_s2.yaml` states, with an availability channel. The displacement -1.42 E / +3.98 N m is removed inside the single resampling. Residual per block after that constant, low-pass, median and 90th percentile: "
                  + ", ".join(f"{q} {v[m_]['median']} and {v[m_]['p90']} m" for q, v in res["quarters"].items()) + " (`docs/evidence_phase2/ugi_seg/e3_s2_residuals/summary.json`)."]
    seed_sd = None
    p = S / "summary_baseline_ejido.json"
    if p.exists():
        seed_sd = json.load(open(p))["arms"][BASE]["ap_nocirc_sd_over_seeds"]
    primary = []
    if rc:
        v_c = rc["status"] if new_rule else ("the input helps" if rc["helps"] else "no gain")
        title_c = f"**Reading (centre {a.centre}, primary, {a.rule}): {v_c}.**"
        if new_rule:
            primary = [title_c, "", f"Gain in AP inside the centre: {rc['gain']:+.4f} (`{S}/summary_{a.tag}_{sub_c}.json`), against the threshold {thr_c:.4f}, three times the SD {sd_c:.4f} of the 81 one-seed-per-fold baseline realisations inside it (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`; CLAUDE.md 13.8). Positive in {rc['folds_positive']} of 4 folds: " + ", ".join(f"{v:+.4f}" for v in rc["per_fold_difference"].values()) + ". " + beside_pooled(rc), "",
                       f"Block interval (13.8: the block bootstrap does not cover seed noise here, so it is no test of a pass; below the threshold it separates undecided from no gain, as the undecided clause is written): [{rc['ci95_block'][0]:+.4f}, {rc['ci95_block'][1]:+.4f}]; with {lrc['resamples']:,} resamples [{lrc['ci95'][0]:+.4f}, {lrc['ci95'][1]:+.4f}].", "",
                       "Tests: " + "; ".join(f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" for k, v in rc["tests"].items()) + ".", ""]
        else:
            primary = [title_c, "", f"Gain in AP inside the centre: {rc['gain']:+.4f} [{rc['ci95_block'][0]:+.4f}, {rc['ci95_block'][1]:+.4f}] (`{S}/summary_{a.tag}_{sub_c}.json`), against the threshold {thr_c:.4f}, three times the seed SD {sd_c:.4f} inside it (`{S}/summary_{sub_c}.json`). With {lrc['resamples']:,} resamples the interval is [{lrc['ci95'][0]:+.4f}, {lrc['ci95'][1]:+.4f}]. Positive in {rc['folds_positive']} of 4 folds: " + ", ".join(f"{v:+.4f}" for v in rc["per_fold_difference"].values()) + ".", "",
                       "Tests: " + "; ".join(f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" for k, v in rc["tests"].items()) + ".", ""]
        seed_pairs = Path("../docs/evidence_phase2/ugi_centre/seed_pairs.csv")
        if seed_pairs.exists():
            cells = [x for x in csv.DictReader(open(seed_pairs)) if x["region"] == a.centre]
            primary += ["The block interval holds no training-seed variance, so the threshold is the test that covers it. Between baseline seeds inside this centre (b0, b1, b2: seeds 0, 1, 2), the same interval gives " + "; ".join(f"{x['pair']} {float(x['ap_difference']):+.4f} [{float(x['ci95_low']):+.4f}, {float(x['ci95_high']):+.4f}]" for x in cells) + " (`docs/evidence_phase2/ugi_centre/seed_pairs.csv`).", ""]
        rows_c = []
        for name, k in ROWS:
            x, y, dp, dc = score_cell(sc, xc, a.arm, k)
            rows_c.append({"score": name, "baseline_inside_centre": round(x, 4), "arm_inside_centre": round(y, 4), "difference_inside_centre": round(dp, 4), "ci95_block_inside_centre": f"[{dc[0]:+.4f}, {dc[1]:+.4f}]"})
        with open(out / "paired_table_centre.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_c[0]))
            w.writeheader()
            w.writerows(rows_c)
        oc_b, pc_b, nc_b = recalls(BASE, sub_c)
        oc_a, pc_a, nc_a = recalls(a.arm, sub_c)
        with open(out / "objects_by_class_centre.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "polygons", "recall_tolerant_baseline", "recall_tolerant_arm", "recall_strict_baseline", "recall_strict_arm", "coverage_baseline", "coverage_arm"])
            for k in oc_b:
                if oc_b[k]["n"]:
                    w.writerow([k, oc_b[k]["n"], round(oc_b[k]["recall_tolerant"], 4), round(oc_a[k]["recall_tolerant"], 4), round(oc_b[k]["recall_strict"], 4), round(oc_a[k]["recall_strict"], 4), round(oc_b[k]["area_weighted_coverage"], 4), round(oc_a[k]["area_weighted_coverage"], 4)])
        primary += ["| score, inside the centre | baseline | arm | difference | 95 % block interval |", "|---|---|---|---|---|"]
        primary += [f"| {x['score']} | {x['baseline_inside_centre']:.4f} | {x['arm_inside_centre']:.4f} | {x['difference_inside_centre']:+.4f} | {x['ci95_block_inside_centre']} |" for x in rows_c]
        primary += ["", f"Precision by area, inside the centre: strict {pc_b['strict']:.4f} against {pc_a['strict']:.4f}, tolerant {pc_b['tolerant']:.4f} against {pc_a['tolerant']:.4f} ({nc_b} and {nc_a} components). Classes: `objects_by_class_centre.csv`.", "", "## The ejido, secondary", ""]
    against_e = (f"; threshold {thr_e:.4f}, three times the SD {sd_e:.4f} of the 81 one-seed-per-fold baseline realisations (`docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv`; CLAUDE.md 13.8)" if new_rule else (f"; the baseline's SD over seeds is {seed_sd:.4f}" if seed_sd else ""))
    L = [f"# {a.arm} against {BASE}, seed 0: pre-registered reading (CLAUDE.md {a.rule if new_rule or rc else '13.6'})", "", *primary, headline.replace("**Reading", "**Secondary reading") if rc else headline, "",
         f"Gain in AP with CIRCUNVALACION excluded, inside the ejido: {r['gain']:+.4f} [{r['ci95_block'][0]:+.4f}, {r['ci95_block'][1]:+.4f}] (`{S}/summary_{a.tag}_ejido.json`)"
         + against_e + f". With {lr['resamples']:,} resamples the interval is [{lr['ci95'][0]:+.4f}, {lr['ci95'][1]:+.4f}]. "
         f"Positive in {r['folds_positive']} of 4 folds: " + ", ".join(f"{v:+.4f}" for v in r["per_fold_difference"].values()) + "." + (" " + beside_pooled(r) if new_rule else ""), "",
         "Tests: " + "; ".join(f"{k.replace('_', ' ')}: {'yes' if v else 'no'}" for k, v in r["tests"].items()) + "."]
    if rb:
        L += ["", f"**B4 tile set** (1,240 tiles with all three dates, `{S}/summary_{a.tag}_ejido_b4.json`): gain {rb['gain']:+.4f} [{rb['ci95_block'][0]:+.4f}, {rb['ci95_block'][1]:+.4f}]; with {lrb['resamples']:,} resamples [{lrb['ci95'][0]:+.4f}, {lrb['ci95'][1]:+.4f}]; positive in {rb['folds_positive']} of 4 folds: "
              + ", ".join(f"{v:+.4f}" for v in rb["per_fold_difference"].values()) + (f". Threshold {thr_b:.4f}, three times the SD {sd_b:.4f} computed the same way on these tiles. " + beside_pooled(rb) + f" By the same tests: {rb['status']}." if new_rule else f". By the same tests: {'helps' if rb['helps'] else 'no gain'}.")]
    if new_rule and (rc["status"] if rc else r["status"]) == "undecided":
        L += ["", "**Undecided (13.8):** the input gets seed 1 before its reading, and the reading uses the mean of the two seeds against the threshold divided by sqrt 2 (`two_seeds/`)." + (" E1c's seed 1 is not in the queue, which stops at the conditional run (13.8): it waits for Shafi's GO." if a.arm == "ugi_e1c_ps1m" else "")]
    elif new_rule and "undecided" in [x["status"] for x in (r, rb) if x]:
        L += ["", "**Undecided outside the centre only:** that triggers no seed (Shafi, 4 Nov: the centre, the primary region, triggers it)."]
    L += caveat
    L += ["", "| score | baseline | arm | difference | 95 % block interval | full grid |" + (" B4 tiles |" if sb else ""), "|---|---|---|---|---|---|" + ("---|" if sb else "")]
    for x in rows:
        L.append(f"| {x['score']} | {x['baseline_inside_ejido']:.4f} | {x['arm_inside_ejido']:.4f} | {x['difference_inside_ejido']:+.4f} | {x['ci95_block_inside_ejido']} | {x['difference_full_grid']:+.4f} {x['ci95_block_full_grid']} |"
                 + (f" {x['difference_b4_tiles']:+.4f} {x['ci95_block_b4_tiles']} |" if sb else ""))
    L += ["", f"Precision by area, inside the ejido: strict {pb['strict']:.4f} against {pa['strict']:.4f}, tolerant {pb['tolerant']:.4f} against {pa['tolerant']:.4f} (baseline against arm; {nb} and {na} components; `components_rule7.csv` of each fold). "
          f"Full grid: strict {pb_f['strict']:.4f} against {pa_f['strict']:.4f}, tolerant {pb_f['tolerant']:.4f} against {pa_f['tolerant']:.4f} ({nb_f} and {na_f} components).", "",
          f"The class table is inside the ejido. On the full grid no recall or coverage in it differs by more than {class_gap:.4f}, because " + ("all but 263 inventory pixels lie inside the ejido." if new_rule else "the inventory lies inside the ejido."), "",
          "| class | polygons | tolerant recall, baseline | arm | area-weighted coverage, baseline | arm |", "|---|---|---|---|---|---|"]
    for k in ob:
        if ob[k]["n"]:
            L.append(f"| {k} | {ob[k]['n']} | {ob[k]['recall_tolerant']:.4f} | {oa[k]['recall_tolerant']:.4f} | {ob[k]['area_weighted_coverage']:.4f} | {oa[k]['area_weighted_coverage']:.4f} |")
    L += ["", "Files: `paired_table.csv`, `per_fold.csv`, `objects_by_class.csv` in this folder. The per-fold values come from the fine histograms of the runs' `ejido/boot_rule7.npz`" + (" (B4 rows: `ejido_b4/boot_rule7.npz`)" if rb else "") + "; `metrics_rule7.json` holds them at 5 decimals. Seed 0. One seed: a helping input is confirmed with seeds 1-2 before it enters the fusion design. Coverage is the pixel-weighted mean of per-polygon values stored at 4 decimals (`objects_rule7.csv`), so its last digit can differ by one from the exact ratio." + (f" The values inside the centre come from the runs' `centre_{a.centre}/boot_rule7.npz`, `metrics_rule7.json` and `objects_rule7.csv`." if new_rule and rc else "")]
    (out / "reading.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if new_rule:
        rj = {"rule": "13.8", "arm": a.arm, "tag": a.tag, "secondary": for_json(r, region="ejido", sd=sd_e), "secondary_verdict": verdict}
        if rc:
            rj["primary"] = for_json(rc, region=sub_c, sd=sd_c)
        if rb:
            rj["secondary_b4"] = for_json(rb, region="ejido_b4", sd=sd_b)
        json.dump(rj, open(out / "reading.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
