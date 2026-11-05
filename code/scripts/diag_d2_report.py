"""D2 of CLAUDE.md 13.8, the table: the trained E1b and E2 models with the added input at its no-data value against the same models as run, seed 0. Reads stored files only; refuses an existing output folder.

For each pixel set (centre sq8km, ejido, and for E1b the B4 tiles inside the ejido, where PlanetScope exists): AP and AUROC without CIRCUNVALACION, pooled and per fold, as run and blanked, from the fine histograms of boot_rule7.npz; the block interval of the AP difference (summary_d2_<tag>_<set>.json); the mean absolute change of P(UGI) per pixel (every 4th pixel of prob_oof.tif, inside the set), and the share of pixels whose side of the run's threshold changes.
Reading of 13.8: a change below the seed SD means the model did not use the input. Its test stays in the columns *_below_seed_sd. The column reading adds the outputs (written before D2 ran, after the review of 13.8; the wording of 13.8 is Shafi's): the words "did not use the input" are printed only if pooled AP and AUROC are below their SDs, no fold is beyond its SD, and every fold has mean |dP| below 0.001, Spearman at least 0.999 and top-k overlap at least 0.99, the range of numerical differences alone. The SDs: the 81-based SD of AP and of AUROC inside each set (docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv); the B4 tiles have no such row, so the ejido's stands there, and that is said. The threshold of each fold's run is unchanged. A change of score scale common to all folds moves neither AP nor AUROC. One that differs between folds moves the pooled numbers with no re-ranking inside any fold: a logit offset of 1 in one fold moves the pooled AP of E2 by -0.0403 to +0.0297 inside sq8km and by -0.0184 to +0.0044 in the ejido, and of E1b by -0.0372 to +0.0284 and by -0.0257 to +0.0121, against seed SDs of 0.0180 and 0.0124, with every fold's AP unchanged. The columns AP at equal FPR and mean of folds do not move under such an offset; the IoU column does.
Per fold, added: the AP change, each fold's SD over the baseline's seeds 0 to 2 (2 degrees of freedom), AP at equal FPR, the pooled change the folds' median logit changes alone would give, and from the two prob_oof.tif (every 4th pixel, the AP's pixels): mean |dP|, share changing side of the threshold, median and robust SD of the logit change, Spearman, overlap of the top k pixels, k being the set's positives; vegetated AP and AUROC from metrics_rule7.json.

Outputs in <out>: d2_table.csv, d2_folds.csv, d2.json, README.md (with diagnostics/d2_notes.md appended). From work_ms/: python scripts/diag_d2_report.py --out ../docs/evidence_phase2/ugi_seg/diagnostics/d2_blank
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

ROOT = Path("runs_itareo/ugi")
S = ROOT / "summary_r"
FOLDS = (1, 2, 3, 4)
ARMS = {"e1b": ("ugi_e1b_ps_plane", "diag_d2_e1b_blank", ("centre_sq8km", "ejido", "ejido_b4")), "e2": ("ugi_e2_tex", "diag_d2_e2_blank", ("centre_sq8km", "ejido"))}
LABEL = {"centre_sq8km": "centre sq8km", "ejido": "ejido", "ejido_b4": "ejido, B4 tiles"}
BASE = "ugi_r1_none"
BIN = 4e-4  # logit width of the fine histograms (itareo.unet_ugi.metrics)
NOTES = Path("../docs/evidence_phase2/ugi_seg/diagnostics/d2_notes.md")
STILL = {"mean_abs_dp": 0.001, "spearman": 0.999, "topk_overlap": 0.99}  # numerical differences alone (CPU against GPU, same weights and input) reach at most 0.0005, at least 0.9994 and at least 0.994


def seed_sds():
    out = {}
    for r in csv.DictReader(open("../docs/evidence_phase2/ugi_seg/e1b/registration_check/combos.csv")):
        out[r["region"]] = (float(r["ap_sd_81_corrected"]), float(r["auroc_sd_81_corrected"]))
    return {"centre_sq8km": out["sq8km"], "ejido": out["ejido"], "ejido_b4": out["ejido"]}


def scores(arm, sub):
    per, hp, hn, t = {}, 0, 0, np.zeros(3)
    for f in FOLDS:
        d = ROOT / arm / f"fold{f}_seed0" / sub
        z = np.load(d / "boot_rule7.npz")
        p, n = z["hist_nocirc_pos"].astype(np.float64), z["hist_nocirc_neg"].astype(np.float64)
        per[f] = auroc_ap_from_hist(p, n)
        hp, hn = hp + p, hn + n
        t += [z[f"block_nocirc_{k}"].sum() for k in ("tp", "fp", "fn")]
    au, ap = auroc_ap_from_hist(hp, hn)
    return {"per_fold": per, "auroc": au, "ap": ap, "iou": t[0] / t.sum()}


def change(arm, blank, sub):
    """Mean |dP| and the share of pixels whose side of the threshold changes, every 4th pixel, inside the set."""
    tot, flips, n = 0.0, 0, 0
    for f in FOLDS:
        a, b = ROOT / arm / f"fold{f}_seed0", ROOT / blank / f"fold{f}_seed0"
        meta = json.load(open(a / "prob_oof.json"))
        r0, c0, H, W_ = meta["region_rows_cols"]
        with rasterio.open(a / "prob_oof.tif") as s:
            p = s.read(1)[::4, ::4]
        with rasterio.open(b / "prob_oof.tif") as s:
            q = s.read(1)[::4, ::4]
        if sub == "centre_sq8km":
            with rasterio.open("derived/ugi_centre/sq8km_grid1m.tif") as s:
                m = s.read(1, window=Window(c0, r0, W_, H))[::4, ::4] == 1
        else:
            with rasterio.open("derived/ugi_labels/ejido_grid1m.tif") as s:
                m = s.read(1, window=Window(c0, r0, W_, H))[::4, ::4] == 1
            if sub == "ejido_b4":
                from itareo.ugi_seg.rescore9 import b4_mask
                m &= b4_mask(r0, c0, H, W_)[::4, ::4]
        ok = m & np.isfinite(p) & np.isfinite(q)
        tot += float(np.abs(p[ok] - q[ok]).sum())
        flips += int(((p[ok] >= meta["threshold"]) != (q[ok] >= meta["threshold"])).sum())
        n += int(ok.sum())
    return tot / n, flips / n


def fold_hist(arm, f, s, sub):
    z = np.load(ROOT / arm / f"fold{f}_seed{s}" / sub / "boot_rule7.npz")
    return z["hist_nocirc_pos"].astype(np.float64), z["hist_nocirc_neg"].astype(np.float64)


def shift(h, delta):
    """The histogram of logit + delta, clipped at the ends."""
    k = int(round(delta / BIN))
    if k == 0:
        return h.copy()
    o = np.zeros_like(h)
    if k > 0:
        o[k:] = h[:-k]
        o[-1] += h[-k:].sum()
    else:
        o[:k] = h[-k:]
        o[0] += h[:-k].sum()
    return o


def ap_equal_fpr(folds):
    """folds: list of (pos, neg). Each fold's bins, from the top score down, are blocks ending at the fold's own FPR; all blocks are pooled in order of that FPR, AP over them. On one fold it is that fold's AP; a per-fold offset does not move it. It uses the held-out labels: a reading aid, not a score."""
    us, ps, ns = [], [], []
    for pos, neg in folds:
        pos, neg = pos[::-1], neg[::-1]
        k = (pos > 0) | (neg > 0)
        us.append(np.cumsum(neg[k]) / neg[k].sum())
        ps.append(pos[k])
        ns.append(neg[k])
    o = np.argsort(np.concatenate(us), kind="stable")
    p, n = np.concatenate(ps)[o], np.concatenate(ns)[o]
    tp, fp = np.cumsum(p), np.cumsum(n)
    k = p > 0
    return float(np.sum(p[k] / p.sum() * tp[k] / np.maximum(tp[k] + fp[k], 1)))


def baseline_sds(sub):
    """Each fold's SD of AP over the baseline's seeds 0 to 2, and the 81-based SDs of AP at equal FPR and of the mean of fold APs (plug-in SD x sqrt(3/2), as combos.csv). The B4 tiles use the ejido's: only seed 0 of the baseline is sure to be scored on them when D2 ends."""
    s0 = "ejido" if sub == "ejido_b4" else sub
    H = {(s, f): fold_hist(BASE, f, s, s0) for s in (0, 1, 2) for f in FOLDS}
    per = np.array([[auroc_ap_from_hist(*H[(s, f)])[1] for f in FOLDS] for s in (0, 1, 2)])
    eq, mf = [], []
    for c in itertools.product((0, 1, 2), repeat=4):
        hh = [H[(c[i], FOLDS[i])] for i in range(4)]
        eq.append(ap_equal_fpr(hh))
        mf.append(np.mean([per[c[i], i] for i in range(4)]))
    return {"fold_sd_ap": dict(zip(FOLDS, per.std(0, ddof=1).tolist())), "sd_81_equal_fpr": float(np.std(eq) * np.sqrt(1.5)), "sd_81_mean_of_folds": float(np.std(mf) * np.sqrt(1.5))}


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def _ranks(a):
    r = np.empty(len(a), np.float64)
    r[np.argsort(a, kind="stable")] = np.arange(len(a))
    return r


def pair_columns(p, q, t, thr):
    dz = _logit(q) - _logit(p)
    a, b, c = np.percentile(dz, [25, 50, 75])
    k = int(t.sum())
    top_p, top_q = np.argpartition(-p, k)[:k], np.argpartition(-q, k)[:k]
    return {"mean_abs_dp": float(np.abs(p - q).mean()), "flip_share": float(((p >= thr) != (q >= thr)).mean()), "dlogit_median": float(b), "dlogit_robust_sd": float((c - a) / 1.349), "spearman": float(np.corrcoef(_ranks(p), _ranks(q))[0, 1]), "topk_overlap": len(np.intersect1d(top_p, top_q)) / k}


def outputs_by_fold(arm, blank, sub, circ_ids):
    """Every 4th pixel, the AP's pixels: inside the set, CIRCUNVALACION left out."""
    res = {}
    for f in FOLDS:
        a, b = ROOT / arm / f"fold{f}_seed0", ROOT / blank / f"fold{f}_seed0"
        meta = json.load(open(a / "prob_oof.json"))
        r0, c0, H, W_ = meta["region_rows_cols"]
        win = Window(c0, r0, W_, H)
        with rasterio.open(a / "prob_oof.tif") as s:
            p = s.read(1)[::4, ::4]
        with rasterio.open(b / "prob_oof.tif") as s:
            q = s.read(1)[::4, ::4]
        with rasterio.open("derived/ugi_labels/ugi_binary_grid1m.tif") as s:
            ub = s.read(1, window=win)[::4, ::4] > 0
        with rasterio.open("derived/ugi_labels/ugi_id_grid1m.tif") as s:
            ids = s.read(1, window=win)[::4, ::4]
        mask = "derived/ugi_centre/sq8km_grid1m.tif" if sub == "centre_sq8km" else "derived/ugi_labels/ejido_grid1m.tif"
        with rasterio.open(mask) as s:
            m = s.read(1, window=win)[::4, ::4] == 1
        if sub == "ejido_b4":
            from itareo.ugi_seg.rescore9 import b4_mask
            m &= b4_mask(r0, c0, H, W_)[::4, ::4]
        m &= ~(ub & np.isin(ids, circ_ids)) & np.isfinite(p) & np.isfinite(q)
        res[f] = pair_columns(p[m], q[m], ub[m], meta["threshold"])
    return res


def vegetated(arm, sub):
    res = {}
    for f in FOLDS:
        v = json.load(open(ROOT / arm / f"fold{f}_seed0" / sub / "metrics_rule7.json"))["rule4"]["no_circunvalacion"]["veg"]
        res[f] = (v["ap"], v["auroc"])
    return res


def fmt(d, n=4, sign=True):
    return ", ".join(f"{v:+.{n}f}" if sign else f"{v:.{n}f}" for v in d.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    sd = seed_sds()
    circ_ids = [int(r["id"]) for r in csv.DictReader(open("derived/ugi_labels/polygon_attributes.csv", encoding="utf-8")) if r["circunvalacion"] == "True"]
    base_sd = {sub: baseline_sds(sub) for sub in ("centre_sq8km", "ejido", "ejido_b4")}
    table, folds, res = [], [], {}
    for tag, (arm, blank, subs) in ARMS.items():
        res[tag] = {}
        for sub in subs:
            x, y = scores(arm, sub), scores(blank, sub)
            dap, dau = y["ap"] - x["ap"], y["auroc"] - x["auroc"]
            d = json.load(open(S / f"summary_d2_{tag}_{sub}.json"))["paired_differences"][f"{blank} - {arm}"]["ap_nocirc"]
            mean_dp, flip = change(arm, blank, sub)
            fold_d = {f: y["per_fold"][f][1] - x["per_fold"][f][1] for f in FOLDS}
            ha, hb = [fold_hist(arm, f, 0, sub) for f in FOLDS], [fold_hist(blank, f, 0, sub) for f in FOLDS]
            eq_a, eq_b = ap_equal_fpr(ha), ap_equal_fpr(hb)
            out_f = outputs_by_fold(arm, blank, sub, circ_ids)
            sh = [(shift(h[0], out_f[f]["dlogit_median"]), shift(h[1], out_f[f]["dlogit_median"])) for f, h in zip(FOLDS, ha)]
            offsets_only = auroc_ap_from_hist(sum(h[0] for h in sh), sum(h[1] for h in sh))[1] - auroc_ap_from_hist(sum(h[0] for h in ha), sum(h[1] for h in ha))[1]
            fsd = base_sd[sub]["fold_sd_ap"]
            beyond = sum(abs(fold_d[f]) > fsd[f] for f in FOLDS)
            below_ap, below_au = bool(abs(dap) < sd[sub][0]), bool(abs(dau) < sd[sub][1])
            still = all(out_f[f]["mean_abs_dp"] < STILL["mean_abs_dp"] and out_f[f]["spearman"] >= STILL["spearman"] and out_f[f]["topk_overlap"] >= STILL["topk_overlap"] for f in FOLDS)
            skill_below = below_ap and below_au and beyond == 0
            reading = "did not use the input (13.8)" if skill_below and still else ("uses the input, no measurable loss of skill without it" if skill_below else "uses the input or what it learned from its masks; removal changes skill")
            va, vb = vegetated(arm, sub), vegetated(blank, sub)
            row = {"experiment": tag, "pixels": LABEL[sub], "ap_as_run": round(x["ap"], 4), "ap_blanked": round(y["ap"], 4), "ap_change": round(dap, 4), "ap_change_ci95_block": f"[{d['ci95_block'][0]:+.4f}, {d['ci95_block'][1]:+.4f}]", "seed_sd_ap": sd[sub][0], "ap_change_below_seed_sd": bool(abs(dap) < sd[sub][0]),
                   "mean_of_fold_ap_as_run": round(float(np.mean([x["per_fold"][f][1] for f in FOLDS])), 4), "mean_of_fold_ap_blanked": round(float(np.mean([y["per_fold"][f][1] for f in FOLDS])), 4), "mean_of_fold_ap_change": round(float(np.mean(list(fold_d.values()))), 4),
                   "auroc_as_run": round(x["auroc"], 4), "auroc_blanked": round(y["auroc"], 4), "auroc_change": round(dau, 4), "seed_sd_auroc": sd[sub][1], "auroc_change_below_seed_sd": bool(abs(dau) < sd[sub][1]), "iou_as_run": round(x["iou"], 4), "iou_blanked": round(y["iou"], 4), "mean_abs_change_of_p": round(mean_dp, 5), "share_of_pixels_changing_side_of_threshold": round(flip, 5),
                   "ap_change_by_fold": fmt(fold_d), "fold_seed_sd_ap": fmt(fsd, sign=False), "folds_beyond_seed_sd": int(beyond), "ap_change_mean_abs_of_folds": round(float(np.mean([abs(v) for v in fold_d.values()])), 4), "ap_equal_fpr_as_run": round(eq_a, 4), "ap_equal_fpr_blanked": round(eq_b, 4), "ap_change_equal_fpr": round(eq_b - eq_a, 4), "sd_81_equal_fpr": round(base_sd[sub]["sd_81_equal_fpr"], 4), "sd_81_mean_of_folds": round(base_sd[sub]["sd_81_mean_of_folds"], 4), "ap_change_fold_offsets_only": round(float(offsets_only), 4),
                   "dlogit_median_by_fold": fmt({f: out_f[f]["dlogit_median"] for f in FOLDS}, 3), "dlogit_robust_sd_by_fold": fmt({f: out_f[f]["dlogit_robust_sd"] for f in FOLDS}, 3, False), "spearman_by_fold": fmt({f: out_f[f]["spearman"] for f in FOLDS}, 4, False), "topk_overlap_by_fold": fmt({f: out_f[f]["topk_overlap"] for f in FOLDS}, 4, False), "mean_abs_dp_by_fold": fmt({f: out_f[f]["mean_abs_dp"] for f in FOLDS}, 4, False), "flip_share_by_fold": fmt({f: out_f[f]["flip_share"] for f in FOLDS}, 4, False),
                   "vegetated_ap_change_by_fold": fmt({f: vb[f][0] - va[f][0] for f in FOLDS}), "vegetated_auroc_change_by_fold": fmt({f: vb[f][1] - va[f][1] for f in FOLDS}), "reading": reading}
            table.append(row)
            res[tag][sub] = row
            for f in FOLDS:
                folds.append({"experiment": tag, "pixels": LABEL[sub], "fold": f, "ap_as_run": round(x["per_fold"][f][1], 5), "ap_blanked": round(y["per_fold"][f][1], 5), "ap_change": round(fold_d[f], 5), "auroc_as_run": round(x["per_fold"][f][0], 5), "auroc_blanked": round(y["per_fold"][f][0], 5), "auroc_change": round(y["per_fold"][f][0] - x["per_fold"][f][0], 5)})
    for name, rows in (("d2_table.csv", table), ("d2_folds.csv", folds)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump(res, open(out / "d2.json", "w"), indent=1, default=float)
    L = ["# D2 (CLAUDE.md 13.8): the added input blanked at inference, seed 0", "", "Reading of 13.8: a change below the seed SD means the model did not use the input; the column reading prints those words only where the outputs support them (below). Blanked = the same weights and threshold, the added input at its no-data value in every window (`scripts/diag_d2.py`; self-test `../d2_selftest_v2/d2_selftest.json`). Runs: `runs_itareo/ugi/diag_d2_e1b_blank/`, `diag_d2_e2_blank/`. Files: `d2_table.csv`, `d2_folds.csv`. The seed SD of the B4 tiles is the ejido's, for want of its own.", "",
         "| experiment | pixels | AP as run | blanked | change | block interval | seed SD | AP, mean of folds, change | AUROC as run | blanked | change | seed SD | IoU as run | blanked | mean abs change of P | side of threshold changes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['experiment']} | {r['pixels']} | {r['ap_as_run']:.4f} | {r['ap_blanked']:.4f} | {r['ap_change']:+.4f} | {r['ap_change_ci95_block']} | {r['seed_sd_ap']:.4f} | {r['mean_of_fold_ap_change']:+.4f} | {r['auroc_as_run']:.4f} | {r['auroc_blanked']:.4f} | {r['auroc_change']:+.4f} | {r['seed_sd_auroc']:.4f} | {r['iou_as_run']:.4f} | {r['iou_blanked']:.4f} | {r['mean_abs_change_of_p']:.4f} | {100 * r['share_of_pixels_changing_side_of_threshold']:.2f} % |")
    L += ["", "| experiment | pixels | reading | below seed SD, AP and AUROC (13.8) | AP change by fold | fold seed SDs | folds beyond | mean abs of folds | AP at equal FPR, change (81-based SD) | mean of folds, 81-based SD | pooled change from fold offsets alone |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['experiment']} | {r['pixels']} | {r['reading']} | {r['ap_change_below_seed_sd']}, {r['auroc_change_below_seed_sd']} | {r['ap_change_by_fold']} | {r['fold_seed_sd_ap']} | {r['folds_beyond_seed_sd']} of 4 | {r['ap_change_mean_abs_of_folds']:.4f} | {r['ap_change_equal_fpr']:+.4f} ({r['sd_81_equal_fpr']:.4f}) | {r['sd_81_mean_of_folds']:.4f} | {r['ap_change_fold_offsets_only']:+.4f} |")
    L += ["", "| experiment | pixels | mean abs change of P by fold | side of threshold changes by fold | median logit change by fold | robust SD of it | Spearman | top-k overlap | vegetated AP change by fold | vegetated AUROC change by fold |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['experiment']} | {r['pixels']} | {r['mean_abs_dp_by_fold']} | {r['flip_share_by_fold']} | {r['dlogit_median_by_fold']} | {r['dlogit_robust_sd_by_fold']} | {r['spearman_by_fold']} | {r['topk_overlap_by_fold']} | {r['vegetated_ap_change_by_fold']} | {r['vegetated_auroc_change_by_fold']} |")
    L += ["", NOTES.read_text(encoding="utf-8").rstrip()]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
