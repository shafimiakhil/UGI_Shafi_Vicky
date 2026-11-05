"""Stage R summaries (CLAUDE.md 13.3, rule 2): pooled scores per arm with spatial block-bootstrap intervals.

Per run (runs_itareo/ugi/<config>/fold<f>_seed<s>/): boot.npz (per 1 km block of GRID_1M: valid, inventory, predicted
pixels, tp/fp/fn and coarse logit histograms of all and of vegetated pixels; full-resolution histograms of the run),
objects.csv (polygons with their block by centroid, strict and tolerant flags), metrics.json. Blocks partition the grid,
so the four folds of one seed are pooled by summing per block.
  * Pooled values: AP and AUROC from the summed full-resolution histograms; IoU from the summed confusion counts; strict
    and tolerant object recall over the concatenated polygons; vegetated AP from the vegetated histograms.
  * Block bootstrap: 2,000 resamples of the 1 km blocks with replacement (seed 0), the same resamples for every arm
    (paired); pixels and polygons go with their block; AP from the coarse block histograms (1,000 logit bins), shifted by
    the arm's fine-minus-coarse AP of the full sample (coarse bins make ties that lower AP; the offset is recorded),
    IoU and recall from the block counts; 95 % interval = 2.5 and 97.5 percentiles. Differences between arms are computed per
    resample. With several seeds the per-seed values are averaged inside each resample. Values are stored with 7 decimals, so
    that a report printing 3 or 4 does not round a rounded value.
  * The polygon-level bootstrap of the A1/A2 round, if shown, is labelled as such and shown only next to this one.
  * Rule 7 (CLAUDE.md 13.4), when the runs carry boot_rule7.npz and objects_rule7.csv (itareo.ugi_seg.rescore7): AP with
    CIRCUNVALACION pixels excluded (ap_nocirc), and strict and tolerant recall of polygons under 6 m wide, with the same
    block bootstrap. Also IoU and strict and tolerant object recall without CIRCUNVALACION (iou_nocirc,
    object_recall_strict_nocirc, object_recall_tolerant_nocirc; 13.5).
  * --ejido (rule 9, CLAUDE.md 13.5) reads each run's ejido/ subfolder (itareo.ugi_seg.rescore9): the same quantities on the
    pixels inside the ejido; the blocks resampled are those with pixels inside it. Without it, the full grid.
  * --decision applies the R1 rule of CLAUDE.md 13.4, written before the runs: the highest pooled ap_nocirc; within
    0.005 of it, the higher tolerant object recall over all polygons. --r1b compares ugi_r1b_both_e120 with ugi_r1_both
    on fold 1 seed 0 by ap_nocirc: below +0.01, 60 epochs stay; otherwise report and wait.
  python -m itareo.ugi_seg.summarize_r --arms ugi_r1_both ugi_r1_outer ugi_r1_none --seeds 0 [--folds 1 2 3 4]
       [--out runs_itareo/ugi/summary_r] [--tag r1] [--ejido]        (from work_ms/; refuses an existing summary_<tag>)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from itareo.safe_write import refuse_existing
from itareo.unet_ugi.metrics import auroc_ap_from_hist

ROOT = Path("runs_itareo/ugi")
NBOOT = 2000
SUB = [""]                  # "ejido": read each run's ejido/ subfolder (rule 9) instead of the run folder


def load(arm, fold, seed):
    d = ROOT / arm / f"fold{fold}_seed{seed}" / SUB[0]
    if not (d / "boot.npz").exists():
        return None
    b = dict(np.load(d / "boot.npz"))
    objs = list(csv.DictReader(open(d / "objects.csv")))
    out = {"boot": b, "objects": objs, "metrics": json.load(open(d / "metrics.json")), "dir": str(d)}
    if (d / "boot_rule7.npz").exists():
        out["boot7"] = dict(np.load(d / "boot_rule7.npz"))
        out["objects7"] = list(csv.DictReader(open(d / "objects_rule7.csv")))
    return out


def pooled_arm_seed(runs):
    b0 = runs[0]["boot"]
    nb = b0["block_valid"].shape[0]
    blk = {k: np.zeros(nb, np.int64) for k in ("valid", "tp", "fp", "fn", "inv", "pred")}
    bh = np.zeros((2, nb, b0["block_hist_pos"].shape[1]), np.int64)
    bhv = np.zeros_like(bh)
    hf, hv = np.zeros((2, b0["hist_full_pos"].shape[0]), np.int64), np.zeros((2, b0["hist_veg_pos"].shape[0]), np.int64)
    for r in runs:
        b = r["boot"]
        for k in blk:
            blk[k] += b[f"block_{k}"]
        bh[0] += b["block_hist_pos"]
        bh[1] += b["block_hist_neg"]
        bhv[0] += b["block_hist_veg_pos"]
        bhv[1] += b["block_hist_veg_neg"]
        hf += np.stack([b["hist_full_pos"], b["hist_full_neg"]])
        hv += np.stack([b["hist_veg_pos"], b["hist_veg_neg"]])
    objs = [o for r in runs for o in r["objects"]]
    ob = np.array([int(o["block"]) for o in objs])
    fs = np.array([o["found_strict"] == "True" for o in objs])
    ft = np.array([o["found_tolerant"] == "True" for o in objs])
    auroc, ap = auroc_ap_from_hist(hf[0], hf[1])
    _, ap_veg = auroc_ap_from_hist(hv[0], hv[1])
    tp, fp, fn = blk["tp"].sum(), blk["fp"].sum(), blk["fn"].sum()
    point = {"ap": ap, "auroc": auroc, "ap_vegetated": ap_veg, "iou": tp / (tp + fp + fn),
             "object_recall_strict": float(fs.mean()), "object_recall_tolerant": float(ft.mean()), "objects": len(objs)}
    per_block = {"blk": blk, "bh": bh, "bhv": bhv,
                 "obj_n": np.bincount(ob, minlength=nb), "obj_s": np.bincount(ob, weights=fs, minlength=nb),
                 "obj_t": np.bincount(ob, weights=ft, minlength=nb)}
    if all("boot7" in r for r in runs):
        bhn = np.zeros_like(bh)
        hn = np.zeros((2, b0["hist_full_pos"].shape[0]), np.int64)
        for r in runs:
            bhn[0] += r["boot7"]["block_hist_nocirc_pos"]
            bhn[1] += r["boot7"]["block_hist_nocirc_neg"]
            hn += np.stack([r["boot7"]["hist_nocirc_pos"], r["boot7"]["hist_nocirc_neg"]])
        point["ap_nocirc"] = auroc_ap_from_hist(hn[0], hn[1])[1]
        o7 = [o for r in runs for o in r["objects7"] if float(o["width_m"]) < 6]
        b6 = np.array([int(o["block"]) for o in o7])
        s6 = np.array([o["found_strict"] == "True" for o in o7])
        t6 = np.array([o["found_tolerant"] == "True" for o in o7])
        point["recall_under_6m_strict"] = float(s6.mean()) if len(o7) else None
        point["recall_under_6m_tolerant"] = float(t6.mean()) if len(o7) else None
        point["objects_under_6m"] = len(o7)
        per_block.update({"bhn": bhn, "o6_n": np.bincount(b6, minlength=nb), "o6_s": np.bincount(b6, weights=s6, minlength=nb),
                          "o6_t": np.bincount(b6, weights=t6, minlength=nb)})
        blk_nc = {k: sum(r["boot7"][f"block_nocirc_{k}"] for r in runs) for k in ("tp", "fp", "fn")}
        point["iou_nocirc"] = blk_nc["tp"].sum() / (blk_nc["tp"].sum() + blk_nc["fp"].sum() + blk_nc["fn"].sum())
        onc = [o for r in runs for o in r["objects7"] if o["circunvalacion"] != "True"]
        bo = np.array([int(o["block"]) for o in onc])
        sn = np.array([o["found_strict"] == "True" for o in onc])
        tn = np.array([o["found_tolerant"] == "True" for o in onc])
        point["object_recall_strict_nocirc"] = float(sn.mean())
        point["object_recall_tolerant_nocirc"] = float(tn.mean())
        point["objects_nocirc"] = len(onc)
        per_block.update({"blk_nc": blk_nc, "onc_n": np.bincount(bo, minlength=nb), "onc_s": np.bincount(bo, weights=sn, minlength=nb),
                          "onc_t": np.bincount(bo, weights=tn, minlength=nb)})
    ones = np.ones((1, nb))
    full = boot_values(per_block, ones)
    point["ap_coarse_offset"] = ap - full["ap"][0]
    point["ap_vegetated_coarse_offset"] = ap_veg - full["ap_vegetated"][0]
    if "bhn" in per_block:
        point["ap_nocirc_coarse_offset"] = point["ap_nocirc"] - full["ap_nocirc"][0]
    return point, per_block


def boot_values(pb, W):
    """W (B, nb) multiplicities of the blocks in each resample -> dict of arrays (B,)."""
    tp, fp, fn = (W @ pb["blk"][k].astype(float) for k in ("tp", "fp", "fn"))
    out = {"iou": tp / np.maximum(tp + fp + fn, 1)}
    for key, h in (("ap", pb["bh"]), ("ap_vegetated", pb["bhv"])):
        pos, neg = W @ h[0].astype(float), W @ h[1].astype(float)
        out[key] = np.array([auroc_ap_from_hist(pos[i], neg[i])[1] or np.nan for i in range(W.shape[0])])
    n = W @ pb["obj_n"]
    out["object_recall_strict"] = (W @ pb["obj_s"]) / np.maximum(n, 1)
    out["object_recall_tolerant"] = (W @ pb["obj_t"]) / np.maximum(n, 1)
    if "bhn" in pb:
        pos, neg = W @ pb["bhn"][0].astype(float), W @ pb["bhn"][1].astype(float)
        out["ap_nocirc"] = np.array([auroc_ap_from_hist(pos[i], neg[i])[1] or np.nan for i in range(W.shape[0])])
        n6 = W @ pb["o6_n"]
        out["recall_under_6m_strict"] = (W @ pb["o6_s"]) / np.maximum(n6, 1)
        out["recall_under_6m_tolerant"] = (W @ pb["o6_t"]) / np.maximum(n6, 1)
        tpn, fpn, fnn = (W @ pb["blk_nc"][k].astype(float) for k in ("tp", "fp", "fn"))
        out["iou_nocirc"] = tpn / np.maximum(tpn + fpn + fnn, 1)
        nn = W @ pb["onc_n"]
        out["object_recall_strict_nocirc"] = (W @ pb["onc_s"]) / np.maximum(nn, 1)
        out["object_recall_tolerant_nocirc"] = (W @ pb["onc_t"]) / np.maximum(nn, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--out", default="runs_itareo/ugi/summary_r")
    ap.add_argument("--tag", default="r")
    ap.add_argument("--ejido", action="store_true", help="scores inside the ejido (rule 9): each run's ejido/ subfolder")
    ap.add_argument("--sub", default=None, help="another subfolder of each run, e.g. ejido_b4")
    ap.add_argument("--decision", action="store_true", help="apply the R1 rule of CLAUDE.md 13.4")
    ap.add_argument("--r1b", action="store_true", help="apply the R1b rule of CLAUDE.md 13.4")
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out / f"summary_{a.tag}.json", out / f"summary_{a.tag}.csv")
    out.mkdir(parents=True, exist_ok=True)
    SUB[0] = a.sub or ("ejido" if a.ejido else "")
    data, missing = {}, []
    for arm in a.arms:
        for s in a.seeds:
            runs = [load(arm, f, s) for f in a.folds]
            if any(r is None for r in runs):
                missing.append(f"{arm} seed {s}")
                continue
            data[(arm, s)] = pooled_arm_seed(runs)
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    nb = next(iter(data.values()))[1]["blk"]["valid"].shape[0]
    used = np.nonzero(next(iter(data.values()))[1]["blk"]["valid"] > 0)[0]
    rng = np.random.default_rng(0)
    W = np.zeros((NBOOT, nb))
    for i in range(NBOOT):
        W[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)
    keys = ["ap", "ap_vegetated", "iou", "object_recall_strict", "object_recall_tolerant"]
    if all("ap_nocirc" in v[0] for v in data.values()):
        keys += ["ap_nocirc", "recall_under_6m_strict", "recall_under_6m_tolerant", "iou_nocirc",
                 "object_recall_strict_nocirc", "object_recall_tolerant_nocirc"]
    def shifted(arm, s):
        v = boot_values(data[(arm, s)][1], W)
        v["ap"] = v["ap"] + data[(arm, s)][0]["ap_coarse_offset"]
        v["ap_vegetated"] = v["ap_vegetated"] + data[(arm, s)][0]["ap_vegetated_coarse_offset"]
        if "ap_nocirc" in v:
            v["ap_nocirc"] = v["ap_nocirc"] + data[(arm, s)][0]["ap_nocirc_coarse_offset"]
        return v
    bv = {arm: {k: np.mean([shifted(arm, s)[k] for s in a.seeds], axis=0) for k in keys} for arm in a.arms}
    rows, res = [], {"reference_area": a.sub or ("inside the ejido (rule 9)" if a.ejido else "full grid"), "blocks_resampled": int(len(used)),
                     "resamples": NBOOT, "seeds": a.seeds, "folds": a.folds, "arms": {}}
    for arm in a.arms:
        pts = [data[(arm, s)][0] for s in a.seeds]
        r = {"arm": arm, "seeds": len(a.seeds),
             "ap_coarse_offset": round(float(np.mean([p["ap_coarse_offset"] for p in pts])), 5),
             "ap_vegetated_coarse_offset": round(float(np.mean([p["ap_vegetated_coarse_offset"] for p in pts])), 5)}
        if "ap_nocirc_coarse_offset" in pts[0]:
            r["ap_nocirc_coarse_offset"] = round(float(np.mean([p["ap_nocirc_coarse_offset"] for p in pts])), 5)
        for k in keys + ["auroc"]:
            v = [p[k] for p in pts]
            r[k] = round(float(np.mean(v)), 7)
            if len(v) > 1:
                r[f"{k}_sd_over_seeds"] = round(float(np.std(v, ddof=1)), 7)
            if k in bv[arm]:
                lo, hi = np.nanpercentile(bv[arm][k], [2.5, 97.5])
                r[f"{k}_ci95_block"] = [round(float(lo), 7), round(float(hi), 7)]
        res["arms"][arm] = r
        rows.append({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    res["paired_differences"] = {}
    for i, a1 in enumerate(a.arms):
        for a2 in a.arms[i + 1:]:
            d = {}
            for k in keys:
                diff = bv[a2][k] - bv[a1][k]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                d[k] = {"point": round(res["arms"][a2][k] - res["arms"][a1][k], 7), "ci95_block": [round(float(lo), 7), round(float(hi), 7)]}
            res["paired_differences"][f"{a2} - {a1}"] = d
    if a.decision:
        best = max(res["arms"].values(), key=lambda r: r["ap_nocirc"])
        close = [r for r in res["arms"].values() if round(best["ap_nocirc"] - r["ap_nocirc"], 5) <= 0.005]   # rounded: no float flip at the boundary
        win = max(close, key=lambda r: r["object_recall_tolerant"])
        res["r1_decision"] = {"rule": "highest pooled AP with CIRCUNVALACION excluded; within 0.005, the higher tolerant "
                                      "object recall over all polygons (CLAUDE.md 13.4, written before the runs)",
                              "highest_ap_nocirc": best["arm"], "within_0.005": [r["arm"] for r in close],
                              "winner": win["arm"]}
        if a.ejido:
            res["r1_decision"]["note"] = ("the same rule inside the ejido, for comparison only; the decision of record is the "
                                          "full-grid one (summary_r1.json), as written in CLAUDE.md 13.4")
    if a.r1b:
        def ap_nc(cfg):
            return json.load(open(ROOT / cfg / "fold1_seed0" / SUB[0] / "metrics_rule7.json"))["pixel"]["no_circunvalacion"]["ap"]
        a60, a120 = ap_nc("ugi_r1_both"), ap_nc("ugi_r1b_both_e120")
        gain = round(a120 - a60, 5)
        res["r1b_decision"] = {"ap_nocirc_60_epochs": a60, "ap_nocirc_120_epochs": a120, "gain": gain,
                               "rule": "if 120 epochs add less than 0.01 held-out AP (CIRCUNVALACION excluded), 60 stays; "
                                       "otherwise report and wait (CLAUDE.md 13.4)",
                               "decision": "60 epochs stay" if gain < 0.01 else "report and wait"}
    json.dump(res, open(out / f"summary_{a.tag}.json", "w"), indent=1)
    with open(out / f"summary_{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
