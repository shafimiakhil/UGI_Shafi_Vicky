"""The seed SD inside a candidate sets the AP threshold of 13.7, so its parts are written out: per seed the pooled AP without CIRCUNVALACION, inside each candidate and in the ejido, and per seed and fold; and, as an independent check on the histograms, the AP of one seed inside a candidate computed directly from the stored prob_oof.tif with scikit-learn on every 4th pixel.
Writes docs/evidence_phase2/ugi_centre/seed_values.csv and seed_check.json. Run from work_ms/ (CPU): python scripts/centre_seed_check.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.unet_ugi.metrics import auroc_ap_from_hist  # noqa: E402

RUNS = WM / "runs_itareo/ugi/ugi_r1_none"
DOC = WM.parent / "docs/evidence_phase2/ugi_centre"
SETS = ["ejido", "centre_ring", "centre_sq3km", "centre_sq5km", "centre_sq8km"]


def pooled_ap(seed, sub, folds=(1, 2, 3, 4)):
    h = None
    for f in folds:
        b = np.load(RUNS / f"fold{f}_seed{seed}" / sub / "boot_rule7.npz")
        x = np.stack([b["hist_nocirc_pos"], b["hist_nocirc_neg"]]).astype(np.float64)
        h = x if h is None else h + x
    return auroc_ap_from_hist(h[0], h[1])


def direct_ap(seed, name):
    from sklearn.metrics import average_precision_score
    reg = json.load(open(WM / "derived/ugi_centre/regions.json"))[name]
    ys, ps = [], []
    with rasterio.open(WM / f"derived/ugi_centre/{name}_grid1m.tif") as sm, rasterio.open(WM / "derived/ugi_labels/ugi_binary_grid1m.tif") as sl, rasterio.open(WM / "derived/ugi_labels/ejido_grid1m.tif") as se:
        for f in (1, 2, 3, 4):
            run = RUNS / f"fold{f}_seed{seed}"
            r0, c0, H, W = json.load(open(run / "prob_oof.json"))["region_rows_cols"]
            a0, a1 = max(r0, reg["row0"]), min(r0 + H, reg["row0"] + reg["rows"])
            b0, b1 = max(c0, reg["col0"]), min(c0 + W, reg["col0"] + reg["cols"])
            with rasterio.open(run / "prob_oof.tif") as s:
                p = s.read(1, window=Window(b0 - c0, a0 - r0, b1 - b0, a1 - a0))
            w = Window(b0, a0, b1 - b0, a1 - a0)
            m = (sm.read(1, window=w) == 1) & (se.read(1, window=w) == 1) & np.isfinite(p)
            y = sl.read(1, window=w) == 1
            ys.append(y[::4, ::4][m[::4, ::4]])
            ps.append(p[::4, ::4][m[::4, ::4]])
    return float(average_precision_score(np.concatenate(ys), np.concatenate(ps)))


def main():
    rows = []
    res = {}
    for sub in SETS:
        v = []
        for s in (0, 1, 2):
            au, ap = pooled_ap(s, sub)
            v.append(ap)
            row = {"region": sub, "seed": s, "fold": "pooled", "ap_nocirc": round(ap, 5), "auroc_nocirc": round(au, 5)}
            rows.append(row)
            for f in (1, 2, 3, 4):
                au1, ap1 = pooled_ap(s, sub, (f,))
                rows.append({"region": sub, "seed": s, "fold": f, "ap_nocirc": round(ap1, 5), "auroc_nocirc": round(au1, 5)})
        res[sub] = {"ap_nocirc_per_seed": [round(x, 5) for x in v], "mean": round(float(np.mean(v)), 5), "sd": round(float(np.std(v, ddof=1)), 5), "range": round(float(max(v) - min(v)), 5)}
    res["direct_check_seed0_ring_every_4th_pixel"] = {"sklearn_ap": round(direct_ap(0, "ring"), 5), "histogram_ap": round(pooled_ap(0, "centre_ring")[1], 5)}
    res["direct_check_seed1_sq5km_every_4th_pixel"] = {"sklearn_ap": round(direct_ap(1, "sq5km"), 5), "histogram_ap": round(pooled_ap(1, "centre_sq5km")[1], 5)}
    with open(DOC / "seed_values.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(res, open(DOC / "seed_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
