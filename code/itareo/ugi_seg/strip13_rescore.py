"""Strip 1_3 before and after v3 (CLAUDE.md 13.4, F1): inference only, labels unchanged.

The A1 30-epoch checkpoints whose held-out quadrant contains strip 1_3 (runs_itareo/ugi/ugi_a1/fold{1,3}_seed{0,1,2})
are run with the sliding-window inference of evaluate.predict_region over the same region on the v1 mosaic
(derived/pneo2022_registered/) and on the v3 mosaic (derived/pneo2022_registered_v3/); the region is the bounding box of
the scored pixels, so only the image differs. Scored pixels: inside the checkpoint's held-out quadrant, strip 1_3 in both
mosaics (strip-id 3 in v1 and in v3) and valid in both images. Per checkpoint AUROC and AP (logit histograms) and IoU at
its own threshold; pooled over the two folds per seed; mean and SD over the three seeds; and the three-seed mean
probability pooled over folds.
Output: docs/evidence_phase2/ugi_seg/strip13_v3_rescore.json and strip13_v3_rescore.csv. Run from work_ms/.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins

from .evaluate import predict_region, pixel_scores
from .model import build
from .oof import quadrant

V1 = Path("derived/pneo2022_registered")
V3 = Path("derived/pneo2022_registered_v3")
OUT = Path("../docs/evidence_phase2/ugi_seg")
FOLDS = {"x_line": 388282.5, "y_line": 6525025.5}


def main(device="cuda"):
    t0 = time.time()
    img1, img3 = V1 / "pneo2022_toa_grid1m.tif", V3 / "pneo2022_toa_grid1m.tif"
    with rasterio.open(V1 / "pneo2022_strip_id_grid1m.tif") as s:
        sid1 = s.read(1)
    with rasterio.open(V3 / "pneo2022_strip_id_grid1m.tif") as s:
        sid3 = s.read(1)
    with rasterio.open("derived/ugi_labels/ugi_binary_grid1m.tif") as s:
        ref_all = s.read(1).astype(bool)
    both = (sid1 == 3) & (sid3 == 3)
    rows, acc = [], {}
    for fold in (1, 2, 3, 4):
        in_r, in_c, _ = quadrant(str(img1), FOLDS, fold)
        m = both & in_r[:, None] & in_c[None, :]
        if not m.any():
            continue
        rr, cc = np.nonzero(m)
        r0, c0, R, C = int(rr.min()), int(cc.min()), int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)
        msk = m[r0:r0 + R, c0:c0 + C]
        ref = ref_all[r0:r0 + R, c0:c0 + C]
        for seed in (0, 1, 2):
            d = Path(f"runs_itareo/ugi/ugi_a1/fold{fold}_seed{seed}")
            ck = torch.load(d / "best.pt", map_location=device, weights_only=False)
            model = build("imagenet")
            model.load_state_dict(ck["model"])
            model = model.to(device)
            probs, oks = {}, {}
            for name, img in (("v1", img1), ("v3", img3)):
                p = np.full((R, C), np.nan, np.float32)
                for b0, pu, _ in predict_region(model, str(img), (r0, c0, R, C), ck["mean"], ck["sd"], device=device):
                    p[b0:b0 + pu.shape[0]] = pu
                with rasterio.open(img) as s:
                    oks[name] = (s.read(window=Window(c0, r0, C, R)) != s.nodata).all(0)
                probs[name] = p
            sel = msk & oks["v1"] & oks["v3"]
            for name in ("v1", "v3"):
                p, t = probs[name][sel], ref[sel]
                b = prob_bins(p)
                h = np.stack([np.bincount(b[t], minlength=NBINS), np.bincount(b[~t], minlength=NBINS)])
                pr = p >= ck["threshold"]
                conf = [int(np.sum(t & pr)), int(np.sum(~t & pr)), int(np.sum(t & ~pr)), int(np.sum(~t & ~pr))]
                a, ap = auroc_ap_from_hist(h[0], h[1])
                rows.append({"fold": fold, "seed": seed, "mosaic": name, "pixels": int(sel.sum()), "ugi_pixels": int(t.sum()),
                             "auroc": round(a, 5), "ap": round(ap, 5), "iou": pixel_scores(*conf)["iou"],
                             "threshold": ck["threshold"], "region_rows_cols": [r0, c0, R, C]})
                k = (seed, name)
                acc.setdefault(k, {"h": np.zeros((2, NBINS), np.int64), "conf": np.zeros(4, np.int64)})
                acc[k]["h"] += h
                acc[k]["conf"] += conf
                e = ("ens", fold, name)
                acc.setdefault(e, {"p": 0, "t": t, "n": 0})
                acc[e]["p"] = acc[e]["p"] + p
                acc[e]["n"] += 1
            print(fold, seed, [r for r in rows[-2:]], round(time.time() - t0), "s", flush=True)
    res = {"pixels": "held-out quadrant, strip 1_3 in v1 and in v3, valid in both images", "per_seed_pooled": {}, "ensemble": {}}
    for name in ("v1", "v3"):
        vals = []
        for seed in (0, 1, 2):
            a = acc[(seed, name)]
            au, ap = auroc_ap_from_hist(*a["h"])
            vals.append({"seed": seed, "auroc": au, "ap": ap, "iou": pixel_scores(*a["conf"])["iou"]})
        res["per_seed_pooled"][name] = {
            "seeds": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in x.items()} for x in vals],
            **{f"{k}_mean": round(float(np.mean([x[k] for x in vals])), 5) for k in ("auroc", "ap", "iou")},
            **{f"{k}_sd": round(float(np.std([x[k] for x in vals], ddof=1)), 5) for k in ("auroc", "ap", "iou")}}
        h = np.zeros((2, NBINS), np.int64)
        for f in (1, 2, 3, 4):
            e = acc.get(("ens", f, name))
            if e is None:
                continue
            pm = e["p"] / e["n"]
            b = prob_bins(pm)
            h += np.stack([np.bincount(b[e["t"]], minlength=NBINS), np.bincount(b[~e["t"]], minlength=NBINS)])
        au, ap = auroc_ap_from_hist(*h)
        res["ensemble"][name] = {"auroc": round(au, 5), "ap": round(ap, 5), "pixels": int(h.sum()), "ugi_pixels": int(h[0].sum())}
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "strip13_v3_rescore.json", "w"), indent=1)
    with open(OUT / "strip13_v3_rescore.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
