"""Matched evaluation (CLAUDE.md 13.2): re-score trained UGI checkpoints on tile subsets, without retraining.

One sliding-window inference per checkpoint (itareo.ugi_seg.evaluate.predict_region, same windows, blending and threshold
as the run), scored on several pixel subsets of the held-out quadrant:
  all  every evaluated pixel (as in the run; adds the probability histograms for pooled AUROC/AP);
  b4   pixels of the non-straddling tiles with all three PlanetScope dates >= 90 % valid
       (derived/ugi_tiles/tiles_b4_planetscope.csv, avail_ps_all3 >= 0.9; 1,240 tiles);
  b8   pixels of the tiles of the B8 cache (runs_itareo/ugi/cache/b8_ple2017_neo2022/tiles.csv; 1,587 tiles).
Per subset: pixel scores (full and without the edge band) with AUROC and AP, object recall (polygons with >= 50 % of
their pixels in the subset; detected at >= 50 % coverage) overall, by group and by size, type accuracy and macro F1, and
the boundary F-score within 2 m (not per strip). Also per Neo strip of the image (registered strip id, pixel and object
scores). Outputs per run in
<run>/rescore/: metrics_<subset>.json, objects_<subset>.csv, hist_<subset>.npz, and done.json last (a run is
skipped only when done.json exists).
  python -m itareo.ugi_seg.rescore --runs runs_itareo/ugi/ugi_a1 runs_itareo/ugi/ugi_a2   (from work_ms/)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from itareo.unet_ugi.data import read_csv
from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins

from .evaluate import GROUP_NAMES, SIZE_EDGES, SIZE_NAMES, boundary_f, pixel_scores, predict_region
from .model import build

SUBSETS = {"b4": ("derived/ugi_tiles/tiles_b4_planetscope.csv", lambda t: not t["straddle"] and t["avail_ps_all3"] >= 0.9),
           "b8": ("runs_itareo/ugi/cache/b8_ple2017_neo2022/tiles.csv", lambda t: True)}
STRIPS = {2: "1_2", 3: "1_3", 5: "1_5", 6: "1_6"}
SID = "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"


def tile_mask_full(shape, tiles, T=512):
    m = np.zeros(shape, bool)
    for t in tiles:
        m[int(t["row0"]):int(t["row0"]) + T, int(t["col0"]):int(t["col0"]) + T] = True
    return m


def new_acc(nid=4000):
    return {"conf": {"full": np.zeros(4, np.int64), "no_edge": np.zeros(4, np.int64)},
            "hist": {k: {"pos": np.zeros(NBINS, np.int64), "neg": np.zeros(NBINS, np.int64)} for k in ("full", "no_edge")},
            "poly_px": np.zeros(nid, np.int64), "poly_pos": np.zeros(nid, np.int64), "poly_g": np.zeros((nid, 5))}


def finish(acc, pred, refm, valid, polys, with_boundary=True):
    res = {}
    for key in acc["conf"]:
        s = pixel_scores(*acc["conf"][key])
        a, ap = auroc_ap_from_hist(acc["hist"][key]["pos"], acc["hist"][key]["neg"])
        s.update({"auroc": None if a is None else round(a, 5), "ap": None if ap is None else round(ap, 5)})
        res[f"pixel_{key}"] = s
    if with_boundary:
        res.update(boundary_f(pred & valid, refm & valid, valid))
    objs = []
    for pid, r in polys.items():
        total = int(r["pixels"])
        if total == 0 or acc["poly_px"][pid] < 0.5 * total:
            continue
        cov = acc["poly_pos"][pid] / acc["poly_px"][pid]
        objs.append({"id": pid, "group": int(r["group"]), "area_m2": float(r["area_m2"]), "coverage": round(float(cov), 4),
                     "detected": bool(cov >= 0.5), "pred_group": int(np.argmax(acc["poly_g"][pid]) + 1)})
    det = np.array([o["detected"] for o in objs])
    res["objects_evaluated"] = len(objs)
    res["object_recall"] = round(float(det.mean()), 5) if len(det) else None
    res["object_recall_by_group"] = {n: {"n": len(d), "recall": round(float(np.mean(d)), 5) if d else None}
                                     for g, n in enumerate(GROUP_NAMES + ["NO DEFINIDO"], 1)
                                     for d in [[o["detected"] for o in objs if o["group"] == g]]}
    res["object_recall_by_size"] = {n: {"n": len(d), "recall": round(float(np.mean(d)), 5) if d else None}
                                    for k, n in enumerate(SIZE_NAMES)
                                    for d in [[o["detected"] for o in objs if SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1]]]}
    typed = [o for o in objs if o["detected"] and 1 <= o["group"] <= 5]
    if typed:
        t = np.array([o["group"] for o in typed])
        p = np.array([o["pred_group"] for o in typed])
        f1s = []
        for g in sorted(set(t)):
            tp = np.sum((t == g) & (p == g))
            pr, rc = tp / max(np.sum(p == g), 1), tp / max(np.sum(t == g), 1)
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        res.update({"type_polygons": len(typed), "type_accuracy": round(float((t == p).mean()), 5),
                    "type_macro_f1": round(float(np.mean(f1s)), 5)})
    return res, objs


def rescore_run(run, device="cuda"):
    t0 = time.time()
    run = Path(run)
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    fold = int(run.name.split("_")[0].replace("fold", ""))
    model = build("imagenet")      # architecture only; the weights come from the checkpoint
    model.load_state_dict(ck["model"])
    model = model.to(device)
    with rasterio.open(cfg["paths"]["image"]) as src:
        Tr, Hh, Ww = src.transform, src.height, src.width
    xl, yl = cfg["folds"]["x_line"], cfg["folds"]["y_line"]
    left = Tr.c + (np.arange(Ww) + 0.5) * Tr.a <= xl
    south = Tr.f + (np.arange(Hh) + 0.5) * Tr.e <= yl
    in_c = left if fold in (1, 3) else ~left
    in_r = south if fold in (1, 2) else ~south
    ci, ri = np.nonzero(in_c)[0], np.nonzero(in_r)[0]
    r_lo, c_lo = int(ri.min()), int(ci.min())
    R, Cn = int(ri.max() - ri.min() + 1), int(ci.max() - ci.min() + 1)
    sub_masks = {"all": np.ones((R, Cn), bool)}
    for name, (path, keep) in SUBSETS.items():
        full = tile_mask_full((Hh, Ww), [t for t in read_csv(path) if keep(t)])
        sub_masks[name] = full[r_lo:r_lo + R, c_lo:c_lo + Cn].copy()
        del full
    strip_names = list(STRIPS.values())
    accs = {k: new_acc() for k in list(sub_masks) + [f"strip_{s}" for s in strip_names]}
    pred = np.zeros((R, Cn), bool)
    refm = np.zeros((R, Cn), bool)
    valid = np.zeros((R, Cn), bool)
    sidm = np.zeros((R, Cn), np.uint8)
    lab = Path(cfg["paths"]["labels"])
    thr = ck["threshold"]
    with rasterio.open(cfg["paths"]["image"]) as im, rasterio.open(lab / "ugi_binary_grid1m.tif") as ub, \
            rasterio.open(lab / "ugi_edge_grid1m.tif") as ue, rasterio.open(lab / "ugi_id_grid1m.tif") as ui, \
            rasterio.open(SID) as sp:
        for b0, pu, pg in predict_region(model, cfg["paths"]["image"], (r_lo, c_lo, R, Cn), ck["mean"], ck["sd"],
                                         n=cfg["eval"]["window"], band_rows=cfg["eval"]["band_rows"], device=device):
            rows = pu.shape[0]
            w = Window(c_lo, r_lo + b0, Cn, rows)
            ok = (im.read(window=w) != im.nodata).all(0) & in_r[r_lo + b0:r_lo + b0 + rows][:, None] & in_c[c_lo:c_lo + Cn][None, :]
            ref = ub.read(1, window=w).astype(bool)
            edge = ue.read(1, window=w).astype(bool)
            ids = ui.read(1, window=w).astype(np.int64)
            sid = sp.read(1, window=w)
            pp = pu >= thr
            sl = slice(b0, b0 + rows)
            pred[sl], refm[sl], valid[sl], sidm[sl] = pp & ok, ref & ok, ok, sid
            for name in accs:
                if name.startswith("strip_"):
                    code = [c for c, s in STRIPS.items() if s == name[6:]][0]
                    m_sub = ok & (sid == code)
                else:
                    m_sub = ok & sub_masks[name][sl]
                a = accs[name]
                for key, m in (("full", m_sub), ("no_edge", m_sub & ~edge)):
                    t, pr = ref[m], pp[m]
                    a["conf"][key] += [np.sum(t & pr), np.sum(~t & pr), np.sum(t & ~pr), np.sum(~t & ~pr)]
                    bins = prob_bins(pu[m])
                    a["hist"][key]["pos"] += np.bincount(bins[t], minlength=NBINS)
                    a["hist"][key]["neg"] += np.bincount(bins[~t], minlength=NBINS)
                sel = m_sub & (ids > 0)
                ii = ids[sel]
                a["poly_px"] += np.bincount(ii, minlength=4000)
                a["poly_pos"] += np.bincount(ii, weights=pp[sel], minlength=4000).astype(np.int64)
                for g in range(5):
                    a["poly_g"][:, g] += np.bincount(ii, weights=pg[g][sel], minlength=4000)
    polys = {int(r["ID_NUEVO"]): r for r in csv.DictReader(open(cfg["paths"]["polygons"], encoding="utf-8"))}
    out = run / "rescore"
    out.mkdir(exist_ok=True)
    summary = {}
    for name, a in accs.items():
        if name.startswith("strip_"):
            m = valid & (sidm == [c for c, s in STRIPS.items() if s == name[6:]][0])
        else:
            m = valid & sub_masks[name]
        res, objs = finish(a, pred, refm, m, polys, with_boundary=not name.startswith("strip_"))
        res.update({"fold": fold, "seed": int(run.name.split("seed")[1]), "subset": name, "threshold": thr,
                    "pixels_evaluated": int(m.sum())})
        json.dump(res, open(out / f"metrics_{name}.json", "w"), indent=1)
        np.savez_compressed(out / f"hist_{name}.npz", **{f"{k}_{p}": a["hist"][k][p] for k in a["hist"] for p in ("pos", "neg")},
                            **{f"conf_{k}": a["conf"][k] for k in a["conf"]})
        if objs:
            with open(out / f"objects_{name}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(objs[0]))
                wr.writeheader()
                wr.writerows(objs)
        summary[name] = {"iou": res["pixel_full"]["iou"], "ap": res["pixel_full"]["ap"], "object_recall": res["object_recall"],
                         "pixels": res["pixels_evaluated"]}
    summary["seconds"] = round(time.time() - t0, 1)
    json.dump(summary, open(out / "done.json", "w"), indent=1)      # written last: marks a complete rescoring
    print(run, json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="config run folders containing fold*_seed* runs")
    a = ap.parse_args()
    for d in a.runs:
        for run in sorted(Path(d).glob("fold*_seed*")):
            if (run / "best.pt").exists() and (run / "metrics.json").exists() and not (run / "rescore" / "done.json").exists():
                rescore_run(run)


if __name__ == "__main__":
    main()
