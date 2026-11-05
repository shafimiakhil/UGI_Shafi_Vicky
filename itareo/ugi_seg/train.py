"""Train and evaluate one held-out fold with one seed (CLAUDE.md Section 0 and 13; family A of the design note).

  * tiles: derived/ugi_tiles/tiles.csv, non-straddling, Neo valid share >= tiles.min_neo (folds by Luigi's quadrants);
  * internal validation: 2,048 m blocks of the training folds (itareo.unet_ugi.data.internal_blocks, fixed per fold,
    every type group present), 128 m buffer ignored in the loss; normalisation and class weights from the internal
    training tiles only;
  * loss: binary BCE + soft Dice of the UGI class (background = certain negative), plus cross-entropy of the groups with
    median-frequency weights (itareo.unet_ugi.metrics.seg_loss, deterministic) on UGI pixels with a defined group;
    invalid pixels and the buffer are ignored by both heads, and the 1 m edge band as set by train.edge_band
    (both | outer | inner | none; default both, the A1/A2 round; CLAUDE.md 13.3 F3, 13.4);
  * checkpoint: best internal-validation average precision of the binary head; threshold: the F1-optimal one of 201
    values in [0, 1] on internal validation at that checkpoint. Validation scores every valid pixel (no band ignored),
    so that selection matches the full-pixel evaluation (the A1/A2 round validated without the band);
  * held-out (CLAUDE.md 13.3 rules 1 and 4): itareo.ugi_seg.oof writes prob_oof.tif and oof_groups.csv over the fold
    quadrant, and itareo.ugi_seg.score scores them (pixel, rule 4 on vegetated and other pixels beside NDVI alone,
    boundary, strict and tolerant objects, object precision, 1 km blocks). metrics.json keeps the keys of the A1/A2
    round (pixel_full, pixel_no_edge, boundary_f, object_recall = strict, ...) and adds the new ones; boot.npz holds the
    per-block counts and coarse histograms for the block bootstrap. paths.ndvi and paths.sid name the index and strip
    rasters of the mosaic in paths.image.
  * model.planetscope in the config (E1, CLAUDE.md 13.6; itareo.ugi_seg.ps_branch) adds the PlanetScope branch and its dataset; model.texture (E2; itareo.ugi_seg.tex_branch) adds
    the texture channels and their dataset; model.plugin (E3 on; CLAUDE.md 13.7) names a module with build(cfg), stats(cfg, fold, tr_i), dataset(cfg, fold, stats, *args, **kw) and window(cfg, fold, ck, region);
    folds.centre (fold C; itareo.ugi_seg.fold_c) holds out the centre and its buffer; a config without them runs as before.

From work_ms/:  python -m itareo.ugi_seg.train --cfg itareo/configs/ugi_a1.yaml --fold 1 --seed 0 [--epochs N] [--out DIR]
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np          # noqa: E402
import rasterio             # noqa: E402
import torch                # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml                 # noqa: E402

from itareo.unet_ugi.data import SeededSampler, buffer_mask, in_rects, internal_blocks, median_frequency_weights, read_csv  # noqa: E402
from itareo.unet_ugi.metrics import NBINS, auroc_ap_from_hist, prob_bins, seg_loss  # noqa: E402
from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.unet_ugi.train import seed_all  # noqa: E402

from .data import IGNORE, N_GROUPS, UGIDataset, label_counts  # noqa: E402
from .model import build  # noqa: E402
from .oof import infer_oof  # noqa: E402
from .score import score_map  # noqa: E402

DEFAULT_PATHS = {"ndvi": "derived/features/neo2022_indices_grid1m.tif",
                 "sid": "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"}

GROUPS = ["road_greens", "squares", "parks_green_areas", "water_corridors", "rail_verges"]


def split(cfg, fold):
    tiles = read_csv(cfg["paths"]["tiles"])
    if cfg["folds"].get("centre"):
        from .fold_c import assign_fold_c
        tiles = assign_fold_c(tiles, cfg["folds"]["centre"])
    elig = [t for t in tiles if not t["straddle"] and t["avail_neo2022"] >= cfg["tiles"]["min_neo"]]
    for t in elig:                                  # internal_blocks needs c1..cK: every group in the validation
        for k, g in enumerate(GROUPS, 1):
            t[f"c{k}"] = t[f"px_{g}"]
    iv = cfg["internal_val"]
    rects, used = internal_blocks(elig, fold, iv["block_m"], iv["frac"], iv["seed"] + fold, 1.0, len(GROUPS))
    trf = [t for t in elig if t["fold"] != fold]
    val = [t for t in trf if in_rects(t, rects)]
    tr = [t for t in trf if not in_rects(t, rects)]
    return elig, tr, val, rects, used


def to_device(x, device):
    """A tensor, or the list [Neo, PlanetScope] of a fusion dataset."""
    if isinstance(x, (list, tuple)):
        return [v.to(device, non_blocking=True) for v in x]
    return x.to(device, non_blocking=True)


def forward(model, x):
    return model(*x) if isinstance(x, (list, tuple)) else model(x)


def val_dataset(cache, val_i, mean, sd, nodata):
    """Internal validation scores every valid pixel, whatever the arm's edge_band (CLAUDE.md 13.3 F3)."""
    return UGIDataset(cache, val_i, mean, sd, nodata, False, edge_band="none")


def binary_dice(logit, y, eps=1.0):
    valid = (y != IGNORE).float()
    p = torch.sigmoid(logit.float()) * valid
    t = (y == 1).float() * valid
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def make_loss(gw):
    ce = seg_loss(N_GROUPS, IGNORE, gw, dice_weight=0.0)

    def fn(o, yb, yg):
        lb = o[:, 0]
        valid = yb != IGNORE
        bce = F.binary_cross_entropy_with_logits(lb.float()[valid], yb[valid].float()) if valid.any() else lb.sum() * 0
        loss = bce + binary_dice(lb, yb)
        if (yg != IGNORE).any():
            loss = loss + ce(o[:, 1:], yg)
        return loss
    return fn


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    pos = np.zeros(NBINS, np.int64)
    neg = np.zeros(NBINS, np.int64)
    lin = np.zeros((2, 201), np.int64)                    # counts with p >= t for t in linspace(0, 1, 201), pos / neg
    th = np.linspace(0, 1, 201)
    gc = np.zeros(2, np.int64)
    for x, yb, yg in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            o = forward(model, to_device(x, device)).float()
        p = torch.sigmoid(o[:, 0]).cpu().numpy()
        g = o[:, 1:].argmax(1).cpu().numpy()
        yb, yg = yb.numpy(), yg.numpy()
        m = yb != IGNORE
        t = yb[m] == 1
        pv = p[m]
        b = prob_bins(pv)
        pos += np.bincount(b[t], minlength=NBINS)
        neg += np.bincount(b[~t], minlength=NBINS)
        idx = np.clip(np.searchsorted(th, pv, side="right"), 0, 201)       # number of thresholds <= p
        cp = np.bincount(idx[t], minlength=202)
        cn = np.bincount(idx[~t], minlength=202)
        lin[0] += np.cumsum(cp[::-1])[::-1][1:202]
        lin[1] += np.cumsum(cn[::-1])[::-1][1:202]
        mg = yg != IGNORE
        gc += [int((g[mg] == yg[mg]).sum()), int(mg.sum())]
    auroc, ap = auroc_ap_from_hist(pos, neg)
    P = pos.sum()
    tp, fp = lin[0], lin[1]
    f1 = np.where(tp > 0, 2 * tp / np.maximum(tp + fp + P, 1), 0.0)
    k = int(np.argmax(f1))
    return {"ap": ap, "auroc": auroc, "f1_best": float(f1[k]), "threshold": float(th[k]),
            "group_acc": gc[0] / max(gc[1], 1)}


def run(cfg, fold, seed, out, epochs=None):
    out = Path(out)
    refuse_existing(*(out / n for n in ("best.pt", "epochs.csv", "metrics.json", "prob_oof.tif", "boot.npz")))
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["train"]
    E = epochs or tc["epochs"]
    edge_band = tc.get("edge_band", "both")
    elig, tr, val, rects, used = split(cfg, fold)
    cache = Path(cfg["paths"]["cache"])
    cinfo = json.load(open(cache / "cache.json"))
    if Path(cinfo["image"]).resolve() != Path(cfg["paths"]["image"]).resolve():
        raise ValueError(f"cache {cache} was built from {cinfo['image']}, not {cfg['paths']['image']}")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(read_csv(cache / "tiles.csv"))}
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    val_i = [index[(t["row0"], t["col0"])] for t in val]
    from itareo.unet_ugi.data import band_stats
    mean, sd = band_stats(cache, tr_i, cinfo["nodata"])
    buf = {}
    for t, i in zip(tr, tr_i):
        m = buffer_mask(t, rects, cfg["internal_val"]["buffer_m"], 512, 1.0)
        if m.any():
            buf[i] = m
    pos, neg, gcnt, gpres = label_counts(cache, tr_i, cinfo["nodata"], buf, edge_band)
    gw = median_frequency_weights(gcnt, gpres)
    b3 = None
    if cfg["model"]["init"] == "b3":
        b3 = Path(cfg["model"]["b3_runs"]) / f"fold{fold}_seed{seed}" / "best.pt"
    ps_cfg = cfg["model"].get("planetscope")
    tex_cfg = cfg["model"].get("texture")
    plug = None
    if cfg["model"].get("plugin"):
        import importlib
        plug = importlib.import_module(cfg["model"]["plugin"]["module"])
    if ps_cfg:
        from .ps_branch import UGIPSDataset, build_e1, ps_stats
        model = build_e1(cfg).to(device)
        ps_mean, ps_sd = ps_stats(cfg["paths"]["ps_cache"], tr_i)
    elif tex_cfg:
        from .tex_branch import UGITexDataset, build_e2, tex_stats
        model = build_e2(cfg).to(device)
        tex_mean, tex_sd = tex_stats(cfg["paths"]["tex_cache"], fold, tr_i)
    elif plug:
        model = plug.build(cfg).to(device)
        plug_stats = plug.stats(cfg, fold, tr_i)
    else:
        model = build(cfg["model"]["init"], b3).to(device)
    info = {"fold": fold, "seed": seed, "tiles_eligible": len(elig), "train_tiles": len(tr_i), "val_tiles": len(val_i),
            "val_blocks": rects, "val_draw_seed": used, "buffer_tiles": len(buf), "band_mean": mean.tolist(),
            "band_sd": sd.tolist(), "binary_pos_px": pos, "binary_neg_px": neg, "group_px": gcnt.tolist(),
            "group_weights": gw.tolist(), "init": cfg["model"]["init"], "b3_checkpoint": str(b3) if b3 else None,
            "edge_band": edge_band, "validation_pixels": "all valid pixels (no band ignored)", "epochs": E}
    if ps_cfg:
        info.update({"planetscope": ps_cfg, "ps_cache": cfg["paths"]["ps_cache"], "ps_mean": ps_mean.tolist(), "ps_sd": ps_sd.tolist()})
    if tex_cfg:
        info.update({"texture": tex_cfg, "tex_cache": cfg["paths"]["tex_cache"], "tex_mean": tex_mean.tolist(), "tex_sd": tex_sd.tolist()})
    if plug:
        info.update({"plugin": cfg["model"]["plugin"], **plug_stats})
    json.dump(info, open(out / "split.json", "w"), indent=1)
    g = torch.Generator().manual_seed(seed)
    nw = tc.get("workers", 4)
    if ps_cfg:
        ds_tr = UGIPSDataset(cfg["paths"]["ps_cache"], ps_mean, ps_sd, cache, tr_i, mean, sd, cinfo["nodata"], True, buf, tc.get("photometric"), edge_band)
        ds_val = UGIPSDataset(cfg["paths"]["ps_cache"], ps_mean, ps_sd, cache, val_i, mean, sd, cinfo["nodata"], False, edge_band="none")
    elif tex_cfg:
        ds_tr = UGITexDataset(cfg["paths"]["tex_cache"], fold, tex_mean, tex_sd, cache, tr_i, mean, sd, cinfo["nodata"], True, buf, tc.get("photometric"), edge_band)
        ds_val = UGITexDataset(cfg["paths"]["tex_cache"], fold, tex_mean, tex_sd, cache, val_i, mean, sd, cinfo["nodata"], False, edge_band="none")
    elif plug:
        ds_tr = plug.dataset(cfg, fold, plug_stats, cache, tr_i, mean, sd, cinfo["nodata"], True, buf, tc.get("photometric"), edge_band)
        ds_val = plug.dataset(cfg, fold, plug_stats, cache, val_i, mean, sd, cinfo["nodata"], False, edge_band="none")
    else:
        ds_tr = UGIDataset(cache, tr_i, mean, sd, cinfo["nodata"], True, buf, tc.get("photometric"), edge_band)
        ds_val = val_dataset(cache, val_i, mean, sd, cinfo["nodata"])
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=tc["batch"], sampler=SeededSampler(len(tr_i), g),
                                        num_workers=nw, drop_last=True, pin_memory=True, persistent_workers=nw > 0)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=tc["batch"], shuffle=False, num_workers=nw,
                                         pin_memory=True, persistent_workers=nw > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E * len(dl_tr))
    loss_fn = make_loss(gw)
    log, best = [], (-1.0, -1, 0.5)
    for ep in range(1, E + 1):
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tot, nb = 0.0, 0
        for x, yb, yg in dl_tr:
            x, yb, yg = to_device(x, device), yb.to(device, non_blocking=True), yg.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                o = forward(model, x)
            loss = loss_fn(o, yb, yg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        t_train = time.time() - t0
        v = validate(model, dl_val, device)
        if v["ap"] is not None and v["ap"] > best[0]:
            best = (v["ap"], ep, v["threshold"])
            torch.save({"model": model.state_dict(), "epoch": ep, "mean": mean.tolist(), "sd": sd.tolist(),
                        "threshold": v["threshold"], "config": cfg, **({"ps_mean": ps_mean.tolist(), "ps_sd": ps_sd.tolist()} if ps_cfg else {}),
                        **({"tex_mean": tex_mean.tolist(), "tex_sd": tex_sd.tolist()} if tex_cfg else {}),
                        **(plug_stats if plug else {})}, out / "best.pt")
        row = {"epoch": ep, "train_loss": round(tot / nb, 5), "val_ap": round(v["ap"], 5), "val_auroc": round(v["auroc"], 5),
               "val_f1_best": round(v["f1_best"], 5), "val_threshold": v["threshold"],
               "val_group_acc": round(v["group_acc"], 5), "seconds_train": round(t_train, 1),
               "seconds_epoch": round(time.time() - t0, 1),
               "gpu_max_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
               "best_epoch_so_far": best[1]}
        log.append(row)
        print(json.dumps(row), flush=True)
        with open(out / "epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    t_eval = time.time()
    prob, region, _ = infer_oof(model, ck, cfg, fold, out, device)
    gs = {int(r["id"]): np.array([float(r[f"sum_p_group{g}"]) for g in range(1, 6)])
          for r in csv.DictReader(open(out / "oof_groups.csv"))}
    sp = {"labels": cfg["paths"]["labels"], "polygons": cfg["paths"]["polygons"],
          "ndvi": cfg["paths"].get("ndvi", DEFAULT_PATHS["ndvi"]), "sid": cfg["paths"].get("sid", DEFAULT_PATHS["sid"])}
    res, objs, unmatched, blocks, boot = score_map(prob, prob >= ck["threshold"], region[0], region[1], sp, gs)
    del prob
    res.update({"boundary_f": res["boundary"]["f"], "object_recall": res["object_recall_strict"],
                "object_recall_by_group": res["object_recall_strict_by_group"],
                "object_recall_by_size": res["object_recall_strict_by_size"],
                "fold": fold, "seed": seed, "edge_band": edge_band, "epochs": E, "best_epoch": best[1],
                "best_val_ap": best[0], "threshold": ck["threshold"], "region_rows_cols": region,
                "scored_from": "prob_oof.tif as stored (float16 precision, read back)", "seconds_eval": round(time.time() - t_eval, 1)})
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    for o in objs:
        o["detected"] = o["found_strict"]
    for name, rows in (("objects.csv", objs), ("components_unmatched.csv", unmatched), ("blocks_1km.csv", blocks)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    np.savez_compressed(out / "boot.npz", **boot)
    t_end = time.time()
    with open(out / "run_log.md", "w", encoding="utf-8") as f:
        f.write(f"# {cfg['name']} fold {fold} seed {seed}\n\n")
        f.write(f"- command: `{' '.join(sys.argv)}` (cwd `{os.getcwd()}`)\n")
        f.write(f"- environment: python {platform.python_version()}, torch {torch.__version__}, "
                f"device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}\n")
        f.write(f"- start {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t_start))}, end "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t_end))}, {round(t_end - t_start)} s\n")
        f.write(f"- epochs {E}; best epoch {best[1]} (internal-validation AP {best[0]:.5f}); threshold {ck['threshold']}\n")
        f.write(f"- config: `{cfg['name']}`; image `{cfg['paths']['image']}`; edge_band {edge_band}; validation on all "
                f"valid pixels\n")
        f.write("- files: `split.json`, `epochs.csv`, `best.pt`, `prob_oof.tif`, `prob_oof.json`, `oof_groups.csv`, "
                "`metrics.json`, `objects.csv`, `components_unmatched.csv`, `blocks_1km.csv`, `boot.npz`\n")
        r4 = res["rule4"]["all"]
        f.write(f"- held-out fold {fold}: IoU {res['pixel_full']['iou']} (no edge {res['pixel_no_edge']['iou']}), F1 "
                f"{res['pixel_full']['f1']}, AUROC {res['pixel_full']['auroc']}, AP {res['pixel_full']['ap']}; vegetated "
                f"AUROC/AP {r4.get('veg', {}).get('auroc')}/{r4.get('veg', {}).get('ap')} (NDVI alone "
                f"{r4.get('veg', {}).get('ndvi_auroc')}/{r4.get('veg', {}).get('ndvi_ap')}); object recall strict "
                f"{res['object_recall_strict']}, tolerant {res['object_recall_tolerant']}; object precision "
                f"{res['components']['object_precision']}; boundary F {res['boundary_f']}; type macro F1 "
                f"{res.get('type_macro_f1')}\n")
    print(json.dumps({k: res[k] for k in ("fold", "seed", "edge_band", "best_epoch", "best_val_ap", "threshold",
                                          "object_recall_strict", "object_recall_tolerant", "boundary_f", "seconds_eval")}),
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8"))
    out = a.out or str(Path(cfg["paths"]["runs"]) / cfg["name"] / f"fold{a.fold}_seed{a.seed}")
    run(cfg, a.fold, a.seed, out, a.epochs)


if __name__ == "__main__":
    main()
