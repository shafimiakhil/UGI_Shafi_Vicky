"""Train and evaluate one held-out fold with one seed (CLAUDE.md Section 0 and 12.3).

  * internal validation: blocks of the training folds (data.internal_blocks), fixed per fold (seed internal_val.seed +
    fold), independent of the training seed; training pixels within buffer_m of a validation block are ignored;
  * normalisation (per-band mean and SD) and median-frequency class weights from the internal training tiles only;
  * loss: weighted cross-entropy + Dice (weight train.dice_weight); boundary band ignored when train.ignore_band;
  * augmentation: flips, 90 degree rotations, photometric gain (global and per band);
  * AdamW with cosine decay over all steps; bf16 autocast; deterministic algorithms; seeds for python, numpy, torch;
  * checkpoint: the epoch with the best internal-validation mIoU on the eroded reference (boundary band excluded);
  * held-out fold: sliding-window inference (infer.py) over the fold's quadrant; confusion matrices on the full and the
    eroded reference, overall and per strip; no threshold is involved (argmax).

From the folder that holds the config's relative paths:
  python -m itareo.unet_ugi.train --cfg <config.yaml> --fold 1 --seed 0 [--epochs N] [--out DIR]
"""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np          # noqa: E402
import rasterio             # noqa: E402
import torch                # noqa: E402
import yaml                 # noqa: E402

from .data import (IGNORE, SeededSampler, TileDataset, band_stats, buffer_mask, class_counts, in_rects,  # noqa: E402
                   internal_blocks, median_frequency_weights, read_csv)
from .infer import evaluate_region  # noqa: E402
from .metrics import auroc_ap_scores, confusion, scores, seg_loss  # noqa: E402
from .model import UNetResNet18  # noqa: E402


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def split(cfg, fold):
    tiles = read_csv(cfg["paths"]["tiles"])
    tc = cfg["tiles"]
    elig = [t for t in tiles if t["share"] >= tc["min_share"] and not t["straddle"]]
    iv = cfg["internal_val"]
    res = abs(cfg["grid_res"])
    rects, used_seed = internal_blocks(elig, fold, iv["block_m"], iv["frac"], iv["seed"] + fold, res,
                                       len(cfg["classes"]))
    train_folds = [t for t in elig if t["fold"] != fold]
    val = [t for t in train_folds if in_rects(t, rects)]
    tr = [t for t in train_folds if not in_rects(t, rects)]
    test = [t for t in elig if t["fold"] == fold]
    return tr, val, test, rects, used_seed


@torch.no_grad()
def validate(model, loader, band_loader_items, n_cls, device):
    model.eval()
    full = np.zeros((n_cls, n_cls), np.int64)
    ero = np.zeros((n_cls, n_cls), np.int64)
    k = 0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            o = model(x.to(device, non_blocking=True))
        pred = o.float().argmax(1).cpu().numpy()
        yy = y.numpy()
        for b in range(len(yy)):
            ok = yy[b] != IGNORE
            band = band_loader_items[k]
            full += confusion(yy[b], pred[b], n_cls, ok)
            ero += confusion(yy[b], pred[b], n_cls, ok & ~band)
            k += 1
    return full, ero


def run(cfg, fold, seed, out, epochs=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = cfg["classes"]
    K = len(names)
    tcfg = cfg["train"]
    E = epochs or tcfg["epochs"]
    tr, val, test, rects, used_seed = split(cfg, fold)
    cache = Path(cfg["paths"]["cache"])
    cinfo = json.load(open(cache / "cache.json"))
    if Path(cinfo["image"]).resolve() != Path(cfg["paths"]["image"]).resolve() or \
            Path(cinfo["labels"]).resolve() != Path(cfg["paths"]["labels"]).resolve():
        raise ValueError(f"tile cache {cache} was built from {cinfo['image']} / {cinfo['labels']}, not from the "
                         f"configured {cfg['paths']['image']} / {cfg['paths']['labels']}")
    index = {(t["row0"], t["col0"]): i for i, t in enumerate(read_csv(cache / "tiles.csv"))}
    tr_i = [index[(t["row0"], t["col0"])] for t in tr]
    val_i = [index[(t["row0"], t["col0"])] for t in val]
    mean, sd = band_stats(cache, tr_i, cinfo["nodata"])
    T = cfg["tiles"]["size"]
    res = abs(cfg["grid_res"])
    buf = {}
    for t, i in zip(tr, tr_i):
        m = buffer_mask(t, rects, cfg["internal_val"]["buffer_m"], T, res)
        if m.any():
            buf[i] = m
    # class weights from the pixels that enter the loss (boundary band and buffer excluded when they are ignored)
    cnt, present = class_counts(cache, tr_i, K, tcfg["ignore_band"], buf)
    weights = median_frequency_weights(cnt, present)
    split_info = {"fold": fold, "seed": seed, "train_tiles": len(tr_i), "val_tiles": len(val_i), "test_tiles": len(test),
                  "val_blocks": rects, "val_draw_seed": used_seed, "train_tiles_with_buffer_pixels": len(buf),
                  "buffer_pixels": int(sum(m.sum() for m in buf.values())), "band_mean": mean.tolist(),
                  "band_sd": sd.tolist(), "class_counts_train": cnt.tolist(), "class_weights": weights.tolist()}
    json.dump(split_info, open(out / "split.json", "w"), indent=1)

    g = torch.Generator().manual_seed(seed)
    ds_tr = TileDataset(cache, tr_i, mean, sd, cinfo["nodata"], tcfg["ignore_band"], True, buf, tcfg.get("photometric"))
    ds_val = TileDataset(cache, val_i, mean, sd, cinfo["nodata"], False, False)
    nw = tcfg.get("workers", 4)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=tcfg["batch"], sampler=SeededSampler(len(tr_i), g),
                                        num_workers=nw, drop_last=True, pin_memory=True, persistent_workers=nw > 0)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=tcfg["batch"], shuffle=False, num_workers=nw,
                                         pin_memory=True, persistent_workers=nw > 0)
    band_arr = np.load(cache / "band.npy", mmap_mode="r")
    val_band = [np.asarray(band_arr[i]) > 0 for i in val_i]

    model = UNetResNet18(K, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["wd"])
    steps = E * len(dl_tr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = seg_loss(K, IGNORE, weights, tcfg["dice_weight"])
    log = []
    best = (-1.0, -1)
    for ep in range(1, E + 1):
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                o = model(x)
            loss = loss_fn(o, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        t_train = time.time() - t0
        full, ero = validate(model, dl_val, val_band, K, device)
        sf, se = scores(full, names), scores(ero, names)
        row = {"epoch": ep, "train_loss": round(tot / nb, 5), "val_miou_full": sf["miou"], "val_miou_eroded": se["miou"],
               "val_oa_full": sf["oa"], "val_oa_eroded": se["oa"], "lr_end": opt.param_groups[0]["lr"],
               "seconds_train": round(t_train, 1), "seconds_epoch": round(time.time() - t0, 1),
               "gpu_max_alloc_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
               "gpu_max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20) if device == "cuda" else None}
        if se["miou"] > best[0]:
            best = (se["miou"], ep)
            torch.save({"model": model.state_dict(), "epoch": ep, "mean": mean.tolist(), "sd": sd.tolist(),
                        "classes": names, "config": cfg}, out / "best.pt")
        row["best_epoch_so_far"] = best[1]
        log.append(row)
        print(json.dumps(row), flush=True)
        with open(out / "epochs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)

    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    t_eval = time.time()
    with rasterio.open(cfg["paths"]["image"]) as src:
        Tr, Hh, Ww = src.transform, src.height, src.width
    xl, yl = cfg["folds"]["x_line"], cfg["folds"]["y_line"]
    cols_c = Tr.c + (np.arange(Ww) + 0.5) * Tr.a
    rows_c = Tr.f + (np.arange(Hh) + 0.5) * Tr.e
    left = cols_c <= xl
    south = rows_c <= yl
    in_c = left if fold in (1, 3) else ~left
    in_r = south if fold in (1, 2) else ~south
    c_idx, r_idx = np.nonzero(in_c)[0], np.nonzero(in_r)[0]
    region = (int(r_idx.min()), int(c_idx.min()), int(r_idx.max() - r_idx.min() + 1), int(c_idx.max() - c_idx.min() + 1))

    def mask_fn(r0, c0, nr, nc):
        return in_r[r0:r0 + nr][:, None] & in_c[c0:c0 + nc][None, :]

    conf, per_strip, hist = evaluate_region(model, cfg["paths"]["image"], cfg["paths"]["labels"], region, ck["mean"],
                                            ck["sd"], K, mask_fn, cfg["paths"].get("strips"), device=device,
                                            n=cfg["eval"]["window"], band_rows=cfg["eval"]["band_rows"])
    strip_names = cfg.get("strip_names", {})
    result = {"fold": fold, "seed": seed, "best_epoch": best[1], "best_val_miou_eroded": best[0],
              "held_out": {"full": scores(conf["full"], names), "eroded": scores(conf["eroded"], names)},
              "held_out_auroc_ap": {r: auroc_ap_scores(hist[r]["pos"], hist[r]["neg"], names) for r in hist},
              "per_strip": {strip_names.get(s, str(s)): {"full": scores(v["full"], names),
                                                          "eroded": scores(v["eroded"], names)}
                            for s, v in sorted(per_strip.items())},
              "region_rows_cols": region, "seconds_eval": round(time.time() - t_eval, 1)}
    json.dump(result, open(out / "metrics.json", "w"), indent=1)
    # probability histograms of the held-out fold, for out-of-fold AUROC and AP pooled over folds
    np.savez_compressed(out / "held_out_prob_hist.npz", **{f"{r}_{k}": hist[r][k] for r in hist for k in ("pos", "neg")})
    t_end = time.time()
    with open(out / "run_log.md", "w", encoding="utf-8") as f:
        f.write(f"# {cfg['name']} fold {fold} seed {seed}\n\n")
        f.write(f"- command: `{' '.join(sys.argv)}` (cwd `{os.getcwd()}`)\n")
        f.write(f"- environment: python {platform.python_version()}, torch {torch.__version__}, "
                f"device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}\n")
        f.write(f"- start {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t_start))}, end "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t_end))}, {round(t_end - t_start)} s\n")
        f.write(f"- epochs {E}; best epoch {best[1]} (internal-validation mIoU, eroded reference, {best[0]})\n")
        f.write("- files: `split.json`, `epochs.csv`, `best.pt`, `metrics.json`\n")
        f.write(f"- held-out fold {fold}: mIoU full {result['held_out']['full']['miou']}, eroded "
                f"{result['held_out']['eroded']['miou']}; OA full {result['held_out']['full']['oa']}, eroded "
                f"{result['held_out']['eroded']['oa']}; macro AUROC full {result['held_out_auroc_ap']['full']['macro_auroc']}, "
                f"eroded {result['held_out_auroc_ap']['eroded']['macro_auroc']}; macro AP full "
                f"{result['held_out_auroc_ap']['full']['macro_ap']}, eroded "
                f"{result['held_out_auroc_ap']['eroded']['macro_ap']} (agreement with the label map)\n")
    print(json.dumps({k: result[k] for k in ("fold", "seed", "best_epoch", "best_val_miou_eroded", "seconds_eval")}),
          flush=True)
    return result


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
