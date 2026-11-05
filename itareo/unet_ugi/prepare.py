"""Tile index, fold report and tile cache (CLAUDE.md 12.3 d).

  python -m itareo.unet_ugi.prepare --cfg <config.yaml> --index    tile index of every tile position with labels
                                                                    (paths.tiles) and report.json next to it
  python -m itareo.unet_ugi.prepare --cfg <config.yaml> --cache    tile cache of paths.image (paths.cache) for the
                                                                    non-straddling tiles with share >= tiles.cache_min_share

report.json: tile counts at labelled-and-valid shares of 30, 50 and 80 % (all, per fold, per strip, straddling tiles
dropped); labelled pixels per class and fold at tiles.min_share; the share of labelled pixels in the 1 px boundary band;
the internal-validation blocks of each held-out fold.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .data import Paths, build_cache, internal_blocks, in_rects, read_csv, tile_index, write_csv


def report(cfg, tiles):
    K = len(cfg["classes"])
    names = cfg["classes"]
    tc = cfg["tiles"]
    out = {"tile_px": tc["size"], "stride_px": tc["stride"], "fold_lines": cfg["folds"], "thresholds": {}}
    for thr in (0.3, 0.5, 0.8):
        sel = [t for t in tiles if t["share"] >= thr]
        keep = [t for t in sel if not t["straddle"]]
        out["thresholds"][str(thr)] = {
            "tiles_at_or_above": len(sel), "straddling_dropped": len(sel) - len(keep), "tiles_kept": len(keep),
            "per_fold": {f: sum(t["fold"] == f for t in keep) for f in (1, 2, 3, 4)},
            "per_strip": {str(s): sum(t.get("strip") == s for t in keep) for s in (1, 2)},
            "labelled_valid_px": int(sum(t["labelled_valid_px"] for t in keep)),
            "band_px": int(sum(t["band_px"] for t in keep))}
        out["thresholds"][str(thr)]["band_share_of_labelled"] = round(
            out["thresholds"][str(thr)]["band_px"] / out["thresholds"][str(thr)]["labelled_valid_px"], 4)
    elig = [t for t in tiles if t["share"] >= tc["min_share"] and not t["straddle"]]
    per_fold = {}
    for f in (1, 2, 3, 4):
        ts = [t for t in elig if t["fold"] == f]
        c = np.array([sum(t[f"c{k}"] for t in ts) for k in range(1, K + 1)], np.int64)
        per_fold[f] = {"tiles": len(ts), "labelled_px": int(c.sum()),
                       "class_px": {names[k]: int(c[k]) for k in range(K)},
                       "class_share": {names[k]: round(float(c[k] / c.sum()), 4) for k in range(K)},
                       "band_share_of_labelled": round(sum(t["band_px"] for t in ts) / max(c.sum(), 1), 4)}
    out["min_share_used"] = tc["min_share"]
    out["class_pixels_per_fold"] = per_fold
    all_lab = sum(t["labelled_valid_px"] for t in tiles)
    out["band_share_all_tiles_with_labels"] = round(sum(t["band_px"] for t in tiles) / all_lab, 4)
    iv = cfg["internal_val"]
    res = abs(cfg["grid_res"])
    out["internal_validation"] = {"block_m": iv["block_m"], "frac": iv["frac"], "buffer_m": iv["buffer_m"], "folds": {}}
    for f in (1, 2, 3, 4):
        rects, used = internal_blocks(elig, f, iv["block_m"], iv["frac"], iv["seed"] + f, res, K)
        trf = [t for t in elig if t["fold"] != f]
        val = [t for t in trf if in_rects(t, rects)]
        c = np.array([sum(t[f"c{k}"] for t in val) for k in range(1, K + 1)], np.int64)
        out["internal_validation"]["folds"][f] = {
            "blocks": len(rects), "draw_seed": used, "val_tiles": len(val), "train_tiles": len(trf) - len(val),
            "val_class_px": {names[k]: int(c[k]) for k in range(K)},
            "block_rects": [{k: r[k] for k in ("x0", "y0", "x1", "y1", "tiles")} for r in rects]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8"))
    p = cfg["paths"]
    paths = Paths(p["image"], p["labels"], p.get("strips"))
    tc = cfg["tiles"]
    if a.index:
        t0 = time.time()
        tiles = tile_index(paths, tc["size"], tc["stride"], cfg["folds"]["x_line"], cfg["folds"]["y_line"],
                           len(cfg["classes"]))
        Path(p["tiles"]).parent.mkdir(parents=True, exist_ok=True)
        write_csv(tiles, p["tiles"])
        rep = report(cfg, tiles)
        rep.update({"image": p["image"], "labels": p["labels"], "seconds": round(time.time() - t0, 1),
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        json.dump(rep, open(Path(p["tiles"]).parent / "report.json", "w"), indent=1)
        print(json.dumps({k: rep[k] for k in ("thresholds", "band_share_all_tiles_with_labels")}, indent=1))
    if a.cache:
        t0 = time.time()
        tiles = [t for t in read_csv(p["tiles"]) if t["share"] >= tc["cache_min_share"] and not t["straddle"]]
        build_cache(paths, tiles, p["cache"])
        write_csv(tiles, Path(p["cache"]) / "tiles.csv")
        print(f"cache of {len(tiles)} tiles in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
