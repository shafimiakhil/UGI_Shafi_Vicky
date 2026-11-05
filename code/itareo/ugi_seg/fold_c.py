"""Fold C, the held-out centre (CLAUDE.md 13.7, step 3): training on the tiles outside the centre plus a buffer, scored on the centre. The hooks are in itareo.ugi_seg.train.split and itareo.ugi_seg.oof (config folds.centre: {name, mask, buffer_mask}).

  * tiles: those that touch the centre or its buffer (derived/ugi_centre/<name>_buffer<m>m_grid1m.tif) are the held-out fold, FOLD_C = 5, and enter no training or internal validation. All the others train, whatever quadrant they lie in. Everything else as ugi_r1_none, seed by seed: the cached tiles (the tiles straddling a quadrant line stay out, because the cache does not hold them), internal validation in 2 km blocks of the training tiles, the loss on all pixels of a tile, those outside the ejido included (13.6).
  * inference: over the bounding box of the centre; prob_oof.tif is NaN outside the centre.
  * caveat of 13.7: the centre is enclosed by training data on all sides, hence the buffer; fewer positives widen the intervals.

  python -m itareo.ugi_seg.fold_c --prepare <name> [--buffer-m 512]    the buffer raster and itareo/configs/ugi_foldc_<name>.yaml (from ugi_r1_none.yaml)
  python -m itareo.ugi_seg.fold_c --selftest <name> --out <a new folder>    (exit 1 if it fails)
Training, from work_ms/: python -m itareo.ugi_seg.train --cfg itareo/configs/ugi_foldc_<name>.yaml --fold 5 --seed <s>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from itareo.safe_write import refuse_existing

FOLD_C = 5
TILE = 512
REG = Path("derived/ugi_centre")


def assign_fold_c(tiles, centre):
    """Tiles touching the centre or its buffer get fold FOLD_C. Returns new dicts; the rest keep their quadrant."""
    out = []
    with rasterio.open(centre["buffer_mask"]) as s:
        b = s.read(1) == 1
    ii = np.pad(np.cumsum(np.cumsum(b.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    for t in tiles:
        a, c = int(t["row0"]), int(t["col0"])
        v = ii[a + TILE, c + TILE] - ii[a, c + TILE] - ii[a + TILE, c] + ii[a, c]
        out.append({**t, "fold": FOLD_C if v > 0 else t["fold"]})
    return out


def region_c(Hh, Ww, centre):
    """(in_r, in_c, region) as oof.quadrant: the bounding box of the centre."""
    b = json.load(open(REG / "regions.json"))[centre["name"]]
    in_r, in_c = np.zeros(Hh, bool), np.zeros(Ww, bool)
    in_r[b["row0"]:b["row0"] + b["rows"]] = True
    in_c[b["col0"]:b["col0"] + b["cols"]] = True
    return in_r, in_c, (b["row0"], b["col0"], b["rows"], b["cols"])


def centre_window(centre):
    return rasterio.open(centre["mask"])


def prepare(name, buffer_m):
    import yaml
    mask = REG / f"{name}_grid1m.tif"
    buf = REG / f"{name}_buffer{buffer_m}m_grid1m.tif"
    cfg_path = Path(f"itareo/configs/ugi_foldc_{name}.yaml")
    refuse_existing(buf, cfg_path)
    with rasterio.open(mask) as s:
        m = s.read(1) == 1
        prof = s.profile
        colormap = s.colormap(1)
    rr, cc = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    a0, a1 = max(rr[0] - buffer_m - 4, 0), min(rr[-1] + buffer_m + 5, m.shape[0])
    c0, c1 = max(cc[0] - buffer_m - 4, 0), min(cc[-1] + buffer_m + 5, m.shape[1])
    d = ndimage.distance_transform_edt(~m[a0:a1, c0:c1])
    out = np.zeros(m.shape, np.uint8)
    out[a0:a1, c0:c1] = d <= buffer_m
    with rasterio.open(buf, "w", **prof) as dst:
        dst.write(out, 1)
        dst.write_colormap(1, colormap)
        dst.set_band_description(1, f"centre candidate {name} plus {buffer_m} m: 1")
    txt = Path("itareo/configs/ugi_r1_none.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(txt)
    assert txt.count("name: ugi_r1_none") == 1 and txt.count("folds: {x_line: 388282.5, y_line: 6525025.5}") == 1
    txt = txt.replace("name: ugi_r1_none", f"name: ugi_foldc_{name}").replace("folds: {x_line: 388282.5, y_line: 6525025.5}", f"folds: {{x_line: 388282.5, y_line: 6525025.5, centre: {{name: {name}, mask: {mask.as_posix()}, buffer_mask: {buf.as_posix()}, buffer_m: {buffer_m}}}}}")
    cfg_path.write_text(f"# Fold C (CLAUDE.md 13.7): ugi_r1_none with the centre {name} and a {buffer_m} m buffer held out (itareo/ugi_seg/fold_c.py). Nothing else differs. Train with --fold 5.\n" + txt, encoding="utf-8")
    print("prepared", buf, cfg_path, flush=True)


def selftest(name, out):
    import csv

    import yaml

    from itareo.unet_ugi.data import read_csv

    from .oof import quadrant
    from .train import split
    out = Path(out)
    refuse_existing(out / "foldc_selftest.json")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = f"itareo/configs/ugi_foldc_{name}.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    res = {"config": cfg_path, "centre": cfg["folds"]["centre"]}
    elig, tr, val, rects, used = split(cfg, FOLD_C)
    held = [t for t in elig if t["fold"] == FOLD_C]
    res.update({"eligible_tiles": len(elig), "held_out_tiles": len(held), "train_tiles": len(tr), "val_tiles": len(val), "val_blocks": len(rects), "val_draw_seed": used})
    res["held_train_val_partition_eligible"] = len(held) + len(tr) + len(val) == len(elig)
    with rasterio.open(cfg["folds"]["centre"]["mask"]) as s:
        mask = s.read(1) == 1
    with rasterio.open(cfg["folds"]["centre"]["buffer_mask"]) as s:
        buf = s.read(1) == 1
    res["buffer_contains_centre"] = bool(not (mask & ~buf).any())
    res["buffer_km2"] = round(buf.sum() / 1e6, 3)
    res["centre_km2"] = round(mask.sum() / 1e6, 3)
    touch = lambda t, m: bool(m[t["row0"]:t["row0"] + TILE, t["col0"]:t["col0"] + TILE].any())
    res["no_training_or_validation_tile_touches_the_buffer"] = not any(touch(t, buf) for t in tr + val)
    # the distance from the training tiles to the centre, measured
    d = ndimage.distance_transform_edt(~mask[::8, ::8]) * 8
    res["min_distance_training_tile_to_centre_m_approx"] = round(float(min(d[t["row0"] // 8:(t["row0"] + TILE) // 8, t["col0"] // 8:(t["col0"] + TILE) // 8].min() for t in tr + val)), 0)
    in_r, in_c, region = quadrant(cfg["paths"]["image"], cfg["folds"], FOLD_C)
    rr, cc = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    res["region_rows_cols"] = list(region)
    res["region_is_bounding_box_of_centre"] = region == (int(rr[0]), int(cc[0]), int(rr[-1] - rr[0] + 1), int(cc[-1] - cc[0] + 1)) and bool(in_r[rr[0]]) and not (rr[0] > 0 and in_r[rr[0] - 1])
    labels = Path(cfg["paths"]["labels"])
    with rasterio.open(labels / "ugi_binary_grid1m.tif") as s:
        ref = s.read(1) == 1
    res["ugi_pixels_centre"] = int((ref & mask).sum())
    tiles = read_csv(cfg["paths"]["tiles"])
    res["ugi_pixels_train_tiles"] = int(sum(t["ugi_px"] for t in tr))
    groups = {}
    for k in ("px_road_greens", "px_squares", "px_parks_green_areas", "px_water_corridors", "px_rail_verges", "px_no_definido"):
        groups[k] = {"train": int(sum(t[k] for t in tr)), "val": int(sum(t[k] for t in val))}
    res["group_pixels"] = groups
    res["every_group_in_train_and_val"] = all(v["train"] > 0 and v["val"] > 0 for k, v in groups.items() if k != "px_no_definido")
    res["passed"] = bool(res["held_train_val_partition_eligible"] and res["buffer_contains_centre"] and res["no_training_or_validation_tile_touches_the_buffer"] and res["region_is_bounding_box_of_centre"] and res["every_group_in_train_and_val"])
    json.dump(res, open(out / "foldc_selftest.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare")
    ap.add_argument("--buffer-m", type=int, default=512)
    ap.add_argument("--selftest")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.prepare, a.buffer_m)
    if a.selftest:
        raise SystemExit(0 if selftest(a.selftest, a.out) else 1)
