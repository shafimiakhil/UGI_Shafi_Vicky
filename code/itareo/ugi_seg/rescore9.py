"""Rule 9 (CLAUDE.md 13.5): scores inside the ejido municipal. Never infers.

Reads the stored probability map (prob_oof.tif of a run; the mean and the vote of the Stage M ensemble), sets prob to NaN
outside derived/ugi_labels/ejido_grid1m.tif ("not evaluated" for the scorers) and runs itareo.ugi_seg.score.score_map and
itareo.ugi_seg.score7.score7 unchanged, at the run's threshold, with the rasters of the run's mosaic. So every polygon,
component and block is what those scorers define on the pixels inside the ejido: predicted components are cut at its limit,
polygons need >= 50 % of their pixels inside, and 1 km blocks hold only their pixels inside. Type scores use the run's
oof_groups.csv as in the full-grid scoring, whose sums cover the whole held-out quadrant.

Writes, under the file names of the full-grid scoring, in <run>/ejido/ (Stage M: runs_itareo/ugi/maps/ejido/):
  metrics.json, objects.csv, components_unmatched.csv, blocks_1km.csv, boot.npz            (score_map)
  metrics_rule7.json, objects_rule7.csv, components_rule7.csv, blocks_1km_errors.csv, boot_rule7.npz   (score7)
  done.json, last. A run with done.json is skipped; a folder without it is a partial result and is refused.
itareo.ugi_seg.summarize_r --ejido reads them. With --b4 the pixels are further limited to the B4 tile set of CLAUDE.md 13.6 (non-straddling tiles with all three
PlanetScope dates >= 90 % valid, derived/ugi_tiles/tiles_b4_planetscope.csv; 1,240 tiles), in <run>/ejido_b4/ (summarize_r --sub ejido_b4).
  python -m itareo.ugi_seg.rescore9 [--stage-m] [--runs runs_itareo/ugi/ugi_r1_both ...] [--b4]        (from work_ms/)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio

from .rescore7 import DEFAULT, write_csv
from .score import read_win, score_map
from .score7 import score7

EJIDO = "derived/ugi_labels/ejido_grid1m.tif"


def b4_mask(r0, c0, H, W_, size=512):
    """The window (r0, c0, H, W_) of GRID_1M: True on the footprints of the B4 tile set."""
    from itareo.unet_ugi.data import read_csv
    m = np.zeros((H, W_), bool)
    for t in read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv"):
        if not t["straddle"] and t["avail_ps_all3"] >= 0.9:
            a, b = max(t["row0"] - r0, 0), max(t["col0"] - c0, 0)
            m[a:max(t["row0"] + size - r0, 0), b:max(t["col0"] + size - c0, 0)] = True
    return m


def score_inside(prob, pred, r0, c0, paths, folds, gs, out, info, b4=False):
    """prob (read for this call) is set to NaN outside the ejido (and, with b4, outside the B4 tiles). Refuses an existing output folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    H, W = prob.shape
    full = int(np.isfinite(prob).sum())
    inside = read_win(EJIDO, r0, c0, H, W)[0] == 1
    if b4:
        inside &= b4_mask(r0, c0, H, W)
    prob[~inside] = np.nan
    n_in = int(np.isfinite(prob).sum())
    del inside
    common = {"reference_area": "inside the ejido municipal (rule 9, CLAUDE.md 13.5)" + (", B4 tiles (CLAUDE.md 13.6)" if b4 else ""), "ejido": EJIDO,
              "pixels_evaluated_full_grid": full, "pixels_evaluated_inside_ejido": n_in, "paths": paths, **info}
    res, objs, unmatched, blocks, boot = score_map(prob, pred, r0, c0, paths, gs)
    res.update(common)
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=float)
    write_csv(out / "objects.csv", objs)
    write_csv(out / "components_unmatched.csv", unmatched)
    write_csv(out / "blocks_1km.csv", blocks)
    np.savez_compressed(out / "boot.npz", **boot)
    res7, objs7, comps7, brows7, boot7 = score7(prob, pred, r0, c0, paths, folds)
    res7.update(common)
    json.dump(res7, open(out / "metrics_rule7.json", "w"), indent=1, default=float)
    write_csv(out / "objects_rule7.csv", objs7)
    write_csv(out / "components_rule7.csv", comps7)
    write_csv(out / "blocks_1km_errors.csv", brows7)
    np.savez_compressed(out / "boot_rule7.npz", **boot7)
    json.dump({**common, "seconds": round(time.time() - t0, 1)}, open(out / "done.json", "w"), indent=1, default=float)
    print(out, "inside the ejido: AP", res["pixel_full"]["ap"], "AP without CIRCUNVALACION", res7["pixel"]["no_circunvalacion"]["ap"],
          "IoU", res["pixel_full"]["iou"], round(time.time() - t0), "s", flush=True)


def rescore_run(run, b4=False):
    run = Path(run)
    meta = json.load(open(run / "prob_oof.json"))
    import torch
    cfg = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"]
    paths = {"labels": cfg["paths"]["labels"], "polygons": cfg["paths"]["polygons"],
             "ndvi": cfg["paths"].get("ndvi", DEFAULT["ndvi"]), "sid": cfg["paths"].get("sid", DEFAULT["sid"])}
    gs = {int(r["id"]): np.array([float(r[f"sum_p_group{g}"]) for g in range(1, 6)])
          for r in csv.DictReader(open(run / "oof_groups.csv"))}
    with rasterio.open(run / "prob_oof.tif") as s:
        p = s.read(1)
    r0, c0 = meta["region_rows_cols"][:2]
    score_inside(p, p >= meta["threshold"], r0, c0, paths, cfg["folds"], gs, run / ("ejido_b4" if b4 else "ejido"),
                 {"run": str(run), "threshold": meta["threshold"], "scored_from": "prob_oof.tif as stored, NaN outside the ejido"}, b4)


def rescore_stage_m():
    from .stage_m import FOLDS, MAPS, PATHS, PRED_TEXT, PROB, VOTE, group_sums
    with rasterio.open(PROB) as s:
        p = s.read(1)
    with rasterio.open(VOTE) as s:
        pred = s.read(1) == 1
    score_inside(p, pred, 0, 0, PATHS, FOLDS, group_sums(), MAPS / "ejido",
                 {"maps": {"prob_mean": str(PROB), "vote": str(VOTE)}, "pred": PRED_TEXT,
                  "scored_from": "the mean and the vote as stored, NaN outside the ejido"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-m", action="store_true", help="the Stage M ensemble")
    ap.add_argument("--runs", nargs="*", default=[], help="config folders holding fold*_seed* runs, or run folders")
    ap.add_argument("--b4", action="store_true", help="the B4 tile set inside the ejido, in <run>/ejido_b4/")
    a = ap.parse_args()
    if a.stage_m and not Path("runs_itareo/ugi/maps/ejido/done.json").exists():
        rescore_stage_m()
    for d in a.runs:
        d = Path(d)
        runs = [d] if (d / "prob_oof.json").exists() else sorted(d.glob("fold*_seed*"))
        for run in runs:
            if (run / "metrics.json").exists() and (run / "prob_oof.tif").exists() and not (run / ("ejido_b4" if a.b4 else "ejido") / "done.json").exists():
                rescore_run(run, a.b4)


if __name__ == "__main__":
    main()
