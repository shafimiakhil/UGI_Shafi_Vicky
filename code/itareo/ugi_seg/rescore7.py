"""Rule 7 for finished runs (CLAUDE.md 13.4): itareo.ugi_seg.score7 on each run's prob_oof.tif at its threshold, with the
index and strip rasters of the run's mosaic (config paths.ndvi, paths.sid). Never infers. Writes beside metrics.json:
metrics_rule7.json, objects_rule7.csv, components_rule7.csv, blocks_1km_errors.csv, boot_rule7.npz. A run that already has
metrics_rule7.json is skipped (no-clobber).
  python -m itareo.ugi_seg.rescore7 --runs runs_itareo/ugi/ugi_r1_both ...        (from work_ms/)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml

from itareo.safe_write import refuse_existing

from .score7 import score7

DEFAULT = {"ndvi": "derived/features/neo2022_indices_grid1m.tif", "sid": "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif"}


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def rescore_run(run, cfg_path=None):
    run = Path(run)
    t0 = time.time()
    refuse_existing(*(run / n for n in ("metrics_rule7.json", "objects_rule7.csv", "components_rule7.csv",
                                        "blocks_1km_errors.csv", "boot_rule7.npz")))
    meta = json.load(open(run / "prob_oof.json"))
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) if cfg_path else None
    if cfg is None:
        import torch
        cfg = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"]
    paths = {"labels": cfg["paths"]["labels"], "polygons": cfg["paths"]["polygons"],
             "ndvi": cfg["paths"].get("ndvi", DEFAULT["ndvi"]), "sid": cfg["paths"].get("sid", DEFAULT["sid"])}
    with rasterio.open(run / "prob_oof.tif") as s:
        p = s.read(1)
    r0, c0 = meta["region_rows_cols"][:2]
    res, objs, comps, brows, boot = score7(p, p >= meta["threshold"], r0, c0, paths, cfg["folds"])
    res.update({"run": str(run), "threshold": meta["threshold"], "scored_from": "prob_oof.tif as stored", "paths": paths,
                "seconds": round(time.time() - t0, 1)})
    write_csv(run / "objects_rule7.csv", objs)
    write_csv(run / "components_rule7.csv", comps)
    write_csv(run / "blocks_1km_errors.csv", brows)
    np.savez_compressed(run / "boot_rule7.npz", **boot)
    json.dump(res, open(run / "metrics_rule7.json", "w"), indent=1, default=float)       # last: main() skips a run that has it
    print(run, "AP", res["pixel"]["all"]["ap"], "AP without CIRCUNVALACION", res["pixel"]["no_circunvalacion"]["ap"],
          round(time.time() - t0), "s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="config folders holding fold*_seed* runs, or run folders")
    a = ap.parse_args()
    for d in a.runs:
        d = Path(d)
        runs = [d] if (d / "prob_oof.json").exists() else sorted(d.glob("fold*_seed*"))
        for run in runs:
            if (run / "metrics.json").exists() and (run / "prob_oof.tif").exists() and not (run / "metrics_rule7.json").exists():
                rescore_run(run)


if __name__ == "__main__":
    main()
