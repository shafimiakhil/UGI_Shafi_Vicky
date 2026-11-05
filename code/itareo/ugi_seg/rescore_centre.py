"""Scores inside a centre candidate (CLAUDE.md 13.7). Never infers. Reads the stored probability map over the bounding box of the region (derived/ugi_centre/<name>_grid1m.tif, regions.json), sets it to NaN outside the region and runs
itareo.ugi_seg.rescore9.score_inside, so itareo.ugi_seg.score and score7 run unchanged. Polygons need >= 50 % of their pixels inside, predicted components are cut at the region's limit, 1 km blocks hold only their pixels inside.

Writes, under the file names of the full-grid scoring, in <run>/centre_<name>/ (Stage M2: runs_itareo/ugi/maps_m2/centre_<name>/), done.json last. A run with done.json is skipped; a folder without it is refused. itareo.ugi_seg.summarize_r --sub centre_<name> reads them.
  python -m itareo.ugi_seg.rescore_centre --regions ring sq3km sq5km sq8km [--stage-m2] [--runs runs_itareo/ugi/ugi_r1_none ...] [--seeds 0 1 2] [--workers 6]        (from work_ms/)
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .rescore7 import DEFAULT
from .rescore9 import score_inside
from .score import read_win

REG = Path("derived/ugi_centre")


def box(name):
    return json.load(open(REG / "regions.json"))[name]


def crop(name, r0, c0, H, W):
    """The overlap of the run's window with the region's bounding box: (row, col, rows, cols), None if empty."""
    b = box(name)
    a0, a1 = max(r0, b["row0"]), min(r0 + H, b["row0"] + b["rows"])
    c_0, c_1 = max(c0, b["col0"]), min(c0 + W, b["col0"] + b["cols"])
    return None if a1 <= a0 or c_1 <= c_0 else (a0, c_0, a1 - a0, c_1 - c_0)


def rescore_run(run, name):
    run = Path(run)
    out = run / f"centre_{name}"
    meta = json.load(open(run / "prob_oof.json"))
    import torch
    cfg = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["config"]
    paths = {"labels": cfg["paths"]["labels"], "polygons": cfg["paths"]["polygons"], "ndvi": cfg["paths"].get("ndvi", DEFAULT["ndvi"]), "sid": cfg["paths"].get("sid", DEFAULT["sid"])}
    gs = {int(r["id"]): np.array([float(r[f"sum_p_group{g}"]) for g in range(1, 6)]) for r in csv.DictReader(open(run / "oof_groups.csv"))}
    r0, c0, H, W = meta["region_rows_cols"]
    w = crop(name, r0, c0, H, W)
    if w is None:
        raise ValueError(f"{run} does not overlap {name}")
    a, b, h, wd = w
    with rasterio.open(run / "prob_oof.tif") as s:
        p = s.read(1, window=Window(b - c0, a - r0, wd, h))
    p[read_win(REG / f"{name}_grid1m.tif", a, b, h, wd)[0] != 1] = np.nan
    score_inside(p, p >= meta["threshold"], a, b, paths, cfg["folds"], gs, out, {"reference_area": f"inside the centre candidate {name} (CLAUDE.md 13.7)", "centre_region": name, "run": str(run), "threshold": meta["threshold"], "scored_from": "prob_oof.tif as stored, NaN outside the region"})


def rescore_m2(name):
    from . import stage_m, stage_m2
    stage_m2.use_m2()
    b = box(name)
    a, c, h, wd = b["row0"], b["col0"], b["rows"], b["cols"]
    win = Window(c, a, wd, h)
    with rasterio.open(stage_m2.PROB) as s:
        p = s.read(1, window=win)
    with rasterio.open(stage_m2.VOTE) as s:
        pred = s.read(1, window=win) == 1
    p[read_win(REG / f"{name}_grid1m.tif", a, c, h, wd)[0] != 1] = np.nan
    score_inside(p, pred, a, c, stage_m2.PATHS, stage_m.FOLDS, stage_m.group_sums(), stage_m2.MAPS / f"centre_{name}", {"reference_area": f"inside the centre candidate {name} (CLAUDE.md 13.7)", "centre_region": name, "pred": "vote map (>= 2 of 3 seeds above their own threshold)", "scored_from": "the mean and the vote of Stage M2 as stored, NaN outside the region"})


def _task(t):
    kind, a, name = t
    if kind == "m2":
        rescore_m2(name)
    else:
        rescore_run(a, name)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--stage-m2", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[], help="config folders holding fold*_seed* runs, or run folders")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="only these seeds of the config folders")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    tasks = []
    for name in a.regions:
        if a.stage_m2 and not Path(f"runs_itareo/ugi/maps_m2/centre_{name}/done.json").exists():
            tasks.append(("m2", None, name))
        for d in a.runs:
            d = Path(d)
            runs = [d] if (d / "prob_oof.json").exists() else sorted(d.glob("fold*_seed*"))
            for run in runs:
                if a.seeds is not None and int(run.name.split("seed")[1]) not in a.seeds:
                    continue
                if (run / "metrics.json").exists() and (run / "prob_oof.tif").exists() and not (run / f"centre_{name}" / "done.json").exists():
                    tasks.append(("run", str(run), name))
    if a.workers <= 1:
        for t in tasks:
            _task(t)
        return
    with ProcessPoolExecutor(a.workers) as ex:
        for t in ex.map(_task, tasks, chunksize=1):
            print("done", t, flush=True)


if __name__ == "__main__":
    main()
