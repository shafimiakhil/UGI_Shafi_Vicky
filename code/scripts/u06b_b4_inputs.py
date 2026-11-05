"""B4 inputs (CLAUDE.md 13.1, proposed): PlanetScope 24 Apr, 10 Sep and 28 Oct 2022 at 3 m, each normalised to its S2 2022
quarter median (scripts/u06_planetscope.py), on the COMMON lattice (GRID_1M nests at factor 3). No training.
Per tile of derived/ugi_tiles/tiles.csv (same 512 px tiles): the valid share of each date and of all three together.
Outputs: derived/ugi_tiles/tiles_b4_planetscope.csv, derived/ugi_tiles/b4_summary.json. Run from work_ms/."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.unet_ugi.data import read_csv   # noqa: E402

DATES = ["20220424", "20220910", "20221028"]
PS = {d: WM / f"derived/planetscope2022/ps{d}_sr_norm_s2_3m.tif" for d in DATES}

t0 = time.time()
tiles = read_csv(WM / "derived/ugi_tiles/tiles.csv")
valid = {}
for d, p in PS.items():
    with rasterio.open(p) as s:
        valid[d] = (s.read() != 0).all(0)
rows = []
for t in tiles:
    r3, c3 = t["row0"] // 3, t["col0"] // 3
    r = {"row0": t["row0"], "col0": t["col0"], "fold": t["fold"], "straddle": t["straddle"], "ugi_px": t["ugi_px"]}
    allv = None
    for d in DATES:
        v = valid[d][r3:r3 + 171, c3:c3 + 171]          # 512 px = 170.7 cells of 3 m
        r[f"avail_ps{d}"] = round(float(v.mean()), 4)
        allv = v if allv is None else allv & v
    r["avail_ps_all3"] = round(float(allv.mean()), 4)
    rows.append(r)
with open(WM / "derived/ugi_tiles/tiles_b4_planetscope.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
keep = [r for r in rows if not r["straddle"]]
summ = {"inputs": {d: str(p.relative_to(WM)) for d, p in PS.items()},
        "tiles_non_straddling": len(keep),
        "tiles_ge_0.9": {k: sum(r[k] >= 0.9 for r in keep) for k in [f"avail_ps{d}" for d in DATES] + ["avail_ps_all3"]},
        "ugi_px_in_tiles_all3_ge_0.9": int(sum(r["ugi_px"] for r in keep if r["avail_ps_all3"] >= 0.9)),
        "ugi_px_total_non_straddling": int(sum(r["ugi_px"] for r in keep)),
        "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(summ, open(WM / "derived/ugi_tiles/b4_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
