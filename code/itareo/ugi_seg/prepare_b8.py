"""Inputs for B8 (CLAUDE.md 13.1, proposed): bi-temporal Siamese encoder over Pléiades 12 Feb 2017 and Neo Oct 2022 on the
four common bands (R, G, B, NIR), with the GEOBIA 7-class head as auxiliary supervision on the 2017 branch. Data only, no
training.

Tile cache (row i = tile i of tiles.csv; tiles non-straddling with Neo and Pléiades 2017 each >= tiles.min_valid valid):
  img2017 (N, 4, 512, 512) uint16  derived/pleiades2017/pleiades2017_toa_hpf_grid1m_harmonised.tif (R, G, B, NIR), 65535 nodata
  img2022 (N, 4, 512, 512) uint16  derived/pneo2022_registered/pneo2022_toa_grid1m.tif bands 1-4 (R, G, B, NIR), 65535 nodata
  ugi, grp, edge (N, 512, 512) uint8  derived/ugi_labels/
  geobia (N, 512, 512) uint8       derived/geobia2017/geobia2017_grid1m.tif (0 nodata, classes 1-7)
Both images are TOA reflectance x 10000 on GRID_1M, each resampled once already (A5 + B1b; A6 + U0.3).
  python -m itareo.ugi_seg.prepare_b8 --out runs_itareo/ugi/cache/b8_ple2017_neo2022 [--min-valid 0.5]   (from work_ms/)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from itareo.unet_ugi.data import read_csv, write_csv

PLE = "derived/pleiades2017/pleiades2017_toa_hpf_grid1m_harmonised.tif"
NEO = "derived/pneo2022_registered/pneo2022_toa_grid1m.tif"
GEO = "derived/geobia2017/geobia2017_grid1m.tif"
LAB = Path("derived/ugi_labels")
TILES = "derived/ugi_tiles/tiles.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-valid", type=float, default=0.5)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = [t for t in read_csv(TILES) if not t["straddle"] and t["avail_neo2022"] >= a.min_valid
             and t["avail_ple2017"] >= a.min_valid]
    N, T = len(tiles), 512
    arr = {"img2017": np.lib.format.open_memmap(out / "img2017.npy", "w+", np.uint16, (N, 4, T, T)),
           "img2022": np.lib.format.open_memmap(out / "img2022.npy", "w+", np.uint16, (N, 4, T, T))}
    for k in ("ugi", "grp", "edge", "geobia"):
        arr[k] = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.uint8, (N, T, T))
    with rasterio.open(PLE) as p, rasterio.open(NEO) as n, rasterio.open(GEO) as g, \
            rasterio.open(LAB / "ugi_binary_grid1m.tif") as ub, rasterio.open(LAB / "ugi_group_grid1m.tif") as ug, \
            rasterio.open(LAB / "ugi_edge_grid1m.tif") as ue:
        assert p.transform == n.transform == g.transform == ub.transform
        for i, t in enumerate(tiles):
            w = Window(int(t["col0"]), int(t["row0"]), T, T)
            arr["img2017"][i] = p.read(window=w)
            arr["img2022"][i] = n.read([1, 2, 3, 4], window=w)
            arr["geobia"][i] = g.read(1, window=w)
            arr["ugi"][i] = ub.read(1, window=w)
            arr["grp"][i] = ug.read(1, window=w)
            arr["edge"][i] = ue.read(1, window=w)
    for v in arr.values():
        v.flush()
    write_csv(tiles, out / "tiles.csv")
    info = {"tiles": N, "per_fold": {f: sum(t["fold"] == f for t in tiles) for f in (1, 2, 3, 4)},
            "min_valid": a.min_valid, "image_2017": PLE, "image_2022": NEO, "bands": "R, G, B, NIR (2022: Neo bands 1-4)",
            "geobia": GEO, "labels": str(LAB), "nodata": 65535, "seconds": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(info, open(out / "cache.json", "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
