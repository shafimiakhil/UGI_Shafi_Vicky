"""Tile cache for UGI segmentation: python -m itareo.ugi_seg.prepare --cfg itareo/configs/ugi_a1.yaml (from work_ms/).
Caches every non-straddling tile of paths.tiles with Neo valid share >= tiles.min_neo, and writes its tiles.csv."""
import argparse
import time
from pathlib import Path

import yaml

from itareo.unet_ugi.data import read_csv, write_csv

from .data import build_cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.cfg, encoding="utf-8"))
    t0 = time.time()
    tiles = [t for t in read_csv(cfg["paths"]["tiles"]) if not t["straddle"] and t["avail_neo2022"] >= cfg["tiles"]["min_neo"]]
    build_cache(cfg["paths"]["image"], cfg["paths"]["labels"], tiles, cfg["paths"]["cache"])
    write_csv(tiles, Path(cfg["paths"]["cache"]) / "tiles.csv")
    print(f"cache of {len(tiles)} tiles in {time.time() - t0:.0f} s")
