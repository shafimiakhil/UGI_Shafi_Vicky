"""Map inference with a trained checkpoint: sliding windows with 50 % overlap and raised-cosine blending (infer.py).

  python -m <package>.predict --ckpt best.pt --image image.tif --out map.tif [--region ROW0 COL0 ROWS COLS]
                              [--window 512] [--band-rows 2048] [--cpu]

Output: band 1 = argmax class 1..K, bands 2..K+1 = class probabilities x 250 (0..250), uint8, nodata 255 in every band
where the input is invalid,
on the image grid (or the given pixel region of it). Normalisation statistics and class names come from the checkpoint.
"""
import argparse
import json

import rasterio
import torch

from .infer import predict_to_raster
from .model import UNetResNet18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", nargs=4, type=int, metavar=("ROW0", "COL0", "ROWS", "COLS"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--band-rows", type=int, default=2048)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    K = len(ck["classes"])
    model = UNetResNet18(K, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    if a.region:
        region = tuple(a.region)
    else:
        with rasterio.open(a.image) as s:
            region = (0, 0, s.height, s.width)
    predict_to_raster(model, a.image, a.out, region, ck["mean"], ck["sd"], K, device=device, n=a.window,
                      band_rows=a.band_rows)
    print(json.dumps({"out": a.out, "region": region, "classes": ck["classes"], "epoch": ck["epoch"]}))


if __name__ == "__main__":
    main()
