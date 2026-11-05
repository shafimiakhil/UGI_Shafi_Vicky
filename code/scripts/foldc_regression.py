"""One epoch of fold C through inference and scoring, with the learning-rate schedule of the 60-epoch run, so that the whole path runs before the queue. It writes only into <out>.
From work_ms/:  python scripts/foldc_regression.py --name <candidate> --out <a new folder>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import torch  # noqa: E402
import yaml   # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.ugi_seg import train  # noqa: E402
from itareo.ugi_seg.fold_c import FOLD_C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out)
    out.mkdir(parents=True)
    cfg = yaml.safe_load(open(f"itareo/configs/ugi_foldc_{a.name}.yaml", encoding="utf-8"))
    real = torch.optim.lr_scheduler.CosineAnnealingLR
    torch.optim.lr_scheduler.CosineAnnealingLR = lambda opt, T_max: real(opt, T_max=60 * T_max)
    try:
        r = train.run(cfg, FOLD_C, 0, out / "run", epochs=1)
    finally:
        torch.optim.lr_scheduler.CosineAnnealingLR = real
    e = next(csv.DictReader(open(out / "run/epochs.csv")))
    res = {"name": a.name, "first_epoch": e, "scored": {"ap": r["pixel_full"]["ap"], "iou": r["pixel_full"]["iou"], "seconds_eval": r["seconds_eval"], "region_rows_cols": r["region_rows_cols"]}}
    json.dump(res, open(out / "regression.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
