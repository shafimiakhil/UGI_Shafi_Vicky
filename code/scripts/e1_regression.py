"""Two checks of the training path before E1 (CLAUDE.md 13.6). They write only into <out>.

  baseline  itareo.ugi_seg.train.run on itareo/configs/ugi_r1_none.yaml, fold 1, seed 0, one epoch, with the learning-rate schedule of the 60-epoch run
            (T_max of CosineAnnealingLR multiplied by 60, so the first epoch's rates are those of the real run). Its epoch row must equal the first row of
            runs_itareo/ugi/ugi_r1_none/fold1_seed0/epochs.csv: the edits made for E1 left the baseline's path unchanged. It goes on through inference and scoring.
  e1        the same on itareo/configs/ugi_e1_ps.yaml: the whole E1 path (training, validation, checkpoint, inference with the PlanetScope windows, scoring) runs.
From work_ms/:  python scripts/e1_regression.py --out <a new folder>
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

KEYS = ("train_loss", "val_ap", "val_auroc", "val_f1_best", "val_threshold", "val_group_acc")


def first_epoch(cfg_path, out):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    real = torch.optim.lr_scheduler.CosineAnnealingLR
    torch.optim.lr_scheduler.CosineAnnealingLR = lambda opt, T_max: real(opt, T_max=60 * T_max)
    try:
        res = train.run(cfg, 1, 0, out, epochs=1)
    finally:
        torch.optim.lr_scheduler.CosineAnnealingLR = real
    return next(csv.DictReader(open(Path(out) / "epochs.csv"))), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out)
    out.mkdir(parents=True)
    got, _ = first_epoch("itareo/configs/ugi_r1_none.yaml", out / "baseline")
    want = next(csv.DictReader(open("runs_itareo/ugi/ugi_r1_none/fold1_seed0/epochs.csv")))
    res = {"baseline_first_epoch": {k: {"this_run": got[k], "ugi_r1_none_fold1_seed0": want[k]} for k in KEYS}}
    res["baseline_equal"] = all(float(got[k]) == float(want[k]) for k in KEYS)
    json.dump(res, open(out / "e1_regression.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    e1, r = first_epoch("itareo/configs/ugi_e1_ps.yaml", out / "e1")
    res["e1_first_epoch"] = {k: e1[k] for k in KEYS + ("seconds_epoch", "gpu_max_alloc_mb")}
    res["e1_scored"] = {"ap": r["pixel_full"]["ap"], "iou": r["pixel_full"]["iou"], "seconds_eval": r["seconds_eval"]}
    res["baseline_seconds_epoch"] = {"this_run": got["seconds_epoch"], "ugi_r1_none_fold1_seed0": want["seconds_epoch"]}
    json.dump(res, open(out / "e1_regression.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["baseline_equal"] else 1)


if __name__ == "__main__":
    main()
