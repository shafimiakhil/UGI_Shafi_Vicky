"""Checks of the training path before E2 (CLAUDE.md 13.6). They write only into <out>. E1 may be training at the same time: these use the same code and write elsewhere.

  baseline  itareo.ugi_seg.train.run on itareo/configs/ugi_r1_none.yaml, fold 1, seed 0, one epoch with the learning-rate schedule of the 60-epoch run (scripts/e1_regression.first_epoch), stopped before inference.
            Its epoch row must equal the first row of runs_itareo/ugi/ugi_r1_none/fold1_seed0/epochs.csv: the edits made for E2 left the baseline's path unchanged.
  e1        the same on itareo/configs/ugi_e1_ps.yaml against runs_itareo/ugi/ugi_e1_ps/fold1_seed0/epochs.csv, because E1's folds started after the edits read the edited files.
  e2        the same on itareo/configs/ugi_e2_tex.yaml through inference and scoring: the whole E2 path runs.
From work_ms/:  python scripts/e2_regression.py --out <a new folder>       (exit 1 if a row differs)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))

from e1_regression import KEYS, first_epoch  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.ugi_seg import train  # noqa: E402


class Stop(Exception):
    pass


def epoch_only(cfg_path, out):
    real = train.infer_oof

    def stop(*a, **k):
        raise Stop
    train.infer_oof = stop
    try:
        first_epoch(cfg_path, out)
    except Stop:
        pass
    finally:
        train.infer_oof = real
    return next(csv.DictReader(open(Path(out) / "epochs.csv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out)
    out.mkdir(parents=True)
    res = {}
    for name, cfg, ref in (("baseline", "itareo/configs/ugi_r1_none.yaml", "runs_itareo/ugi/ugi_r1_none/fold1_seed0/epochs.csv"), ("e1", "itareo/configs/ugi_e1_ps.yaml", "runs_itareo/ugi/ugi_e1_ps/fold1_seed0/epochs.csv")):
        got = epoch_only(cfg, out / name)
        want = next(csv.DictReader(open(ref)))
        res[f"{name}_first_epoch"] = {k: {"this_run": got[k], "reference": want[k]} for k in KEYS}
        res[f"{name}_equal"] = all(float(got[k]) == float(want[k]) for k in KEYS)
        json.dump(res, open(out / "e2_regression.json", "w"), indent=1)
        print(name, res[f"{name}_equal"], flush=True)
    e2, r = first_epoch("itareo/configs/ugi_e2_tex.yaml", out / "e2")
    res["e2_first_epoch"] = {k: e2[k] for k in KEYS + ("seconds_epoch", "gpu_max_alloc_mb")}
    res["e2_scored"] = {"ap": r["pixel_full"]["ap"], "iou": r["pixel_full"]["iou"], "seconds_eval": r["seconds_eval"]}
    json.dump(res, open(out / "e2_regression.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["baseline_equal"] and res["e1_equal"] else 1)


if __name__ == "__main__":
    main()
