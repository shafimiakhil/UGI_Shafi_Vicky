"""The hooks added to train.py and oof.py for CLAUDE.md 13.7 (E1b, E3, fold C) change nothing for a config without the new keys. They write only into <out>; E1 or E2 may be training at the same time.

  baseline, e1   one epoch with the learning-rate schedule of the 60-epoch run, stopped before inference: the row must equal the first row of the real run (scripts/e2_regression.epoch_only).
  e2             the same on itareo/configs/ugi_e2_tex.yaml through inference and scoring: the row and the scores must equal runs_itareo/ugi/e2_regression, which ran on the files before the hooks.
From work_ms/:  python scripts/hooks_regression.py --out <a new folder>       (exit 1 if anything differs)
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
from e2_regression import epoch_only  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402


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
        res[f"{name}_equal"] = all(float(got[k]) == float(want[k]) for k in KEYS)
        print(name, res[f"{name}_equal"], flush=True)
        json.dump(res, open(out / "hooks_regression.json", "w"), indent=1)
    e2, r = first_epoch("itareo/configs/ugi_e2_tex.yaml", out / "e2")
    old = json.load(open("runs_itareo/ugi/e2_regression/e2_regression.json"))
    res["e2_first_epoch_equal"] = all(float(e2[k]) == float(old["e2_first_epoch"][k]) for k in KEYS)
    res["e2_scored"] = {"ap": r["pixel_full"]["ap"], "iou": r["pixel_full"]["iou"]}
    res["e2_scored_equal"] = bool(res["e2_scored"]["ap"] == old["e2_scored"]["ap"] and res["e2_scored"]["iou"] == old["e2_scored"]["iou"])
    res["passed"] = bool(res["baseline_equal"] and res["e1_equal"] and res["e2_first_epoch_equal"] and res["e2_scored_equal"])
    json.dump(res, open(out / "hooks_regression.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
