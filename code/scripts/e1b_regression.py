"""One E1b epoch through inference and scoring, with the learning-rate schedule of the 60-epoch run (scripts/e1_regression.first_epoch), so that the whole path runs before the night queue. It writes only into <out>.
From work_ms/:  python scripts/e1b_regression.py --out <a new folder>
"""
import argparse
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))

from e1_regression import KEYS, first_epoch  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="itareo/configs/ugi_e1b_ps_plane.yaml")
    a = ap.parse_args()
    out = Path(a.out)
    refuse_existing(out)
    out.mkdir(parents=True)
    e, r = first_epoch(a.cfg, out / "run")
    res = {"config": a.cfg, "first_epoch": {k: e[k] for k in KEYS + ("seconds_epoch", "gpu_max_alloc_mb")}, "scored": {"ap": r["pixel_full"]["ap"], "iou": r["pixel_full"]["iou"], "seconds_eval": r["seconds_eval"]}}
    json.dump(res, open(out / "regression.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
