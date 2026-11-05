"""Self-tests of ugi_seg (synthetic data; the B3 checkpoint is used only if it exists).

  python -m itareo.ugi_seg.selftest [--out DIR]
  1. model: 6-band conv1 (RE, DB = mean of channels 1-4), output 6 channels; A2 encoder equal to the B3 checkpoint.
  2. loss: BCE part equals torch's on the valid pixels; group part ignores background and NO DEFINIDO.
  3. targets: edge band, invalid image and buffer ignored in both heads; group codes 1-5 -> 0-4, 6 ignored.
  4. boundary F-score: identical masks give 1; a mask shifted by 3 px gives a lower score than one shifted by 1 px.
  5. validation threshold: the F1-optimal threshold of the 201-value grid equals a brute-force search.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE, targets
from .evaluate import boundary_f
from .model import selftest as model_selftest
from .train import make_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    ap.add_argument("--b3", default="runs_itareo/d1_l1a/d1_l1a_main/fold1_seed0/best.pt")
    a = ap.parse_args()
    res = {}
    m = model_selftest(a.b3 if Path(a.b3).exists() else None)
    m["passed"] = m["imagenet_re_db_equal_mean_1_4"] and m["output_shape"] == [2, 6, 128, 128] and \
        all(v for k, v in m.items() if k.startswith("b3_"))
    res["model"] = m
    # targets
    ugi = np.array([[1, 1, 0, 0]] * 4, np.uint8)
    grp = np.array([[1, 6, 0, 0]] * 4, np.uint8)
    edge = np.zeros((4, 4), np.uint8)
    edge[0] = 1
    ok = np.ones((4, 4), bool)
    ok[1, 3] = False
    buf = np.zeros((4, 4), bool)
    buf[3, 2] = True
    yb, yg = targets(ugi, grp, edge, ok, buf)
    t_ok = (yb[0] == IGNORE).all() and yb[1, 3] == IGNORE and yb[3, 2] == IGNORE and yb[2, 0] == 1 and yb[2, 3] == 0 \
        and yg[2, 0] == 0 and yg[2, 1] == IGNORE and yg[2, 2] == IGNORE
    res["targets"] = {"passed": bool(t_ok)}
    # loss
    torch.manual_seed(0)
    o = torch.randn(2, 6, 8, 8)
    yb_t = torch.randint(0, 2, (2, 8, 8))
    yb_t[:, 0] = IGNORE
    yg_t = torch.full((2, 8, 8), IGNORE, dtype=torch.long)
    lf = make_loss(np.ones(5))
    l_bin_only = lf(o, yb_t, yg_t)
    v = yb_t != IGNORE
    bce = F.binary_cross_entropy_with_logits(o[:, 0][v], yb_t[v].float())
    p = torch.sigmoid(o[:, 0]) * v
    t = (yb_t == 1).float() * v
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    res["loss"] = {"abs_diff": float(abs(l_bin_only - bce - dice)), "passed": float(abs(l_bin_only - bce - dice)) < 1e-5}
    # boundary F
    ref = np.zeros((100, 100), bool)
    ref[30:70, 30:70] = True
    valid = np.ones_like(ref)
    f0 = boundary_f(ref, ref, valid)["boundary_f"]
    f1 = boundary_f(np.roll(ref, 1, 1), ref, valid)["boundary_f"]
    f3 = boundary_f(np.roll(ref, 3, 1), ref, valid)["boundary_f"]
    res["boundary_f"] = {"same": f0, "shift_1px": f1, "shift_3px": f3, "passed": f0 == 1.0 and f1 == 1.0 and f3 < f1}
    # threshold search, brute force against the validate() counting
    rng = np.random.default_rng(1)
    pv = rng.random(5000)
    tt = rng.random(5000) < pv
    th = np.linspace(0, 1, 201)
    idx = np.clip(np.searchsorted(th, pv, side="right"), 0, 201)
    cp, cn = np.bincount(idx[tt], minlength=202), np.bincount(idx[~tt], minlength=202)
    tp, fp = np.cumsum(cp[::-1])[::-1][1:202], np.cumsum(cn[::-1])[::-1][1:202]
    f1v = np.where(tp > 0, 2 * tp / np.maximum(tp + fp + tt.sum(), 1), 0)
    brute = [2 * np.sum(tt & (pv >= x)) / max(np.sum(pv >= x) + tt.sum(), 1) for x in th]
    res["threshold_search"] = {"max_abs_diff": float(np.max(np.abs(f1v - np.array(brute)))),
                               "passed": float(np.max(np.abs(f1v - np.array(brute)))) < 1e-12}
    res = json.loads(json.dumps(res, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    res["passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / "selftest.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert res["passed"], "ugi_seg self-test failed"


if __name__ == "__main__":
    main()
