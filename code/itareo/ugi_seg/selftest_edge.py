"""Self-test of the edge_band key (CLAUDE.md 13.3, F3), on synthetic data; run before Stage R.

  python -m itareo.ugi_seg.selftest_edge --out <a new folder>   (refuses an existing JSON or synthetic cache)
  1. targets: both = the band ignored on both sides; outer = only band pixels outside the inventory ignored (inner band
     pixels are positives and keep their group); inner = only band pixels inside the inventory ignored (outer band
     pixels are negatives); none = no band; invalid and buffer pixels ignored in every mode; both
     heads follow the mode; an unknown mode raises.
  2. datasets: a cache of two synthetic tiles; the training dataset follows the arm's mode; the validation dataset of
     train.val_dataset scores every valid pixel (no band ignored) whatever the arm; label_counts follows the mode.
  3. loss: with outer, the inner band enters the BCE (more valid pixels than both).
  4. configs: every Stage R config sets train.edge_band to an allowed value.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from itareo.safe_write import refuse_existing

from .data import EDGE_BANDS, IGNORE, UGIDataset, label_counts, targets
from .train import make_loss, val_dataset


def synthetic(T=16):
    ugi = np.zeros((T, T), np.uint8)
    ugi[4:12, 4:12] = 1
    grp = np.where(ugi == 1, 2, 0).astype(np.uint8)
    edge = np.zeros((T, T), np.uint8)
    inner = (ugi == 1) & ~np.pad(np.ones((6, 6), bool), 5)          # 1 px ring inside the square
    outer = np.pad(np.ones((10, 10), bool), 3) & (ugi == 0)          # 1 px ring outside it
    edge[inner | outer] = 1
    ok = np.ones((T, T), bool)
    ok[0, 0] = False
    buf = np.zeros((T, T), bool)
    buf[15, :] = True
    return ugi, grp, edge, ok, buf, inner, outer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="a new folder for this self-test's JSON and synthetic cache")
    a = ap.parse_args()
    res, ok_all = {}, True
    ugi, grp, edge, ok, buf, inner, outer = synthetic()
    t = {}
    for mode in EDGE_BANDS:
        yb, yg = targets(ugi, grp, edge, ok, buf, mode)
        t[mode] = (yb, yg)
        r = {"invalid_ignored": bool(yb[0, 0] == IGNORE and yg[0, 0] == IGNORE),
             "buffer_ignored": bool((yb[15] == IGNORE).all() and (yg[15] == IGNORE).all()),
             "inner_band_ignored": bool((yb[inner] == IGNORE).all()), "outer_band_ignored": bool((yb[outer] == IGNORE).all()),
             "inner_band_group_ignored": bool((yg[inner] == IGNORE).all()),
             "inner_band_positive": bool((yb[inner] == 1).all()), "outer_band_negative": bool((yb[outer & ok & ~buf] == 0).all()),
             "group_on_inner_band": bool((yg[inner] == 1).all())}
        exp = {"both": dict(inner_band_ignored=True, outer_band_ignored=True, inner_band_group_ignored=True),
               "outer": dict(inner_band_ignored=False, outer_band_ignored=True, inner_band_positive=True, group_on_inner_band=True),
               "inner": dict(inner_band_ignored=True, outer_band_ignored=False, inner_band_group_ignored=True,
                             outer_band_negative=True),
               "none": dict(inner_band_ignored=False, outer_band_ignored=False, inner_band_positive=True,
                            outer_band_negative=True, group_on_inner_band=True)}[mode]
        r["passed"] = r["invalid_ignored"] and r["buffer_ignored"] and all(r[k] == v for k, v in exp.items())
        res[f"targets_{mode}"] = r
        ok_all &= r["passed"]
    try:
        targets(ugi, grp, edge, ok, None, "middle")             # not a mode
        res["unknown_mode_raises"] = False
    except ValueError:
        res["unknown_mode_raises"] = True
    ok_all &= res["unknown_mode_raises"]
    # datasets on a synthetic cache
    refuse_existing(Path(a.out) / "edge_band_selftest.json")
    cache = Path(a.out) / "edge_band_cache"                          # tiny synthetic cache (2 tiles of 16 x 16)
    refuse_existing(cache)
    cache.mkdir(parents=True, exist_ok=True)
    img = np.full((2, 6, 16, 16), 1000, np.uint16)
    img[:, :, 0, 0] = 65535
    np.save(cache / "img.npy", img)
    np.save(cache / "ugi.npy", np.stack([ugi, ugi]))
    np.save(cache / "grp.npy", np.stack([grp, grp]))
    np.save(cache / "edge.npy", np.stack([edge, edge]))
    mean, sd = np.zeros(6), np.ones(6)
    ds = {}
    for mode in EDGE_BANDS:
        d = UGIDataset(cache, [0, 1], mean, sd, 65535, False, None, None, mode)
        x, yb, yg = d[0]
        ds[mode] = int((yb.numpy() == IGNORE).sum())
    v = val_dataset(cache, [0, 1], mean, sd, 65535)
    _, ybv, _ = v[0]
    res["dataset_ignored_px"] = ds
    res["validation_ignored_px"] = int((ybv.numpy() == IGNORE).sum())
    res["validation_ignores_only_invalid"] = res["validation_ignored_px"] == 1
    lc = {m: label_counts(cache, [0, 1], 65535, None, m)[:2] for m in EDGE_BANDS}
    res["label_counts_pos_neg"] = {m: [int(x) for x in v_] for m, v_ in lc.items()}
    res["datasets_passed"] = (ds["both"] > ds["outer"] > ds["none"] == 1 and ds["both"] > ds["inner"] > ds["none"]
                              and res["validation_ignores_only_invalid"]
                              and lc["outer"][0] == lc["none"][0] > lc["both"][0] and lc["both"][1] == lc["outer"][1] < lc["none"][1]
                              and lc["inner"][0] == lc["both"][0] and lc["inner"][1] == lc["none"][1])
    ok_all &= res["datasets_passed"]
    # loss: outer adds the inner band to the BCE
    loss_fn = make_loss(np.ones(5))
    o = torch.zeros(1, 6, 16, 16)
    ln = {}
    for mode in ("both", "outer", "inner"):
        yb, yg = (torch.from_numpy(z)[None] for z in t[mode])
        ln[mode] = {"valid_binary_px": int((yb != IGNORE).sum()), "loss": round(float(loss_fn(o, yb, yg)), 6)}
    res["loss"] = ln
    res["loss_passed"] = (ln["outer"]["valid_binary_px"] - ln["both"]["valid_binary_px"] == int(inner.sum())
                          and ln["inner"]["valid_binary_px"] - ln["both"]["valid_binary_px"] == int((outer & ~buf).sum()))
    ok_all &= res["loss_passed"]
    cfgs = sorted(Path("itareo/configs").glob("ugi_r*.yaml"))
    res["stage_r_configs"] = {c.name: yaml.safe_load(open(c, encoding="utf-8"))["train"].get("edge_band") for c in cfgs}
    res["configs_passed"] = bool(cfgs) and all(v in EDGE_BANDS for v in res["stage_r_configs"].values())
    ok_all &= res["configs_passed"]
    res["passed"] = bool(ok_all)
    out = Path(a.out) / "edge_band_selftest.json"
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
