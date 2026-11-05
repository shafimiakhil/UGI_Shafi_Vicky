"""D1 of CLAUDE.md 13.8, the gate: how far the zero-initialised part of each fusion model has moved by the checkpoint the run kept (best.pt; the run stores no later one), and how large what it adds is against what it adds to. Reads checkpoints and caches; refuses an existing output folder; CPU is enough.

  * PlanetScope (E1, E1b) and Sentinel-2 (E3): the 1 x 1 gate. Its weight and bias norms (Frobenius), and on held-out tiles of the fold, in evaluation mode, the root mean square of the stage features it adds to (layer1 output, 4 m, for PlanetScope; layer2 output, 8 m, for Sentinel-2), of the added term, and of the part of the added term that depends on the input (added term minus the added term for the no-data input). Ratios to the stage features: ratio_added, ratio_input_dependent. Median over the tiles.
  * New conv1 channels (E2, E1c): the Frobenius norm of the weights of the new channels and of the six Neo channels, per channel on average (rms_per_channel: norm / sqrt(channels)), the ratio, each new channel's norm; and on held-out tiles the root mean square of conv1 applied to the new channels alone against conv1 applied to Neo alone (before the batch norm). Median over the tiles.
Held-out tiles: 32 drawn with seed = fold from the non-straddling tiles of the fold in the Neo cache; for PlanetScope and for E1c among those with all three dates at least 90 % valid, because a tile without PlanetScope has nothing to show.

Outputs in <out>: d1_gate.csv (gates), d1_conv1.csv (new conv1 channels). From work_ms/: python scripts/diag_d1.py --out ../docs/evidence_phase2/ugi_seg/diagnostics/d1_gate --arms ugi_e1_ps ugi_e1b_ps_plane ugi_e2_tex
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from itareo.ugi_seg import ps_branch, s2_branch, tex_branch  # noqa: E402
from itareo.unet_ugi.data import read_csv  # noqa: E402

ROOT = Path("runs_itareo/ugi")
KIND = {"ugi_e1_ps": "ps", "ugi_e1b_ps_plane": "ps", "ugi_e3_s2": "s2", "ugi_e2_tex": "tex", "ugi_e1c_ps1m": "ps1m"}
N_TILES = 32


def rms(t, dims=None):
    return torch.sqrt(torch.mean(t.float() ** 2)) if dims is None else torch.sqrt(torch.mean(t.float() ** 2, dim=dims))


def tiles_for(cfg, fold, need_ps):
    cache = Path(cfg["paths"]["cache"])
    tiles = read_csv(cache / "tiles.csv")
    b4 = {(t["row0"], t["col0"]): t["avail_ps_all3"] for t in read_csv("derived/ugi_tiles/tiles_b4_planetscope.csv")}
    idx = [i for i, t in enumerate(tiles) if t["fold"] == fold and not t["straddle"] and (not need_ps or b4.get((t["row0"], t["col0"]), 0) >= 0.9)]
    rng = np.random.default_rng(fold)
    return cache, json.load(open(cache / "cache.json")), [idx[k] for k in rng.choice(len(idx), min(N_TILES, len(idx)), replace=False)]


@torch.no_grad()
def gate_row(arm, fold, seed):
    ck = torch.load(ROOT / arm / f"fold{fold}_seed{seed}" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    kind = KIND[arm]
    cache, cinfo, idx = tiles_for(cfg, fold, kind == "ps")
    if kind == "ps":
        m = ps_branch.build_e1(cfg)
        ds = ps_branch.UGIPSDataset(cfg["paths"]["ps_cache"], ck["ps_mean"], ck["ps_sd"], cache, idx, ck["mean"], ck["sd"], cinfo["nodata"], False, edge_band="none")
        gate, branch = m.ps_gate, m.ps_branch
    else:
        m = s2_branch.build(cfg)
        ds = s2_branch.dataset(cfg, fold, ck, cache, idx, ck["mean"], ck["sd"], cinfo["nodata"], False, edge_band="none")
        gate, branch = m.s2_gate, m.s2_branch
    m.load_state_dict(ck["model"])
    m.eval()
    e = m.base.enc
    stage, added, dep = [], [], []
    for j in range(len(idx)):
        [x, p], _, _ = ds[j]
        x, p = x[None], p[None]
        s = e.layer1(e.maxpool(e.relu(e.bn1(e.conv1(x)))))
        if kind == "s2":
            s = e.layer2(s)
        g = gate(branch(p))
        g0 = gate(branch(torch.zeros_like(p)))
        stage.append(float(rms(s)))
        added.append(float(rms(g)))
        dep.append(float(rms(g - g0)))
    stage, added, dep = np.array(stage), np.array(added), np.array(dep)
    return {"arm": arm, "fold": fold, "seed": seed, "best_epoch": ck["epoch"], "gate_weight_norm": round(float(gate.weight.norm()), 5), "gate_bias_norm": round(float(gate.bias.norm()), 5), "tiles": len(idx),
            "stage_rms_median": round(float(np.median(stage)), 5), "added_rms_median": round(float(np.median(added)), 5), "input_dependent_rms_median": round(float(np.median(dep)), 5),
            "ratio_added": round(float(np.median(added / stage)), 5), "ratio_input_dependent": round(float(np.median(dep / stage)), 5)}


@torch.no_grad()
def conv1_row(arm, fold, seed):
    ck = torch.load(ROOT / arm / f"fold{fold}_seed{seed}" / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    kind = KIND[arm]
    cache, cinfo, idx = tiles_for(cfg, fold, kind == "ps1m")
    if kind == "tex":
        m = tex_branch.build_e2(cfg)
        ds = tex_branch.UGITexDataset(cfg["paths"]["tex_cache"], fold, ck["tex_mean"], ck["tex_sd"], cache, idx, ck["mean"], ck["sd"], cinfo["nodata"], False, edge_band="none")
    else:
        from itareo.ugi_seg import ps1m_branch
        m = ps1m_branch.build(cfg)
        ds = ps1m_branch.dataset(cfg, fold, ck, cache, idx, ck["mean"], ck["sd"], cinfo["nodata"], False, edge_band="none")
    m.load_state_dict(ck["model"])
    m.eval()
    c = m.base.enc.conv1
    w = c.weight
    neo, new = w[:, :6], w[:, 6:]
    a_neo, a_new = [], []
    for j in range(len(idx)):
        [x, p], _, _ = ds[j]
        a_neo.append(float(rms(F.conv2d(x[None], neo, stride=c.stride, padding=c.padding))))
        a_new.append(float(rms(F.conv2d(p[None], new, stride=c.stride, padding=c.padding))))
    a_neo, a_new = np.array(a_neo), np.array(a_new)
    n_neo, n_new = float(neo.norm()), float(new.norm())
    pc_neo, pc_new = n_neo / np.sqrt(neo.shape[1]), n_new / np.sqrt(new.shape[1])
    return {"arm": arm, "fold": fold, "seed": seed, "best_epoch": ck["epoch"], "neo_weight_norm": round(n_neo, 5), "new_weight_norm": round(n_new, 5), "neo_rms_per_channel": round(pc_neo, 5), "new_rms_per_channel": round(pc_new, 5), "ratio_per_channel": round(pc_new / pc_neo, 5), "tiles": len(idx),
            "conv1_output_rms_neo_median": round(float(np.median(a_neo)), 5), "conv1_output_rms_new_median": round(float(np.median(a_new)), 5), "ratio_conv1_output": round(float(np.median(a_new / a_neo)), 5),
            "new_channel_norms": json.dumps([round(float(new[:, k].norm()), 4) for k in range(new.shape[1])])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True, choices=list(KIND))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {"gate": [], "conv1": []}
    for arm in a.arms:
        for s in a.seeds:
            for f in (1, 2, 3, 4):
                if KIND[arm] in ("ps", "s2"):
                    r = gate_row(arm, f, s)
                    rows["gate"].append(r)
                else:
                    r = conv1_row(arm, f, s)
                    rows["conv1"].append(r)
                print({k: v for k, v in r.items() if k != "new_channel_norms"}, flush=True)
    for name, rr in rows.items():
        if rr:
            with open(out / f"d1_{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0]))
                w.writeheader()
                w.writerows(rr)


if __name__ == "__main__":
    main()
