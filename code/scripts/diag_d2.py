"""D2 of CLAUDE.md 13.8, input blanked at inference: the trained E1b and E2 checkpoints scored on the held-out fold with the added input at its no-data value (availability 0, masks 0), everything else as the run: same weights, same threshold, same sliding-window inference (itareo.ugi_seg.oof.infer_oof through the plugin itareo.ugi_seg.blank_input) and the same scoring (itareo.ugi_seg.score.score_map). Nothing is trained.

Outputs, as a run folder, in runs_itareo/ugi/diag_d2_<tag>_blank/fold<f>_seed<s>/: prob_oof.tif, prob_oof.json, oof_groups.csv, metrics.json, objects.csv, components_unmatched.csv, blocks_1km.csv, boot.npz, best.pt (the config and the source checkpoint's path only, no weights; the rescoring reads the config from it), run_log.md, last. Refuses an existing run_log.md or prob_oof.tif. Rules 7 and 9 and the centre then run on it as on any run (scripts/night_post_13_8.py).

  python scripts/diag_d2.py --arm ugi_e1b_ps_plane --fold 1 --seed 0          (from work_ms/; GPU)
  python scripts/diag_d2.py --selftest --out <new folder>

Self-test (exit 1 if it fails), on fold 1 of both arms, over a window of 1,024 px on the lattice of the stored run: the no-data tensors of ps_tensor and tex_tensor are all 0, so the blank is what the data loader gives where the input does not exist (how often training holds that state differs: E1b, no date on 4.83 to 11.93 % of the training cells of folds 1 to 4, none inside sq8km; E2, both masks 0 on 0.0017 to 0.0045 % of training pixels, so a whole window in that state lies outside training); the normal inference reproduces the stored prob_oof.tif (interior of the window: mean absolute difference below 1e-4, maximum below 5e-3); the blank inference differs from it, so the blank reaches the model (reported).
The limits: inference runs under bf16 autocast. One rounding step of the final logit moves p by at most 1.6e-3: the step is 2/128 = 0.0156 between 2 and 4, where the slope of the sigmoid is at most 0.105 (the slope 0.25 holds at logit 0, where the step is far smaller). The same inference with batch 16 and batch 7 differs by up to 8.9e-3 (E1b) and 7.8e-3 (E2), means 4.5e-5 and 4.0e-5 (noise_max_abs, noise_mean_abs), which is 5.4 and 4.8 times that step, so the differences arise upstream of the final logit. The limit of 5e-3 on the maximum is therefore also below the measured noise; it holds because the test compares at batch 16, the batch of the stored runs, where its maxima (2.6e-3, 3.4e-3) repeat to all digits between the two self-tests. The mean carries the test: limit 1e-4, noise 4.5e-5 and 4.0e-5, measured 1.5e-5 and 1.0e-5. The first self-test (d2_selftest/, limit 2e-3 on the maximum) failed on that limit alone: maxima 2.6e-3 and 3.4e-3, means 1.5e-5 and 1.0e-5.
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from itareo.safe_write import refuse_existing  # noqa: E402
from itareo.ugi_seg.evaluate import predict_region  # noqa: E402
from itareo.ugi_seg.oof import infer_oof  # noqa: E402
from itareo.ugi_seg.ps_branch import PS_CHANNELS, N_PS, STAGE_PX, WindowPS, build_e1, ps_tensor  # noqa: E402
from itareo.ugi_seg.score import score_map  # noqa: E402
from itareo.ugi_seg.tex_branch import N_LAYERS, TEX_CHANNELS, WindowTex, build_e2, tex_tensor  # noqa: E402

ROOT = Path("runs_itareo/ugi")
ARMS = {"ugi_e1b_ps_plane": {"tag": "e1b", "shape": [PS_CHANNELS, STAGE_PX, STAGE_PX], "build": build_e1, "drop": "planetscope"},
        "ugi_e2_tex": {"tag": "e2", "shape": [TEX_CHANNELS, 512, 512], "build": build_e2, "drop": "texture"}}


def load(arm, fold, seed, device):
    ck = torch.load(ROOT / arm / f"fold{fold}_seed{seed}" / "best.pt", map_location=device, weights_only=False)
    model = ARMS[arm]["build"](ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def blank_cfg(arm, cfg):
    c = copy.deepcopy(cfg)
    del c["model"][ARMS[arm]["drop"]]
    c["model"]["plugin"] = {"module": "itareo.ugi_seg.blank_input", "shape": ARMS[arm]["shape"]}
    return c


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def run(arm, fold, seed):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src = ROOT / arm / f"fold{fold}_seed{seed}"
    out = ROOT / f"diag_d2_{ARMS[arm]['tag']}_blank" / f"fold{fold}_seed{seed}"
    refuse_existing(out / "run_log.md", out / "prob_oof.tif")
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load(arm, fold, seed, device)
    cfg = blank_cfg(arm, ck["config"])
    torch.save({"config": cfg, "source_checkpoint": str(src / "best.pt")}, out / "best.pt")        # first: the rescoring reads the config from it as soon as metrics.json exists
    prob, region, secs = infer_oof(model, ck, cfg, fold, out, device)
    gs = {int(r["id"]): np.array([float(r[f"sum_p_group{g}"]) for g in range(1, 6)]) for r in csv.DictReader(open(out / "oof_groups.csv"))}
    sp = {"labels": cfg["paths"]["labels"], "polygons": cfg["paths"]["polygons"], "ndvi": cfg["paths"]["ndvi"], "sid": cfg["paths"]["sid"]}
    res, objs, unmatched, blocks, boot = score_map(prob, prob >= ck["threshold"], region[0], region[1], sp, gs)
    res.update({"fold": fold, "seed": seed, "threshold": ck["threshold"], "region_rows_cols": region, "blanked": f"the added input of {arm} at its no-data value", "source_checkpoint": str(src / "best.pt"), "scored_from": "prob_oof.tif as stored"})
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=float)
    write_csv(out / "objects.csv", objs)
    write_csv(out / "components_unmatched.csv", unmatched)
    write_csv(out / "blocks_1km.csv", blocks)
    np.savez_compressed(out / "boot.npz", **boot)
    with open(out / "run_log.md", "w", encoding="utf-8") as f:
        f.write(f"# D2 (CLAUDE.md 13.8): {arm} fold {fold} seed {seed}, the added input blanked at inference\n\n- command: `{' '.join(sys.argv)}` (cwd `{os.getcwd()}`)\n- checkpoint `{src / 'best.pt'}`, best epoch {ck['epoch']}, threshold {ck['threshold']}\n- start {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0))}, end {time.strftime('%Y-%m-%dT%H:%M:%S')}, {round(time.time() - t0)} s, inference {secs} s\n- held-out AP {res['pixel_full']['ap']}, AUROC {res['pixel_full']['auroc']}, IoU {res['pixel_full']['iou']}\n")
    print(arm, "fold", fold, "seed", seed, "AP", res["pixel_full"]["ap"], "AUROC", res["pixel_full"]["auroc"], round(time.time() - t0), "s", flush=True)


def infer_window(model, ck, cfg, region, extra, device, batch=16):
    p = np.full(region[2:], np.nan, np.float32)
    for b0, pu, _ in predict_region(model, cfg["paths"]["image"], tuple(region), ck["mean"], ck["sd"], n=cfg["eval"]["window"], band_rows=cfg["eval"]["band_rows"], batch=batch, device=device, extra=extra):
        p[b0:b0 + pu.shape[0]] = pu
    return p


def selftest(out):
    out = Path(out)
    refuse_existing(out / "d2_selftest.json")
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"device": device, "ps_nodata_all_zero": bool((ps_tensor(np.full((N_PS, STAGE_PX, STAGE_PX), np.nan, np.float32), np.ones(N_PS), np.ones(N_PS)) == 0).all()),
           "tex_nodata_all_zero": bool((tex_tensor(np.full((N_LAYERS, 512, 512), np.nan, np.float32), np.ones(N_LAYERS), np.ones(N_LAYERS)) == 0).all()), "arms": {}}
    H, N = 256, 1024
    for arm in ARMS:
        t0 = time.time()
        model, ck = load(arm, 1, 0, device)
        cfg = ck["config"]
        run_d = ROOT / arm / "fold1_seed0"
        r_lo, c_lo, R, Cn = json.load(open(run_d / "prob_oof.json"))["region_rows_cols"]
        region = (r_lo + 16 * H, c_lo + 16 * H, N, N)           # on the lattice of the stored run, windows every 256 px from its corner
        if arm == "ugi_e1b_ps_plane":
            extra = WindowPS(ck["ps_mean"], ck["ps_sd"], cfg["model"]["planetscope"]["registration"])
        else:
            extra = WindowTex(cfg, 1, ck["tex_mean"], ck["tex_sd"], region)
        p = infer_window(model, ck, cfg, region, extra, device)
        p7 = infer_window(model, ck, cfg, region, extra, device, batch=7)
        from itareo.ugi_seg.blank_input import window
        q = infer_window(model, ck, cfg, region, window(blank_cfg(arm, cfg), 1, ck, region), device)
        with rasterio.open(run_d / "prob_oof.tif") as s:
            stored = s.read(1, window=Window(region[1] - c_lo, region[0] - r_lo, N, N))
        inner = (slice(H, N - H), slice(H, N - H))
        ok = np.isfinite(stored[inner])
        d = np.abs(p[inner] - stored[inner])[ok]
        res["arms"][arm] = {"window_row_col_rows_cols": region, "interior_pixels_compared": int(ok.sum()), "max_abs_diff_to_stored": float(d.max()), "mean_abs_diff_to_stored": float(d.mean()), "share_of_pixels_over_1e-3": float((d > 1e-3).mean()), "noise_max_abs": float(np.abs(p[inner] - p7[inner]).max()), "noise_mean_abs": float(np.abs(p[inner] - p7[inner]).mean()), "mean_abs_diff_blank_vs_normal": float(np.abs(q[inner] - p[inner])[ok].mean()), "seconds": round(time.time() - t0, 1)}
        print(arm, res["arms"][arm], flush=True)
    res["passed"] = bool(res["ps_nodata_all_zero"] and res["tex_nodata_all_zero"] and all(v["max_abs_diff_to_stored"] < 5e-3 and v["mean_abs_diff_to_stored"] < 1e-4 and v["interior_pixels_compared"] > 1000 and v["mean_abs_diff_blank_vs_normal"] > 0 for v in res["arms"].values()))
    json.dump(res, open(out / "d2_selftest.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return res["passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest(a.out) else 1)
    run(a.arm, a.fold, a.seed)
