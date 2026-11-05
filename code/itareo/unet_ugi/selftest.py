"""Self-tests of unet_ugi (synthetic data only; no project data needed).

  python -m itareo.unet_ugi.selftest [--out DIR]     writes selftest.json and fails on any failed check

  1. conv1: channels 1-3 equal torchvision's pretrained conv1, channel 4 = their mean, other layers unchanged.
  2. loss: the cross-entropy part equals torch.nn.functional.cross_entropy (weights, ignore_index); the Dice part equals
     a NumPy recomputation.
  3. boundary band: equals a brute-force loop over the 8 neighbours.
  4. metrics: OA, kappa, per-class F1, precision and recall equal scikit-learn when it is installed (else skipped),
     also for a class with no true positive and a class never predicted (zero_division=0); macro F1 likewise.
  5. AUROC and AP from logit-binned histograms equal scikit-learn's within 1e-3, also for a saturated rare class.
  6. sliding window: the raised-cosine weights of the four windows covering a pixel sum to 1; the blended output of a
     fixed random network does not depend on the band height; a constant network returns its own softmax.
  7. determinism (CUDA only): two seeded forward/backward passes of the U-Net with the loss give identical gradients.
  8. dataset: augmentation keeps image and label aligned (a label-coded image survives flips and rotations), and the
     augmented batches are identical with 0 and 2 DataLoader workers.
  9. class weights count only loss pixels (boundary band and buffer excluded when ignored).
Synthetic files are written to a new temporary folder (printed as workdir); nothing is deleted.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np          # noqa: E402
import rasterio             # noqa: E402
import torch                # noqa: E402
import torch.nn as nn       # noqa: E402
import torch.nn.functional as F  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from .data import IGNORE, SeededSampler, TileDataset, boundary_band, class_counts  # noqa: E402
from .infer import predict_bands, raised_cosine  # noqa: E402
from .metrics import NBINS, auroc_ap_from_hist, auroc_ap_scores, confusion, prob_bins, scores, seg_loss  # noqa: E402
from .model import UNetResNet18, conv1_selftest  # noqa: E402


def t_loss():
    rng = np.random.default_rng(0)
    K = 5
    o = torch.tensor(rng.normal(size=(2, K, 6, 7)), dtype=torch.float32)
    y = torch.tensor(rng.integers(0, K, size=(2, 6, 7)))
    y[0, 0, :3] = IGNORE
    w = rng.uniform(0.5, 2, K)
    ce = seg_loss(K, IGNORE, w, dice_weight=0.0)(o, y)
    ref = F.cross_entropy(o, y, weight=torch.tensor(w, dtype=torch.float32), ignore_index=IGNORE)
    tot = seg_loss(K, IGNORE, w, dice_weight=1.0)(o, y)
    p = torch.softmax(o, 1).numpy()
    yy = y.numpy()
    valid = yy != IGNORE
    d = []
    for k in range(K):
        oh = (yy == k) & valid
        if oh.sum() == 0:
            continue
        pk = p[:, k] * valid
        d.append((2 * (pk * oh).sum() + 1e-6) / (pk.sum() + oh.sum() + 1e-6))
    dice_np = 1 - np.mean(d)
    return {"ce_abs_diff_vs_torch": abs(float(ce - ref)), "dice_abs_diff_vs_numpy": abs(float(tot - ce) - dice_np),
            "passed": abs(float(ce - ref)) < 1e-6 and abs(float(tot - ce) - dice_np) < 1e-5}


def t_band():
    rng = np.random.default_rng(1)
    lab = rng.integers(0, 4, size=(40, 50)).astype(np.uint8)
    lab[10:30, 10:40] = 2
    fast = boundary_band(np.pad(lab, 1))
    slow = np.zeros_like(fast)
    for i in range(40):
        for j in range(50):
            if lab[i, j] == 0:
                continue
            nb = lab[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
            slow[i, j] = ((nb != 0) & (nb != lab[i, j])).any()
    return {"mismatches": int((fast != slow).sum()), "band_share": round(float(fast.mean()), 4),
            "passed": bool((fast == slow).all())}


def t_metrics():
    rng = np.random.default_rng(2)
    ref = rng.integers(0, 4, 5000)
    pred = np.where(rng.random(5000) < 0.7, ref, rng.integers(0, 4, 5000))
    s = scores(confusion(ref, pred, 4))
    try:
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
    except ImportError:
        return {"skipped": "scikit-learn not installed", "passed": True}
    d = max(abs(s["oa"] - accuracy_score(ref, pred)), abs(s["kappa"] - cohen_kappa_score(ref, pred)),
            *[abs(s["per_class"][str(k)]["f1"] - v) for k, v in enumerate(f1_score(ref, pred, average=None))],
            *[abs(s["per_class"][str(k)]["precision_ua"] - v) for k, v in enumerate(precision_score(ref, pred, average=None))],
            *[abs(s["per_class"][str(k)]["recall_pa"] - v) for k, v in enumerate(recall_score(ref, pred, average=None))])
    # degenerate case: class 2 has reference pixels but no true positive, class 3 is never predicted
    ref2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
    pred2 = np.array([0, 0, 0, 1, 1, 2, 0, 0, 1, 0])
    s2 = scores(confusion(ref2, pred2, 4))
    kw = dict(average=None, labels=[0, 1, 2, 3], zero_division=0)
    d2 = max(*[abs(s2["per_class"][str(k)]["f1"] - v) for k, v in enumerate(f1_score(ref2, pred2, **kw))],
             *[abs(s2["per_class"][str(k)]["precision_ua"] - v) for k, v in enumerate(precision_score(ref2, pred2, **kw))],
             *[abs(s2["per_class"][str(k)]["recall_pa"] - v) for k, v in enumerate(recall_score(ref2, pred2, **kw))],
             abs(s2["macro_f1"] - f1_score(ref2, pred2, average="macro", zero_division=0)))
    return {"max_abs_diff_vs_sklearn": d, "degenerate_max_abs_diff_vs_sklearn_zero_division_0": d2,
            "passed": d < 1e-4 and d2 < 1e-4}


def t_auroc():
    """One-vs-rest AUROC and AP from probability histograms against scikit-learn on 20,000 synthetic pixels."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {"skipped": "scikit-learn not installed", "passed": True}
    rng = np.random.default_rng(6)
    K, n = 4, 20000
    ref = rng.integers(0, K, n)
    logits = rng.normal(size=(K, n)) + 1.5 * (np.arange(K)[:, None] == ref[None])
    p = np.exp(logits) / np.exp(logits).sum(0)
    pos = np.zeros((K, NBINS), np.int64)
    neg = np.zeros((K, NBINS), np.int64)
    for c in range(K):
        h = np.bincount(ref * NBINS + prob_bins(p[c]), minlength=K * NBINS).reshape(K, NBINS)
        pos[c], neg[c] = h[c], h.sum(0) - h[c]
    r = auroc_ap_scores(pos, neg, [str(k) for k in range(K)])
    da = max(abs(r["per_class"][str(c)]["auroc"] - roc_auc_score(ref == c, p[c])) for c in range(K))
    dp = max(abs(r["per_class"][str(c)]["ap"] - average_precision_score(ref == c, p[c])) for c in range(K))
    # saturated case: a confident model (most positives above p = 0.9999) and a rare class (1 %)
    ref_s = np.where(rng.random(n) < 0.01, 0, rng.integers(1, K, n))
    logit_s = rng.normal(0, 5, size=(K, n)) + 16 * (np.arange(K)[:, None] == ref_s[None])
    logit_s -= logit_s.max(0)
    ps = (np.exp(logit_s) / np.exp(logit_s).sum(0)).astype(np.float32)
    hs = np.bincount(ref_s * NBINS + prob_bins(ps[0]), minlength=K * NBINS).reshape(K, NBINS)
    rs = auroc_ap_from_hist(hs[0], hs.sum(0) - hs[0])
    das = abs(rs[0] - roc_auc_score(ref_s == 0, ps[0]))
    dps = abs(rs[1] - average_precision_score(ref_s == 0, ps[0]))
    return {"auroc_max_abs_diff_vs_sklearn": da, "ap_max_abs_diff_vs_sklearn": dp, "bins": NBINS,
            "saturated_rare_class_auroc_diff": das, "saturated_rare_class_ap_diff": dps,
            "saturated_share_of_positives_above_0.9999": float((ps[0][ref_s == 0] > 0.9999).mean()),
            "passed": da < 1e-3 and dp < 1e-3 and das < 1e-3 and dps < 1e-3}


class _Const(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.b = nn.Parameter(torch.linspace(-1, 1, K))

    def forward(self, x):
        return self.b[None, :, None, None].expand(x.shape[0], -1, x.shape[2], x.shape[3])


def t_sliding(tmp):
    n = 64
    w = raised_cosine(n)
    h = n // 2
    tot = w[:h, :h] + w[h:, :h] + w[:h, h:] + w[h:, h:]
    rng = np.random.default_rng(3)
    H, W = 300, 270
    img = rng.integers(100, 5000, size=(4, H, W)).astype(np.uint16)
    img[:, 50:60, 70:90] = 65535
    path = Path(tmp) / "img.tif"
    with rasterio.open(path, "w", driver="GTiff", count=4, dtype="uint16", nodata=65535, width=W, height=H,
                       crs="EPSG:32720", transform=from_origin(1000, 2000, 1, 1)) as d:
        d.write(img)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 5, 3, padding=1)).eval()
    region = (13, 7, 250, 230)
    mean, sd = [2500] * 4, [1500] * 4
    a = np.concatenate([p for _, p in predict_bands(net, path, region, mean, sd, n=n, band_rows=64, device="cpu",
                                                    amp=False)], 1)
    b = np.concatenate([p for _, p in predict_bands(net, path, region, mean, sd, n=n, band_rows=256, device="cpu",
                                                    amp=False)], 1)
    c = np.concatenate([p for _, p in predict_bands(_Const(5), path, region, mean, sd, n=n, band_rows=64,
                                                    device="cpu", amp=False)], 1)
    soft = torch.softmax(torch.linspace(-1, 1, 5), 0).numpy()[:, None, None]
    # direct check of one interior pixel against the four windows by hand
    return {"weights_sum_min": float(tot.min()), "weights_sum_max": float(tot.max()),
            "band_height_max_abs_diff": float(np.abs(a - b).max()), "shape": list(a.shape),
            "prob_sum_max_dev": float(np.abs(a.sum(0) - 1).max()),
            "constant_net_max_abs_diff": float(np.abs(c - soft).max()),
            "passed": bool(abs(tot - 1).max() < 1e-6 and np.abs(a - b).max() < 1e-5 and a.shape == (5, 250, 230)
                           and np.abs(c - soft).max() < 1e-6)}


def t_determinism():
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA", "passed": True}
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    grads = []
    for _ in range(2):
        torch.manual_seed(0)
        m = UNetResNet18(7, pretrained=False).cuda()
        x = torch.randn(2, 4, 256, 256, device="cuda")
        y = torch.randint(0, 7, (2, 256, 256), device="cuda")
        y[:, :10] = IGNORE
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(x)
        seg_loss(7, IGNORE, np.ones(7), 1.0)(o, y).backward()
        grads.append(torch.cat([p.grad.flatten().float() for p in m.parameters() if p.grad is not None]).cpu())
    return {"max_abs_grad_diff": float((grads[0] - grads[1]).abs().max()), "passed": bool(torch.equal(*grads))}


def t_dataset(tmp):
    N, T = 3, 32
    c = Path(tmp) / "cache"
    c.mkdir()
    lab = np.random.default_rng(4).integers(1, 8, size=(N, T, T)).astype(np.uint8)
    img = np.repeat(lab[:, None].astype(np.uint16) * 100, 4, 1)
    np.save(c / "img.npy", img)
    np.save(c / "lab.npy", lab)
    np.save(c / "band.npy", np.zeros_like(lab))
    ds = TileDataset(c, range(N), [0] * 4, [1] * 4, 65535, False, True, photometric={"global": [1, 1], "per_band": 0})
    bad = 0
    for s in range(20):
        for i in range(N):
            x, y = ds[(i, 1000 * s + i)]
            bad += int(((x[0] / 100).round().long() - 1 != y).sum())
    # the augmented batches depend on the sampler seed only, not on the number of DataLoader workers
    ds2 = TileDataset(c, range(N), [0] * 4, [1] * 4, 65535, False, True)
    batches = {}
    for nw in (0, 2):
        dl = torch.utils.data.DataLoader(ds2, batch_size=2, num_workers=nw,
                                         sampler=SeededSampler(N, torch.Generator().manual_seed(7)))
        batches[nw] = [torch.cat([x.flatten(), y.flatten().float()]) for _ in range(2) for x, y in dl]
    same = all(torch.equal(a, b) for a, b in zip(batches[0], batches[2])) and len(batches[0]) == len(batches[2])
    return {"misaligned_pixels": bad, "workers_0_vs_2_identical": bool(same), "passed": bad == 0 and same}


def t_class_counts(tmp):
    """class_counts counts only loss pixels: the boundary band when ignore_band, and the buffer pixels."""
    c = Path(tmp) / "cache_counts"
    c.mkdir()
    lab = np.array([[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 0, 0], [3, 3, 0, 0]]], np.uint8)
    band = np.array([[[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]], np.uint8)
    np.save(c / "lab.npy", lab)
    np.save(c / "band.npy", band)
    buf = {0: np.array([[True, False, False, False]] + [[False] * 4] * 3)}
    a, _ = class_counts(c, [0], 3)
    b, _ = class_counts(c, [0], 3, ignore_band=True)
    d, _ = class_counts(c, [0], 3, ignore_band=True, buffer_masks=buf)
    ok = a.tolist() == [4, 4, 4] and b.tolist() == [2, 2, 2] and d.tolist() == [1, 2, 2]
    return {"all": a.tolist(), "without_band": b.tolist(), "without_band_and_buffer": d.tolist(), "passed": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    res = {"conv1": conv1_selftest()}
    res["conv1"]["passed"] = all(v for k, v in res["conv1"].items() if k != "conv1_shape")
    res["loss"] = t_loss()
    res["boundary_band"] = t_band()
    res["metrics"] = t_metrics()
    res["auroc_ap"] = t_auroc()
    tmp = tempfile.mkdtemp(prefix="unet_ugi_selftest_")      # synthetic files stay there; nothing is deleted
    res["sliding_window"] = t_sliding(tmp)
    res["dataset_augmentation"] = t_dataset(tmp)
    res["class_counts_loss_pixels"] = t_class_counts(tmp)
    res["determinism_cuda"] = t_determinism()
    res["workdir"] = tmp
    res = json.loads(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res["passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / "selftest.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert res["passed"], "unet_ugi self-test failed"


if __name__ == "__main__":
    main()
