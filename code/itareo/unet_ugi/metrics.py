"""Loss, confusion matrices and scores.

Confusion matrix C[K, K]: rows = reference class, columns = predicted class, counts of pixels.
Per class: IoU = C_cc / (row_c + col_c - C_cc); precision (user's accuracy) = C_cc / col_c; recall (producer's
accuracy) = C_cc / row_c; F1 = 2 P R / (P + R). OA = trace / total; Cohen's kappa; mIoU and macro F1 over the classes
with reference pixels. Results against the GEOBIA map are agreement with that map, not accuracy.
"""
import numpy as np
import torch
import torch.nn.functional as F


def seg_loss(n_cls, ignore, weights, dice_weight=1.0, eps=1e-6):
    """Weighted cross-entropy over labelled pixels (sum w[y] nll / sum w[y], torch's weighted mean) plus dice_weight x
    (1 - mean soft Dice over the classes present in the batch labels). Written with one-hot products, which have
    deterministic CUDA kernels (torch's CrossEntropyLoss on [B, C, H, W] has none)."""
    w = torch.as_tensor(np.asarray(weights, np.float32))

    def loss_fn(o, y):
        valid = y != ignore
        oh = F.one_hot(torch.where(valid, y, torch.zeros_like(y)), n_cls).permute(0, 3, 1, 2).float() * valid[:, None]
        lp = F.log_softmax(o.float(), 1)
        ohw = oh * w.to(o.device)[None, :, None, None]
        loss = -(lp * ohw).sum() / ohw.sum().clamp(min=1e-12)
        if dice_weight:
            p = lp.exp() * valid[:, None]
            inter, card, n_c = (p * oh).sum((0, 2, 3)), p.sum((0, 2, 3)) + oh.sum((0, 2, 3)), oh.sum((0, 2, 3))
            present = n_c > 0
            if present.any():
                loss = loss + dice_weight * (1 - ((2 * inter + eps) / (card + eps))[present].mean())
        return loss
    return loss_fn


def confusion(ref, pred, n_cls, mask=None):
    """ref, pred: integer arrays (ref in 0..K-1 where counted); mask: boolean of pixels to count."""
    ref, pred = np.asarray(ref).ravel(), np.asarray(pred).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel()
        ref, pred = ref[m], pred[m]
    return np.bincount(ref.astype(np.int64) * n_cls + pred, minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def scores(C, names=None):
    """Zero-division rule as scikit-learn's zero_division=0 for classes with reference pixels: a class never predicted
    has precision 0, a class with no true positive has F1 0. Classes without reference pixels get None for recall and
    are left out of mIoU and macro F1."""
    C = np.asarray(C, np.float64)
    n = C.sum()
    tp, row, col = np.diag(C), C.sum(1), C.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(row + col - tp > 0, tp / (row + col - tp), np.nan)
        prec = np.where(col > 0, tp / col, np.where(row > 0, 0.0, np.nan))
        rec = np.where(row > 0, tp / row, np.nan)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), np.where(row > 0, 0.0, np.nan))
    pe = (row * col).sum() / n ** 2 if n else np.nan
    oa = tp.sum() / n if n else np.nan
    present = row > 0
    names = names or [str(k) for k in range(len(tp))]
    per = {nm: {"iou": iou[k], "f1": f1[k], "precision_ua": prec[k], "recall_pa": rec[k], "reference_px": int(row[k]),
                "predicted_px": int(col[k])} for k, nm in enumerate(names)}
    per = {k: {a: (None if isinstance(b, float) and not np.isfinite(b) else (round(float(b), 5) if isinstance(b, float)
                                                                              else b)) for a, b in v.items()}
           for k, v in per.items()}
    return {"oa": round(float(oa), 5), "kappa": round(float((oa - pe) / (1 - pe)), 5) if n else None,
            "miou": round(float(np.mean(iou[present])), 5), "macro_f1": round(float(np.mean(f1[present])), 5),
            "pixels": int(n), "per_class": per, "confusion": C.astype(np.int64).tolist()}


NBINS = 100000      # histogram bins for AUROC and AP, on the logit scale
LOGIT_MAX = 20.0    # logit(p) is clipped to [-20, 20]; bin width 4e-4 in logit units


def prob_bins(p):
    """Bin index 0..NBINS-1 of probabilities in [0, 1], on the logit scale so that saturated probabilities (close to 0
    or 1) are not tied in one bin; monotone in p, so the ranking is kept up to ties within a bin."""
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore"):
        z = np.log(p) - np.log1p(-p)
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=LOGIT_MAX, neginf=-LOGIT_MAX), -LOGIT_MAX, LOGIT_MAX)
    return np.minimum(((z + LOGIT_MAX) / (2 * LOGIT_MAX) * NBINS).astype(np.int64), NBINS - 1)


def auroc_ap_from_hist(pos, neg):
    """One-vs-rest AUROC and average precision from histograms of the class probability over positives and
    negatives (NBINS bins). Thresholds run down the bins; scores within one bin are ties (trapezoidal ROC, and the
    precision at each tied block as in scikit-learn's average_precision_score). Returns (auroc, ap); None if a class
    has no positives or no negatives."""
    pos, neg = np.asarray(pos, np.float64)[::-1], np.asarray(neg, np.float64)[::-1]
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return None, None
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    tpr, fpr = np.concatenate([[0], tp / P]), np.concatenate([[0], fp / N])
    auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    keep = pos > 0
    prec = tp / np.maximum(tp + fp, 1)
    ap = float(np.sum((pos[keep] / P) * prec[keep]))
    return auroc, ap


def auroc_ap_scores(hist_pos, hist_neg, names):
    """hist_pos, hist_neg: (K, NBINS) arrays. Per-class AUROC and AP, and their means over classes with positives."""
    per = {}
    for k, nm in enumerate(names):
        a, p = auroc_ap_from_hist(hist_pos[k], hist_neg[k])
        per[nm] = {"auroc": None if a is None else round(a, 5), "ap": None if p is None else round(p, 5),
                   "positives": int(np.sum(hist_pos[k]))}
    va = [v["auroc"] for v in per.values() if v["auroc"] is not None]
    vp = [v["ap"] for v in per.values() if v["ap"] is not None]
    return {"macro_auroc": round(float(np.mean(va)), 5) if va else None,
            "macro_ap": round(float(np.mean(vp)), 5) if vp else None, "per_class": per,
            "method": f"one-vs-rest on the blended softmax probability, histograms of {NBINS} bins of logit(p) in [-{LOGIT_MAX}, {LOGIT_MAX}]"}
