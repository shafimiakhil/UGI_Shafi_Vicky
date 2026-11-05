"""Summary of the UGI runs (A1, A2, ...): pooled over folds per seed, per fold, object recall by group and by size,
and a paired comparison of two configurations.

Sources per run folder runs_itareo/ugi/<config>/fold<f>_seed<s>/:
  subset 'run'            metrics.json + objects.csv (the run's own held-out evaluation);
  subset 'all'|'b4'|'b8'  rescore/metrics_<subset>.json + objects_<subset>.csv + hist_<subset>.npz (itareo.ugi_seg.rescore).
Pooling over the four held-out folds of one seed: pixel confusion counts summed (IoU, F1, precision, recall); AUROC and
AP from the summed probability histograms (only where rescore histograms exist); objects concatenated (each polygon is
held out in exactly one fold) for object recall overall, by group and by size and for the type scores; boundary F is the
mean over folds (it is not poolable from the stored numbers). Seeds: mean and SD (ddof 1) of the pooled values.
Comparison (B against A, same fold and seed): differences of the pooled pixel scores per seed; object recall difference
with a 95 % bootstrap interval over held-out polygons (2,000 resamples of polygon ids, the same ids for both
configurations, recall averaged over seeds).
  python -m itareo.ugi_seg.summarize --configs ugi_a1 ugi_a2 [--subset run] [--out runs_itareo/ugi/summary]   (from work_ms/)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from itareo.unet_ugi.metrics import auroc_ap_from_hist

from .evaluate import GROUP_NAMES, SIZE_EDGES, SIZE_NAMES, pixel_scores

ROOT = Path("runs_itareo/ugi")
GROUPS = GROUP_NAMES + ["NO DEFINIDO"]


def load_run(run, subset):
    if subset == "run":
        m, o, h = run / "metrics.json", run / "objects.csv", None
    else:
        m, o, h = run / "rescore" / f"metrics_{subset}.json", run / "rescore" / f"objects_{subset}.csv", \
            run / "rescore" / f"hist_{subset}.npz"
    if not m.exists():
        return None
    met = json.load(open(m))
    objs = [{"id": int(r["id"]), "group": int(r["group"]), "area_m2": float(r["area_m2"]),
             "detected": r["detected"] == "True", "pred_group": int(r["pred_group"])}
            for r in csv.DictReader(open(o))] if o.exists() else []
    hist = dict(np.load(h)) if h is not None and h.exists() else None
    return {"metrics": met, "objects": objs, "hist": hist, "path": str(run)}


def object_scores(objs):
    det = [o["detected"] for o in objs]
    out = {"objects": len(objs), "object_recall": float(np.mean(det)) if det else None}
    for g, n in enumerate(GROUPS, 1):
        d = [o["detected"] for o in objs if o["group"] == g]
        out[f"recall_group_{n}"] = float(np.mean(d)) if d else None
        out[f"n_group_{n}"] = len(d)
    for k, n in enumerate(SIZE_NAMES):
        d = [o["detected"] for o in objs if SIZE_EDGES[k] <= o["area_m2"] < SIZE_EDGES[k + 1]]
        out[f"recall_size_{n}"] = float(np.mean(d)) if d else None
        out[f"n_size_{n}"] = len(d)
    typed = [o for o in objs if o["detected"] and 1 <= o["group"] <= 5]
    if typed:
        t = np.array([o["group"] for o in typed])
        p = np.array([o["pred_group"] for o in typed])
        f1s = []
        for g in sorted(set(t)):
            tp = np.sum((t == g) & (p == g))
            pr, rc = tp / max(np.sum(p == g), 1), tp / max(np.sum(t == g), 1)
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        out.update({"type_polygons": len(typed), "type_accuracy": float((t == p).mean()), "type_macro_f1": float(np.mean(f1s))})
    return out


def pooled(runs):
    """runs: the four fold runs of one configuration and seed."""
    out = {}
    for key in ("full", "no_edge"):
        conf = np.array([sum(r["metrics"][f"pixel_{key}"][c] for r in runs) for c in ("tp", "fp", "fn", "tn")])
        s = pixel_scores(*conf)
        if all(r["hist"] is not None for r in runs):
            pos = np.sum([r["hist"][f"{key}_pos"] for r in runs], axis=0)
            neg = np.sum([r["hist"][f"{key}_neg"] for r in runs], axis=0)
            s["auroc"], s["ap"] = auroc_ap_from_hist(pos, neg)
        else:
            s["auroc"] = s["ap"] = None
        out.update({f"{key}_{k}": v for k, v in s.items() if k in ("iou", "f1", "precision", "recall", "auroc", "ap")})
    bf = [r["metrics"].get("boundary_f") for r in runs]
    out["boundary_f_mean_over_folds"] = float(np.mean(bf)) if all(b is not None for b in bf) else None
    out.update(object_scores([o for r in runs for o in r["objects"]]))
    return out


def per_fold(r):
    m = r["metrics"]
    out = {f"{k}_{x}": m[f"pixel_{k}"][x] for k in ("full", "no_edge") for x in ("iou", "f1", "precision", "recall", "auroc", "ap")}
    out.update({"boundary_f": m.get("boundary_f"), "threshold": m.get("threshold"), "best_epoch": m.get("best_epoch"),
                "best_val_ap": m.get("best_val_ap")})
    out.update(object_scores(r["objects"]))
    return out


def mean_sd(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[f"{k}_mean"] = float(np.mean(v)) if v else None
        out[f"{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        out[f"{k}_n"] = len(v)
    return out


def collect(config, subset, folds=(1, 2, 3, 4), seeds=(0, 1, 2)):
    runs = {}
    for s in seeds:
        for f in folds:
            r = load_run(ROOT / config / f"fold{f}_seed{s}", subset)
            if r is not None:
                runs[(f, s)] = r
    return runs


def bootstrap_recall_diff(objs_a, objs_b, n=2000, seed=0):
    """objs_*: {seed: {polygon id: detected}}; same ids in both. Returns (diff, lo, hi) of mean-over-seeds recall B - A."""
    seeds = sorted(set(objs_a) & set(objs_b))
    ids = sorted(set.intersection(*[set(objs_a[s]) & set(objs_b[s]) for s in seeds]))
    if not ids:
        return None, None, None
    A = np.array([[objs_a[s][i] for i in ids] for s in seeds], float).mean(0)
    B = np.array([[objs_b[s][i] for i in ids] for s in seeds], float).mean(0)
    d = B - A
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(ids), len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["ugi_a1", "ugi_a2"])
    ap.add_argument("--subset", default="run", choices=["run", "all", "b4", "b8", "strip_1_2", "strip_1_3", "strip_1_5", "strip_1_6"])
    ap.add_argument("--out", default="runs_itareo/ugi/summary")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.subset
    pooled_rows, fold_rows, seed_rows = [], [], []
    store = {}
    for c in a.configs:
        runs = collect(c, a.subset)
        store[c] = runs
        for (f, s), r in sorted(runs.items()):
            fold_rows.append({"config": c, "fold": f, "seed": s, **per_fold(r), "source": r["path"]})
        for s in sorted({s for _, s in runs}):
            rs = [runs[(f, s)] for f in (1, 2, 3, 4) if (f, s) in runs]
            if len(rs) == 4:
                seed_rows.append({"config": c, "seed": s, **pooled(rs)})
        keys = [k for k in (seed_rows[-1] if seed_rows else {}) if k not in ("config", "seed")
                and not k.startswith("n_") and k not in ("objects", "type_polygons")]
        cs = [r for r in seed_rows if r["config"] == c]
        if cs:
            pooled_rows.append({"config": c, "seeds": len(cs), **mean_sd(cs, keys)})
    for name, rows in (("pooled_per_seed", seed_rows), ("pooled_mean_sd", pooled_rows), ("per_fold", fold_rows)):
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}_{tag}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    # per fold, mean and SD over seeds
    pf = []
    for c in a.configs:
        for f in (1, 2, 3, 4):
            rows = [r for r in fold_rows if r["config"] == c and r["fold"] == f]
            if rows:
                pf.append({"config": c, "fold": f, "seeds": len(rows),
                           **mean_sd(rows, ["full_iou", "full_f1", "full_auroc", "full_ap", "no_edge_iou", "boundary_f",
                                            "object_recall", "type_macro_f1"])})
    if pf:
        with open(out / f"per_fold_mean_sd_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in pf for k in r)))
            w.writeheader()
            w.writerows(pf)
    comp = []
    if len(a.configs) >= 2:
        ca, cb = a.configs[0], a.configs[1]
        for s in sorted({r["seed"] for r in seed_rows if r["config"] == ca} & {r["seed"] for r in seed_rows if r["config"] == cb}):
            ra = next(r for r in seed_rows if r["config"] == ca and r["seed"] == s)
            rb = next(r for r in seed_rows if r["config"] == cb and r["seed"] == s)
            comp.append({"seed": s, **{f"d_{k}": (rb[k] - ra[k]) if rb.get(k) is not None and ra.get(k) is not None else None
                                       for k in ("full_iou", "full_f1", "full_auroc", "full_ap", "no_edge_iou", "object_recall",
                                                 "type_macro_f1", "boundary_f_mean_over_folds")}})
        oa = {s: {o["id"]: o["detected"] for (f, ss), r in store[ca].items() if ss == s for o in r["objects"]}
              for s in {s for _, s in store[ca]}}
        ob = {s: {o["id"]: o["detected"] for (f, ss), r in store[cb].items() if ss == s for o in r["objects"]}
              for s in {s for _, s in store[cb]}}
        complete = lambda st, s: all((f, s) in st for f in (1, 2, 3, 4))
        oa = {s: v for s, v in oa.items() if complete(store[ca], s)}
        ob = {s: v for s, v in ob.items() if complete(store[cb], s)}
        bt = bootstrap_recall_diff(oa, ob) if oa and ob else (None,) * 5
        comp_summary = {"a": ca, "b": cb, "object_recall_diff_b_minus_a": bt[0], "ci95": [bt[1], bt[2]] if bt[0] is not None else None,
                        "polygons": bt[3] if len(bt) > 3 else None, "seeds": bt[4] if len(bt) > 4 else None,
                        "per_seed": comp}
        if comp:
            with open(out / f"compare_{ca}_{cb}_{tag}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(comp[0]))
                w.writeheader()
                w.writerows(comp)
        json.dump(comp_summary, open(out / f"compare_{ca}_{cb}_{tag}.json", "w"), indent=1)
        print(json.dumps(comp_summary, indent=1))
    print(json.dumps(pooled_rows, indent=1))


if __name__ == "__main__":
    main()
