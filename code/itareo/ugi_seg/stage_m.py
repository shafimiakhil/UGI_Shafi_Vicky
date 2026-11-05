"""Stage M, the map (CLAUDE.md 13.3): the out-of-fold ensemble of the 12 A1 30-epoch checkpoints on the mosaic they were
trained on (derived/pneo2022_registered/, v1). Inference is done by itareo.ugi_seg.oof (prob_oof.tif per run); this
module only reads rasters.

  --ensemble  runs_itareo/ugi/maps/ugi_a1_e30_prob_mean.tif (mean P of the three seeds of the fold that holds the pixel
              out; float32 with NBITS=16, COG, NaN where not evaluated) and ugi_a1_e30_vote.tif (uint8: 1 where at least
              two seeds exceed their own internal-validation threshold, 0 otherwise, 255 where not evaluated)
  --score     itareo.ugi_seg.score on the whole grid (prob = the mean, pred = the vote; type from the summed group
              probabilities of the three seeds): maps/ensemble_metrics.json, objects.csv, components.csv (unmatched
              predicted components), blocks_1km.csv
  --zooms     five 1.5 x 1.5 km windows on 500 m steps by fixed rules -> maps/zooms.csv
  --figures   JPGs in docs/evidence_phase2/ugi_seg/maps/ (no titles; axes in UTM 20S km; legends and colour bars only)
From work_ms/:  python -m itareo.ugi_seg.stage_m --ensemble --score --zooms --figures
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .oof import write_prob_cog
from .score import BLOCK, score_map

RUNS = Path("runs_itareo/ugi/ugi_a1")
MAPS = Path("runs_itareo/ugi/maps")
FIGS = Path("../docs/evidence_phase2/ugi_seg/maps")
PATHS = {"labels": "derived/ugi_labels", "polygons": "derived/ugi_labels/polygons.csv",
         "ndvi": "derived/features/neo2022_indices_grid1m.tif", "sid": "derived/pneo2022_registered/pneo2022_strip_id_grid1m.tif",
         "image": "derived/pneo2022_registered/pneo2022_toa_grid1m.tif"}
FOLDS = {"x_line": 388282.5, "y_line": 6525025.5}
PROB = MAPS / "ugi_a1_e30_prob_mean.tif"
VOTE = MAPS / "ugi_a1_e30_vote.tif"
SEEDS = (0, 1, 2)   # one seed in the maps of stage_m2.use_one_seed (CLAUDE.md 8.3)
PROB_TEXT = "mean P(UGI) of three seeds, out of fold"
VOTE_TEXT = "UGI vote: >= 2 of 3 seeds above their own threshold"
PRED_TEXT = "vote map (>= 2 of 3 seeds above their own threshold)"
CB_TEXT = "P(UGI), mean of three seeds, out of fold"
ZOOM_PX, STEP = 1500, 500
# colours: the validated categorical slots 1-3 (dataviz reference palette; all-pairs CVD check passed)
C_TP, C_FP, C_FN = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]
INK, INK2, SURFACE, NODATA_GREY = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def runs():
    out = []
    for f in (1, 2, 3, 4):
        for s in SEEDS:
            d = RUNS / f"fold{f}_seed{s}"
            out.append((f, s, d, json.load(open(d / "prob_oof.json"))))
    return out


def grid():
    with rasterio.open(PATHS["image"]) as s:
        return s.height, s.width, s.transform, s.crs


def ensemble():
    t0 = time.time()
    H, W, T, crs = grid()
    psum = np.zeros((H, W), np.float32)
    n = np.zeros((H, W), np.uint8)
    votes = np.zeros((H, W), np.uint8)
    info = []
    for f, s, d, meta in runs():
        with rasterio.open(d / "prob_oof.tif") as src:
            p = src.read(1)
            r0 = round((src.transform.f - T.f) / T.e)
            c0 = round((src.transform.c - T.c) / T.a)
        ok = np.isfinite(p)
        sl = (slice(r0, r0 + p.shape[0]), slice(c0, c0 + p.shape[1]))
        psum[sl] += np.where(ok, p, 0)
        n[sl] += ok
        votes[sl] += ok & (p >= meta["threshold"])
        info.append({"fold": f, "seed": s, "threshold": meta["threshold"], "evaluated_px": int(ok.sum())})
    k = len(SEEDS)
    assert set(np.unique(n)) <= {0, k}, f"every evaluated pixel must have exactly {k} seeds"
    mean = np.where(n == k, psum / np.maximum(n, 1), np.nan).astype(np.float32)
    vote = np.where(n == k, (votes >= k // 2 + 1).astype(np.uint8), 255).astype(np.uint8)
    MAPS.mkdir(parents=True, exist_ok=True)
    b1 = write_prob_cog(PROB, mean, T, crs, PROB_TEXT)
    prof = dict(driver="COG", width=W, height=H, count=1, dtype="uint8", nodata=255, crs=crs, transform=T, compress="DEFLATE",
                BLOCKSIZE=512, BIGTIFF="YES")
    with rasterio.open(VOTE, "w", **prof) as dst:
        dst.write(vote[None])
        dst.set_band_description(1, VOTE_TEXT)
    out = {"runs": info, "evaluated_px": int((n == k).sum()), "vote_ugi_px": int((vote == 1).sum()),
           "prob_mean_bytes": b1, "vote_bytes": VOTE.stat().st_size, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(MAPS / "ensemble_build.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}), flush=True)


def group_sums():
    """Per polygon the summed group probabilities of the three seeds (each seed over the same evaluated pixels)."""
    acc = {}
    for f, s, d, _ in runs():
        for r in csv.DictReader(open(d / "oof_groups.csv")):
            v = np.array([float(r[f"sum_p_group{g}"]) for g in range(1, 6)])
            acc[int(r["id"])] = acc.get(int(r["id"]), 0) + v
    return acc


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def score():
    t0 = time.time()
    with rasterio.open(PROB) as s:
        prob = s.read(1)
    with rasterio.open(VOTE) as s:
        pred = s.read(1) == 1
    res, objs, unmatched, blocks, boot = score_map(prob, pred, 0, 0, PATHS, group_sums())
    res.update({"maps": {"prob_mean": str(PROB), "vote": str(VOTE)}, "checkpoints": str(RUNS / "fold*_seed*/best.pt"),
                "mosaic": PATHS["image"], "pred": PRED_TEXT,
                "scores_from": "AUROC/AP from the mean probability; thresholded scores, objects and blocks from the vote",
                "seconds": round(time.time() - t0, 1)})
    json.dump(res, open(MAPS / "ensemble_metrics.json", "w"), indent=1)
    write_csv(MAPS / "objects.csv", objs)
    write_csv(MAPS / "components.csv", unmatched)
    write_csv(MAPS / "blocks_1km.csv", blocks)
    np.savez_compressed(MAPS / "ensemble_blocks_boot.npz", **boot)
    print(json.dumps({k: res[k] for k in ("pixel_full", "object_recall_strict", "object_recall_tolerant", "components",
                                          "blocks_1km", "seconds")}, indent=1), flush=True)


def check_run(run="fold1_seed0"):
    """Rule-1 agreement check: one run scored from its prob_oof.tif at its own threshold, beside its metrics.json
    -> maps/check_<run>_from_raster.json."""
    d = RUNS / run
    meta = json.load(open(d / "prob_oof.json"))
    own = json.load(open(d / "metrics.json"))
    with rasterio.open(d / "prob_oof.tif") as s:
        p = s.read(1)
    r0, c0 = meta["region_rows_cols"][:2]
    res, _, _, _, _ = score_map(p, p >= meta["threshold"], r0, c0, PATHS)
    out = {"run": str(d), "threshold": meta["threshold"],
           "from_raster": {"iou": res["pixel_full"]["iou"], "ap": res["pixel_full"]["ap"], "auroc": res["pixel_full"]["auroc"],
                           "boundary_f": res["boundary"]["f"], "object_recall_strict": res["object_recall_strict"],
                           "objects": res["objects_evaluated"]},
           "metrics_json": {"iou": own["pixel_full"]["iou"], "ap": own["pixel_full"]["ap"], "auroc": own["pixel_full"]["auroc"],
                            "boundary_f": own["boundary_f"], "object_recall_strict": own["object_recall"],
                            "objects": own["objects_evaluated"]}}
    json.dump(out, open(MAPS / f"check_{run}_from_raster.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


def rule7():
    """Rule 7 (CLAUDE.md 13.4) on the ensemble: itareo.ugi_seg.score7 on the whole grid (prob = the mean, pred = the vote).
    maps/ensemble_metrics_rule7.json, components_rule7.csv (every component >= 50 m2 with its strict / tolerant match and
    the polygons it covers), blocks_1km_errors.csv, ensemble_boot_rule7.npz; objects.csv gains the rule-7 columns (the
    previous objects.csv goes to the archive first)."""
    from .score7 import score7
    t0 = time.time()
    with rasterio.open(PROB) as s:
        prob = s.read(1)
    with rasterio.open(VOTE) as s:
        pred = s.read(1) == 1
    res, objs, comps, brows, boot = score7(prob, pred, 0, 0, PATHS, FOLDS)
    res.update({"maps": {"prob_mean": str(PROB), "vote": str(VOTE)}, "seconds": round(time.time() - t0, 1)})
    json.dump(res, open(MAPS / "ensemble_metrics_rule7.json", "w"), indent=1, default=float)
    write_csv(MAPS / "components_rule7.csv", comps)
    write_csv(MAPS / "blocks_1km_errors.csv", brows)
    np.savez_compressed(MAPS / "ensemble_boot_rule7.npz", **boot)
    objects_add_attributes()
    print(json.dumps({k: res[k] for k in ("objects", "object_precision", "calibration")}, indent=1, default=float)[:3000])


def objects_add_attributes():
    """objects.csv gains the rule-7 columns (type, source, core, CIRCUNVALACION, width, fold-line crossing) from
    derived/ugi_labels/polygon_attributes.csv; the previous objects.csv is archived first (no-clobber)."""
    from itareo.archive_nc import archive_nc
    old = MAPS / "objects.csv"
    attrs = {int(r["id"]): r for r in csv.DictReader(open("derived/ugi_labels/polygon_attributes.csv", encoding="utf-8"))}
    prev = list(csv.DictReader(open(old)))
    if "width_m" in prev[0]:
        print("objects.csv already has the rule-7 columns")
        return
    archive_nc("../archive/2026-11-03/stage_m_objects_before_rule7", old)
    add = ("Tipo_evp_2", "source", "core", "circunvalacion", "width_m", "crosses_fold_line")
    merged = [{**o, **{k: attrs[int(o["id"])][k] for k in add}} for o in prev]
    write_csv(old, merged)
    print("objects.csv:", len(merged), "rows, columns added:", add)


def zooms():
    """Windows of 1,500 m on 500 m steps inside the grid, chosen by fixed rules; ties go to the first window in row-major
    order; a window chosen by an earlier rule is not chosen again (the next best is taken)."""
    H, W, T, _ = grid()
    objs = list(csv.DictReader(open(MAPS / "objects.csv")))
    comps = list(csv.DictReader(open(MAPS / "components.csv")))
    pos_r = list(range(0, H - ZOOM_PX + 1, STEP))
    pos_c = list(range(0, W - ZOOM_PX + 1, STEP))
    wins = [(r, c) for r in pos_r for c in pos_c]

    def rc(o):
        return (T.f - float(o["y"])) / -T.e - 0.5, (float(o["x"]) - T.c) / T.a - 0.5

    def count_centroids(rows, cond):
        pts = [rc(o) for o in rows if cond(o)]
        return np.array([sum(r <= y < r + ZOOM_PX and c <= x < c + ZOOM_PX for y, x in pts) for r, c in wins], float)
    # pixel sums by 500 m cells, then 3 x 3 cells per window
    with rasterio.open(PATHS["labels"] + "/ugi_group_grid1m.tif") as s:
        grp = s.read(1)
    with rasterio.open(PATHS["labels"] + "/ugi_binary_grid1m.tif") as s:
        ugi = s.read(1)

    def cellsum(a):
        nr, nc = H // STEP, W // STEP
        return a[:nr * STEP, :nc * STEP].reshape(nr, STEP, nc, STEP).sum(axis=(1, 3))

    def winsum(cells):
        return np.array([cells[r // STEP:r // STEP + 3, c // STEP:c // STEP + 3].sum() for r, c in wins], float)
    water = winsum(cellsum((grp == 4).astype(np.int64)))
    share = winsum(cellsum(ugi.astype(np.int64))) / ZOOM_PX ** 2
    with rasterio.open(VOTE) as s:
        vote = s.read(1) == 1
    from scipy import ndimage
    comp, _ = ndimage.label(vote, structure=np.ones((3, 3), bool))
    ids = np.array([int(c["component"]) for c in comps], np.int64)
    look = np.zeros(comp.max() + 1, bool)
    look[ids] = True
    um = look[comp]
    del comp
    unm = winsum(cellsum(um.astype(np.int64)))
    crit = [("squares", "most Squares polygons (centroids)", count_centroids(objs, lambda o: o["group"] == "2")),
            ("road_greens", "most Road-greens polygons (centroids)", count_centroids(objs, lambda o: o["group"] == "1")),
            ("water_corridors", "most Water-corridors area (px)", water),
            ("unmatched", "largest unmatched predicted area (px)", unm)]
    chosen, rows = set(), []
    for name, rule, v in crit:
        for k in np.argsort(-v, kind="stable"):
            if wins[k] not in chosen:
                break
        chosen.add(wins[k])
        rows.append((name, rule, k, float(v[k])))
    rng = np.random.default_rng(0)
    elig = [k for k in range(len(wins)) if share[k] >= 0.03 and wins[k] not in chosen]
    k = int(rng.choice(elig))
    rows.append(("random", f"random (seed 0) among {len(elig)} windows with UGI share >= 3 % (not already chosen)", k,
                 float(share[k])))
    out = []
    for i, (name, rule, k, val) in enumerate(rows, 1):
        r, c = wins[k]
        out.append({"zoom": i, "name": name, "rule": rule, "criterion_value": val, "row0": r, "col0": c,
                    "x_min": T.c + c, "y_max": T.f - r, "x_max": T.c + c + ZOOM_PX, "y_min": T.f - r - ZOOM_PX,
                    "ugi_share": round(float(share[k]), 4)})
    write_csv(MAPS / "zooms.csv", out)
    print(json.dumps(out, indent=1))


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False})
    return plt


def _km_axes(ax, x0, x1, y0, y1):
    ax.set_xlabel("Easting, UTM 20S (km)")
    ax.set_ylabel("Northing, UTM 20S (km)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.1f}" if (x1 - x0) < 5000 else f"{v / 1000:.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.1f}" if (y1 - y0) < 5000 else f"{v / 1000:.0f}")


def _save_native(fig, ax, ncols, path, quality=92):
    """Save at the dpi that gives every data column of the image at least one output pixel (the axes keep their size in
    inches), so nearest-neighbour display drops no cell. Returns the dpi."""
    import math as _m
    fig.canvas.draw()
    w_in = ax.get_window_extent().width / fig.dpi
    dpi = max(int(_m.ceil(1.01 * ncols / w_in)), 150)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": quality})
    return dpi


def figures(parts=("city", "zooms", "scatter")):
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patheffects import withStroke
    FIGS.mkdir(parents=True, exist_ok=True)
    H, W, T, _ = grid()
    ext = (T.c, T.c + W * T.a, T.f + H * T.e, T.f)
    with rasterio.open(PROB) as s:
        prob = s.read(1)
    h4, w4 = H // 4, W // 4
    dpis = {}
    if "city" in parts:
        _city(plt, FIGS, H, W, T, ext, prob, h4, w4, dpis)
    if "zooms" in parts:
        _zooms(plt, dpis)
    if "scatter" in parts:
        _scatter(plt)
    old = json.load(open(MAPS / "figures_dpi.json")) if (MAPS / "figures_dpi.json").exists() else {"dpi": {}}
    old["dpi"].update(dpis)
    json.dump({"dpi": old["dpi"], "rule": "every data column of the image gets at least one output pixel"},
              open(MAPS / "figures_dpi.json", "w"), indent=1)
    print(dpis, sorted(str(p) for p in FIGS.glob("*.jpg")))


def _city(plt, FIGS, H, W, T, ext, prob, h4, w4, dpis):
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    # ---- city_prob.jpg: mean P at 4 m ---------------------------------------------------------------------------
    p4 = prob[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4)
    with np.errstate(invalid="ignore"):
        p4 = np.nanmean(p4, axis=(1, 3))
    cmap = LinearSegmentedColormap.from_list("blue_seq", [SURFACE] + SEQ[1:])     # P = 0 recedes to the surface
    cmap.set_bad(NODATA_GREY)
    fig, ax = plt.subplots(figsize=(8.2, 7.6), dpi=200)
    im = ax.imshow(p4, extent=(T.c, T.c + w4 * 4, T.f - h4 * 4, T.f), cmap=cmap, vmin=0, vmax=1, interpolation="none")
    ax.axvline(FOLDS["x_line"], color=INK, ls=(0, (6, 4)), lw=1.0)
    ax.axhline(FOLDS["y_line"], color=INK, ls=(0, (6, 4)), lw=1.0)
    _km_axes(ax, *ext)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(CB_TEXT)
    cb.outline.set_edgecolor(INK2)
    hs = [Line2D([], [], color=INK, ls=(0, (6, 4)), lw=1.0, label="Fold boundaries")]
    if np.isnan(prob).any():
        hs.append(Patch(color=NODATA_GREY, label="Not evaluated"))
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(hs), fontsize=8)
    dpis["city_prob.jpg"] = _save_native(fig, ax, w4, FIGS / "city_prob.jpg")
    plt.close(fig)
    del p4
    # ---- city_outcome.jpg ----------------------------------------------------------------------------------------
    objs = {int(o["id"]): o for o in csv.DictReader(open(MAPS / "objects.csv"))}
    with rasterio.open(PATHS["labels"] + "/ugi_id_grid1m.tif") as s:
        ids = s.read(1)
    found = np.zeros(ids.max() + 1, np.uint8)       # 1 found, 2 missed, 0 not evaluated / no polygon
    for pid, o in objs.items():
        found[pid] = 1 if o["found_tolerant"] == "True" else 2
    cat = found[ids]
    del ids
    with rasterio.open(VOTE) as s:
        vote = s.read(1)
    from scipy import ndimage
    comp, _ = ndimage.label(vote == 1, structure=np.ones((3, 3), bool))
    unm_ids = np.array([int(c["component"]) for c in csv.DictReader(open(MAPS / "components.csv"))], np.int64)
    look = np.zeros(comp.max() + 1, bool)
    look[unm_ids] = True
    cat[look[comp]] = 3
    del comp
    cat[(vote == 255) & (cat == 0)] = 4
    # 4 m display: a 4 x 4 cell shows unmatched > missed > found > not evaluated > background
    c4 = cat[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4)
    disp = np.zeros((h4, w4), np.uint8)
    for code in (4, 1, 2, 3):
        disp[(c4 == code).any(axis=(1, 3))] = code
    del c4, cat
    cmap2 = ListedColormap([SURFACE, C_TP, C_FN, C_FP, NODATA_GREY])
    fig, ax = plt.subplots(figsize=(8.2, 7.6), dpi=200)
    ax.imshow(disp, extent=(T.c, T.c + w4 * 4, T.f - h4 * 4, T.f), cmap=cmap2, vmin=-0.5, vmax=4.5, interpolation="none")
    _km_axes(ax, *ext)
    n_found = sum(o["found_tolerant"] == "True" for o in objs.values())
    hs = [Patch(color=C_TP, label=f"Inventory UGI found, tolerant rule ({n_found})"),
          Patch(color=C_FN, label=f"Inventory UGI missed ({len(objs) - n_found})"),
          Patch(color=C_FP, label=f"Predicted UGI outside the inventory ({len(unm_ids)} components >= 50 m²)")]
    if (vote == 255).any():
        hs.append(Patch(color=NODATA_GREY, label="Not evaluated"))
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7.5)
    dpis["city_outcome.jpg"] = _save_native(fig, ax, w4, FIGS / "city_outcome.jpg")
    plt.close(fig)
    del disp


def _zooms(plt, dpis):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.patheffects import withStroke
    from scipy import ndimage  # noqa: F401
    with rasterio.open(VOTE) as s:
        vote = s.read(1)
    # ---- zooms ---------------------------------------------------------------------------------------------------
    with rasterio.open(PATHS["labels"] + "/ugi_binary_grid1m.tif") as s:
        ref_all = s.read(1)
    for z in csv.DictReader(open(MAPS / "zooms.csv")):
        r, c = int(z["row0"]), int(z["col0"])
        w = Window(c, r, ZOOM_PX, ZOOM_PX)
        with rasterio.open(PATHS["image"]) as s:
            rgb = s.read([1, 2, 3], window=w).astype(np.float32)
            nod = (rgb == s.nodata).any(0)
        out = np.zeros((ZOOM_PX, ZOOM_PX, 3), np.float32)
        for k in range(3):
            v = rgb[k][~nod]
            lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
            out[..., k] = np.clip((rgb[k] - lo) / max(hi - lo, 1), 0, 1)
        out[nod] = 1.0
        vt = vote[r:r + ZOOM_PX, c:c + ZOOM_PX]
        ref = ref_all[r:r + ZOOM_PX, c:c + ZOOM_PX].astype(bool)
        pr = vt == 1
        ev = vt != 255
        alpha = 0.5
        for m, col in ((pr & ref, C_TP), (pr & ~ref & ev, C_FP), (~pr & ref & ev, C_FN)):
            rgbc = np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)
            out[m] = (1 - alpha) * out[m] + alpha * rgbc
        with rasterio.open(PATHS["labels"] + "/ugi_id_grid1m.tif") as s:
            pid = s.read(1, window=w)
        outline = np.zeros(pid.shape, bool)            # pixels whose left or upper neighbour has another polygon id
        outline[:, 1:] |= pid[:, 1:] != pid[:, :-1]
        outline[1:, :] |= pid[1:, :] != pid[:-1, :]
        out[outline] = np.array([1.0, 1.0, 1.0], np.float32)
        x0, y1 = float(z["x_min"]), float(z["y_max"])
        fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=220)
        ax.imshow(out, extent=(x0, x0 + ZOOM_PX, y1 - ZOOM_PX, y1), interpolation="none")
        _km_axes(ax, x0, x0 + ZOOM_PX, y1 - ZOOM_PX, y1)
        ax.legend(handles=[Patch(color=C_TP, alpha=0.8, label="In the inventory and mapped"),
                           Patch(color=C_FP, alpha=0.8, label="Mapped outside the inventory"),
                           Patch(color=C_FN, alpha=0.8, label="In the inventory, not mapped"),
                           Line2D([], [], color="white", lw=1.5, label="Inventory outline",
                                  path_effects=[withStroke(linewidth=3.5, foreground=INK)])],
                  loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
        dpis[f"zoom{z['zoom']}_{z['name']}.jpg"] = _save_native(fig, ax, ZOOM_PX, FIGS / f"zoom{z['zoom']}_{z['name']}.jpg")
        plt.close(fig)


def _scatter(plt):
    from matplotlib.lines import Line2D  # noqa: F401
    # ---- blocks_scatter.jpg --------------------------------------------------------------------------------------
    b = [x for x in csv.DictReader(open(MAPS / "blocks_1km.csv")) if int(x["valid"]) >= 500_000]
    inv = np.array([float(x["inventory_share"]) for x in b]) * 100
    mp = np.array([float(x["mapped_share"]) for x in b]) * 100
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    hi = max(inv.max(), mp.max()) * 1.05
    ax.plot([0, hi], [0, hi], color=INK2, lw=1.2, ls=(0, (4, 3)), label="1:1", zorder=1)
    ax.scatter(inv, mp, s=22, color=C_TP, edgecolor=SURFACE, linewidth=0.8, label=f"1 km blocks ({len(b)})", zorder=2)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Inventory UGI share of the block (%)")
    ax.set_ylabel("Mapped UGI share of the block (%)")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIGS / "blocks_scatter.jpg", dpi=200, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("ensemble", "score", "zooms", "figures", "figures_zooms", "check_run", "rule7", "objects_attrs"):
        ap.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    if a.ensemble:
        ensemble()
    if a.score:
        score()
    if a.zooms:
        zooms()
    if a.figures:
        figures()
    if a.figures_zooms:
        figures(parts=("zooms",))
    if a.check_run:
        check_run()
    if a.rule7:
        rule7()
    if a.objects_attrs:
        objects_add_attributes()
