"""Stage M2, the converged map (CLAUDE.md 13.6): the out-of-fold ensemble of ugi_r1_none, seeds 0-2, from its 12 fold*_seed*/prob_oof.tif,
built exactly as Stage M by the functions of itareo.ugi_seg.stage_m (same mean, vote and threshold rules), on the v3 mosaic. CPU only, no
inference. Scored with rules 4 and 7 on the full grid (secondary) and inside the ejido (rule 9, primary, itareo.ugi_seg.rescore9).

  --ensemble      runs_itareo/ugi/maps_m2/ugi_r1_none_prob_mean.tif, ugi_r1_none_vote.tif, ensemble_build.json; zooms.csv, the five windows of Stage M
  --score         ensemble_metrics.json, objects.csv, components.csv, blocks_1km.csv, ensemble_blocks_boot.npz and the rule-7 files (full grid)
  --ejido         the same scores inside the ejido, in maps_m2/ejido/
  --qgis          maps_m2/qgis/: layers as in maps/qgis/, with one more code in the comparison raster, 4 = mapped outside the ejido (no reference)
  --figures       docs/evidence_phase2/ugi_seg/maps_m2/: the figures of Stage M, same code, same windows; ejido/city_outcome.jpg and
                  ejido/blocks_scatter.jpg are the outcome and the blocks inside the ejido
  --before-after  before_after.csv: Stage M against M2 inside the ejido, with the protocol differences read from the two configs; paired block
                  bootstrap (rule 2) on the difference where the block files allow it
  --readme        maps_m2/README.md
Every step refuses to overwrite. From work_ms/:  bash jobs/stage_m2.sh
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.windows import Window

from itareo.safe_write import refuse_existing

from . import rescore9, stage_m
from . import summarize_r as sr

M1 = Path("runs_itareo/ugi/maps")
MAPS = Path("runs_itareo/ugi/maps_m2")
FIGS = Path("../docs/evidence_phase2/ugi_seg/maps_m2")
RUNS = Path("runs_itareo/ugi/ugi_r1_none")
PROB = MAPS / "ugi_r1_none_prob_mean.tif"
VOTE = MAPS / "ugi_r1_none_vote.tif"
EJIDO = rescore9.EJIDO
PATHS = {"labels": "derived/ugi_labels", "polygons": "derived/ugi_labels/polygons.csv", "ndvi": "derived/features/neo2022_v3_indices_grid1m.tif",
         "sid": "derived/pneo2022_registered_v3/pneo2022_strip_id_grid1m.tif", "image": "derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif"}
GROUPS = ["Road greens", "Squares", "Parks and green areas", "Water corridors", "Rail verges", "NO DEFINIDO"]
STEM, PROB_STEM = "ugi_m2_r1_none", "ugi_r1_none"
VALUES_TEXT = "1 = mapped as UGI (vote of three seeds), 0 = nodata"
SOURCE_TEXT = "Stage M2 ensemble, ugi_r1_none, 60 epochs, v3 mosaic, three-seed out-of-fold vote; inventory 22174_UrbanGreenInfras.gpkg rasterised on GRID_1M"
QGIS_HEAD = "# QGIS layers of the converged map, Stage M2 (CLAUDE.md 13.6)\n\nAs `../../maps/qgis/`, from `python -m itareo.ugi_seg.stage_m2 --qgis`."
BLUE, ORANGE, GREEN, GREY = (43, 108, 212, 255), (232, 116, 59, 255), (25, 169, 121, 255), (150, 150, 150, 255)


def use_one_seed(which):
    """One seed instead of the three-seed ensemble, built and scored exactly as M2 (CLAUDE.md 8.3): a1 = ugi_r1_none, a2 = ugi_r3_a2_none, seed 0. The probability is the run's, the map its threshold."""
    global RUNS, MAPS, FIGS, PROB, VOTE, STEM, PROB_STEM, VALUES_TEXT, SOURCE_TEXT, QGIS_HEAD
    arm, folder, STEM = {"a1": ("ugi_r1_none", "maps_a1_seed0", "ugi_a1_seed0"), "a2": ("ugi_r3_a2_none", "maps_a2", "ugi_a2_seed0")}[which]
    RUNS, MAPS, FIGS = Path(f"runs_itareo/ugi/{arm}"), Path(f"runs_itareo/ugi/{folder}"), Path(f"../docs/evidence_phase2/ugi_seg/{folder}")
    PROB_STEM = f"{arm}_seed0"
    PROB, VOTE = MAPS / f"{PROB_STEM}_prob.tif", MAPS / f"{PROB_STEM}_map.tif"
    stage_m.SEEDS = (0,)
    stage_m.PROB_TEXT = f"P(UGI) of {arm}, seed 0, out of fold"
    stage_m.VOTE_TEXT = "UGI: seed 0 above its own threshold"
    stage_m.PRED_TEXT = "seed 0 above its own internal-validation threshold, per fold"
    stage_m.CB_TEXT = "P(UGI), seed 0, out of fold"
    VALUES_TEXT = "1 = mapped as UGI (seed 0 above its own threshold), 0 = nodata"
    SOURCE_TEXT = f"{arm}, seed 0, 60 epochs, v3 mosaic, out-of-fold map; inventory 22174_UrbanGreenInfras.gpkg rasterised on GRID_1M"
    QGIS_HEAD = f"# QGIS layers of {arm}, seed 0 (CLAUDE.md 8.3)\n\nBuilt as the layers of `../../maps_m2/qgis/`, with one seed instead of the three-seed vote."


def use_m2():
    """stage_m's functions read these module constants when they run."""
    stage_m.RUNS, stage_m.MAPS, stage_m.FIGS, stage_m.PATHS, stage_m.PROB, stage_m.VOTE = RUNS, MAPS, FIGS, PATHS, PROB, VOTE
    stage_m.objects_add_attributes = objects_add_attributes


def objects_add_attributes():
    """objects.csv, written moments before by --score, gains the rule-7 columns, as Stage M's did."""
    attrs = {int(r["id"]): r for r in csv.DictReader(open("derived/ugi_labels/polygon_attributes.csv", encoding="utf-8"))}
    prev = list(csv.DictReader(open(MAPS / "objects.csv")))
    add = ("Tipo_evp_2", "source", "core", "circunvalacion", "width_m", "crosses_fold_line")
    stage_m.write_csv(MAPS / "objects.csv", [{**o, **{k: attrs[int(o["id"])][k] for k in add}} for o in prev])


def do_ensemble():
    refuse_existing(MAPS)
    MAPS.mkdir(parents=True)
    shutil.copy2(M1 / "zooms.csv", MAPS / "zooms.csv")
    stage_m.ensemble()


def do_score():
    refuse_existing(MAPS / "ensemble_metrics.json", MAPS / "ensemble_metrics_rule7.json")
    stage_m.score()
    stage_m.rule7()


def do_ejido():
    rescore9.rescore_stage_m()


def do_qgis():
    q = MAPS / "qgis"
    refuse_existing(q)
    q.mkdir()
    counts = {"tp": 0, "fp": 0, "fn": 0, "mapped_outside_ejido": 0, "evaluated_inside": 0}
    with rasterio.open(VOTE) as v, rasterio.open(PATHS["labels"] + "/ugi_binary_grid1m.tif") as g, rasterio.open(EJIDO) as e:
        assert v.crs == g.crs and v.transform == g.transform and v.shape == g.shape == e.shape, "grids differ"
        prof = {"driver": "GTiff", "dtype": "uint8", "count": 1, "width": v.width, "height": v.height, "crs": v.crs, "transform": v.transform, "nodata": 0, "tiled": True,
                "blockxsize": 512, "blockysize": 512, "compress": "deflate", "zlevel": 6, "BIGTIFF": "IF_SAFER"}
        p_pred, p_cmp = q / f"{STEM}_predicted.tif", q / f"{STEM}_vs_inventory.tif"
        with rasterio.open(p_pred, "w", **prof) as dp, rasterio.open(p_cmp, "w", **prof) as dc:
            for r0 in range(0, v.height, 2048):
                w = Window(0, r0, v.width, min(2048, v.height - r0))
                vv, gg, ee = v.read(1, window=w), g.read(1, window=w), e.read(1, window=w) == 1
                ev = vv != 255
                pr = (vv == 1) & ev
                inv = (gg == 1) & ev
                cmp_ = np.zeros(vv.shape, np.uint8)
                cmp_[pr & inv & ee] = 1
                cmp_[pr & ~inv & ee] = 2
                cmp_[~pr & inv & ee] = 3
                cmp_[pr & ~ee] = 4
                dp.write(pr.astype(np.uint8), 1, window=w)
                dc.write(cmp_, 1, window=w)
                counts["tp"] += int((cmp_ == 1).sum())
                counts["fp"] += int((cmp_ == 2).sum())
                counts["fn"] += int((cmp_ == 3).sum())
                counts["mapped_outside_ejido"] += int((cmp_ == 4).sum())
                counts["evaluated_inside"] += int((ev & ee).sum())
            dp.write_colormap(1, {0: (0, 0, 0, 0), 1: BLUE})
            dc.write_colormap(1, {0: (0, 0, 0, 0), 1: BLUE, 2: ORANGE, 3: GREEN, 4: GREY})
            dp.update_tags(1, VALUES=VALUES_TEXT)
            dc.update_tags(1, VALUES="inside the ejido: 1 = in the inventory and mapped; 2 = mapped outside the inventory; 3 = in the inventory, not mapped; "
                                     "4 = mapped outside the ejido (no reference); 0 = nodata")
            for d in (dp, dc):
                d.update_tags(SOURCE=SOURCE_TEXT)
        for p in (p_pred, p_cmp):
            with rasterio.open(p, "r+") as d:
                d.build_overviews([2, 4, 8, 16, 32, 64], Resampling.nearest)
                d.update_tags(ns="rio_overview", resampling="nearest")
    s = (M1 / "qgis/ugi_baseline_a1_e30_predicted.qml").read_text(encoding="utf-8")
    (q / f"{STEM}_predicted.qml").write_text(s, encoding="utf-8")
    s = (M1 / "qgis/ugi_baseline_a1_e30_vs_inventory.qml").read_text(encoding="utf-8")
    key = "</colorPalette>"
    assert s.count(key) == 1
    (q / f"{STEM}_vs_inventory.qml").write_text(s.replace(key, '  <paletteEntry value="4" alpha="255" color="#969696" label="Mapped outside the ejido (no reference)"/>\n      ' + key),
                                                       encoding="utf-8")
    shutil.copy2(M1 / "qgis/ugi_a1_e30_prob_mean_style.qml", q / f"{PROB_STEM}_prob_style.qml" if STEM != "ugi_m2_r1_none" else q / "ugi_r1_none_prob_mean_style.qml")
    json.dump(counts, open(q / "counts.json", "w"), indent=1)
    (q / "README.md").write_text(
        QGIS_HEAD + " Pixel counts in `counts.json`; inside the ejido they equal `../ejido/metrics.json`.\n\n"
        "| file | what it holds |\n|---|---|\n"
        f"| `{STEM}_vs_inventory.tif` | uint8. Inside the ejido: 1 = in the inventory and mapped; 2 = mapped outside the inventory; 3 = in the inventory, not mapped. 4 = mapped outside the ejido, where there is no reference. 0 = nodata (transparent) |\n"
        f"| `{STEM}_predicted.tif` | uint8. 1 = mapped as UGI, 0 = nodata (transparent) |\n"
        f"| `{STEM}_vs_inventory.qml`, `{STEM}_predicted.qml` | styles; QGIS applies them on loading because they share the raster's name and folder |\n"
        f"| `{Path(PROB).stem}_style.qml` | style for `../{Path(PROB).name}` (load by hand) |\n\n"
        "Set-up: `derived/pneo2022_registered_v3/pneo2022_toa_grid1m.tif` (red = band 1, green = band 2, blue = band 3, cumulative 2-98 % stretch); the comparison raster above it; "
        "`raw_data/22174_UrbanGreenInfras.gpkg` and `raw_data/ejido-municipal_iAEC1kw/EjidoMunicipal.shp` on top, no fill, thin outlines. "
        "Rasters on GRID_1M, tiled, DEFLATE, embedded colour table, internal overviews (nearest).\n", encoding="utf-8")
    print("qgis", json.dumps(counts), flush=True)


def city_outcome_inside(plt, out):
    """city_outcome of Stage M restricted to the reference area: outcome from ejido/objects.csv and ejido/components_unmatched.csv."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy import ndimage
    H, W, T, _ = stage_m.grid()
    h4, w4 = H // 4, W // 4
    objs = {int(o["id"]): o for o in csv.DictReader(open(MAPS / "ejido/objects.csv"))}
    with rasterio.open(PATHS["labels"] + "/ugi_id_grid1m.tif") as s:
        ids = s.read(1)
    found = np.zeros(ids.max() + 1, np.uint8)
    for pid, o in objs.items():
        found[pid] = 1 if o["found_tolerant"] == "True" else 2
    cat = found[ids]
    del ids
    with rasterio.open(VOTE) as s:
        vote = s.read(1)
    with rasterio.open(EJIDO) as s:
        inside = s.read(1) == 1
    comp, _ = ndimage.label((vote == 1) & inside, structure=np.ones((3, 3), bool))
    unm = np.array([int(c["component"]) for c in csv.DictReader(open(MAPS / "ejido/components_unmatched.csv"))], np.int64)
    look = np.zeros(comp.max() + 1, bool)
    look[unm] = True
    cat[look[comp]] = 3
    del comp
    cat[~inside] = 4
    c4 = cat[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4)
    disp = np.zeros((h4, w4), np.uint8)
    for code in (4, 1, 2, 3):
        disp[(c4 == code).any(axis=(1, 3))] = code
    del c4, cat
    fig, ax = plt.subplots(figsize=(8.2, 7.6), dpi=200)
    ax.imshow(disp, extent=(T.c, T.c + w4 * 4, T.f - h4 * 4, T.f), cmap=ListedColormap([stage_m.SURFACE, stage_m.C_TP, stage_m.C_FN, stage_m.C_FP, stage_m.NODATA_GREY]),
              vmin=-0.5, vmax=4.5, interpolation="none")
    stage_m._km_axes(ax, T.c, T.c + W * T.a, T.f + H * T.e, T.f)
    n_found = sum(o["found_tolerant"] == "True" for o in objs.values())
    hs = [Patch(color=stage_m.C_TP, label=f"Inventory UGI found, tolerant rule ({n_found})"), Patch(color=stage_m.C_FN, label=f"Inventory UGI missed ({len(objs) - n_found})"),
          Patch(color=stage_m.C_FP, label=f"Predicted UGI outside the inventory ({len(unm)} components >= 50 m²)"), Patch(color=stage_m.NODATA_GREY, label="Outside the ejido (no reference)")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7.5)
    dpi = stage_m._save_native(fig, ax, w4, out)
    plt.close(fig)
    return dpi


def do_figures():
    refuse_existing(FIGS)
    FIGS.mkdir(parents=True)
    (FIGS / "ejido").mkdir()
    stage_m.figures()
    plt = stage_m._style()
    dpi = city_outcome_inside(plt, FIGS / "ejido/city_outcome.jpg")
    stage_m.FIGS, stage_m.MAPS = FIGS / "ejido", MAPS / "ejido"
    stage_m._scatter(plt)
    stage_m.FIGS, stage_m.MAPS = FIGS, MAPS
    json.dump({"ejido/city_outcome.jpg": dpi}, open(MAPS / "figures_dpi_ejido.json", "w"), indent=1)
    print(sorted(str(p) for p in FIGS.rglob("*.jpg")), flush=True)


def _load(folder):
    return {"boot": dict(np.load(folder / "boot.npz")), "objects": list(csv.DictReader(open(folder / "objects.csv"))),
            "boot7": dict(np.load(folder / "boot_rule7.npz")), "objects7": list(csv.DictReader(open(folder / "objects_rule7.csv")))}


def _paired(a, b):
    """M2 minus Stage M, paired block bootstrap: the resampling of itareo.ugi_seg.summarize_r."""
    pa, pb = sr.pooled_arm_seed([a]), sr.pooled_arm_seed([b])
    nb = pa[1]["blk"]["valid"].shape[0]
    used = np.nonzero(pa[1]["blk"]["valid"] > 0)[0]
    rng = np.random.default_rng(0)
    W_ = np.zeros((sr.NBOOT, nb))
    for i in range(sr.NBOOT):
        W_[i] = np.bincount(rng.choice(used, len(used), replace=True), minlength=nb)

    def values(p):
        v = sr.boot_values(p[1], W_)
        for k in ("ap", "ap_vegetated", "ap_nocirc"):
            v[k] = v[k] + p[0][f"{k}_coarse_offset"]
        return v
    va, vb = values(pa), values(pb)
    out = {}
    for k in va:
        lo, hi = np.nanpercentile(vb[k] - va[k], [2.5, 97.5])
        out[k] = (pb[0][k] - pa[0][k], float(lo), float(hi))
    return out, len(used)


def do_before_after():
    out = MAPS / "before_after.csv"
    refuse_existing(out, FIGS / "before_after.csv")
    a1, r1 = (yaml.safe_load(open(f"itareo/configs/{n}.yaml", encoding="utf-8")) for n in ("ugi_a1", "ugi_r1_none"))

    def flat(d, pre=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
        return o
    fa, fr = flat(a1), flat(r1)
    rows = []
    for k in sorted(set(fa) | set(fr)):
        if k != "name" and fa.get(k) != fr.get(k):
            rows.append({"kind": "protocol", "item": f"config {k}", "stage_m": fa.get(k, "not in the config (code default: both)" if k == "train.edge_band" else "not in the config"), "m2": fr.get(k), "difference": "", "difference_ci95_block": ""})
    sa, sb = (json.load(open(f"runs_itareo/ugi/{n}/fold1_seed0/split.json")) for n in ("ugi_a1", "ugi_r1_none"))
    rows.append({"kind": "protocol", "item": "internal validation pixels (split.json; CLAUDE.md 13.3 F3)", "stage_m": sa.get("validation_pixels", "without the edge band"), "m2": sb.get("validation_pixels"), "difference": "", "difference_ci95_block": ""})
    ba, bb = (json.load(open(p / "ensemble_build.json")) for p in (M1, MAPS))
    th = lambda b: f"{min(r['threshold'] for r in b['runs']):.3f} to {max(r['threshold'] for r in b['runs']):.3f}"
    rows.append({"kind": "protocol", "item": "per-seed thresholds, internal validation (ensemble_build.json)", "stage_m": th(ba), "m2": th(bb), "difference": "", "difference_ci95_block": ""})
    nblocks = _score_rows(rows, M1 / "ejido", MAPS / "ejido", "stage_m", "m2")
    stage_m.write_csv(out, rows)
    shutil.copy2(out, FIGS / "before_after.csv")
    print("before_after rows", len(rows), "blocks resampled", nblocks, flush=True)


def _score_rows(rows, A, B, left, right, extra=None):
    """The score rows of the table: B minus A, from two scored folders (metrics.json, metrics_rule7.json, the bootstrap files)."""
    extra = extra or {}
    ma, mb, m7a, m7b = (json.load(open(p)) for p in (A / "metrics.json", B / "metrics.json", A / "metrics_rule7.json", B / "metrics_rule7.json"))
    boot, nblocks = _paired(_load(A), _load(B))

    def add(item, x, y, bkey=None):
        d, ci = "", ""
        if x is not None and y is not None:
            d = round(y - x, 5)
        if bkey:
            d, lo, hi = boot[bkey]
            d, ci = round(d, 5), f"[{lo:+.4f}, {hi:+.4f}]"
        rows.append({**extra, "kind": "score", "item": item, left: x, right: y, "difference": d, "difference_ci95_block": ci})
    pf, pg = ma["pixel_full"], mb["pixel_full"]
    add("AUROC", pf["auroc"], pg["auroc"])
    add("AP", pf["ap"], pg["ap"], "ap")
    add("IoU", pf["iou"], pg["iou"], "iou")
    add("F1", pf["f1"], pg["f1"])
    add("Precision", pf["precision"], pg["precision"])
    add("Recall", pf["recall"], pg["recall"])
    add("IoU, without the 1 m edge band", ma["pixel_no_edge"]["iou"], mb["pixel_no_edge"]["iou"])
    nca, ncb = m7a["pixel"]["no_circunvalacion"], m7b["pixel"]["no_circunvalacion"]
    add("AUROC, CIRCUNVALACION excluded", nca["auroc"], ncb["auroc"])
    add("AP, CIRCUNVALACION excluded", nca["ap"], ncb["ap"], "ap_nocirc")
    add("IoU, CIRCUNVALACION excluded", nca["iou"], ncb["iou"], "iou_nocirc")
    for name, key in (("vegetated pixels", "veg"), ("other pixels", "other")):
        x, y = ma["rule4"]["all"][key], mb["rule4"]["all"][key]
        add(f"AUROC, {name}", x["auroc"], y["auroc"])
        add(f"AP, {name}", x["ap"], y["ap"], "ap_vegetated" if key == "veg" else None)
        add(f"NDVI alone, AUROC, {name}", x["ndvi_auroc"], y["ndvi_auroc"])
        add(f"NDVI alone, AP, {name}", x["ndvi_ap"], y["ndvi_ap"])
    add("Object recall, strict", ma["object_recall_strict"], mb["object_recall_strict"], "object_recall_strict")
    add("Object recall, tolerant", ma["object_recall_tolerant"], mb["object_recall_tolerant"], "object_recall_tolerant")
    oa, ob = m7a["objects"], m7b["objects"]
    add("Object recall, tolerant, CIRCUNVALACION excluded", oa["no_circunvalacion"]["recall_tolerant"], ob["no_circunvalacion"]["recall_tolerant"], "object_recall_tolerant_nocirc")
    add("Object recall, tolerant, polygons under 6 m wide", oa["under_6m_wide"]["recall_tolerant"], ob["under_6m_wide"]["recall_tolerant"], "recall_under_6m_tolerant")
    add("Area-weighted coverage, all polygons", oa["all"]["area_weighted_coverage"], ob["all"]["area_weighted_coverage"])
    for g in GROUPS:
        add(f"Object recall, tolerant, {g}", oa["by_group"][g]["recall_tolerant"], ob["by_group"][g]["recall_tolerant"])
    for g in ("Parks and green areas", "Water corridors"):
        add(f"Area-weighted coverage, {g}", oa["by_group"][g]["area_weighted_coverage"], ob["by_group"][g]["area_weighted_coverage"])
    for s in oa["by_size"]:
        add(f"Object recall, tolerant, polygons {s} m2", oa["by_size"][s]["recall_tolerant"], ob["by_size"][s]["recall_tolerant"])
    for s in oa["by_width"]:
        add(f"Object recall, tolerant, polygons {s} m wide", oa["by_width"][s]["recall_tolerant"], ob["by_width"][s]["recall_tolerant"])
    add("Object precision (rule 4: tolerant, by count)", ma["components"]["object_precision"], mb["components"]["object_precision"])
    for kind in ("strict", "tolerant"):
        for by in ("by_count", "by_area"):
            add(f"Object precision, {kind}, {by.replace('_', ' ')}", m7a["object_precision"][kind][by], m7b["object_precision"][kind][by])
    add("Predicted components, >= 50 m2", ma["components"]["components"], mb["components"]["components"])
    add("Boundary F, 2 m", ma["boundary"]["f"], mb["boundary"]["f"])
    for k, name in (("pearson_r", "Pearson r"), ("slope", "slope"), ("bias_pp", "bias, pp"), ("mae_pp", "MAE, pp")):
        add(f"1 km blocks, {name}", ma["blocks_1km"][k], mb["blocks_1km"][k])
    add("Expected calibration error, mean P", m7a["calibration"]["ece"], m7b["calibration"]["ece"])
    add("One-to-one object F1 (supplementary)", m7a["one_to_one"]["f1"], m7b["one_to_one"]["f1"])
    if "pixels_evaluated_inside_ejido" in ma:
        add("Pixels evaluated inside the ejido", ma["pixels_evaluated_inside_ejido"], mb["pixels_evaluated_inside_ejido"])
    return nblocks


def do_centre():
    """Scores inside sq8km with this module's constants (rescore_centre.rescore_m2 would import a second copy of this module, with M2's)."""
    from .rescore_centre import REG, box
    from .rescore9 import score_inside
    from .score import read_win
    name = "sq8km"
    bx = box(name)
    r0, c0, h, w = bx["row0"], bx["col0"], bx["rows"], bx["cols"]
    win = Window(c0, r0, w, h)
    with rasterio.open(PROB) as s:
        p = s.read(1, window=win)
    with rasterio.open(VOTE) as s:
        pred = s.read(1, window=win) == 1
    p[read_win(REG / f"{name}_grid1m.tif", r0, c0, h, w)[0] != 1] = np.nan
    score_inside(p, pred, r0, c0, PATHS, stage_m.FOLDS, stage_m.group_sums(), MAPS / f"centre_{name}",
                 {"reference_area": f"inside the centre {name}", "centre_region": name, "maps": {"prob": str(PROB), "map": str(VOTE)}, "pred": stage_m.PRED_TEXT})


def do_pair():
    """One CSV: A2 against A1, both seed 0, built as maps, inside the ejido and sq8km. The self-test of this path first: each map's scores must equal those of its four runs."""
    a1, a2 = Path("runs_itareo/ugi/maps_a1_seed0"), Path("runs_itareo/ugi/maps_a2")
    out = Path("../docs/evidence_phase2/ugi_seg/maps_a2/pair_a2_vs_a1_seed0.csv")
    refuse_existing(out)
    arms = {"a1": ("ugi_r1_none", a1), "a2": ("ugi_r3_a2_none", a2)}
    for region, sub in (("ejido", "ejido"), ("sq8km", "centre_sq8km")):
        s = json.load(open(f"runs_itareo/ugi/summary_r/summary_{'r3_ejido' if region == 'ejido' else 'r3_centre_sq8km'}.json"))
        for _, (arm, folder) in arms.items():
            m, m7 = json.load(open(folder / sub / "metrics.json")), json.load(open(folder / sub / "metrics_rule7.json"))
            got = {"ap": m["pixel_full"]["ap"], "iou": m["pixel_full"]["iou"], "ap_nocirc": m7["pixel"]["no_circunvalacion"]["ap"]}
            for k, v in got.items():
                want = s["arms"][arm][k]
                assert abs(v - want) < 6e-4, f"self-test: {region} {arm} {k} map {v} runs {want}"
            # A polygon that crosses a fold line is scored whole on a map and by its part in one fold in a run; every other polygon must get the same result.
            runs = {}
            for f in (1, 2, 3, 4):
                for o in csv.DictReader(open(f"runs_itareo/ugi/{arm}/fold{f}_seed0/{sub}/objects_rule7.csv", encoding="utf-8")):
                    runs[o["id"]] = o
            objs = {o["id"]: o for o in csv.DictReader(open(folder / sub / "objects_rule7.csv", encoding="utf-8"))}
            assert set(runs) == set(objs), f"self-test: {region} {arm} polygons differ"
            differ = [k for k in objs if objs[k]["found_tolerant"] != runs[k]["found_tolerant"]]
            assert all(objs[k]["crosses_fold_line"] == "True" for k in differ), f"self-test: {region} {arm} polygons not on a fold line differ: {differ}"
            print("self-test", region, arm, {k: round(v, 4) for k, v in got.items()}, "polygons", len(objs), "differing, all on a fold line", differ, flush=True)
    rows = []
    for region, sub in (("ejido", "ejido"), ("sq8km", "centre_sq8km")):
        n = _score_rows(rows, a1 / sub, a2 / sub, "a1_seed0", "a2_seed0", {"region": region})
        print(region, "blocks resampled", n, flush=True)
    stage_m.write_csv(out, rows)
    print("pair rows", len(rows), str(out), flush=True)


def do_readme():
    refuse_existing(MAPS / "README.md")
    b = json.load(open(MAPS / "ensemble_build.json"))
    m, m7 = json.load(open(MAPS / "ejido/metrics.json")), json.load(open(MAPS / "ejido/metrics_rule7.json"))
    f = json.load(open(MAPS / "ensemble_metrics.json"))
    t = lambda x: json.dumps(x)
    (MAPS / "README.md").write_text(
        "# Stage M2, the converged map (CLAUDE.md 13.6)\n\n"
        "**Code** (from `work_ms/`): `bash jobs/stage_m2.sh`, which runs `itareo/ugi_seg/stage_m2.py` on the functions of `stage_m.py` and `rescore9.py`. CPU only, no inference.\n\n"
        "## Inputs\n\n"
        "- **Runs:** `runs_itareo/ugi/ugi_r1_none/fold{1-4}_seed{0-2}/prob_oof.tif`, with each run's internal-validation threshold from `prob_oof.json`.\n"
        f"- **Image, indices, strips:** `{PATHS['image']}`, `{PATHS['ndvi']}`, `{PATHS['sid']}` (v3).\n"
        "- **Labels:** `derived/ugi_labels/`. **Reference area:** `derived/ugi_labels/ejido_grid1m.tif`.\n\n"
        "## Build (`ensemble_build.json`)\n\n"
        f"The rules of Stage M: the mean P of the three seeds of the fold that holds each pixel out; a pixel is voted UGI when at least two seeds exceed their own threshold. {b['evaluated_px']:,} pixels evaluated, {b['vote_ugi_px']:,} voted UGI.\n\n"
        "## Scores\n\n"
        "- **Inside the ejido, primary:** `ejido/metrics.json`, `ejido/metrics_rule7.json` and the object, component and block files beside them.\n"
        "- **Full grid, secondary:** `ensemble_metrics.json`, `ensemble_metrics_rule7.json`, `objects.csv`, `components.csv`, `blocks_1km.csv`.\n"
        "- **Against Stage M:** `before_after.csv`, with the protocol differences first.\n\n"
        f"Inside the ejido: pixel scores {t(m['pixel_full'])}; AP and IoU with CIRCUNVALACION excluded {m7['pixel']['no_circunvalacion']['ap']}, {m7['pixel']['no_circunvalacion']['iou']}.\n"
        f"Full grid: pixel scores {t(f['pixel_full'])}.\n\n"
        "## Figures (`docs/evidence_phase2/ugi_seg/maps_m2/`)\n\n"
        "`city_prob.jpg`, `city_outcome.jpg`, `zoom1_squares.jpg` to `zoom5_random.jpg`, `blocks_scatter.jpg`: the code and the five windows of Stage M (`zooms.csv` is a copy of `../maps/zooms.csv`; all five lie wholly inside the ejido). "
        "`ejido/city_outcome.jpg`, `ejido/blocks_scatter.jpg`: the outcome and the blocks inside the ejido, the reference area. QGIS layers: `qgis/`.\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("ensemble", "score", "ejido", "centre", "qgis", "figures", "before-after", "readme", "pair"):
        ap.add_argument(f"--{k}", action="store_true")
    ap.add_argument("--one-seed", choices=["a1", "a2"], default=None, help="the map of one seed instead of the M2 ensemble (CLAUDE.md 8.3)")
    a = ap.parse_args()
    if a.one_seed:
        use_one_seed(a.one_seed)
    use_m2()
    t0 = time.time()
    for flag, fn in ((a.ensemble, do_ensemble), (a.score, do_score), (a.ejido, do_ejido), (a.centre, do_centre), (a.qgis, do_qgis), (a.pair, do_pair), (a.figures, do_figures), (a.before_after, do_before_after), (a.readme, do_readme)):
        if flag:
            fn()
    print("seconds", round(time.time() - t0), flush=True)
