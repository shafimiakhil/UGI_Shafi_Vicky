"""Morning report of the night plan (CLAUDE.md 13.5): at most twenty lines, every number with its path, written from the
result files only; no number is typed here. A result that does not exist is reported as missing. docs/evidence_phase2/
ugi_seg/NIGHT_2026-11-03.md; refuses to overwrite; prints the report. Paths in the report are relative to work_ms/.
Run from work_ms/: python scripts/night_report.py
"""
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
from itareo.safe_write import refuse_existing  # noqa: E402

OUT = WM.parent / "docs/evidence_phase2/ugi_seg/NIGHT_2026-11-03.md"
SUM = "runs_itareo/ugi/summary_r"
MAPS = "runs_itareo/ugi/maps"
SURF = "../docs/evidence_phase2/ugi_surface"
ARMS = ("both", "outer", "inner", "none")


def load(rel):
    p = WM / rel
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def ci(r, k):
    lo, hi = r[f"{k}_ci95_block"]
    return f"{r[k]:.3f} [{lo:.3f}, {hi:.3f}]"


def sd(r, k):
    return f"{r[k]:.4f} ± {r[f'{k}_sd_over_seeds']:.4f}"


def interval(d, n=4):
    return f"{d['point']:+.{n}f} [{d['ci95_block'][0]:+.{n}f}, {d['ci95_block'][1]:+.{n}f}]"


def main():
    refuse_existing(OUT)
    r1, r1e = load(f"{SUM}/summary_r1.json"), load(f"{SUM}/summary_r1_ejido.json")
    dec, b = r1["r1_decision"], r1["r1b_decision"]
    win = dec["winner"]
    band = win.removeprefix("ugi_r1_")
    r3 = f"ugi_r3_a2_{band}"
    L = ["# Night report (CLAUDE.md 13.5; paths relative to `work_ms/`)"]
    L.append(f"**R1, full grid** (`{SUM}/summary_r1.json`; folds 1-4, seed 0; [95 % block bootstrap]; 'no C' = CIRCUNVALACION excluded):")
    L.append("| arm | AP | AP no C | IoU | strict recall | tolerant recall | under 6 m strict | under 6 m tolerant |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        r = r1["arms"][f"ugi_r1_{a}"]
        L.append(f"| {a} | " + " | ".join(ci(r, k) for k in ("ap", "ap_nocirc", "iou", "object_recall_strict", "object_recall_tolerant",
                                                               "recall_under_6m_strict", "recall_under_6m_tolerant")) + " |")
    if r1e:
        de = r1e["r1_decision"]
        same = "the same winner" if de["winner"] == win else f"a different winner, {de['winner']} (not acted on)"
        inside = f"The same rule inside the ejido gives {same} (`{SUM}/summary_r1_ejido.json`)."
    else:
        inside = f"`{SUM}/summary_r1_ejido.json` is missing."
    hi = dec["highest_ap_nocirc"]
    pair = ""
    if hi != win:
        pd1 = r1["paired_differences"]
        v, sign = (pd1[f"{win} - {hi}"], 1) if f"{win} - {hi}" in pd1 else (pd1[f"{hi} - {win}"], -1)
        flip = lambda x: x if sign == 1 else {"point": -x["point"], "ci95_block": [-x["ci95_block"][1], -x["ci95_block"][0]]}
        pair = f" {win} minus {hi}: AP no C {interval(flip(v['ap_nocirc']), 3)}, tolerant recall {interval(flip(v['object_recall_tolerant']), 3)}."
    L.append(f"**R1 decision** (`{SUM}/summary_r1.json`, rule of 13.4): {win}; highest AP no C {hi}, within 0.005: "
             f"{', '.join(dec['within_0.005'])}.{pair} {inside}")
    L.append(f"**R1b** (`{SUM}/summary_r1.json`): 120 against 60 epochs, fold 1, seed 0, AP no C {b['ap_nocirc_120_epochs']:.3f} against "
             f"{b['ap_nocirc_60_epochs']:.3f}, gain {b['gain']:+.3f}: {b['decision']}; reported, not acted on (13.5)."
             + (f" Inside the ejido (`{SUM}/summary_r1_ejido.json`): {r1e['r1b_decision']['ap_nocirc_120_epochs']:.3f} against {r1e['r1b_decision']['ap_nocirc_60_epochs']:.3f}, "
                f"gain {r1e['r1b_decision']['gain']:+.3f}." if r1e else ""))
    logp = WM / "runs_itareo/ugi/night_queue_log.csv"
    rows = list(csv.DictReader(open(logp))) if logp.exists() else []
    ok = [x for x in rows if x["exit"] == "0"]
    n = lambda c, s: sum(x["config"] == c and x["seed"] == str(s) for x in ok)
    qp = WM / "runs_itareo/ugi/night_queue_stdout.txt"
    status = [x for x in qp.read_text(encoding="utf-8").splitlines() if x.startswith(("waiting for", "done", "STOP"))] if qp.exists() else []
    stop = f" The queue's last status: {status[-1]}." if status and not status[-1].startswith("done") else ""
    L.append(f"**Runs finished** (`runs_itareo/ugi/night_queue_log.csv`): {len(ok)} of 12 with exit 0: R2a {n(win, 1)} ({win}, seed 1), "
             f"R3 {n(r3, 0)} ({r3}, seed 0), R2b {n(win, 2)} ({win}, seed 2).{stop}")
    be, bf = load(f"{SUM}/summary_baseline_ejido.json"), load(f"{SUM}/summary_baseline.json")
    keys = (("AP", "ap"), ("IoU", "iou"), ("strict", "object_recall_strict"), ("tolerant", "object_recall_tolerant"))
    keys_nc = (("AP", "ap_nocirc"), ("IoU", "iou_nocirc"), ("strict", "object_recall_strict_nocirc"), ("tolerant", "object_recall_tolerant_nocirc"))
    fmt = lambda r, ks: ", ".join(f"{n_} {sd(r, k)}" for n_, k in ks)
    if be and bf:
        be, bf = be["arms"][win], bf["arms"][win]
        L.append(f"**Three-seed baseline, inside the ejido** ({win}, mean ± SD over seeds 0-2, `{SUM}/summary_baseline_ejido.json`): {fmt(be, keys)}; "
                 f"no C: {fmt(be, keys_nc)}.")
        L.append(f"**The same, full grid** (`{SUM}/summary_baseline.json`): {fmt(bf, keys)}; no C: {fmt(bf, keys_nc)}.")
    else:
        L.append(f"**Three-seed baseline:** missing (`{SUM}/summary_baseline_ejido.json`, `summary_baseline.json`): not every run of seeds 1 and 2 finished.")
    d3e, d3f = load(f"{SUM}/summary_r3_ejido.json"), load(f"{SUM}/summary_r3.json")
    if d3e and d3f:
        pk = f"{r3} - {win}"
        d = d3e["paired_differences"][pk]["ap_nocirc"]
        excl = d["ci95_block"][0] > 0 or d["ci95_block"][1] < 0
        df_ = d3f["paired_differences"][pk]["ap_nocirc"]
        sens = load(f"{SUM}/r3_interval_sensitivity.json")
        firm = ""
        if sens:
            by_seed = sens["a2_seed0_minus_a1_by_a1_seed"]
            folds = ", ".join(f"{v['difference']:+.3f}" for v in sens["per_fold_seed0"].values())
            le, lf = sens["long_run"]["inside_ejido"], sens["long_run"]["full_grid"]
            me, mf_ = sens["many_seeds"]["inside_ejido"], sens["many_seeds"]["full_grid"]
            firm = (f" How firm (`{SUM}/r3_interval_sensitivity.json`): narrowly. With {le['resamples']:,} resamples the interval is [{le['ci95'][0]:+.4f}, {le['ci95'][1]:+.4f}]; at 2,000 resamples "
                    f"{100 * me['share_including_zero']:.0f} % of {me['seeds']} other resampling seeds give an interval that includes zero. Against A1 seed 1 {interval(by_seed['1'], 3)}, "
                    f"against A1 seed 2 {interval(by_seed['2'], 3)}; folds 1-4: {folds}.")
        L.append(f"**A2 against A1** (`{SUM}/summary_r3_ejido.json`; inside the ejido, no C, seed 0; one seed, reported as such): AP {ci(d3e['arms'][r3], 'ap_nocirc')} against "
                 f"{ci(d3e['arms'][win], 'ap_nocirc')}; paired difference {interval(d)}: the interval "
                 f"{'excludes' if excl else 'includes'} zero, so under the pre-registered reading A2 {('differs from A1, ' + ('lower' if d['point'] < 0 else 'higher')) if excl else 'does not differ from A1'}."
                 f"{firm} Full grid (`{SUM}/summary_r3.json`): {interval(df_)}"
                 + (f", {100 * mf_['share_including_zero']:.0f} % of the other seeds include zero, so the full grid does not clearly exclude it." if sens else "."))
    else:
        L.append(f"**A2 against A1:** missing (`{SUM}/summary_r3_ejido.json`, `summary_r3.json`): not every R3 run finished.")
    m, m7 = load(f"{MAPS}/ejido/metrics.json"), load(f"{MAPS}/ejido/metrics_rule7.json")
    mf, mf7 = load(f"{MAPS}/ensemble_metrics.json"), load(f"{MAPS}/ensemble_metrics_rule7.json")
    px = lambda x: f"AUROC {x['auroc']:.3f}, AP {x['ap']:.3f}, IoU {x['iou']:.3f}, P {x['precision']:.3f}, R {x['recall']:.3f}, F1 {x['f1']:.3f}"
    if m and m7:
        L.append(f"**Stage M inside the ejido** (`{MAPS}/ejido/metrics.json`, `metrics_rule7.json`): {px(m['pixel_full'])}; no C: AP {m7['pixel']['no_circunvalacion']['ap']:.3f}, "
                 f"IoU {m7['pixel']['no_circunvalacion']['iou']:.3f}; strict and tolerant recall {m['object_recall_strict']:.3f}, {m['object_recall_tolerant']:.3f}; object precision "
                 f"{m['components']['object_precision']:.3f}. All pixels (`{MAPS}/ensemble_metrics.json`, `ensemble_metrics_rule7.json`): {px(mf['pixel_full'])}; no C: AP "
                 f"{mf7['pixel']['no_circunvalacion']['ap']:.3f}, IoU {mf7['pixel']['no_circunvalacion']['iou']:.3f}.")
    else:
        L.append(f"**Stage M inside the ejido:** missing (`{MAPS}/ejido/`).")
    if r1e:
        L.append(f"**R1 inside the ejido** (`{SUM}/summary_r1_ejido.json`; AP, AP no C, IoU, tolerant recall): " + "; ".join(
            f"{a} {r1e['arms'][f'ugi_r1_{a}']['ap']:.3f}, {r1e['arms'][f'ugi_r1_{a}']['ap_nocirc']:.3f}, {r1e['arms'][f'ugi_r1_{a}']['iou']:.3f}, "
            f"{r1e['arms'][f'ugi_r1_{a}']['object_recall_tolerant']:.3f}" for a in ARMS) + ".")
    else:
        L.append(f"**R1 inside the ejido:** missing (`{SUM}/summary_r1_ejido.json`).")
    ty = list(csv.DictReader(open(WM / SURF / "surface_by_type.csv", encoding="utf-8")))
    pc = lambda r, k: f"{100 * float(r[k]):.0f}"
    L.append(f"**Surface composition by type** (`{SURF}/surface_by_type.csv`; % of area, always vegetated / seasonal / never vegetated / water): " + "; ".join(
        f"{r['Tipo_evp_2']} {pc(r, 'share_always_vegetated')}/{pc(r, 'share_seasonal')}/{pc(r, 'share_never_vegetated')}/{pc(r, 'share_water')}" for r in ty) + ".")
    sl = load(f"{SURF}/surface_log.json")
    pd_ = sl["planetscope_dates_available_pixels"]
    s2d = load(f"{SURF}/surface_s2_displacement.json")
    s2n = (f" Sentinel-2 is read without its measured displacement ({s2d['dx_east_m_all']:+.1f} E, {s2d['dy_north_m_all']:+.1f} N m); read displaced, "
           f"{100 * s2d['effect_of_reading_s2_displaced']['share_changing_class']:.1f} % of the pixels would change class (`{SURF}/surface_s2_displacement.json`).") if s2d else ""
    L.append(f"Seasonal is the remainder of the rule, not a phenology class; buildings cannot be separated without a 2022 building layer. PlanetScope dates available at inventory pixels "
             f"(`{SURF}/surface_log.json`), 0/1/2/3: " + "/".join(f"{100 * pd_[k] / sl['inventory_pixels']:.1f}" for k in "0123") + f" %.{s2n}")
    st = load("derived/s1_2021_2022/status.json")
    L.append(f"**Sentinel-1 2021-2022** (`derived/s1_2021_2022/status.json`): {st['status']}" if st
             else "**Sentinel-1 2021-2022:** not attempted; `derived/s1_2021_2022/status.json` does not exist.")
    if len(L) > 20:
        raise SystemExit(f"{len(L)} lines, the limit is twenty")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
