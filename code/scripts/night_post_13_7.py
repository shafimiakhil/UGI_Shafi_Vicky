"""CPU side of the night queue of CLAUDE.md 13.7 (jobs/night_queue_13_7.sh). Reads and writes only the scoring files of finished runs; never touches the GPU. Every 120 s:
  * rules 7 and 9 on each finished run of E1b, E3, fold C and their later seeds; the B4 tile set for E1b; the chosen centre for those and for E1 and E2 (derived/ugi_centre/chosen.txt, written on Shafi's word; nothing centre-related runs before it exists);
  * when E1b or E3 has its four folds of seed 0 and the centre exists: the paired summaries against ugi_r1_none seed 0 and the pre-registered reading (scripts/exp_report.py --centre: the centre primary, with the threshold 3 x seed SD; the ejido secondary), once;
  * when fold C has its three seeds: the summary inside the centre and the table against the four-quadrant protocol (docs/evidence_phase2/ugi_seg/foldc/README.md), once.
It ends when the queue has printed done or STOP and nothing is left to do. A failed step prints "STOP: ..." and ends it; it can be relaunched once the failed step's partial output is archived (a folder without done.json, a reading folder without reading.md, rule 7 files without metrics_rule7.json): every step skips what is finished and refuses what is partial.
If E2 passes, its seeds 1-2 get rules 7 and 9 here, because jobs/ugi_e2_post.sh has ended by then. The baseline's seeds 1-2 get the B4 tile set once E1b's seeds 1-2 start.
Run from work_ms/: python scripts/night_post_13_7.py >> runs_itareo/ugi/night2_post_stdout.txt 2>&1
"""
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("runs_itareo/ugi")
QUEUE = ROOT / "night2_stdout.txt"
CHOSEN = Path("derived/ugi_centre/chosen.txt")
SUM = ROOT / "summary_r"
DOC = Path("../docs/evidence_phase2/ugi_seg")
FOLDS = (1, 2, 3, 4)
ARMS = {"ugi_e1b_ps_plane": {"tag": "e1b", "b4": True}, "ugi_e3_s2": {"tag": "e3", "b4": False}}


def sh(*a):
    r = subprocess.run([sys.executable, "-W", "ignore", *a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(f"STOP: step failed: {' '.join(a)} {time.strftime('%F %T')}", flush=True)
        raise SystemExit(1)


def chosen():
    return CHOSEN.read_text(encoding="utf-8").strip() if CHOSEN.exists() else None


def finished(arm, seeds=(0,), folds=FOLDS, sub=None):
    return all((ROOT / arm / f"fold{f}_seed{s}" / (sub or "") / ("done.json" if sub else "run_log.md")).exists() for s in seeds for f in folds)


def summaries(arm, tag, b4, name):
    if not (SUM / f"summary_{tag}_ejido.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", "ugi_r1_none", arm, "--seeds", "0", "--ejido", "--out", str(SUM), "--tag", f"{tag}_ejido")
    if not (SUM / f"summary_{tag}.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", "ugi_r1_none", arm, "--seeds", "0", "--out", str(SUM), "--tag", tag)
    if b4 and not (SUM / f"summary_{tag}_ejido_b4.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", "ugi_r1_none", arm, "--seeds", "0", "--sub", "ejido_b4", "--out", str(SUM), "--tag", f"{tag}_ejido_b4")
    if not (SUM / f"summary_{tag}_centre_{name}.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", "ugi_r1_none", arm, "--seeds", "0", "--sub", f"centre_{name}", "--out", str(SUM), "--tag", f"{tag}_centre_{name}")


def f1_seeds(arm, folds, name):
    """F1 and F1 without CIRCUNVALACION per seed, from the block counts inside the centre: 2 tp / (2 tp + fp + fn)."""
    a, b = [], []
    for s in (0, 1, 2):
        t = np.zeros(3)
        n = np.zeros(3)
        for f in folds:
            d = ROOT / arm / f"fold{f}_seed{s}" / f"centre_{name}"
            x, y = np.load(d / "boot.npz"), np.load(d / "boot_rule7.npz")
            t += [x[f"block_{k}"].sum() for k in ("tp", "fp", "fn")]
            n += [y[f"block_nocirc_{k}"].sum() for k in ("tp", "fp", "fn")]
        a.append(2 * t[0] / (2 * t[0] + t[1] + t[2]))
        b.append(2 * n[0] / (2 * n[0] + n[1] + n[2]))
    return a, b


def foldc_table(name):
    arm = f"ugi_foldc_{name}"
    out = DOC / "foldc"
    if (out / "README.md").exists():
        return
    if not (SUM / f"summary_foldc_{name}.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", arm, "--seeds", "0", "1", "2", "--folds", "5", "--sub", f"centre_{name}", "--out", str(SUM), "--tag", f"foldc_{name}")
    c = json.load(open(SUM / f"summary_foldc_{name}.json"))["arms"][arm]
    q = json.load(open(SUM / f"summary_centre_{name}.json"))["arms"]["ugi_r1_none"]
    keys = (("AP, no C", "ap_nocirc"), ("AP", "ap"), ("AUROC", "auroc"), ("IoU, no C", "iou_nocirc"), ("IoU", "iou"), ("Object recall, tolerant", "object_recall_tolerant"), ("Object recall, strict", "object_recall_strict"))
    fc, fq = f1_seeds(arm, (5,), name), f1_seeds("ugi_r1_none", FOLDS, name)
    n_c = len(list(csv.DictReader(open(ROOT / arm / "fold5_seed0" / f"centre_{name}" / "objects.csv"))))
    n_q = sum(len(list(csv.DictReader(open(ROOT / "ugi_r1_none" / f"fold{f}_seed0" / f"centre_{name}" / "objects.csv")))) for f in FOLDS)
    L = [f"# Fold C: the centre {name} held out, with a 512 m buffer (CLAUDE.md 13.7, step 3)", "", f"`ugi_r1_none`, seeds 0-2, config `itareo/configs/ugi_foldc_{name}.yaml`; `runs_itareo/ugi/{arm}/`; `runs_itareo/ugi/summary_r/summary_foldc_{name}.json`. Against the same three seeds in the four-quadrant protocol, scored inside the same centre (`summary_centre_{name}.json`).", "",
         "| score | fold C, mean | SD | 95 % block interval | four quadrants, mean | SD |", "|---|---|---|---|---|---|"]
    for lab, k in keys:
        ci = c.get(f"{k}_ci95_block")
        L.append(f"| {lab} | {c[k]:.4f} | {c.get(f'{k}_sd_over_seeds', float('nan')):.4f} | " + (f"{ci[0]:.4f} to {ci[1]:.4f}" if ci else "") + f" | {q[k]:.4f} | {q.get(f'{k}_sd_over_seeds', float('nan')):.4f} |")
    L += [f"| F1 | {np.mean(fc[0]):.4f} | {np.std(fc[0], ddof=1):.4f} | | {np.mean(fq[0]):.4f} | {np.std(fq[0], ddof=1):.4f} |", f"| F1, no C | {np.mean(fc[1]):.4f} | {np.std(fc[1], ddof=1):.4f} | | {np.mean(fq[1]):.4f} | {np.std(fq[1], ddof=1):.4f} |"]
    L += ["", f"Pixel scores are on the same pixels in both columns. Object recall is on {n_c} polygons in fold C and {n_q} in the four quadrants: a polygon split by a quadrant line is scored per quadrant on the right and whole in fold C.", "", "Caveat of 13.7: the centre is enclosed by training data on all sides, hence the buffer; fewer positives widen the intervals. Pixels outside the ejido stay in the loss, as in the sensor series (13.6); tiles straddling a quadrant line stay out, because the cache does not hold them."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def queue_ended():
    if not QUEUE.exists():
        return False
    last = [x for x in QUEUE.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"^(self-tests|done|STOP)", x)]
    return bool(last) and bool(re.match(r"^(done|STOP)", last[-1]))


def one_pass():
    name = chosen()
    runs = [str(ROOT / a) for a in ARMS]
    if name:
        runs.append(str(ROOT / f"ugi_foldc_{name}"))
    existing = [r for r in runs if Path(r).exists()]
    e2_later = [str(ROOT / f"ugi_e2_tex/fold{f}_seed{s}") for s in (1, 2) for f in FOLDS if (ROOT / f"ugi_e2_tex/fold{f}_seed{s}").exists()]
    if existing or e2_later:
        sh("-m", "itareo.ugi_seg.rescore7", "--runs", *existing, *e2_later)
        sh("-m", "itareo.ugi_seg.rescore9", "--runs", *existing, *e2_later)
    if (ROOT / "ugi_e1b_ps_plane").exists():
        seeds = (0, 1, 2) if (ROOT / "ugi_e1b_ps_plane/fold1_seed1").exists() else (0,)
        base = [str(ROOT / f"ugi_r1_none/fold{f}_seed{s}") for s in seeds for f in FOLDS]
        sh("-m", "itareo.ugi_seg.rescore9", "--b4", "--runs", str(ROOT / "ugi_e1b_ps_plane"), *base)
    if name:
        more = [str(ROOT / a) for a in ("ugi_e1_ps", "ugi_e2_tex")] + existing
        sh("-m", "itareo.ugi_seg.rescore_centre", "--regions", name, "--runs", *[m for m in more if Path(m).exists()], "--workers", "4")
        for arm, v in ARMS.items():
            tag = v["tag"]
            if finished(arm) and all((ROOT / arm / f"fold{f}_seed0" / "metrics_rule7.json").exists() for f in FOLDS) and finished(arm, sub="ejido") and finished(arm, sub=f"centre_{name}") and (not v["b4"] or finished(arm, sub="ejido_b4")) and not (DOC / tag / "reading.md").exists():
                summaries(arm, tag, v["b4"], name)
                a = ["scripts/exp_report.py", "--arm", arm, "--tag", tag, "--out", str(DOC / tag), "--centre", name]
                sh(*(a + (["--b4"] if v["b4"] else [])))
        c = f"ugi_foldc_{name}"
        if finished(c, (0, 1, 2), (5,)) and finished(c, (0, 1, 2), (5,), sub=f"centre_{name}"):
            foldc_table(name)


if __name__ == "__main__":
    print(f"night post start {time.strftime('%F %T')}", flush=True)
    while True:
        ended = queue_ended()
        one_pass()
        if ended:
            print(f"night post done {time.strftime('%F %T')}", flush=True)
            break
        time.sleep(120)
