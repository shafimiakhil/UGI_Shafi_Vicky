"""E1 and E2 keep the ejido reading they were registered with; the centre is reported beside it (CLAUDE.md 13.7, step 2). For an arm: itareo.ugi_seg.rescore_centre on its seed 0 runs for every candidate, the paired summary against ugi_r1_none seed 0 inside each, and a table. Reads files only.
From work_ms/: python scripts/centre_rescore_report.py --arm ugi_e1_ps --tag e1 --out ../docs/evidence_phase2/ugi_seg/e1/centre_rescoring     (refuses an existing output folder)
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import exp_report as er  # noqa: E402

NAMES = ["ring", "sq3km", "sq5km", "sq8km"]
SUM = Path("runs_itareo/ugi/summary_r")
BASE = "ugi_r1_none"
ROWS = [("AP", "ap_nocirc"), ("AUROC", "auroc_nocirc"), ("IoU", "iou_nocirc"), ("F1", "f1_nocirc"), ("Object recall, tolerant", "object_recall_tolerant"), ("Object recall, strict", "object_recall_strict")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-W", "ignore", "-m", "itareo.ugi_seg.rescore_centre", "--regions", *NAMES, "--runs", f"runs_itareo/ugi/{a.arm}", "--seeds", "0", "--workers", "4"], check=True)
    rows = []
    for n in NAMES:
        sub = f"centre_{n}"
        tag = f"{a.tag}_{sub}"
        if not (SUM / f"summary_{tag}.json").exists():
            subprocess.run([sys.executable, "-W", "ignore", "-m", "itareo.ugi_seg.summarize_r", "--arms", BASE, a.arm, "--seeds", "0", "--sub", sub, "--out", str(SUM), "--tag", tag], check=True, stdout=subprocess.DEVNULL)
        s = json.load(open(SUM / f"summary_{tag}.json"))
        x = er.auroc_f1(a.arm, sub)
        thr = 3 * json.load(open(SUM / f"summary_{sub}.json"))["arms"][BASE]["ap_nocirc_sd_over_seeds"]
        fb, fa = er.per_fold(BASE, sub), er.per_fold(a.arm, sub)
        for label, k in ROWS:
            b_, a_, d, c = er.score_cell(s, x, a.arm, k)
            rows.append({"candidate": n, "score": label, "baseline_seed0": round(b_, 4), "arm": round(a_, 4), "difference": round(d, 4), "ci95_block_low": round(c[0], 4), "ci95_block_high": round(c[1], 4)})
        rows.append({"candidate": n, "score": "per-fold difference in AP (folds 1-4)", "baseline_seed0": "", "arm": "", "difference": " ".join(f"{fa[f] - fb[f]:+.4f}" for f in er.FOLDS), "ci95_block_low": "", "ci95_block_high": ""})
        rows.append({"candidate": n, "score": "3 x seed SD of AP inside it, for reference only", "baseline_seed0": "", "arm": "", "difference": round(thr, 4), "ci95_block_low": "", "ci95_block_high": ""})
    with open(out / "centre_rescoring.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = [f"# {a.arm} against {BASE}, seed 0, inside the centre candidates (CLAUDE.md 13.7, step 2)", "", f"For information: this input keeps the ejido reading it was registered with. No CIRCUNVALACION polygon lies inside a candidate. Summaries: `runs_itareo/ugi/summary_r/summary_{a.tag}_centre_<name>.json`.", "", "| candidate | score | baseline | arm | difference | 95 % block interval |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["ci95_block_low"] != "":
            L.append(f"| {r['candidate']} | {r['score']} | {r['baseline_seed0']:.4f} | {r['arm']:.4f} | {r['difference']:+.4f} | [{r['ci95_block_low']:+.4f}, {r['ci95_block_high']:+.4f}] |")
        else:
            L.append(f"| {r['candidate']} | {r['score']} | | | {r['difference']} | |")
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
