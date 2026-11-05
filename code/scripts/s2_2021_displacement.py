"""Sentinel-2 2021 quarters against Neo v3 (CLAUDE.md 13.8), measured as for 2022. Measurement only; nothing enters E3, which is 2022 only as registered. Uses the 2022 functions unchanged, with their paths pointed at the 2021 files:

  displacement.json          scripts/ugi_surface_checks.s2_displacement: per quarter, the correlation peak of Neo v3 NDVI averaged on the 10 m cells with the quarter's NDVI (band 1, 4, 7, 10 of derived/features/s2_2021_indices_10m.tif), 25 windows of 3 km, shifts of +-7 m, parabolic refinement. Convention: content at Neo position (x, y) lies at (x + dx, y + dy). The same function gave -1.42 E / +3.98 N m for 2022 (docs/evidence_phase2/ugi_surface/surface_s2_displacement.json); the 2022 values stand beside it. Neo v3 is October 2022, so the correlation of a 2021 quarter is lower than that of the same quarter of 2022 for reasons that are not geometry.
  residuals_summary.json, blocks_2021Q<k>.csv   scripts/e3_s2_residuals.measure and summary: phase correlation per 1 km block of each quarterly median against Neo v3, with the residual after the 2022 constant and after the quarter's own median.

Outputs in <out> (refuses an existing folder). Run from work_ms/ (CPU): python scripts/s2_2021_displacement.py --out ../docs/evidence_phase2/s2_2021
"""
import argparse
import csv
import json
import sys
from pathlib import Path

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
import e3_s2_residuals as E  # noqa: E402
import ugi_surface_checks as C  # noqa: E402

INDICES = WM / "derived/features/s2_2021_indices_10m.tif"
MEDIANS = WM.parent / "gee_exports/2026-11-04_s2o112_2021/s2_sr_median_{}.tif"
QUARTERS = ("2021Q1", "2021Q2", "2021Q3", "2021Q4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    C.S2 = INDICES
    d = C.s2_displacement()
    d["quarters"] = {k.replace("2022", "2021"): v for k, v in d["quarters"].items()}         # the function names the quarters 2022; it read the 2021 file
    d["source"] = str(INDICES.relative_to(WM))
    y2022 = json.load(open(WM.parent / "docs/evidence_phase2/ugi_surface/surface_s2_displacement.json"))
    d["same_measure_for_2022"] = y2022
    json.dump(d, open(out / "displacement.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in d.items() if k != "same_measure_for_2022"}, indent=1), flush=True)
    E.S2 = str(MEDIANS)
    res = {"what": "displacement of each Sentinel-2 2021 quarter relative to Neo v3 (m east, m north), per 1 km block, as docs/evidence_phase2/ugi_seg/e3_s2_residuals/ for 2022; constant = the one E3 removes for 2022", "constant_m_east_north": list(E.CONST), "quarters": {}}
    for q in QUARTERS:
        rows = E.measure(q)
        with open(out / f"blocks_{q}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        res["quarters"][q] = E.summary(rows)
        print(q, json.dumps(res["quarters"][q]), flush=True)
    json.dump(res, open(out / "residuals_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
