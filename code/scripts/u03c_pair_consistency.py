"""Consistency of the 1_3 -> 1_2 pair measurement (CLAUDE.md 13.3 F1, found 3 Nov 2026 during --quad13-check).

On every used block of the pair (blocks_measure.csv), the displacement of 1_2 relative to 1_3 is measured three ways with
the same A7 phase correlation (1 km blocks, used blocks only):
  table        the stored value (unregistered measurement copies, derived/pneo2022_registered/measure/);
  remeasured   the same copies measured again (must equal the table);
  registered   the registered measurement copies (measure_registered/check_1_2_registered and check_1_3_translation).
If phase correlation were exactly shift-equivariant, registered = table + (t13 - t12) = table + (1.05, -1.70) m (the v1
translations move each strip's content by minus its displacement). Each strip's own shift is measured as well (registered
copy against its unregistered copy; expected minus its displacement: 1_2 (+1.30, +0.30), 1_3 (+0.25, +2.00)).
Output: derived/pneo2022_registered/pair_consistency_13_12.json and _blocks.csv. Run from work_ms/.
--lowpass L adds the estimate weighted toward low frequencies (itareo.phasecorr lowpass, cycles per px; A5 uses 0.1 to
check resampled images) and writes pair_consistency_13_12_lowpass.json and _lowpass_blocks.csv (the unweighted files are
left as they are). Added 3 Nov 2026 after the code review found the unweighted estimate biased on these copies.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a7_coregistration as A7      # noqa: E402
import u03_neo_register as U        # noqa: E402


def main(lowpass=None):
    reg = json.loads((U.OUT / "registration.json").read_text(encoding="utf-8"))["applied_displacements"]
    t12 = np.array([reg["1_2"]["dx_east_m"], reg["1_2"]["dy_north_m"]])
    t13 = np.array([reg["1_3"]["dx_east_m"], reg["1_3"]["dy_north_m"]])
    MR = U.OUT / "measure_registered"
    rows = [r for r in csv.DictReader(open(U.OUT / "blocks_measure.csv")) if r["pair"] == "1_3->1_2" and r["used"] == "True"]
    out = []
    for r in rows:
        r0, c0 = int(r["block_row"]), int(r["block_col"])
        u13, u12 = U.neo_block(U.MEAS / "measure_1_3_grid1m.tif", r0, c0, U.BLOCK), U.neo_block(U.MEAS / "measure_1_2_grid1m.tif", r0, c0, U.BLOCK)
        g13, g12 = U.neo_block(MR / "check_1_3_translation_grid1m.tif", r0, c0, U.BLOCK), U.neo_block(MR / "check_1_2_registered_grid1m.tif", r0, c0, U.BLOCK)
        m = {"remeasured": A7.correlate({}, *u13, *u12, 1.0, lowpass), "registered": A7.correlate({}, *g13, *g12, 1.0, lowpass),
             "shift_1_2": A7.correlate({}, *u12, *g12, 1.0, lowpass), "shift_1_3": A7.correlate({}, *u13, *g13, 1.0, lowpass)}
        row = {"block_row": r0, "block_col": c0, "table_dx_m": float(r["dx_m"]), "table_dy_m": float(r["dy_m"])}
        for k, v in m.items():
            row[f"{k}_used"] = bool(v.get("used"))
            row[f"{k}_dx_m"], row[f"{k}_dy_m"] = v.get("dx_m"), v.get("dy_m")
            if lowpass:
                row[f"{k}_lp_dx_m"], row[f"{k}_lp_dy_m"] = v.get("dx_lp_m"), v.get("dy_lp_m")
        out.append(row)
    suffix = "_lowpass" if lowpass else ""
    with open(U.OUT / f"pair_consistency_13_12{suffix}_blocks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    def med(key, cond=None):
        v = [(r[f"{key}_dx_m"], r[f"{key}_dy_m"]) for r in out if r[f"{key}_used"] and (cond is None or cond(r))]
        a = np.array(v, float)
        return {"blocks": len(v), "dx_median_m": round(float(np.median(a[:, 0])), 3) if v else None,
                "dy_median_m": round(float(np.median(a[:, 1])), 3) if v else None}
    exp = t13 - t12
    both = [r for r in out if r["registered_used"]]
    dev = np.array([(r["registered_dx_m"] - r["table_dx_m"] - exp[0], r["registered_dy_m"] - r["table_dy_m"] - exp[1]) for r in both])
    res = {"blocks": len(out), "expected_registered_minus_table_m": exp.round(3).tolist(),
           "table": {"dx_median_m": round(float(np.median([r["table_dx_m"] for r in out])), 3),
                     "dy_median_m": round(float(np.median([r["table_dy_m"] for r in out])), 3)},
           "remeasured_equals_table": all(r["remeasured_used"] and r["remeasured_dx_m"] == r["table_dx_m"]
                                          and r["remeasured_dy_m"] == r["table_dy_m"] for r in out),
           "registered": med("registered"),
           "registered_minus_table_minus_expected_m": {"blocks": len(dev), "dx_median": round(float(np.median(dev[:, 0])), 3),
                                                        "dy_median": round(float(np.median(dev[:, 1])), 3),
                                                        "dx_iqr": np.percentile(dev[:, 0], [25, 75]).round(3).tolist(),
                                                        "dy_iqr": np.percentile(dev[:, 1], [25, 75]).round(3).tolist()},
           "shift_1_2_registered_vs_unregistered": {**med("shift_1_2"), "expected_m": (-t12).round(3).tolist()},
           "shift_1_3_registered_vs_unregistered": {**med("shift_1_3"), "expected_m": (-t13).round(3).tolist()}}
    if lowpass:
        def medlp(key):
            v = np.array([(r[f"{key}_lp_dx_m"], r[f"{key}_lp_dy_m"]) for r in out if r[f"{key}_used"]], float)
            return {"blocks": len(v), "dx_median_m": round(float(np.median(v[:, 0])), 3), "dy_median_m": round(float(np.median(v[:, 1])), 3)}
        devl = np.array([(r["registered_lp_dx_m"] - r["remeasured_lp_dx_m"] - exp[0], r["registered_lp_dy_m"] - r["remeasured_lp_dy_m"] - exp[1])
                         for r in out if r["registered_used"] and r["remeasured_used"]])
        res["lowpass"] = {"cycles_per_px": lowpass, "table_remeasured": medlp("remeasured"), "registered": medlp("registered"),
                          "registered_minus_table_minus_expected_m": {"blocks": len(devl), "dx_median": round(float(np.median(devl[:, 0])), 3),
                                                                       "dy_median": round(float(np.median(devl[:, 1])), 3),
                                                                       "dx_iqr": np.percentile(devl[:, 0], [25, 75]).round(3).tolist(),
                                                                       "dy_iqr": np.percentile(devl[:, 1], [25, 75]).round(3).tolist()},
                          "shift_1_2_registered_vs_unregistered": {**medlp("shift_1_2"), "expected_m": (-t12).round(3).tolist()},
                          "shift_1_3_registered_vs_unregistered": {**medlp("shift_1_3"), "expected_m": (-t13).round(3).tolist()}}
    (U.OUT / f"pair_consistency_13_12{suffix}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowpass", type=float)
    main(ap.parse_args().lowpass)
