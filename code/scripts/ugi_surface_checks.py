"""Checks on the inventory surface composition (CLAUDE.md 13.5, after the independent review). They read the classes and
change nothing; the classes stay as scripts/ugi_surface.py wrote them. Outputs in docs/evidence_phase2/ugi_surface/ (refuses
existing files):
  surface_vs_geobia2017.csv     per GEOBIA 2017 class (Rodriguez Gonzalez et al. 2023; a classification of 2017, a cross-check
                                only) the inventory pixels and their shares of the four surface classes, for all pixels and
                                for those with Neo v3 NDVI < 0.1, still without vegetation at 1 m in 2022;
  surface_seasonal_anatomy.json what class 4 holds: the share of its pixels that cross both thresholds (one observation below
                                0.2 and one at or above 0.3) within PlanetScope or within the Sentinel-2 quarters, and how
                                many pixels with Neo NDVI below 0.2 or at or above 0.3 fail the other sensors' tests;
  surface_s2_displacement.json  Sentinel-2 content against Neo v3: displacement per quarter from the correlation peak of Neo NDVI
                                averaged on the 10 m cells with each quarter's NDVI (25 windows of 3 km, shifts of +-7 m, parabolic
                                refinement), and what the classes would be if Sentinel-2 were read at the displaced position,
                                as PlanetScope is; the classes are also recomputed at the nominal position and compared with
                                the stored raster, pixel by pixel.
Run from work_ms/: python scripts/ugi_surface_checks.py
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
import numpy as np                      # noqa: E402
import rasterio                         # noqa: E402
from rasterio.windows import Window     # noqa: E402

WM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM))
sys.path.insert(0, str(WM / "scripts"))
from itareo.safe_write import refuse_existing  # noqa: E402
from ugi_surface import ATTRS, CHUNK, CLASSES, IDS, NEO, NEO_NDVI, NEO_NDWI, NID, OFFSETS, OUT, PS, S2, S2_NDVI, ps_flags  # noqa: E402

GEOBIA = WM / "derived/geobia2017/geobia2017_grid1m.tif"
GEO_NAMES = {0: "no label", 1: "Trees", 2: "Low vegetation", 3: "Water", 4: "Buildings", 5: "Asphalt", 6: "Bare soil", 7: "Shadows"}
SHIFTS = list(range(-7, 8))


def parabola(a, b, c):
    d = a - 2 * b + c
    return 0.5 * (a - c) / d if d else 0.0


def peak(acc):
    i, j = np.unravel_index(acc.argmax(), acc.shape)
    dj = parabola(acc[i, j - 1], acc[i, j], acc[i, j + 1]) if 0 < j < len(SHIFTS) - 1 else 0.0
    di = parabola(acc[i - 1, j], acc[i, j], acc[i + 1, j]) if 0 < i < len(SHIFTS) - 1 else 0.0
    return SHIFTS[j] + dj, SHIFTS[i] + di, acc[i, j]


def s2_displacement():
    """S2 content at Neo position (x, y) lies at (x + dx, y + dy): the sign used for PlanetScope in b4_ps_offsets.json."""
    with rasterio.open(S2) as s:
        q = s.read(S2_NDVI).astype(np.float32) / 1e4
    peaks = {k: [] for k in range(4)}
    acc_all = {k: np.zeros((len(SHIFTS), len(SHIFTS))) for k in range(4)}
    nw = 0
    with rasterio.open(NEO) as sn:
        nd, H, W = sn.nodata, sn.height, sn.width
        for r0 in range(1993, H - 3100, 4000):
            for c0 in range(1993, W - 3100, 4000):
                raw = sn.read(NEO_NDVI, window=Window(c0 - 10, r0 - 10, 3020, 3020))
                if (raw == nd).mean() > 0.02:
                    continue
                n = raw.astype(np.float32) / 1e4
                sc0, sr0 = (c0 + 7) // 10, (r0 + 7) // 10
                nw += 1
                cells = {}
                for i, dy in enumerate(SHIFTS):
                    for j, dx in enumerate(SHIFTS):
                        sub = n[10 + dy:10 + dy + 3000, 10 - dx:10 - dx + 3000]    # Neo at (X - dx, Y - dy): columns -dx, rows +dy
                        cells[(i, j)] = sub.reshape(300, 10, 300, 10).mean(axis=(1, 3)).ravel()
                for k in range(4):
                    v = q[k, sr0:sr0 + 300, sc0:sc0 + 300].ravel()
                    acc = np.zeros((len(SHIFTS), len(SHIFTS)))
                    for (i, j), m in cells.items():
                        acc[i, j] = np.corrcoef(m, v)[0, 1]
                    acc_all[k] += acc
                    peaks[k].append(peak(acc))
    out = {"windows": nw, "quarters": {}}
    for k in range(4):
        p = np.array(peaks[k])
        mean_acc = acc_all[k] / nw
        out["quarters"][f"2022Q{k + 1}"] = {
            "dx_east_m_median": round(float(np.median(p[:, 0])), 3), "dy_north_m_median": round(float(np.median(p[:, 1])), 3),
            "dx_east_m_iqr": [round(float(x), 3) for x in np.percentile(p[:, 0], [25, 75])],
            "dy_north_m_iqr": [round(float(x), 3) for x in np.percentile(p[:, 1], [25, 75])],
            "r_at_peak_mean_surface": round(float(peak(mean_acc)[2]), 4), "r_at_zero_mean_surface": round(float(mean_acc[SHIFTS.index(0), SHIFTS.index(0)]), 4)}
    all_p = np.concatenate([np.array(peaks[k]) for k in range(4)])
    out["dx_east_m_all"], out["dy_north_m_all"] = round(float(np.median(all_p[:, 0])), 3), round(float(np.median(all_p[:, 1])), 3)
    return out


def main():
    t0 = time.time()
    outputs = [OUT / n for n in ("surface_vs_geobia2017.csv", "surface_seasonal_anatomy.json", "surface_s2_displacement.json")]
    refuse_existing(*outputs)
    disp = s2_displacement()
    print(json.dumps(disp, indent=1), round(time.time() - t0), "s", flush=True)
    dx2, dy2 = disp["dx_east_m_all"], disp["dy_north_m_all"]
    off = json.load(open(OFFSETS))["offsets_relative_to_registered_neo2022"]
    flags = {d: ps_flags(p) for d, p in PS.items()}
    ps_h, ps_w = next(iter(flags.values()))[0].shape
    with rasterio.open(S2) as s:
        q = s.read(S2_NDVI)
        s2_h, s2_w = s.height, s.width
    q_max, q_min = q.max(0), q.min(0)                 # every inventory pixel has all four quarters (surface_log.json)
    attrs = {int(r["id"]): r for r in csv.DictReader(open(ATTRS, encoding="utf-8"))}
    types = sorted({r["Tipo_evp_2"] for r in attrs.values()})
    ty_of = np.zeros(NID, np.int64)
    for i, r in attrs.items():
        ty_of[i] = types.index(r["Tipo_evp_2"])
    geo_all, geo_bare = np.zeros((8, 5), np.int64), np.zeros((8, 5), np.int64)
    tab_nom, tab_dis = np.zeros((len(types), 5), np.int64), np.zeros((len(types), 5), np.int64)
    changed = different = 0
    an = {"class4": 0, "class4_cross_planetscope": 0, "class4_cross_sentinel2": 0, "class4_cross_one_sensor": 0,
          "neo_below_02_not_water": 0, "of_which_never": 0, "fail_planetscope_only": 0, "fail_sentinel2_only": 0, "fail_both": 0,
          "neo_from_03": 0, "of_which_always": 0, "always_fail_planetscope_only": 0, "always_fail_sentinel2_only": 0, "always_fail_both": 0}
    with rasterio.open(IDS) as sid, rasterio.open(NEO) as sn, rasterio.open(GEOBIA) as sg, rasterio.open(OUT / "surface_class_grid1m.tif") as sc:
        H, W = sid.height, sid.width
        for a in range(0, H, CHUNK):
            win = Window(0, a, W, min(CHUNK, H - a))
            ids = sid.read(1, window=win)
            vi = ids > 0
            if not vi.any():
                continue
            rr, cc = np.nonzero(vi)
            rows = rr + a
            ii = ids[vi].astype(np.int64)
            stored = sc.read(1, window=win)[vi]
            ndvi, ndwi = sn.read(NEO_NDVI, window=win)[vi], sn.read(NEO_NDWI, window=win)[vi]
            geo = sg.read(1, window=win)[vi].astype(np.int64)
            n = len(ii)
            all_lt, all_ge, any_lt, any_ge = np.ones(n, bool), np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool)
            for d, (av, lt, ge) in flags.items():
                pr = np.floor((rows + 0.5 - off[d]["dy_north_m"]) / 3).astype(np.int64)
                pc = np.floor((cc + 0.5 + off[d]["dx_east_m"]) / 3).astype(np.int64)
                ins = (pr >= 0) & (pr < ps_h) & (pc >= 0) & (pc < ps_w)
                have = np.zeros(n, bool)
                have[ins] = av[pr[ins], pc[ins]]
                is_lt, is_ge = np.zeros(n, bool), np.zeros(n, bool)
                is_lt[have] = lt[pr[have], pc[have]]
                is_ge[have] = ge[pr[have], pc[have]]
                all_lt &= is_lt | ~have
                all_ge &= is_ge | ~have
                any_lt |= is_lt
                any_ge |= is_ge

            def classify(sr, sc_):
                never_s2, always_s2 = q_max[sr, sc_] < 3000, q_min[sr, sc_] >= 3000
                c = np.full(n, 4, np.uint8)
                c[(ndvi >= 3000) & all_ge & always_s2] = 3
                c[(ndvi < 2000) & all_lt & never_s2] = 2
                c[(ndwi > 1000) & (ndvi < 1000)] = 1
                return c, never_s2, always_s2
            sr0, sc0 = (rows + 7) // 10, (cc + 7) // 10
            nom, never_s2, always_s2 = classify(sr0, sc0)
            different += int((nom != stored).sum())
            sr1 = np.clip(np.floor((rows + 7.5 - dy2) / 10).astype(np.int64), 0, s2_h - 1)
            sc1 = np.clip(np.floor((cc + 7.5 + dx2) / 10).astype(np.int64), 0, s2_w - 1)
            dis, _, _ = classify(sr1, sc1)
            changed += int((nom != dis).sum())
            np.add.at(tab_nom, (ty_of[ii], nom), 1)
            np.add.at(tab_dis, (ty_of[ii], dis), 1)
            np.add.at(geo_all, (geo, nom), 1)
            bare = ndvi < 1000
            np.add.at(geo_bare, (geo[bare], nom[bare]), 1)
            k4 = nom == 4
            s2_cross = (q_min[sr0, sc0] < 2000) & (q_max[sr0, sc0] >= 3000)
            ps_cross = any_lt & any_ge
            an["class4"] += int(k4.sum())
            an["class4_cross_planetscope"] += int((k4 & ps_cross).sum())
            an["class4_cross_sentinel2"] += int((k4 & s2_cross).sum())
            an["class4_cross_one_sensor"] += int((k4 & (ps_cross | s2_cross)).sum())
            low = (ndvi < 2000) & (nom != 1)
            an["neo_below_02_not_water"] += int(low.sum())
            an["of_which_never"] += int((low & (nom == 2)).sum())
            an["fail_planetscope_only"] += int((low & ~all_lt & never_s2).sum())
            an["fail_sentinel2_only"] += int((low & all_lt & ~never_s2).sum())
            an["fail_both"] += int((low & ~all_lt & ~never_s2).sum())
            high = ndvi >= 3000
            an["neo_from_03"] += int(high.sum())
            an["of_which_always"] += int((high & (nom == 3)).sum())
            an["always_fail_planetscope_only"] += int((high & ~all_ge & always_s2).sum())
            an["always_fail_sentinel2_only"] += int((high & all_ge & ~always_s2).sum())
            an["always_fail_both"] += int((high & ~all_ge & ~always_s2).sum())
    if different:
        raise SystemExit(f"{different} pixels differ between the stored raster and this recomputation")
    with open(outputs[0], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pixels_with", "geobia2017_class", "pixels"] + [f"share_{v}" for v in CLASSES.values()])
        for label, t in (("all inventory pixels", geo_all), ("Neo v3 NDVI < 0.1", geo_bare)):
            for g in sorted(GEO_NAMES):
                n = int(t[g].sum())
                if n:
                    w.writerow([label, GEO_NAMES[g], n] + [round(int(t[g, k]) / n, 5) for k in CLASSES])
    n4 = an["class4"]
    an["class4_share_cross_one_sensor"] = round(an["class4_cross_one_sensor"] / n4, 5)
    an["note"] = "cross = at least one observation with NDVI < 0.2 and at least one with NDVI >= 0.3 within one sensor (PlanetScope dates available at the pixel; Sentinel-2 quarters)"
    json.dump(an, open(outputs[1], "w"), indent=1)

    def shares(t):
        n = int(t.sum())
        return {v: round(int(t[k]) / n, 5) for k, v in CLASSES.items()}
    n_all = int(tab_nom.sum())
    disp.update({"convention": "S2 content at Neo position (x, y) lies at (x + dx, y + dy), the sign of runs_itareo/ugi/b4_ps_offsets.json",
                 "effect_of_reading_s2_displaced": {"offset_used_m": [dx2, dy2], "inventory_pixels": n_all, "pixels_changing_class": changed,
                                                    "share_changing_class": round(changed / n_all, 5), "shares_nominal": shares(tab_nom.sum(0)),
                                                    "shares_displaced": shares(tab_dis.sum(0)),
                                                    "by_type": {ty: {"nominal": shares(tab_nom[i]), "displaced": shares(tab_dis[i])} for i, ty in enumerate(types)}},
                 "stored_raster_recomputed": {"pixels": n_all, "different": different}, "seconds": round(time.time() - t0, 1)})
    json.dump(disp, open(outputs[2], "w"), indent=1)
    print(json.dumps(an, indent=1))
    print(json.dumps({k: v for k, v in disp["effect_of_reading_s2_displaced"].items() if k != "by_type"}, indent=1))
    print("stored raster recomputed, different pixels:", different, "seconds", round(time.time() - t0))


if __name__ == "__main__":
    main()
