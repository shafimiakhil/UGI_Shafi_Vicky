"""U0.3 (CLAUDE.md 13, 12.2): register each Pléiades Neo 2022 strip to the Pléiades 2017 geometry, then re-mosaic with
one resampling per strip.

  --measure   each strip warped alone to GRID_1M (bilinear, R, G, B, NIR; cloud polygons set to nodata) into
              derived/pneo2022_registered/measure/ (measurement copies, never used as data); phase correlation
              (itareo.phasecorr via scripts/a7_coregistration.correlate: valid >= 0.9, stable >= 0.2, quality >= 2) of
              1 km blocks on a 500 m stride against the corrected harmonised Pléiades 2017 GRID_1M product
              (mean of R, G, B); per strip: median displacement of the Neo content relative to Pléiades 2017 (m east,
              m north), IQR, blocks, and a plane fit (m per km) as a trend check. Also the pairwise offsets between
              strips where two measurement copies overlap (before registration).
  --mosaic    scripts/a6_pneo2022.run_warp with each strip's source geotransform (EPSG:4326 VRT origin) and cloud
              polygons translated by minus its median displacement, so every strip is still resampled once, from its
              DIMAP tiles to GRID_1M; same priority order (lowest incidence first), cloud rule and outputs, written to
              derived/pneo2022_registered/.
  --check     the registered mosaic against Pléiades 2017 per strip (blocks dominated >= 95 % by one strip): residual.
Outputs: derived/pneo2022_registered/registration.json, blocks_measure.csv, blocks_check.csv, warp_log.json.
Run from work_ms/.
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from shapely.affinity import translate
from shapely.geometry import mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a6_pneo2022 as A6                       # noqa: E402
import a7_coregistration as A7                 # noqa: E402
from itareo.grid import GRID_1M                # noqa: E402

OUT = A6.W / "derived" / "pneo2022_registered"
MEAS = OUT / "measure"
PLE = A6.W / "derived/pleiades2017/pleiades2017_toa_hpf_grid1m_harmonised.tif"
BLOCK, STRIDE, STRIP_SHARE = 1000, 500, 0.95
LAT0 = -31.4                                   # metres per degree at the extent's latitude (translations only)
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))
M_PER_DEG_LAT = 110574.0


def ple_block(r0, c0, n):
    a = A7.read_window(PLE, GRID_1M, r0, c0, n, n, [1, 2, 3, 4], 1e-4, 65535)
    return (*A7.vis_ndvi(a, [0, 1, 2], 0, 3), np.zeros((n, n), bool))


def neo_block(path, r0, c0, n):
    a = A7.read_window(path, GRID_1M, r0, c0, n, n, [1, 2, 3, 4], 1e-4, 65535)
    return (*A7.vis_ndvi(a, [0, 1, 2], 0, 3), np.zeros((n, n), bool))


def stats(rows):
    if not rows:
        return {"blocks": 0}
    dx = np.array([r["dx_m"] for r in rows])
    dy = np.array([r["dy_m"] for r in rows])
    cx = np.array([r["block_col"] + BLOCK / 2 for r in rows]) / 1000.0
    cy = -np.array([r["block_row"] + BLOCK / 2 for r in rows]) / 1000.0
    out = {"blocks": len(rows), "dx_east_m_median": round(float(np.median(dx)), 3),
           "dy_north_m_median": round(float(np.median(dy)), 3),
           "dx_iqr": [round(float(v), 3) for v in np.percentile(dx, [25, 75])],
           "dy_iqr": [round(float(v), 3) for v in np.percentile(dy, [25, 75])]}
    if len(rows) >= 6:
        A = np.column_stack([np.ones_like(cx), cx, cy])
        out["plane_dx_m_per_km_east_north"] = [round(float(v), 3) for v in np.linalg.lstsq(A, dx, rcond=None)[0][1:]]
        out["plane_dy_m_per_km_east_north"] = [round(float(v), 3) for v in np.linalg.lstsq(A, dy, rcond=None)[0][1:]]
    return out


def measure():
    t0 = time.time()
    A6.RUN = OUT
    MEAS.mkdir(parents=True, exist_ok=True)
    paths = {}
    for strip in A6.STRIPS:
        vrt, lat = A6.build_vrt(strip)
        clouds, _ = A6.cloud_polygons(strip, lat)
        p = MEAS / f"measure_{strip}_grid1m.tif"
        paths[strip] = p
        if p.exists():                                  # measurement copies are reused when present
            continue
        prof = GRID_1M.profile(count=4, dtype="uint16", nodata=A6.ND16, predictor=2, **A6.GTIFF)
        with rasterio.open(vrt) as src, rasterio.open(p, "w", **prof) as d:
            for r0 in range(0, GRID_1M.height, 2048):
                nr = min(2048, GRID_1M.height - r0)
                tr = from_origin(GRID_1M.left, GRID_1M.top - r0, 1.0, 1.0)
                buf = np.full((4, nr, GRID_1M.width), np.nan, np.float32)
                reproject(rasterio.band(src, [1, 2, 3, 4]), buf, dst_transform=tr, dst_crs=GRID_1M.crs,
                          resampling=Resampling.bilinear, src_nodata=A6.ND_DN, dst_nodata=np.nan, num_threads=8)
                ok = np.isfinite(buf).all(0)
                if clouds:
                    ok &= ~features.rasterize([(g, 1) for g in clouds], out_shape=ok.shape, transform=tr, fill=0,
                                              dtype="uint8").astype(bool)
                out = np.where(ok[None], np.clip(np.rint(buf), 0, A6.ND16 - 1), A6.ND16).astype(np.uint16)
                d.write(out, window=Window(0, r0, GRID_1M.width, nr))
        paths[strip] = p
        print(strip, "measurement copy", round(time.time() - t0), "s", flush=True)
    rows = []
    for r0 in range(0, GRID_1M.height - BLOCK + 1, STRIDE):
        for c0 in range(0, GRID_1M.width - BLOCK + 1, STRIDE):
            pa = ple_block(r0, c0, BLOCK)
            ple_ok = np.isfinite(pa[0]).mean() >= A7.VALID_MIN
            nb = {s: neo_block(p, r0, c0, BLOCK) for s, p in paths.items()}
            for s, b in nb.items():
                if not ple_ok or np.isfinite(b[0]).mean() < A7.VALID_MIN:
                    continue
                rows.append(A7.correlate({"pair": f"pleiades2017->{s}", "strip": s, "block_row": r0, "block_col": c0},
                                         *pa, *b, 1.0))
            ss = [s for s, b in nb.items() if np.isfinite(b[0]).mean() >= A7.VALID_MIN]
            for i, s1 in enumerate(ss):
                for s2 in ss[i + 1:]:
                    rows.append(A7.correlate({"pair": f"{s1}->{s2}", "strip": f"{s1}|{s2}", "block_row": r0,
                                              "block_col": c0}, *nb[s1], *nb[s2], 1.0))
    write_rows(OUT / "blocks_measure.csv", rows)
    used = [r for r in rows if r.get("used")]
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "reference": str(PLE.relative_to(A6.W)),
           "block_px": BLOCK, "stride_px": STRIDE,
           "displacement_of_neo_relative_to_pleiades2017": {s: stats([r for r in used if r["pair"] == f"pleiades2017->{s}"])
                                                           for s in A6.STRIPS},
           "pairwise_before": {p: stats([r for r in used if r["pair"] == p]) for p in sorted({r["pair"] for r in used})
                               if "pleiades" not in p},
           "seconds": round(time.time() - t0)}
    (OUT / "registration.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def resolve_shifts(reg, min_blocks=5):
    """Displacement of each strip relative to Pléiades 2017: measured directly where >= min_blocks blocks exist;
    otherwise through the strip pair with the most blocks to a directly measured strip (pair a->b gives p_b - p_a)."""
    direct = {s: v for s, v in reg["displacement_of_neo_relative_to_pleiades2017"].items()
              if v.get("blocks", 0) >= min_blocks}
    out = {s: (v["dx_east_m_median"], v["dy_north_m_median"], "direct") for s, v in direct.items()}
    for s in A6.STRIPS:
        if s in out:
            continue
        best = None
        for pair, v in reg["pairwise_before"].items():
            a, b = pair.split("->")
            if v.get("blocks", 0) < min_blocks or s not in (a, b):
                continue
            ref = b if a == s else a
            if ref not in direct:
                continue
            if best is None or v["blocks"] > best[0]:
                sign = 1 if a == s else -1                 # p_s = p_ref - (p_ref - p_s)
                best = (v["blocks"], ref, pair, sign * v["dx_east_m_median"], sign * v["dy_north_m_median"])
        if best is None:
            raise SystemExit(f"strip {s}: no direct measurement and no pair to a measured strip")
        _, ref, pair, ddx, ddy = best
        out[s] = (round(out[ref][0] - ddx, 3), round(out[ref][1] - ddy, 3), f"via {ref} ({pair}, {best[0]} blocks)")
    return out


def mosaic():
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    res = resolve_shifts(reg)
    reg["applied_displacements"] = {s: {"dx_east_m": v[0], "dy_north_m": v[1], "via": v[2]} for s, v in res.items()}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    shifts = {s: v[:2] for s, v in res.items()}
    saved = (A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons)
    A6.RUN = OUT
    A6.OUT = A6.W / "derived"        # A6 logs paths relative to OUT; the common parent of pneo2022/ and pneo2022_registered/
    orig_vrt, orig_cloud = A6.build_vrt, A6.cloud_polygons

    def cloud_polygons(strip, lattice):
        geoms, info = orig_cloud(strip, lattice)
        dx, dy = shifts[strip]
        return [mapping(translate(shape(g), xoff=-dx, yoff=-dy)) for g in geoms], info

    A6.build_vrt, A6.cloud_polygons = registered_vrt_builder(orig_vrt, shifts), cloud_polygons
    try:
        A6.run_warp()
    finally:
        A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons = saved


def registered_vrt_builder(orig_vrt, shifts):
    """A6.build_vrt replacement: the strip VRT (written in A6.RUN/vrt) with its geotransform translated by minus the
    strip's displacement (shifts[strip] = (dx east m, dy north m))."""
    def build_vrt(strip):
        vrt, lat = orig_vrt(strip)
        dx, dy = shifts[strip]
        ddeg_x, ddeg_y = -dx / M_PER_DEG_LON, -dy / M_PER_DEG_LAT       # move the content back by minus its offset
        txt = vrt.read_text(encoding="utf-8")
        m = re.search(r"<GeoTransform>([^<]+)</GeoTransform>", txt)
        g = [float(v) for v in m.group(1).split(",")]
        g[0] += ddeg_x
        g[3] += ddeg_y
        txt = txt.replace(m.group(0), "<GeoTransform>" + ", ".join(repr(float(v)) for v in g) + "</GeoTransform>")
        vrt.write_text(txt, encoding="utf-8")
        lat = dict(lat, x0=g[0], y0=g[3], applied_shift_m_east_north=[-dx, -dy], applied_shift_deg=[ddeg_x, ddeg_y])
        return vrt, lat
    return build_vrt


def restore_v1_vrts(strips=("1_2", "1_3")):
    """The first affine_13 run rebuilt derived/pneo2022_registered/vrt/1_2.vrt and 1_3.vrt with the unregistered
    geotransforms (warp_copy wrote into A6.RUN = OUT; fixed). The overwritten files and the two *_check.vrt files left
    there are moved to archive/2026-11-02/u03_vrt_overwritten/, and the registered VRTs are rebuilt with the applied
    translations; their origins are compared with the v1 warp_log.json lattices (the mosaic TIFFs were never touched)."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    shifts = {st: (v["dx_east_m"], v["dy_north_m"]) for st, v in reg["applied_displacements"].items()}
    arch = A6.W.parent / "archive/2026-11-02/u03_vrt_overwritten"
    wl = json.loads((OUT / "warp_log.json").read_text(encoding="utf-8"))["strips"]
    if any((arch / f"{st}.vrt").exists() for st in strips):
        # restored before: never move anything over the archived evidence; only verify the current VRTs
        for st in strips:
            g = [float(v) for v in re.search(r"<GeoTransform>([^<]+)</GeoTransform>",
                                              (OUT / "vrt" / f"{st}.vrt").read_text(encoding="utf-8")).group(1).split(",")]
            assert abs(g[0] - wl[st]["lattice"]["x0"]) < 1e-12 and abs(g[3] - wl[st]["lattice"]["y0"]) < 1e-12, st
        print("already restored; current VRT origins equal warp_log.json")
        return
    arch.mkdir(parents=True, exist_ok=True)
    for name in [f"{st}.vrt" for st in strips] + ["1_2_check.vrt", "1_3_check.vrt"]:
        f = OUT / "vrt" / name
        if f.exists():
            f.replace(arch / name)
    saved = A6.RUN
    A6.RUN = OUT
    try:
        build = registered_vrt_builder(A6.build_vrt, shifts)
        lat = {st: build(st)[1] for st in strips}
    finally:
        A6.RUN = saved
    out = {st: {"x0": lat[st]["x0"], "y0": lat[st]["y0"], "warp_log_x0": wl[st]["lattice"]["x0"],
                "warp_log_y0": wl[st]["lattice"]["y0"],
                "equal": abs(lat[st]["x0"] - wl[st]["lattice"]["x0"]) < 1e-12 and abs(lat[st]["y0"] - wl[st]["lattice"]["y0"]) < 1e-12}
           for st in strips}
    (arch / "restore_log.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    assert all(v["equal"] for v in out.values())


def check():
    sid_path = OUT / "pneo2022_strip_id_grid1m.tif"
    toa = OUT / "pneo2022_toa_grid1m.tif"
    rows = []
    with rasterio.open(sid_path) as s:
        for r0 in range(0, GRID_1M.height - BLOCK + 1, STRIDE):
            for c0 in range(0, GRID_1M.width - BLOCK + 1, STRIDE):
                sid = s.read(1, window=Window(c0, r0, BLOCK, BLOCK))
                vals, cnt = np.unique(sid[sid > 0], return_counts=True)
                if not vals.size or cnt.max() / sid.size < STRIP_SHARE:
                    continue
                strip = A7.NEO_STRIP_CODE[int(vals[np.argmax(cnt)])]
                rows.append(A7.correlate({"pair": f"pleiades2017->registered_{strip}", "strip": strip, "block_row": r0,
                                          "block_col": c0}, *ple_block(r0, c0, BLOCK), *neo_block(toa, r0, c0, BLOCK),
                                         1.0))
    write_rows(OUT / "blocks_check.csv", rows)
    used = [r for r in rows if r.get("used")]
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    reg["residual_after_registration"] = {s: stats([r for r in used if r["strip"] == s]) for s in A6.STRIPS}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(reg["residual_after_registration"], indent=1))


def check_s2():
    """Independent check of all four strips, including 1_3 and 1_6 (outside Pléiades 2017): the registered mosaic,
    averaged to the Sentinel-2 10 m lattice (GRID_1M offset 7 px), against the S2 SR median 2022 Q4 (mean of B2, B3, B4),
    on 2 km blocks (200 px) dominated >= 95 % by one strip; phase correlation as A7 (valid >= 0.9, stable >= 0.2,
    quality >= 2). The residual of a strip = its offset to S2 minus the mean offset of the directly measured strips."""
    s2p = A6.W.parent / "gee_exports/2026-10-28_nodata/s2_sr_median_2022Q4.tif"
    toa, sidp = OUT / "pneo2022_toa_grid1m.tif", OUT / "pneo2022_strip_id_grid1m.tif"
    rows = []
    bs = 200
    with rasterio.open(s2p) as s2, rasterio.open(toa) as nt, rasterio.open(sidp) as sp:
        assert abs(s2.transform.c - (GRID_1M.left - 7)) < 1e-6 and abs(s2.transform.f - (GRID_1M.top + 7)) < 1e-6
        for r0 in range(1, s2.height - bs, bs):                 # s2 row 1 starts at GRID_1M row 3 (10 - 7)
            for c0 in range(1, s2.width - bs, bs):
                g_r, g_c = r0 * 10 - 7, c0 * 10 - 7
                if g_r < 0 or g_c < 0 or g_r + bs * 10 > GRID_1M.height or g_c + bs * 10 > GRID_1M.width:
                    continue
                w1 = Window(g_c, g_r, bs * 10, bs * 10)
                sid = sp.read(1, window=w1)
                vals, cnt = np.unique(sid[sid > 0], return_counts=True)
                if not vals.size or cnt.max() / sid.size < STRIP_SHARE:
                    continue
                a = nt.read([1, 2, 3, 4], window=w1).astype(np.float64)
                a[:, (a == nt.nodata).any(0)] = np.nan
                a = a.reshape(4, bs, 10, bs, 10).mean(axis=(2, 4)) * 1e-4
                b = s2.read([1, 2, 3, 7], window=Window(c0, r0, bs, bs)).astype(np.float64)
                b[:, (b == s2.nodata).any(0)] = np.nan
                b *= 1e-4
                va, na = A7.vis_ndvi(a, [0, 1, 2], 0, 3)
                vb, nbv = A7.vis_ndvi(b, [0, 1, 2], 2, 3)
                strip = A7.NEO_STRIP_CODE[int(vals[np.argmax(cnt)])]
                rows.append(A7.correlate({"pair": f"s2_2022Q4->registered_{strip}", "strip": strip, "block_row": g_r,
                                          "block_col": g_c}, vb, nbv, np.zeros_like(vb, bool), va, na,
                                         np.zeros_like(va, bool), 10.0))
    write_rows(OUT / "blocks_check_s2.csv", rows)
    used = [r for r in rows if r.get("used")]
    st = {s: stats([r for r in used if r["strip"] == s]) for s in A6.STRIPS}
    direct = [s for s in ("1_2", "1_5") if st[s].get("blocks", 0) >= 5]
    ref = (np.mean([st[s]["dx_east_m_median"] for s in direct]), np.mean([st[s]["dy_north_m_median"] for s in direct]))
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    reg["check_against_s2_2022Q4"] = {
        "offset_of_registered_strip_relative_to_s2_m": st,
        "reference_mean_of_direct_strips": {"strips": direct, "dx_east_m": round(float(ref[0]), 3),
                                            "dy_north_m": round(float(ref[1]), 3)},
        "residual_relative_to_direct_strips_m": {s: ([round(st[s]["dx_east_m_median"] - ref[0], 3),
                                                      round(st[s]["dy_north_m_median"] - ref[1], 3)]
                                                     if st[s].get("blocks", 0) else None) for s in A6.STRIPS}}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(reg["check_against_s2_2022Q4"], indent=1))


def check_ref(name, ref_path, ref_bands, ref_nodata, factor, off_x, off_y, block_m=1000, stride_m=500):
    """Residual check of every strip against an independent reference on its own lattice (factor = reference pixel in
    GRID_1M px; off_x, off_y = GRID_1M column and row of the reference origin, may be negative): the registered
    mosaic is averaged factor x factor onto the reference lattice; 1 km blocks on a 500 m stride dominated >= 95 % by
    one strip; phase correlation (A7 criteria). Residual = strip offset minus the mean offset of 1_2 and 1_5."""
    toa, sidp = OUT / "pneo2022_toa_grid1m.tif", OUT / "pneo2022_strip_id_grid1m.tif"
    bs, st_ = int(block_m // factor), int(stride_m // factor)
    rows = []
    with rasterio.open(ref_path) as rf, rasterio.open(toa) as nt, rasterio.open(sidp) as sp:
        for r0 in range(0, rf.height - bs, st_):
            for c0 in range(0, rf.width - bs, st_):
                g_r, g_c = r0 * factor + off_y, c0 * factor + off_x
                if g_r < 0 or g_c < 0 or g_r + bs * factor > GRID_1M.height or g_c + bs * factor > GRID_1M.width:
                    continue
                w1 = Window(g_c, g_r, bs * factor, bs * factor)
                sid = sp.read(1, window=w1)
                vals, cnt = np.unique(sid[sid > 0], return_counts=True)
                if not vals.size or cnt.max() / sid.size < STRIP_SHARE:
                    continue
                a = nt.read([1, 2, 3, 4], window=w1).astype(np.float64)
                a[:, (a == nt.nodata).any(0)] = np.nan
                a = a.reshape(4, bs, factor, bs, factor).mean(axis=(2, 4)) * 1e-4
                b = rf.read(ref_bands, window=Window(c0, r0, bs, bs)).astype(np.float64)
                b[:, (b == ref_nodata).any(0)] = np.nan
                b *= 1e-4
                va, na = A7.vis_ndvi(a, [0, 1, 2], 0, 3)
                vb, nbv = A7.vis_ndvi(b, [0, 1, 2], 2, 3)
                strip = A7.NEO_STRIP_CODE[int(vals[np.argmax(cnt)])]
                rows.append(A7.correlate({"pair": f"{name}->registered_{strip}", "strip": strip, "block_row": g_r,
                                          "block_col": g_c}, vb, nbv, np.zeros_like(vb, bool), va, na,
                                         np.zeros_like(va, bool), float(factor)))
    write_rows(OUT / f"blocks_check_{name}.csv", rows)
    used = [r for r in rows if r.get("used")]
    st = {s: stats([r for r in used if r["strip"] == s]) for s in A6.STRIPS}
    direct = [s for s in ("1_2", "1_5") if st[s].get("blocks", 0) >= 5]
    ref = (np.mean([st[s]["dx_east_m_median"] for s in direct]), np.mean([st[s]["dy_north_m_median"] for s in direct]))
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    reg[f"check_against_{name}_1km"] = {
        "offset_of_registered_strip_relative_to_reference_m": st,
        "reference_mean_of_direct_strips": {"strips": direct, "dx_east_m": round(float(ref[0]), 3),
                                            "dy_north_m": round(float(ref[1]), 3)},
        "residual_relative_to_direct_strips_m": {s: ([round(st[s]["dx_east_m_median"] - ref[0], 3),
                                                      round(st[s]["dy_north_m_median"] - ref[1], 3)]
                                                     if st[s].get("blocks", 0) else None) for s in A6.STRIPS}}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(name, json.dumps(reg[f"check_against_{name}_1km"]["residual_relative_to_direct_strips_m"]),
          {s: v.get("blocks", 0) for s, v in st.items()}, flush=True)


def warp_copy(strip, gt_fn, path):
    """Measurement copy of one strip on GRID_1M (bilinear, R, G, B, NIR, clouds as nodata), with its VRT geotransform
    replaced by gt_fn(gt) (6-tuple GDAL order)."""
    saved = A6.RUN
    A6.RUN = Path(path).parent          # VRTs in <measurement folder>/vrt, never in the registered mosaic's vrt/
    try:
        vrt, lat = A6.build_vrt(strip)
    finally:
        A6.RUN = saved
    txt = vrt.read_text(encoding="utf-8")
    m = re.search(r"<GeoTransform>([^<]+)</GeoTransform>", txt)
    g = gt_fn([float(v) for v in m.group(1).split(",")])
    vrt2 = vrt.with_name(f"{Path(path).stem}.vrt")
    vrt2.write_text(txt.replace(m.group(0), "<GeoTransform>" + ", ".join(repr(float(v)) for v in g) + "</GeoTransform>"),
                    encoding="utf-8")
    copy_from_vrt(vrt2, path)
    return g


def copy_from_vrt(vrt2, path):
    """Measurement copy on GRID_1M (bilinear, R, G, B, NIR) of a strip VRT (geotransform or GCPs)."""
    prof = GRID_1M.profile(count=4, dtype="uint16", nodata=A6.ND16, predictor=2, **A6.GTIFF)
    with rasterio.open(vrt2) as src, rasterio.open(path, "w", **prof) as d:
        for r0 in range(0, GRID_1M.height, 2048):
            nr = min(2048, GRID_1M.height - r0)
            tr = from_origin(GRID_1M.left, GRID_1M.top - r0, 1.0, 1.0)
            buf = np.full((4, nr, GRID_1M.width), np.nan, np.float32)
            reproject(rasterio.band(src, [1, 2, 3, 4]), buf, dst_transform=tr, dst_crs=GRID_1M.crs,
                      resampling=Resampling.bilinear, src_nodata=A6.ND_DN, dst_nodata=np.nan, num_threads=8)
            ok = np.isfinite(buf).all(0)
            out = np.where(ok[None], np.clip(np.rint(buf), 0, A6.ND16 - 1), A6.ND16).astype(np.uint16)
            d.write(out, window=Window(0, r0, GRID_1M.width, nr))


def affine_13():
    """Strip 1_3 (CLAUDE.md 13.2): an affine model of the 1_3 -> 1_2 pair displacement over their overlap
    (blocks_measure.csv, used blocks; robust least squares, 3 x MAD, 5 passes), turned into an affine correction of the
    1_3 geotransform that makes 1_3 agree with the registered 1_2 everywhere: D13(x, y) = t12 - d_pair(x, y), content
    moved by -D13. Checked against Sentinel-2 2022 Q4 at 10 m on identical 1 km blocks for the corrected 1_3 and the
    registered 1_2 (measurement copies), and by columns as in check_ref."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    t12 = np.array([reg["applied_displacements"]["1_2"]["dx_east_m"], reg["applied_displacements"]["1_2"]["dy_north_m"]])
    t13 = np.array([reg["applied_displacements"]["1_3"]["dx_east_m"], reg["applied_displacements"]["1_3"]["dy_north_m"]])
    rows = [r for r in csv.DictReader(open(OUT / "blocks_measure.csv")) if r["pair"] == "1_3->1_2" and r["used"] == "True"]
    X = np.array([GRID_1M.left + float(r["block_col"]) + BLOCK / 2 for r in rows])
    Y = np.array([GRID_1M.top - float(r["block_row"]) - BLOCK / 2 for r in rows])
    D = np.column_stack([[float(r["dx_m"]) for r in rows], [float(r["dy_m"]) for r in rows]])
    x0, y0 = float(np.mean(X)), float(np.mean(Y))
    A = np.column_stack([np.ones_like(X), (X - x0) / 1000, (Y - y0) / 1000])
    coefs, keep_all = [], np.ones(len(X), bool)
    for k in range(2):
        keep = np.ones(len(X), bool)
        for _ in range(5):
            c = np.linalg.lstsq(A[keep], D[keep, k], rcond=None)[0]
            res = D[:, k] - A @ c
            mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
            nk = np.abs(res) <= max(3 * mad, 1e-6)
            if (nk == keep).all():
                break
            keep = nk
        coefs.append(c)
        keep_all &= keep
    fit_res = np.concatenate([D[keep_all, k] - A[keep_all] @ coefs[k] for k in range(2)])
    # correction in metres at a map point p: content displacement D13(p) = t12 - d_pair(p); new position p - D13(p)
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GRID_1M.crs, always_xy=True)
    to_ll = Transformer.from_crs(GRID_1M.crs, "EPSG:4326", always_xy=True)

    def corrected(lon, lat):
        x, y = to_m.transform(lon, lat)
        a = np.array([1.0, (x - x0) / 1000, (y - y0) / 1000])
        d13 = t12 - np.array([a @ coefs[0], a @ coefs[1]])
        return to_ll.transform(x - d13[0], y - d13[1])

    def gt_fn(g):
        # the corrected position of three reference pixel corners defines the new affine geotransform
        ref = [(0.0, 0.0), (10000.0, 0.0), (0.0, 10000.0)]
        P = [(g[0] + c * g[1] + r * g[2], g[3] + c * g[4] + r * g[5]) for c, r in ref]
        Q = [corrected(*p) for p in P]
        a1 = (Q[1][0] - Q[0][0]) / 10000.0
        a2 = (Q[2][0] - Q[0][0]) / 10000.0
        b1 = (Q[1][1] - Q[0][1]) / 10000.0
        b2 = (Q[2][1] - Q[0][1]) / 10000.0
        return [Q[0][0], a1, a2, Q[0][1], b1, b2]

    MR = OUT / "measure_registered"
    MR.mkdir(parents=True, exist_ok=True)
    g13 = warp_copy("1_3", gt_fn, MR / "check_1_3_affine_grid1m.tif")

    def translate_fn(dx, dy):
        return lambda g: [g[0] - dx / M_PER_DEG_LON, g[1], g[2], g[3] - dy / M_PER_DEG_LAT, g[4], g[5]]
    warp_copy("1_2", translate_fn(*t12), MR / "check_1_2_registered_grid1m.tif")
    warp_copy("1_3", translate_fn(*t13), MR / "check_1_3_translation_grid1m.tif")
    s2p = A6.W.parent / "gee_exports/2026-10-28_nodata/s2_sr_median_2022Q4.tif"
    paired = {"affine": [], "translation": []}
    with rasterio.open(s2p) as s2:
        for r0 in range(1, s2.height - 100, 50):
            for c0 in range(1, s2.width - 100, 50):
                g_r, g_c = r0 * 10 - 7, c0 * 10 - 7
                if g_r < 0 or g_c < 0 or g_r + 1000 > GRID_1M.height or g_c + 1000 > GRID_1M.width:
                    continue
                b = s2.read([1, 2, 3, 7], window=Window(c0, r0, 100, 100)).astype(np.float64)
                b[:, (b == s2.nodata).any(0)] = np.nan
                vb, nbv = A7.vis_ndvi(b * 1e-4, [0, 1, 2], 2, 3)
                res_ = {}
                for name, p in (("r12", MR / "check_1_2_registered_grid1m.tif"), ("affine", MR / "check_1_3_affine_grid1m.tif"),
                                ("translation", MR / "check_1_3_translation_grid1m.tif")):
                    with rasterio.open(p) as src:
                        a = src.read([1, 2, 3, 4], window=Window(g_c, g_r, 1000, 1000)).astype(np.float64)
                    a[:, (a == A6.ND16).any(0)] = np.nan
                    if np.isfinite(a[0]).mean() < 0.9:
                        break
                    a = a.reshape(4, 100, 10, 100, 10).mean(axis=(2, 4)) * 1e-4
                    va, na = A7.vis_ndvi(a, [0, 1, 2], 0, 3)
                    res_[name] = A7.correlate({}, vb, nbv, np.zeros_like(vb, bool), va, na, np.zeros_like(va, bool), 10.0)
                if len(res_) < 3 or not all(v.get("used") for v in res_.values()):
                    continue
                for k in ("affine", "translation"):
                    paired[k].append({"block_row": g_r, "block_col": g_c,
                                      "dx_m": res_[k]["dx_m"] - res_["r12"]["dx_m"], "dy_m": res_[k]["dy_m"] - res_["r12"]["dy_m"]})
    out = {"pair_blocks_used": int(keep_all.sum()), "pair_blocks": len(rows), "centre_m": [x0, y0],
           "pair_fit_coef_dx_m_per_km": [round(float(v), 4) for v in coefs[0]],
           "pair_fit_coef_dy_m_per_km": [round(float(v), 4) for v in coefs[1]],
           "pair_fit_rms_m": round(float(np.sqrt(np.mean(fit_res ** 2))), 3),
           "affine_geotransform_1_3": g13}
    for k, v in paired.items():
        if v:
            dx = np.array([r["dx_m"] for r in v])
            dy = np.array([r["dy_m"] for r in v])
            out[f"s2_paired_residual_{k}"] = {"blocks": len(v), "dx_median_m": round(float(np.median(dx)), 3),
                                               "dy_median_m": round(float(np.median(dy)), 3),
                                               "magnitude_m": round(float(np.hypot(np.median(dx), np.median(dy))), 3),
                                               "dx_iqr": [round(float(q), 3) for q in np.percentile(dx, [25, 75])],
                                               "dy_iqr": [round(float(q), 3) for q in np.percentile(dy, [25, 75])]}
        else:
            out[f"s2_paired_residual_{k}"] = {"blocks": 0}
    write_rows(OUT / "blocks_check_1_3_affine.csv", [dict(r, model=k) for k, v in paired.items() for r in v] or [{"none": 1}])
    out["rule_s2_gate"] = "S2-paired residual magnitude below 0.5 m (cannot decide: 0.5 m steps; see strip13_check())"
    out["s2_gate_pass"] = bool(out["s2_paired_residual_affine"].get("magnitude_m", 99) < 0.5)
    prev = reg.get("strip_1_3_affine", {})
    for k in ("apply", "decision", "apply_superseded_by"):     # set only by strip13_check(); kept on a re-run
        if k in prev:
            out[k] = prev[k]
    out.setdefault("apply", False)
    reg["strip_1_3_affine"] = out
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def fit_linear_mad(A, D):
    """The affine_13() fit: per component, least squares with 3 x MAD rejection, 5 passes; returns (coefs, keep_all)."""
    coefs, keep_all = [], np.ones(len(D), bool)
    for k in range(2):
        keep = np.ones(len(D), bool)
        for _ in range(5):
            c = np.linalg.lstsq(A[keep], D[keep, k], rcond=None)[0]
            res = D[:, k] - A @ c
            mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
            nk = np.abs(res) <= max(3 * mad, 1e-6)
            if (nk == keep).all():
                break
            keep = nk
        coefs.append(c)
        keep_all &= keep
    return coefs, keep_all


def fit_gross(A, D, limit=1.0):
    """Least squares per component, then one refit without blocks whose residual exceeds `limit` m (gross errors only)."""
    coefs = []
    for k in range(2):
        c = np.linalg.lstsq(A, D[:, k], rcond=None)[0]
        keep = np.abs(D[:, k] - A @ c) <= limit
        coefs.append(np.linalg.lstsq(A[keep], D[keep, k], rcond=None)[0])
    return coefs


def cv_models(X, Y, D, x0, y0, bin_m=3000, folds=5):
    """Translation, affine and quadratic_y models of the 1_3 -> 1_2 pair displacement D at block centres (X, Y): residual
    of the corrected 1_3 relative to the registered 1_2 (= model - D) on all blocks and by cross-validation (5 folds of whole
    block rows contiguous in northing; training without the blocks that overlap a test block), medians per 3 km band.
    Returns (models summary, per-band rows, fold sizes)."""
    def design(x, y, quad):
        cols = [np.ones_like(x), (x - x0) / 1000, (y - y0) / 1000]
        return np.column_stack(cols + ([((y - y0) / 1000) ** 2] if quad else []))

    def fit(model, idx):
        if model == "translation":
            med = np.median(D[idx], axis=0)
            return lambda x, y: np.tile(med, (len(x), 1))
        if model == "affine":
            c, _ = fit_linear_mad(design(X[idx], Y[idx], False), D[idx])
            return lambda x, y: np.column_stack([design(x, y, False) @ c[0], design(x, y, False) @ c[1]])
        c = fit_gross(design(X[idx], Y[idx], True), D[idx])
        return lambda x, y: np.column_stack([design(x, y, True) @ c[0], design(x, y, True) @ c[1]])

    all_idx = np.arange(len(D))
    edges = np.arange(math.floor(Y.min() / bin_m) * bin_m, Y.max() + bin_m, bin_m)
    rows_y = np.unique(Y)
    cum = np.cumsum([int(np.sum(Y == y)) for y in rows_y])
    row_fold = np.minimum((cum - 1) * folds // len(Y), folds - 1)
    fold_of = row_fold[np.searchsorted(rows_y, Y)]
    assert len(set(fold_of)) == folds
    models, bins_rows = {}, []
    for model in ("translation", "affine", "quadratic_y"):
        res = fit(model, all_idx)(X, Y) - D
        cv = np.zeros_like(D)
        excluded = 0
        for k in range(folds):
            te = all_idx[fold_of == k]
            near = ((np.abs(X[:, None] - X[te][None]) < BLOCK) & (np.abs(Y[:, None] - Y[te][None]) < BLOCK)).any(1)
            tr = all_idx[(fold_of != k) & ~near]
            excluded += int(np.sum((fold_of != k) & near))
            cv[te] = fit(model, tr)(X[te], Y[te]) - D[te]
        per_bin = []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            m = (Y >= b0) & (Y < b1)
            if not m.any():
                continue
            mdx, mdy = float(np.median(res[m, 0])), float(np.median(res[m, 1]))
            cdx, cdy = float(np.median(cv[m, 0])), float(np.median(cv[m, 1]))
            per_bin.append({"model": model, "northing_from": int(b0), "northing_to": int(b1), "blocks": int(m.sum()),
                            "dx_13_rel_12_median_m": round(mdx, 3), "dy_13_rel_12_median_m": round(mdy, 3),
                            "magnitude_m": round(math.hypot(mdx, mdy), 3),
                            "cv_dx_13_rel_12_median_m": round(cdx, 3), "cv_dy_13_rel_12_median_m": round(cdy, 3),
                            "cv_magnitude_m": round(math.hypot(cdx, cdy), 3)})
        bins_rows += per_bin
        models[model] = {"rms_all_blocks_m": round(float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), 3),
                         "rms_cv_m": round(float(np.sqrt(np.mean(np.sum(cv ** 2, axis=1)))), 3),
                         "max_bin_median_magnitude_m": max(r["magnitude_m"] for r in per_bin),
                         "max_bin_cv_median_magnitude_m": max(r["cv_magnitude_m"] for r in per_bin),
                         "cv_training_blocks_excluded_by_buffer_total": excluded}
    return models, bins_rows, [int(np.sum(fold_of == k)) for k in range(folds)]


def lowpass_table():
    """The 1_3 -> 1_2 pair table measured with the low-pass estimate on the unregistered copies (used blocks of
    pair_consistency_13_12_lowpass_blocks.csv): (X, Y, D)."""
    pc = [r for r in csv.DictReader(open(OUT / "pair_consistency_13_12_lowpass_blocks.csv")) if r["remeasured_used"] == "True"]
    X = np.array([GRID_1M.left + float(r["block_col"]) + BLOCK / 2 for r in pc])
    Y = np.array([GRID_1M.top - float(r["block_row"]) - BLOCK / 2 for r in pc])
    D = np.column_stack([[float(r["remeasured_lp_dx_m"]) for r in pc], [float(r["remeasured_lp_dy_m"]) for r in pc]])
    return X, Y, D


def strip13_cv_lowpass(bin_m=3000, folds=5):
    """F1 gate (CLAUDE.md 13.4): the models of cv_models() on the low-pass pair table; v3 is built only if the
    cross-validated worst 3 km band of the quadratic is below 0.5 m. No label enters. Writes strip13_check_lowpass.json
    and strip13_check_lowpass_bins.csv; records the decision in registration.json["strip_1_3_quadratic_lowpass"]."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    X, Y, D = lowpass_table()
    x0, y0 = reg["strip_1_3_affine"]["centre_m"]
    assert abs(x0 - X.mean()) < 1e-6 and abs(y0 - Y.mean()) < 1e-6
    models, bins_rows, fold_sizes = cv_models(X, Y, D, x0, y0, bin_m, folds)
    write_rows(OUT / "strip13_check_lowpass_bins.csv", bins_rows)
    worst = models["quadratic_y"]["max_bin_cv_median_magnitude_m"]
    corrected, info = quad13_model((X, Y, D))
    out = {"table": "pair_consistency_13_12_lowpass_blocks.csv (remeasured, low-pass 0.1 cycles per px)", "blocks": len(D),
           "cv_folds_contiguous_in_northing": folds, "cv_blocks_per_fold": fold_sizes,
           "cv_design": "whole block rows per fold; training excludes blocks overlapping a test block (as strip13_check)",
           "models": models, "quadratic_model": info,
           "rule_13_4": "build v3 only if the cross-validated worst 3 km band of the quadratic refitted on the low-pass pair "
                        "table is below 0.5 m",
           "quadratic_cv_worst_band_m": worst, "build_v3": bool(worst < 0.5)}
    (OUT / "strip13_check_lowpass.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    reg["strip_1_3_quadratic_lowpass"] = {**info, "check": "strip13_check_lowpass.json", "cv_worst_band_m": worst,
                                          "pass": out["build_v3"], "estimator": "low-pass 0.1 cycles per px (CLAUDE.md 13.4)"}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def strip13_check(bin_m=3000, folds=5):
    """Strip 1_3 (CLAUDE.md 13.2), the decisive check of the affine fit, on the Neo-to-Neo overlap itself.
    The Sentinel-2 gate of affine_13() cannot decide: at 10 m with itareo.phasecorr's upsampling of 20 its shifts come in
    0.5 m steps, both medians were exactly 0, and it has no stable blocks in the north of 1_3. Here, on every used block of
    the 1_3 -> 1_2 pair (blocks_measure.csv, 1 km blocks at a 500 m stride, 0.05 m steps), the residual displacement of the
    corrected 1_3 relative to the registered 1_2 (m east, m north; = model - measured pair displacement, because the pair
    value is the displacement of 1_2 relative to 1_3) is computed for three models of the pair displacement:
      translation  the pair median (what v1 applies: t13 = t12 - median);
      affine       the affine_13() fit (linear in x and y, 3 x MAD rejection, the coefficients stored in registration.json);
      quadratic_y  linear in x and y plus y^2, with gross-error rejection only (|residual| > 1 m), for information.
    Reported: medians per 3 km band of northing (all blocks), the RMS over all blocks, and residuals predicted by
    cross-validation: 5 folds of whole block rows, contiguous in northing, each predicted from a fit on the other folds
    without the blocks that overlap a test block (1 km blocks at a 500 m stride overlap their neighbours). Rule (13.2):
    the affine is applied only if its median residual is below 0.5 m in every band. Writes strip13_check.json and
    strip13_check_bins.csv in OUT and updates registration.json["strip_1_3_affine"] (apply, decision). The first version
    (archive/2026-11-02/u03_strip13_check_v1/) split block rows between folds, had no buffer and reported the opposite sign;
    its in-sample numbers are unchanged."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    aff = reg["strip_1_3_affine"]
    rows = [r for r in csv.DictReader(open(OUT / "blocks_measure.csv")) if r["pair"] == "1_3->1_2" and r["used"] == "True"]
    X = np.array([GRID_1M.left + float(r["block_col"]) + BLOCK / 2 for r in rows])
    Y = np.array([GRID_1M.top - float(r["block_row"]) - BLOCK / 2 for r in rows])
    D = np.column_stack([[float(r["dx_m"]) for r in rows], [float(r["dy_m"]) for r in rows]])
    x0, y0 = aff["centre_m"]
    assert abs(x0 - X.mean()) < 1e-6 and abs(y0 - Y.mean()) < 1e-6

    def design(x, y, quad):
        cols = [np.ones_like(x), (x - x0) / 1000, (y - y0) / 1000]
        return np.column_stack(cols + ([((y - y0) / 1000) ** 2] if quad else []))

    def fit(model, idx):
        if model == "translation":
            med = np.median(D[idx], axis=0)
            return lambda x, y: np.tile(med, (len(x), 1))
        if model == "affine":
            c, _ = fit_linear_mad(design(X[idx], Y[idx], False), D[idx])
            return lambda x, y: np.column_stack([design(x, y, False) @ c[0], design(x, y, False) @ c[1]])
        c = fit_gross(design(X[idx], Y[idx], True), D[idx])
        return lambda x, y: np.column_stack([design(x, y, True) @ c[0], design(x, y, True) @ c[1]])

    all_idx = np.arange(len(D))
    c_aff, keep_aff = fit_linear_mad(design(X, Y, False), D)
    stored = [aff["pair_fit_coef_dx_m_per_km"], aff["pair_fit_coef_dy_m_per_km"]]
    assert all(abs(round(float(c_aff[k][j]), 4) - stored[k][j]) < 1e-9 for k in range(2) for j in range(3)), "refit differs"
    t12 = np.array([reg["applied_displacements"]["1_2"]["dx_east_m"], reg["applied_displacements"]["1_2"]["dy_north_m"]])
    t13 = np.array([reg["applied_displacements"]["1_3"]["dx_east_m"], reg["applied_displacements"]["1_3"]["dy_north_m"]])
    med_all = np.median(D, axis=0)
    edges = np.arange(math.floor(Y.min() / bin_m) * bin_m, Y.max() + bin_m, bin_m)
    rows_y = np.unique(Y)                                   # northings of the block rows, ascending
    cum = np.cumsum([int(np.sum(Y == y)) for y in rows_y])
    row_fold = np.minimum((cum - 1) * folds // len(Y), folds - 1)
    fold_of = row_fold[np.searchsorted(rows_y, Y)]          # whole block rows per fold, contiguous in northing
    assert len(set(fold_of)) == folds
    out = {"blocks": len(D), "blocks_kept_by_affine_fit": int(keep_aff.sum()), "centre_m": [x0, y0],
           "translation_pair_median_m": [round(float(v), 3) for v in med_all],
           "v1_translation_check_t12_minus_t13_m": [round(float(v), 3) for v in t12 - t13],
           "bin_m": bin_m, "cv_folds_contiguous_in_northing": folds,
           "cv_design": "whole block rows per fold; training excludes blocks overlapping a test block",
           "cv_blocks_per_fold": [int(np.sum(fold_of == k)) for k in range(folds)],
           "sign": "residuals are the displacement of the corrected 1_3 relative to the registered 1_2 (m east, m north)",
           "models": {}}
    bins_rows = []
    for model in ("translation", "affine", "quadratic_y"):
        pred = fit(model, all_idx)(X, Y)
        res = pred - D
        cv = np.zeros_like(D)
        excluded = 0
        for k in range(folds):
            te = all_idx[fold_of == k]
            near = ((np.abs(X[:, None] - X[te][None]) < BLOCK) & (np.abs(Y[:, None] - Y[te][None]) < BLOCK)).any(1)
            tr = all_idx[(fold_of != k) & ~near]
            excluded += int(np.sum((fold_of != k) & near))
            cv[te] = fit(model, tr)(X[te], Y[te]) - D[te]
        per_bin = []
        for b0, b1 in zip(edges[:-1], edges[1:]):
            m = (Y >= b0) & (Y < b1)
            if not m.any():
                continue
            mdx, mdy = float(np.median(res[m, 0])), float(np.median(res[m, 1]))
            cdx, cdy = float(np.median(cv[m, 0])), float(np.median(cv[m, 1]))
            row = {"model": model, "northing_from": int(b0), "northing_to": int(b1), "blocks": int(m.sum()),
                   "dx_13_rel_12_median_m": round(mdx, 3), "dy_13_rel_12_median_m": round(mdy, 3),
                   "magnitude_m": round(math.hypot(mdx, mdy), 3),
                   "cv_dx_13_rel_12_median_m": round(cdx, 3), "cv_dy_13_rel_12_median_m": round(cdy, 3),
                   "cv_magnitude_m": round(math.hypot(cdx, cdy), 3)}
            per_bin.append(row)
            bins_rows.append(row)
        out["models"][model] = {
            "rms_all_blocks_m": round(float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), 3),
            "rms_cv_m": round(float(np.sqrt(np.mean(np.sum(cv ** 2, axis=1)))), 3),
            "max_bin_median_magnitude_m": max(r["magnitude_m"] for r in per_bin),
            "max_bin_cv_median_magnitude_m": max(r["cv_magnitude_m"] for r in per_bin),
            "cv_training_blocks_excluded_by_buffer_total": excluded}
    write_rows(OUT / "strip13_check_bins.csv", bins_rows)
    worst = out["models"]["affine"]["max_bin_median_magnitude_m"]
    apply_ = bool(worst < 0.5)
    out["rule"] = "apply the affine only if its median residual against the registered 1_2 is below 0.5 m in every 3 km band"
    out["apply_affine"] = apply_
    out["decision"] = ("affine applied" if apply_ else
                       f"affine not applied: its worst band median is {worst} m (translation "
                       f"{out['models']['translation']['max_bin_median_magnitude_m']} m); strip 1_3 keeps the v1 translation "
                       "and held-out scores are reported per strip")
    (OUT / "strip13_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    if "rule" in aff:                                       # the Sentinel-2 gate's rule, kept under its own name
        aff["rule_s2_gate"] = aff.pop("rule")
    aff["apply"] = apply_
    aff["apply_superseded_by"] = "strip13_check.json (Neo-to-Neo check; the Sentinel-2 gate above is quantised in 0.5 m steps)"
    aff["decision"] = out["decision"]
    reg["strip_1_3_affine"] = aff
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def quad13_model(table=None):
    """The quadratic_y model of strip13_check() fitted on all used 1_3 -> 1_2 blocks (linear in x and y plus y^2, gross
    errors > 1 m rejected): coefficients of the pair displacement (m; per km about the centre), and the correction
    corrected(lon, lat) -> (lon, lat): content displacement D13 = t12 - d_pair(x, y), new position = p - D13."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    if table is None:                                   # the stored pair table (unweighted phase correlation)
        rows = [r for r in csv.DictReader(open(OUT / "blocks_measure.csv")) if r["pair"] == "1_3->1_2" and r["used"] == "True"]
        X = np.array([GRID_1M.left + float(r["block_col"]) + BLOCK / 2 for r in rows])
        Y = np.array([GRID_1M.top - float(r["block_row"]) - BLOCK / 2 for r in rows])
        D = np.column_stack([[float(r["dx_m"]) for r in rows], [float(r["dy_m"]) for r in rows]])
    else:
        X, Y, D = table
    x0, y0 = reg["strip_1_3_affine"]["centre_m"]
    A = np.column_stack([np.ones_like(X), (X - x0) / 1000, (Y - y0) / 1000, ((Y - y0) / 1000) ** 2])
    coefs = fit_gross(A, D)
    t12 = np.array([reg["applied_displacements"]["1_2"]["dx_east_m"], reg["applied_displacements"]["1_2"]["dy_north_m"]])
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GRID_1M.crs, always_xy=True)
    to_ll = Transformer.from_crs(GRID_1M.crs, "EPSG:4326", always_xy=True)

    def corrected(lon, lat):
        x, y = to_m.transform(np.asarray(lon, float), np.asarray(lat, float))
        a = np.stack([np.ones_like(x), (x - x0) / 1000, (y - y0) / 1000, ((y - y0) / 1000) ** 2])
        d13x, d13y = t12[0] - coefs[0] @ a, t12[1] - coefs[1] @ a
        return to_ll.transform(x - d13x, y - d13y)
    info = {"model": "pair displacement = c0 + c1 x' + c2 y' + c3 y'^2, x' = (x - x0)/1 km, y' = (y - y0)/1 km; "
                     "least squares, then refit without blocks with |residual| > 1 m",
            "blocks": len(D), "centre_m": [x0, y0], "coef_dx_m": [round(float(v), 6) for v in coefs[0]],
            "coef_dy_m": [round(float(v), 6) for v in coefs[1]], "t12_m": t12.tolist()}
    return corrected, info


def gcp_vrt(strip, corrected, folder, step=500, name=None, builder=None):
    """The strip VRT (A6.build_vrt, written in folder/vrt) with its geotransform replaced by a GCP list: pixel/line corners
    every `step` px mapped through corrected(). GDAL's polynomial GCP transformer (order 2, chosen automatically for
    >= 10 GCPs) carries the quadratic correction into the warp; the fit of the polynomial is checked here."""
    saved = A6.RUN
    A6.RUN = Path(folder)
    try:
        vrt, lat = (builder or A6.build_vrt)(strip)       # builder: the unpatched A6.build_vrt inside a mosaic
    finally:
        A6.RUN = saved
    txt = vrt.read_text(encoding="utf-8")
    m = re.search(r"<GeoTransform>([^<]+)</GeoTransform>", txt)
    g = [float(v) for v in m.group(1).split(",")]
    W_, H_ = lat["width"], lat["height"]
    cols = np.unique(np.r_[np.arange(0, W_, step), W_]).astype(float)
    lines = np.unique(np.r_[np.arange(0, H_, step), H_]).astype(float)
    C, L = np.meshgrid(cols, lines)
    lon = g[0] + C * g[1] + L * g[2]
    la = g[3] + C * g[4] + L * g[5]
    X, Y = corrected(lon.ravel(), la.ravel())
    from rasterio.crs import CRS
    wkt = CRS.from_epsg(4326).to_wkt().replace('"', "&quot;")
    gl = [f'  <GCPList Projection="{wkt}">']
    gl += [f'    <GCP Id="{i}" Pixel="{float(c)!r}" Line="{float(l)!r}" X="{float(x)!r}" Y="{float(y)!r}" Z="0"/>'
           for i, (c, l, x, y) in enumerate(zip(C.ravel(), L.ravel(), X, Y))]
    gl.append("  </GCPList>")
    txt = re.sub(r"\s*<SRS>.*?</SRS>", "", txt, flags=re.S)
    txt = txt.replace(m.group(0), "\n".join(gl).lstrip())
    out = vrt if name is None else vrt.with_name(name)
    out.write_text(txt, encoding="utf-8")
    # fidelity: GDAL's GCP polynomial against the exact correction at points between the GCPs
    from rasterio.control import GroundControlPoint
    from rasterio.transform import GCPTransformer
    gcps = [GroundControlPoint(row=l, col=c, x=float(x), y=float(y)) for c, l, x, y in zip(C.ravel(), L.ravel(), X, Y)]
    rng = np.random.default_rng(0)
    pc, pl = rng.uniform(0, W_, 2000), rng.uniform(0, H_, 2000)
    ex = corrected(g[0] + pc * g[1] + pl * g[2], g[3] + pc * g[4] + pl * g[5])
    with GCPTransformer(gcps) as gt_:
        px, py = gt_.xy(pl, pc, offset="ul")
    from pyproj import Transformer
    to_m = Transformer.from_crs("EPSG:4326", GRID_1M.crs, always_xy=True)
    ax, ay = to_m.transform(np.asarray(px), np.asarray(py))
    bx, by = to_m.transform(np.asarray(ex[0]), np.asarray(ex[1]))
    err = np.hypot(ax - bx, ay - by)
    return out, {"gcps": len(gcps), "gcp_step_px": step, "vrt_size": [W_, H_], "lattice": lat,
                 "polynomial_error_m": {"max": round(float(err.max()), 4), "p99": round(float(np.percentile(err, 99)), 4),
                                        "median": round(float(np.median(err)), 4), "points": 2000}}


def band_residuals(rows, model, bin_m=3000, kx="dx_13_rel_12_m", ky="dy_13_rel_12_m"):
    """Per 3 km band of northing: median residual of 1_3 relative to 1_2 (m) from image blocks."""
    Y = np.array([GRID_1M.top - r["block_row"] - BLOCK / 2 for r in rows])
    R = np.array([[r[kx], r[ky]] for r in rows])
    out = []
    if not len(rows):
        return out, None
    for b0 in np.arange(math.floor(Y.min() / bin_m) * bin_m, Y.max() + bin_m, bin_m):
        m = (Y >= b0) & (Y < b0 + bin_m)
        if not m.any():
            continue
        mdx, mdy = float(np.median(R[m, 0])), float(np.median(R[m, 1]))
        out.append({"model": model, "northing_from": int(b0), "northing_to": int(b0 + bin_m), "blocks": int(m.sum()),
                    "dx_13_rel_12_median_m": round(mdx, 3), "dy_13_rel_12_median_m": round(mdy, 3),
                    "magnitude_m": round(math.hypot(mdx, mdy), 3)})
    return out, max(r["magnitude_m"] for r in out)


def s2_paired(models, ref12):
    """The Sentinel-2 2022 Q4 paired check of affine_13() for any set of 1_3 copies (name -> path) against ref12."""
    s2p = A6.W.parent / "gee_exports/2026-10-28_nodata/s2_sr_median_2022Q4.tif"
    paired = {k: [] for k in models}
    with rasterio.open(s2p) as s2:
        for r0 in range(1, s2.height - 100, 50):
            for c0 in range(1, s2.width - 100, 50):
                g_r, g_c = r0 * 10 - 7, c0 * 10 - 7
                if g_r < 0 or g_c < 0 or g_r + 1000 > GRID_1M.height or g_c + 1000 > GRID_1M.width:
                    continue
                b = s2.read([1, 2, 3, 7], window=Window(c0, r0, 100, 100)).astype(np.float64)
                b[:, (b == s2.nodata).any(0)] = np.nan
                vb, nbv = A7.vis_ndvi(b * 1e-4, [0, 1, 2], 2, 3)
                res_ = {}
                for name, path in [("r12", ref12)] + list(models.items()):
                    with rasterio.open(path) as src:
                        a = src.read([1, 2, 3, 4], window=Window(g_c, g_r, 1000, 1000)).astype(np.float64)
                    a[:, (a == A6.ND16).any(0)] = np.nan
                    if np.isfinite(a[0]).mean() < 0.9:
                        break
                    a = a.reshape(4, 100, 10, 100, 10).mean(axis=(2, 4)) * 1e-4
                    va, na = A7.vis_ndvi(a, [0, 1, 2], 0, 3)
                    res_[name] = A7.correlate({}, vb, nbv, np.zeros_like(vb, bool), va, na, np.zeros_like(va, bool), 10.0)
                if len(res_) < len(models) + 1 or not all(v.get("used") for v in res_.values()):
                    continue
                for k in models:
                    paired[k].append({"block_row": g_r, "block_col": g_c, "dx_m": res_[k]["dx_m"] - res_["r12"]["dx_m"],
                                      "dy_m": res_[k]["dy_m"] - res_["r12"]["dy_m"]})
    out = {}
    for k, v in paired.items():
        if not v:
            out[k] = {"blocks": 0}
            continue
        dx, dy = np.array([r["dx_m"] for r in v]), np.array([r["dy_m"] for r in v])
        out[k] = {"blocks": len(v), "dx_median_m": round(float(np.median(dx)), 3), "dy_median_m": round(float(np.median(dy)), 3),
                  "magnitude_m": round(float(np.hypot(np.median(dx), np.median(dy))), 3),
                  "dx_mean_m": round(float(dx.mean()), 3), "dy_mean_m": round(float(dy.mean()), 3)}
    return out, paired


def quad13_check():
    """F1 checks before the v3 mosaic (CLAUDE.md 13.3): the quadratic correction of 1_3 as a GCP VRT; a measurement copy;
    image blocks of the corrected 1_3 against the registered 1_2 (every block of the 1_3 -> 1_2 pair in
    blocks_measure.csv, 1 km at a 500 m stride, A7 phase correlation, used blocks only) for the quadratic, the
    translation (v1) and the affine copies, medians per 3 km band of northing; the Sentinel-2 paired check of affine_13()
    for the same copies. Rule (13.3): if the quadratic's worst band exceeds 0.5 m, stop and report.
    Writes registration.json["strip_1_3_quadratic"], quad13_check.json and quad13_check_bins.csv in OUT."""
    t0 = time.time()
    corrected, info = quad13_model()
    MR = OUT / "measure_registered"
    vrt, fid = gcp_vrt("1_3", corrected, MR, name="check_1_3_quadratic_grid1m.vrt")
    info["gcp_vrt"] = {"path": str(vrt.relative_to(OUT)), **fid}
    q = MR / "check_1_3_quadratic_grid1m.tif"
    copy_from_vrt(vrt, q)
    copies = {"quadratic": q, "translation": MR / "check_1_3_translation_grid1m.tif",
              "affine": MR / "check_1_3_affine_grid1m.tif"}
    r12 = MR / "check_1_2_registered_grid1m.tif"
    cand = {(int(r["block_row"]), int(r["block_col"])) for r in csv.DictReader(open(OUT / "blocks_measure.csv"))
            if r["pair"] == "1_3->1_2"}
    rows = {k: [] for k in copies}
    for r0, c0 in sorted(cand):
        b12 = neo_block(r12, r0, c0, BLOCK)
        if np.isfinite(b12[0]).mean() < A7.VALID_MIN:
            continue
        for k, path in copies.items():
            b13 = neo_block(path, r0, c0, BLOCK)
            if np.isfinite(b13[0]).mean() < A7.VALID_MIN:
                continue
            row = A7.correlate({"model": k, "block_row": r0, "block_col": c0}, *b13, *b12, 1.0)
            if row.get("used"):
                row["dx_13_rel_12_m"], row["dy_13_rel_12_m"] = -row["dx_m"], -row["dy_m"]
                rows[k].append(row)
    bins, worst = [], {}
    for k, v in rows.items():
        b, w = band_residuals(v, k)
        bins += b
        worst[k] = {"blocks": len(v), "worst_band_median_magnitude_m": w,
                    "dx_13_rel_12_median_m": round(float(np.median([r["dx_13_rel_12_m"] for r in v])), 3) if v else None,
                    "dy_13_rel_12_median_m": round(float(np.median([r["dy_13_rel_12_m"] for r in v])), 3) if v else None}
    write_rows(OUT / "quad13_check_bins.csv", bins)
    write_rows(OUT / "quad13_check_blocks.csv", [r for v in rows.values() for r in v])
    s2, _ = s2_paired(copies, r12)
    out = {"model": info, "image_check": worst, "s2_paired": s2,
           "rule": "the v3 mosaic is built only if the quadratic's worst 3 km band median (image blocks) is below 0.5 m",
           "pass": bool(worst["quadratic"]["worst_band_median_magnitude_m"] is not None
                        and worst["quadratic"]["worst_band_median_magnitude_m"] < 0.5),
           "seconds": round(time.time() - t0, 1)}
    (OUT / "quad13_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    reg["strip_1_3_quadratic"] = {**info, "check": "quad13_check.json", "pass": out["pass"]}
    (OUT / "registration.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    if not out["pass"]:
        raise SystemExit("quadratic worst band >= 0.5 m: stop and report (CLAUDE.md 13.3 F1)")


def quad13_lowpass(lowpass=0.1):
    """After the code review of 3 Nov 2026 (CLAUDE.md 13.3 F1): the unweighted phase correlation of quad13_check() is
    biased on bilinear-resampled copies (pair_consistency_13_12.json); itareo.phasecorr's estimate weighted toward low
    frequencies (lowpass 0.1 cycles per px, as A5 uses for resampled images) recovers the known shifts
    (pair_consistency_13_12_lowpass.json). Measurement only, for Shafi's decision: registration.json is not changed and
    no mosaic is built.
      1. the 1_3 -> 1_2 pair table re-measured with the low-pass estimate on the unregistered copies (same used blocks);
      2. the quadratic refitted on it (as quad13_model), its GCP VRT and measurement copy check_1_3_quadratic_lp_grid1m.tif;
      3. image check of the copies (quadratic fitted on the stored table, quadratic refitted, affine, translation) against
         the registered 1_2 with both estimates; worst 3 km band median for each.
    Writes quad13_lowpass.json, quad13_lowpass_bins.csv and quad13_lowpass_blocks.csv in OUT."""
    t0 = time.time()
    pc = [r for r in csv.DictReader(open(OUT / "pair_consistency_13_12_lowpass_blocks.csv")) if r["remeasured_used"] == "True"]
    X = np.array([GRID_1M.left + float(r["block_col"]) + BLOCK / 2 for r in pc])
    Y = np.array([GRID_1M.top - float(r["block_row"]) - BLOCK / 2 for r in pc])
    D = np.column_stack([[float(r["remeasured_lp_dx_m"]) for r in pc], [float(r["remeasured_lp_dy_m"]) for r in pc]])
    corrected, info = quad13_model((X, Y, D))
    info["table"] = f"pair_consistency_13_12_lowpass_blocks.csv (remeasured, lowpass {lowpass}), {len(D)} blocks"
    MR = OUT / "measure_registered"
    q_lp = MR / "check_1_3_quadratic_lp_grid1m.tif"
    vrt, fid = gcp_vrt("1_3", corrected, MR, name="check_1_3_quadratic_lp_grid1m.vrt")
    info["gcp_vrt"] = {"path": str(vrt.relative_to(OUT)), **fid}
    if not q_lp.exists():
        copy_from_vrt(vrt, q_lp)
    copies = {"quadratic_lp_fit": q_lp, "quadratic_stored_fit": MR / "check_1_3_quadratic_grid1m.tif",
              "affine": MR / "check_1_3_affine_grid1m.tif", "translation": MR / "check_1_3_translation_grid1m.tif"}
    r12 = MR / "check_1_2_registered_grid1m.tif"
    cand = {(int(r["block_row"]), int(r["block_col"])) for r in csv.DictReader(open(OUT / "blocks_measure.csv"))
            if r["pair"] == "1_3->1_2"}
    rows = {k: [] for k in copies}
    for r0, c0 in sorted(cand):
        b12 = neo_block(r12, r0, c0, BLOCK)
        if np.isfinite(b12[0]).mean() < A7.VALID_MIN:
            continue
        for k, path in copies.items():
            b13 = neo_block(path, r0, c0, BLOCK)
            if np.isfinite(b13[0]).mean() < A7.VALID_MIN:
                continue
            row = A7.correlate({"model": k, "block_row": r0, "block_col": c0}, *b13, *b12, 1.0, lowpass)
            if row.get("used"):
                row.update(dx_13_rel_12_m=-row["dx_m"], dy_13_rel_12_m=-row["dy_m"],
                           dx_lp_13_rel_12_m=-row["dx_lp_m"], dy_lp_13_rel_12_m=-row["dy_lp_m"])
                rows[k].append(row)
    bins, summ = [], {}
    for k, v in rows.items():
        b_u, w_u = band_residuals(v, k)
        b_l, w_l = band_residuals(v, k, kx="dx_lp_13_rel_12_m", ky="dy_lp_13_rel_12_m")
        for b in b_u:
            b["estimator"] = "unweighted"
        for b in b_l:
            b["estimator"] = f"lowpass {lowpass}"
        bins += b_u + b_l
        summ[k] = {"blocks": len(v), "worst_band_unweighted_m": w_u, "worst_band_lowpass_m": w_l,
                   "median_lowpass_13_rel_12_m": [round(float(np.median([r["dx_lp_13_rel_12_m"] for r in v])), 3),
                                                  round(float(np.median([r["dy_lp_13_rel_12_m"] for r in v])), 3)] if v else None}
    write_rows(OUT / "quad13_lowpass_bins.csv", bins)
    write_rows(OUT / "quad13_lowpass_blocks.csv", [r for v in rows.values() for r in v])
    out = {"lowpass_cycles_per_px": lowpass, "refit_model": info, "image_check": summ,
           "rule_13_3": "build v3 only if the worst 3 km band median is below 0.5 m (the estimator is Shafi's decision)",
           "seconds": round(time.time() - t0, 1)}
    (OUT / "quad13_lowpass.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def lowpass_residuals_v1(lowpass=0.1):
    """CLAUDE.md 13.4, measurement only: low-pass residuals of the v1 translations. Registered measurement copies of 1_2,
    1_5 (warp_copy with the applied translation) against Pleiades 2017 on the blocks of the pleiades2017 -> strip pairs of
    blocks_measure.csv; 1_6 against the registered 1_2 on the blocks of the 1_6 -> 1_2 pair. Residual = displacement of
    the registered strip relative to the reference (m east, m north), low-pass and unweighted. Writes
    v1_lowpass_residuals.json and v1_lowpass_residuals_blocks.csv in OUT."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    MR = OUT / "measure_registered"

    def translate_fn(dx, dy):
        return lambda g: [g[0] - dx / M_PER_DEG_LON, g[1], g[2], g[3] - dy / M_PER_DEG_LAT, g[4], g[5]]
    copies = {}
    for st in ("1_2", "1_5", "1_6"):
        path = MR / f"check_{st}_registered_grid1m.tif"
        if not path.exists():
            t = reg["applied_displacements"][st]
            warp_copy(st, translate_fn(t["dx_east_m"], t["dy_north_m"]), path)
        copies[st] = path
    bm = list(csv.DictReader(open(OUT / "blocks_measure.csv")))
    rows, summ = [], {}
    for st, (pair, ref) in {"1_2": ("pleiades2017->1_2", "ple"), "1_5": ("pleiades2017->1_5", "ple"),
                            "1_6": ("1_6->1_2", "r12")}.items():
        cand = sorted({(int(r["block_row"]), int(r["block_col"])) for r in bm if r["pair"] == pair})
        got = []
        for r0, c0 in cand:
            b = neo_block(copies[st], r0, c0, BLOCK)
            a = ple_block(r0, c0, BLOCK) if ref == "ple" else neo_block(copies["1_2"], r0, c0, BLOCK)
            if np.isfinite(a[0]).mean() < A7.VALID_MIN or np.isfinite(b[0]).mean() < A7.VALID_MIN:
                continue
            row = A7.correlate({"strip": st, "reference": "pleiades2017" if ref == "ple" else "registered 1_2",
                                "block_row": r0, "block_col": c0}, *a, *b, 1.0, lowpass)
            if row.get("used"):
                got.append(row)
        rows += got
        v = np.array([(r["dx_lp_m"], r["dy_lp_m"], r["dx_m"], r["dy_m"]) for r in got], float)
        summ[st] = {"reference": got[0]["reference"] if got else None, "blocks": len(got),
                    "lowpass_median_m": [round(float(np.median(v[:, 0])), 3), round(float(np.median(v[:, 1])), 3)] if got else None,
                    "lowpass_iqr_e": np.percentile(v[:, 0], [25, 75]).round(3).tolist() if got else None,
                    "lowpass_iqr_n": np.percentile(v[:, 1], [25, 75]).round(3).tolist() if got else None,
                    "unweighted_median_m": [round(float(np.median(v[:, 2])), 3), round(float(np.median(v[:, 3])), 3)] if got else None,
                    "applied_translation_m": [reg["applied_displacements"][st]["dx_east_m"], reg["applied_displacements"][st]["dy_north_m"]]}
    write_rows(OUT / "v1_lowpass_residuals_blocks.csv", rows)
    out = {"lowpass_cycles_per_px": lowpass, "residual": "displacement of the registered strip relative to the reference",
           "strips": summ}
    (OUT / "v1_lowpass_residuals.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def mosaic_v3():
    """Registered mosaic v3 (CLAUDE.md 13.3 F1, 13.4): strip 1_3 corrected with the quadratic-in-northing model refitted on
    the low-pass pair table (GCP VRT; strip13_cv_lowpass() passed), 1_2, 1_5, 1_6 with their v1 translations; each strip resampled once from its DIMAP tiles, as
    mosaic(). Output derived/pneo2022_registered_v3/; v1 stays (it backs Stage M and A1/A2)."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    assert reg.get("strip_1_3_quadratic_lowpass", {}).get("pass"), "run --strip13-cv-lowpass first (and it must pass)"
    corrected, _ = quad13_model(lowpass_table())         # the quadratic refitted on the low-pass pair table (13.4)
    shifts = {st: (v["dx_east_m"], v["dy_north_m"]) for st, v in reg["applied_displacements"].items()}
    out = A6.W / "derived" / "pneo2022_registered_v3"
    out.mkdir(parents=True, exist_ok=True)
    saved = (A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons)
    A6.RUN = out
    A6.OUT = A6.W / "derived"
    orig_vrt, orig_cloud = A6.build_vrt, A6.cloud_polygons
    translated = registered_vrt_builder(orig_vrt, shifts)

    def build_vrt(strip):
        if strip != "1_3":
            return translated(strip)
        vrt, info = gcp_vrt("1_3", corrected, out, builder=orig_vrt)
        lat = dict(info["lattice"], correction="quadratic in northing (GCP VRT)", gcps=info["gcps"],
                   polynomial_error_m=info["polynomial_error_m"])
        return vrt, lat

    def cloud_polygons(strip, lattice):
        if strip == "1_3":
            geoms, info = orig_cloud(strip, lattice)
            assert not geoms, "1_3 has cloud polygons: transform them with the quadratic first"
            return geoms, info
        geoms, info = orig_cloud(strip, lattice)
        dx, dy = shifts[strip]
        return [mapping(translate(shape(g), xoff=-dx, yoff=-dy)) for g in geoms], info

    A6.build_vrt, A6.cloud_polygons = build_vrt, cloud_polygons
    try:
        A6.run_warp()
    finally:
        A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons = saved


def compare_v3_v1():
    """v3 against v1: identical wherever neither mosaic takes strip 1_3 (strips 1_2, 1_5 and 1_6 are unchanged)."""
    v1, v3 = OUT, A6.W / "derived" / "pneo2022_registered_v3"
    res = {"pixels_compared": 0, "pixels_differing": 0, "sid_changed": 0, "valid_v1": 0, "valid_v3": 0}
    with rasterio.open(v1 / "pneo2022_toa_grid1m.tif") as a1, rasterio.open(v3 / "pneo2022_toa_grid1m.tif") as a3, \
            rasterio.open(v1 / "pneo2022_strip_id_grid1m.tif") as s1, rasterio.open(v3 / "pneo2022_strip_id_grid1m.tif") as s3:
        for r0 in range(0, GRID_1M.height, 2048):
            w = Window(0, r0, GRID_1M.width, min(2048, GRID_1M.height - r0))
            i1, i3 = s1.read(1, window=w), s3.read(1, window=w)
            m = (i1 != 3) & (i3 != 3)
            t1, t3 = a1.read(window=w), a3.read(window=w)
            res["pixels_compared"] += int(m.sum())
            res["pixels_differing"] += int(((t1 != t3).any(0) & m).sum())
            res["sid_changed"] += int((i1 != i3).sum())
            res["valid_v1"] += int((i1 > 0).sum())
            res["valid_v3"] += int((i3 > 0).sum())
    res["identical_outside_1_3"] = res["pixels_differing"] == 0
    (A6.W / "derived" / "pneo2022_registered_v3" / "compare_v1.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


def mosaic_v2():
    """Registered mosaic v2 (CLAUDE.md 13.2): as mosaic(), but strip 1_3 takes the affine geotransform of affine_13();
    1_2, 1_5, 1_6 keep their translations. One resampling per strip as before. Output: derived/pneo2022_registered_v2/.
    Run once on 2 Nov 2026 after the Sentinel-2 gate of affine_13() passed; strip13_check() then showed that the affine
    is worse than the translation in the north of 1_3 (apply set to false) and that output was archived
    (archive/2026-11-02/pneo2022_registered_v2_affine13_not_applied/). v1 stays the registered mosaic."""
    reg = json.loads((OUT / "registration.json").read_text(encoding="utf-8"))
    aff = reg["strip_1_3_affine"]
    chk = json.loads((OUT / "strip13_check.json").read_text(encoding="utf-8"))
    assert aff["apply"] and chk["apply_affine"], "the 1_3 affine did not pass strip13_check()"
    g13 = [float(v) for v in aff["affine_geotransform_1_3"]]
    shifts = {st: (v["dx_east_m"], v["dy_north_m"]) for st, v in reg["applied_displacements"].items()}
    out = A6.W / "derived" / "pneo2022_registered_v2"
    out.mkdir(parents=True, exist_ok=True)
    saved = (A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons)
    A6.RUN = out
    A6.OUT = A6.W / "derived"
    orig_vrt, orig_cloud = A6.build_vrt, A6.cloud_polygons

    def build_vrt(strip):
        vrt, lat = orig_vrt(strip)
        txt = vrt.read_text(encoding="utf-8")
        m = re.search(r"<GeoTransform>([^<]+)</GeoTransform>", txt)
        g = [float(v) for v in m.group(1).split(",")]
        if strip == "1_3":
            g = g13
            how = "affine (strip_1_3_affine)"
        else:
            dx, dy = shifts[strip]
            g[0] -= dx / M_PER_DEG_LON
            g[3] -= dy / M_PER_DEG_LAT
            how = f"translation {-dx}, {-dy} m"
        txt = txt.replace(m.group(0), "<GeoTransform>" + ", ".join(repr(float(v)) for v in g) + "</GeoTransform>")
        vrt.write_text(txt, encoding="utf-8")
        return vrt, dict(lat, x0=g[0], y0=g[3], geotransform=g, correction=how)

    def cloud_polygons(strip, lattice):
        geoms, info = orig_cloud(strip, lattice)
        if strip == "1_3":
            assert not geoms, "1_3 has cloud polygons: transform them with the affine first"
            return geoms, info
        dx, dy = shifts[strip]
        return [mapping(translate(shape(g), xoff=-dx, yoff=-dy)) for g in geoms], info

    A6.build_vrt, A6.cloud_polygons = build_vrt, cloud_polygons
    try:
        A6.run_warp()
    finally:
        A6.RUN, A6.OUT, A6.build_vrt, A6.cloud_polygons = saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--affine-13", action="store_true")
    ex = ap.add_mutually_exclusive_group()          # one mosaic per call (they patch A6 differently)
    ex.add_argument("--mosaic-v2", action="store_true")
    ex.add_argument("--mosaic", action="store_true")
    ex.add_argument("--mosaic-v3", action="store_true")
    ap.add_argument("--strip13-check", action="store_true")
    ap.add_argument("--quad13-check", action="store_true")
    ap.add_argument("--quad13-lowpass", action="store_true")
    ap.add_argument("--strip13-cv-lowpass", action="store_true")
    ap.add_argument("--lowpass-residuals-v1", action="store_true")
    ap.add_argument("--compare-v3", action="store_true")
    ap.add_argument("--restore-v1-vrts", action="store_true")
    ap.add_argument("--check-ref", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-s2", action="store_true")
    a = ap.parse_args()
    if a.check_s2:
        check_s2()
    if a.affine_13:
        affine_13()
    if a.strip13_check:
        strip13_check()
    if a.restore_v1_vrts:
        restore_v1_vrts()
    if a.quad13_check:
        quad13_check()
    if a.quad13_lowpass:
        quad13_lowpass()
    if a.strip13_cv_lowpass:
        strip13_cv_lowpass()
    if a.lowpass_residuals_v1:
        lowpass_residuals_v1()
    if a.mosaic_v2:
        mosaic_v2()
    if a.mosaic_v3:
        mosaic_v3()
    if a.compare_v3:
        compare_v3_v1()
    if a.check_ref:
        check_ref("s2_2022Q4", A6.W.parent / "gee_exports/2026-10-28_nodata/s2_sr_median_2022Q4.tif", [1, 2, 3, 7], 65535,
                  10, -7, -7)
        check_ref("ps20220424", A6.W.parent / "raw_data/PRISMA/PSScene/20220424_135836_31_2446_3B_AnalyticMS_SR_8b_clip.tif",
                  [2, 4, 6, 8], 0, 3, 33, 681)
    if a.measure:
        measure()
    if a.mosaic:
        mosaic()
    if a.check:
        check()
