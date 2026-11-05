"""Sentinel-2 quarterly median composites of the 10 bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 on the native 10 m
lattice GRID_S2_10M (no resampling in Earth Engine; 20 m bands become 2 x 2 duplicates), uint16 reflectance x 10000
(nodata 65535). Clouds masked with Cloud Score+ (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, cs_cdf >= CLEAR).
  - 2022 Q1-Q4 and Oct 2024 - Sep 2025 (2024Q4, 2025Q1-Q3): surface reflectance, COPERNICUS/S2_SR_HARMONIZED;
  - Jan-Mar 2017, to pair with the Pleiades scene of 12 Feb 2017 and its GEOBIA labels: TOP-OF-ATMOSPHERE reflectance,
    COPERNICUS/S2_HARMONIZED (L1C), exported 28 Oct 2026; not radiometrically comparable with the SR composites.
    CORRECTION 28 Oct 2026: the earlier note that Earth Engine had no harmonised SR for early 2017 over this area was
    wrong. A read-only query found 6 COPERNICUS/S2_SR_HARMONIZED scenes over GRID_S2_10M in Jan-Mar 2017 (tile T20JLL:
    30 Jan, 12 Feb, 19 Feb, 22 Feb, 1 Mar, 14 Mar; CLOUDY_PIXEL_PERCENTAGE 0.0-2.5), the 12 Feb one on the Pleiades
    date, and 24 SR scenes in all of 2017 (docs/evidence_phase2/gee_landing/s2_2017Q1_scenes.csv). SR for 2017 is
    product s2sr2017 (tasks_sr2017, added 29 Oct 2026): the quarterly median Jan-Mar 2017 and the single scene of
    12 Feb 2017 (expect_count = 1: run_exports --start refuses the task if the one-day window holds another image).
"""
from itareo.gee.common import GRID_S2_10M, QUARTERS, task

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLEAR = 0.60
SR, TOA = "COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S2_HARMONIZED"


def _col(ee, collection, start, end):
    from itareo.gee.common import region
    s2 = ee.ImageCollection(collection).filterBounds(region(ee, GRID_S2_10M)).filterDate(start, end)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    return s2.linkCollection(cs, ["cs_cdf"]).map(lambda im: im.updateMask(im.select("cs_cdf").gte(CLEAR)))


def build(collection, start, end):
    return lambda ee: _col(ee, collection, start, end).select(BANDS).median().toUint16()


def count(collection, start, end):
    return lambda ee: _col(ee, collection, start, end).size()


def tasks():
    out = [task(f"s2_sr_median_{q}", build(SR, s, e), GRID_S2_10M, BANDS, "uint16",
                f"S2 SR harmonised quarterly median {s} to {e}", count(SR, s, e)) for q, (s, e) in QUARTERS.items()]
    out.append(task("s2_toa_median_2017Q1", build(TOA, "2017-01-01", "2017-04-01"), GRID_S2_10M, BANDS, "uint16",
                    "S2 L1C TOA (NOT SR) median Jan-Mar 2017, pairs with the Pleiades 2017 labels",
                    count(TOA, "2017-01-01", "2017-04-01")))
    return out


def tasks_sr2017():
    """Product s2sr2017 (29 Oct 2026): surface reflectance for the Pleiades date. The quarterly median Jan-Mar 2017 and
    the single scene of 12 Feb 2017 (a one-day window: the median of one image is that image), both Cloud Score+ masked
    like every S2 task."""
    scene = task("s2_sr_scene_20170212", build(SR, "2017-02-12", "2017-02-13"), GRID_S2_10M, BANDS, "uint16",
                 "S2 SR single scene of 12 Feb 2017, the Pleiades date (tile T20JLL), Cloud Score+ masked",
                 count(SR, "2017-02-12", "2017-02-13"))
    scene["expect_count"] = 1        # exactly one granule: a second image would turn the "scene" into a median
    return [task("s2_sr_median_2017Q1", build(SR, "2017-01-01", "2017-04-01"), GRID_S2_10M, BANDS, "uint16",
                 "S2 SR harmonised quarterly median 2017-01-01 to 2017-04-01, pairs with the Pleiades 2017 labels",
                 count(SR, "2017-01-01", "2017-04-01")), scene]


# Product s2_2026 (2 Nov 2026, stage A8; windows corrected after review the same day): SR medians on the NISAR seasons
# of CLAUDE.md 11.4, to pair with NISAR. End dates are exclusive.
#   2026winter  17 Jun - 31 Aug 2026, winter, leaf-off (the first PROVISIONAL NISAR date is 17 Jun);
#   2026sep     September 2026, kept separate;
#   2026oct     October 2026, spring, leaf-on; the spring season runs "to the latest acquisition", so the end is fixed
#               at 1 Nov (1 Nov excluded) to keep the composite reproducible whenever it is exported.
# Source images per window on 2 Nov 2026 (run_exports --check --only s2_2026, export_check.csv): winter 39, September 15,
# October 15 (1 Nov excluded).
PERIODS_2026 = {"2026winter": ("2026-06-17", "2026-09-01"), "2026sep": ("2026-09-01", "2026-10-01"),
                "2026oct": ("2026-10-01", "2026-11-01")}


# Product s2_2021 (4 Nov 2026; CLAUDE.md 13.5 and 13.7): SR quarterly medians of 2021, the same build as 2022, so the time series of the optical-only design reaches 2021-2022. End dates are exclusive.
QUARTERS_2021 = {"2021Q1": ("2021-01-01", "2021-04-01"), "2021Q2": ("2021-04-01", "2021-07-01"), "2021Q3": ("2021-07-01", "2021-10-01"), "2021Q4": ("2021-10-01", "2022-01-01")}


def tasks_2021():
    return [task(f"s2_sr_median_{q}", build(SR, s, e), GRID_S2_10M, BANDS, "uint16", f"S2 SR harmonised quarterly median {s} to {e}", count(SR, s, e)) for q, (s, e) in QUARTERS_2021.items()]


def tasks_2026():
    return [task(f"s2_sr_median_{q}", build(SR, s, e), GRID_S2_10M, BANDS, "uint16",
                 f"S2 SR harmonised median {s} to {e} (end exclusive), NISAR season, to pair with NISAR", count(SR, s, e))
            for q, (s, e) in PERIODS_2026.items()]
