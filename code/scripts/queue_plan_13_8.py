"""What the queue of CLAUDE.md 13.8 runs after the diagnostics: prints one "config fold seed" per line, in order. Reads docs/evidence_phase2/ugi_seg/e3/reading.json and diagnostics/d3_screen/d3.json only.

  1. E3, if its reading asks for more seeds. Undecided in the centre, the primary region: seed 1, before its reading (13.8). 13.8 names no region; Shafi ruled on 4 Nov, after the review, that the centre alone triggers it, so an undecided in the ejido alone triggers nothing. The seed decides nothing by itself, the reading then uses the mean of two seeds against the threshold / sqrt 2.
  2. The conditional run of 13.8: the PlanetScope screen at or above 0.60, E1c (ugi_e1c_ps1m), seed 0, folds 1-4; otherwise seed 1 of E1b and of E2.
The queue stops there ("Either way the queue stops there and reports", 13.8; "Stop there and report", Shafi's item 2). If E3 passes in the centre, its seeds 1-2 (13.6) are for Shafi's GO; the report says so. --test runs the logic on synthetic readings and exits 1 if any case differs. Run from work_ms/: python scripts/queue_plan_13_8.py
"""
import json
import sys
from pathlib import Path

DOC = Path("../docs/evidence_phase2/ugi_seg")
FOLDS = (1, 2, 3, 4)


def plan(reading, d3):
    runs = []
    if reading["primary"]["status"] == "undecided":
        runs += [("ugi_e3_s2", f, 1) for f in FOLDS]
    if d3["ps_reaches_0_60"]:
        runs += [("ugi_e1c_ps1m", f, 0) for f in FOLDS]
    else:
        runs += [("ugi_e1b_ps_plane", f, 1) for f in FOLDS] + [("ugi_e2_tex", f, 1) for f in FOLDS]
    return runs


def test():
    def r(p, s):
        return {"primary": {"status": p}, "secondary": {"status": s}}
    t, f = {"ps_reaches_0_60": True}, {"ps_reaches_0_60": False}
    e1c = [("ugi_e1c_ps1m", k, 0) for k in FOLDS]
    seeds = [("ugi_e1b_ps_plane", k, 1) for k in FOLDS] + [("ugi_e2_tex", k, 1) for k in FOLDS]
    s1 = [("ugi_e3_s2", k, 1) for k in FOLDS]
    cases = [(r("no gain", "no gain"), t, e1c), (r("no gain", "no gain"), f, seeds), (r("undecided", "no gain"), t, s1 + e1c), (r("no gain", "undecided"), t, e1c), (r("undecided", "undecided"), f, s1 + seeds),
             (r("the input helps", "no gain"), t, e1c), (r("the input helps", "undecided"), f, seeds), (r("no gain", "the input helps"), t, e1c)]
    bad = [(a, b, plan(a, b), c) for a, b, c in cases if plan(a, b) != c]
    print("queue plan cases", len(cases), "failed", len(bad))
    return not bad


if __name__ == "__main__":
    if "--test" in sys.argv:
        raise SystemExit(0 if test() else 1)
    rd = json.load(open(DOC / "e3" / "reading.json"))
    d = json.load(open(DOC / "diagnostics" / "d3_screen" / "d3.json"))
    for c, f_, s_ in plan(rd, d):
        print(c, f_, s_)
