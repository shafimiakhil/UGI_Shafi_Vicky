"""CPU side of the queue of CLAUDE.md 13.8 (jobs/night_queue_13_8.sh). Reads and writes only the scoring files of finished runs; never touches the GPU. It starts its work when the post of 13.7 has ended, then every 120 s:
  * rules 7 and 9 and the centre sq8km on each finished run of E3, D2 (diag_d2_*_blank), E1c, and seed 1 of E1b and E2; the B4 tile set for the E1b blank and for E1c (with the baseline's seeds 1-2, whose SD on these tiles E1c's reading needs);
  * E3 seed 0 complete: the summaries and the reading (scripts/exp_report.py --centre sq8km), once; E3 passed to this post when the queue of 13.7 stopped on its fold 1;
  * D2 complete: the summaries and scripts/diag_d2_report.py, once;
  * E1c seed 0 complete: the summaries and the reading (scripts/exp_report.py --centre sq8km --b4), once;
  * E3 seed 1 complete: the two-seed summaries and scripts/exp_two_seeds.py --mode reading, once;
  * seed 1 of E1b or of E2 complete: the same with --mode size, once.
It ends when the queue has printed done or STOP and nothing is left to do. A failed step prints "STOP: ..." and ends it; every step skips what is finished and refuses what is partial.
Run from work_ms/, after the queue has started, with PYTHONUTF8=1 TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=-1 GDAL_PAM_ENABLED=NO: python -W ignore scripts/night_post_13_8.py >> runs_itareo/ugi/night3_post_stdout.txt 2>&1
Before a relaunch archive night3_stdout.txt and this log (no-clobber): an old STOP there ends this job at once, and the queue would then wait two hours for a reading and stop. The queue does not look at this job after E3's reading, so the report checks this log for STOP and for the line "post of 13.8 done"; if that line is missing, this job is rerun alone.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("runs_itareo/ugi")
OLDPOST = ROOT / "night2_post_stdout.txt"
QUEUE = ROOT / "night3_stdout.txt"
SUM = ROOT / "summary_r"
DOC = Path("../docs/evidence_phase2/ugi_seg")
FOLDS = (1, 2, 3, 4)
NAME = "sq8km"
BASE = "ugi_r1_none"
BLANK = {"e1b": ("ugi_e1b_ps_plane", "diag_d2_e1b_blank"), "e2": ("ugi_e2_tex", "diag_d2_e2_blank")}


def sh(*a):
    r = subprocess.run([sys.executable, "-W", "ignore", *a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1500:], flush=True)
    if r.returncode != 0:
        print(f"STOP: step failed: {' '.join(a)} {time.strftime('%F %T')}", flush=True)
        raise SystemExit(1)


def done(arm, seeds=(0,), sub=None):
    """Every fold of these seeds has finished, and carries the scoring file of sub."""
    for s in seeds:
        for f in FOLDS:
            d = ROOT / arm / f"fold{f}_seed{s}"
            if not (d / "run_log.md").exists():
                return False
            if sub == "rule7" and not (d / "metrics_rule7.json").exists():
                return False
            if sub and sub != "rule7" and not (d / sub / "done.json").exists():
                return False
    return True


def scored(arm, seeds=(0,), subs=("rule7", "ejido", f"centre_{NAME}")):
    return all(done(arm, seeds, s) for s in subs)


def summ(tag, arms, seeds, sub):
    if not (SUM / f"summary_{tag}.json").exists():
        sh("-m", "itareo.ugi_seg.summarize_r", "--arms", *arms, "--seeds", *[str(s) for s in seeds], "--sub", sub, "--out", str(SUM), "--tag", tag)


def queue_ended():
    if not QUEUE.exists():
        return False
    last = [x for x in QUEUE.read_text(encoding="utf-8", errors="replace").splitlines() if re.match(r"^(self-tests|done|STOP)", x)]
    return bool(last) and bool(re.match(r"^(done|STOP)", last[-1]))


def later_runs(arm, seeds):
    return [str(d) for s in seeds for d in sorted((ROOT / arm).glob(f"fold*_seed{s}"))] if (ROOT / arm).exists() else []


def one_pass():
    runs = [str(ROOT / a) for _, a in BLANK.values() if (ROOT / a).exists()]
    for arm in ("ugi_e1c_ps1m", "ugi_e3_s2"):
        if (ROOT / arm).exists():
            runs.append(str(ROOT / arm))
    runs += later_runs("ugi_e1b_ps_plane", (1,)) + later_runs("ugi_e2_tex", (1,))
    if runs:
        sh("-m", "itareo.ugi_seg.rescore7", "--runs", *runs)
        sh("-m", "itareo.ugi_seg.rescore9", "--runs", *runs)
        sh("-m", "itareo.ugi_seg.rescore_centre", "--regions", NAME, "--runs", *runs, "--workers", "4")
    if (ROOT / "diag_d2_e1b_blank").exists():
        sh("-m", "itareo.ugi_seg.rescore9", "--b4", "--runs", str(ROOT / "diag_d2_e1b_blank"))
    if (ROOT / "ugi_e1c_ps1m").exists():
        base = [str(ROOT / BASE / f"fold{f}_seed{s}") for s in (1, 2) for f in FOLDS]
        sh("-m", "itareo.ugi_seg.rescore9", "--b4", "--runs", str(ROOT / "ugi_e1c_ps1m"), *base)
    # ---- E3 ---------------------------------------------------------------------------------------------------------------------------------------
    if scored("ugi_e3_s2") and not (DOC / "e3/reading.md").exists():
        summ("e3_ejido", (BASE, "ugi_e3_s2"), (0,), "ejido")
        if not (SUM / "summary_e3.json").exists():
            sh("-m", "itareo.ugi_seg.summarize_r", "--arms", BASE, "ugi_e3_s2", "--seeds", "0", "--out", str(SUM), "--tag", "e3")
        summ(f"e3_centre_{NAME}", (BASE, "ugi_e3_s2"), (0,), f"centre_{NAME}")
        sh("scripts/exp_report.py", "--arm", "ugi_e3_s2", "--tag", "e3", "--out", str(DOC / "e3"), "--centre", NAME)
    # ---- D2 ---------------------------------------------------------------------------------------------------------------------------------------
    if all(scored(b) for _, b in BLANK.values()) and done("diag_d2_e1b_blank", sub="ejido_b4") and not (DOC / "diagnostics/d2_blank/README.md").exists():
        for tag, (arm, blank) in BLANK.items():
            summ(f"d2_{tag}_ejido", (arm, blank), (0,), "ejido")
            summ(f"d2_{tag}_centre_{NAME}", (arm, blank), (0,), f"centre_{NAME}")
        summ("d2_e1b_ejido_b4", BLANK["e1b"], (0,), "ejido_b4")
        sh("scripts/diag_d2_report.py", "--out", str(DOC / "diagnostics/d2_blank"))
    # ---- E1c --------------------------------------------------------------------------------------------------------------------------------------
    if scored("ugi_e1c_ps1m") and done("ugi_e1c_ps1m", sub="ejido_b4") and all(done(BASE, (s,), "ejido_b4") for s in (1, 2)) and not (DOC / "e1c/reading.md").exists():
        summ("e1c_ejido", (BASE, "ugi_e1c_ps1m"), (0,), "ejido")
        if not (SUM / "summary_e1c.json").exists():
            sh("-m", "itareo.ugi_seg.summarize_r", "--arms", BASE, "ugi_e1c_ps1m", "--seeds", "0", "--out", str(SUM), "--tag", "e1c")
        summ("e1c_ejido_b4", (BASE, "ugi_e1c_ps1m"), (0,), "ejido_b4")
        summ(f"e1c_centre_{NAME}", (BASE, "ugi_e1c_ps1m"), (0,), f"centre_{NAME}")
        sh("scripts/exp_report.py", "--arm", "ugi_e1c_ps1m", "--tag", "e1c", "--out", str(DOC / "e1c"), "--centre", NAME, "--b4")
    # ---- two seeds --------------------------------------------------------------------------------------------------------------------------------
    for arm, tag, mode in (("ugi_e3_s2", "e3", "reading"), ("ugi_e1b_ps_plane", "e1b", "size"), ("ugi_e2_tex", "e2", "size")):
        if (ROOT / arm / "fold1_seed1").exists() and scored(arm, (1,), ("rule7", "ejido", f"centre_{NAME}")) and not (DOC / tag / "two_seeds/two_seeds.md").exists():
            summ(f"{tag}_s01_ejido", (BASE, arm), (0, 1), "ejido")
            summ(f"{tag}_s01_centre_{NAME}", (BASE, arm), (0, 1), f"centre_{NAME}")
            sh("scripts/exp_two_seeds.py", "--arm", arm, "--tag", tag, "--mode", mode, "--out", str(DOC / tag / "two_seeds"))


if __name__ == "__main__":
    print(f"post of 13.8 waiting for the post of 13.7 {time.strftime('%F %T')}", flush=True)
    while not (OLDPOST.exists() and re.search(r"night post done|STOP", OLDPOST.read_text(encoding="utf-8", errors="replace"))):
        time.sleep(60)
    if "STOP" in OLDPOST.read_text(encoding="utf-8", errors="replace"):
        print(f"STOP: the post of 13.7 stopped {time.strftime('%F %T')}", flush=True)
        raise SystemExit(1)
    print(f"post of 13.8 start {time.strftime('%F %T')}", flush=True)
    while True:
        ended = queue_ended()
        one_pass()
        if ended:
            print(f"post of 13.8 done {time.strftime('%F %T')}", flush=True)
            break
        time.sleep(120)
