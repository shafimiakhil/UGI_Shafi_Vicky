#!/usr/bin/env bash
# The queue of CLAUDE.md 13.8, one GPU job at a time, and no other GPU job beside it (E3 fold 1 failed at 21:43 on 4 Nov with CUDA out of memory because an E1c regression ran its inference beside it). It takes E3 over from jobs/night_queue_13_7.sh, which stopped there: E3 folds 1-4 seed 0, then waits for E3's reading (scripts/night_post_13_8.py), runs D2 (input blanked at inference, scripts/diag_d2.py: E1b and E2, folds 1-4), then the runs of scripts/queue_plan_13_8.py, and stops there.
# D1 and D3 run on the CPU before it starts (their files must exist). Rescoring, summaries and readings: scripts/night_post_13_8.py, started separately.
# Stops on a failed run, on a partial run folder (train.py refuses to overwrite), on no reading of E3 after two hours, or below 30 GB free. Prints "done" at the end or "STOP: ..." on a stop. Every job runs through scripts/guard_run.py, because a job that fails on the GPU can stay stuck at exit (E3 fold 1, 4 Nov) and a plain call would wait on it for ever.
# Run from work_ms/:  bash jobs/night_queue_13_8.sh >> runs_itareo/ugi/night3_stdout.txt 2>&1   then start the post.
# Before a relaunch: archive night3_stdout.txt, night3_post_stdout.txt and night3_log.csv (no-clobber) and move any partial run folder, because both jobs read their old lines (an old STOP ends the post at once and stops the wait for E3's reading). If a reading failed, also move its folder (docs/evidence_phase2/ugi_seg/e3, e1c or diagnostics/d2_blank), which the report refuses to overwrite. Queue first, then the post.
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
LOG=$ROOT/night3_log.csv
DIAG=../docs/evidence_phase2/ugi_seg/diagnostics
OLDPOST=$ROOT/night2_post_stdout.txt
POST=$ROOT/night3_post_stdout.txt
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG

ok=$($PY -c "
import json
a = json.load(open('$DIAG/d2_selftest_v2/d2_selftest.json'))['passed'] is True
b = json.load(open('$DIAG/d3_screen/d3.json'))
c = json.load(open('../docs/evidence_phase2/ugi_seg/selftest_e1c/e1c_selftest.json'))['passed'] is True and json.load(open('../docs/evidence_phase2/ugi_seg/selftest_e3/e3_selftest.json'))['passed'] is True
d = 'scored' in json.load(open('$ROOT/e1c_regression/regression.json'))
e = json.load(open('../docs/evidence_phase2/ugi_seg/selftest_guard/guard_selftest.json'))['passed'] is True
print(int(a and 'ps_reaches_0_60' in b and c and d and e))" 2>/dev/null | tr -d '\r\n')
if [ "$ok" != "1" ]; then echo "STOP: a self-test or regression did not pass; nothing started"; exit 1; fi
if ! $PY -W ignore scripts/queue_plan_13_8.py --test > /dev/null; then echo "STOP: the queue plan failed its test"; exit 1; fi
echo "self-tests passed $(date '+%F %T')"
until grep -qE 'night post done|STOP' $OLDPOST; do sleep 30; done

free_ok() {
  local free
  free=$(df -BG --output=avail /c | tail -1 | tr -dc 0-9)
  if ! [ "$free" -ge 30 ] 2>/dev/null; then echo "STOP: free disk '${free}' GB is below 30 GB or unreadable before $1"; exit 2; fi
  FREE=$free
}

run() {  # config fold seed
  local cfg=$1 f=$2 s=$3 out=$ROOT/$1/fold$2_seed$3
  if [ -f $out/run_log.md ]; then echo "finished already: $cfg fold $f seed $s"; return 0; fi
  if [ -e $out ]; then echo "STOP: partial run folder $out"; exit 1; fi
  free_ok "$cfg fold $f seed $s"
  mkdir -p $out
  local t0
  t0=$(date '+%F %T')
  echo "start $cfg fold $f seed $s $t0"
  $PY -W ignore scripts/guard_run.py --log $out/stdout.txt -- -W ignore -m itareo.ugi_seg.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out < /dev/null
  local rc=$?
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc,$FREE" >> $LOG
  if [ $rc -ne 0 ]; then echo "STOP: run failed: $cfg fold $f seed $s (exit $rc)"; exit 1; fi
  if [ ! -f $out/run_log.md ]; then echo "STOP: run ended with exit 0 and no run_log.md: $cfg fold $f seed $s"; exit 1; fi
}

for f in 1 2 3 4; do run ugi_e3_s2 $f 0; done
echo "E3 trained; waiting for its reading $(date '+%F %T')"
n=0
until [ -f ../docs/evidence_phase2/ugi_seg/e3/reading.json ] || { [ -f $POST ] && grep -q 'STOP' $POST; } || [ $n -ge 120 ]; do sleep 60; n=$((n+1)); done
if [ ! -f ../docs/evidence_phase2/ugi_seg/e3/reading.json ]; then echo "STOP: no reading of E3 (the post stopped, or none came in two hours) $(date '+%F %T')"; exit 1; fi

for arm in ugi_e1b_ps_plane ugi_e2_tex; do
  tag=$([ $arm = ugi_e1b_ps_plane ] && echo e1b || echo e2)
  for f in 1 2 3 4; do
    if [ -f $ROOT/diag_d2_${tag}_blank/fold${f}_seed0/run_log.md ]; then echo "finished already: D2 $arm fold $f"; continue; fi
    if [ -e $ROOT/diag_d2_${tag}_blank/fold${f}_seed0/prob_oof.tif ]; then echo "STOP: partial D2 folder $arm fold $f"; exit 1; fi
    free_ok "D2 $arm fold $f"
    t0=$(date '+%F %T')
    echo "start D2 $arm fold $f $t0"
    $PY -W ignore scripts/guard_run.py --log $ROOT/diag_d2_${tag}_fold${f}_stdout.txt -- -W ignore scripts/diag_d2.py --arm $arm --fold $f --seed 0 < /dev/null
    rc=$?
    echo "diag_d2_$tag,$f,0,$t0,$(date '+%F %T'),$rc,$FREE" >> $LOG
    if [ $rc -ne 0 ]; then echo "STOP: D2 failed: $arm fold $f (exit $rc)"; exit 1; fi
    if [ ! -f $ROOT/diag_d2_${tag}_blank/fold${f}_seed0/run_log.md ]; then echo "STOP: D2 ended with exit 0 and no run_log.md: $arm fold $f"; exit 1; fi
  done
done
echo "D2 done $(date '+%F %T')"

plan=$($PY -W ignore scripts/queue_plan_13_8.py) || { echo "STOP: the queue plan failed $(date '+%F %T')"; exit 1; }
plan=$(printf '%s' "$plan" | tr -d '\r')
echo "plan: $(printf '%s' "$plan" | tr '\n' ';') $(date '+%F %T')"

while read -r cfg f s; do
  [ -n "$cfg" ] && run "$cfg" "$f" "$s"
done <<< "$plan"
echo "done $(date '+%F %T')"
