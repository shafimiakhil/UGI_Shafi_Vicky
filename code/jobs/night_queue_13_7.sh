#!/usr/bin/env bash
# The night queue of CLAUDE.md 13.7, one job at a time: E1b folds 1-4 seed 0; fold C of ugi_r1_none, seeds 0-2, for the chosen centre; E3 folds 1-4 seed 0; then seeds 1-2 of whatever passed (scripts/pass_list.py).
# Waits for E2 to print "done"; does not start if E2 stopped or if a self-test or regression did not pass. The centre is the one line of derived/ugi_centre/chosen.txt, written on Shafi's word: if it does not exist when fold C's turn comes, E3 goes first and fold C follows when it exists.
# Before seeds 1-2 it waits for the readings of E1, E2, E1b and E3 (scripts/night_post_13_7.py, started separately). Stops on a failed run, on a partial run folder (train.py refuses to overwrite), or below 30 GB free. Prints "done" at the end or "STOP: ..." on a stop.
# Run from work_ms/:  bash jobs/night_queue_13_7.sh >> runs_itareo/ugi/night2_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
LOG=$ROOT/night2_log.csv
E2=$ROOT/e2_stdout.txt
CHOSEN=derived/ugi_centre/chosen.txt
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG

ok=$($PY -c "
import json
a = json.load(open('../docs/evidence_phase2/ugi_seg/selftest_e1b/e1_selftest.json'))['passed'] is True
b = json.load(open('../docs/evidence_phase2/ugi_seg/e1b_ps_plane_run2/residuals.json'))['passed'] is True
c = 'scored' in json.load(open('$ROOT/e1b_regression/regression.json'))
d = json.load(open('../docs/evidence_phase2/ugi_seg/selftest_e3/e3_selftest.json'))['passed'] is True
e = 'scored' in json.load(open('$ROOT/e3_regression/regression.json'))
f = 'scored' in json.load(open('$ROOT/foldc_regression_sq3km/regression.json'))
print(int(a and b and c and d and e and f))" 2>/dev/null | tr -d '\r\n')
if [ "$ok" != "1" ]; then echo "STOP: a self-test or regression did not pass; nothing started"; exit 1; fi
echo "self-tests and regressions passed; waiting for E2 $(date '+%F %T')"
until grep -qE '^(done|STOP)' $E2; do sleep 60; done
if ! grep -q '^done' $E2; then echo "STOP: E2 stopped; the queue did not start $(date '+%F %T')"; exit 1; fi

run() {  # config fold seed
  local cfg=$1 f=$2 s=$3 out=$ROOT/$1/fold$2_seed$3
  if [ -f $out/run_log.md ]; then echo "finished already: $cfg fold $f seed $s"; return 0; fi
  if [ -e $out ]; then echo "STOP: partial run folder $out"; exit 1; fi
  local free
  free=$(df -BG --output=avail /c | tail -1 | tr -dc 0-9)
  if ! [ "$free" -ge 30 ] 2>/dev/null; then echo "STOP: free disk '${free}' GB is below 30 GB or unreadable before $cfg fold $f seed $s"; exit 2; fi
  mkdir -p $out
  local t0
  t0=$(date '+%F %T')
  echo "start $cfg fold $f seed $s $t0"
  $PY -W ignore -m itareo.ugi_seg.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out > $out/stdout.txt 2>&1
  local rc=$?
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc,$free" >> $LOG
  if [ $rc -ne 0 ]; then echo "STOP: run failed: $cfg fold $f seed $s (exit $rc)"; exit 1; fi
}

fold_c() {
  local name
  name=$(tr -d '\r\n ' < $CHOSEN)
  [ -f itareo/configs/ugi_foldc_$name.yaml ] || { echo "STOP: no config for the chosen centre '$name'"; exit 1; }
  for s in 0 1 2; do run ugi_foldc_$name 5 $s; done
}

for f in 1 2 3 4; do run ugi_e1b_ps_plane $f 0; done
c_done=0
if [ -f $CHOSEN ]; then fold_c; c_done=1; else echo "fold C deferred: no centre chosen yet $(date '+%F %T')"; fi
for f in 1 2 3 4; do run ugi_e3_s2 $f 0; done
if [ $c_done -eq 0 ]; then
  until [ -f $CHOSEN ]; do sleep 300; done
  fold_c
fi
echo "waiting for the readings $(date '+%F %T')"
until [ -f ../docs/evidence_phase2/ugi_seg/e1/reading.md ] && [ -f ../docs/evidence_phase2/ugi_seg/e2/reading.md ] && [ -f ../docs/evidence_phase2/ugi_seg/e1b/reading.md ] && [ -f ../docs/evidence_phase2/ugi_seg/e3/reading.md ]; do sleep 120; done
arms=$($PY -W ignore scripts/pass_list.py) || { echo "STOP: pass_list failed $(date '+%F %T')"; exit 1; }
arms=$(printf '%s' "$arms" | tr -d '\r')
echo "seeds 1-2 of: ${arms:-none} $(date '+%F %T')"
for cfg in $arms; do
  for s in 1 2; do
    for f in 1 2 3 4; do run $cfg $f $s; done
  done
done
echo "done $(date '+%F %T')"
