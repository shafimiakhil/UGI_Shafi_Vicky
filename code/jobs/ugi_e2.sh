#!/usr/bin/env bash
# E2, optical texture in the stem at 1 m (CLAUDE.md 13.6): itareo/configs/ugi_e2_tex.yaml, seed 0, folds 1-4, one job at a time, 60 epochs. Waits for E1 to print "done" (jobs/ugi_e1.sh); does not start if E1 stopped, or if the
# self-test or the regression did not pass. Stops on a failed run, on a partial run folder (train.py refuses to overwrite), or below 30 GB free. Prints "done" at the end or "STOP: ..." on a stop.
# Run from work_ms/:  bash jobs/ugi_e2.sh >> runs_itareo/ugi/e2_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
LOG=$ROOT/e2_log.csv
E1=$ROOT/e1_stdout.txt
SELFTEST=../docs/evidence_phase2/ugi_seg/selftest_e2_run2/e2_selftest.json
REGRESSION=$ROOT/e2_regression/e2_regression.json
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG

ok=$($PY -c "
import json
s = json.load(open('$SELFTEST')); r = json.load(open('$REGRESSION'))
print(int(s['passed'] is True and r['baseline_equal'] is True and r['e1_equal'] is True and 'e2_scored' in r))" 2>/dev/null | tr -d '\r\n')
if [ "$ok" != "1" ]; then echo "STOP: the self-test or the regression did not pass; nothing started"; exit 1; fi
echo "self-test and regression passed; waiting for E1 $(date '+%F %T')"
until grep -qE '^(done|STOP)' $E1; do sleep 60; done
if ! grep -q '^done' $E1; then echo "STOP: E1 stopped; E2 not started $(date '+%F %T')"; exit 1; fi

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

for f in 1 2 3 4; do run ugi_e2_tex $f 0; done
echo "done $(date '+%F %T')"
