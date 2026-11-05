#!/usr/bin/env bash
# E1, PlanetScope 2022 at the 4 m stage (CLAUDE.md 13.6): itareo/configs/ugi_e1_ps.yaml, seed 0, folds 1-4, one job at a time, 60 epochs. Starts only if the
# self-test and the baseline regression passed. Stops on a failed run, on a partial run folder (train.py refuses to overwrite), or below 30 GB free. Prints "done" at the
# end or "STOP: ..." on a stop.
# Run from work_ms/:  bash jobs/ugi_e1.sh >> runs_itareo/ugi/e1_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
LOG=$ROOT/e1_log.csv
SELFTEST=../docs/evidence_phase2/ugi_seg/selftest_e1_run2/e1_selftest.json
REGRESSION=$ROOT/e1_regression/e1_regression.json
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG

ok=$($PY -c "
import json
s = json.load(open('$SELFTEST')); r = json.load(open('$REGRESSION'))
print(int(s['passed'] is True and r['baseline_equal'] is True and 'e1_scored' in r))" 2>/dev/null | tr -d '\r\n')
if [ "$ok" != "1" ]; then echo "STOP: the self-test or the regression did not pass; nothing started"; exit 1; fi

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

echo "self-test and regression passed $(date '+%F %T')"
for f in 1 2 3 4; do run ugi_e1_ps $f 0; done
echo "done $(date '+%F %T')"
