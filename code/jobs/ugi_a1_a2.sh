#!/usr/bin/env bash
# A1 and A2 (design note family A; Shafi's GO of 2 Nov 2026, night): 4 folds x 3 seeds each, after B3 and after the A1
# dry run passed. A2 needs the B3 checkpoint of the same fold and seed. Sequential on the local GPU; a run whose
# metrics.json exists is skipped (resume). Run from work_ms/:  bash jobs/ugi_a1_a2.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=runs_itareo/ugi/a1_a2_runner_log.csv
mkdir -p runs_itareo/ugi
[ -f $LOG ] || echo "config,fold,seed,start,end,exit" > $LOG
run() {  # config fold seed
  local cfg=$1 f=$2 s=$3 out=runs_itareo/ugi/$1/fold$2_seed$3
  if [ -f $out/metrics.json ]; then return; fi
  mkdir -p $out
  local t0=$(date '+%F %T')
  $PY -m itareo.ugi_seg.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out > $out/stdout.txt 2>&1
  local rc=$?
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc" >> $LOG
}
for s in 0 1 2; do for f in 1 2 3 4; do run ugi_a1 $f $s; done; done
for s in 0 1 2; do for f in 1 2 3 4; do run ugi_a2 $f $s; done; done
echo "done $(date '+%F %T')"
