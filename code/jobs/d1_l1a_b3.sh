#!/usr/bin/env bash
# Stage B3 (CLAUDE.md 12.3 f; Shafi's GO of 2 Nov 2026): main configuration 4 folds x 3 seeds, then the two ablations
# (boundary band not ignored; nominal-lattice imagery) on fold 1 x 3 seeds. Sequential on the local GPU; a run whose
# metrics.json exists is skipped (resume). Run from work_ms/:  bash jobs/d1_l1a_b3.sh
# Restarted after the third review's fixes; the earlier runs are in archive/2026-11-02/d1_l1a_before_review_fixes/.
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=runs_itareo/d1_l1a/b3_runner_log.csv
mkdir -p runs_itareo/d1_l1a
[ -f $LOG ] || echo "config,fold,seed,start,end,exit" > $LOG
run() {  # config fold seed
  local cfg=$1 f=$2 s=$3 out=runs_itareo/d1_l1a/$1/fold$2_seed$3
  if [ -f $out/metrics.json ]; then return; fi
  mkdir -p $out
  local t0=$(date '+%F %T')
  $PY -m itareo.unet_ugi.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out > $out/stdout.txt 2>&1
  local rc=$?                      # exit code of the training run (read before any other command)
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc" >> $LOG
}
for s in 0 1 2; do for f in 1 2 3 4; do run d1_l1a_main $f $s; done; done
for s in 0 1 2; do run d1_l1a_band_not_ignored 1 $s; done
if [ ! -f runs_itareo/d1_l1a/cache/nominal_harmonised/cache.json ]; then
  $PY -m itareo.unet_ugi.prepare --cfg itareo/configs/d1_l1a_nominal.yaml --cache > runs_itareo/d1_l1a/cache_nominal_stdout.txt 2>&1
fi
for s in 0 1 2; do run d1_l1a_nominal 1 $s; done
echo "done $(date '+%F %T')"
