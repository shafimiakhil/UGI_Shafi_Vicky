#!/usr/bin/env bash
# Stage R, R1 and R1b (CLAUDE.md 13.4; Shafi's GO): A1 on the v3 mosaic, 60 epochs, edge_band both / outer / inner / none,
# seed 0, folds 1-4 (16 runs), then R1b = ugi_r1b_both_e120 fold 1 seed 0 (120 epochs). One job at a time (F4). A run
# whose metrics.json exists is skipped (resume). Stops on a failed run or when free disk on C: is below 30 GB.
# Run from work_ms/:  bash jobs/ugi_r1.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=runs_itareo/ugi/r1_runner_log.csv
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG
run() {  # config fold seed
  local cfg=$1 f=$2 s=$3 out=runs_itareo/ugi/$1/fold$2_seed$3
  if [ -f $out/metrics.json ]; then return 0; fi
  local free
  free=$(df -BG --output=avail /c | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 30 ]; then echo "STOP: free disk ${free} GB < 30 GB before $cfg fold $f"; exit 2; fi
  mkdir -p $out
  local t0
  t0=$(date '+%F %T')
  $PY -m itareo.ugi_seg.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out > $out/stdout.txt 2>&1
  local rc=$?
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc,$free" >> $LOG
  if [ $rc -ne 0 ]; then echo "STOP: run failed: $cfg fold $f seed $s (exit $rc)"; exit 1; fi
}
for f in 1 2 3 4; do for arm in both outer inner none; do run ugi_r1_$arm $f 0; done; done
run ugi_r1b_both_e120 1 0
echo "done $(date '+%F %T')"
