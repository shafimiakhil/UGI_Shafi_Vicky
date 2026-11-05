#!/usr/bin/env bash
# After the A1/A2 queue (CLAUDE.md 13.2): waits for "done" in runs_itareo/ugi/a1_a2_runner_stdout.txt, then
#  1. the convergence check: A1, fold 1, seed 0, 60 epochs (Shafi's GO), in runs_itareo/ugi/ugi_a1_e60/fold1_seed0;
#  2. the matched evaluation: every A1, A2 and A1-60-epoch checkpoint re-scored on all held-out pixels, the B4 tile set
#     (1,240) and the B8 tile set (1,587), without retraining (itareo.ugi_seg.rescore; outputs in <run>/rescore/).
# Run from work_ms/:  bash jobs/ugi_after_queue.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG=runs_itareo/ugi/after_queue_log.csv
[ -f $LOG ] || echo "step,start,end,exit" > $LOG
until grep -q '^done' runs_itareo/ugi/a1_a2_runner_stdout.txt 2>/dev/null; do sleep 60; done
out=runs_itareo/ugi/ugi_a1_e60/fold1_seed0
if [ ! -f $out/metrics.json ]; then
  mkdir -p $out
  t0=$(date '+%F %T')
  $PY -m itareo.ugi_seg.train --cfg itareo/configs/ugi_a1.yaml --fold 1 --seed 0 --epochs 60 --out $out > $out/stdout.txt 2>&1
  rc=$?
  echo "a1_60_epochs_fold1_seed0,$t0,$(date '+%F %T'),$rc" >> $LOG
fi
t0=$(date '+%F %T')
$PY -m itareo.ugi_seg.rescore --runs runs_itareo/ugi/ugi_a1 runs_itareo/ugi/ugi_a2 runs_itareo/ugi/ugi_a1_e60 \
  > runs_itareo/ugi/rescore_stdout.txt 2>&1
rc=$?
echo "rescore,$t0,$(date '+%F %T'),$rc" >> $LOG
echo "done $(date '+%F %T')"
