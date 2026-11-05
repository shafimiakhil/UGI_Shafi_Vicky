#!/usr/bin/env bash
# Rule-7 rescoring of the R1 / R1b runs as they finish (CPU, from prob_oof.tif), then, when jobs/ugi_r1.sh prints
# "done", the R1 summary with the decision rule and the R1b rule of CLAUDE.md 13.4. Nothing further if it prints STOP.
# Run from work_ms/:  bash jobs/ugi_r1_post.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1
OUT=runs_itareo/ugi/r1_post_stdout.txt
RUNS="runs_itareo/ugi/ugi_r1_both runs_itareo/ugi/ugi_r1_outer runs_itareo/ugi/ugi_r1_inner runs_itareo/ugi/ugi_r1_none runs_itareo/ugi/ugi_r1b_both_e120"
ST=runs_itareo/ugi/r1_runner_stdout.txt
until grep -q '^done\|^STOP' $ST 2>/dev/null; do
  $PY -m itareo.ugi_seg.rescore7 --runs $RUNS >> $OUT 2>&1
  sleep 300
done
$PY -m itareo.ugi_seg.rescore7 --runs $RUNS >> $OUT 2>&1
if grep -q '^done' $ST; then
  $PY -m itareo.ugi_seg.summarize_r --arms ugi_r1_both ugi_r1_outer ugi_r1_inner ugi_r1_none --seeds 0 \
    --out runs_itareo/ugi/summary_r --tag r1 --decision --r1b > runs_itareo/ugi/summary_r1_stdout.txt 2>&1
  echo "summary exit $?" >> $OUT
fi
echo "post done $(date '+%F %T')" >> $OUT
