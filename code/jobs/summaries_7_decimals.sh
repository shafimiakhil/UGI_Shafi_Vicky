#!/usr/bin/env bash
# The six Stage R summaries again with 7 decimals (itareo/ugi_seg/summarize_r.py), after the independent check of the morning report
# (CLAUDE.md 13.5): they read files only; the same arguments as jobs/ugi_r1_post.sh and jobs/ugi_night_post.sh. The 5-decimal versions must
# already be in archive/2026-11-04/summary_r_5decimals/. Then the sensitivity of the A2 reading. Stops on the first failure.
# Run from work_ms/:  bash jobs/summaries_7_decimals.sh
set -eu
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1
OUT=runs_itareo/ugi/summary_r
LOG=runs_itareo/ugi/summaries_7_decimals_stdout.txt
R1="ugi_r1_both ugi_r1_outer ugi_r1_inner ugi_r1_none"
$PY -m itareo.ugi_seg.summarize_r --arms $R1 --seeds 0 --out $OUT --tag r1 --decision --r1b >> $LOG 2>&1
$PY -m itareo.ugi_seg.summarize_r --arms $R1 --seeds 0 --ejido --decision --r1b --out $OUT --tag r1_ejido >> $LOG 2>&1
$PY -m itareo.ugi_seg.summarize_r --arms ugi_r1_none --seeds 0 1 2 --ejido --out $OUT --tag baseline_ejido >> $LOG 2>&1
$PY -m itareo.ugi_seg.summarize_r --arms ugi_r1_none --seeds 0 1 2 --out $OUT --tag baseline >> $LOG 2>&1
$PY -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_r3_a2_none --seeds 0 --ejido --out $OUT --tag r3_ejido >> $LOG 2>&1
$PY -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_r3_a2_none --seeds 0 --out $OUT --tag r3 >> $LOG 2>&1
echo "summaries done $(date '+%F %T')"
$PY scripts/r3_interval_sensitivity.py >> $LOG 2>&1
echo "sensitivity done $(date '+%F %T')"
