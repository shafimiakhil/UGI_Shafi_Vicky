#!/usr/bin/env bash
# CPU side of E2 (CLAUDE.md 13.6). Rules 7 and 9 on each fold as it finishes. When jobs/ugi_e2.sh prints "done": the paired summaries and the pre-registered reading (scripts/exp_report.py).
# No partial folds as a result: nothing is summarised if it prints STOP. Stops, with a "STOP:" line in its log, if a step fails. It idles until E2 has started.
# Run from work_ms/:  bash jobs/ugi_e2_post.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=-1 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
OUT=$ROOT/e2_post_stdout.txt
Q=$ROOT/e2_stdout.txt
SUM=$ROOT/summary_r
ARM=$ROOT/ugi_e2_tex

rescore() {
  $PY -W ignore -m itareo.ugi_seg.rescore7 --runs $ARM >> $OUT 2>&1 || { echo "STOP: rule 7 failed $(date '+%F %T')" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.rescore9 --runs $ARM >> $OUT 2>&1 || { echo "STOP: rule 9 failed $(date '+%F %T')" >> $OUT; exit 1; }
}

ended() { grep -E '^(self-test|done|STOP)' $Q 2>/dev/null | tail -1 | grep -q '^done\|^STOP'; }

echo "e2 post start $(date '+%F %T')" >> $OUT
until [ -d $ARM ] || ended; do sleep 120; done
until ended; do
  rescore
  sleep 120
done
if [ -d $ARM ]; then rescore; fi
if grep -E '^(self-test|done|STOP)' $Q | tail -1 | grep -q '^done'; then
  $PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_e2_tex --seeds 0 --ejido --out $SUM --tag e2_ejido >> $ROOT/summary_e2_stdout.txt 2>&1 || { echo "STOP: summary e2_ejido failed" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_e2_tex --seeds 0 --out $SUM --tag e2 >> $ROOT/summary_e2_stdout.txt 2>&1 || { echo "STOP: summary e2 failed" >> $OUT; exit 1; }
  $PY -W ignore scripts/exp_report.py --arm ugi_e2_tex --tag e2 --out ../docs/evidence_phase2/ugi_seg/e2 >> $OUT 2>&1 || { echo "STOP: reading failed" >> $OUT; exit 1; }
fi
echo "e2 post done $(date '+%F %T')" >> $OUT
