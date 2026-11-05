#!/usr/bin/env bash
# CPU side of E1 (CLAUDE.md 13.6). Rules 7 and 9 on each fold as it finishes, and the B4 tile set inside the ejido, for E1 and for the baseline's seed 0. When jobs/ugi_e1.sh prints "done":
# the paired summaries and the pre-registered reading (scripts/exp_report.py). No partial folds as a result: nothing is summarised if it prints STOP. Stops, with a "STOP:" line in its log, if a step fails.
# Run from work_ms/:  bash jobs/ugi_e1_post.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=-1 GDAL_PAM_ENABLED=NO
ROOT=runs_itareo/ugi
OUT=$ROOT/e1_post_stdout.txt
Q=$ROOT/e1_stdout.txt
SUM=$ROOT/summary_r
ARM=$ROOT/ugi_e1_ps
BASE="$ROOT/ugi_r1_none/fold1_seed0 $ROOT/ugi_r1_none/fold2_seed0 $ROOT/ugi_r1_none/fold3_seed0 $ROOT/ugi_r1_none/fold4_seed0"

rescore() {
  $PY -W ignore -m itareo.ugi_seg.rescore7 --runs $ARM >> $OUT 2>&1 || { echo "STOP: rule 7 failed $(date '+%F %T')" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.rescore9 --runs $ARM >> $OUT 2>&1 || { echo "STOP: rule 9 failed $(date '+%F %T')" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.rescore9 --b4 --runs $ARM $BASE >> $OUT 2>&1 || { echo "STOP: B4 tile set failed $(date '+%F %T')" >> $OUT; exit 1; }
}

ended() { grep -E '^(self-test|done|STOP)' $Q 2>/dev/null | tail -1 | grep -q '^done\|^STOP'; }

echo "e1 post start $(date '+%F %T')" >> $OUT
until ended; do
  rescore
  sleep 120
done
rescore
if grep -E '^(self-test|done|STOP)' $Q | tail -1 | grep -q '^done'; then
  $PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_e1_ps --seeds 0 --ejido --out $SUM --tag e1_ejido >> $ROOT/summary_e1_stdout.txt 2>&1 || { echo "STOP: summary e1_ejido failed" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_e1_ps --seeds 0 --out $SUM --tag e1 >> $ROOT/summary_e1_stdout.txt 2>&1 || { echo "STOP: summary e1 failed" >> $OUT; exit 1; }
  $PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_e1_ps --seeds 0 --sub ejido_b4 --out $SUM --tag e1_ejido_b4 >> $ROOT/summary_e1_stdout.txt 2>&1 || { echo "STOP: summary e1_ejido_b4 failed" >> $OUT; exit 1; }
  $PY -W ignore scripts/exp_report.py --arm ugi_e1_ps --tag e1 --out ../docs/evidence_phase2/ugi_seg/e1 --b4 >> $OUT 2>&1 || { echo "STOP: reading failed" >> $OUT; exit 1; }
fi
echo "e1 post done $(date '+%F %T')" >> $OUT
