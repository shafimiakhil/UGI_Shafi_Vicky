#!/usr/bin/env bash
# CPU side of the night plan (CLAUDE.md 13.5). Rule 9 for the Stage M ensemble and every R1 run, then rules 7 and 9 for each
# new run as it finishes (reads prob_oof.tif, never infers); the R1 summary inside the ejido once it can be made; when the
# GPU queue (jobs/ugi_night_queue.sh) prints "done" or "STOP", the three-seed baseline and A2 against A1, each inside the ejido
# and on the full grid, for whatever finished, then the morning report. Only the queue's latest start counts, so an earlier
# STOP in its appended stdout does not end a restarted night. Rule 7 waits for the end of jobs/ugi_r1_post.sh, which runs it on
# the R1 folders itself. Stops, with a "STOP:" line in its log, if a rescoring or a summary fails.
# Run from work_ms/:  bash jobs/ugi_night_post.sh
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1
ROOT=runs_itareo/ugi
OUT=$ROOT/night_post_stdout.txt
Q=$ROOT/night_queue_stdout.txt
R1POST=$ROOT/r1_post_stdout.txt
SUM=$ROOT/summary_r
DEC=$SUM/summary_r1.json
CFG="$ROOT/ugi_r1_both $ROOT/ugi_r1_outer $ROOT/ugi_r1_inner $ROOT/ugi_r1_none $ROOT/ugi_r1b_both_e120 $ROOT/ugi_r3_a2_both $ROOT/ugi_r3_a2_outer $ROOT/ugi_r3_a2_inner $ROOT/ugi_r3_a2_none"

rescore() {
  if grep -q '^post done' $R1POST 2>/dev/null; then
    $PY -m itareo.ugi_seg.rescore7 --runs $CFG >> $OUT 2>&1 || { echo "STOP: rule 7 rescoring failed $(date '+%F %T')" >> $OUT; exit 1; }
  fi
  $PY -m itareo.ugi_seg.rescore9 --stage-m --runs $CFG >> $OUT 2>&1 || { echo "STOP: rule 9 rescoring failed $(date '+%F %T')" >> $OUT; exit 1; }
}

summary() {  # tag, then the arguments of summarize_r; once only
  local tag=$1
  shift
  if [ -f $SUM/summary_$tag.json ]; then return 0; fi
  $PY -m itareo.ugi_seg.summarize_r "$@" --out $SUM --tag $tag >> $ROOT/summary_${tag}_stdout.txt 2>&1
  local rc=$?
  echo "summary $tag exit $rc $(date '+%F %T')" >> $OUT
  if [ $rc -ne 0 ]; then echo "STOP: summary $tag failed" >> $OUT; exit 1; fi
}

r1_inside() {  # when the decision and every R1 run's inside-ejido scores exist
  [ -f $DEC ] || return 0
  [ -f $ROOT/ugi_r1b_both_e120/fold1_seed0/ejido/done.json ] || return 0
  local a f
  for a in both outer inner none; do
    for f in 1 2 3 4; do
      [ -f $ROOT/ugi_r1_$a/fold${f}_seed0/ejido/done.json ] || return 0
    done
  done
  summary r1_ejido --arms ugi_r1_both ugi_r1_outer ugi_r1_inner ugi_r1_none --seeds 0 --ejido --decision --r1b
}

queue_ended() {  # the last status line of the queue's latest start is done or STOP
  grep -E '^(waiting for|done|STOP)' $Q 2>/dev/null | tail -1 | grep -q '^done\|^STOP'
}

have_runs() {  # config seed: all four folds carry their rule-7 and inside-ejido scores
  local f
  for f in 1 2 3 4; do
    [ -f $ROOT/$1/fold${f}_seed$2/metrics_rule7.json ] && [ -f $ROOT/$1/fold${f}_seed$2/boot_rule7.npz ] && [ -f $ROOT/$1/fold${f}_seed$2/ejido/done.json ] || return 1
  done
}

echo "night post start $(date '+%F %T')" >> $OUT
until queue_ended; do
  rescore
  r1_inside
  sleep 120
done
until grep -q '^post done' $R1POST 2>/dev/null; do sleep 60; done
rescore
r1_inside
if [ -f $DEC ]; then
  W=$($PY -c "import json; print(json.load(open('$DEC'))['r1_decision']['winner'])" | tr -d '\r\n')
  R3=ugi_r3_a2_${W#ugi_r1_}
  if have_runs $W 1 && have_runs $W 2; then
    summary baseline_ejido --arms $W --seeds 0 1 2 --ejido
    summary baseline --arms $W --seeds 0 1 2
  fi
  if have_runs $R3 0; then
    summary r3_ejido --arms $W $R3 --seeds 0 --ejido
    summary r3 --arms $W $R3 --seeds 0
  fi
fi
if [ -f scripts/night_report.py ]; then
  $PY scripts/night_report.py >> $OUT 2>&1
  echo "morning report exit $? $(date '+%F %T')" >> $OUT
fi
echo "night post done $(date '+%F %T')" >> $OUT
