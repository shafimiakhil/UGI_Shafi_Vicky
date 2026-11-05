#!/usr/bin/env bash
# GPU side of the night plan (CLAUDE.md 13.5). Waits until jobs/ugi_r1.sh and jobs/ugi_r1_post.sh have written the R1 and
# the R1b decisions in runs_itareo/ugi/summary_r/summary_r1.json, runs the edge-band self-test, then, one job at a time,
# folds 1-4, 60 epochs:
#   R2a  the R1 winner's config unchanged, seed 1
#   R3   A2 (D1 L1a encoder of the same fold, seed 0) with the winner's edge band
#   R2b  the winner, seed 2
# R1b is reported, not acted on. Nothing starts if the R1 runner printed STOP. Stops on a failed run or self-test, on a
# partial run folder (train.py refuses to overwrite), or below 30 GB free. Prints "done" at the end or "STOP: ..." on a stop.
# Run from work_ms/:  bash jobs/ugi_night_queue.sh >> runs_itareo/ugi/night_queue_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT=runs_itareo/ugi
LOG=$ROOT/night_queue_log.csv
DEC=$ROOT/summary_r/summary_r1.json
RUNNER=$ROOT/r1_runner_stdout.txt
[ -f $LOG ] || echo "config,fold,seed,start,end,exit,free_gb_before" > $LOG

winner() {  # the winner once both decisions are in a complete JSON; empty otherwise
  $PY -c "
import json
d = json.load(open('$DEC'))
w = d['r1_decision']['winner']
d['r1b_decision']['decision']
assert w in ('ugi_r1_both', 'ugi_r1_outer', 'ugi_r1_inner', 'ugi_r1_none')
print(w)" 2>/dev/null | tr -d '\r\n'
}

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
  $PY -m itareo.ugi_seg.train --cfg itareo/configs/$cfg.yaml --fold $f --seed $s --out $out > $out/stdout.txt 2>&1
  local rc=$?
  echo "$cfg,$f,$s,$t0,$(date '+%F %T'),$rc,$free" >> $LOG
  if [ $rc -ne 0 ]; then echo "STOP: run failed: $cfg fold $f seed $s (exit $rc)"; exit 1; fi
}

echo "waiting for the R1 and R1b decisions $(date '+%F %T')"
W=""
SAID=""
until [ -n "$W" ]; do
  if grep -q '^STOP' $RUNNER 2>/dev/null; then echo "STOP: the R1 runner stopped; nothing started on the GPU"; exit 1; fi
  if [ -f $DEC ]; then W=$(winner); fi
  if [ -z "$W" ]; then
    if [ -z "$SAID" ] && grep -q '^summary exit [1-9]' $ROOT/r1_post_stdout.txt 2>/dev/null; then
      echo "still waiting: the R1 summary failed, see $ROOT/r1_post_stdout.txt and $ROOT/summary_r1_stdout.txt; it is to be fixed and rerun"
      SAID=1
    fi
    sleep 60
  fi
done
BAND=${W#ugi_r1_}
R3=ugi_r3_a2_$BAND
echo "decision found $(date '+%F %T'): winner $W, edge band $BAND, R3 config $R3"
if [ ! -f itareo/configs/$W.yaml ] || [ ! -f itareo/configs/$R3.yaml ]; then echo "STOP: config missing for $W or $R3"; exit 1; fi
for f in 1 2 3 4; do
  if [ ! -f runs_itareo/d1_l1a/d1_l1a_main/fold${f}_seed0/best.pt ]; then echo "STOP: D1 L1a encoder missing for fold $f seed 0"; exit 1; fi
done
ST=$ROOT/selftest_night_$(date '+%Y%m%d_%H%M%S')
$PY -m itareo.ugi_seg.selftest_edge --out $ST > ${ST}_stdout.txt 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "STOP: self-test failed (exit $rc), see ${ST}_stdout.txt"; exit 1; fi
echo "self-test passed: $ST/edge_band_selftest.json"

for f in 1 2 3 4; do run $W $f 1; done
for f in 1 2 3 4; do run $R3 $f 0; done
for f in 1 2 3 4; do run $W $f 2; done
echo "done $(date '+%F %T')"
