#!/usr/bin/env bash
# Night CPU task 3 of CLAUDE.md 13.5, optional and last: scripts/s1_2021_2022.py starts once rule 9 has finished for the Stage M
# ensemble and every R1 run, that is, when jobs/ugi_night_post.sh has written the R1 summary inside the ejido; the surface
# composition and the review are done by then. Nothing starts if that job stopped.
# Run from work_ms/:  bash jobs/s1_after_rule9.sh >> runs_itareo/ugi/s1_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1
echo "waiting for rule 9 on Stage M and R1 $(date '+%F %T')"
until [ -f runs_itareo/ugi/summary_r/summary_r1_ejido.json ]; do
  if grep -q '^STOP' runs_itareo/ugi/night_post_stdout.txt 2>/dev/null; then echo "STOP: the night post job stopped; Sentinel-1 not started"; exit 1; fi
  sleep 60
done
echo "start $(date '+%F %T')"
$PY scripts/s1_2021_2022.py
rc=$?
echo "exit $rc $(date '+%F %T')"
exit $rc
