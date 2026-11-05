#!/usr/bin/env bash
# Stage M2, the converged map (CLAUDE.md 13.6). CPU only, no inference. The ensemble first; then the two scorings and the QGIS layers side by side; then the
# figures and the table. Stops on the first failure, with a "STOP:" line. Prints the total time.
# Run from work_ms/:  bash jobs/stage_m2.sh >> runs_itareo/ugi/stage_m2_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 GDAL_PAM_ENABLED=NO CUDA_VISIBLE_DEVICES=-1
M="$PY -m itareo.ugi_seg.stage_m2"
T0=$(date +%s)
echo "start $(date '+%F %T')"
$M --ensemble || { echo "STOP: ensemble failed"; exit 1; }
echo "ensemble done $(date '+%F %T')"
$M --score & P1=$!
$M --ejido & P2=$!
$M --qgis & P3=$!
wait $P1; R1=$?
wait $P2; R2=$?
wait $P3; R3=$?
if [ $R1 -ne 0 ] || [ $R2 -ne 0 ] || [ $R3 -ne 0 ]; then echo "STOP: score $R1, ejido $R2, qgis $R3"; exit 1; fi
echo "scores and layers done $(date '+%F %T')"
$M --figures || { echo "STOP: figures failed"; exit 1; }
$M --before-after --readme || { echo "STOP: before_after failed"; exit 1; }
echo "done $(date '+%F %T'), $(( $(date +%s) - T0 )) s"
