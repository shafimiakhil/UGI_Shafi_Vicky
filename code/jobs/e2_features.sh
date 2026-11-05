#!/usr/bin/env bash
# E2 inputs (CLAUDE.md 13.6), CPU only: the PAN lawn layers on v3 (self-test first; strip 1_3 and the merge only if it passes) and, beside them, the NDVI texture. Prints "STOP: ..." on a failure.
# Run from work_ms/:  bash jobs/e2_features.sh > runs_itareo/ugi/e2_features_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=-1 GDAL_PAM_ENABLED=NO
echo "start $(date '+%F %T')"
( $PY -W ignore scripts/e2_ndvi_texture.py && echo "ndvi texture done $(date '+%F %T')" || echo "STOP: ndvi texture failed $(date '+%F %T')" ) &
$PY -W ignore scripts/e2_pan_lawn_v3.py --selftest || { echo "STOP: PAN self-test failed $(date '+%F %T')"; wait; exit 1; }
echo "PAN self-test passed $(date '+%F %T')"
$PY -W ignore scripts/e2_pan_lawn_v3.py --strip13 || { echo "STOP: strip 1_3 failed $(date '+%F %T')"; wait; exit 1; }
$PY -W ignore scripts/e2_pan_lawn_v3.py --merge || { echo "STOP: merge failed $(date '+%F %T')"; wait; exit 1; }
echo "PAN done $(date '+%F %T')"
wait
echo "end $(date '+%F %T')"
