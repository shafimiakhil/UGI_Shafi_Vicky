#!/usr/bin/env bash
# One-seed maps of ugi_r3_a2_none (A2) and ugi_r1_none (A1), seed 0 (CLAUDE.md 8.3), built exactly as Stage M2, then one CSV with the pair's scores inside the ejido
# and sq8km, whose self-test is that each map's scores equal those of its four runs. CPU only, no inference. Stops on the first failure, with a "STOP:" line.
# Run from work_ms/:  bash jobs/maps_a2_pair.sh >> runs_itareo/ugi/maps_a2_stdout.txt 2>&1
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 GDAL_PAM_ENABLED=NO CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
M="$PY -W ignore -m itareo.ugi_seg.stage_m2"
T0=$(date +%s)
echo "start $(date '+%F %T')"
$PY -W ignore -m itareo.ugi_seg.rescore_centre --regions sq8km --runs runs_itareo/ugi/ugi_r3_a2_none --seeds 0 --workers 4 || { echo "STOP: rescore of A2 inside sq8km failed"; exit 1; }
$PY -W ignore -m itareo.ugi_seg.summarize_r --arms ugi_r1_none ugi_r3_a2_none --seeds 0 --sub centre_sq8km --tag r3_centre_sq8km || { echo "STOP: summary of A2 inside sq8km failed"; exit 1; }
echo "A2 inside sq8km done $(date '+%F %T')"
for w in a1 a2; do
  $M --one-seed $w --ensemble || { echo "STOP: $w ensemble failed"; exit 1; }
  $M --one-seed $w --score & P1=$!
  $M --one-seed $w --ejido & P2=$!
  $M --one-seed $w --centre & P3=$!
  $M --one-seed $w --qgis & P4=$!
  wait $P1; R1=$?
  wait $P2; R2=$?
  wait $P3; R3=$?
  wait $P4; R4=$?
  if [ $R1 -ne 0 ] || [ $R2 -ne 0 ] || [ $R3 -ne 0 ] || [ $R4 -ne 0 ]; then echo "STOP: $w score $R1, ejido $R2, centre $R3, qgis $R4"; exit 1; fi
  $M --one-seed $w --figures || { echo "STOP: $w figures failed"; exit 1; }
  echo "$w done $(date '+%F %T')"
done
$M --pair || { echo "STOP: pair failed"; exit 1; }
echo "done $(date '+%F %T'), $(( $(date +%s) - T0 )) s"
