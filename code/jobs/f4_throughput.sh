#!/usr/bin/env bash
# F4 (CLAUDE.md 13.3): 3 epochs of one job alone, then two at once; GPU memory sampled every 0.5 s by nvidia-smi.
# Decision (13.3): two concurrent jobs if aggregate throughput rises by >= 30 % with peak GPU memory < 20 GB.
set -u
PY=/c/Users/SHAFI/anaconda3/envs/cordoba/python.exe
export PYTHONUTF8=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT=runs_itareo/ugi/ugi_f4_throughput
mkdir -p $OUT
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 500 > $OUT/nvidia_smi.csv &
SMI=$!
date '+alone_start %F %T' >> $OUT/phases.txt
$PY -m itareo.ugi_seg.train --cfg itareo/configs/ugi_f4_throughput.yaml --fold 1 --seed 0 --out $OUT/alone_fold1_seed0 > $OUT/alone.txt 2>&1
echo "alone_exit $?" >> $OUT/phases.txt
date '+pair_start %F %T' >> $OUT/phases.txt
$PY -m itareo.ugi_seg.train --cfg itareo/configs/ugi_f4_throughput.yaml --fold 1 --seed 0 --out $OUT/pair_a_fold1_seed0 > $OUT/pair_a.txt 2>&1 &
A=$!
$PY -m itareo.ugi_seg.train --cfg itareo/configs/ugi_f4_throughput.yaml --fold 2 --seed 0 --out $OUT/pair_b_fold2_seed0 > $OUT/pair_b.txt 2>&1 &
B=$!
wait $A; echo "pair_a_exit $?" >> $OUT/phases.txt
wait $B; echo "pair_b_exit $?" >> $OUT/phases.txt
date '+end %F %T' >> $OUT/phases.txt
kill $SMI
echo done
