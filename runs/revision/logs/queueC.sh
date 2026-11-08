#!/bin/bash
cd /home/claude/work/NSCLC__bio
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "== E9 run $(date +%T)"; python3 -m pmtrd.revision.e9 run --datasets GSE89843 --n-jobs 1 || { echo "E9 failed $?"; exit 1; }
echo "== E9 external run $(date +%T)"; python3 -m pmtrd.revision.e9_external run || { echo "E9x failed $?"; exit 1; }
echo "== E9 summarise $(date +%T)"; python3 -m pmtrd.revision.e9 summarise || echo "E9 summarise failed $?"
echo "== E9 external summarise $(date +%T)"; python3 -m pmtrd.revision.e9_external summarise || echo "E9x summarise failed $?"
echo "== queue C done $(date +%T)"
