#!/bin/bash
cd /home/claude/work/NSCLC__bio
export PYTHONWARNINGS=ignore
echo "== E17 run $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pmtrd.revision.e17 run || { echo "E17 failed $?"; exit 1; }
echo "== E17 summarise $(date +%T)"; OMP_NUM_THREADS=1 python3 -m pmtrd.revision.e17 summarise || echo "E17 summarise failed $?"
echo "== E15 full B $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pmtrd.revision.e15 full --env B || { echo "E15B failed $?"; exit 1; }
echo "== E15 summarise $(date +%T)"; python3 -m pmtrd.revision.e15 summarise || echo "E15 summarise failed $?"
echo "== queue E done $(date +%T)"
