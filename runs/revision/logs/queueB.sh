#!/bin/bash
cd /home/claude/work/NSCLC__bio
export PYTHONWARNINGS=ignore
echo "== E10 $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pmtrd.revision.e10 || echo "E10 failed $?"
echo "== E8 $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pmtrd.revision.e8 || echo "E8 failed $?"
echo "== E15 full A $(date +%T)"; python3 -m pmtrd.revision.e15 full --env A || echo "E15A failed $?"
echo "== E15 nested $(date +%T)"; python3 -m pmtrd.revision.e15 nested || echo "E15n failed $?"
echo "== E17 run $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pmtrd.revision.e17 run || echo "E17 failed $?"
echo "== E15 full B $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pmtrd.revision.e15 full --env B || echo "E15B failed $?"
echo "== queue B done $(date +%T)"
