#!/bin/bash
# resumes queue B after its 2-hour stop: every step skips units already in its cache
cd /home/claude/work/NSCLC__bio
export PYTHONWARNINGS=ignore
echo "== E15 full A (resume) $(date +%T)"; python3 -m pmtrd.revision.e15 full --env A || { echo "E15A failed $?"; exit 1; }
echo "== E15 nested $(date +%T)"; python3 -m pmtrd.revision.e15 nested || { echo "E15n failed $?"; exit 1; }
echo "== E17 run $(date +%T)"; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pmtrd.revision.e17 run || { echo "E17 failed $?"; exit 1; }
echo "== E17 summarise $(date +%T)"; OMP_NUM_THREADS=1 python3 -m pmtrd.revision.e17 summarise || echo "E17 summarise failed $?"
echo "== queue D done $(date +%T)"
