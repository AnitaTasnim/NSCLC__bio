#!/bin/bash
# E9 external members on the second core, while queue C finishes the GSE89843 units.
# When queue C reaches its own external step, it is stopped so that the two do not duplicate work.
cd /home/claude/work/NSCLC__bio
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
(
  until grep -q "== E9 external run" runs_queueC.out; do sleep 2; done
  kids=$(pgrep -P 4804); kill 4804; sleep 0.5; [ -n "$kids" ] && kill $kids
  echo "== queue C stopped at its external step $(date +%T); external run continues in queue F"
) &
echo "== E9 external run (queue F) $(date +%T)"; python3 -m pmtrd.revision.e9_external run || { echo "E9x failed $?"; exit 1; }
wait
echo "== queue F external done $(date +%T)"
