#!/bin/bash
# Re-draw the supplementary figures into figures/build/ (run from anywhere inside the repository).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PMTRD_ROOT="$ROOT"
cd "$ROOT/figures/code"
python make_all.py figS fig_funnel fig_benchmark "$@"
echo "Figures written to $ROOT/figures/build/"
