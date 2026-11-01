#!/usr/bin/env bash
# Reproduce every table and figure of the paper.
#
# Usage:   bash run_all.sh /path/to/2022_abi_stat.xlsx [output_dir]
# Quick functional test (1 CV repeat, 10 bootstrap/permutation refits; numbers will differ from the paper):
#          QUICK=1 bash run_all.sh /path/to/2022_abi_stat.xlsx test_output
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash run_all.sh /path/to/2022_abi_stat.xlsx [output_dir]"; exit 1
fi
SRC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_ARG="${2:-results}"
mkdir -p "$OUT_ARG"
OUT="$(cd "$OUT_ARG" && pwd)"
mkdir -p "$OUT"/{interp,reg,tables,figs}

if [ "${QUICK:-0}" = "1" ]; then
  REP_MAIN=1; REP_DUR=1; BOOT=10; HPERM=10
else
  REP_MAIN=10; REP_DUR=2; BOOT=200; HPERM=100
fi

cd "$(dirname "$0")/scripts"
python 01_prepare_cohort.py "$SRC" "$OUT"
python 02_nested_cv.py "$OUT/analytic_cohort.csv" "$OUT" --repeats "$REP_MAIN" --tag main     # about 1-2 h on 2 cores
Rscript ../R/03_mars_nested.R "$OUT/main"
python 02_nested_cv.py "$OUT/analytic_cohort.csv" "$OUT" --repeats "$REP_DUR" --with-duration --tag duration
Rscript ../R/03_mars_nested.R "$OUT/duration"
python 04_interpretation.py "$OUT/analytic_cohort.csv" "$OUT/interp" --boot "$BOOT" --hperm "$HPERM"
Rscript ../R/06_mars_full.R "$OUT/interp"
python 05_regression.py "$OUT/analytic_cohort.csv" "$SRC" "$OUT/reg"
python 08_tables.py "$OUT/analytic_cohort.csv" "$OUT/main" "$OUT/tables"
python 09_sensitivity_table.py "$OUT/main" "$OUT/duration" "$OUT/tables/tableS1_duration.csv"
python 07a_schematics.py "$OUT/figs" "$OUT/cohort_flow.json"
python 07b_results_figures.py "$OUT/main" "$OUT/interp" "$OUT/reg" "$OUT/figs"
echo "Done. Outputs are in $OUT"
