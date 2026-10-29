#!/usr/bin/env bash
# One command: round 1 (preprocessing -> features -> baselines -> deep model -> evaluation ->
# error analysis / calibration -> controls -> pre-registered bar) and round 2 (grouped CV on
# the 22 non-test nights, the added configurations, the leak checks, and reproduction of the
# shipped predictions.csv from the recorded selection). Place the data at data/ECG_P1 (or set
# ECG_DATA_DIR). Pass --no-cnn to skip the CNN re-trainings inside the control stages on a slow CPU.
# The round-2 test scoring itself (s15 --confirm) is NOT re-run: the selection, seed, smoothing
# constant and threshold recorded in outputs/round2_test.json are reproduced without computing
# a test metric.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
PY=${PYTHON:-python}
"$PY" "$HERE/src/s1_data.py"
"$PY" "$HERE/src/s3_features.py"
"$PY" "$HERE/src/s2_split.py"      # after s3: its regression test also checks the cached features
"$PY" "$HERE/src/s4_baseline.py"
"$PY" "$HERE/src/s5_deep.py"
"$PY" "$HERE/src/s6_evaluate.py"
"$PY" "$HERE/src/s7_errors_calib.py"
"$PY" "$HERE/src/s8_controls.py" "$@"
"$PY" "$HERE/src/s9_posthoc_controls.py" "$@"
"$PY" "$HERE/src/s10_prereg_bar.py"
mv -f "$HERE/predictions.csv" "$HERE/predictions_round1.csv"
"$PY" "$HERE/src/s14_round2_cv.py"
"$PY" "$HERE/src/s14b_round2_cv_extra.py"
"$PY" "$HERE/src/s15_round2_final.py" --dry-run
"$PY" "$HERE/src/s15_round2_final.py" --reproduce-only --out predictions.csv
"$PY" "$HERE/src/s17_leak_checks.py"
"$PY" "$HERE/src/s7_errors_calib.py" --round2
echo "Done. predictions.csv (round 2) and predictions_round1.csv written; see outputs/"
