# Error analysis and calibration (test split, round-1 final model per task)

## Task A — cnn seed 0, threshold 0.007

### Per record

| record | n | prevalence | AUPRC | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|---|
| a16 | 481 | 0.67 | 0.916 | 0.838 | 1.000 | 0.975 | 157 | 0 |
| a18 | 489 | 0.90 | 0.969 | 0.810 | 1.000 | 1.000 | 51 | 0 |
| a19 | 502 | 0.41 | 0.422 | 0.509 | 0.000 | 0.000 | 0 | 205 |
| a20 | 510 | 0.62 | 0.720 | 0.656 | 0.000 | 0.000 | 0 | 315 |
| b05 | 432 | 0.13 | 0.086 | 0.266 | 0.000 | 0.037 | 14 | 57 |
| c08 | 513 | 0.00 | – | – | – | 0.766 | 393 | 0 |
| c09 | 467 | 0.00 | 0.006 | 0.362 | 0.000 | 0.013 | 6 | 2 |
| c10 | 430 | 0.00 | 0.111 | 0.981 | 1.000 | 0.340 | 146 | 0 |
Quality axis: beat-template SQI of the last input minute; tercile sizes {'worst': 1275, 'best': 1275, 'mid': 1274}; AUROC(SQI -> label) on train = 0.435 (0.5 = quality unrelated to the label). Flags: 98 flat/clipped minutes, 22 implausible-beat minutes, 18 minutes with < 3 beats (SQI set to 0, so they sit in the worst tercile).


### by_quality_tercile(sqi)

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| worst | 1275 | 0.29 | 0.416 | 0.037 | 0.231 | 208 | 361 |
| mid | 1274 | 0.34 | 0.659 | 0.594 | 0.280 | 237 | 174 |
| best | 1275 | 0.42 | 0.886 | 0.918 | 0.435 | 322 | 44 |

### by_beats_plausible

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 22 | 0.00 | – | – | 0.455 | 10 | 0 |
| True | 3802 | 0.35 | 0.660 | 0.567 | 0.307 | 757 | 579 |

### by_flat_or_clipped

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3726 | 0.36 | 0.663 | 0.570 | 0.303 | 728 | 570 |
| True | 98 | 0.13 | 0.340 | 0.308 | 0.459 | 39 | 9 |

### by_no_running_baseline

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3784 | 0.35 | 0.660 | 0.569 | 0.307 | 752 | 575 |
| True | 40 | 0.10 | 0.118 | 0.000 | 0.417 | 15 | 4 |

### by_boundary_minute

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3588 | 0.34 | 0.674 | 0.587 | 0.309 | 730 | 506 |
| True | 236 | 0.47 | 0.510 | 0.348 | 0.298 | 37 | 73 |

### by_minutes_since_last_apnea

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| <=1 | 1396 | 0.96 | 0.677 | 0.567 | 0.310 | 18 | 579 |
| 2-3 | 94 | 0.00 | – | – | 0.266 | 25 | 0 |
| 4-10 | 210 | 0.00 | – | – | 0.233 | 49 | 0 |
| 11-30 | 301 | 0.00 | – | – | 0.150 | 45 | 0 |
| >30 / never | 1823 | 0.00 | – | – | 0.346 | 630 | 0 |

### by_night_third

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| first | 1274 | 0.26 | 0.656 | 0.538 | 0.293 | 277 | 152 |
| middle | 1275 | 0.38 | 0.658 | 0.590 | 0.321 | 255 | 197 |
| last | 1267 | 0.42 | 0.656 | 0.564 | 0.315 | 233 | 230 |

Excluding boundary minutes: AUPRC 0.561, AUROC 0.674, F1 0.538 (n=3588).
R-peak miss rate vs .qrs by quality tercile: worst: 0.566, mid: 0.294, best: 0.076
R-peak miss rate on FP minutes 0.031 vs TN minutes 0.353.

Calibration: raw Brier 0.290, ECE 0.304; after isotonic (fit on val) Brier 0.248, ECE 0.253.

## Task B — xgb_rel seed 2, threshold 0.430

### Per record

| record | n | prevalence | AUPRC | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|---|
| a16 | 156 | 0.08 | 0.252 | 0.649 | 0.154 | 0.021 | 3 | 11 |
| a18 | 47 | 0.13 | 0.274 | 0.443 | 0.167 | 0.073 | 3 | 5 |
| a19 | 296 | 0.05 | 0.574 | 0.944 | 0.067 | 0.000 | 0 | 14 |
| a20 | 190 | 0.08 | 0.297 | 0.774 | 0.125 | 0.017 | 3 | 14 |
| b05 | 371 | 0.02 | 0.031 | 0.645 | 0.000 | 0.008 | 3 | 7 |
| c08 | 508 | 0.00 | – | – | – | 0.026 | 13 | 0 |
| c09 | 460 | 0.00 | 0.014 | 0.738 | 0.000 | 0.009 | 4 | 2 |
| c10 | 424 | 0.00 | 0.004 | 0.378 | 0.000 | 0.005 | 2 | 1 |
Quality axis: beat-template SQI of the last input minute; tercile sizes {'worst': 818, 'mid': 817, 'best': 817}; AUROC(SQI -> label) on train = 0.544 (0.5 = quality unrelated to the label). Flags: 243 flat/clipped minutes, 5 implausible-beat minutes, 6 minutes with < 3 beats (SQI set to 0, so they sit in the worst tercile).


### by_quality_tercile(sqi)

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| worst | 818 | 0.03 | 0.752 | 0.080 | 0.019 | 15 | 23 |
| mid | 817 | 0.02 | 0.708 | 0.150 | 0.009 | 7 | 17 |
| best | 817 | 0.02 | 0.661 | 0.067 | 0.011 | 9 | 14 |

### by_beats_plausible

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 5 | 0.00 | – | – | 0.000 | 0 | 0 |
| True | 2447 | 0.02 | 0.713 | 0.100 | 0.013 | 31 | 54 |

### by_flat_or_clipped

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2209 | 0.02 | 0.683 | 0.094 | 0.013 | 29 | 48 |
| True | 243 | 0.03 | 0.913 | 0.143 | 0.008 | 2 | 6 |

### by_no_running_baseline

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2415 | 0.02 | 0.719 | 0.102 | 0.013 | 30 | 53 |
| True | 37 | 0.03 | 0.472 | 0.000 | 0.028 | 1 | 1 |

### by_boundary_minute

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2341 | 0.00 | – | – | 0.013 | 30 | 0 |
| True | 111 | 0.54 | 0.675 | 0.100 | 0.020 | 1 | 54 |

### by_minutes_since_last_apnea

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| <=1 | 57 | 0.11 | 0.748 | 0.167 | 0.020 | 1 | 5 |
| 2-3 | 94 | 0.15 | 0.685 | 0.143 | 0.013 | 1 | 12 |
| 4-10 | 209 | 0.06 | 0.861 | 0.077 | 0.010 | 2 | 12 |
| 11-30 | 301 | 0.04 | 0.827 | 0.091 | 0.000 | 0 | 10 |
| >30 / never | 1791 | 0.01 | 0.601 | 0.062 | 0.015 | 27 | 15 |

### by_night_third

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| first | 924 | 0.02 | 0.679 | 0.053 | 0.022 | 20 | 18 |
| middle | 794 | 0.02 | 0.735 | 0.211 | 0.012 | 9 | 15 |
| last | 734 | 0.03 | 0.736 | 0.045 | 0.003 | 2 | 21 |

Excluding boundary minutes: not defined for Task B (every onset is a boundary minute by construction).
R-peak miss rate vs .qrs by quality tercile: worst: 0.441, mid: 0.248, best: 0.077
R-peak miss rate on FP minutes 0.050 vs TN minutes 0.249.

Calibration: raw Brier 0.028, ECE 0.023; after isotonic (fit on val) Brier 0.023, ECE 0.006.
