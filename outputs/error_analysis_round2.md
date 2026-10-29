# Error analysis and calibration (test split, round-2 final model per task)

## Task A — ctx5_rz|shipped seed 2, threshold 0.252

### Per record

| record | n | prevalence | AUPRC | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|---|
| a16 | 481 | 0.67 | 0.968 | 0.931 | 0.947 | 0.404 | 65 | 17 |
| a18 | 489 | 0.90 | 0.968 | 0.794 | 0.950 | 0.667 | 34 | 22 |
| a19 | 502 | 0.41 | 0.971 | 0.973 | 0.824 | 0.034 | 10 | 36 |
| a20 | 510 | 0.62 | 0.929 | 0.905 | 0.854 | 0.205 | 40 | 46 |
| b05 | 432 | 0.13 | 0.776 | 0.937 | 0.632 | 0.035 | 13 | 21 |
| c08 | 513 | 0.00 | – | – | – | 0.224 | 115 | 0 |
| c09 | 467 | 0.00 | 0.025 | 0.775 | 0.500 | 0.075 | 35 | 1 |
| c10 | 430 | 0.00 | 0.003 | 0.263 | 0.000 | 0.075 | 32 | 1 |
Quality axis: beat-template SQI of the last input minute; tercile sizes {'worst': 1275, 'best': 1275, 'mid': 1274}; AUROC(SQI -> label) on train = 0.435 (0.5 = quality unrelated to the label). Flags: 98 flat/clipped minutes, 22 implausible-beat minutes, 18 minutes with < 3 beats (SQI set to 0, so they sit in the worst tercile).


### by_quality_tercile(sqi)

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| worst | 1275 | 0.29 | 0.942 | 0.813 | 0.084 | 76 | 70 |
| mid | 1274 | 0.34 | 0.959 | 0.904 | 0.141 | 119 | 41 |
| best | 1275 | 0.42 | 0.949 | 0.938 | 0.201 | 149 | 33 |

### by_beats_plausible

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 22 | 0.00 | – | – | 0.000 | 0 | 0 |
| True | 3802 | 0.35 | 0.948 | 0.892 | 0.140 | 344 | 144 |

### by_flat_or_clipped

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3726 | 0.36 | 0.948 | 0.893 | 0.142 | 340 | 142 |
| True | 98 | 0.13 | 0.978 | 0.846 | 0.047 | 4 | 2 |

### by_no_running_baseline

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3784 | 0.35 | 0.950 | 0.895 | 0.140 | 343 | 140 |
| True | 40 | 0.10 | 0.556 | 0.000 | 0.028 | 1 | 4 |

### by_boundary_minute

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 3588 | 0.34 | 0.962 | 0.921 | 0.126 | 298 | 97 |
| True | 236 | 0.47 | 0.673 | 0.580 | 0.371 | 46 | 47 |

### by_minutes_since_last_apnea

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| <=1 | 1396 | 0.96 | 0.867 | 0.892 | 0.448 | 26 | 144 |
| 2-3 | 94 | 0.00 | – | – | 0.277 | 26 | 0 |
| 4-10 | 210 | 0.00 | – | – | 0.152 | 32 | 0 |
| 11-30 | 301 | 0.00 | – | – | 0.090 | 27 | 0 |
| >30 / never | 1823 | 0.00 | – | – | 0.128 | 233 | 0 |

### by_night_third

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| first | 1274 | 0.26 | 0.938 | 0.851 | 0.129 | 122 | 49 |
| middle | 1275 | 0.38 | 0.947 | 0.892 | 0.131 | 104 | 52 |
| last | 1267 | 0.42 | 0.956 | 0.919 | 0.160 | 118 | 43 |

Excluding boundary minutes: AUPRC 0.930, AUROC 0.962, F1 0.851 (n=3588).
R-peak miss rate vs .qrs by quality tercile: worst: 0.566, mid: 0.294, best: 0.076
R-peak miss rate on FP minutes 0.153 vs TN minutes 0.270.

Calibration: raw Brier 0.088, ECE 0.043; after isotonic (fit on OOF) Brier 0.109, ECE 0.146.

## Task B — W5_r1_window|lr_C0.5 seed 1, threshold 0.732

### Per record

| record | n | prevalence | AUPRC | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|---|
| a16 | 156 | 0.08 | 0.289 | 0.664 | 0.692 | 0.441 | 63 | 4 |
| a18 | 47 | 0.13 | 0.151 | 0.524 | 0.000 | 0.049 | 2 | 6 |
| a19 | 296 | 0.05 | 0.159 | 0.830 | 0.400 | 0.107 | 30 | 9 |
| a20 | 190 | 0.08 | 0.162 | 0.626 | 0.250 | 0.098 | 17 | 12 |
| b05 | 371 | 0.02 | 0.021 | 0.508 | 0.000 | 0.118 | 43 | 7 |
| c08 | 508 | 0.00 | – | – | – | 0.110 | 56 | 0 |
| c09 | 460 | 0.00 | 0.030 | 0.888 | 0.000 | 0.063 | 29 | 2 |
| c10 | 424 | 0.00 | 0.003 | 0.265 | 0.000 | 0.059 | 25 | 1 |
Quality axis: beat-template SQI of the last input minute; tercile sizes {'worst': 818, 'mid': 817, 'best': 817}; AUROC(SQI -> label) on train = 0.544 (0.5 = quality unrelated to the label). Flags: 243 flat/clipped minutes, 5 implausible-beat minutes, 6 minutes with < 3 beats (SQI set to 0, so they sit in the worst tercile).


### by_quality_tercile(sqi)

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| worst | 818 | 0.03 | 0.614 | 0.240 | 0.166 | 132 | 19 |
| mid | 817 | 0.02 | 0.775 | 0.250 | 0.080 | 64 | 15 |
| best | 817 | 0.02 | 0.843 | 0.533 | 0.086 | 69 | 7 |

### by_beats_plausible

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 5 | 0.00 | – | – | 0.000 | 0 | 0 |
| True | 2447 | 0.02 | 0.727 | 0.317 | 0.111 | 265 | 41 |

### by_flat_or_clipped

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2209 | 0.02 | 0.723 | 0.264 | 0.077 | 166 | 39 |
| True | 243 | 0.03 | 0.746 | 0.714 | 0.419 | 99 | 2 |

### by_no_running_baseline

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2415 | 0.02 | 0.725 | 0.322 | 0.111 | 261 | 40 |
| True | 37 | 0.03 | 0.889 | 0.000 | 0.111 | 4 | 1 |

### by_boundary_minute

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| False | 2341 | 0.00 | – | – | 0.105 | 245 | 0 |
| True | 111 | 0.54 | 0.439 | 0.317 | 0.392 | 20 | 41 |

### by_minutes_since_last_apnea

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| <=1 | 57 | 0.11 | 0.605 | 0.500 | 0.392 | 20 | 3 |
| 2-3 | 94 | 0.15 | 0.397 | 0.214 | 0.412 | 33 | 11 |
| 4-10 | 209 | 0.06 | 0.625 | 0.462 | 0.153 | 30 | 7 |
| 11-30 | 301 | 0.04 | 0.762 | 0.273 | 0.069 | 20 | 8 |
| >30 / never | 1791 | 0.01 | 0.640 | 0.250 | 0.091 | 162 | 12 |

### by_night_third

| group | n | prevalence | AUROC | TPR | FPR | FP | FN |
|---|---|---|---|---|---|---|---|
| first | 924 | 0.02 | 0.712 | 0.211 | 0.112 | 101 | 15 |
| middle | 794 | 0.02 | 0.732 | 0.316 | 0.106 | 82 | 13 |
| last | 734 | 0.03 | 0.744 | 0.409 | 0.115 | 82 | 13 |

Excluding boundary minutes: not defined for Task B (every onset is a boundary minute by construction).
R-peak miss rate vs .qrs by quality tercile: worst: 0.441, mid: 0.248, best: 0.077
R-peak miss rate on FP minutes 0.308 vs TN minutes 0.239.

Calibration: raw Brier 0.195, ECE 0.340; after isotonic (fit on OOF) Brier 0.025, ECE 0.013.
