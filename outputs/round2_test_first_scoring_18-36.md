# Round 2 — single test scoring of the promoted pipelines

Confirmation on record: Candidate, 2026-10-29 18:35 IST, in writing: 'then do that' in reply to the recommended option 1, 'Apply the rule as written, score once' (Task A ctx5_rz|shipped, Task B W5_r1_window|lr_C0.5; retrain on 22 nights; one test scoring).
Scored at 2026-10-29T18:36:16

| task | pipeline | seed | α | threshold | OOF AUPRC | OOF within-night | test prevalence | test AUPRC (CI) | test AUROC (CI) | test within-night (CI) | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | ctx5_rz|shipped | 2 | 1.0 | 0.252 | 0.785 | 0.860 | 0.350 | 0.914 (0.729–0.956) | 0.949 (0.920–0.965) | 0.908 (0.850–0.951) | 0.776 | 0.892 | 0.830 |
| B | W5_r1_window|lr_C0.5 | 1 | 1.0 | 0.891 | 0.115 | 0.699 | 0.024 | 0.080 (0.021–0.197) | 0.761 (0.660–0.833) | 0.624 (0.541–0.684) | 0.085 | 0.203 | 0.119 |

Task A after isotonic recalibration (fit on OOF, threshold 0.356): AUPRC 0.904, AUROC 0.948, P 0.776, R 0.892, F1 0.830, Brier 0.109 (raw 0.088), ECE 0.146 (raw 0.043), [TN FP / FN TP] = [2142 344 / 144 1194].

Per night, Task A:

| record | n | prevalence | AUROC | AUPRC | TPR | FPR | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| a16 | 481 | 0.67 | 0.931 | 0.968 | 0.947 | 0.404 | 303 | 65 | 17 |
| a18 | 489 | 0.90 | 0.794 | 0.968 | 0.950 | 0.667 | 416 | 34 | 22 |
| a19 | 502 | 0.41 | 0.973 | 0.971 | 0.824 | 0.034 | 169 | 10 | 36 |
| a20 | 510 | 0.62 | 0.905 | 0.929 | 0.854 | 0.205 | 269 | 40 | 46 |
| b05 | 432 | 0.13 | 0.937 | 0.776 | 0.632 | 0.035 | 36 | 13 | 21 |
| c08 | 513 | 0.00 | – | – | – | 0.224 | 0 | 115 | 0 |
| c09 | 467 | 0.00 | 0.775 | 0.025 | 0.500 | 0.075 | 1 | 35 | 1 |
| c10 | 430 | 0.00 | 0.263 | 0.003 | 0.000 | 0.075 | 0 | 32 | 1 |

Task B after isotonic recalibration (fit on OOF, threshold 0.158): AUPRC 0.074, AUROC 0.755, P 0.085, R 0.203, F1 0.119, Brier 0.024 (raw 0.192), ECE 0.013 (raw 0.322), [TN FP / FN TP] = [2226 130 / 47 12].

Per night, Task B:

| record | n | prevalence | AUROC | AUPRC | TPR | FPR | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| a16 | 151 | 0.09 | 0.676 | 0.317 | 0.462 | 0.181 | 6 | 25 | 7 |
| a18 | 42 | 0.14 | 0.458 | 0.154 | 0.000 | 0.000 | 0 | 0 | 6 |
| a19 | 294 | 0.05 | 0.701 | 0.100 | 0.143 | 0.068 | 2 | 19 | 12 |
| a20 | 185 | 0.09 | 0.664 | 0.210 | 0.188 | 0.083 | 3 | 14 | 13 |
| b05 | 366 | 0.02 | 0.620 | 0.030 | 0.000 | 0.078 | 0 | 28 | 7 |
| c08 | 503 | 0.00 | – | – | – | 0.058 | 0 | 29 | 0 |
| c09 | 455 | 0.00 | 0.948 | 0.106 | 0.500 | 0.022 | 1 | 10 | 1 |
| c10 | 419 | 0.00 | 0.136 | 0.003 | 0.000 | 0.012 | 0 | 5 | 1 |