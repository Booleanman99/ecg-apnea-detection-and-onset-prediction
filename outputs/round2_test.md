# Round 2 — single test scoring of the promoted pipelines

Confirmation on record: Candidate, 2026-10-29 18:35 IST, in writing: 'then do that' in reply to the recommended option 1, 'Apply the rule as written, score once' (Task A ctx5_rz|shipped, Task B W5_r1_window|lr_C0.5; retrain on 22 nights; one test scoring).
Scored at 2026-10-29T18:38:38

| task | pipeline | seed | α | threshold | OOF AUPRC | OOF within-night | test prevalence | test AUPRC (CI) | test AUROC (CI) | test within-night (CI) | P | R | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | ctx5_rz|shipped | 2 | 1.0 | 0.252 | 0.785 | 0.860 | 0.350 | 0.914 (0.729–0.956) | 0.949 (0.920–0.965) | 0.908 (0.850–0.951) | 0.776 | 0.892 | 0.830 |
| B | W5_r1_window|lr_C0.5 | 1 | 1.0 | 0.732 | 0.102 | 0.710 | 0.024 | 0.059 (0.016–0.178) | 0.727 (0.585–0.822) | 0.630 (0.538–0.736) | 0.067 | 0.317 | 0.110 |

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

Task B after isotonic recalibration (fit on OOF, threshold 0.106): AUPRC 0.054, AUROC 0.722, P 0.060, R 0.250, F1 0.096, Brier 0.025 (raw 0.195), ECE 0.013 (raw 0.340), [TN FP / FN TP] = [2156 236 / 45 15].

Per night, Task B:

| record | n | prevalence | AUROC | AUPRC | TPR | FPR | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| a16 | 156 | 0.08 | 0.664 | 0.289 | 0.692 | 0.441 | 9 | 63 | 4 |
| a18 | 47 | 0.13 | 0.524 | 0.151 | 0.000 | 0.049 | 0 | 2 | 6 |
| a19 | 296 | 0.05 | 0.830 | 0.159 | 0.400 | 0.107 | 6 | 30 | 9 |
| a20 | 190 | 0.08 | 0.626 | 0.162 | 0.250 | 0.098 | 4 | 17 | 12 |
| b05 | 371 | 0.02 | 0.508 | 0.021 | 0.000 | 0.118 | 0 | 43 | 7 |
| c08 | 508 | 0.00 | – | – | – | 0.110 | 0 | 56 | 0 |
| c09 | 460 | 0.00 | 0.888 | 0.030 | 0.000 | 0.063 | 0 | 29 | 2 |
| c10 | 424 | 0.00 | 0.265 | 0.003 | 0.000 | 0.059 | 0 | 25 | 1 |