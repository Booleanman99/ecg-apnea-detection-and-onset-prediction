| Task | Model | seeds | Val AUPRC | Test prevalence | Test AUPRC | Test AUROC | Precision | Recall | F1 | Within-record AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Logistic regression (HRV/EDR window features) | 1 | 0.603 | 0.350 | 0.525 | 0.750 | 0.554 | 0.591 | 0.572 | 0.770 |
| A | XGBoost (HRV/EDR window features) | 3 | 0.579 | 0.350 | 0.584 ± 0.004 | 0.793 ± 0.002 | 0.604 | 0.647 | 0.625 ± 0.008 | 0.848 |
| A | 1-D CNN + GRU (raw ECG window) | 3 | 0.790 | 0.350 | 0.537 ± 0.012 | 0.651 ± 0.026 | 0.546 | 0.566 | 0.555 ± 0.018 | 0.653 |
| A | XGBoost + subject-relative features (night-so-far context; second baseline) | 3 | 0.544 | 0.350 | 0.621 ± 0.015 | 0.811 ± 0.002 | 0.610 | 0.723 | 0.661 ± 0.007 | 0.859 |
| B | Logistic regression (HRV/EDR window features) | 1 | 0.090 | 0.024 | 0.073 | 0.695 | 0.071 | 0.367 | 0.118 | 0.646 |
| B | XGBoost (HRV/EDR window features) | 3 | 0.091 | 0.024 | 0.121 ± 0.009 | 0.741 ± 0.006 | 0.186 | 0.167 | 0.154 ± 0.038 | 0.692 |
| B | 1-D CNN + GRU (raw ECG window) | 3 | 0.094 | 0.024 | 0.052 ± 0.011 | 0.516 ± 0.014 | 0.086 | 0.222 | 0.122 ± 0.021 | 0.517 |
| B | XGBoost + subject-relative features (night-so-far context; second baseline) | 3 | 0.118 | 0.024 | 0.109 ± 0.005 | 0.706 ± 0.006 | 0.180 | 0.100 | 0.128 ± 0.005 | 0.684 |

**Task A final model (selected on validation):** 1-D CNN + GRU (raw ECG window) (seed 0), threshold 0.007. Test AUPRC 0.552 (record-bootstrap 95% CI 0.088–0.865; prevalence 0.350, lift 1.6×), AUROC 0.658 (CI 0.238–0.886). Within-record AUROC 0.616 over 5 records (CI 0.423–0.785). Confusion matrix [TN FP / FN TP] = [1719 767 / 579 759]. Brier 0.290, ECE 0.304.

**Task B final model (selected on validation):** XGBoost + subject-relative features (night-so-far context; second baseline) (seed 2), threshold 0.430. Test AUPRC 0.102 (record-bootstrap 95% CI 0.032–0.236; prevalence 0.024, lift 4.2×), AUROC 0.713 (CI 0.569–0.817). Within-record AUROC 0.691 over 5 records (CI 0.564–0.842). Confusion matrix [TN FP / FN TP] = [2361 31 / 54 6]. Brier 0.028, ECE 0.023.
