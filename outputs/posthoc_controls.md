# Post-hoc diagnostics (val split only; decided after test numbers were seen; see ANALYSIS.md)

## 1. Label-shift test, Task A val split (score vs labels shifted by k minutes within each record)

| model | k=-3 | k=-2 | k=-1 | k=+0 | k=+1 | k=+2 | k=+3 |
|---|---|---|---|---|---|---|---|
| A_xgb_rel_s0 AUROC | 0.705 | 0.722 | 0.734 | 0.743 | 0.745 | 0.727 | 0.716 |
| A_xgb_rel_s0 AUPRC | 0.514 | 0.531 | 0.544 | 0.553 | 0.564 | 0.545 | 0.538 |
| A_xgb_rel_s1 AUROC | 0.703 | 0.720 | 0.730 | 0.739 | 0.742 | 0.724 | 0.713 |
| A_xgb_rel_s1 AUPRC | 0.502 | 0.517 | 0.529 | 0.538 | 0.546 | 0.528 | 0.520 |
| A_xgb_rel_s2 AUROC | 0.700 | 0.716 | 0.728 | 0.735 | 0.737 | 0.720 | 0.709 |
| A_xgb_rel_s2 AUPRC | 0.504 | 0.518 | 0.531 | 0.540 | 0.546 | 0.528 | 0.522 |
| A_cnn_s0 AUROC | 0.877 | 0.878 | 0.877 | 0.875 | 0.875 | 0.873 | 0.871 |
| A_cnn_s0 AUPRC | 0.787 | 0.788 | 0.786 | 0.788 | 0.788 | 0.782 | 0.777 |
| A_cnn_s1 AUROC | 0.896 | 0.896 | 0.895 | 0.894 | 0.894 | 0.893 | 0.892 |
| A_cnn_s1 AUPRC | 0.798 | 0.798 | 0.794 | 0.796 | 0.799 | 0.799 | 0.796 |
| A_cnn_s2 AUROC | 0.886 | 0.887 | 0.886 | 0.886 | 0.884 | 0.881 | 0.878 |
| A_cnn_s2 AUPRC | 0.788 | 0.790 | 0.787 | 0.786 | 0.785 | 0.778 | 0.774 |
| A_lr_s0 AUROC | 0.754 | 0.766 | 0.770 | 0.778 | 0.784 | 0.770 | 0.758 |
| A_lr_s0 AUPRC | 0.586 | 0.597 | 0.596 | 0.603 | 0.610 | 0.597 | 0.589 |

## 2. Future-minute detector: Task A model applied to the HORIZON minute, scored on the Task B onset label (val)

| model | prevalence | AUPRC | AUROC | within-record AUROC |
|---|---|---|---|---|
| xgb_rel | 0.029 | 0.041 ± 0.000 | 0.625 ± 0.007 | 0.688 |
| xgb | 0.029 | 0.039 ± 0.000 | 0.617 ± 0.004 | 0.607 |

## 5. Gap sweep: onset in (t, t+1] from a 5-min history ending at t − g (val; window-only XGBoost, 3 seeds, common prediction times)

| gap g (min) | prevalence | AUPRC | lift | AUROC | within-record AUROC |
|---|---|---|---|---|---|
| 0 | 0.029 | 0.103 ± 0.014 | 3.54 | 0.709 ± 0.011 | 0.648 |
| 1 | 0.029 | 0.061 ± 0.004 | 2.08 | 0.636 ± 0.008 | 0.543 |
| 2 | 0.029 | 0.054 ± 0.005 | 1.87 | 0.622 ± 0.004 | 0.523 |

## 3/4. CNN variants (3 seeds; mean ± sd; val split)

| variant | task | AUPRC | AUROC | within-record AUROC |
|---|---|---|---|---|
| cnn_selected_by_within_record_auroc | A | 0.771 ± 0.001 | 0.879 ± 0.004 | 0.655 ± 0.010 |
| cnn_selected_by_within_record_auroc | B | 0.081 ± 0.021 | 0.753 ± 0.033 | 0.590 ± 0.004 |
| cnn_polarity_flip_aug | A | 0.767 ± 0.007 | 0.858 ± 0.010 | 0.635 ± 0.045 |
| cnn_polarity_flip_aug | B | 0.120 ± 0.005 | 0.758 ± 0.017 | 0.502 ± 0.036 |