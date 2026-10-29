# Round 2 — grouped 6-fold CV on the 22 non-test nights (3 seeds; mean)

Folds (balanced by class): {'a01': 0, 'a02': 1, 'a03': 2, 'a05': 3, 'a06': 4, 'a07': 5, 'a08': 0, 'a09': 1, 'a11': 2, 'a12': 3, 'a13': 4, 'a14': 5, 'a15': 0, 'b01': 0, 'b02': 1, 'b04': 2, 'c01': 0, 'c02': 1, 'c03': 2, 'c04': 3, 'c06': 4, 'c07': 5}

## Task A — detection

| candidate | smoothing α | OOF AUPRC | OOF AUROC | within-night AUROC | fold sd | seed sd |
|---|---|---|---|---|---|---|
| r1_window|shipped | 1.0 | 0.560 | 0.670 | 0.763 | 0.052 | 0.002 |
| r1_window|shipped | 0.7 | 0.599 | 0.680 | 0.786 | 0.052 | 0.002 |
| r1_window|shipped | 0.5 | 0.615 | 0.684 | 0.794 | 0.051 | 0.003 |
| r1_window|shipped | 0.3 | 0.621 | 0.681 | 0.791 | 0.051 | 0.002 |
| r1_window|grid | 1.0 | 0.527 | 0.646 | 0.732 | 0.046 | 0.005 |
| r1_window|grid | 0.7 | 0.556 | 0.655 | 0.758 | 0.045 | 0.006 |
| r1_window|grid | 0.5 | 0.567 | 0.657 | 0.766 | 0.045 | 0.006 |
| r1_window|grid | 0.3 | 0.571 | 0.655 | 0.763 | 0.047 | 0.007 |
| r1_rel|shipped | 1.0 | 0.714 | 0.752 | 0.782 | 0.034 | 0.003 |
| r1_rel|shipped | 0.7 | 0.741 | 0.764 | 0.804 | 0.031 | 0.003 |
| r1_rel|shipped | 0.5 | 0.750 | 0.767 | 0.811 | 0.028 | 0.003 |
| r1_rel|shipped | 0.3 | 0.754 | 0.765 | 0.809 | 0.027 | 0.003 |
| r1_rel|grid | 1.0 | 0.704 | 0.750 | 0.770 | 0.034 | 0.003 |
| r1_rel|grid | 0.7 | 0.735 | 0.765 | 0.796 | 0.033 | 0.004 |
| r1_rel|grid | 0.5 | 0.747 | 0.770 | 0.802 | 0.032 | 0.004 |
| r1_rel|grid | 0.3 | 0.753 | 0.771 | 0.798 | 0.033 | 0.005 |
| ctx5|shipped | 1.0 | 0.763 | 0.777 | 0.835 | 0.057 | 0.004 |
| ctx5|shipped | 0.7 | 0.767 | 0.778 | 0.834 | 0.055 | 0.004 |
| ctx5|shipped | 0.5 | 0.767 | 0.776 | 0.829 | 0.053 | 0.003 |
| ctx5|shipped | 0.3 | 0.764 | 0.773 | 0.817 | 0.049 | 0.002 |
| ctx5|grid | 1.0 | 0.774 | 0.794 | 0.833 | 0.047 | 0.001 |
| ctx5|grid | 0.7 | 0.780 | 0.796 | 0.835 | 0.045 | 0.000 |
| ctx5|grid | 0.5 | 0.782 | 0.795 | 0.830 | 0.043 | 0.000 |
| ctx5|grid | 0.3 | 0.780 | 0.792 | 0.818 | 0.041 | 0.001 |
| ctx5_rz|shipped | 1.0 | 0.785 | 0.803 | 0.860 | 0.038 | 0.000 |
| ctx5_rz|shipped | 0.7 | 0.787 | 0.804 | 0.859 | 0.037 | 0.000 |
| ctx5_rz|shipped | 0.5 | 0.787 | 0.803 | 0.853 | 0.037 | 0.000 |
| ctx5_rz|shipped | 0.3 | 0.784 | 0.799 | 0.839 | 0.036 | 0.000 |
| ctx5_rz|grid | 1.0 | 0.784 | 0.805 | 0.857 | 0.028 | 0.001 |
| ctx5_rz|grid | 0.7 | 0.789 | 0.807 | 0.857 | 0.027 | 0.000 |
| ctx5_rz|grid | 0.5 | 0.790 | 0.806 | 0.851 | 0.026 | 0.001 |
| ctx5_rz|grid | 0.3 | 0.787 | 0.802 | 0.836 | 0.028 | 0.001 |

Comparator (best round-1 family under CV): r1_rel|shipped. Best Task A by within-night AUROC (α = 1): ctx5_rz|shipped; its nested OOF detector probabilities feed the Task B detector-history features.

Paired per-fold differences vs comparator (within-night AUROC, α = 1): r1_window|shipped: -0.024, 2×SE 0.022, passes False, AUPRC not lower False; r1_window|grid: -0.050, 2×SE 0.021, passes False, AUPRC not lower False; r1_rel|grid: -0.011, 2×SE 0.022, passes False, AUPRC not lower False; ctx5|shipped: +0.046, 2×SE 0.028, passes True, AUPRC not lower True; ctx5|grid: +0.045, 2×SE 0.020, passes True, AUPRC not lower True; ctx5_rz|shipped: +0.078, 2×SE 0.017, passes True, AUPRC not lower True; ctx5_rz|grid: +0.075, 2×SE 0.019, passes True, AUPRC not lower True

Eligible nights (≥ 5 positives and negatives): ['a01', 'a02', 'a03', 'a05', 'a06', 'a07', 'a08', 'a09', 'a11', 'a12', 'a13', 'a14', 'a15', 'b01', 'b02', 'b04']

## Task B — onset prediction (primary set)

| candidate | OOF AUPRC | OOF AUROC | within-night AUROC | fold sd | seed sd | nights |
|---|---|---|---|---|---|---|
| W5_r1_rel|shipped | 0.102 | 0.748 | 0.695 | 0.041 | 0.004 | 14 |
| W5_r1_rel+det|shipped | 0.110 | 0.758 | 0.702 | 0.046 | 0.006 | 14 |
| W5_r1_rel|grid | 0.124 | 0.773 | 0.709 | 0.061 | 0.003 | 14 |
| W5_r1_rel+det|grid | 0.132 | 0.782 | 0.717 | 0.062 | 0.003 | 14 |
| W10_rel|shipped | 0.100 | 0.759 | 0.687 | 0.079 | 0.008 | 14 |
| W10_rel+det|shipped | 0.112 | 0.777 | 0.697 | 0.072 | 0.007 | 14 |
| W10_rel|grid | 0.126 | 0.777 | 0.705 | 0.097 | 0.002 | 14 |
| W10_rel+det|grid | 0.132 | 0.789 | 0.712 | 0.096 | 0.002 | 14 |

Comparator: W5_r1_rel|grid. Paired per-fold differences: W5_r1_rel|shipped: -0.009, 2×SE 0.029, passes False, AUPRC not lower False; W5_r1_rel+det|shipped: -0.002, 2×SE 0.025, passes False, AUPRC not lower False; W5_r1_rel+det|grid: +0.007, 2×SE 0.006, passes True, AUPRC not lower True; W10_rel|shipped: -0.022, 2×SE 0.032, passes False, AUPRC not lower False; W10_rel+det|shipped: -0.011, 2×SE 0.035, passes False, AUPRC not lower False; W10_rel|grid: -0.008, 2×SE 0.040, passes False, AUPRC not lower True; W10_rel+det|grid: -0.000, 2×SE 0.039, passes False, AUPRC not lower True

Eligible nights: ['a02', 'a03', 'a05', 'a06', 'a07', 'a08', 'a09', 'a11', 'a12', 'a13', 'a14', 'a15', 'b01', 'b02']