# Controls (test split; mean ± sd over seeds where applicable)

| control | prevalence | AUPRC | lift | AUROC |
|---|---|---|---|---|
| shuffle_global_taskA [xgb_rel] | 0.350 | 0.339 ± 0.020 | 0.969 ± 0.057 | 0.499 ± 0.039 |
| shuffle_global_taskA [cnn] | 0.350 | 0.277 | 0.791 | 0.386 |
| shuffle_within_record_taskA [xgb_rel] | 0.350 | 0.541 ± 0.012 | 1.548 ± 0.033 | 0.729 ± 0.008 |
| shuffle_within_record_taskA [cnn] | 0.350 | 0.503 | 1.438 | 0.621 |
| shuffle_global_taskB [xgb_rel] | 0.024 | 0.028 ± 0.002 | 1.127 ± 0.098 | 0.515 ± 0.026 |
| shuffle_global_taskB [cnn] | 0.024 | 0.019 | 0.784 | 0.391 |
| shuffle_within_record_taskB [xgb_rel] | 0.024 | 0.058 ± 0.007 | 2.385 ± 0.303 | 0.679 ± 0.009 |
| shuffle_within_record_taskB [cnn] | 0.024 | 0.035 | 1.435 | 0.546 |
| taskB_primary_ongoing_excluded [xgb_rel] | 0.024 | 0.109 ± 0.005 | 4.441 ± 0.195 | 0.706 ± 0.006 |
| taskB_primary_ongoing_excluded [cnn] | 0.024 | 0.039 | 1.587 | 0.501 |
| taskB_deployment_all_windows_onset_only [xgb_rel] | 0.016 | 0.056 ± 0.007 | 3.560 ± 0.456 | 0.704 ± 0.006 |
| taskB_deployment_all_windows_onset_only [cnn] | 0.016 | 0.046 | 2.874 | 0.701 |
| taskB_persistence_inflated_apnea_in_horizon [xgb_rel] | 0.353 | 0.615 ± 0.003 | 1.743 ± 0.009 | 0.824 ± 0.000 |
| taskB_persistence_inflated_apnea_in_horizon [cnn] | 0.353 | 0.549 | 1.558 | 0.660 |
| detector_on_prediction_windows [xgb] | 0.024 | 0.035 ± 0.000 | 1.411 ± 0.003 | 0.658 ± 0.000 |
| detector_on_prediction_windows [cnn] | 0.024 | 0.029 | 1.188 | 0.452 |
| horizon_sweep H1 [xgb_rel] | 0.024 | 0.088 ± 0.005 | 3.620 ± 0.188 | 0.705 ± 0.008 |
| horizon_sweep H2 [xgb_rel] | 0.046 | 0.090 ± 0.008 | 1.934 ± 0.176 | 0.619 ± 0.007 |
| horizon_sweep H3 [xgb_rel] | 0.065 | 0.083 ± 0.004 | 1.278 ± 0.055 | 0.587 ± 0.006 |
| horizon_sweep H5 [xgb_rel] | 0.097 | 0.120 ± 0.009 | 1.232 ± 0.088 | 0.559 ± 0.020 |
| horizon_sweep H10 [xgb_rel] | 0.162 | 0.220 ± 0.013 | 1.357 ± 0.081 | 0.611 ± 0.021 |
| history_sweep W1 [xgb] | 0.024 | 0.093 ± 0.007 | 3.796 ± 0.276 | 0.770 ± 0.006 |
| history_sweep W3 [xgb] | 0.024 | 0.108 ± 0.020 | 4.419 ± 0.834 | 0.736 ± 0.004 |
| history_sweep W5 [xgb] | 0.024 | 0.134 ± 0.010 | 5.498 ± 0.392 | 0.738 ± 0.009 |
| history_sweep W10 [xgb] | 0.024 | 0.180 ± 0.013 | 7.388 ± 0.552 | 0.769 ± 0.014 |