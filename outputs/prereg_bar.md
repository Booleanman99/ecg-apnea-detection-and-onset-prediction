# Pre-registered Task B bar — evaluation

Final model xgb_rel seed 2, frozen threshold 0.430.

## (a) AUPRC vs prevalence, per seed (record-bootstrap 95% CI on test)

| seed | prevalence | AUPRC | lift | CI | point lift >= 2x | CI clear of prevalence (literal) | CI clear of 2x prevalence (strict) |
|---|---|---|---|---|---|---|---|
| 0 | 0.024 | 0.111 | 4.5x | 0.038–0.224 | True | True | False |
| 1 | 0.024 | 0.113 | 4.6x | 0.030–0.237 | True | True | False |
| 2 | 0.024 | 0.102 | 4.2x | 0.032–0.236 | True | True | False |

All seeds, literal reading: True; strict reading: False.

## (i) within-record AUROC, stamped test plus val (8 a/b records with >= 5 onsets)

| record | split | n | onsets | AUROC |
|---|---|---|---|---|
| a13 | val | 247 | 20 | 0.723 |
| a14 | val | 121 | 8 | 0.729 |
| a15 | val | 136 | 16 | 0.642 |
| a16 | test | 156 | 13 | 0.649 |
| a18 | test | 47 | 6 | 0.443 |
| a19 | test | 296 | 15 | 0.944 |
| a20 | test | 190 | 16 | 0.774 |
| b05 | test | 371 | 7 | 0.645 |

Mean 0.694, CI 0.599–0.781 -> clause (i) MET.

## (i) within-record AUROC, test only (5 a/b records with >= 5 onsets)

| record | split | n | onsets | AUROC |
|---|---|---|---|---|
| a16 | test | 156 | 13 | 0.649 |
| a18 | test | 47 | 6 | 0.443 |
| a19 | test | 296 | 15 | 0.944 |
| a20 | test | 190 | 16 | 0.774 |
| b05 | test | 371 | 7 | 0.645 |

Mean 0.691, CI 0.550–0.842 -> clause (i) MET.

## (ii) TPR in worst vs best SQI tercile at the frozen threshold

Quality index: beat-template SQI of last input minute; tercile sizes {'worst': 818, 'mid': 817, 'best': 817}; SQI = 0 minutes: 6; AUROC(SQI -> label) on train 0.544.

worst: n=818, positives=25, TPR 0.080; best: n=817, TPR 0.067 -> clause (ii) MET.

**Overall (literal reading): MET. Overall (strict reading): NOT MET.**

# Task A diagnostics

## Per record: prevalence, mean predicted probability and within-record AUROC

| record | split | n | prev | R+ frac | CNN mean p | CNN within AUROC | XGB-rel mean p | XGB-rel within | XGB mean p | XGB within |
|---|---|---|---|---|---|---|---|---|---|---|
| a16 | test | 481 | 0.67 | 0.99 | 0.059 | 0.838 | 0.540 | 0.891 | 0.511 | 0.877 |
| a18 | test | 489 | 0.90 | 1.00 | 0.291 | 0.810 | 0.826 | 0.884 | 0.797 | 0.875 |
| a19 | test | 502 | 0.41 | 1.00 | 0.002 | 0.509 | 0.195 | 0.940 | 0.207 | 0.955 |
| a20 | test | 510 | 0.62 | 0.99 | 0.002 | 0.656 | 0.301 | 0.813 | 0.325 | 0.754 |
| b05 | test | 432 | 0.13 | 0.31 | 0.003 | 0.266 | 0.054 | 0.772 | 0.163 | 0.804 |
| c08 | test | 513 | 0.00 | 0.99 | 0.023 | – | 0.664 | – | 0.646 | – |
| c09 | test | 467 | 0.00 | 0.99 | 0.008 | – | 0.111 | – | 0.140 | – |
| c10 | test | 430 | 0.00 | 1.00 | 0.125 | – | 0.154 | – | 0.231 | – |
| a13 | val | 494 | 0.49 | 1.00 | 0.009 | 0.650 | 0.468 | 0.943 | 0.407 | 0.917 |
| a14 | val | 509 | 0.75 | 1.00 | 0.791 | 0.859 | 0.526 | 0.745 | 0.656 | 0.666 |
| a15 | val | 509 | 0.72 | 1.00 | 0.381 | 0.465 | 0.494 | 0.802 | 0.584 | 0.820 |
| b04 | val | 428 | 0.02 | 1.00 | 0.021 | 0.342 | 0.466 | 0.661 | 0.478 | 0.617 |
| c06 | val | 467 | 0.00 | 1.00 | 0.002 | – | 0.154 | – | 0.175 | – |
| c07 | val | 428 | 0.01 | 1.00 | 0.002 | – | 0.599 | – | 0.549 | – |

## Between-night vs within-night

| model | split | Spearman(mean p, prevalence) across records | mean within-record AUROC | n |
|---|---|---|---|---|
| cnn | val | 0.943 | 0.579 | 4 |
| cnn | test | 0.095 | 0.616 | 5 |
| xgb_rel | val | 0.429 | 0.788 | 4 |
| xgb_rel | test | 0.405 | 0.860 | 5 |
| xgb | val | 0.771 | 0.755 | 4 |
| xgb | test | 0.357 | 0.853 | 5 |

## Paired record bootstrap on test (model − CNN)

| comparison | pooled AUPRC diff | CI | within-record AUROC diff | CI |
|---|---|---|---|---|
| xgb_rel_minus_cnn | +0.083 | -0.206–+0.430 | +0.244 | +0.084–+0.421 |
| xgb_minus_cnn | +0.037 | -0.251–+0.361 | +0.237 | +0.063–+0.434 |