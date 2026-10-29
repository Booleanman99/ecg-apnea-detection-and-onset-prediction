# REPORT — ECG apnea detection (Task A) and prospective onset prediction (Task B)

The test split is 8 nights. Two rounds of model selection are reported (rules and timestamps in `PREREGISTRATION.md`): round 1 on a 6-night validation split; round 2 by 6-fold grouped cross-validation on all 22 non-test nights under a rule fixed before it ran, scored on test once per final pipeline (§10).

## 1. Data, windows and labels

**Data.** 30 overnight single-lead ECG records ("nights": 17 severe, 4 borderline, 9 control), 244.6 h at 100 Hz; 5,725 of 14,626 annotated minutes are apnea (prevalence 0.391), per-night 0.00–0.96, so the imbalance is between nights. ECG-only (8 records carry respiration/SpO₂).

**Alignment, verified.** Every `.apn` annotation sits at sample k·6000 in steps of exactly 6000 (asserted in s1), so annotation k labels minute [k, k+1); unannotated trailing signal is dropped.

**Pre-processing.** R-peaks from our own Pan–Tompkins-style detector (validated against the supplied `.qrs`, never an input), a beat-template signal-quality index (SQI) and flat/clip/beat-count flags; nothing is dropped for quality. Every input is causal: Task A uses the labelled minute (plus the four before it in the final model); Task B uses the 5-minute history only, proven by `s2_split.py` (noise after t, identical features, 30/30 windows).

**Windows.** Task A: one window per annotated minute, 14,626 (test 3,824, prevalence 0.350). Task B: t at every minute boundary with 5 annotated minutes of history and 1 of horizon; an **onset** is an A minute whose predecessor is N; label = "an onset occurs in (t, t+1 min]"; windows with apnea in progress at t (5,708) are excluded, as the brief requires. Primary set 8,768 windows (test 2,452; onset prevalence 0.024, 60 onsets).

## 2. Split and leakage

Split **by record** (`src/common.py`): train 16, validation 6, test 8 nights, each spanning the severity range; labels persist (P(same as previous minute) = 0.965; mean run 21.7 min), so a window-level split would leak. No whole-night statistic reaches a model (per-window normalisation; night-so-far features from earlier windows only). Round 1 read test once, after every choice was fixed on validation. Round 2 selected by grouped CV on the 22 non-test nights (folds balanced by severity, 3 seeds) under a comparator, paired per-fold test and promotion rule stamped before any CV number existed, retrained on the 22 nights, thresholded at maximum F1 on out-of-fold predictions and scored test once per final pipeline (§10); the Task A final passed five leak checks first (`outputs/leak_checks_round2_A.json`).

## 3. Class imbalance

Test prevalence: Task A 0.350, Task B 0.024 (60 onsets). Class-weighted losses (`class_weight="balanced"`, `scale_pos_weight`, BCE `pos_weight`) and a maximum-F1 threshold on held-out predictions (validation in round 1, out-of-fold in round 2); AUPRC beside prevalence, because at 2.4 % positives AUROC alone hides how few flagged windows are real.

## 4–6. Models and results

**Baselines (§4).** Logistic regression and XGBoost on 21 HRV, ECG-derived-respiration and quality features per window (README), chosen for the heart-rate cycling and R-amplitude modulation apnea produces; for Task B over the 5-minute history with per-minute trajectories. A second XGBoost adds each feature minus its running median over the night so far ("night-so-far"). Under 22-night grouped CV the linear baseline was the strongest round-1 detector (within-night 0.792 vs 0.782 and 0.763), so it is round 2's comparator. The round-2 Task A final adds the four preceding minutes, whole-context HRV and a causal running median/IQR z-score of every feature.

**Temporal deep model (§5).** One architecture for both tasks: a 1-D CNN minute encoder (4 conv blocks, global average pooling, 64-d) feeding a GRU over the minute sequence (length 1 for Task A, where it is inert; 5 for Task B): convolutions for beat morphology and spacing, the GRU for state across minutes; 59,969 parameters, weighted BCE, early stopping on validation AUPRC, 3 seeds.

**Selection.** Round 1: highest mean validation AUPRC per task, fixed in code before any test metric — the **CNN for Task A**, **XGBoost + night-so-far for Task B**. Round 2: a candidate replaces the round-1 final if its paired per-fold within-night AUROC gain over the best round-1 family exceeds twice its standard error with out-of-fold AUPRC not lower; failing that, a round-1 family ranked above the round-1 final by CV is promoted. It chose the **5-minute-context XGBoost for Task A** (+0.070 ± 0.024 over logistic regression; CV within-night 0.860 vs 0.792) and **logistic regression for Task B** (0.709 vs 0.695 within-night over the round-1 final: inside noise, promoted because the clause had no noise margin, §10).

| Task | Model | round | Test prev. | Test AUPRC | Test AUROC | Prec. | Recall | F1 | Within-night AUROC |
|---|---|---|---|---|---|---|---|---|---|
| A | Logistic regression (window feats) | 1 | 0.350 | 0.525 | 0.750 | 0.554 | 0.591 | 0.572 | 0.770 |
| A | XGBoost (window feats) | 1 | 0.350 | 0.584 ± 0.004 | 0.793 ± 0.002 | 0.604 | 0.647 | 0.625 | 0.848 |
| A | XGBoost (+ night-so-far feats) | 1 | 0.350 | 0.621 ± 0.015 | 0.811 ± 0.002 | 0.610 | 0.723 | 0.661 | 0.859 |
| A | CNN + GRU (raw ECG), round-1 final | 1 | 0.350 | 0.537 ± 0.012 | 0.651 ± 0.026 | 0.546 | 0.566 | 0.555 | 0.653 |
| A | **XGBoost, 5-min context + night-so-far scaling — FINAL** | 2 | 0.350 | **0.914** | **0.949** | 0.776 | 0.892 | 0.830 | **0.908** |
| B | Logistic regression (window feats) | 1 | 0.024 | 0.073 | 0.695 | 0.071 | 0.367 | 0.118 | 0.646 |
| B | XGBoost (window feats) | 1 | 0.024 | 0.121 ± 0.009 | 0.741 ± 0.006 | 0.186 | 0.167 | 0.154 | 0.692 |
| B | XGBoost (+ night-so-far feats), round-1 final | 1 | 0.024 | 0.109 ± 0.005 | 0.706 ± 0.006 | 0.180 | 0.100 | 0.128 | 0.684 |
| B | CNN + GRU (raw ECG) | 1 | 0.024 | 0.052 ± 0.011 | 0.516 ± 0.014 | 0.086 | 0.222 | 0.122 | 0.517 |
| B | **Logistic regression (window feats) — FINAL** | 2 | 0.024 | **0.059** | **0.727** | 0.067 | 0.317 | 0.110 | **0.630** |

Round-1 rows: seed means (± sd) at validation thresholds (logistic regression is deterministic); round-2 rows: the 22-night retrain at its out-of-fold threshold. Within-night AUROC: mean over the 5 test nights with ≥ 5 of each class. Confusion matrices: `outputs/results.json`, `outputs/round2_test.json`.

**Final models, with uncertainty.** Task A (threshold 0.252): AUPRC 0.914, record-bootstrap 95 % CI 0.729–0.956; AUROC 0.949 (0.920–0.965); within-night 0.908 (0.850–0.951); [TN 2142, FP 344; FN 144, TP 1194]; above its CV estimate (0.785 / 0.860) as every feature model was; leak checks passed (§2). Task B (threshold 0.732): AUPRC 0.059 (0.016–0.178; lift 2.5×); AUROC 0.727; within-night 0.630 (0.538–0.736); [TN 2127, FP 265; FN 41, TP 19]: lower than the round-1 final on AUPRC and within-night, higher on pooled AUROC, CIs overlapping, so indistinguishable on 60 onsets; the selection was not revisited after the number was seen.

**The deep model does not earn its place.** On Task A the CNN won validation (0.790 vs ≤ 0.603) and lost test through a between-night shortcut: its mean probability tracks night prevalence (Spearman 0.943) while its within-night AUROC is 0.579, and the mapping fails on test (Spearman 0.095). A paired record bootstrap gives the round-1 night-so-far XGBoost +0.244 within-night AUROC over it (CI +0.084 to +0.421). On Task B the CNN is at chance (0.516) against 0.70–0.74 for the feature models.

## 7. Error analysis and calibration (final models)

Per-night and quality tables for both finals are in `outputs/round2_test.md` and `outputs/error_analysis_round2.md`; the concentration they show:

**Task A.** False alarms concentrate (c08 holds 115 of 344; c08 + a16 + a20 64 %); misses are spread, no severe night below TPR 0.63 (lowest b05, whose inverted lead broke the round-1 CNN), and round 1's whole-night failures (a19 + a20, 90 % of its misses) are gone. Boundary minutes (236, 6 %) hold a third of the misses (AUROC 0.673 vs 0.962 elsewhere); quality no longer matters (0.942 / 0.959 / 0.949 by SQI tercile); the first five minutes of each night, before the running baseline exists, are at chance (40 windows).

**Task B.** Catches concentrate on a16 (9 of 19, where 44 % of normal windows are also flagged); a18 and b05 sit at chance with 13 of the 41 misses and no catch. Flat or clipped windows (243, 10 %) hold 99 of the 265 false alarms (FPR 0.419 vs 0.077): the linear model reads imputed features and flags as apnea-like (AUROC 0.614 / 0.775 / 0.843 by SQI tercile). Most of the one-minute skill sits in the final minute before t (validation-only gap sweep), which at minute-label resolution cannot separate an onset already under way from a short precursor (ANALYSIS §3, §6).

**Calibration** (test; isotonic fitted on 22-night out-of-fold predictions; BSS = Brier skill vs base rate):

| model | Brier raw → isotonic | ECE raw → isotonic | BSS raw → isotonic | ΔAUPRC |
|---|---|---|---|---|
| A (final) | 0.088 → 0.109 | 0.043 → 0.146 | +0.61 → +0.52 | 0.914 → 0.904 |
| B (final) | 0.195 → 0.025 | 0.340 → 0.013 | −7.2 → ≈ 0 | 0.059 → 0.054 |

Ranking and Task A's decisions are unchanged (AUROC 0.949 → 0.948, same confusion matrix); Task B's operating point moves (TP 19 → 15, FP 265 → 236). Task A's raw probabilities are already calibrated; a calibrator fitted on six fold models and applied to the 22-night model makes them worse. Task B's raw output is not a probability (balanced weights at 2.4 % prevalence); recalibration fixes the scale, and a BSS near zero says the calibrated probabilities forecast no better than the base rate: the model's value is ranking only.

## 8. Reproducibility

One command (`run_all.sh`) reproduces both rounds; fixed seeds, pinned dependencies, byte-identical re-runs; timings in README.

## 9. With more time

A CNN learning curve; event-level Task B scoring under leave-one-night-out CV; a sub-minute gap sweep; a noise margin in every promotion clause.

## 10. Limitations

Eight test nights: every CI is wide. An earlier configuration selected on a 6-night validation split was scored on the test set once; the final models were then selected by 22-night grouped cross-validation before a single further test scoring per final pipeline (one mis-mapped Task B run was caught and re-scored with the same pre-fixed choices; `PREREGISTRATION.md`), a second look at the test set. Round-1 numbers rest on a 6-night validation split used for early stopping, selection and thresholds. The round-2 clause "ranked above the round-1 final" carried no noise margin and promoted a Task B model statistically indistinguishable from the one it replaced. The Task B primary set filters on a label (apnea at t−1) a deployed system would not have. The record naming resembles a public apnea-ECG corpus; only the supplied files were used.
