# Temporal AI for ECG-based apnea detection and short-horizon onset prediction

Take-home assignment, Position 1 (PRS-I, SBILab / HMI Lab, IIIT-Delhi). Two binary tasks on
overnight single-lead ECG (100 Hz, WFDB, minute-level expert labels):

- **Task A — detection**: one annotated minute of ECG → P(apnea in that minute).
- **Task B — prospective onset prediction**: 5 minutes of ECG history ending at t → P(an apnea
  event *starts* in the next minute). Nothing at or after t is used; windows where an event is
  already in progress at t are excluded from the primary set (and reported separately).

Results, error analysis and calibration are in `REPORT.md`; `ANALYSIS.md` answers the
leakage-versus-difficulty question. This file says how to run it and what we assumed.

## Layout

```
src/common.py            seeds, the fixed subject split, metrics (implemented once)
src/s1_data.py           §1 load, alignment check, audit, R-peaks, quality flags, windows
src/s2_split.py          §2 split report, label autocorrelation, causality regression test
src/s3_features.py       §3/§4 HRV / ECG-derived-respiration / quality features
src/s4_baseline.py       §4 logistic regression + XGBoost (window-only and subject-relative)
src/s5_deep.py           §5 one temporal deep model: 1-D CNN minute encoder + GRU, 3 seeds
src/s6_evaluate.py       §6 model selection on validation, thresholds, test metrics, predictions.csv
src/s7_errors_calib.py   §7 error analysis, reliability diagram, Brier/ECE, isotonic recalibration
src/s8_controls.py       ANALYSIS controls: label shuffles, Task B variants, horizon/history sweeps,
                         detector on prediction windows
src/s9_posthoc_controls.py  post-hoc diagnostics, validation split only (label-shift test, future-minute
                         detector, gap sweep, CNN variants)
src/s10_prereg_bar.py    evaluation of the pre-registered Task B bar; Task A selection diagnostics
src/s11_val_tuning.py    round 2: validation-only hyper-parameter grid for the feature baselines
src/s14_round2_cv.py     round 2: grouped 6-fold CV on the 22 non-test nights (candidates, paired tests)
src/s14b_round2_cv_extra.py  round 2: added configurations on the same folds, merged into the CV table
src/s15_round2_final.py  round 2: promotion rule, retrain on 22 nights, single test scoring (gated)
run_all.sh               one command, both rounds; round 2 reproduces predictions.csv from the recorded selection with no test metric (s15 --reproduce-only)
requirements.txt         pinned dependencies
PREREGISTRATION.md       the stamped selection rules and bars, with timestamps
REPORT.md                renders to 3 A4 pages at 10.5 pt (headless Chrome print, table at 8 pt)
predictions.csv          final test-set predictions, both tasks (see format below)
outputs/                 written by the pipeline: results_table.md, error_analysis.md, controls.md,
                         posthoc_controls.md, *.json, figures/, models/, preds/ (the zip ships only the cited .md/.json/.csv files, figures and outputs/predictions_round1.csv)
data/ECG_P1/             place the supplied dataset here (not included in the zip)
```

## Setup

Python 3.12 (3.10+ should work). From the repo root:

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.11.0`. On a CPU-only machine, install the CPU wheel first to
avoid the CUDA download: `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`.
On Colab, see "Running on Google Colab" below.

**Data.** Unzip `ECG_P1.zip` so that `data/ECG_P1/RECORDS` exists (the `.dat/.hea/.apn/.qrs`
files next to it). Any other location works with `ECG_DATA_DIR=/path/to/ECG_P1`.

## Run

```
bash run_all.sh            # both rounds; regenerates predictions.csv in place (byte-identical to the shipped file, see runtime table)
bash run_all.sh --no-cnn   # same, but the control stages skip the CNN re-trainings (slow CPU)
# to run stages individually (same order as run_all.sh):
python src/s11_val_tuning.py          # round 2, validation-only grid (optional)
python src/s14_round2_cv.py           # round 2: grouped CV on the 22 non-test nights
python src/s14b_round2_cv_extra.py    # round 2: added configurations on the same folds
python src/s15_round2_final.py --dry-run           # promotion decision, no test row read
python src/s15_round2_final.py --confirm "<text>"  # retrain on 22 nights, score test once
python src/s17_leak_checks.py         # leak checks on the round-2 Task A final
python src/s7_errors_calib.py --round2            # error analysis of the round-2 finals
```

**Two rounds.** Round 1 (s1–s10) selected on a 6-night validation split and produced
`outputs/predictions_round1.csv`. Round 2 (s14–s15) selected by 6-fold grouped cross-validation on all 22
non-test nights under rules stamped in `PREREGISTRATION.md`, retrained the promoted pipelines on
those 22 nights and scored the test split once per task. The shipped `predictions.csv` is the
round-2 output (Task A: XGBoost on a causal 5-minute context with night-so-far scaling; Task B:
logistic regression on 5-minute-history window features). `outputs/round2_cv.md` has the
full candidate table, `outputs/round2_test.md` the test scoring with per-night tables, and
`outputs/leak_checks_round2_A.json` the leak checks run before the Task A number was quoted.

Stages can be run one at a time in the order listed in `run_all.sh`; each reads the previous
stage's cache. `src/s1_data.py --force` rebuilds the per-record cache.

Hardware and runtime are tabulated at the end of this file (measured, not estimated).

## Reproducibility

- Seeds are set for Python, NumPy and PyTorch (`common.seed_everything`; cuDNN deterministic).
  Re-running s4–s6 on this machine reproduced every validation and test prediction file
  byte-for-byte (md5).
- In round 1 the test split is loaded by exactly one stage, `s6_evaluate.py`, after the final
  model per task has been chosen on validation; s4/s5 never read test rows. In round 2 only
  `s15_round2_final.py --confirm` reads test labels, once per final pipeline (a mis-mapped Task B run was caught and re-run with the same pre-fixed choices; `PREREGISTRATION.md`); its `--reproduce-only` path reads none.
- `s2_split.py` doubles as a test. It corrupts every sample at or after the prediction time for
  30 random Task B windows, asserts that the feature vectors are unchanged and fails the run
  otherwise. `s1_data.py` asserts the annotation grid; `s6`/`s7` assert the
  `predictions.csv` contract.

## predictions.csv

One row per evaluated test window, both tasks, from the single final model per task.

| column | meaning |
|---|---|
| `subject_id` | record name (e.g. `a16`) |
| `window_start`, `window_end` | the **observation** window, seconds from record start (Task A: the evaluated minute; Task B: the 5-minute history, `window_end` = prediction time t) |
| `task` | `A_detection` or `B_onset_prediction` |
| `probability` | the final model's raw output, in [0, 1] |
| `prediction` | `probability >= threshold`, threshold = maximum F1 on the 22-night out-of-fold predictions (round 2), frozen before the test scoring |
| `horizon_start`, `horizon_end` | the interval the label refers to, seconds (Task A: same as the window; Task B: `(t, t+60]`) |
| `probability_calibrated` | isotonic regression fitted on the out-of-fold predictions applied to `probability`; same ranking. For Task B read this column as the probability: the raw output is NOT calibrated (ECE 0.34, balanced class weights). For Task A the raw probabilities are the better calibrated (REPORT §7) |

`outputs/predictions_round1.csv` has the same layout for the round-1 finals: CNN+GRU for Task A (threshold
0.007, as its raw probabilities are compressed) and XGBoost with night-so-far features for Task B.
Its calibrated column was fitted on the validation split.

Task B rows are the primary set only: prediction times at which no apnea was in progress
(minute t−1 labelled N). Windows with an ongoing event are excluded, as the brief requires, and
are analysed separately in `outputs/controls.md`.
The six required columns come first, in the stated order. The last three are additions, so that
the Task B observation window and horizon are unambiguous and calibrated probabilities are
available. No NaN; all probabilities in [0, 1] (asserted).

## Assumptions and trade-offs (documented, as the brief asks)

1. **ECG only.** Respiration/SpO₂ exist for 8 of 30 records. An ECG-only model covers every
   record and matches the stated single-lead setting. The companion `r`/`er` files are never
   read except to set an audit flag.
2. **`.qrs` files are not an input.** Our own R-peak detector is used everywhere; `.qrs` is read
   once, in the audit, to report the detector's agreement (PPV/sensitivity at 50/100 ms).
3. **Labels.** Annotation i describes minute [i, i+1), verified for all 30 records (every
   annotation at sample i·6000, steps of exactly 6000). Trailing minutes with signal but no
   annotation (−1 to 13 per record) are dropped. An onset is a minute labelled A whose
   predecessor is N.
4. **Task B primary set excludes windows with apnea at minute t−1** ("ongoing at t"). This is
   the strict reading of "must not be counted as a future prediction". It filters on a label
   that is not available at inference, so `outputs/controls.md` also reports the
   all-windows "deployment view" and the persistence-inflated variant.
5. **Causality.** Every Task B input comes from `ecg[t−300 s, t)` only: beat detection and the
   quality index run on the history slice, beats within 5 s of t are ignored, the CNN input is
   filtered and z-scored per window, and the night-so-far features use earlier windows
   only. Task A uses whole-record beat detection because the window itself is the evaluated
   interval.
6. **Split.** Fixed by hand, by record: train 16 / validation 6 / test 8 nights, each with
   severe, borderline and control recordings (`common.SPLIT`). Chosen for severity coverage,
   not by any score.
7. **Imbalance.** Class-weighted losses (`class_weight="balanced"`, `scale_pos_weight`,
   `pos_weight`) plus a validation-chosen threshold; no resampling, no synthetic data.
8. **Round-1 hyper-parameters are fixed, not searched.** LR C = 0.5; XGBoost 400 trees, depth 4,
   lr 0.05; CNN as in `s5_deep.py`. The brief scores evidence over leaderboard. The risk is a
   slightly under-tuned baseline (round 2 then searched a small validation grid; `PREREGISTRATION.md`).
9. **Model selection was fixed in code before any test metric (REPORT §4–6)**: highest mean validation AUPRC per task, ties within
   0.005 to the simpler model, the seed with the median validation AUPRC as "the model". It was
   not changed after test numbers existed, even where that leaves a weaker model as the final
   one (REPORT §4–6). Round 2 then re-selected by 22-night grouped CV under `PREREGISTRATION.md`.
10. **The second baseline (`xgb_rel`) sees night-so-far context** (running medians of earlier
    windows) that the CNN does not. The headline deep-vs-baseline comparison is against
    the window-only models, and the table shows both.
11. **For Task A the GRU runs on a sequence of length 1** and is inert; Task A is effectively
    CNN-only. One architecture is used for both tasks, as asked.
12. The record naming resembles a well-known public apnea-ECG corpus. Only the supplied files
    were used; no public labels, splits or published results were consulted.

## The 21 window features

Beat count; mean and SD of heart rate; SDNN; RMSSD; pNN50; RR min, max, range and CV; Lomb–Scargle VLF, LF and HF power and LF/HF; ECG-derived respiration from the R-amplitude series (SD, 0.1–0.5 Hz power, peak frequency); ECG SD and peak-to-peak; flat and clip fractions; plus the beat-template signal-quality index (SQI). For Task B the same features are computed over the 5-minute history, with per-minute trajectories of heart rate, SDNN, RMSSD and EDR SD and the last minute's deviation from the earlier ones. The round-2 Task A final adds the per-minute features of the labelled minute and the four before it, whole-context features, and a causal running median/IQR z-score of every feature over the night so far (272 features).

## Running on Google Colab

Upload or clone the folder and change into it; `pip install -r requirements.txt` (Colab's preinstalled torch may differ from the pin; the pinned versions are the ones measured). Place the data at `data/ECG_P1/` or mount Drive and set `ECG_DATA_DIR`. Then `bash run_all.sh` on a GPU runtime, or `bash run_all.sh --no-cnn` on a CPU runtime to skip the CNN re-trainings in the control stages. Measured on a laptop GPU (table below); a Colab T4 should be comparable; CPU-only was not measured. No Colab run was made.

## Hardware and runtime

Measured on a clean checkout (fresh virtual environment from `requirements.txt`, data placed as above, `bash run_all.sh` stage by stage) on a laptop with an AMD Ryzen 9 270 (16 threads), 14 GB RAM and an RTX 5060 8 GB GPU. CPU-only timings were not measured on this machine; the GPU-bound stages are s5, s8 and s9 (CNN training and re-training), and `--no-cnn` removes the CNN re-trainings from the control stages.

| stage | seconds (GPU laptop) |
|---|---|
| s1 data, R-peaks, windows | 33 |
| s3 features (Task A + B) | 172 |
| s2 split report and causality test | 4 |
| s4 baselines (2 tasks, 7 fits) | 10 |
| s5 CNN (2 tasks × 3 seeds) | 174 |
| s6 + s7 evaluation, errors, calibration | 29 |
| s8 controls (incl. CNN re-trainings) | 716 |
| s9 post-hoc diagnostics (validation) | 653 |
| s10 pre-registered bar | 38 |
| round 1 total | 1,831 (≈ 31 min) |
| s14 + s14b round-2 cross-validation | 1,634 (≈ 27 min) |
| s15 reproduce-only (refit on 22 nights, predictions) | 16 |

From the clean checkout, the round-1 predictions matched `outputs/predictions_round1.csv` exactly (6,276 rows, maximum probability difference 0, no prediction mismatch), the round-2 cross-validation tables matched to 6 decimals, and the round-2 refit reproduced `predictions.csv` exactly (6,276 rows, maximum probability difference 0, no prediction mismatch). Whole run: about 58 minutes on the GPU laptop.
