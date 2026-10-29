"""§2 Data split and leakage.

Reports the fixed subject-level split (train 16 / val 6 / test 8 records, each containing
severe, borderline and control nights) and runs two leakage checks that are claims in
the report and therefore have to be measured, not asserted:

 1. Label autocorrelation. Apnea minutes come in runs, so neighbouring windows share
    state; a random split of windows would leak. We report P(label_k == label_{k-1}) to
    show how strong that is, which is the reason for splitting by record.
 2. Causality regression test. Task B beats, SQI and features are computed from the
    history slice ecg[t-W:t] only. To prove the code path reads nothing after t, every
    sample at or after t is replaced by noise for 30 random windows and the feature
    vectors must be identical.

Run:  python src/s2_split.py   (writes outputs/split_report.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FS, OUT_DIR, SPLIT, save_json  # noqa: E402
from s1_data import REC_CACHE  # noqa: E402


def split_table() -> pd.DataFrame:
    A = pd.read_csv(CACHE_DIR / "windows_A.csv")
    B = pd.read_csv(CACHE_DIR / "windows_B.csv")
    B = B[B.ongoing_at_t == 0]
    rows = []
    for name, recs in SPLIT.items():
        a, b = A[A.split == name], B[B.split == name]
        rows.append({"split": name, "records": " ".join(recs), "n_records": len(recs),
                     "severe(a)": sum(r.startswith("a") for r in recs), "borderline(b)": sum(r.startswith("b") for r in recs),
                     "control(c)": sum(r.startswith("c") for r in recs),
                     "taskA_windows": len(a), "taskA_prevalence": round(a.label.mean(), 3),
                     "taskB_windows": len(b), "taskB_onset_prevalence": round(b.label.mean(), 3)})
    return pd.DataFrame(rows)


def label_autocorrelation() -> dict:
    A = pd.read_csv(CACHE_DIR / "windows_A.csv")
    same, runs = [], []
    for _, g in A.groupby("subject_id"):
        y = g.sort_values("minute").label.values
        same.append(np.mean(y[1:] == y[:-1]))
        # mean run length of apnea minutes
        r, cur = [], 0
        for v in y:
            if v == 1:
                cur += 1
            elif cur:
                r.append(cur); cur = 0
        if cur:
            r.append(cur)
        runs.append(np.mean(r) if r else 0)
    return {"p_same_as_previous_minute": float(np.mean(same)), "mean_apnea_run_length_min": float(np.mean(runs))}


def causality_regression_test(n_windows: int = 30, seed: int = 0) -> dict:
    """Task B features must depend on ecg[:t] only. For n random primary Task B windows the
    SHARED feature function (s3_features.task_b_features, the one that writes
    features_B.parquet) is run twice: on the real record and on a copy in which every
    sample at or after t is replaced by white noise; the vectors must be identical. The
    same is done for last_minute_features_causal, the input path of the detector-on-
    prediction-windows control. Finally the real-record vector is compared with the row
    stored in features_B.parquet, which ties the test to the artefact the models train on."""
    from s3_features import feature_columns, last_minute_features_causal, task_b_features
    rng = np.random.default_rng(seed)
    B = pd.read_csv(CACHE_DIR / "windows_B.csv")
    B = B[B.ongoing_at_t == 0].sample(n_windows, random_state=seed)
    pq = CACHE_DIR / "features_B.parquet"
    assert pq.exists(), "run s3_features.py first: the test must compare against the cached features"
    cached = pd.read_parquet(pq).set_index(["subject_id", "minute"])
    assert cached.index.is_unique, "features_B.parquet has duplicate (subject_id, minute) rows"
    res = {"windows_checked": 0, "taskB_changed": 0, "control_changed": 0, "parquet_mismatch": 0,
           "parquet_rows_compared": 0, "n_features": None, "n_cols_compared": None}

    def same(a, b):
        return all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)

    for row in B.itertuples(index=False):
        ecg = np.load(REC_CACHE / f"{row.subject_id}.npz", allow_pickle=True)["ecg"]
        s, e = int(row.window_start * FS), int(row.window_end * FS)
        corrupted = ecg.copy()
        corrupted[e:] = rng.normal(0, 1, len(ecg) - e).astype(np.float32)
        a, b = task_b_features(ecg, s, e), task_b_features(corrupted, s, e)
        res["taskB_changed"] += int(not same(a, b))
        c, d = last_minute_features_causal(ecg, s, e), last_minute_features_causal(corrupted, s, e)
        res["control_changed"] += int(not same(c, d))
        stored = cached.loc[(row.subject_id, row.minute)]
        missing = set(a) - set(stored.index)
        assert not missing, f"features missing from the parquet: {sorted(missing)}"
        res["n_features"], res["n_cols_compared"] = len(a), len(a)
        res["parquet_rows_compared"] += 1
        res["parquet_mismatch"] += int(not same(a, {k: float(stored[k]) for k in a}))
        res["windows_checked"] += 1
    res["passed"] = (res["taskB_changed"] == 0 and res["control_changed"] == 0 and res["parquet_mismatch"] == 0
                     and res["parquet_rows_compared"] == res["windows_checked"] > 0)
    return res


def relative_features_label_independence(seed: int = 0) -> dict:
    """The subject-relative (_rel) features are computed from earlier windows' features
    only. Permuting every label in the cached table and recomputing them must give the
    identical columns, which proves no label enters the baseline pool."""
    from s3_features import CORE, add_running_baseline
    out = {}
    for task in ("A", "B"):
        df = pd.read_parquet(CACHE_DIR / f"features_{task}.parquet")
        base = df[["subject_id", "minute"] + [f"{c}_rel" for c in CORE]].sort_values(["subject_id", "minute"]).reset_index(drop=True)
        shuffled = df.drop(columns=[f"{c}_rel" for c in CORE]).copy()
        shuffled["label"] = np.random.default_rng(seed).permutation(shuffled.label.values)
        if "ongoing_at_t" in shuffled:
            shuffled["ongoing_at_t"] = np.random.default_rng(seed + 1).permutation(shuffled.ongoing_at_t.values)
        again = add_running_baseline(shuffled)[["subject_id", "minute"] + [f"{c}_rel" for c in CORE]].sort_values(["subject_id", "minute"]).reset_index(drop=True)
        same = all(np.allclose(base[c], again[c], equal_nan=True) for c in base.columns if c.endswith("_rel"))
        out[task] = {"rows": len(base), "identical_after_label_permutation": bool(same)}
        assert same, f"Task {task}: relative features changed when labels were permuted"
    return out


def main() -> None:
    tab = split_table()
    ac = label_autocorrelation()
    caus = causality_regression_test()
    rel = relative_features_label_independence()
    print("Relative features identical after label permutation:", rel)
    print(tab.to_string(index=False))
    print(f"\nLabel autocorrelation: P(same as previous minute) = {ac['p_same_as_previous_minute']:.3f}, "
          f"mean apnea run = {ac['mean_apnea_run_length_min']:.1f} min")
    print(f"Causality regression test on {caus['windows_checked']} Task B windows (all samples >= t replaced by noise): "
          f"Task B feature vectors changed in {caus['taskB_changed']}, control-input vectors changed in "
          f"{caus['control_changed']}, cached parquet rows mismatching the live function: {caus['parquet_mismatch']} "
          f"of {caus['parquet_rows_compared']} -> {'PASSED' if caus['passed'] else 'FAILED'}")
    assert caus["passed"], "Task B features read samples at or after the prediction time"
    save_json({"split": tab.to_dict("records"), "label_autocorrelation": ac, "causality_regression_test": caus, "relative_features_label_independence": rel},
              OUT_DIR / "split_report.json")


if __name__ == "__main__":
    main()
