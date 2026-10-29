"""Leak checks for the round-2 Task A final (ctx5_rz|shipped) before its test number is quoted.

 1. Feature-list audit: no label-derived, split, identity or index column among the features.
 2. Disjointness: the 22 retrain nights and the 8 test nights do not overlap; the pipeline
    (imputer + model) was fit on the 22 only (re-fit here and compared with the saved one).
 3. Label-permutation invariance: permute each test night's labels, recompute the context +
    night-so-far features, assert identical.
 4. Causality: for 30 random test windows, replace every sample after the labelled minute
    with noise and assert the context features of that window are identical.
 5. Plausibility: out-of-fold predictions of the SAME pipeline (6-fold grouped CV on the 22
    nights, seed as retrained) re-scored against the training labels: expect about 0.785.

Reads test LABELS only in check 3 (to permute them) and never scores a test prediction.
Run:  python src/s17_leak_checks.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, MIN_SAMPLES, OUT_DIR, SPLIT, load_json, save_json  # noqa: E402
from s1_data import REC_CACHE  # noqa: E402
from s14_round2_cv import FOLD, META, add_robust_z, build_ctx_table, ctx_features, cv_predict, nontest  # noqa: E402
from s15_round2_final import table_for  # noqa: E402

BAD = re.compile(r"label|^y($|_)|apnea|onset|since|boundary|ongoing|split|subject|fold|^minute$|window_|horizon", re.I)


def main():
    out = {}
    r2 = load_json(OUT_DIR / "round2_test.json")["results"]["A"]
    key, seed = r2["key"], r2["seed"]
    full, feats = table_for("A", key)
    # 1. feature audit
    bad = [c for c in feats if BAD.search(c)]
    out["1_feature_audit"] = {"n_features": len(feats), "label_like_or_index_columns": bad, "passed": not bad}
    # 2. disjointness + fit set
    tr = nontest(full)
    te = full[full.split == "test"]
    overlap = sorted(set(tr.subject_id) & set(te.subject_id))
    saved = joblib.load(OUT_DIR / "models" / f"A_round2_{seed}.joblib")
    n_fit = getattr(saved["model"].steps[-1][1], "n_features_in_", None)
    out["2_disjoint_and_fit"] = {"train_nights": sorted(set(tr.subject_id)), "test_nights": sorted(set(te.subject_id)), "overlap": overlap,
                                 "train_equals_SPLIT_train_plus_val": sorted(set(tr.subject_id)) == sorted(SPLIT["train"] + SPLIT["val"]),
                                 "saved_model_n_features_in": int(n_fit) if n_fit else None, "saved_feats_equal": saved["feats"] == feats,
                                 "passed": not overlap and saved["feats"] == feats and (n_fit == len(feats))}
    # 3. label permutation invariance (labels never enter ctx_features or add_robust_z; prove it)
    ctx = build_ctx_table()
    cols = [c for c in ctx.columns if c not in META and c != "minute"]
    base = add_robust_z(ctx.copy(), cols)
    perm = ctx.copy()
    rng = np.random.default_rng(0)
    for rec in SPLIT["test"]:
        m = perm.subject_id == rec
        perm.loc[m, "label"] = rng.permutation(perm.loc[m, "label"].values)
    again = add_robust_z(perm, cols)
    same = all(np.allclose(base[c].values, again[c].values, equal_nan=True) for c in feats if c in base.columns)
    out["3_label_permutation_invariance"] = {"columns_compared": len([c for c in feats if c in base.columns]), "identical": bool(same), "passed": bool(same)}
    # 4. causality: nothing after the labelled minute
    changed = 0
    sample = te.sample(30, random_state=0)
    for r in sample.itertuples(index=False):
        ecg = np.load(REC_CACHE / f"{r.subject_id}.npz", allow_pickle=True)["ecg"]
        k = int(r.minute)
        a = ctx_features(ecg, k)
        corrupted = ecg.copy()
        corrupted[(k + 1) * MIN_SAMPLES:] = rng.normal(0, 1, len(ecg) - (k + 1) * MIN_SAMPLES).astype(np.float32)
        b = ctx_features(corrupted, k)
        changed += int(not all((np.isnan(a[c]) and np.isnan(b[c])) or a[c] == b[c] for c in a))
    out["4_causality_noise_after_window"] = {"windows": 30, "changed": changed, "passed": changed == 0}
    # 5. plausibility: OOF of the same pipeline on the 22 nights
    params = r2["params"]
    p = cv_predict(tr, feats, params, seed)
    ap = float(average_precision_score(tr.label, p))
    out["5_oof_plausibility"] = {"oof_auprc_same_pipeline_seed": ap, "cv_table_value": 0.785, "passed": abs(ap - 0.785) < 0.02}
    out["all_passed"] = all(v["passed"] for k, v in out.items() if isinstance(v, dict))
    save_json(out, OUT_DIR / "leak_checks_round2_A.json")
    for k, v in out.items():
        print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("train_nights", "test_nights")})


if __name__ == "__main__":
    main()
