"""Validation-only tuning pass (round 2). Reads no test row.

A small, justified search (the brief allows one) over the feature baselines that were
shipped with fixed hyper-parameters: XGBoost depth / learning rate / trees / min_child_weight
and logistic-regression C, for both tasks, on the cached features. Scores validation pooled
AUPRC (the pre-fixed selection metric) and within-night AUROC. Writes outputs/val_tuning.md
and .json. Whether anything from this stage is promoted to a second test evaluation is a
separate, disclosed decision; this stage never touches the test split.

Run:  python src/s11_val_tuning.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, SEEDS, save_json, seed_everything  # noqa: E402
from s3_features import feature_columns, window_only_columns  # noqa: E402
from s4_baseline import load_task  # noqa: E402
from s6_evaluate import within_record  # noqa: E402

GRID_XGB = {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400], "min_child_weight": [5, 20]}
GRID_LR = {"C": [0.05, 0.2, 0.5, 2.0, 10.0]}
SHIPPED = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400, "min_child_weight": 5}


def run_xgb(tr, va, feats, params, seed):
    seed_everything(seed)
    pos_w = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    m = make_pipeline(SimpleImputer(strategy="median"),
                      XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=pos_w,
                                    eval_metric="aucpr", random_state=seed, n_jobs=8, tree_method="hist", **params))
    m.fit(tr[feats].values, tr.label.values)
    return m.predict_proba(va[feats].values)[:, 1]


def run_lr(tr, va, feats, C):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    m.fit(tr[feats].values, tr.label.values)
    return m.predict_proba(va[feats].values)[:, 1]


def score(va, p):
    return {"auprc": float(average_precision_score(va.label, p)),
            "within_auroc": within_record(va.assign(p=p), n_boot=50)["mean_auroc"]}


def main():
    R = {}
    for task in ("A", "B"):
        df = load_task(task)
        tr, va = (df[df.split == s].reset_index(drop=True) for s in ("train", "val"))
        for fam, feats in (("xgb", window_only_columns(df)), ("xgb_rel", feature_columns(df))):
            rows = []
            for combo in itertools.product(*GRID_XGB.values()):
                params = dict(zip(GRID_XGB.keys(), combo))
                sc = [score(va, run_xgb(tr, va, feats, params, s)) for s in SEEDS]
                rows.append({**params, "val_auprc_mean": np.mean([x["auprc"] for x in sc]), "val_auprc_sd": np.std([x["auprc"] for x in sc]),
                             "val_within_auroc_mean": np.mean([x["within_auroc"] for x in sc]), "shipped": params == SHIPPED})
                print(task, fam, params, f"AUPRC {rows[-1]['val_auprc_mean']:.3f} within {rows[-1]['val_within_auroc_mean']:.3f}")
            R[f"{task}_{fam}"] = pd.DataFrame(rows).sort_values("val_auprc_mean", ascending=False).to_dict("records")
        rows = []
        for C in GRID_LR["C"]:
            sc = score(va, run_lr(tr, va, window_only_columns(df), C))
            rows.append({"C": C, **{f"val_{k}": v for k, v in sc.items()}, "shipped": C == 0.5})
        R[f"{task}_lr"] = rows
    save_json(R, OUT_DIR / "val_tuning.json")
    md = ["# Validation-only tuning (round 2; test never read)", ""]
    for key, rows in R.items():
        md += [f"## {key}", ""]
        t = pd.DataFrame(rows)
        md.append(t.head(8).to_markdown(index=False, floatfmt=".3f"))
        ship = t[t.shipped] if "shipped" in t else t.iloc[0:0]
        if len(ship):
            md.append(f"\nShipped setting: AUPRC {ship.iloc[0].get('val_auprc_mean', ship.iloc[0].get('val_auprc')):.3f}; rank {int(np.where(t.shipped)[0][0]) + 1} of {len(t)}.")
        md.append("")
    (OUT_DIR / "val_tuning.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
