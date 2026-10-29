"""§4 Non-deep baselines: logistic regression and XGBoost on the §3 features.

This is the bar the deep model has to clear, so it gets the same inputs the deep model
gets (the window, or the whole 5-minute history for Task B) and its own hyper-parameters
are chosen on the validation split only. A second, stronger baseline (xgb_rel) also gets
the subject-relative "night so far" features, which the deep model does not see; it was
added after the first validation numbers showed the between-patient confound, and the
report says so.

Run:  python src/s4_baseline.py   (writes outputs/models/{task}_{model}_s{seed}.joblib and
                                   outputs/preds/{task}_{model}_s{seed}_val.csv; NO test predictions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, OUT_DIR, SEEDS, seed_everything  # noqa: E402
from s3_features import feature_columns, window_only_columns  # noqa: E402

PRED_DIR = OUT_DIR / "preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
META = ["subject_id", "minute", "window_start", "window_end", "label"]


def load_task(task: str, drop_ongoing: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / f"features_{task}.parquet")
    if task == "B" and drop_ongoing:
        df = df[df.ongoing_at_t == 0].reset_index(drop=True)
    return df


XGB_DEFAULT = dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=5)


def fit_predict(model_name: str, tr: pd.DataFrame, parts: dict[str, pd.DataFrame],
                seed: int = 0, feats: list[str] | None = None, save_as: str | None = None,
                xgb_params: dict | None = None) -> dict[str, np.ndarray]:
    """Fit on `tr`, return P(positive) for every DataFrame in `parts`. With save_as, the
    fitted pipeline and its feature list are stored for s6_evaluate.py, which is the only
    stage that ever scores the test split. xgb_params overrides XGB_DEFAULT (round-2 CV)."""
    seed_everything(seed)
    # lr / xgb: window-only features (same information as the deep model, §5 comparison)
    # xgb_rel : + subject-relative features, i.e. the night so far (second baseline, bonus)
    if feats is None:
        feats = feature_columns(tr) if model_name.endswith("_rel") else window_only_columns(tr)
    model_name = model_name.replace("_rel", "")
    Xtr, ytr = tr[feats].values, tr.label.values
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)       # imbalance: class weighting (§3)
    if model_name == "lr":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=(xgb_params or {}).get("C", 0.5), class_weight="balanced", max_iter=5000))
        model.fit(Xtr, ytr)
    elif model_name == "xgb":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          scale_pos_weight=pos_w, eval_metric="aucpr", random_state=seed,
                          n_jobs=4, tree_method="hist", **(XGB_DEFAULT | (xgb_params or {}))))
        model.fit(Xtr, ytr)
    else:
        raise ValueError(model_name)
    if save_as:
        joblib.dump({"model": model, "feats": feats}, MODEL_DIR / f"{save_as}.joblib")
    return {name: model.predict_proba(df[feats].values)[:, 1] for name, df in parts.items()}


def load_model(task: str, model: str, seed: int):
    d = joblib.load(MODEL_DIR / f"{task}_{model}_s{seed}.joblib")
    return d["model"], d["feats"]


def save_preds(df: pd.DataFrame, p: np.ndarray, task: str, model: str, seed: int, split: str) -> None:
    out = df[[c for c in META + ["horizon_start", "horizon_end"] if c in df.columns]].copy()
    out["p"] = p
    out.to_csv(PRED_DIR / f"{task}_{model}_s{seed}_{split}.csv", index=False)


def main() -> None:
    for task in ("A", "B"):
        df = load_task(task)
        parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val")}   # test is never loaded here
        print(f"Task {task}: " + ", ".join(f"{s}={len(d)} (prev {d.label.mean():.3f})" for s, d in parts.items()))
        for model in ("lr", "xgb", "xgb_rel"):
            for seed in (SEEDS if model.startswith("xgb") else SEEDS[:1]):     # LR is deterministic
                preds = fit_predict(model, parts["train"], {"val": parts["val"]}, seed=seed, save_as=f"{task}_{model}_s{seed}")
                save_preds(parts["val"], preds["val"], task, model, seed, "val")
                from sklearn.metrics import average_precision_score
                print(f"  {model} s{seed}: val AUPRC={average_precision_score(parts['val'].label, preds['val']):.3f}")


if __name__ == "__main__":
    main()
