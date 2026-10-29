"""§6 Evaluation, threshold selection and the final predictions file.

This is the ONLY stage that scores the test split. Order of operations:
 1. Load every model's VALIDATION predictions (written by s4/s5), choose each model's
    threshold (max-F1 on validation) and the final model per task:
      best mean validation AUPRC; a tie (within 0.005) goes to the simpler model in the
      order LR < XGB < XGB-relative < CNN; for a stochastic family "the model" is the single
      seed whose validation AUPRC is the median of the three.
    (fixed in this file before any test metric was computed; see REPORT §4–6)
 2. Only then load the saved models, predict the test split for the chosen and the
    comparison models in one pass, and compute the §6 metrics.
Seeds give model variance (mean ± sd); a record-level bootstrap gives sampling variance
over the 8 test nights (95% CI). Both are reported because they measure different things.

Run:  python src/s6_evaluate.py   (writes outputs/preds/*_test.csv, outputs/results.json,
                                   outputs/results_table.md, predictions.csv)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, ROOT, evaluate, save_json, select_threshold  # noqa: E402
from s4_baseline import PRED_DIR, load_model, load_task, save_preds  # noqa: E402
from s5_deep import WindowDataset, load_cnn, load_ecg, predict  # noqa: E402

MODEL_LABEL = {"lr": "Logistic regression (HRV/EDR window features)",
               "xgb": "XGBoost (HRV/EDR window features)",
               "cnn": "1-D CNN + GRU (raw ECG window)",
               "xgb_rel": "XGBoost + subject-relative features (night-so-far context; second baseline)"}
SIMPLER_FIRST = ["lr", "xgb", "xgb_rel", "cnn"]
TIE = 0.005


def available(task: str) -> list[tuple[str, int]]:
    runs = set()
    for f in PRED_DIR.glob(f"{task}_*_s*_val.csv"):
        m = re.match(rf"{task}_(\w+?)_s(\d+)_val\.csv", f.name)
        runs.add((m.group(1), int(m.group(2))))
    return sorted(runs)


def load_pred(task: str, model: str, seed: int, split: str) -> pd.DataFrame:
    return pd.read_csv(PRED_DIR / f"{task}_{model}_s{seed}_{split}.csv")


def predict_test(task: str, model: str, seed: int, test_df: pd.DataFrame, ecg: dict) -> np.ndarray:
    if model == "cnn":
        return predict(load_cnn(task, seed), WindowDataset(test_df, ecg))
    pipe, feats = load_model(task, model, seed)
    return pipe.predict_proba(test_df[feats].values)[:, 1]


def record_bootstrap(df: pd.DataFrame, n: int = 1000, seed: int = 0) -> dict:
    """Resample TEST RECORDS with replacement; AUPRC/AUROC CI over nights."""
    rng = np.random.default_rng(seed)
    recs = df.subject_id.unique()
    groups = {r: df[df.subject_id == r] for r in recs}
    ap, au = [], []
    for _ in range(n):
        b = pd.concat([groups[r] for r in rng.choice(recs, len(recs), replace=True)])
        if b.label.nunique() < 2:
            continue
        ap.append(average_precision_score(b.label, b.p))
        au.append(roc_auc_score(b.label, b.p))
    return {"auprc_ci95": [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))],
            "auroc_ci95": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))]}


def within_record(df: pd.DataFrame, min_pos: int = 5, n_boot: int = 1000, seed: int = 0) -> dict:
    """Discrimination INSIDE each night (removes the patient-identity shortcut). Only records
    with >= min_pos positives and negatives are defined. CI = bootstrap over those records."""
    rows = []
    for r, g in df.groupby("subject_id"):
        if (g.label == 1).sum() >= min_pos and (g.label == 0).sum() >= min_pos:
            rows.append({"record": r, "n": len(g), "prevalence": g.label.mean(),
                         "auprc": average_precision_score(g.label, g.p),
                         "auroc": roc_auc_score(g.label, g.p)})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"records": [], "n_records": 0, "mean_auroc": float("nan"), "mean_auprc": float("nan"),
                "mean_lift": float("nan"), "auroc_ci95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    boots = [t.auroc.values[rng.integers(0, len(t), len(t))].mean() for _ in range(n_boot)]
    return {"records": t.to_dict("records"), "n_records": int(len(t)),
            "mean_auroc": float(t.auroc.mean()), "mean_auprc": float(t.auprc.mean()),
            "mean_lift": float((t.auprc / t.prevalence).mean()),
            "auroc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def choose_final(fam: dict[str, list[dict]]) -> tuple[str, dict]:
    means = {m: float(np.mean([r["val"]["auprc"] for r in runs])) for m, runs in fam.items()}
    best = max(means.values())
    candidates = [m for m in SIMPLER_FIRST if m in means and means[m] >= best - TIE]
    model = candidates[0]                                            # simplest within the tie band
    runs = sorted(fam[model], key=lambda r: r["val"]["auprc"])
    return model, runs[len(runs) // 2]                               # median-validation seed


def main() -> None:
    ecg = load_ecg()
    results, final = {}, {}
    for task in ("A", "B"):
        # ---- 1. validation only: thresholds and model selection
        results[task], fam = {}, {}
        for model, seed in available(task):
            va = load_pred(task, model, seed, "val")
            thr = select_threshold(va.label.values, va.p.values)
            r = {"model": model, "seed": seed, "threshold": thr, "val": evaluate(va.label.values, va.p.values, thr)}
            results[task][f"{model}_s{seed}"] = r
            fam.setdefault(model, []).append(r)
        best_model, chosen = choose_final(fam)
        print(f"Task {task}: final model selected on validation = {best_model} seed {chosen['seed']} "
              f"(val AUPRC {chosen['val']['auprc']:.3f}); family means: "
              + ", ".join(f"{m}={np.mean([r['val']['auprc'] for r in rs]):.3f}" for m, rs in fam.items()))

        # ---- 2. test, scored once, for every model (comparison table) after selection
        test_df = load_task(task)
        test_df = test_df[test_df.split == "test"].reset_index(drop=True)
        for key, r in results[task].items():
            p = predict_test(task, r["model"], r["seed"], test_df, ecg)
            save_preds(test_df, p, task, r["model"], r["seed"], "test")
            te = test_df.assign(p=p)
            r["test"] = evaluate(te.label.values, p, r["threshold"])
            r["test_within_record"] = within_record(te)
        summary = {}
        for model, runs in fam.items():
            tests = pd.DataFrame([r["test"] for r in runs])
            vals = pd.DataFrame([r["val"] for r in runs])
            summary[model] = {"n_seeds": len(runs), "val_auprc_mean": float(vals.auprc.mean()),
                              **{f"test_{c}_mean": float(tests[c].mean()) for c in tests.columns},
                              **{f"test_{c}_sd": float(tests[c].std(ddof=0)) for c in ("auprc", "auroc", "f1")},
                              "test_within_record_auroc_mean": float(np.mean([r["test_within_record"]["mean_auroc"] for r in runs]))}
        te = load_pred(task, best_model, chosen["seed"], "test")
        final[task] = {"model": best_model, "seed": chosen["seed"], "threshold": chosen["threshold"],
                       "val": chosen["val"], "test": chosen["test"], "test_bootstrap": record_bootstrap(te),
                       "test_within_record": chosen["test_within_record"]}
        results[task]["_summary"] = summary
        results[task]["_final"] = final[task]

    save_json(results, OUT_DIR / "results.json")

    # ---- one results table covering every model, both tasks
    lines = ["| Task | Model | seeds | Val AUPRC | Test prevalence | Test AUPRC | Test AUROC | Precision | Recall | F1 | Within-record AUROC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in ("A", "B"):
        s = results[task]["_summary"]
        for model in ("lr", "xgb", "cnn", "xgb_rel"):
            if model not in s:
                continue
            r = s[model]
            pm = lambda c: f"{r[f'test_{c}_mean']:.3f} ± {r[f'test_{c}_sd']:.3f}" if r["n_seeds"] > 1 else f"{r[f'test_{c}_mean']:.3f}"
            lines.append(f"| {task} | {MODEL_LABEL[model]} | {r['n_seeds']} | {r['val_auprc_mean']:.3f} | "
                         f"{r['test_prevalence_mean']:.3f} | {pm('auprc')} | {pm('auroc')} | "
                         f"{r['test_precision_mean']:.3f} | {r['test_recall_mean']:.3f} | {pm('f1')} | "
                         f"{r['test_within_record_auroc_mean']:.3f} |")
    lines.append("")
    for task in ("A", "B"):
        f = final[task]
        t, w = f["test"], f["test_within_record"]
        lines.append(f"**Task {task} final model (selected on validation):** {MODEL_LABEL[f['model']]} (seed {f['seed']}), "
                     f"threshold {f['threshold']:.3f}. Test AUPRC {t['auprc']:.3f} (record-bootstrap 95% CI "
                     f"{f['test_bootstrap']['auprc_ci95'][0]:.3f}–{f['test_bootstrap']['auprc_ci95'][1]:.3f}; prevalence {t['prevalence']:.3f}, "
                     f"lift {t['auprc'] / t['prevalence']:.1f}×), AUROC {t['auroc']:.3f} (CI {f['test_bootstrap']['auroc_ci95'][0]:.3f}–{f['test_bootstrap']['auroc_ci95'][1]:.3f}). "
                     f"Within-record AUROC {w['mean_auroc']:.3f} over {w['n_records']} records (CI {w['auroc_ci95'][0]:.3f}–{w['auroc_ci95'][1]:.3f}). "
                     f"Confusion matrix [TN FP / FN TP] = [{t['tn']} {t['fp']} / {t['fn']} {t['tp']}]. Brier {t['brier']:.3f}, ECE {t['ece']:.3f}.")
        lines.append("")
    (OUT_DIR / "results_table.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # ---- predictions.csv (both tasks, final model each)
    rows = []
    for task in ("A", "B"):
        f = final[task]
        te = load_pred(task, f["model"], f["seed"], "test")
        rows.append(pd.DataFrame({
            "subject_id": te.subject_id, "window_start": te.window_start, "window_end": te.window_end,
            "task": "A_detection" if task == "A" else "B_onset_prediction",
            "probability": te.p.clip(0, 1), "prediction": (te.p >= f["threshold"]).astype(int),
            # for detection the evaluated interval is the window itself
            "horizon_start": te.horizon_start if "horizon_start" in te else te.window_start,
            "horizon_end": te.horizon_end if "horizon_end" in te else te.window_end,
        }))
    pred = pd.concat(rows, ignore_index=True)
    assert pred.probability.between(0, 1).all() and not pred.isna().any().any()
    pred.to_csv(ROOT / "predictions.csv", index=False)
    print(f"predictions.csv: {len(pred)} rows")


if __name__ == "__main__":
    main()
