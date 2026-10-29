"""Post-hoc diagnostics for ANALYSIS.md — VALIDATION SPLIT ONLY.

Everything in this file was decided AFTER the test numbers were first seen (29 Oct 2026,
12:08 IST). The brief forbids repeated experimentation on the test set, so nothing here
reads a test row: new models are trained on train, selected on validation and scored on
validation; the label-shift test re-scores the validation predictions written by s4/s5.
Nothing here changes the final models, the thresholds or predictions.csv.

 1. Label-shift test (alignment). Validation predictions are re-scored against labels
    shifted by k minutes within each record, k in -3..+3. If minute k's input really
    carries minute k's label, the score peaks at k = 0 and falls off on both sides.
 2. Future-minute detector. The Task A feature model is applied to the HORIZON minute
    itself (future information, illegal as a predictor) and scored on the Task B onset
    label. It shows how visible the first minute of an event is even with the future in
    hand, for comparison with the legitimate predictor.
 3. CNN selected by within-record AUROC instead of pooled AUPRC (3 seeds).
 4. CNN with random polarity flips during training (3 seeds). Lead polarity identifies a
    night but carries no apnea information.
 3/4 ask whether either remedy brings the CNN's within-night ranking near the feature
 models'. Validation answers that as well as test would.

Run:  python src/s9_posthoc_controls.py [--no-cnn]   (writes outputs/posthoc_controls.json / .md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, SEEDS, save_json  # noqa: E402
from s4_baseline import PRED_DIR, load_model, load_task  # noqa: E402
from s5_deep import WindowDataset, load_ecg, predict, train_model  # noqa: E402
from s6_evaluate import within_record  # noqa: E402

USE_CNN = "--no-cnn" not in sys.argv
SPLIT = "val"                                   # never "test" in this file


def pooled(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    ok = y.min() != y.max()
    return {"n": int(len(y)), "prevalence": float(y.mean()),
            "auprc": float(average_precision_score(y, p)) if ok else None,
            "auroc": float(roc_auc_score(y, p)) if ok else None}


def scored(df: pd.DataFrame) -> dict:
    w = within_record(df, n_boot=200)
    return {**pooled(df.label, df.p), "within_record_auroc": w["mean_auroc"], "within_record_n": w["n_records"]}


# ----------------------------------------------------------------------------- 1. shift
def label_shift(task: str, model: str, seed: int, shifts=range(-3, 4)) -> dict:
    va = pd.read_csv(PRED_DIR / f"{task}_{model}_s{seed}_{SPLIT}.csv").sort_values(["subject_id", "minute"])
    out = {}
    for k in shifts:
        parts = []
        for _, g in va.groupby("subject_id"):
            y, p = g.label.values, g.p.values
            if k > 0:
                parts.append(pd.DataFrame({"subject_id": g.subject_id.values[:-k], "label": y[k:], "p": p[:-k]}))
            elif k < 0:
                parts.append(pd.DataFrame({"subject_id": g.subject_id.values[-k:], "label": y[:k], "p": p[-k:]}))
            else:
                parts.append(g[["subject_id", "label", "p"]])
        out[f"shift_{k:+d}"] = scored(pd.concat(parts, ignore_index=True))
    return out


# ----------------------------------------------------------------------------- 2. future-minute detector
def future_minute_detector(model: str = "xgb_rel") -> dict:
    A = load_task("A")
    B = load_task("B")                                    # primary set: ongoing-at-t excluded
    A, B = A[A.split == SPLIT], B[B.split == SPLIT]
    out = {}
    for seed in SEEDS:
        pipe, feats = load_model("A", model, seed)
        A = A.assign(p=pipe.predict_proba(A[feats].values)[:, 1])
        m = B[["subject_id", "minute", "label"]].merge(A[["subject_id", "minute", "p"]], on=["subject_id", "minute"], how="inner")
        out[seed] = {SPLIT: scored(m)}
    return summarise(out)


# ----------------------------------------------------------------------------- 3/4. CNN
def cnn_variant(task: str, ecg: dict, select: str, flip_aug: bool) -> dict:
    df = load_task(task)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val")}    # no test rows
    out = {}
    for seed in SEEDS:
        model, info = train_model(parts["train"], parts["val"], ecg, seed, verbose=False, select=select, flip_aug=flip_aug)
        d = parts["val"].assign(p=predict(model, WindowDataset(parts["val"], ecg)))
        res = {SPLIT: scored(d), "epochs": info["epochs"], "best_val_selection_score": info["best_val_auprc"]}
        out[seed] = res
        print(f"  {task} select={select} flip={flip_aug} seed {seed}: val AUPRC {res['val']['auprc']:.3f} "
              f"AUROC {res['val']['auroc']:.3f} within {res['val']['within_record_auroc']:.3f}")
    return summarise(out)


# ----------------------------------------------------------------------------- 5. gap sweep
def gap_sweep(gaps=(0, 1, 2), W=5, model: str = "xgb", params: dict | None = None) -> dict:
    """Is the H = 1 skill prediction, or detection of an onset already under way in the last
    history minute (label-grid smear)? Predict onset in (t, t+1] from a history that ENDS at
    t − g minutes, g in {0, 1, 2}, on one common set of prediction times. If the skill
    collapses at g = 1, it lives in the last 60 s before t. Window-only XGBoost, 3 seeds,
    train -> validation only."""
    from common import MIN_SAMPLES, split_of
    from s1_data import REC_CACHE, list_records
    from s3_features import task_b_features, window_only_columns
    from s4_baseline import fit_predict
    rows = []
    for rec in list_records():
        sp = split_of(rec)
        if sp == "test":
            continue
        z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
        lab, ecg = z["labels"], z["ecg"]
        onset = np.zeros(len(lab), bool)
        onset[1:] = (lab[1:] == 1) & (lab[:-1] == 0)
        for k in range(W + max(gaps), len(lab)):
            if (lab[k - W - max(gaps):k + 1] < 0).any() or lab[k - 1] == 1:
                continue
            for g in gaps:
                s, e = (k - W - g) * MIN_SAMPLES, (k - g) * MIN_SAMPLES
                rows.append({"subject_id": rec, "split": sp, "minute": k, "gap": g, "label": int(onset[k]),
                             **task_b_features(ecg, s, e)})
        print(f"  gap sweep features: {rec} done ({len(rows)} rows)")
    df = pd.DataFrame(rows)
    feats = [c for c in window_only_columns(df) if c != "gap"]
    out = {}
    for g in gaps:
        d = df[df.gap == g]
        parts = {s: d[d.split == s].reset_index(drop=True) for s in ("train", "val")}
        per_seed = {}
        for seed in SEEDS:
            p = fit_predict(model, parts["train"], {"val": parts["val"]}, seed=seed, feats=feats, xgb_params=params)["val"]
            per_seed[seed] = {SPLIT: scored(parts["val"].assign(p=p))}
        out[f"gap_{g}"] = summarise(per_seed)
        print(f"  gap={g}: val AUPRC {out[f'gap_{g}']['val_auprc_mean']:.3f} AUROC {out[f'gap_{g}']['val_auroc_mean']:.3f} within {out[f'gap_{g}']['val_within_record_auroc_mean']:.3f}")
    return out


def gap_sweep_lr_main():
    """Fix requested in review: the gap sweep with the round-2 Task B final family (logistic
    regression, C = 0.5) on the same features and prediction times. Validation only."""
    C = {"split": SPLIT, "gap_sweep_lr_C0.5": gap_sweep(model="lr", params={"C": 0.5})}
    save_json(C, OUT_DIR / "posthoc_gap_sweep_lr.json")
    md = ["# Gap sweep with logistic regression C = 0.5 (validation; same features and prediction times as the XGBoost sweep)", "",
          "| gap g (min) | prevalence | AUPRC | lift | AUROC | within-record AUROC (3 nights) |", "|---|---|---|---|---|---|"]
    for g, r in C["gap_sweep_lr_C0.5"].items():
        md.append(f"| {g[-1]} | {r['val_prevalence_mean']:.3f} | {r['val_auprc_mean']:.3f} ± {r['val_auprc_sd']:.3f} | {r['val_auprc_mean'] / r['val_prevalence_mean']:.2f} | "
                  f"{r['val_auroc_mean']:.3f} ± {r['val_auroc_sd']:.3f} | {r['val_within_record_auroc_mean']:.3f} |")
    (OUT_DIR / "posthoc_gap_sweep_lr.md").write_text("\n".join(md))
    print("\n".join(md))


def summarise(per_seed: dict) -> dict:
    out = {"per_seed": per_seed}
    for k in ("auprc", "auroc", "within_record_auroc", "prevalence"):
        vals = [r[SPLIT][k] for r in per_seed.values() if r[SPLIT].get(k) is not None]
        out[f"{SPLIT}_{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{SPLIT}_{k}_sd"] = float(np.std(vals)) if vals else None
    return out


def main() -> None:
    C = {"split": SPLIT, "label_shift": {}}
    for model, seeds in (("xgb_rel", SEEDS), ("cnn", SEEDS), ("lr", SEEDS[:1])):
        for seed in seeds:
            C["label_shift"][f"A_{model}_s{seed}"] = label_shift("A", model, seed)
        print("label shift", model, {k: round(v["auroc"], 3) for k, v in C["label_shift"][f"A_{model}_s{seeds[0]}"].items()})

    C["future_minute_detector"] = {"xgb_rel": future_minute_detector("xgb_rel"), "xgb": future_minute_detector("xgb")}
    print("future-minute detector", {m: round(v[f"{SPLIT}_auprc_mean"], 3) for m, v in C["future_minute_detector"].items()})

    C["gap_sweep"] = gap_sweep()

    if USE_CNN:
        ecg = load_ecg()
        C["cnn_selected_by_within_record_auroc"] = {t: cnn_variant(t, ecg, "within_record_auroc", False) for t in ("A", "B")}
        C["cnn_polarity_flip_aug"] = {t: cnn_variant(t, ecg, "pooled_auprc", True) for t in ("A", "B")}

    save_json(C, OUT_DIR / "posthoc_controls.json")

    md = [f"# Post-hoc diagnostics ({SPLIT} split only; decided after test numbers were seen; see ANALYSIS.md)", "",
          f"## 1. Label-shift test, Task A {SPLIT} split (score vs labels shifted by k minutes within each record)", "",
          "| model | " + " | ".join(f"k={k:+d}" for k in range(-3, 4)) + " |", "|---|" + "---|" * 7]
    for key, sh in C["label_shift"].items():
        md.append(f"| {key} AUROC | " + " | ".join(f"{v['auroc']:.3f}" for v in sh.values()) + " |")
        md.append(f"| {key} AUPRC | " + " | ".join(f"{v['auprc']:.3f}" for v in sh.values()) + " |")
    md += ["", f"## 2. Future-minute detector: Task A model applied to the HORIZON minute, scored on the Task B onset label ({SPLIT})", "",
           "| model | prevalence | AUPRC | AUROC | within-record AUROC |", "|---|---|---|---|---|"]
    for m, r in C["future_minute_detector"].items():
        md.append(f"| {m} | {r[f'{SPLIT}_prevalence_mean']:.3f} | {r[f'{SPLIT}_auprc_mean']:.3f} ± {r[f'{SPLIT}_auprc_sd']:.3f} | "
                  f"{r[f'{SPLIT}_auroc_mean']:.3f} ± {r[f'{SPLIT}_auroc_sd']:.3f} | {r[f'{SPLIT}_within_record_auroc_mean']:.3f} |")
    md += ["", f"## 5. Gap sweep: onset in (t, t+1] from a 5-min history ending at t − g ({SPLIT}; window-only XGBoost, 3 seeds, common prediction times)", "",
           "| gap g (min) | prevalence | AUPRC | lift | AUROC | within-record AUROC |", "|---|---|---|---|---|---|"]
    for g, r in C["gap_sweep"].items():
        md.append(f"| {g[-1]} | {r['val_prevalence_mean']:.3f} | {r['val_auprc_mean']:.3f} ± {r['val_auprc_sd']:.3f} | "
                  f"{r['val_auprc_mean'] / r['val_prevalence_mean']:.2f} | {r['val_auroc_mean']:.3f} ± {r['val_auroc_sd']:.3f} | {r['val_within_record_auroc_mean']:.3f} |")
    if USE_CNN:
        md += ["", f"## 3/4. CNN variants (3 seeds; mean ± sd; {SPLIT} split)", "",
               "| variant | task | AUPRC | AUROC | within-record AUROC |", "|---|---|---|---|---|"]
        for name in ("cnn_selected_by_within_record_auroc", "cnn_polarity_flip_aug"):
            for t, r in C[name].items():
                md.append(f"| {name} | {t} | {r[f'{SPLIT}_auprc_mean']:.3f} ± {r[f'{SPLIT}_auprc_sd']:.3f} | {r[f'{SPLIT}_auroc_mean']:.3f} ± {r[f'{SPLIT}_auroc_sd']:.3f} | "
                          f"{r[f'{SPLIT}_within_record_auroc_mean']:.3f} ± {r[f'{SPLIT}_within_record_auroc_sd']:.3f} |")
    (OUT_DIR / "posthoc_controls.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    if "--gap-lr" in sys.argv:
        gap_sweep_lr_main()
    else:
        main()
