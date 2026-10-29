"""Controls for ANALYSIS.md: is detection leaking, or is prediction hard?

 1. Shuffled labels, two ways. GLOBAL shuffle of training labels: any score above
    prevalence/0.5 is a pipeline leak. WITHIN-RECORD shuffle: keeps each night's
    severity, so whatever survives is the patient-identity shortcut, not timing.
 2. Task B three ways: primary (ongoing-at-t excluded), deployment view (all windows,
    positive only on a new onset), persistence-inflated (positive if apnea anywhere in
    the horizon: the shortcut a careless setup would reward).
 3. Horizon sweep H in {1,2,3,5,10} on ONE fixed set of prediction times (valid for
    H=10), reporting lift = AUPRC/prevalence and AUROC so rising prevalence cannot pose
    as skill.
 4. History sweep W in {1,3,5,10} with H=1 on a common set of prediction times.
 5. Detection model evaluated on prediction windows: the Task A model scores the last
    history minute (which is 'N' by construction in the primary set); if it is still
    informative about the coming onset there is a pre-onset signature.
 6. Seeds: everything stochastic is run with 3 seeds.

Run:  python src/s8_controls.py [--no-cnn]   (writes outputs/controls.json / .md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FS, OUT_DIR, SEEDS, seed_everything, save_json  # noqa: E402
from s1_data import REC_CACHE, list_records, task_b_windows  # noqa: E402
from s3_features import feature_columns, last_minute_features_causal, task_b_windows_features, window_only_columns  # noqa: E402
from s4_baseline import fit_predict, load_task  # noqa: E402
from s5_deep import DEVICE, MODEL_DIR, CNNGRU, WindowDataset, load_ecg, predict, train_model  # noqa: E402

USE_CNN = "--no-cnn" not in sys.argv


def score(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "prevalence": float(y.mean()),
            "auprc": float(average_precision_score(y, p)) if y.min() != y.max() else None,
            "lift": float(average_precision_score(y, p) / y.mean()) if 0 < y.mean() < 1 else None,
            "auroc": float(roc_auc_score(y, p)) if y.min() != y.max() else None}


def parts_of(df: pd.DataFrame) -> dict:
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def run_xgb(parts: dict, seeds=SEEDS, label_col: str = "label", model: str = "xgb_rel",
            feats: list[str] | None = None) -> dict:
    """feats must be given explicitly whenever the frame carries extra label columns (y, y_H*),
    otherwise feature_columns() would sweep them in and the model would train on its target."""
    if feats is None:
        feats = feature_columns(parts["train"]) if model.endswith("_rel") else window_only_columns(parts["train"])
    bad = [c for c in feats if c in ("label", "y", "ongoing_at_t", "apnea_in_horizon") or c.startswith("y_H")]
    assert not bad, f"label-like columns in the feature list: {bad}"
    out = {}
    for seed in seeds:
        tr = parts["train"].copy()
        tr["label"] = tr[label_col]
        p = fit_predict(model, tr, {"val": parts["val"], "test": parts["test"]}, seed=seed, feats=feats)
        out[seed] = {s: score(parts[s][label_col], p[s]) for s in ("val", "test")}
    return summarise(out)


def run_cnn(parts: dict, ecg: dict, seed: int = 0, label_col: str = "label") -> dict:
    tr, va, te = (parts[s].copy() for s in ("train", "val", "test"))
    for d in (tr, va, te):
        d["label"] = d[label_col]
    model, _ = train_model(tr, va, ecg, seed, verbose=False)
    return summarise({seed: {s: score(d.label, predict(model, WindowDataset(d, ecg))) for s, d in (("val", va), ("test", te))}})


def summarise(per_seed: dict) -> dict:
    out = {"per_seed": per_seed}
    for split in ("val", "test"):
        for k in ("auprc", "lift", "auroc", "prevalence"):
            vals = [r[split][k] for r in per_seed.values() if r[split][k] is not None]
            out[f"{split}_{k}_mean"] = float(np.mean(vals)) if vals else None
            out[f"{split}_{k}_sd"] = float(np.std(vals)) if vals else None
    return out


def shuffle_labels(df: pd.DataFrame, how: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    y = df.label.values.copy()
    if how == "global":
        rng.shuffle(y)
    else:                                  # within record
        for rec in df.subject_id.unique():
            m = (df.subject_id == rec).values
            y[m] = rng.permutation(y[m])
    return pd.Series(y, index=df.index)


def main() -> None:
    ecg = load_ecg() if USE_CNN else None
    C = {}

    # ------------------------------------------------------------- 1. shuffled labels
    for task in ("A", "B"):
        parts = parts_of(load_task(task))
        for how in ("global", "within_record"):
            key = f"shuffle_{how}_task{task}"
            res = {}
            for seed in SEEDS:
                p = {k: v.copy() for k, v in parts.items()}
                p["train"]["label"] = shuffle_labels(p["train"], how, seed).values
                pr = fit_predict("xgb_rel", p["train"], {"val": p["val"], "test": p["test"]}, seed=seed)
                res[seed] = {s: score(parts[s].label, pr[s]) for s in ("val", "test")}
            C[key] = {"xgb_rel": summarise(res)}
            if USE_CNN:
                p = {k: v.copy() for k, v in parts.items()}
                p["train"]["label"] = shuffle_labels(p["train"], how, 0).values
                model, _ = train_model(p["train"], p["val"], ecg, 0, max_epochs=10, verbose=False)
                C[key]["cnn"] = summarise({0: {s: score(parts[s].label, predict(model, WindowDataset(parts[s], ecg)))
                                              for s in ("val", "test")}})
            print(key, {m: (round(C[key][m]["test_auprc_mean"], 3), round(C[key][m]["test_auroc_mean"], 3)) for m in C[key]})

    # ------------------------------------------------------------- 2. Task B three ways
    full = load_task("B", drop_ongoing=False)
    variants = {
        "primary_ongoing_excluded": full[full.ongoing_at_t == 0].assign(y=lambda d: d.label),
        "deployment_all_windows_onset_only": full.assign(y=lambda d: d.label),
        "persistence_inflated_apnea_in_horizon": full.assign(y=lambda d: d.apnea_in_horizon),
    }
    FEATS_REL = feature_columns(load_task("B"))           # the pristine Task B feature list (no y / y_H* columns)
    for name, df in variants.items():
        parts = parts_of(df)
        C[f"taskB_{name}"] = {"xgb_rel": run_xgb(parts, label_col="y", feats=FEATS_REL)}
        if USE_CNN:
            C[f"taskB_{name}"]["cnn"] = run_cnn(parts, ecg, label_col="y")
        print(name, {m: round(C[f'taskB_{name}'][m]["test_auprc_mean"], 3) for m in C[f"taskB_{name}"]})

    # ------------------------------------------------------------- 3. horizon sweep
    featB = load_task("B", drop_ongoing=False)
    labels = {r: np.load(REC_CACHE / f"{r}.npz", allow_pickle=True)["labels"] for r in list_records()}
    rows = []
    for rec, lab in labels.items():
        onset = np.zeros(len(lab), bool)
        onset[1:] = (lab[1:] == 1) & (lab[:-1] == 0)
        for k in range(5, len(lab) - 10 + 1):
            if (lab[k - 5:k + 10] < 0).any() or lab[k - 1] == 1:
                continue
            rows.append({"subject_id": rec, "minute": k, **{f"y_H{H}": int(onset[k:k + H].any()) for H in (1, 2, 3, 5, 10)}})
    sweep = featB.merge(pd.DataFrame(rows), on=["subject_id", "minute"])
    parts = parts_of(sweep)
    C["horizon_sweep"] = {f"H{H}": run_xgb(parts, label_col=f"y_H{H}", feats=FEATS_REL) for H in (1, 2, 3, 5, 10)}
    print("horizon sweep", {H: (round(v["test_prevalence_mean"], 3), round(v["test_lift_mean"], 2), round(v["test_auroc_mean"], 3)) for H, v in C["horizon_sweep"].items()})

    # ------------------------------------------------------------- 4. history sweep
    C["history_sweep"] = {}
    for W in (1, 3, 5, 10):
        feats = []
        for rec, lab in labels.items():
            z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
            e_ = z["ecg"]
            win = task_b_windows(rec, lab, history_min=W, horizon_min=1)
            win = win[(win.ongoing_at_t == 0) & (win.minute >= 10)]
            feats += task_b_windows_features(e_, win)          # same causal function as s3
        df = pd.DataFrame(feats)
        C["history_sweep"][f"W{W}"] = run_xgb(parts_of(df), model="xgb")   # no running baseline here
        print(f"history W={W}", round(C["history_sweep"][f"W{W}"]["test_auprc_mean"], 3), round(C["history_sweep"][f"W{W}"]["test_auroc_mean"], 3))

    # ------------------------------------------------------------- 5. detector on prediction windows
    A = load_task("A")
    B = load_task("B")                                  # primary set
    B["last_minute"] = (B.window_end // 60 - 1).astype(int)
    featsA = window_only_columns(A)        # the control input carries no subject baseline
    # Input = Task A-style features of the last history minute, computed CAUSALLY (beats and
    # SQI from the history slice via last_minute_features_causal), never from features_A,
    # whose whole-record beats would look past t.
    rows = []
    for rec, g in B.groupby("subject_id"):
        rec_ecg = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)["ecg"]    # not `ecg`: that is the dict the CNN uses below
        for r in g.itertuples(index=False):
            rows.append({"subject_id": rec, "minute": r.minute, **last_minute_features_causal(rec_ecg, int(r.window_start * FS), int(r.window_end * FS))})
    last = B[["subject_id", "minute"]].merge(pd.DataFrame(rows), on=["subject_id", "minute"], how="left")
    partsA = parts_of(A)
    res = {}
    for seed in SEEDS:
        pr = fit_predict("xgb", partsA["train"], {"last": last}, seed=seed, feats=featsA)["last"]
        res[seed] = {s: score(B.label[B.split == s], pr[(B.split == s).values]) for s in ("val", "test")}
    C["detector_on_prediction_windows"] = {"xgb": summarise(res)}
    if USE_CNN and (MODEL_DIR / "A_cnn_s0.pt").exists():
        model = CNNGRU().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_DIR / "A_cnn_s0.pt", map_location=DEVICE))
        lastwin = B[["subject_id", "last_minute", "split", "label"]].rename(columns={"last_minute": "minute"})
        lastwin["window_start"], lastwin["window_end"] = lastwin.minute * 60, (lastwin.minute + 1) * 60
        p = predict(model, WindowDataset(lastwin, ecg))
        C["detector_on_prediction_windows"]["cnn"] = summarise({0: {s: score(lastwin.label[lastwin.split == s], p[(lastwin.split == s).values]) for s in ("val", "test")}})
    print("detector on prediction windows", {m: (round(v["test_auprc_mean"], 3), round(v["test_auroc_mean"], 3)) for m, v in C["detector_on_prediction_windows"].items()})

    save_json(C, OUT_DIR / "controls.json")

    # ------------------------------------------------------------- markdown summary
    md = ["# Controls (test split; mean ± sd over seeds where applicable)", ""]
    def line(name, r):
        f = lambda k: "–" if r.get(f"test_{k}_mean") is None else f"{r[f'test_{k}_mean']:.3f}" + (f" ± {r[f'test_{k}_sd']:.3f}" if r.get(f"test_{k}_sd") else "")
        return f"| {name} | {r['test_prevalence_mean']:.3f} | {f('auprc')} | {f('lift')} | {f('auroc')} |"
    md += ["| control | prevalence | AUPRC | lift | AUROC |", "|---|---|---|---|---|"]
    for key in [k for k in C if k.startswith("shuffle") or k.startswith("taskB_") or k == "detector_on_prediction_windows"]:
        for m, r in C[key].items():
            md.append(line(f"{key} [{m}]", r))
    for key in ("horizon_sweep", "history_sweep"):
        for sub, r in C[key].items():
            md.append(line(f"{key} {sub} [{'xgb' if key == 'history_sweep' else 'xgb_rel'}]", r))
    (OUT_DIR / "controls.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
