"""Round 2: grouped cross-validation on the 22 non-test nights. Reads no test LABEL.

Round 1 selected on a 6-night validation split. Round 2 selects by 6-fold grouped CV over
all train + validation nights (folds balanced by severity class), scoring pooled
out-of-fold AUPRC and mean within-night AUROC. Candidates, all causal:

 Task A (detection of the labelled minute)
   r1_window   round-1 window-only features (reference)
   r1_rel      round-1 features + subject-relative (reference)
   ctx5        causal 5-minute context: per-minute features of the labelled minute and the
               four before it, whole-context features, last-minute-minus-earlier trend
   ctx5_rz     ctx5 + night-so-far robust z-scores (running median / IQR of earlier windows)
   each with XGBoost at the shipped setting and at the best setting of the validation grid;
   then causal exponential smoothing of the out-of-fold probability over past minutes.
 Task B (onset in the next minute, primary set)
   r1_rel      round-1 features (reference), shipped and grid-best XGBoost
   r1_rel+det  + detector-history features from the OUT-OF-FOLD Task A probabilities of the
               best Task A candidate (apnea burden over the last 5 / 15 / 30 minutes, max
               over the last 5, minutes since the last out-of-fold "apnea" minute)
   W10         10-minute history features (computed causally here) with the same additions

Feature tables are computed for every record (test rows included, labels untouched) so that
the promotion stage can reuse them; nothing here reads a test label or scores a test row.
Writes outputs/round2_cv.md / .json and cache/round2_*.parquet.

Run:  python src/s14_round2_cv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FS, MIN_SAMPLES, OUT_DIR, SEEDS, SPLIT, save_json  # noqa: E402
from s1_data import REC_CACHE, detect_rpeaks, list_records, minute_sqi  # noqa: E402
from s3_features import CORE, TRAJ, feature_columns, segment_features, task_b_features, window_only_columns  # noqa: E402
from s4_baseline import fit_predict, load_task  # noqa: E402

PARAMS = {"A": {"shipped": dict(max_depth=4, learning_rate=0.05, n_estimators=400, min_child_weight=5),
                "grid": dict(max_depth=6, learning_rate=0.1, n_estimators=400, min_child_weight=20)},
          "B": {"shipped": dict(max_depth=4, learning_rate=0.05, n_estimators=400, min_child_weight=5),
                "grid": dict(max_depth=4, learning_rate=0.03, n_estimators=200, min_child_weight=20)}}
ALPHAS = (1.0, 0.7, 0.5, 0.3)
N_FOLDS = 6
META = {"subject_id", "split", "minute", "window_start", "window_end", "horizon_start", "horizon_end",
        "label", "ongoing_at_t", "apnea_in_horizon", "fold"}


# ----------------------------------------------------------------------------- folds
def folds_for(records: list[str]) -> dict[str, int]:
    out = {}
    for cls in "abc":
        for i, r in enumerate(sorted(r for r in records if r[0] == cls)):
            out[r] = i % N_FOLDS
    return out


# ----------------------------------------------------------------------------- Task A context features
def ctx_features(ecg: np.ndarray, k: int, n_ctx: int = 5) -> dict:
    """Features of the labelled minute k with up to n_ctx-1 earlier minutes as context.
    Beats and SQI come from the context slice only (nothing after minute k)."""
    s, e = max(0, k - n_ctx + 1) * MIN_SAMPLES, (k + 1) * MIN_SAMPLES
    seg = ecg[s:e]
    rp = detect_rpeaks(seg)
    sq = minute_sqi(seg, rp, (e - s) // MIN_SAMPLES)
    rp = rp + s
    f = {f"ctx_{a}": b for a, b in segment_features(ecg, rp, s, e).items()}
    f["ctx_sqi_mean"] = float(sq.mean())
    per = []
    for j in range(n_ctx):                         # j = 0 is the labelled minute, j = 1 the one before…
        m = k - j
        if m < 0:
            fm = {c: np.nan for c in CORE}
            f[f"sqi_l{j}"] = np.nan
        else:
            fm = segment_features(ecg, rp, m * MIN_SAMPLES, (m + 1) * MIN_SAMPLES)
            f[f"sqi_l{j}"] = float(sq[m - (s // MIN_SAMPLES)])
        per.append(fm)
        for c in CORE:
            f[f"{c}_l{j}"] = fm.get(c, np.nan)
    earlier = pd.DataFrame(per[1:])
    for c in TRAJ:
        f[f"{c}_l0_minus_mean"] = per[0].get(c, np.nan) - earlier[c].mean() if len(earlier) else np.nan
    return f


def build_ctx_table() -> pd.DataFrame:
    path = CACHE_DIR / "round2_featA_ctx5.parquet"
    if path.exists():
        return pd.read_parquet(path)
    A = pd.read_csv(CACHE_DIR / "windows_A.csv")
    rows = []
    for rec in list_records():
        ecg = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)["ecg"]
        for r in A[A.subject_id == rec].itertuples(index=False):
            rows.append({**r._asdict(), **ctx_features(ecg, int(r.minute))})
        print(f"  ctx5 features: {rec} done ({len(rows)} windows)")
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def add_robust_z(df: pd.DataFrame, cols: list[str], min_periods: int = 5) -> pd.DataFrame:
    """Night-so-far robust z: (x − running median) / running IQR over EARLIER windows only."""
    df = df.sort_values(["subject_id", "minute"]).reset_index(drop=True)
    g = df.groupby("subject_id")
    for c in cols:
        prev = g[c].shift(1)
        med = prev.groupby(df.subject_id).expanding(min_periods).median().reset_index(level=0, drop=True)
        q75 = prev.groupby(df.subject_id).expanding(min_periods).quantile(0.75).reset_index(level=0, drop=True)
        q25 = prev.groupby(df.subject_id).expanding(min_periods).quantile(0.25).reset_index(level=0, drop=True)
        df[f"{c}_rz"] = (df[c] - med) / ((q75 - q25).abs() + 1e-6)
    return df


# ----------------------------------------------------------------------------- CV machinery
def within_night(df: pd.DataFrame, p: np.ndarray, min_pos: int = 5) -> pd.Series:
    vals = {}
    for r, g in df.assign(p=p).groupby("subject_id"):
        if (g.label == 1).sum() >= min_pos and (g.label == 0).sum() >= min_pos:
            vals[r] = roc_auc_score(g.label, g.p)
    return pd.Series(vals)


def smooth(df: pd.DataFrame, p: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential smoothing of probabilities along each night (past outputs only)."""
    out = np.empty_like(p)
    d = df.assign(p=p, i=np.arange(len(p))).sort_values(["subject_id", "minute"])
    for _, g in d.groupby("subject_id"):
        prev = None
        for i, v in zip(g.i.values, g.p.values):
            prev = v if prev is None else alpha * v + (1 - alpha) * prev
            out[i] = prev
    return out


def cv_predict(df: pd.DataFrame, feats: list[str], params: dict, seed: int, model: str = "xgb") -> np.ndarray:
    """Out-of-fold probabilities over the 22 non-test nights. params: XGBoost overrides, or
    {"C": ...} for model="lr" (amendment 3)."""
    p = np.full(len(df), np.nan)
    for f in range(N_FOLDS):
        tr, te = df[df.fold != f], df[df.fold == f]
        if te.empty:
            continue
        pr = fit_predict(model, tr, {"te": te}, seed=seed, feats=feats, xgb_params=params)["te"]
        p[te.index.values] = pr
    return p


def fold_within_means(wn: pd.Series) -> list[float]:
    """Mean within-night AUROC per fold (nights with >= 5 of each class), NaN for empty folds."""
    return [float(wn[[r for r in wn.index if FOLD[r] == f]].mean()) if any(FOLD[r] == f for r in wn.index) else float("nan")
            for f in range(N_FOLDS)]


def evaluate_candidate(df: pd.DataFrame, feats: list[str], params: dict, name: str, alphas=(1.0,),
                       oof_override: dict | None = None, model: str = "xgb") -> dict:
    """CV one candidate over 3 seeds. oof_override: precomputed OOF probabilities per seed
    (used by the nested Task B stacking). Returns per-alpha summaries and the per-fold
    within-night means (seed-averaged) for the paired comparison with the comparator."""
    res, oof, folds = {}, {}, {}
    for seed in SEEDS:
        p = oof_override[seed] if oof_override is not None else cv_predict(df, feats, params, seed, model=model)
        oof[seed] = p
        for a in alphas:
            ps = smooth(df, p, a) if a < 1 else p
            wn = within_night(df, ps)
            fm = fold_within_means(wn)
            folds.setdefault(a, []).append(fm)
            res.setdefault(a, []).append({"auprc": float(average_precision_score(df.label, ps)), "auroc": float(roc_auc_score(df.label, ps)),
                                          "within": float(wn.mean()), "within_fold_sd": float(np.nanstd(fm)), "n_nights": int(len(wn)),
                                          "eligible_nights": sorted(wn.index.tolist())})
    out = {}
    for a, rows in res.items():
        t = pd.DataFrame(rows)
        out[f"alpha_{a}"] = {k: float(t[k].mean()) for k in ("auprc", "auroc", "within", "within_fold_sd")} | {
            "n_nights": int(t.n_nights.iloc[0]), "eligible_nights": rows[0]["eligible_nights"],
            "auprc_seed_sd": float(t.auprc.std(ddof=0)), "within_seed_sd": float(t.within.std(ddof=0)),
            "fold_within_means": np.nanmean(np.array(folds[a], dtype=float), axis=0).tolist()}
        print(f"  {name} alpha={a}: OOF AUPRC {out[f'alpha_{a}']['auprc']:.3f} AUROC {out[f'alpha_{a}']['auroc']:.3f} within {out[f'alpha_{a}']['within']:.3f} ± {out[f'alpha_{a}']['within_fold_sd']:.3f} (fold sd)")
    return {"results": out, "oof": oof, "feats": feats, "params": params}


def paired_vs(results: dict, cand: str, comp: str, alpha: str = "alpha_1.0") -> dict:
    """Mean of per-fold paired differences (candidate − comparator) and 2 × SE; clause 3."""
    a = np.array(results[cand][alpha]["fold_within_means"], dtype=float)
    b = np.array(results[comp][alpha]["fold_within_means"], dtype=float)
    d = a - b
    d = d[~np.isnan(d)]
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"comparator": comp, "mean_diff": float(d.mean()), "se": se, "n_folds": int(len(d)), "passes_2se": bool(d.mean() > 2 * se),
            "auprc_not_lower": bool(results[cand][alpha]["auprc"] >= results[comp][alpha]["auprc"])}


def nested_detector_frames(A_df: pd.DataFrame, feats: list[str], params: dict, seed: int) -> dict:
    """Per outer fold f: detector probabilities for all 22 nights where the training nights'
    values are inner 5-fold OOF among the training folds and the held-out fold's values come
    from a detector trained on the training folds. Also the max-F1 threshold on the inner
    OOF. Nothing in fold f's held-out nights touches the features their Task B model trains on."""
    out = {}
    for f in range(N_FOLDS):
        tr, te = A_df[A_df.fold != f], A_df[A_df.fold == f]
        p = np.full(len(A_df), np.nan)
        for g in range(N_FOLDS):
            if g == f:
                continue
            itr, ite = tr[tr.fold != g], tr[tr.fold == g]
            if ite.empty:
                continue
            p[ite.index.values] = fit_predict("xgb", itr, {"x": ite}, seed=seed, feats=feats, xgb_params=params)["x"]
        p[te.index.values] = fit_predict("xgb", tr, {"x": te}, seed=seed, feats=feats, xgb_params=params)["x"]
        inner = tr.assign(p=p[tr.index.values])
        cands = np.unique(np.quantile(inner.p, np.linspace(0.01, 0.99, 99)))
        thr = float(cands[int(np.argmax([f1_score(inner.label, inner.p >= t) for t in cands]))])
        out[f] = {"det": A_df[["subject_id", "minute"]].assign(p=p), "thr": thr}
    return out


def cv_predict_stacked(B: pd.DataFrame, base_feats: list[str], det_frames: dict, params: dict, seed: int) -> np.ndarray:
    """Task B OOF with nested detector-history features (one feature frame per outer fold)."""
    p = np.full(len(B), np.nan)
    for f in range(N_FOLDS):
        dh = detector_history_features(B, det_frames[f]["det"], det_frames[f]["thr"])
        Bf = B.merge(dh, on=["subject_id", "minute"], how="left")
        detcols = [c for c in dh.columns if c.startswith("det_")]
        tr, te = Bf[Bf.fold != f], Bf[Bf.fold == f]
        if te.empty:
            continue
        p[te.index.values] = fit_predict("xgb", tr, {"x": te}, seed=seed, feats=base_feats + detcols, xgb_params=params)["x"]
    return p


# ----------------------------------------------------------------------------- Task B additions
def detector_history_features(B: pd.DataFrame, det: pd.DataFrame, thr: float) -> pd.DataFrame:
    """det: per (subject_id, minute) out-of-fold detector probability. For each Task B window at
    prediction time t = k minutes: burden over minutes [k−5, k), [k−15, k), [k−30, k), max over
    [k−5, k), and minutes since the last minute with p >= thr (history only)."""
    out = []
    for rec, g in B.groupby("subject_id"):
        d = det[det.subject_id == rec].set_index("minute").p
        arr = np.full(int(d.index.max()) + 2, np.nan)
        arr[d.index.values] = d.values
        for r in g.itertuples(index=False):
            k = int(r.minute)
            h5, h15, h30 = arr[max(0, k - 5):k], arr[max(0, k - 15):k], arr[max(0, k - 30):k]
            above = np.flatnonzero(arr[:k] >= thr)
            out.append({"subject_id": rec, "minute": k, "det_burden5": np.nanmean(h5) if len(h5) else np.nan,
                        "det_burden15": np.nanmean(h15) if len(h15) else np.nan, "det_burden30": np.nanmean(h30) if len(h30) else np.nan,
                        "det_max5": np.nanmax(h5) if len(h5) else np.nan,
                        "det_since_last": float(k - above[-1]) if len(above) else 1000.0})
    return pd.DataFrame(out)


def build_w10_table() -> pd.DataFrame:
    path = CACHE_DIR / "round2_featB_W10.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from s1_data import task_b_windows
    rows = []
    for rec in list_records():
        z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
        win = task_b_windows(rec, z["labels"], history_min=10, horizon_min=1)
        for r in win.itertuples(index=False):
            rows.append({**r._asdict(), **task_b_features(z["ecg"], int(r.window_start * FS), int(r.window_end * FS))})
        print(f"  W10 features: {rec} done ({len(rows)} windows)")
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def nontest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.split != "test"].copy()
    d["fold"] = d.subject_id.map(FOLD)
    return d.reset_index(drop=True)


FOLD = folds_for(SPLIT["train"] + SPLIT["val"])


def main():
    R = {"folds": FOLD, "params": PARAMS}
    # ------------------------------------------------------------------ Task A
    A1 = load_task("A")
    A1 = nontest(A1)
    ctx = build_ctx_table()
    ctx_rz = add_robust_z(ctx, [c for c in ctx.columns if c not in META and c not in {"minute"}])
    ctxA = nontest(ctx_rz)
    featsA = {
        "r1_window": (A1, window_only_columns(A1)),
        "r1_rel": (A1, feature_columns(A1)),
        "ctx5": (ctxA, [c for c in ctxA.columns if c not in META and not c.endswith("_rz")]),
        "ctx5_rz": (ctxA, [c for c in ctxA.columns if c not in META]),
    }
    R["A"], best, best_key, store = {}, -1, None, {}
    for fname, (df, feats) in featsA.items():
        for pname, params in PARAMS["A"].items():
            key = f"{fname}|{pname}"
            cand = evaluate_candidate(df, feats, params, key, ALPHAS)
            R["A"][key] = cand["results"]
            store[key] = (df, feats, params)
            score = cand["results"]["alpha_1.0"]["within"]
            if score > best:
                best, best_key = score, key
    # comparator = best ROUND-1 family under the same CV (clause 3 as amended)
    r1_keys = [k for k in R["A"] if k.startswith("r1_")]
    compA = max(r1_keys, key=lambda k: R["A"][k]["alpha_1.0"]["within"])
    R["A"]["_comparator"] = compA
    R["A"]["_paired"] = {k: paired_vs(R["A"], k, compA) for k in R["A"] if not k.startswith("_") and k != compA}
    R["A"]["_best_by_within_alpha1"] = best_key
    print(f"Task A comparator {compA}; best {best_key}; paired: " + ", ".join(f"{k}: {v['mean_diff']:+.3f} (2SE {2*v['se']:.3f})" for k, v in R["A"]["_paired"].items()))

    # ------------------------------------------------------------------ Task B (nested stacking)
    detA_df, detA_feats, detA_params = store[best_key]
    B5 = nontest(load_task("B"))
    B10 = nontest(build_w10_table())
    B10 = B10[B10.ongoing_at_t == 0].reset_index(drop=True)
    R["B"] = {}
    for fname, df, base in (("W5_r1_rel", B5, feature_columns(B5)), ("W10_rel", B10, feature_columns(B10))):
        for pname, params in PARAMS["B"].items():
            R["B"][f"{fname}|{pname}"] = evaluate_candidate(df, base, params, f"{fname}|{pname}")["results"]
            oof = {}
            for seed in SEEDS:
                frames = nested_detector_frames(detA_df, detA_feats, detA_params, seed)
                oof[seed] = cv_predict_stacked(df, base, frames, params, seed)
            R["B"][f"{fname}+det|{pname}"] = evaluate_candidate(df, base, params, f"{fname}+det|{pname}", oof_override=oof)["results"]
    r1_keysB = [k for k in R["B"] if k.startswith("W5_r1_rel|")]
    compB = max(r1_keysB, key=lambda k: R["B"][k]["alpha_1.0"]["within"])
    R["B"]["_comparator"] = compB
    R["B"]["_paired"] = {k: paired_vs(R["B"], k, compB) for k in R["B"] if not k.startswith("_") and k != compB}
    R["B"]["_detector_for_B"] = {"candidate": best_key}
    print(f"Task B comparator {compB}; paired: " + ", ".join(f"{k}: {v['mean_diff']:+.3f} (2SE {2*v['se']:.3f})" for k, v in R["B"]["_paired"].items()))
    save_json(R, OUT_DIR / "round2_cv.json")

    md = ["# Round 2 — grouped 6-fold CV on the 22 non-test nights (3 seeds; mean)", "",
          f"Folds (balanced by class): {FOLD}", "",
          "## Task A — detection", "", "| candidate | smoothing α | OOF AUPRC | OOF AUROC | within-night AUROC | fold sd | seed sd |", "|---|---|---|---|---|---|---|"]
    for key, res in R["A"].items():
        if key.startswith("_"):
            continue
        for a, r in res.items():
            md.append(f"| {key} | {a[6:]} | {r['auprc']:.3f} | {r['auroc']:.3f} | {r['within']:.3f} | {r['within_fold_sd']:.3f} | {r['within_seed_sd']:.3f} |")
    md += ["", f"Comparator (best round-1 family under CV): {compA}. Best Task A by within-night AUROC (α = 1): {best_key}; its nested OOF detector probabilities feed the Task B detector-history features.", "",
           "Paired per-fold differences vs comparator (within-night AUROC, α = 1): " +
           "; ".join(f"{k}: {v['mean_diff']:+.3f}, 2×SE {2 * v['se']:.3f}, passes {v['passes_2se']}, AUPRC not lower {v['auprc_not_lower']}" for k, v in R["A"]["_paired"].items()), "",
           f"Eligible nights (≥ 5 positives and negatives): {R['A'][compA]['alpha_1.0']['eligible_nights']}", "",
           "## Task B — onset prediction (primary set)", "", "| candidate | OOF AUPRC | OOF AUROC | within-night AUROC | fold sd | seed sd | nights |", "|---|---|---|---|---|---|---|"]
    for key, res in R["B"].items():
        if key.startswith("_"):
            continue
        r = res["alpha_1.0"]
        md.append(f"| {key} | {r['auprc']:.3f} | {r['auroc']:.3f} | {r['within']:.3f} | {r['within_fold_sd']:.3f} | {r['within_seed_sd']:.3f} | {r['n_nights']} |")
    md += ["", f"Comparator: {compB}. Paired per-fold differences: " +
           "; ".join(f"{k}: {v['mean_diff']:+.3f}, 2×SE {2 * v['se']:.3f}, passes {v['passes_2se']}, AUPRC not lower {v['auprc_not_lower']}" for k, v in R["B"]["_paired"].items()),
           "", f"Eligible nights: {R['B'][compB]['alpha_1.0']['eligible_nights']}"]
    (OUT_DIR / "round2_cv.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
