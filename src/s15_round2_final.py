"""Round 2 final: apply the stamped promotion rule, retrain on all 22 non-test nights, and
score the test split ONCE for the promoted pipelines (PREREGISTRATION.md, Round 2).

Steps
 1. Read outputs/round2_cv.json. Per task, the candidates that pass clause 3 (paired per-fold
    within-night difference vs the comparator > 2 × SE, pooled OOF AUPRC not lower); the
    promoted one is the passing candidate with the highest CV within-night AUROC. If none
    passes, the task keeps its round-1 final and nothing is scored.
 2. Recompute the promoted candidate's out-of-fold predictions on the 22 nights (3 seeds) to
    fix the threshold (max-F1 on pooled OOF), the median seed (by OOF AUPRC) and, for Task A,
    the causal smoothing constant (best OOF within-night AUROC among the stamped four).
 3. Retrain on all 22 nights. For a Task B pipeline with detector-history features, training
    features come from 6-fold OOF detector predictions and test features from the detector
    trained on all 22 nights (PREREGISTRATION §5).
 4. Score test once: §6 metrics, record bootstrap, within-night AUROC. Write
    outputs/round2_test.json / .md and predictions_round2.csv. predictions.csv is NOT
    overwritten here; the swap is a separate, documented step.

The test scoring runs only with --confirm "<text>", where <text> is the candidate's written
confirmation, which is stored in outputs/round2_test.json.

Run:  python src/s15_round2_final.py --dry-run          (promotion decision only, no test)
      python src/s15_round2_final.py --confirm "..."    (promotion + single test scoring)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, OUT_DIR, ROOT, SEEDS, evaluate, load_json, save_json  # noqa: E402
from s3_features import feature_columns, window_only_columns  # noqa: E402
from s4_baseline import fit_predict, load_task  # noqa: E402
from s6_evaluate import record_bootstrap, within_record  # noqa: E402
from s14_round2_cv import (ALPHAS, FOLD, META, PARAMS, add_robust_z, build_ctx_table, build_w10_table,  # noqa: E402
                           cv_predict, detector_history_features, nontest, smooth, within_night)


ROUND1_FINAL_CV_KEY = {"A": None, "B": "W5_r1_rel|shipped"}   # the round-1 final's CV key (A's CNN is not in CV)


def promoted(cv: dict, task: str) -> tuple[str | None, dict]:
    """Clause 3 as amended: (a) a round-2 candidate replaces the round-1 final if its paired
    within-night difference vs the comparator > 2 × SE and its OOF AUPRC is not lower; among
    several passers the one with the highest CV within-night AUROC; (b) otherwise, if CV ranks
    a round-1 family (the comparator, a round-1 configuration) above the round-1 final on
    within-night AUROC and OOF AUPRC, that alone counts as a promotion, disclosed."""
    comp = cv[task]["_comparator"]
    paired = cv[task]["_paired"]
    passing = {k: v for k, v in paired.items() if v["passes_2se"] and v["auprc_not_lower"]}
    if passing:
        best = max(passing, key=lambda k: cv[task][k]["alpha_1.0"]["within"])
        return best, {"comparator": comp, "passing": sorted(passing), "chosen": best, "paired": paired[best], "route": "clause 3a (paired > 2 SE)"}
    r1_final = ROUND1_FINAL_CV_KEY[task]
    if r1_final and r1_final != comp:
        c, f = cv[task][comp]["alpha_1.0"], cv[task][r1_final]["alpha_1.0"]
        if c["within"] > f["within"] and c["auprc"] >= f["auprc"]:
            return comp, {"comparator": comp, "passing": [], "chosen": comp, "route": "clause 3b (round-1 family ranked above round-1 final under CV; no noise margin in the stamped text)",
                          "vs_round1_final": {"round1_final": r1_final, "within": [c["within"], f["within"]], "oof_auprc": [c["auprc"], f["auprc"]],
                                              "paired_diff_final_vs_comp": paired.get(r1_final)}}
    return None, {"comparator": comp, "passing": []}


def table_for(task: str, key: str) -> tuple[pd.DataFrame, list[str]]:
    """Full feature table (all 30 nights) and feature list for a candidate key."""
    fname = key.split("|")[0]
    if task == "A":
        if fname.startswith("r1_"):
            df = load_task("A")
            return df, (window_only_columns(df) if fname == "r1_window" else feature_columns(df))
        ctx = add_robust_z(build_ctx_table(), [c for c in build_ctx_table().columns if c not in META and c != "minute"])
        feats = [c for c in ctx.columns if c not in META and (fname == "ctx5_rz" or not c.endswith("_rz"))]
        return ctx, feats
    base = fname.replace("+det", "")
    if base.startswith("W5_"):
        df = load_task("B")                                    # 5-minute history, primary set
    else:
        df = build_w10_table()                                 # 10-minute history
        df = df[df.ongoing_at_t == 0].reset_index(drop=True)
    # exactly the feature list the CV evaluated for this key (s14 / s14b)
    return df, (window_only_columns(df) if base.endswith("_window") else feature_columns(df))


def reproduce(decision: dict, cv: dict, out_name: str = "predictions_round2_repro.csv") -> None:
    """Clean-checkout reproduction: same selection, seed, smoothing constant and threshold as
    recorded in outputs/round2_test.json; refit on the 22 nights; recompute the out-of-fold
    predictions for the isotonic column; write the full predictions file. No metric is
    computed and no test label is read."""
    from sklearn.isotonic import IsotonicRegression
    rec = load_json(OUT_DIR / "round2_test.json")["results"]
    rows = []
    for task in ("A", "B"):
        key = decision["tasks"][task]["promoted"]
        if key is None or task not in rec:
            continue
        r = rec[task]
        pname = key.split("|")[1]
        model = "lr" if pname.startswith("lr_") else "xgb"
        params = r["params"]
        full, feats = table_for(task, key)
        tr = nontest(full)
        te = full[full.split == "test"].reset_index(drop=True)
        p_oof = cv_predict(tr, feats, params, r["seed"], model=model)
        if r["alpha"] < 1:
            p_oof = smooth(tr, p_oof, r["alpha"])
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_oof, tr.label.values)
        p_te = fit_predict(model, tr, {"te": te}, seed=r["seed"], feats=feats, xgb_params=params)["te"]
        if r["alpha"] < 1:
            p_te = smooth(te, p_te, r["alpha"])
        rows.append(pd.DataFrame({"subject_id": te.subject_id, "window_start": te.window_start, "window_end": te.window_end,
                                  "task": "A_detection" if task == "A" else "B_onset_prediction",
                                  "probability": np.clip(p_te, 0, 1), "prediction": (p_te >= r["threshold"]).astype(int),
                                  "horizon_start": te.horizon_start if "horizon_start" in te else te.window_start,
                                  "horizon_end": te.horizon_end if "horizon_end" in te else te.window_end,
                                  "probability_calibrated": np.clip(iso.predict(p_te), 0, 1)}))
        print(f"Task {task}: refit {key} seed {r['seed']} on {tr.subject_id.nunique()} nights; {len(te)} test predictions written (no metric computed)")
    pd.concat(rows, ignore_index=True).to_csv(ROOT / out_name, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", default=None)
    ap.add_argument("--task", default="AB", help="which tasks to score (default both)")
    ap.add_argument("--note", default="", help="free-text provenance note stored with the results")
    ap.add_argument("--out", default="predictions_round2_repro.csv", help="output file name for --reproduce-only")
    ap.add_argument("--reproduce-only", action="store_true",
                    help="refit the promoted pipelines on the 22 nights and write test PREDICTIONS to predictions_round2_repro.csv; "
                         "no metric is computed and no test label is read (clean-checkout verification)")
    args = ap.parse_args()
    cv = load_json(OUT_DIR / "round2_cv.json")
    decision = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "tasks": {}}
    for task in ("A", "B"):
        key, info = promoted(cv, task)
        decision["tasks"][task] = {"promoted": key, **info}
        print(f"Task {task}: comparator {info['comparator']}; passing {info['passing']}; promoted -> {key}")
    save_json(decision, OUT_DIR / "round2_decision.json")
    if args.reproduce_only:
        reproduce(decision, cv, args.out)
        return
    if args.dry_run or args.confirm is None:
        print("Dry run: no test row read." if args.dry_run else "No --confirm given: no test row read.")
        return

    prev_path = OUT_DIR / "round2_test.json"
    prev = load_json(prev_path) if prev_path.exists() else None
    R = {"confirmation": args.confirm, "note": args.note, "decision": decision,
         "results": dict(prev["results"]) if prev and args.task != "AB" else {},
         "superseded": (prev.get("superseded", []) + [{"scored_at": prev.get("scored_at"), "results": {t: prev["results"][t] for t in args.task if t in prev["results"]}}]) if prev and args.task != "AB" else []}
    rows = []
    if prev and args.task != "AB" and (ROOT / "predictions_round2.csv").exists():
        keep = pd.read_csv(ROOT / "predictions_round2.csv")
        rows.append(keep[~keep.task.str.startswith(tuple(f"{t}_" for t in args.task))])
    detA = None
    for task in [t for t in ("A", "B") if t in args.task]:
        key = decision["tasks"][task]["promoted"]
        if key is None:
            continue
        pname = key.split("|")[1]
        model = "lr" if pname.startswith("lr_") else "xgb"
        params = cv["params"]["lr"][pname] if model == "lr" else (cv["params"][task].get(pname) or PARAMS[task][pname])
        full, feats = table_for(task, key)
        tr = nontest(full)
        te = full[full.split == "test"].reset_index(drop=True)
        # ---- detector-history features for Task B, if the key asks for them
        if task == "B" and "+det" in key:
            keyA = cv["A"]["_best_by_within_alpha1"]
            fullA, featsA = table_for("A", keyA)
            trA = nontest(fullA)
            seedA = SEEDS[0]
            oofA = cv_predict(trA, featsA, PARAMS["A"][keyA.split("|")[1]], seedA)
            detA_tr = trA[["subject_id", "minute", "label"]].assign(p=oofA)
            cands = np.unique(np.quantile(detA_tr.p, np.linspace(0.01, 0.99, 99)))
            thrA = float(cands[int(np.argmax([f1_score(detA_tr.label, detA_tr.p >= t) for t in cands]))])
            pA_te = fit_predict("xgb", trA, {"te": fullA[fullA.split == "test"]}, seed=seedA, feats=featsA, xgb_params=PARAMS["A"][keyA.split("|")[1]])["te"]
            detA_te = fullA[fullA.split == "test"][["subject_id", "minute"]].assign(p=pA_te)
            tr = tr.merge(detector_history_features(tr, detA_tr[["subject_id", "minute", "p"]], thrA), on=["subject_id", "minute"], how="left")
            te = te.merge(detector_history_features(te, detA_te, thrA), on=["subject_id", "minute"], how="left")
            feats = feats + [c for c in tr.columns if c.startswith("det_")]
        # ---- OOF on 22 nights: threshold, median seed, smoothing constant
        oof = {s: cv_predict(tr, feats, params, s, model=model) if not (task == "B" and "+det" in key) else None for s in SEEDS}
        if task == "B" and "+det" in key:        # nested OOF is expensive; reuse non-nested OOF of the base + det features for the threshold only
            oof = {s: cv_predict(tr, feats, params, s, model=model) for s in SEEDS}
        seed_order = sorted(SEEDS, key=lambda s: average_precision_score(tr.label, oof[s]))
        med = seed_order[len(seed_order) // 2]
        alpha = 1.0
        if task == "A":
            alpha = max(ALPHAS, key=lambda a: within_night(tr, smooth(tr, oof[med], a) if a < 1 else oof[med]).mean())
        p_oof = smooth(tr, oof[med], alpha) if alpha < 1 else oof[med]
        cands = np.unique(np.quantile(p_oof, np.linspace(0.01, 0.99, 197)))
        thr = float(cands[int(np.argmax([f1_score(tr.label, p_oof >= t) for t in cands]))])
        # ---- retrain on 22 nights, score test ONCE
        p_te = fit_predict(model, tr, {"te": te}, seed=med, feats=feats, xgb_params=params, save_as=f"{task}_round2_{med}")["te"]
        if alpha < 1:
            p_te = smooth(te, p_te, alpha)
        te = te.assign(p=p_te)
        # isotonic recalibration fitted on the OOF predictions only (amendment 3): same ranking,
        # threshold re-chosen on calibrated OOF, so the effect on every reported metric is measurable
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_oof, tr.label.values)
        p_oof_iso, p_te_iso = iso.predict(p_oof), iso.predict(p_te)
        cands_i = np.unique(np.quantile(p_oof_iso, np.linspace(0.01, 0.99, 197)))
        thr_iso = float(cands_i[int(np.argmax([f1_score(tr.label, p_oof_iso >= t) for t in cands_i]))])
        per_night = []
        for rec, g in te.groupby("subject_id"):
            yhat = g.p >= thr
            per_night.append({"record": rec, "n": int(len(g)), "prevalence": float(g.label.mean()),
                              "auroc": float(roc_auc_score(g.label, g.p)) if g.label.nunique() > 1 else None,
                              "auprc": float(average_precision_score(g.label, g.p)) if g.label.nunique() > 1 else None,
                              "tpr": float(yhat[g.label == 1].mean()) if (g.label == 1).any() else None,
                              "fpr": float(yhat[g.label == 0].mean()) if (g.label == 0).any() else None,
                              "tp": int((yhat & (g.label == 1)).sum()), "fp": int((yhat & (g.label == 0)).sum()), "fn": int((~yhat & (g.label == 1)).sum())})
        res = {"key": key, "params": params, "seed": med, "alpha": alpha, "threshold": thr, "n_features": len(feats),
               "oof": evaluate(tr.label.values, p_oof, thr), "oof_within_night": float(within_night(tr, p_oof).mean()),
               "test": evaluate(te.label.values, p_te, thr), "test_bootstrap": record_bootstrap(te), "test_within_record": within_record(te),
               "test_isotonic": {"threshold": thr_iso, **evaluate(te.label.values, p_te_iso, thr_iso)},
               "per_night": per_night}
        R["results"][task] = res
        te = te.assign(p_iso=np.clip(p_te_iso, 0, 1))
        print(f"Task {task} round 2 ({key}, seed {med}, alpha {alpha}): test AUPRC {res['test']['auprc']:.3f} "
              f"(CI {res['test_bootstrap']['auprc_ci95'][0]:.3f}-{res['test_bootstrap']['auprc_ci95'][1]:.3f}), AUROC {res['test']['auroc']:.3f}, "
              f"within-night {res['test_within_record']['mean_auroc']:.3f}")
        rows.append(pd.DataFrame({"subject_id": te.subject_id, "window_start": te.window_start, "window_end": te.window_end,
                                  "task": "A_detection" if task == "A" else "B_onset_prediction",
                                  "probability": np.clip(p_te, 0, 1), "prediction": (p_te >= thr).astype(int),
                                  "horizon_start": te.horizon_start if "horizon_start" in te else te.window_start,
                                  "horizon_end": te.horizon_end if "horizon_end" in te else te.window_end,
                                  "probability_calibrated": te.p_iso}))
    R["scored_at"] = dt.datetime.now().isoformat(timespec="seconds")
    save_json(R, OUT_DIR / "round2_test.json")
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(ROOT / "predictions_round2.csv", index=False)
    md = ["# Round 2 — single test scoring of the promoted pipelines", "", f"Confirmation on record: {args.confirm}", f"Scored at {dt.datetime.now().isoformat(timespec='seconds')}", "",
          "| task | pipeline | seed | α | threshold | OOF AUPRC | OOF within-night | test prevalence | test AUPRC (CI) | test AUROC (CI) | test within-night (CI) | P | R | F1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task, r in R["results"].items():
        t, b, w = r["test"], r["test_bootstrap"], r["test_within_record"]
        md.append(f"| {task} | {r['key']} | {r['seed']} | {r['alpha']} | {r['threshold']:.3f} | {r['oof']['auprc']:.3f} | {r['oof_within_night']:.3f} | {t['prevalence']:.3f} | "
                  f"{t['auprc']:.3f} ({b['auprc_ci95'][0]:.3f}–{b['auprc_ci95'][1]:.3f}) | {t['auroc']:.3f} ({b['auroc_ci95'][0]:.3f}–{b['auroc_ci95'][1]:.3f}) | "
                  f"{w['mean_auroc']:.3f} ({w['auroc_ci95'][0]:.3f}–{w['auroc_ci95'][1]:.3f}) | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} |")
    for task, r in R["results"].items():
        i = r["test_isotonic"]
        md += ["", f"Task {task} after isotonic recalibration (fit on OOF, threshold {i['threshold']:.3f}): AUPRC {i['auprc']:.3f}, AUROC {i['auroc']:.3f}, "
               f"P {i['precision']:.3f}, R {i['recall']:.3f}, F1 {i['f1']:.3f}, Brier {i['brier']:.3f} (raw {r['test']['brier']:.3f}), ECE {i['ece']:.3f} (raw {r['test']['ece']:.3f}), "
               f"[TN FP / FN TP] = [{i['tn']} {i['fp']} / {i['fn']} {i['tp']}].", "",
               f"Per night, Task {task}:", "", "| record | n | prevalence | AUROC | AUPRC | TPR | FPR | TP | FP | FN |", "|---|---|---|---|---|---|---|---|---|---|"]
        f_ = lambda v: "–" if v is None else f"{v:.3f}"
        md += [f"| {g['record']} | {g['n']} | {g['prevalence']:.2f} | {f_(g['auroc'])} | {f_(g['auprc'])} | {f_(g['tpr'])} | {f_(g['fpr'])} | {g['tp']} | {g['fp']} | {g['fn']} |" for g in r["per_night"]]
    for task, d in decision["tasks"].items():
        if d["promoted"] is None:
            md.append(f"\nTask {task}: no candidate passed the promotion rule (comparator {d['comparator']}); round 1 stands and test was not scored.")
    (OUT_DIR / "round2_test.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
