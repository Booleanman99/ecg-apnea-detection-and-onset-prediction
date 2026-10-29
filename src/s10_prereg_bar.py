"""Evaluation of the pre-registered Task B bar, and Task A selection diagnostics.

Reads only files written by s1, s6 and s7 (cached records, prediction files,
results.json, error_analysis.json). Trains nothing.

1. Task B bar (stamped 11:46 IST 29 Oct 2026 and amended before the respective numbers
   existed; the stamped text is quoted in ANALYSIS §7):
   (a) H = 1 onset model beats prevalence AUPRC by >= 2x (point estimate), with the
       record-bootstrap CI clear of prevalence (literal reading) — the stricter reading,
       CI clear of 2x prevalence, is reported beside it; both for the final seed and for
       every seed ("across seeds").
   (b) clause (i): within-record AUROC > 0.5 with a record-bootstrap CI clear of 0.5 over
       the a/b TEST+VAL records with >= 5 onsets, as stamped; test-only beside it.
   (c) clause (ii): true-positive rate in the worst beat-template-SQI tercile not lower
       than in the best tercile, at the frozen threshold, with group sizes and the number
       of SQI = 0 minutes.
2. Task A diagnostics for the CNN-vs-baseline discussion: per-record prevalence, mean
   predicted probability and within-record AUROC for the CNN and the feature models on
   validation and test (the patient-recognition signature is high between-night
   separation together with poor within-night ranking); R-wave polarity per record; and
   a PAIRED record bootstrap of (XGB-rel − CNN) and (XGB − CNN) for pooled AUPRC and
   within-record AUROC on test.

Run:  python src/s10_prereg_bar.py   (writes outputs/prereg_bar.json / .md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, SEEDS, load_json, save_json  # noqa: E402
from s1_data import REC_CACHE, list_records  # noqa: E402
from s4_baseline import PRED_DIR  # noqa: E402
from s6_evaluate import record_bootstrap  # noqa: E402

MIN_ONSETS = 5


def load_pred(task, model, seed, split):
    return pd.read_csv(PRED_DIR / f"{task}_{model}_s{seed}_{split}.csv").assign(split=split)


def within_auroc_records(df: pd.DataFrame, only_ab: bool = True, min_pos: int = MIN_ONSETS) -> pd.DataFrame:
    rows = []
    for r, g in df.groupby("subject_id"):
        if only_ab and not r[0] in "ab":
            continue
        if (g.label == 1).sum() >= min_pos and (g.label == 0).sum() >= min_pos:
            rows.append({"record": r, "split": g.split.iloc[0], "n": len(g), "onsets": int(g.label.sum()),
                         "auroc": roc_auc_score(g.label, g.p), "mean_p": float(g.p.mean()), "prevalence": float(g.label.mean())})
    return pd.DataFrame(rows)


def boot_mean(values: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    b = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ----------------------------------------------------------------------------- Task B bar
def task_b_bar(res: dict) -> dict:
    f = res["B"]["_final"]
    model, final_seed, thr = f["model"], f["seed"], f["threshold"]
    out = {"model": model, "final_seed": final_seed, "threshold": thr, "per_seed": {}}
    for seed in SEEDS:
        te = load_pred("B", model, seed, "test")
        prev = float(te.label.mean())
        ap = float(average_precision_score(te.label, te.p))
        ci = record_bootstrap(te)["auprc_ci95"]
        out["per_seed"][seed] = {
            "test_prevalence": prev, "test_auprc": ap, "lift": ap / prev, "auprc_ci95": ci,
            "point_lift_ge_2": ap >= 2 * prev,
            "ci_clear_of_prevalence(literal)": ci[0] > prev,
            "ci_clear_of_2x_prevalence(strict)": ci[0] > 2 * prev,
        }
    out["clause_a_literal_all_seeds"] = all(v["point_lift_ge_2"] and v["ci_clear_of_prevalence(literal)"] for v in out["per_seed"].values())
    out["clause_a_strict_all_seeds"] = all(v["point_lift_ge_2"] and v["ci_clear_of_2x_prevalence(strict)"] for v in out["per_seed"].values())

    # clause (i): a/b test+val records with >= 5 onsets, as stamped; test-only beside it
    both = pd.concat([load_pred("B", model, final_seed, "val"), load_pred("B", model, final_seed, "test")], ignore_index=True)
    for name, df in (("stamped_test_plus_val", both), ("test_only", both[both.split == "test"])):
        t = within_auroc_records(df)
        ci = boot_mean(t.auroc.values) if len(t) else [float("nan")] * 2
        out[f"clause_i_{name}"] = {"records": t.to_dict("records"), "n_records": int(len(t)),
                                   "mean_within_record_auroc": float(t.auroc.mean()) if len(t) else None,
                                   "auroc_ci95": ci, "met": bool(len(t) and ci[0] > 0.5)}

    # clause (ii): from the s7 error analysis (frozen threshold, beat-template SQI terciles)
    ea = load_json(OUT_DIR / "error_analysis.json")["B"]
    terc = {g["group"]: g for g in ea["by_quality_tercile(sqi)"]}
    worst, best = terc.get("worst"), terc.get("best")
    out["clause_ii"] = {
        "quality_index": ea["quality_index"], "tercile_sizes": ea["quality_tercile_sizes"],
        "sqi_zero_minutes": ea["flags"]["sqi_undefined_minutes"],
        "auroc_sqi_to_label_on_train": ea["auroc_sqi_to_label_on_train"],
        "worst": worst, "best": best,
        "decidable": bool(worst and best and worst["tpr"] is not None and best["tpr"] is not None),
    }
    out["clause_ii"]["met"] = (out["clause_ii"]["decidable"] and worst["tpr"] >= best["tpr"])
    out["overall_literal"] = bool(out["clause_a_literal_all_seeds"] and out["clause_i_stamped_test_plus_val"]["met"] and out["clause_ii"]["met"])
    out["overall_strict"] = bool(out["clause_a_strict_all_seeds"] and out["clause_i_stamped_test_plus_val"]["met"] and out["clause_ii"]["met"])
    return out


# ----------------------------------------------------------------------------- Task A diagnostics
def polarity() -> dict:
    out = {}
    for rec in list_records():
        z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
        ecg, rp = z["ecg"], z["rpeaks"]
        amp = ecg[rp] - np.median(ecg)
        out[rec] = {"frac_positive_r": float((amp > 0).mean()), "median_r_amplitude_mV": float(np.median(amp)),
                    "ecg_std_mV": float(ecg.std())}
    return out


def per_record_table(task: str, runs: dict[str, tuple[str, int]]) -> list[dict]:
    rows = {}
    for label, (model, seed) in runs.items():
        for split in ("val", "test"):
            df = load_pred(task, model, seed, split)
            for r, g in df.groupby("subject_id"):
                row = rows.setdefault(r, {"record": r, "split": split, "n": len(g), "prevalence": float(g.label.mean())})
                row[f"{label}_mean_p"] = float(g.p.mean())
                ok = (g.label == 1).sum() >= MIN_ONSETS and (g.label == 0).sum() >= MIN_ONSETS
                row[f"{label}_within_auroc"] = float(roc_auc_score(g.label, g.p)) if ok else None
    return sorted(rows.values(), key=lambda d: (d["split"], d["record"]))


def between_vs_within(table: list[dict], label: str) -> dict:
    t = pd.DataFrame(table)
    out = {}
    for split in ("val", "test"):
        s = t[t.split == split]
        rho = spearmanr(s[f"{label}_mean_p"], s.prevalence).correlation
        w = s[f"{label}_within_auroc"].dropna()
        out[split] = {"spearman_mean_p_vs_prevalence_across_records": float(rho),
                      "mean_within_record_auroc": float(w.mean()) if len(w) else None, "n_within": int(len(w))}
    return out


def paired_bootstrap(task: str, a: tuple[str, int], b: tuple[str, int], n: int = 1000, seed: int = 0) -> dict:
    """Record-level paired bootstrap of model a minus model b on the test split."""
    A, B = load_pred(task, *a, "test"), load_pred(task, *b, "test")
    m = A.merge(B[["subject_id", "minute", "p"]], on=["subject_id", "minute"], suffixes=("_a", "_b"))
    recs = m.subject_id.unique()
    groups = {r: m[m.subject_id == r] for r in recs}
    rng = np.random.default_rng(seed)
    d_ap, d_wr = [], []
    for _ in range(n):
        pick = rng.choice(recs, len(recs), replace=True)
        s = pd.concat([groups[r] for r in pick])
        if s.label.nunique() < 2:
            continue
        d_ap.append(average_precision_score(s.label, s.p_a) - average_precision_score(s.label, s.p_b))
        wa, wb = [], []
        for r in pick:
            g = groups[r]
            if (g.label == 1).sum() >= MIN_ONSETS and (g.label == 0).sum() >= MIN_ONSETS:
                wa.append(roc_auc_score(g.label, g.p_a)); wb.append(roc_auc_score(g.label, g.p_b))
        if wa:
            d_wr.append(np.mean(wa) - np.mean(wb))
    point_ap = average_precision_score(m.label, m.p_a) - average_precision_score(m.label, m.p_b)
    return {"pooled_auprc_diff": float(point_ap), "pooled_auprc_diff_ci95": [float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5))],
            "within_record_auroc_diff_ci95": [float(np.percentile(d_wr, 2.5)), float(np.percentile(d_wr, 97.5))],
            "within_record_auroc_diff_mean": float(np.mean(d_wr))}


def main() -> None:
    res = load_json(OUT_DIR / "results.json")
    out = {"task_b_bar": task_b_bar(res)}

    fA = res["A"]["_final"]
    runs = {"cnn": (fA["model"], fA["seed"]) if fA["model"] == "cnn" else ("cnn", 0), "xgb_rel": ("xgb_rel", 0), "xgb": ("xgb", 0)}
    table = per_record_table("A", runs)
    pol = polarity()
    for row in table:
        row.update({"frac_positive_r": pol[row["record"]]["frac_positive_r"], "ecg_std_mV": pol[row["record"]]["ecg_std_mV"]})
    out["task_a"] = {
        "runs": runs, "per_record": table,
        "between_vs_within": {k: between_vs_within(table, k) for k in runs},
        "polarity_all_records": pol,
        "paired_bootstrap_test": {"xgb_rel_minus_cnn": paired_bootstrap("A", runs["xgb_rel"], runs["cnn"]),
                                  "xgb_minus_cnn": paired_bootstrap("A", runs["xgb"], runs["cnn"])},
    }
    save_json(out, OUT_DIR / "prereg_bar.json")

    b = out["task_b_bar"]
    md = ["# Pre-registered Task B bar — evaluation", "",
          f"Final model {b['model']} seed {b['final_seed']}, frozen threshold {b['threshold']:.3f}.", "",
          "## (a) AUPRC vs prevalence, per seed (record-bootstrap 95% CI on test)", "",
          "| seed | prevalence | AUPRC | lift | CI | point lift >= 2x | CI clear of prevalence (literal) | CI clear of 2x prevalence (strict) |",
          "|---|---|---|---|---|---|---|---|"]
    for s, v in b["per_seed"].items():
        md.append(f"| {s} | {v['test_prevalence']:.3f} | {v['test_auprc']:.3f} | {v['lift']:.1f}x | {v['auprc_ci95'][0]:.3f}–{v['auprc_ci95'][1]:.3f} | "
                  f"{v['point_lift_ge_2']} | {v['ci_clear_of_prevalence(literal)']} | {v['ci_clear_of_2x_prevalence(strict)']} |")
    md += ["", f"All seeds, literal reading: {b['clause_a_literal_all_seeds']}; strict reading: {b['clause_a_strict_all_seeds']}.", ""]
    for name in ("stamped_test_plus_val", "test_only"):
        c = b[f"clause_i_{name}"]
        md += [f"## (i) within-record AUROC, {name.replace('_', ' ')} ({c['n_records']} a/b records with >= {MIN_ONSETS} onsets)", "",
               "| record | split | n | onsets | AUROC |", "|---|---|---|---|---|"]
        md += [f"| {r['record']} | {r['split']} | {r['n']} | {r['onsets']} | {r['auroc']:.3f} |" for r in c["records"]]
        md += ["", f"Mean {c['mean_within_record_auroc']:.3f}, CI {c['auroc_ci95'][0]:.3f}–{c['auroc_ci95'][1]:.3f} -> clause (i) {'MET' if c['met'] else 'NOT MET'}.", ""]
    c = b["clause_ii"]
    md += ["## (ii) TPR in worst vs best SQI tercile at the frozen threshold", "",
           f"Quality index: {c['quality_index']}; tercile sizes {c['tercile_sizes']}; SQI = 0 minutes: {c['sqi_zero_minutes']}; "
           f"AUROC(SQI -> label) on train {c['auroc_sqi_to_label_on_train']:.3f}.", ""]
    if c["decidable"]:
        md.append(f"worst: n={c['worst']['n']}, positives={c['worst']['fn'] + int(round(c['worst']['tpr'] * (c['worst']['fn'] / (1 - c['worst']['tpr']) if c['worst']['tpr'] < 1 else 0)))}, TPR {c['worst']['tpr']:.3f}; "
                  f"best: n={c['best']['n']}, TPR {c['best']['tpr']:.3f} -> clause (ii) {'MET' if c['met'] else 'NOT MET'}.")
    else:
        md.append("Not decidable: a tercile has no positives at the frozen threshold (see error_analysis.md).")
    md += ["", f"**Overall (literal reading): {'MET' if b['overall_literal'] else 'NOT MET'}. Overall (strict reading): {'MET' if b['overall_strict'] else 'NOT MET'}.**", "",
           "# Task A diagnostics", "", "## Per record: prevalence, mean predicted probability and within-record AUROC", "",
           "| record | split | n | prev | R+ frac | CNN mean p | CNN within AUROC | XGB-rel mean p | XGB-rel within | XGB mean p | XGB within |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{v:.3f}"
    for r in out["task_a"]["per_record"]:
        md.append(f"| {r['record']} | {r['split']} | {r['n']} | {r['prevalence']:.2f} | {r['frac_positive_r']:.2f} | {f(r['cnn_mean_p'])} | {f(r['cnn_within_auroc'])} | "
                  f"{f(r['xgb_rel_mean_p'])} | {f(r['xgb_rel_within_auroc'])} | {f(r['xgb_mean_p'])} | {f(r['xgb_within_auroc'])} |")
    md += ["", "## Between-night vs within-night", "", "| model | split | Spearman(mean p, prevalence) across records | mean within-record AUROC | n |", "|---|---|---|---|---|"]
    for k, v in out["task_a"]["between_vs_within"].items():
        for split, s in v.items():
            md.append(f"| {k} | {split} | {s['spearman_mean_p_vs_prevalence_across_records']:.3f} | {f(s['mean_within_record_auroc'])} | {s['n_within']} |")
    md += ["", "## Paired record bootstrap on test (model − CNN)", "", "| comparison | pooled AUPRC diff | CI | within-record AUROC diff | CI |", "|---|---|---|---|---|"]
    for k, v in out["task_a"]["paired_bootstrap_test"].items():
        md.append(f"| {k} | {v['pooled_auprc_diff']:+.3f} | {v['pooled_auprc_diff_ci95'][0]:+.3f}–{v['pooled_auprc_diff_ci95'][1]:+.3f} | "
                  f"{v['within_record_auroc_diff_mean']:+.3f} | {v['within_record_auroc_diff_ci95'][0]:+.3f}–{v['within_record_auroc_diff_ci95'][1]:+.3f} |")
    (OUT_DIR / "prereg_bar.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
