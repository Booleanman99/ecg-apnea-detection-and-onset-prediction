"""§7 Error analysis and calibration for the final model of each task.

Error analysis links false positives / negatives to: individual records, signal quality
(flat-line fraction, implausible beat counts, R-peak miss rate vs the supplied .qrs),
event boundaries (minutes whose label differs from a neighbour), and, for Task B, how
long after the previous event an onset occurs and where in the night it sits.

Calibration: reliability diagram, Brier score and ECE on test for the raw probabilities,
then after isotonic regression fitted on VALIDATION only (bonus item).

Run:  python src/s7_errors_calib.py   (writes outputs/error_analysis.json / .md, figures)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wfdb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, DATA_DIR, FIG_DIR, FS, MIN_SAMPLES, OUT_DIR, ROOT, evaluate, load_json, save_json  # noqa: E402
from s1_data import REC_CACHE  # noqa: E402
from s4_baseline import PRED_DIR  # noqa: E402


def attach_context(task: str, df: pd.DataFrame) -> pd.DataFrame:
    """Join quality features, neighbour labels and R-peak miss rate onto the predictions."""
    feats = pd.read_parquet(CACHE_DIR / f"features_{task}.parquet")
    cols = ["subject_id", "minute", "flat_frac", "n_beats", "clip_frac", "ecg_std", "sqi_last", "sqi_mean"]
    df = df.merge(feats[cols], on=["subject_id", "minute"], how="left")
    extra = []
    for rec, g in df.groupby("subject_id"):
        z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
        lab, rp = z["labels"], z["rpeaks"]
        try:
            ref = wfdb.rdann(str(DATA_DIR / rec), "qrs").sample
        except FileNotFoundError:
            ref = np.array([], dtype=int)
        # last minute of the input window (the minute just before t for Task B, the window for A)
        last = (g.window_end // 60 - 1).astype(int).values
        prev = np.where(last - 1 >= 0, lab[np.clip(last - 1, 0, None)], -1)
        nxt = np.where(last + 1 < len(lab), lab[np.clip(last + 1, 0, len(lab) - 1)], -1)
        boundary = (lab[last] != prev) | (lab[last] != nxt)
        # minutes since the previous apnea minute (Task B: how fresh is the last event)
        since = []
        for k in last:
            prior = np.flatnonzero(lab[:k + 1] == 1)
            since.append(k - prior[-1] if len(prior) else 10_000)
        # R-peak sensitivity vs .qrs inside the last minute
        miss = []
        for k in last:
            s, e = k * MIN_SAMPLES, (k + 1) * MIN_SAMPLES
            r_ref = ref[(ref >= s) & (ref < e)]
            r_det = rp[(rp >= s - FS) & (rp < e + FS)]
            if len(r_ref) == 0 or len(r_det) == 0:
                miss.append(np.nan)
                continue
            idx = np.clip(np.searchsorted(r_det, r_ref), 1, len(r_det) - 1)
            d = np.minimum(np.abs(r_det[idx] - r_ref), np.abs(r_det[idx - 1] - r_ref))
            miss.append(float(np.mean(d > 0.05 * FS)))
        extra.append(pd.DataFrame({"idx": g.index, "boundary": boundary, "minutes_since_apnea": since,
                                   "rpeak_miss_rate": miss, "night_frac": last / len(lab)}))
    ex = pd.concat(extra).set_index("idx").sort_index()
    return pd.concat([df, ex], axis=1)


def by_group(df: pd.DataFrame, col: str, thr: float) -> list[dict]:
    rows = []
    for name, g in df.groupby(col, observed=True):
        yhat = g.p >= thr
        rows.append({"group": str(name), "n": len(g), "prevalence": float(g.label.mean()),
                     "auprc": float(average_precision_score(g.label, g.p)) if g.label.nunique() > 1 else None,
                     "auroc": float(roc_auc_score(g.label, g.p)) if g.label.nunique() > 1 else None,
                     "tpr": float(yhat[g.label == 1].mean()) if (g.label == 1).any() else None,
                     "fpr": float(yhat[g.label == 0].mean()) if (g.label == 0).any() else None,
                     "fp": int((yhat & (g.label == 0)).sum()), "fn": int((~yhat & (g.label == 1)).sum())})
    return rows


def reliability(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()),
             "mean_p": float(p[idx == b].mean()) if (idx == b).any() else None,
             "frac_pos": float(y[idx == b].mean()) if (idx == b).any() else None} for b in range(n_bins)]


def load_round2(task: str):
    """Round-2 final: predictions_round2.csv (test) with labels re-attached from the window
    tables; the calibrator is refitted on the 22-night OOF predictions is not stored, so the
    'isotonic' column of predictions_round2.csv (fit on OOF) is used as the calibrated score."""
    r2 = load_json(OUT_DIR / "round2_test.json")["results"][task]
    src = ROOT / "predictions_round2.csv"
    p = pd.read_csv(src if src.exists() else ROOT / "predictions.csv")   # fresh run_all writes round 2 straight to predictions.csv
    p = p[p.task.str.startswith(task + "_")].copy()
    win = pd.read_csv(CACHE_DIR / f"windows_{task}.csv")
    te = p.merge(win[["subject_id", "window_start", "minute", "label"]], on=["subject_id", "window_start"], how="left")
    te = te.rename(columns={"probability": "p"})
    f = {"model": r2["key"], "seed": r2["seed"], "threshold": r2["threshold"]}
    return f, te, te.probability_calibrated.values


def main() -> None:
    round2 = "--round2" in sys.argv
    res = load_json(OUT_DIR / "results.json")
    report, md = {}, [f"# Error analysis and calibration (test split, {'round-2' if round2 else 'round-1'} final model per task)", ""]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, task in zip(axes, ("A", "B")):
        if round2:
            f, te, p_iso_pre = load_round2(task)
            thr = f["threshold"]
            va = None
        else:
            f = res[task]["_final"]
            thr = f["threshold"]
            te = pd.read_csv(PRED_DIR / f"{task}_{f['model']}_s{f['seed']}_test.csv")
            va = pd.read_csv(PRED_DIR / f"{task}_{f['model']}_s{f['seed']}_val.csv")
        te = attach_context(task, te)
        # Quality axis = beat-template SQI of the last input minute (pre-registered amendment 2):
        # continuous, not label-driven. Flat / clipped / implausible-beat minutes stay binary flags.
        te["quality_tercile"] = pd.qcut(te.sqi_last, 3, labels=["worst", "mid", "best"], duplicates="drop")
        n_min = int(((te.window_end - te.window_start) // 60).iloc[0])        # 1 for Task A, 5 for Task B
        te["beats_plausible"] = te.n_beats.between(20 * n_min, 220 * n_min)   # 20-220 bpm over the input window
        te["flat_or_clipped"] = (te.flat_frac > 0) | (te.clip_frac > 0)
        te["no_running_baseline"] = te.sqi_mean.isna() | te.minute.lt(5 + (5 if task == "B" else 0))   # first 5 windows of a night: _rel imputed
        tr_feats = pd.read_parquet(CACHE_DIR / f"features_{task}.parquet")
        tr_feats = tr_feats[tr_feats.split == "train"]
        sqi_label_auroc = float(roc_auc_score(tr_feats.label, tr_feats.sqi_last))
        te["since_bin"] = pd.cut(te.minutes_since_apnea, [-1, 1, 3, 10, 30, 1e9],
                                 labels=["<=1", "2-3", "4-10", "11-30", ">30 / never"])
        te["night_third"] = pd.cut(te.night_frac, [0, 1 / 3, 2 / 3, 1.0001], labels=["first", "middle", "last"])
        r = {
            "model": f["model"], "seed": f["seed"], "threshold": thr,
            "per_record": by_group(te, "subject_id", thr),
            "quality_index": "beat-template SQI of last input minute",
            "quality_tercile_sizes": te.quality_tercile.value_counts().to_dict(),
            "auroc_sqi_to_label_on_train": sqi_label_auroc,
            "flags": {"flat_or_clipped_minutes": int(te.flat_or_clipped.sum()), "implausible_beat_minutes": int((~te.beats_plausible).sum()),
                      "sqi_undefined_minutes": int((te.sqi_last == 0).sum()),
                      "no_running_baseline_windows": int(te.no_running_baseline.sum())},
            "by_quality_tercile(sqi)": by_group(te, "quality_tercile", thr),
            "by_beats_plausible": by_group(te, "beats_plausible", thr),
            "by_flat_or_clipped": by_group(te, "flat_or_clipped", thr),
            "by_no_running_baseline": by_group(te, "no_running_baseline", thr),
            "by_boundary_minute": by_group(te, "boundary", thr),
            # Task B: every onset is at a boundary by definition, so this is only meaningful for Task A
            "metrics_excluding_boundary": (evaluate(te.label[~te.boundary].values, te.p[~te.boundary].values, thr)
                                           if te.label[~te.boundary].nunique() > 1 else None),
            "by_minutes_since_last_apnea": by_group(te, "since_bin", thr),
            "by_night_third": by_group(te, "night_third", thr),
            "rpeak_miss_rate_by_quality": te.groupby("quality_tercile", observed=True).rpeak_miss_rate.mean().to_dict(),
            "rpeak_miss_rate_fp_vs_tn": {
                "fp": float(te.rpeak_miss_rate[(te.p >= thr) & (te.label == 0)].mean()),
                "tn": float(te.rpeak_miss_rate[(te.p < thr) & (te.label == 0)].mean())},
        }
        # calibration: raw, then isotonic fitted on validation only (round 1) or on the
        # 22-night out-of-fold predictions (round 2, column carried in predictions_round2.csv)
        if round2:
            p_iso = p_iso_pre
        else:
            iso = IsotonicRegression(out_of_bounds="clip").fit(va.p, va.label)
            p_iso = iso.predict(te.p)
        raw, cal = evaluate(te.label.values, te.p.values, thr), evaluate(te.label.values, p_iso, 0.5)
        r["calibration"] = {"raw": {"brier": raw["brier"], "ece": raw["ece"], "reliability": reliability(te.label.values, te.p.values)},
                            "isotonic_on_val": {"brier": cal["brier"], "ece": cal["ece"], "auprc": cal["auprc"],
                                                "reliability": reliability(te.label.values, p_iso)}}
        report[task] = r
        # predictions.csv keeps the pre-registered raw probability and threshold; the
        # validation-fitted isotonic probability is added as an extra column for readers
        # who want calibrated numbers (same ranking, so AUPRC/AUROC are unchanged).
        # round-1 calibrated column goes to predictions_round1.csv when round 2 is the shipped predictions.csv;
        # the round-2 file already carries its calibrated column (s15)
        r1_path = ROOT / "predictions_round1.csv"
        pred_path = None if round2 else (r1_path if r1_path.exists() else ROOT / "predictions.csv")
        if pred_path is not None:
          pred = pd.read_csv(pred_path)
          key = "A_detection" if task == "A" else "B_onset_prediction"
          cal_df = te[["subject_id", "window_start"]].assign(probability_calibrated=np.clip(p_iso, 0, 1))
          sel = pred.task == key
          merged = pred.loc[sel, ["subject_id", "window_start"]].merge(cal_df, on=["subject_id", "window_start"], how="left")
          if not round2:
            assert len(merged) == sel.sum() and merged.probability_calibrated.notna().all()
            if "probability_calibrated" not in pred:
                pred["probability_calibrated"] = np.nan
            pred.loc[sel, "probability_calibrated"] = merged.probability_calibrated.values
            pred.to_csv(pred_path, index=False)
        for name, p, mk in (("raw", te.p.values, "o-"), ("isotonic (fit on val)", p_iso, "s--")):
            rel = [b for b in reliability(te.label.values, p) if b["n"] > 0]
            ax.plot([b["mean_p"] for b in rel], [b["frac_pos"] for b in rel], mk, label=name)
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(title=f"Task {task} reliability ({f['model']})", xlabel="mean predicted probability", ylabel="fraction positive")
        ax.legend()

        md += [f"## Task {task} — {f['model']} seed {f['seed']}, threshold {thr:.3f}", "",
               "### Per record", "", "| record | n | prevalence | AUPRC | AUROC | TPR | FPR | FP | FN |", "|---|---|---|---|---|---|---|---|---|"]
        for g in r["per_record"]:
            fmt = lambda v: "–" if v is None else f"{v:.3f}"
            md.append(f"| {g['group']} | {g['n']} | {g['prevalence']:.2f} | {fmt(g['auprc'])} | {fmt(g['auroc'])} | {fmt(g['tpr'])} | {fmt(g['fpr'])} | {g['fp']} | {g['fn']} |")
        md += [f"Quality axis: beat-template SQI of the last input minute; tercile sizes {r['quality_tercile_sizes']}; "
               f"AUROC(SQI -> label) on train = {sqi_label_auroc:.3f} (0.5 = quality unrelated to the label). "
               f"Flags: {r['flags']['flat_or_clipped_minutes']} flat/clipped minutes, {r['flags']['implausible_beat_minutes']} implausible-beat minutes, "
               f"{r['flags']['sqi_undefined_minutes']} minutes with < 3 beats (SQI set to 0, so they sit in the worst tercile).", ""]
        for key in ("by_quality_tercile(sqi)", "by_beats_plausible", "by_flat_or_clipped", "by_no_running_baseline", "by_boundary_minute",
                    "by_minutes_since_last_apnea", "by_night_third"):
            md += ["", f"### {key}", "", "| group | n | prevalence | AUROC | TPR | FPR | FP | FN |", "|---|---|---|---|---|---|---|---|"]
            for g in r[key]:
                fmt = lambda v: "–" if v is None else f"{v:.3f}"
                md.append(f"| {g['group']} | {g['n']} | {g['prevalence']:.2f} | {fmt(g['auroc'])} | {fmt(g['tpr'])} | {fmt(g['fpr'])} | {g['fp']} | {g['fn']} |")
        eb = r["metrics_excluding_boundary"]
        md += ["", (f"Excluding boundary minutes: AUPRC {eb['auprc']:.3f}, AUROC {eb['auroc']:.3f}, F1 {eb['f1']:.3f} (n={eb['n']})." if eb
                    else "Excluding boundary minutes: not defined for Task B (every onset is a boundary minute by construction)."),
               f"R-peak miss rate vs .qrs by quality tercile: " + ", ".join(f"{k}: {v:.3f}" for k, v in r["rpeak_miss_rate_by_quality"].items()),
               f"R-peak miss rate on FP minutes {r['rpeak_miss_rate_fp_vs_tn']['fp']:.3f} vs TN minutes {r['rpeak_miss_rate_fp_vs_tn']['tn']:.3f}.",
               "", f"Calibration: raw Brier {raw['brier']:.3f}, ECE {raw['ece']:.3f}; after isotonic (fit on {'OOF' if round2 else 'val'}) Brier {cal['brier']:.3f}, ECE {cal['ece']:.3f}.", ""]
    fig.tight_layout()
    sfx = "_round2" if round2 else ""
    fig.savefig(FIG_DIR / f"reliability{sfx}.png", dpi=130)
    save_json(report, OUT_DIR / f"error_analysis{sfx}.json")
    (OUT_DIR / f"error_analysis{sfx}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
