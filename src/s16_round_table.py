"""One table per task: every round-1 model and every round-2 candidate, side by side, for
review before any round-2 test scoring. Reads round-1 results (validation and the already
published test numbers) and the round-2 CV results. Runs nothing on test.

Columns: model / features / config | round | val AUPRC (round 1, 6 nights) | CV pooled OOF
AUPRC ± seed sd | CV within-night AUROC ± SE over folds | paired diff vs comparator (mean ± SE)
| passes rule | round-1 test AUPRC / AUROC / within-night (round-1 models only; round-2 rows
say "no test number yet"). Exactly one row per task carries "SELECTED BY RULE".

Run:  python src/s16_round_table.py   (writes outputs/round_table.md)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, load_json  # noqa: E402

R1_LABEL = {"lr": "Logistic regression / window feats / C=0.5", "xgb": "XGBoost / window feats / shipped",
            "cnn": "CNN+GRU / raw ECG / shipped", "xgb_rel": "XGBoost / window + night-so-far feats / shipped"}
# which CV candidate key corresponds to each round-1 family (for the CV columns of round-1 rows)
R1_TO_CV = {"A": {"lr": "r1_window|lr_C0.5", "xgb": "r1_window|shipped", "xgb_rel": "r1_rel|shipped"},
            "B": {"lr": "W5_r1_window|lr_C0.5", "xgb": "W5_r1_window|shipped", "xgb_rel": "W5_r1_rel|shipped"}}


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def main():
    r1 = load_json(OUT_DIR / "results.json")
    cv = load_json(OUT_DIR / "round2_cv.json")
    dec = load_json(OUT_DIR / "round2_decision.json") if (OUT_DIR / "round2_decision.json").exists() else None
    md = ["# Round 1 and round 2 in one table (for review; no round-2 test number exists)", "",
          "Round-1 columns: 6-night validation AUPRC and the published 8-night test numbers. Round-2 columns: 6-fold grouped CV on the 22 non-test nights "
          "(3 seeds), paired per-fold within-night difference against the best round-1 family under the same CV; the rule (PREREGISTRATION.md, Round 2 clause 3) is "
          "mean paired difference > 2 × SE and pooled OOF AUPRC not lower. Smoothing constant shown for Task A at α = 1 (no smoothing); other α in outputs/round2_cv.md.", ""]
    for task in ("A", "B"):
        comp = cv[task]["_comparator"]
        paired = cv[task]["_paired"]
        passing = [k for k, v in paired.items() if v["passes_2se"] and v["auprc_not_lower"]]
        d = dec["tasks"][task] if dec else None
        selected = d["promoted"] if d else (max(passing, key=lambda k: cv[task][k]["alpha_1.0"]["within"]) if passing else None)
        route = d.get("route", "") if d else ""
        r1_final = r1[task]["_final"]["model"]
        if d and "3b" in route:
            v = d["vs_round1_final"]
            pdf = v.get("paired_diff_final_vs_comp") or {}
            header = (f"No round-2 candidate passes clause 3a. Clause 3b promotes `{selected}` (round-1 family ranked above the round-1 final `{v['round1_final']}` under CV: "
                      f"within-night {v['within'][0]:.3f} vs {v['within'][1]:.3f}, OOF AUPRC {v['oof_auprc'][0]:.3f} vs {v['oof_auprc'][1]:.3f}; per-fold paired difference "
                      f"{-pdf.get('mean_diff', float('nan')):+.3f} ± {pdf.get('se', float('nan')):.3f}; no noise margin in the stamped text).")
        elif selected:
            header = f"Rule-selected candidate: `{selected}` via {route or 'clause 3a'}."
        else:
            header = "No candidate passes either route; round 1 stands."
        md += [f"## Task {task}", "", f"Comparator under CV (best round-1 configuration): `{comp}`. {header} Round-1 final: `{r1_final}`.", "",
               "| model / features / config | round | val AUPRC (6 nights) | CV OOF AUPRC ± seed sd | CV within-night AUROC ± SE (folds) | paired Δ within vs comparator (mean ± SE) | passes rule | test AUPRC / AUROC / within-night | note |",
               "|---|---|---|---|---|---|---|---|---|"]
        summary = r1[task]["_summary"]
        for fam in ("lr", "xgb", "cnn", "xgb_rel"):
            if fam not in summary:
                continue
            s = summary[fam]
            cvkey = R1_TO_CV[task].get(fam)
            c = cv[task].get(cvkey, {}).get("alpha_1.0") if cvkey else None
            pd_ = paired.get(cvkey) if cvkey else None
            se_f = (c["within_fold_sd"] / np.sqrt(6)) if c else None
            note = "round-1 FINAL" if fam == r1_final else ""
            if cvkey == comp:
                note = (note + "; " if note else "") + "CV comparator"
            if cvkey and cvkey == selected:
                note = "**PROMOTED (3b)** / " + note
            md.append(f"| {R1_LABEL[fam]} | 1 | {s['val_auprc_mean']:.3f} | "
                      f"{fmt(c['auprc']) + ' ± ' + fmt(c['auprc_seed_sd']) if c else 'not in CV (deep model: validation screen only)' if fam == 'cnn' else 'not in CV'} | "
                      f"{fmt(c['within']) + ' ± ' + fmt(se_f) if c else '–'} | "
                      f"{(fmt(pd_['mean_diff']) + ' ± ' + fmt(pd_['se'])) if pd_ else ('comparator' if cvkey == comp else '–')} | "
                      f"{('yes' if (pd_ and pd_['passes_2se'] and pd_['auprc_not_lower']) else 'no') if pd_ else '–'} | "
                      f"{s['test_auprc_mean']:.3f} / {s['test_auroc_mean']:.3f} / {s['test_within_record_auroc_mean']:.3f} | {note} |")
        for key, res in cv[task].items():
            if key.startswith("_") or key in R1_TO_CV[task].values():
                continue
            c = res["alpha_1.0"]
            pd_ = paired.get(key)
            se_f = c["within_fold_sd"] / np.sqrt(6)
            note = "**SELECTED BY RULE**" if key == selected else ""
            md.append(f"| {key} | 2 | – | {c['auprc']:.3f} ± {c['auprc_seed_sd']:.3f} | {c['within']:.3f} ± {se_f:.3f} | "
                      f"{(fmt(pd_['mean_diff']) + ' ± ' + fmt(pd_['se'])) if pd_ else 'comparator'} | "
                      f"{('yes' if (pd_ and pd_['passes_2se'] and pd_['auprc_not_lower']) else 'no') if pd_ else '–'} | no test number yet | {note} |")
        if task == "A":
            best_a = {}
            for key, res in cv["A"].items():
                if key.startswith("_"):
                    continue
                ba = max(res.items(), key=lambda kv: kv[1]["within"])
                best_a[key] = (ba[0], ba[1]["within"])
            md += ["", "Best smoothing constant per Task A candidate (by CV within-night AUROC): " + "; ".join(f"{k}: {a} → {w:.3f}" for k, (a, w) in best_a.items())]
        md.append("")
    md += ["Deep-model round-2 screen (validation only, not CV; never a test candidate): see outputs/val_arch_screen.md when complete."]
    (OUT_DIR / "round_table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
