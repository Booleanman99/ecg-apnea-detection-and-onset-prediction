"""Round 2, amendments 2 and 3: configurations added after s14 was launched, run on the SAME
folds and seeds and merged into outputs/round2_cv.json; comparators and paired differences
are then recomputed as "best round-1 family under CV". Reads no test label.

 Task B (amendment 2): round-1 window-only XGBoost at the shipped and the validation-tuned
   setting (3 / 0.03 / 200 / 20).
 Both tasks (amendment 3): logistic regression on round-1 window-only features, C = 0.5
   (shipped) and C = 0.05 (top of the validation grid). LR is deterministic; seeds coincide.

Run:  python src/s14b_round2_cv_extra.py   (after s14_round2_cv.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, load_json, save_json  # noqa: E402
from s3_features import window_only_columns  # noqa: E402
from s4_baseline import load_task  # noqa: E402
from s14_round2_cv import PARAMS, evaluate_candidate, nontest, paired_vs  # noqa: E402

TUNED_WINDOW_B = dict(max_depth=3, learning_rate=0.03, n_estimators=200, min_child_weight=20)
LR_SETTINGS = {"lr_C0.5": {"C": 0.5}, "lr_C0.05": {"C": 0.05}}


ROUND1_SETTINGS = ("shipped", "lr_C0.5")      # the round-1 configurations; tuned settings are round-2 candidates


def recompute(R: dict, task: str, r1_prefix: str) -> str:
    """Comparator = best ROUND-1 configuration (round-1 features at the round-1 setting) under CV."""
    r1 = [k for k in R[task] if k.startswith(r1_prefix) and "+det" not in k and not k.startswith("_")
          and k.split("|")[1] in ROUND1_SETTINGS]
    comp = max(r1, key=lambda k: R[task][k]["alpha_1.0"]["within"])
    R[task]["_comparator"] = comp
    R[task]["_paired"] = {k: paired_vs(R[task], k, comp) for k in R[task] if not k.startswith("_") and k != comp}
    return comp


def main():
    R = load_json(OUT_DIR / "round2_cv.json")
    if "--recompute-only" in sys.argv:          # comparators and paired differences only; no fits
        compA, compB = recompute(R, "A", "r1_"), recompute(R, "B", "W5_r1_")
        save_json(R, OUT_DIR / "round2_cv.json")
        print(f"comparators: A {compA}, B {compB}")
        return
    # Task B: window-only XGBoost (amendment 2) and LR (amendment 3)
    B5 = nontest(load_task("B"))
    featsB = window_only_columns(B5)
    for pname, params in (("shipped", PARAMS["B"]["shipped"]), ("tuned_window", TUNED_WINDOW_B)):
        key = f"W5_r1_window|{pname}"
        R["B"][key] = evaluate_candidate(B5, featsB, params, key)["results"]
    for pname, params in LR_SETTINGS.items():
        key = f"W5_r1_window|{pname}"
        R["B"][key] = evaluate_candidate(B5, featsB, params, key, model="lr")["results"]
    # Task A: LR (amendment 3)
    A1 = nontest(load_task("A"))
    featsA = window_only_columns(A1)
    for pname, params in LR_SETTINGS.items():
        key = f"r1_window|{pname}"
        R["A"][key] = evaluate_candidate(A1, featsA, params, key, alphas=(1.0, 0.7, 0.5, 0.3), model="lr")["results"]
    R["params"]["B"]["tuned_window"] = TUNED_WINDOW_B
    R["params"]["lr"] = LR_SETTINGS
    compA = recompute(R, "A", "r1_")
    compB = recompute(R, "B", "W5_r1_")
    save_json(R, OUT_DIR / "round2_cv.json")
    for task, comp in (("A", compA), ("B", compB)):
        print(f"Task {task} comparator now {comp}; paired: " + ", ".join(f"{k}: {v['mean_diff']:+.3f} (2SE {2 * v['se']:.3f})" for k, v in R[task]["_paired"].items()))


if __name__ == "__main__":
    main()
