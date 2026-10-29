"""Shared configuration, seeding and metrics.

Metrics are implemented once here and reused by every section (assignment §1, §6).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ECG_DATA_DIR", ROOT / "data" / "ECG_P1"))
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
for d in (CACHE_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

FS = 100                       # Hz, verified from every .hea in s1_data.audit()
MIN_SAMPLES = 60 * FS          # 6000 samples per annotated minute
HISTORY_MIN = 5                # Task B: minutes of ECG history before t
HORIZON_MIN = 1                # Task B: predict onset within (t, t+H]
SEEDS = (0, 1, 2)

# Subject-level split (assignment §2). Records were assigned by hand so that every split
# contains severe ("a"), borderline ("b") and control ("c") recordings, i.e. the full
# prevalence range. s4/s5 never load the test rows; s6_evaluate.py is the only stage that
# scores TEST, after selecting the final model per task on validation.
SPLIT = {
    "train": ["a01", "a02", "a03", "a05", "a06", "a07", "a08", "a09", "a11", "a12",
              "b01", "b02", "c01", "c02", "c03", "c04"],
    "val":   ["a13", "a14", "a15", "b04", "c06", "c07"],
    "test":  ["a16", "a18", "a19", "a20", "b05", "c08", "c09", "c10"],
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def split_of(record: str) -> str:
    for name, recs in SPLIT.items():
        if record in recs:
            return name
    raise KeyError(record)


# ----------------------------------------------------------------------------- metrics
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold maximising F1 on the VALIDATION set (never on test)."""
    cands = np.unique(np.quantile(p_val, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y_val, p_val >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """All §6 metrics for one model/task on one split. AUPRC is reported next to the
    positive prevalence, which is the AUPRC of a random classifier."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "auprc": float(average_precision_score(y, p)) if y.min() != y.max() else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
    return out


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
