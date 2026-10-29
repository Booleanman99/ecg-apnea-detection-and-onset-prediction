"""§3/§4 Feature extraction for the non-deep baselines.

Every feature is computed from samples inside the window only (for Task B: inside the
history [t-W, t)), so nothing after the prediction time is used.

Why these features: obstructive apnea leaves two signatures in a single-lead ECG —
(1) cyclic variation of heart rate (bradycardia during the event, tachycardia at
arousal), visible as RR-interval variability and low-frequency HRV power, and
(2) respiration-modulated R-wave amplitude (ECG-derived respiration, EDR), which
changes when breathing stops. Plus signal-quality features so the model can discount
bad minutes rather than the pipeline dropping them.

Run:  python src/s3_features.py   (writes cache/features_A.parquet, cache/features_B.parquet)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FS, MIN_SAMPLES  # noqa: E402
from s1_data import REC_CACHE, detect_rpeaks, list_records, minute_sqi  # noqa: E402

CORE = ["n_beats", "hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "rr_min", "rr_max",
        "rr_range", "rr_cv", "vlf", "lf", "hf", "lf_hf", "edr_std", "edr_resp_power",
        "edr_peak_freq", "ecg_std", "ecg_ptp", "flat_frac", "clip_frac"]
TRAJ = ["hr_mean", "sdnn", "rmssd", "edr_std"]       # per-minute trajectory for Task B
GUARD_S = 5                                            # seconds before t ignored in Task B


def history_rpeaks(ecg: np.ndarray, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Task B: detect beats on the history slice ecg[s:e] ONLY, so the detector's adaptive
    threshold, its record-level floor and its zero-phase filters never see a sample at or
    after the prediction time t = e. Returns absolute peak positions earlier than
    e - GUARD_S (belt and braces against filter edge effects) and the per-minute SQI of
    the slice computed from those peaks."""
    seg = ecg[s:e]
    rp = detect_rpeaks(seg)
    sqi = minute_sqi(seg, rp, (e - s) // MIN_SAMPLES)   # SQI uses all history-only beats
    rp = rp[rp < (e - s) - GUARD_S * FS] + s             # guard applies to beat-derived features
    return rp, sqi


def task_b_features(ecg: np.ndarray, s: int, e: int) -> dict:
    """The complete Task B feature vector for history ecg[s:e]. This single function is
    what build() writes to features_B.parquet AND what the s2 regression test calls, so
    the test exercises the code that trains the models, not a copy."""
    rp, sq = history_rpeaks(ecg, s, e)
    feats = segment_features(ecg, rp, s, e)
    feats["sqi_mean"], feats["sqi_last"] = float(sq.mean()), float(sq[-1])
    # per-minute trajectory of a few core features across the history, plus the last-minute
    # deviation from the earlier minutes (is the night changing?)
    mins = []
    for m in range((e - s) // MIN_SAMPLES):
        fm = segment_features(ecg, rp, s + m * MIN_SAMPLES, s + (m + 1) * MIN_SAMPLES)
        mins.append({k: fm.get(k, np.nan) for k in TRAJ})
        for k in TRAJ:
            feats[f"{k}_m{m}"] = fm.get(k, np.nan)
    mdf = pd.DataFrame(mins)
    for k in TRAJ:
        feats[f"{k}_last_minus_mean"] = mdf[k].iloc[-1] - mdf[k].iloc[:-1].mean() if len(mdf) > 1 else np.nan
    return feats


def task_b_windows_features(ecg: np.ndarray, win: pd.DataFrame) -> list[dict]:
    """task_b_features for every row of a window table (used by s3 and the s8 sweeps)."""
    return [{**row._asdict(), **task_b_features(ecg, int(row.window_start * FS), int(row.window_end * FS))}
            for row in win.itertuples(index=False)]


def last_minute_features_causal(ecg: np.ndarray, s: int, e: int) -> dict:
    """Task A-style features of the LAST minute of a history [s, e), with beats and SQI from
    history_rpeaks over the whole history (so detector warm-up matches Task B and nothing
    at or after e is read). Used by the 'detection model on prediction windows' control."""
    rp, sq = history_rpeaks(ecg, s, e)
    feats = segment_features(ecg, rp, e - MIN_SAMPLES, e)
    feats["sqi_mean"] = feats["sqi_last"] = float(sq[-1])
    return feats


def _band(freqs: np.ndarray, pgram: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(np.trapezoid(pgram[m], freqs[m])) if m.sum() > 1 else 0.0


def segment_features(ecg: np.ndarray, rpeaks: np.ndarray, start: int, end: int) -> dict:
    """Features of the signal in [start, end) samples. rpeaks are absolute sample indices."""
    x = ecg[start:end]
    r = rpeaks[(rpeaks >= start) & (rpeaks < end)] - start
    seg = x.reshape(-1, FS) if len(x) % FS == 0 else x[: len(x) // FS * FS].reshape(-1, FS)
    f = {
        "n_beats": len(r),
        "ecg_std": float(x.std()), "ecg_ptp": float(x.max() - x.min()),
        "flat_frac": float(((seg.max(1) - seg.min(1)) < 0.05).mean()),
        "clip_frac": float((np.abs(x) > 10.0).mean()),
    }
    if len(r) < 5:                       # not enough beats: HRV undefined -> NaN (imputed later)
        for k in CORE:
            f.setdefault(k, np.nan)
        return f
    t = r / FS
    rr = np.diff(t)
    ok = (rr > 0.3) & (rr < 2.0)         # physiologic RR only (30-200 bpm); rejects false peaks
    rr, trr = rr[ok], t[1:][ok]
    if len(rr) < 4:
        for k in CORE:
            f.setdefault(k, np.nan)
        return f
    hr = 60.0 / rr
    d = np.diff(rr)
    f.update({
        "hr_mean": hr.mean(), "hr_std": hr.std(), "sdnn": rr.std(),
        "rmssd": np.sqrt(np.mean(d ** 2)) if len(d) else np.nan,
        "pnn50": float(np.mean(np.abs(d) > 0.05)) if len(d) else np.nan,
        "rr_min": rr.min(), "rr_max": rr.max(), "rr_range": rr.max() - rr.min(),
        "rr_cv": rr.std() / rr.mean(),
    })
    # HRV spectrum on the unevenly sampled RR series (Lomb-Scargle, no interpolation)
    freqs = np.linspace(0.005, 0.5, 100)
    p = lombscargle(trr - trr.mean(), rr - rr.mean(), 2 * np.pi * freqs, normalize=True)
    vlf, lf, hf = _band(freqs, p, 0.003, 0.04), _band(freqs, p, 0.04, 0.15), _band(freqs, p, 0.15, 0.4)
    f.update({"vlf": vlf, "lf": lf, "hf": hf, "lf_hf": lf / (hf + 1e-9)})
    # ECG-derived respiration: R-peak amplitude relative to the window median
    amp = x[r] - np.median(x)
    amp_t = r / FS
    f["edr_std"] = float(amp.std())
    fr = np.linspace(0.05, 0.7, 80)
    pe = lombscargle(amp_t - amp_t.mean(), amp - amp.mean(), 2 * np.pi * fr, normalize=True)
    f["edr_resp_power"] = _band(fr, pe, 0.1, 0.5)
    f["edr_peak_freq"] = float(fr[int(np.argmax(pe))])
    return f


def build(task: str) -> pd.DataFrame:
    win = pd.read_csv(CACHE_DIR / f"windows_{task}.csv")
    out = []
    for rec in list_records():
        z = np.load(REC_CACHE / f"{rec}.npz", allow_pickle=True)
        ecg, rpeaks, sqi = z["ecg"], z["rpeaks"], z["sqi"]
        for row in win[win.subject_id == rec].itertuples(index=False):
            s, e = int(row.window_start * FS), int(row.window_end * FS)
            if task == "B":
                # Causality by construction: beats, SQI and every feature from ecg[s:e] only.
                feats = task_b_features(ecg, s, e)
            else:
                # Task A is detection on the window itself: whole-record beats are fine.
                m0, m1 = int(row.window_start // 60), int(row.window_end // 60)
                feats = segment_features(ecg, rpeaks, s, e)
                feats["sqi_mean"], feats["sqi_last"] = float(sqi[m0:m1].mean()), float(sqi[m1 - 1])
            out.append({**row._asdict(), **feats})
        print(f"{task} {rec} done ({len(out)} windows so far)")
    df = add_running_baseline(pd.DataFrame(out))
    df.to_parquet(CACHE_DIR / f"features_{task}.parquet", index=False)
    return df


REL_MIN_PERIODS = 5


def add_running_baseline(df: pd.DataFrame, cols: list[str] = CORE) -> pd.DataFrame:
    """Subject-relative features, causal. For each record and feature f, f_rel at window k
    = f - median(f over this record's EARLIER windows). Uses only minutes before the
    current window (shift(1) + expanding median, min 5 earlier windows, else NaN ->
    imputed later), so nothing after the prediction time is touched. Motivation: resting
    HRV differs between patients far more than apnea shifts it within a night, so pooled
    window features discriminate poorly across subjects (measured: within-record AUROC
    0.7-0.9 vs pooled 0.63 for SDNN); the running baseline restores the within-night
    contrast with information a deployed system would also have (the night so far)."""
    df = df.sort_values(["subject_id", "minute"]).reset_index(drop=True)
    for c in cols:
        base = df.groupby("subject_id")[c].transform(lambda x: x.shift(1).expanding(REL_MIN_PERIODS).median())
        df[f"{c}_rel"] = df[c] - base
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {"subject_id", "split", "minute", "window_start", "window_end", "horizon_start",
            "horizon_end", "label", "ongoing_at_t", "apnea_in_horizon"}
    return [c for c in df.columns if c not in meta]


def window_only_columns(df: pd.DataFrame) -> list[str]:
    """Ablation: the same features without the subject-relative (_rel) ones."""
    return [c for c in feature_columns(df) if not c.endswith("_rel")]


if __name__ == "__main__":
    for task in ("A", "B"):
        df = build(task)
        print(task, df.shape, "NaN rows:", int(df[feature_columns(df)].isna().any(axis=1).sum()))
