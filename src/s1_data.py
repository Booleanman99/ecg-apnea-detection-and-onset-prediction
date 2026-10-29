"""§1 Data handling and setup.

Loads every ECG record, verifies that annotation minutes line up with signal samples,
produces the dataset audit, detects R-peaks, flags poor-quality minutes and builds the
window tables for Task A (detection) and Task B (prospective onset prediction).

Run:  python src/s1_data.py            (writes cache/records/*.npz, cache/windows_*.csv,
                                         outputs/audit.csv, outputs/alignment.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE_DIR, DATA_DIR, FS, HISTORY_MIN, HORIZON_MIN, MIN_SAMPLES,  # noqa: E402
                    OUT_DIR, save_json, split_of)

REC_CACHE = CACHE_DIR / "records"
REC_CACHE.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- loading
def list_records() -> list[str]:
    """ECG records only. 'xxxr' (respiration) and 'xxxer' (combined) are companions."""
    recs = [l.strip() for l in open(DATA_DIR / "RECORDS") if l.strip()]
    return [r for r in recs if not r.endswith("r") and not r.endswith("er")]


def load_record(rec: str) -> dict:
    sig, fields = wfdb.rdsamp(str(DATA_DIR / rec))
    assert fields["fs"] == FS, f"{rec}: fs={fields['fs']} != {FS}"
    ecg = sig[:, 0].astype(np.float32)          # mV; the only channel in every xxx.dat
    ann = wfdb.rdann(str(DATA_DIR / rec), "apn")
    return {"rec": rec, "ecg": ecg, "ann_sample": ann.sample, "ann_symbol": np.array(ann.symbol)}


def minute_labels(d: dict) -> np.ndarray:
    """Label per minute of signal: 1 = apnea ('A'), 0 = normal ('N'), -1 = not annotated.

    Alignment rule (verified in check_alignment): annotation i sits at sample i*6000 and
    describes the minute [i*6000, (i+1)*6000). Trailing minutes with signal but no
    annotation are marked -1 and never used.
    """
    n_min = len(d["ecg"]) // MIN_SAMPLES
    lab = np.full(n_min, -1, dtype=np.int8)
    for s, sym in zip(d["ann_sample"], d["ann_symbol"]):
        k = s // MIN_SAMPLES
        if k < n_min:
            lab[k] = 1 if sym == "A" else 0
    return lab


def check_alignment(d: dict) -> dict:
    """Assignment §1: a misalignment raises no error, so verify instead of assuming."""
    s = d["ann_sample"]
    n_min_signal = len(d["ecg"]) // MIN_SAMPLES
    return {
        "first_ann_sample": int(s[0]),
        "all_on_minute_grid": bool(np.all(s % MIN_SAMPLES == 0)),
        "all_steps_one_minute": bool(np.all(np.diff(s) == MIN_SAMPLES)),
        "n_annotations": int(len(s)),
        "n_signal_minutes": int(n_min_signal),
        "unannotated_trailing_minutes": int(n_min_signal - len(s)),
        "symbols": sorted(set(d["ann_symbol"].tolist())),
    }


# ----------------------------------------------------------------------------- R-peaks
_BP = butter(2, [5, 15], btype="bandpass", fs=FS, output="sos")


def detect_rpeaks(ecg: np.ndarray) -> np.ndarray:
    """Vectorised Pan-Tompkins-style detector: 5-15 Hz band-pass -> derivative -> square
    -> 150 ms moving integration -> peak picking with a 300 ms refractory period and a
    LOCAL adaptive threshold (25% of the median 5-s block maximum over 25 s). Blocks whose
    level is below 5% of the record's median level are treated as flat/lead-off and yield
    no beats, so flat or noisy stretches cannot stall the detector or crash the pipeline
    (xqrs, tried first, locked up on record b04's flat opening). Peak positions are
    refined to the largest |band-passed| sample within +-100 ms.

    Not causal: the local threshold is a centred median (sees ~15 s ahead) and the floor is
    a record-level median. For the prospective task (Task B) s3_features.history_rpeaks
    therefore runs this detector on the history slice ecg[t-W:t] alone, and s2_split.py
    checks by regression test that no Task B feature changes when everything at or after
    t is replaced by noise. Task A (detection on the window itself) uses one pass per record.
    Accuracy against the supplied .qrs is reported in outputs/audit.csv (PPV/sensitivity
    at 50 ms and 100 ms); on some records the .qrs fiducial sits a constant ~50 ms earlier
    than our R location, which leaves RR intervals unchanged.
    """
    y = sosfiltfilt(_BP, ecg.astype(float))
    energy = np.diff(y, prepend=y[0]) ** 2
    w = int(0.15 * FS)
    integ = np.convolve(energy, np.ones(w) / w, mode="same")
    blk = 5 * FS
    n = len(integ) // blk
    block_max = integ[: n * blk].reshape(n, blk).max(1)
    local = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(block_max, 2, mode="edge"), 5), axis=1)
    floor = 0.05 * np.median(block_max[block_max > 0]) if (block_max > 0).any() else 0.0
    thr_block = np.where(local < floor, np.inf, 0.25 * local)
    thr = np.repeat(thr_block, blk)
    thr = np.concatenate([thr, np.full(len(integ) - len(thr), thr[-1] if len(thr) else np.inf)])
    peaks, _ = find_peaks(integ, distance=int(0.3 * FS), height=thr)
    refined = [a + int(np.argmax(np.abs(y[a:b]))) for q in peaks for a, b in [(max(q - 10, 0), min(q + 10, len(y)))]]
    return np.unique(np.array(refined, dtype=np.int64))


def minute_sqi(ecg: np.ndarray, rpeaks: np.ndarray, n_min: int, half: int = 25) -> np.ndarray:
    """Beat-template signal-quality index per minute: median Pearson correlation of every
    beat (R +- 0.25 s) with that minute's median beat. Continuous, falls with noise,
    artefact and mis-detection, and only weakly related to apnea (checked in s7 as
    AUROC(sqi -> label)). Minutes with fewer than 3 usable beats get 0 (worst)."""
    sqi = np.zeros(n_min, dtype=np.float32)
    rp = rpeaks[(rpeaks >= half) & (rpeaks < len(ecg) - half)]
    minute = rp // MIN_SAMPLES
    for k in np.unique(minute):
        if k >= n_min:
            continue
        r = rp[minute == k]
        if len(r) < 3:
            continue
        seg = np.stack([ecg[i - half:i + half + 1] for i in r]).astype(float)
        seg = seg - seg.mean(1, keepdims=True)
        tmpl = np.median(seg, axis=0)
        tmpl = tmpl - tmpl.mean()
        num = seg @ tmpl
        den = np.linalg.norm(seg, axis=1) * np.linalg.norm(tmpl) + 1e-9
        sqi[k] = float(np.median(num / den))
    return sqi


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from every element of a to its nearest element of sorted b (searchsorted,
    so whole-record sizes are fine)."""
    idx = np.searchsorted(b, a)
    lo = b[np.clip(idx - 1, 0, len(b) - 1)]
    hi = b[np.clip(idx, 0, len(b) - 1)]
    return np.minimum(np.abs(a - lo), np.abs(a - hi))


def rpeak_agreement_with_supplied_qrs(rec: str, rpeaks: np.ndarray) -> dict:
    """Sanity check of our detector against the supplied .qrs annotations, both ways:
    ppv = fraction of OUR peaks within tol of a .qrs beat, sensitivity = fraction of .qrs
    beats within tol of one of ours (missed beats in noisy stretches show up here).
    The .qrs files are used ONLY for this check, never as a model input."""
    out = {}
    try:
        ref = wfdb.rdann(str(DATA_DIR / rec), "qrs").sample
    except FileNotFoundError:
        ref = np.array([], dtype=int)
    for tol_ms in (50, 100):
        tol = tol_ms / 1000 * FS
        ok = len(ref) > 0 and len(rpeaks) > 0
        out[f"ppv_{tol_ms}ms"] = float(np.mean(_nn_dist(rpeaks, ref) <= tol)) if ok else None
        out[f"sens_{tol_ms}ms"] = float(np.mean(_nn_dist(ref, rpeaks) <= tol)) if ok else None
    return out


# ----------------------------------------------------------------------------- quality
def minute_quality(ecg: np.ndarray, rpeaks: np.ndarray, n_min: int) -> pd.DataFrame:
    """Per-minute signal-quality flags. Nothing is dropped on their account (§8); they feed
    the error analysis (§7) and are imputed/zeroed for the models.

    flat_frac : share of 1-s segments with peak-to-peak < 0.05 mV (lead-off / flat line)
    clip_frac : share of samples at the ADC rails (|x| > 10 mV at gain 200, 12-bit)
    n_beats   : detected R-peaks in the minute
    plausible : 20 <= n_beats <= 220 and flat_frac < 0.5
    """
    x = ecg[: n_min * MIN_SAMPLES].reshape(n_min, 60, FS)
    ptp = x.max(-1) - x.min(-1)
    flat_frac = (ptp < 0.05).mean(-1)
    clip_frac = (np.abs(x) > 10.0).reshape(n_min, -1).mean(-1)
    n_beats = np.bincount(np.minimum(rpeaks // MIN_SAMPLES, n_min - 1), minlength=n_min)[:n_min]
    plausible = (n_beats >= 20) & (n_beats <= 220) & (flat_frac < 0.5)
    return pd.DataFrame({"flat_frac": flat_frac, "clip_frac": clip_frac,
                         "n_beats": n_beats, "plausible": plausible})


# ----------------------------------------------------------------------------- windows
def task_a_windows(rec: str, lab: np.ndarray) -> pd.DataFrame:
    """One row per annotated minute: the window IS the annotated minute."""
    k = np.flatnonzero(lab >= 0)
    return pd.DataFrame({
        "subject_id": rec, "split": split_of(rec), "minute": k,
        "window_start": k * 60, "window_end": (k + 1) * 60, "label": lab[k].astype(int),
    })


def task_b_windows(rec: str, lab: np.ndarray, history_min: int = HISTORY_MIN,
                   horizon_min: int = HORIZON_MIN) -> pd.DataFrame:
    """Prospective windows. For prediction time t = k minutes:
        history  = minutes [k-W, k)            (input; everything < t)
        horizon  = minutes [k, k+H)            (target; everything > t)
        onset(m) = lab[m]==1 and lab[m-1]==0   (event starts in minute m)
        label    = any onset in the horizon
        ongoing  = lab[k-1]==1  -> an event is in progress at t. These rows are kept in
                   the table but flagged; the default pipeline EXCLUDES them (brief: an
                   ongoing event must not count as a future prediction). s8_controls.py
                   uses the flag to show what happens if they are kept.
    Only windows whose history and horizon are fully annotated are produced.
    """
    n = len(lab)
    onset = np.zeros(n, dtype=bool)
    onset[1:] = (lab[1:] == 1) & (lab[:-1] == 0)
    rows = []
    for k in range(history_min, n - horizon_min + 1):
        hist, hor = lab[k - history_min:k], lab[k:k + horizon_min]
        if (hist < 0).any() or (hor < 0).any():
            continue
        rows.append((rec, split_of(rec), k, (k - history_min) * 60, k * 60, k * 60,
                     (k + horizon_min) * 60, int(onset[k:k + horizon_min].any()),
                     int(lab[k - 1] == 1), int(lab[k] == 1)))
    return pd.DataFrame(rows, columns=["subject_id", "split", "minute", "window_start",
                                       "window_end", "horizon_start", "horizon_end",
                                       "label", "ongoing_at_t", "apnea_in_horizon"])


# ----------------------------------------------------------------------------- main
def process_record(rec: str, force: bool = False) -> dict:
    path = REC_CACHE / f"{rec}.npz"
    if path.exists() and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"rec": rec}
    d = load_record(rec)
    lab = minute_labels(d)
    rpeaks = detect_rpeaks(d["ecg"])
    q = minute_quality(d["ecg"], rpeaks, len(lab))
    sqi = minute_sqi(d["ecg"], rpeaks, len(lab))
    np.savez_compressed(path, ecg=d["ecg"], labels=lab, rpeaks=rpeaks, sqi=sqi,
                        flat_frac=q.flat_frac.values, clip_frac=q.clip_frac.values,
                        n_beats=q.n_beats.values, plausible=q.plausible.values,
                        ann_sample=d["ann_sample"], ann_symbol=d["ann_symbol"])
    return process_record(rec)


def main(force: bool = False) -> None:
    recs = list_records()
    audit, align, wa, wb = [], {}, [], []
    for rec in recs:
        d = process_record(rec, force=force)
        raw = {"ecg": d["ecg"], "ann_sample": d["ann_sample"], "ann_symbol": d["ann_symbol"]}
        align[rec] = check_alignment(raw)
        lab = d["labels"]
        ok = lab >= 0
        # physiological sanity: apnea minutes should show larger HR variability (cyclic
        # variation of heart rate) than normal minutes within the same night
        rr = np.diff(d["rpeaks"]) / FS
        rr_min = (d["rpeaks"][1:] // MIN_SAMPLES)
        sdnn = pd.Series(rr).groupby(rr_min).std().reindex(range(len(lab)))
        sd_a = float(sdnn[lab == 1].mean()) if (lab == 1).any() else float("nan")
        sd_n = float(sdnn[lab == 0].mean()) if (lab == 0).any() else float("nan")
        audit.append({
            "record": rec, "split": split_of(rec), "fs": FS,
            "hours": len(d["ecg"]) / FS / 3600, "annotated_minutes": int(ok.sum()),
            "apnea_minutes": int((lab == 1).sum()), "prevalence": float((lab == 1).sum() / ok.sum()),
            "flat_minutes(>50%)": int((d["flat_frac"] > 0.5).sum()),
            "implausible_minutes": int((~d["plausible"]).sum()),
            "sqi_median": float(np.median(d["sqi"][ok])), "sqi_p10": float(np.percentile(d["sqi"][ok], 10)),
            **{f"rpeak_{k}_vs_qrs": v for k, v in rpeak_agreement_with_supplied_qrs(rec, d["rpeaks"]).items()},
            "sdnn_apnea_s": sd_a, "sdnn_normal_s": sd_n,
            "has_resp_spo2": (DATA_DIR / f"{rec}r.hea").exists(),
        })
        wa.append(task_a_windows(rec, lab))
        wb.append(task_b_windows(rec, lab))
        print(f"{rec}: {align[rec]} ; prevalence={audit[-1]['prevalence']:.2f} "
              f"rpeak ppv50={audit[-1]['rpeak_ppv_50ms_vs_qrs']} sens50={audit[-1]['rpeak_sens_50ms_vs_qrs']} "
              f"sens100={audit[-1]['rpeak_sens_100ms_vs_qrs']} sqi={audit[-1]['sqi_median']:.2f}")

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT_DIR / "audit.csv", index=False)
    save_json(align, OUT_DIR / "alignment.json")
    assert all(a["all_on_minute_grid"] and a["all_steps_one_minute"] and a["first_ann_sample"] == 0
               for a in align.values()), "annotation grid is not aligned to 1-min sample blocks"
    A = pd.concat(wa, ignore_index=True)
    B = pd.concat(wb, ignore_index=True)
    A.to_csv(CACHE_DIR / "windows_A.csv", index=False)
    B.to_csv(CACHE_DIR / "windows_B.csv", index=False)
    print("\nAudit:\n", audit_df.to_string(index=False))
    print(f"\nTask A windows: {len(A)}  prevalence={A.label.mean():.3f}")
    Bx = B[B.ongoing_at_t == 0]
    print(f"Task B windows (ongoing excluded): {len(Bx)}  onset prevalence={Bx.label.mean():.3f}"
          f"   (all rows {len(B)}, ongoing_at_t={int(B.ongoing_at_t.sum())})")
    print(A.groupby('split').label.agg(['count', 'mean']))
    print(Bx.groupby('split').label.agg(['count', 'mean']))


if __name__ == "__main__":
    main(force="--force" in sys.argv)
