"""§5 One temporal deep model for both tasks.

Architecture (one, used for both tasks):
    raw ECG minute (6000 samples) -> small 1-D CNN minute encoder -> 64-d embedding
    sequence of minute embeddings  -> GRU -> last hidden state -> logit
Task A feeds one minute (sequence length 1), so for Task A the GRU is inert and the model
is effectively CNN-only; Task B feeds the five history minutes (sequence length 5) and
the GRU is where the temporal modelling happens. ~60k parameters, trains in minutes on a
laptop GPU and in well under an hour on a free Colab CPU.

Why this shape: labels are per minute, so a per-minute encoder matches the annotation
unit; the GRU is the temporal component that lets Task B see how the night is evolving
across minutes (the same information the baseline gets through its per-minute feature
trajectory). Convolutions see the local morphology (R-peaks, their amplitude, their
spacing) that the HRV/EDR features summarise by hand.

Inputs are normalised per window (mean/std of the window itself), so nothing after the
prediction time is ever used, and a 0.5 Hz high-pass inside the window removes baseline
wander.

Run:  python src/s5_deep.py   (writes outputs/models/{task}_cnn_s{seed}.pt and
                               outputs/preds/{task}_cnn_s{seed}_val.csv; NO test predictions)

"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, FS, MIN_SAMPLES, OUT_DIR, SEEDS, seed_everything  # noqa: E402
from s1_data import REC_CACHE, list_records  # noqa: E402
from s4_baseline import META, PRED_DIR, save_preds  # noqa: E402

MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HP = butter(2, 0.5 / (FS / 2), btype="highpass", output="sos")


def load_ecg() -> dict[str, np.ndarray]:
    return {r: np.load(REC_CACHE / f"{r}.npz", allow_pickle=True)["ecg"] for r in list_records()}


RUNNING_MIN = 10                                                  # causal context for norm="running"


class WindowDataset(Dataset):
    """norm='window' (shipped): z-score with the window's own mean/std.
    norm='running' (round-2 variant): z-score with the median/IQR of the RUNNING_MIN minutes
    that END at the window start (earlier signal only, so still nothing after t); falls back
    to the window's own statistics when fewer than 2 earlier minutes exist. Amplitude and
    polarity cues then describe the night so far, like the baselines' running features."""

    def __init__(self, df: pd.DataFrame, ecg: dict[str, np.ndarray], train: bool = False, flip_aug: bool = False,
                 norm: str = "window"):
        self.df, self.ecg, self.train, self.flip_aug, self.norm = df.reset_index(drop=True), ecg, train, flip_aug, norm
        self.n_min = int((df.window_end - df.window_start).iloc[0] // 60)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        s, e = int(r.window_start * FS), int(r.window_end * FS)
        x = self.ecg[r.subject_id][s:e].astype(np.float64)
        x = sosfiltfilt(_HP, x)                                   # window-local, no future samples
        if self.norm == "running" and s >= 2 * MIN_SAMPLES:
            ctx = sosfiltfilt(_HP, self.ecg[r.subject_id][max(0, s - RUNNING_MIN * MIN_SAMPLES):s].astype(np.float64))
            med, iqr = np.median(ctx), np.subtract(*np.percentile(ctx, [75, 25]))
            x = (x - med) / (iqr + 1e-6)
        else:
            x = (x - x.mean()) / (x.std() + 1e-6)                 # window-local z-score
        x = x.reshape(self.n_min, MIN_SAMPLES).astype(np.float32)
        if self.train:                                             # light augmentation
            x = x * np.float32(np.random.uniform(0.8, 1.2)) + np.random.normal(0, 0.05, x.shape).astype(np.float32)
            if self.flip_aug and np.random.rand() < 0.5:
                x = -x
        return torch.from_numpy(x), torch.tensor(float(r.label))


class MinuteEncoder(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        def block(ci, co, k, s):
            return nn.Sequential(nn.Conv1d(ci, co, k, stride=s, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU())
        self.net = nn.Sequential(block(1, 16, 7, 2), block(16, 32, 7, 2), nn.MaxPool1d(4),
                                 block(32, 64, 5, 2), block(64, d, 5, 2), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):                      # x: (batch, samples)
        return self.net(x.unsqueeze(1)).squeeze(-1)


class CNNGRU(nn.Module):
    def __init__(self, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.enc = MinuteEncoder(d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):                      # x: (batch, minutes, samples)
        b, m, n = x.shape
        z = self.enc(x.reshape(b * m, n)).reshape(b, m, -1)
        _, h = self.gru(z)
        return self.head(h[-1]).squeeze(-1)


@torch.no_grad()
def predict(model: nn.Module, ds: Dataset, batch: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2):
        out.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def within_record_auroc(df: pd.DataFrame, p: np.ndarray, min_pos: int = 5) -> float:
    from sklearn.metrics import roc_auc_score
    vals = []
    for r, g in df.assign(p=p).groupby("subject_id"):
        if (g.label == 1).sum() >= min_pos and (g.label == 0).sum() >= min_pos:
            vals.append(roc_auc_score(g.label, g.p))
    return float(np.mean(vals)) if vals else float("nan")


def train_model(tr: pd.DataFrame, va: pd.DataFrame, ecg: dict, seed: int, max_epochs: int = 25,
                patience: int = 5, lr: float = 1e-3, verbose: bool = True,
                select: str = "pooled_auprc", flip_aug: bool = False, norm: str = "window",
                model_fn=None) -> tuple[nn.Module, dict]:
    """select: 'pooled_auprc' (pre-registered) or 'within_record_auroc' (post-hoc control in
    s9). flip_aug: random polarity flip per window (post-hoc control; lead polarity carries
    no apnea information but identifies patients). norm: see WindowDataset. model_fn: a
    zero-argument constructor for an alternative architecture (s13 screen); default CNNGRU."""
    seed_everything(seed)
    ds_tr, ds_va = WindowDataset(tr, ecg, train=True, flip_aug=flip_aug, norm=norm), WindowDataset(va, ecg, norm=norm)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=2, generator=g, drop_last=True)
    model = (model_fn() if model_fn is not None else CNNGRU()).to(DEVICE)
    pos_w = torch.tensor((tr.label == 0).sum() / max((tr.label == 1).sum(), 1), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)                 # imbalance: weighted loss (§3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        p_va = predict(model, ds_va)
        ap = average_precision_score(va.label, p_va) if select == "pooled_auprc" else within_record_auroc(va, p_va)
        hist.append({"epoch": ep, "train_loss": tot / len(ds_tr), f"val_{select}": float(ap)})
        if verbose:
            print(f"    ep{ep:02d} loss={tot/len(ds_tr):.4f} val_auprc={ap:.4f} ({time.time()-t0:.0f}s)")
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:                                      # early stopping on val AUPRC
                break
    model.load_state_dict(best_state)
    return model, {"best_val_auprc": best, "select": select, "epochs": len(hist), "history": hist,
                   "seconds": time.time() - t0, "device": DEVICE,
                   "params": sum(p.numel() for p in model.parameters())}


def main() -> None:
    ecg = load_ecg()
    for task in ("A", "B"):
        df = pd.read_parquet(CACHE_DIR / f"features_{task}.parquet")
        if task == "B":
            df = df[df.ongoing_at_t == 0]
        parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val")}   # test is never loaded here
        for seed in SEEDS:
            print(f"Task {task} CNN+GRU seed {seed} on {DEVICE}")
            model, info = train_model(parts["train"], parts["val"], ecg, seed)
            torch.save(model.state_dict(), MODEL_DIR / f"{task}_cnn_s{seed}.pt")
            pd.DataFrame(info["history"]).to_csv(MODEL_DIR / f"{task}_cnn_s{seed}_history.csv", index=False)
            save_preds(parts["val"], predict(model, WindowDataset(parts["val"], ecg)), task, "cnn", seed, "val")
            print(f"  -> best val AUPRC {info['best_val_auprc']:.4f} in {info['epochs']} epochs, "
                  f"{info['seconds']:.0f}s, {info['params']} params")


def load_cnn(task: str, seed: int) -> nn.Module:
    model = CNNGRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / f"{task}_cnn_s{seed}.pt", map_location=DEVICE))
    return model


if __name__ == "__main__":
    main()
