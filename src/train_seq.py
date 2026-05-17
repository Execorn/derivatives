"""
Heston Temporal Dynamics — LSTM Training Pipeline.

Trains HestonDynamicsLSTM to predict tomorrow's Heston parameters from
a 10-day sequence of Total Variance surfaces (W = IV² × T).

Design principles:
  - Z-score normalized labels: prevents gradient starvation from scale
    differences across the 5 Heston parameters (κ ≈ 3.0 vs v₀ ≈ 0.02)
  - Gaussian noise augmentation (σ=0.001) on input surfaces: regularizes
    the LSTM against grid-level artifacts; applied only during training
  - AdamW + weight_decay=1e-4: decoupled weight decay for better
    generalization vs vanilla Adam on financial time series
  - CosineAnnealingLR: smooth LR decay without abrupt steps; avoids
    the instability of ReduceLROnPlateau on noisy val losses
  - Early stopping on normalized val MSE (patience=20 epochs)
  - Gradient clipping (max_norm=1.0): guards against LSTM gradient spikes

Usage:
    cd path/to/derivatives
    python src/train_seq.py [--epochs 200] [--batch-size 512] [--lr 3e-4]

Output:
    artifacts/weights/heston_lstm_best.pth   — best model state dict
    artifacts/scalers/lstm_label_stats.npz   — label_mean, label_std
    logs/lstm_training.csv                   — epoch-level metrics log
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Path setup ──────────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from seq_model import HestonDynamicsLSTM, PARAM_NAMES

# ── Artifact paths ──────────────────────────────────────────────────────────────
DATA_PATH    = PROJECT_ROOT / "data"      / "seq_dataset.npz"
WEIGHTS_DIR  = PROJECT_ROOT / "artifacts" / "weights"
SCALERS_DIR  = PROJECT_ROOT / "artifacts" / "scalers"
LOG_DIR      = PROJECT_ROOT / "logs"
CHECKPOINT   = WEIGHTS_DIR  / "heston_lstm_best.pth"
LABEL_STATS  = SCALERS_DIR  / "lstm_label_stats.npz"
TRAIN_LOG    = LOG_DIR      / "lstm_training.csv"

# ── Default hyperparameters ─────────────────────────────────────────────────────
EPOCHS           = 200
BATCH_SIZE       = 512
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 1.0
NOISE_STD        = 0.001   # Gaussian augmentation on input W surfaces
EARLY_STOP_PAT   = 20      # epochs of no val improvement before stopping
COSINE_T_MAX     = 100     # CosineAnnealingLR period (restarts every T_max epochs)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class HestonSequenceDataset(Dataset):
    """
    PyTorch Dataset for (10-day W surface sequence, next-day parameters) pairs.

    Applies Z-score label normalization using training-set statistics and
    optional Gaussian noise augmentation on the input surfaces.

    Args:
        X:           Input surface sequences, shape (N, seq_len, 88).
        y:           Raw Heston parameter labels, shape (N, 5).
        label_mean:  Per-parameter mean computed from the training set.
        label_std:   Per-parameter std  computed from the training set.
        augment:     If True, add Gaussian noise (σ=``noise_std``) to X.
        noise_std:   Standard deviation of augmentation noise (default 0.001).
    """

    def __init__(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        label_mean: np.ndarray,
        label_std:  np.ndarray,
        augment:    bool  = False,
        noise_std:  float = NOISE_STD,
    ) -> None:
        self.X          = torch.tensor(X, dtype=torch.float32)
        self.label_mean = torch.tensor(label_mean, dtype=torch.float32)
        self.label_std  = torch.tensor(label_std,  dtype=torch.float32)
        self.augment    = augment
        self.noise_std  = noise_std

        # Pre-normalize labels: ŷ = (y − μ) / σ
        y_tensor   = torch.tensor(y, dtype=torch.float32)
        self.y_norm = (y_tensor - self.label_mean) / self.label_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y_norm[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitors validation loss and signals early stopping when no improvement
    occurs within ``patience`` epochs.

    Args:
        patience:  Epochs to wait after last improvement (default: 20).
        min_delta: Minimum absolute change to qualify as improvement.
    """

    def __init__(self, patience: int = EARLY_STOP_PAT, min_delta: float = 1e-6) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:     HestonDynamicsLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    train:     bool,
) -> float:
    """
    Run one full epoch (train or eval).

    Returns:
        Mean MSE loss over all batches.
    """
    model.train(train)
    total_loss = 0.0

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)           # raw logits, shape (B, 5)
            loss = criterion(pred, y_batch) # MSE on Z-score normalized labels

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def train(
    epochs:     int   = EPOCHS,
    batch_size: int   = BATCH_SIZE,
    lr:         float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    data_path:  Path  = DATA_PATH,
) -> None:
    """
    Full training pipeline for HestonDynamicsLSTM.

    Loads data, builds datasets/loaders, trains with early stopping,
    saves best weights and label statistics for inference.
    """
    # ── Output directories ───────────────────────────────────────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)
    if device.type == "cuda":
        device_str += f" — {torch.cuda.get_device_name(0)}"
    print(f"[Device] {device_str}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\n[Data] Loading from {data_path.name} …")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run:  python scripts/generate_seq_data.py  first."
        )
    ds = np.load(data_path)

    label_mean = ds["label_mean"].astype(np.float32)  # (5,)
    label_std  = ds["label_std"].astype(np.float32)   # (5,)

    train_ds = HestonSequenceDataset(
        ds["X_train"], ds["y_train"], label_mean, label_std, augment=True
    )
    val_ds = HestonSequenceDataset(
        ds["X_val"], ds["y_val"], label_mean, label_std, augment=False
    )
    test_ds = HestonSequenceDataset(
        ds["X_test"], ds["y_test"], label_mean, label_std, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    print(f"    Train : {len(train_ds):>8,} sequences  ({len(train_loader)} batches)")
    print(f"    Val   : {len(val_ds):>8,} sequences  ({len(val_loader)} batches)")
    print(f"    Test  : {len(test_ds):>8,} sequences  ({len(test_loader)} batches)")

    # ── Save label statistics (needed at inference) ───────────────────────────
    np.savez(LABEL_STATS, label_mean=label_mean, label_std=label_std)
    print(f"\n[Scalers] Label stats saved → {LABEL_STATS}")
    for name, mu, sd in zip(PARAM_NAMES, label_mean, label_std):
        print(f"    {name:6s}: μ={mu:.5f}  σ={sd:.5f}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = HestonDynamicsLSTM().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] HestonDynamicsLSTM — {n_params:,} parameters")

    # ── Optimizer and scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=COSINE_T_MAX, eta_min=lr * 1e-2
    )
    criterion  = nn.MSELoss()
    stopper    = EarlyStopping(patience=EARLY_STOP_PAT)

    # ── Training log ─────────────────────────────────────────────────────────
    log_file   = open(TRAIN_LOG, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_mse", "val_mse", "lr", "best"])

    # ── Header ───────────────────────────────────────────────────────────────
    col_w = 12
    header = (
        f"{'Epoch':>6}  {'Train MSE':>{col_w}}  {'Val MSE':>{col_w}}  "
        f"{'Val RMSE':>{col_w}}  {'LR':>10}  {'':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    best_val_mse = float("inf")
    best_epoch   = 0
    t0_total     = time.perf_counter()

    for epoch in range(1, epochs + 1):

        train_mse = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_mse   = run_epoch(model, val_loader,   criterion, None,      device, train=False)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Checkpoint ───────────────────────────────────────────────────────
        marker = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch   = epoch
            torch.save(model.state_dict(), CHECKPOINT)
            marker = "★ best"

        # ── Logging ──────────────────────────────────────────────────────────
        log_writer.writerow([epoch, f"{train_mse:.6f}", f"{val_mse:.6f}",
                             f"{current_lr:.2e}", marker])
        log_file.flush()

        if epoch <= 5 or epoch % 10 == 0 or marker:
            print(
                f"{epoch:>6}  {train_mse:>{col_w}.6f}  {val_mse:>{col_w}.6f}  "
                f"{val_mse**0.5:>{col_w}.6f}  {current_lr:>10.2e}  {marker}"
            )

        # ── Early stopping ───────────────────────────────────────────────────
        if stopper.step(val_mse, epoch):
            print(f"\n[Early stopping] No improvement for {EARLY_STOP_PAT} epochs. "
                  f"Best epoch: {best_epoch}")
            break

    log_file.close()
    total_time = time.perf_counter() - t0_total
    print(f"{sep}")
    print(f"\n✓ Training complete.")
    print(f"  Best epoch   : {best_epoch}")
    print(f"  Best val MSE : {best_val_mse:.6f}  (normalized)")
    print(f"  Total time   : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Weights      → {CHECKPOINT}")
    print(f"  Training log → {TRAIN_LOG}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[Test evaluation] Loading best checkpoint …")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False))
    test_mse = run_epoch(model, test_loader, criterion, None, device, train=False)

    # Convert test MSE back to original (denormalized) per-parameter RMSE
    # MSE is computed on Z-score normalized labels: ŷ = (y-μ)/σ
    # Denormalized RMSE per param ≈ sqrt(test_mse) * σ_i  (approx, not exact per-param)
    print(f"\n{'='*62}")
    print(f"  {'Metric':<30}  {'Value':>12}")
    print(f"{'='*62}")
    print(f"  {'Test MSE (normalized)':<30}  {test_mse:>12.6f}")
    print(f"  {'Test RMSE (normalized)':<30}  {test_mse**0.5:>12.6f}")
    print(f"\n  Approx. per-parameter RMSE in original scale:")
    for name, sd in zip(PARAM_NAMES, label_std):
        approx_rmse = test_mse**0.5 * sd
        print(f"    {name:6s}: ≈ {approx_rmse:.5f}")
    print(f"{'='*62}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train HestonDynamicsLSTM")
    parser.add_argument("--epochs",       type=int,   default=EPOCHS,       help=f"Max epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,   help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE, help=f"Initial learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,  help=f"AdamW weight decay (default: {WEIGHT_DECAY})")
    parser.add_argument("--data",         type=Path,  default=DATA_PATH,     help=f"Path to seq_dataset.npz")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_path=args.data,
    )


if __name__ == "__main__":
    main()
