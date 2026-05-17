"""
Heston Surrogate MLP — Training Pipeline.

Trains HestonSurrogateMLP to approximate the Heston pricing function:
    Heston parameters (5) → Total Variance surface (88 = W = IV² × T)

Scalers are saved by data_loader.py; best weights are saved to
artifacts/weights/heston_best.pth.

Usage:
    cd path/to/derivatives
    python src/train.py [--epochs N] [--lr LR] [--batch-size B]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Ensure src/ is on the path so sibling imports work
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import get_dataloaders
from model import HestonSurrogateMLP

# ─── Paths ─────────────────────────────────────────────────────────────────────

DATA_PATH = PROJECT_ROOT / "data" / "HestonTrainSet.txt.gz"
SCALERS_DIR = str(PROJECT_ROOT / "artifacts" / "scalers")
WEIGHTS_DIR = PROJECT_ROOT / "artifacts" / "weights"
CHECKPOINT = WEIGHTS_DIR / "heston_best.pth"

# ─── Hyperparameters ───────────────────────────────────────────────────────────

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_SIZE = 0.15
RANDOM_STATE = 42
SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6


# ─── Training Loop ─────────────────────────────────────────────────────────────


def train(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LEARNING_RATE):
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=SCHEDULER_MIN_LR,
    )

    best_val_loss = float("inf")
    best_epoch = 0

    header = (
        f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'Val RMSE':>10}  {'LR':>10}  {'':>6}"
    )
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                val_loss += criterion(model(X_batch), Y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Checkpoint ──
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT)
            marker = "★ best"

        # ── Logging ──
        if epoch <= 10 or epoch % 10 == 0 or marker:
            print(
                f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
                f"{val_loss**0.5:>10.6f}  {current_lr:>10.2e}  {marker}"
            )

    print(f"{'='*70}")
    print(f"\n✓ Training complete.")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best val MSE : {best_val_loss:.6f}  |  RMSE : {best_val_loss**0.5:.6f}")
    print(f"  Weights saved → {CHECKPOINT}")
    return best_val_loss


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train HestonSurrogateMLP")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help=f"Number of epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Device] {device}"
        + (f" — {torch.cuda.get_device_name(0)}" if device.type == "cuda" else "")
    )

    # Create output directories
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data via data_loader (also saves scalers)
    print(f"\n[Data] Loading from {DATA_PATH.name} …")
    train_loader, val_loader = get_dataloaders(
        filepath=str(DATA_PATH),
        batch_size=args.batch_size,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        scalers_dir=SCALERS_DIR,
    )
    print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"[Data] Scalers saved → {SCALERS_DIR}/")

    # Build model
    model = HestonSurrogateMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] HestonSurrogateMLP — {n_params:,} parameters")

    # Train
    t0 = time.time()
    train(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)
    elapsed = time.time() - t0
    print(f"  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
