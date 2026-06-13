"""
train_fno.py — Training pipeline for FiLM-conditioned Mirror-Padded FNO.

Three changes vs previous runs:
  1. Z-score normalisation of input parameters (ParameterNormalizer)
  2. Per-grid-point z-score normalisation of IV surfaces (IVSurfaceNormalizer)
  3. ATM-weighted Huber loss (δ=0.05, ATM weight 2×)

Both normalizers are fit on the training split and saved alongside weights.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from normalizers import ParameterNormalizer, IVSurfaceNormalizer


# ─── ATM-Weighted Huber Loss ──────────────────────────────────────────────────

def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor,
                        K_grid: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """
    Spatially weighted Huber loss over the (T, K) IV grid.

    Weighting rationale
    -------------------
    Skew (ρ) and smile curvature (σ) are primarily identified from the
    ATM region |K| < 0.1.  Upweighting ATM errors forces the network to
    resolve parameter-dependent geometry in the most informative region.

    Huber δ=0.05 (5% absolute IV) smoothly suppresses Monte Carlo outliers
    at the roughness boundary (T=0.1, deep OTM) without the exploding
    gradients of pure L1 or the outlier-sensitivity of pure L2.

    Parameters
    ----------
    pred, target : (B, 8, 11)  — in NORMALISED z-score space
    K_grid       : (11,)       — raw log-moneyness values
    delta        : Huber threshold

    Returns
    -------
    Scalar weighted Huber loss.
    """
    # Weight = 2.0 near ATM, 1.0 in wings
    atm_mask = (K_grid.abs() < 0.1).float()           # (11,)
    weights  = 1.0 + atm_mask.view(1, 1, 11)          # (1, 1, 11) — broadcast over B, T

    huber = F.huber_loss(pred, target, reduction='none', delta=delta)  # (B, 8, 11)
    return (weights * huber).mean()


# ─── Training function ────────────────────────────────────────────────────────

def train_fno(epochs: int = 500, batch_size: int = 1024, lr: float = 1e-3,
              data_path: str = "data/DeepRoughDataset_v2_fourier.npz") -> None:
    """
    Train the FiLM-conditioned Mirror-Padded FNO on the v2 Fourier-COS dataset.

    Key upgrade vs previous run:
      - Labels are exact Fourier-COS prices (no Monte-Carlo O(dt^0.08) bias)
      - nan_mask weighting: interpolated wing cells get 0.5× loss weight
        (K=±0.5/±0.3 at T=0.1 where option price is below float64 precision)

    Saves:
      artifacts/models/fno_v2_best.pth          — best validation checkpoint
      artifacts/weights/fno_v2_final_prod.pth   — SWA-averaged final model
      artifacts/models/param_normalizer_v2.npz  — ParameterNormalizer scalers
      artifacts/models/iv_normalizer_v2.npz     — IVSurfaceNormalizer scalers
    """
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Run generate_dataset_v2.py first.")
        return

    # ── Load raw data ──────────────────────────────────────────────────────
    print(f"Loading dataset from {data_path}...")
    npz      = np.load(data_path)
    data     = npz["dataset"]                              # (N, 94)
    nan_mask = npz["nan_mask"] if "nan_mask" in npz else None  # (N, 8, 11) bool
    print(f"  Dataset shape: {data.shape}  "
          f"({data.shape[0]} samples, {data.shape[1]-6} IV grid points)")
    if nan_mask is not None:
        valid_pct = nan_mask.mean() * 100
        print(f"  nan_mask valid fraction: {valid_pct:.2f}%  "
              f"(interpolated: {100-valid_pct:.2f}%)")

    X_raw = data[:, :6]                               # (N, 6)
    Y_raw = np.clip(data[:, 6:], 1e-4, None)          # (N, 88)  — clip negatives
    Y_raw = Y_raw.reshape(-1, 8, 11)                  # (N, 8, 11)

    # ── Train/val split BEFORE fitting normalizers ─────────────────────────
    split  = int(0.8 * len(X_raw))
    X_train_raw, X_val_raw = X_raw[:split], X_raw[split:]
    Y_train_raw, Y_val_raw = Y_raw[:split], Y_raw[split:]
    if nan_mask is not None:
        mask_train = nan_mask[:split].astype(np.float32)   # 1.0=valid, 0=interp
        mask_val   = nan_mask[split:].astype(np.float32)
    else:
        mask_train = np.ones((len(X_train_raw), 8, 11), dtype=np.float32)
        mask_val   = np.ones((len(X_val_raw),   8, 11), dtype=np.float32)

    # ── Fit and save normalizers on training split ONLY ───────────────────
    print("Fitting normalizers on training split...")
    param_norm = ParameterNormalizer().fit(X_train_raw)
    iv_norm    = IVSurfaceNormalizer().fit(Y_train_raw)
    print(param_norm.summary())
    print(iv_norm.summary())

    os.makedirs("artifacts/models",  exist_ok=True)
    os.makedirs("artifacts/weights", exist_ok=True)
    param_norm.save("artifacts/models/param_normalizer_v2.npz")
    iv_norm.save("artifacts/models/iv_normalizer_v2.npz")
    print("Normalizers saved (v2).")

    # ── Normalise inputs and outputs ──────────────────────────────────────
    X_train = param_norm.transform(X_train_raw).astype(np.float32)
    X_val   = param_norm.transform(X_val_raw).astype(np.float32)
    Y_train = iv_norm.transform(Y_train_raw).astype(np.float32)
    Y_val   = iv_norm.transform(Y_val_raw).astype(np.float32)

    # ── Build coordinate grids (normalised to [-1, 1]) ───────────────────
    T_raw   = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    K_raw   = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    T_norm  = (T_raw - T_raw.mean()) / T_raw.std()    # (8,)
    K_norm  = K_raw / 0.5                             # (11,) already in [-1,1]

    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing="ij")  # (8,11) each
    # Spatial input per sample: (8, 11, 2)
    coord_field = np.stack([T_mesh, K_mesh], axis=-1).astype(np.float32)  # (8, 11, 2)

    # ── DataLoaders ───────────────────────────────────────────────────────
    N_train = len(X_train)
    N_val   = len(X_val)

    coord_train = np.broadcast_to(coord_field, (N_train, 8, 11, 2)).copy()
    coord_val   = np.broadcast_to(coord_field, (N_val,   8, 11, 2)).copy()

    train_ds = TensorDataset(
        torch.tensor(X_train),       # theta (normalised)
        torch.tensor(Y_train),       # IV surface (normalised)
        torch.tensor(coord_train),   # spatial coords
        torch.tensor(mask_train),    # nan_mask (1=valid COS, 0=interpolated)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(Y_val),
        torch.tensor(coord_val),
        torch.tensor(mask_val),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Model, optimiser, scheduler ──────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model     = MirrorPaddedFNO2d().to(device)
    swa_model = AveragedModel(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"FiLM-FNO parameters: {n_params:,}")

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    swa_start  = int(epochs * 0.75)
    swa_sched  = SWALR(optimizer, swa_lr=lr * 0.1)

    # Grids for arbitrage regularization (real IV space needs T_grid)
    T_grid_dev = torch.tensor(T_raw, dtype=torch.float32, device=device)
    K_grid_dev = torch.tensor(K_raw, dtype=torch.float32, device=device)
    iv_mean_t  = torch.tensor(iv_norm.mean, dtype=torch.float32, device=device)
    iv_std_t   = torch.tensor(iv_norm.std,  dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    print("Starting Training Loop...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for theta_n, iv_n, coords, mask in train_loader:
            theta_n = theta_n.to(device)
            iv_n    = iv_n.to(device)
            coords  = coords.to(device)
            mask    = mask.to(device)   # (B, 8, 11) 1.0=valid / 0.5=interpolated

            # Interpolated cells get half the loss weight
            cell_weights = 0.5 + 0.5 * mask             # (B, 8, 11): 0.5 or 1.0

            optimizer.zero_grad()
            pred_n = model(coords, theta_n)              # (B, 8, 11)

            # ATM-weighted Huber, further modulated by nan_mask cell weights
            atm_mask  = (K_grid_dev.abs() < 0.1).float()
            atm_w     = 1.0 + atm_mask.view(1, 1, 11)   # 1.0 or 2.0
            combined_w = cell_weights * atm_w            # (B, 8, 11)

            huber = F.huber_loss(pred_n, iv_n, reduction='none', delta=0.05)
            huber_loss = (combined_w * huber).mean()

            # Arbitrage regularization in real IV space.
            # Do NOT clamp pred_real before the penalty: clamping zeros gradients
            # for cells where pred<0, removing the corrective signal.
            # Clamp only at inference time (calibrate.py).
            pred_real = pred_n * iv_std_t + iv_mean_t    # UN-clamped real IV
            arb = arbitrage_free_regularization(pred_real, T_grid_dev, K_grid_dev)

            loss = huber_loss + 0.05 * arb
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * theta_n.size(0)

        train_loss /= len(train_ds)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for theta_n, iv_n, coords, mask in val_loader:
                theta_n = theta_n.to(device)
                iv_n    = iv_n.to(device)
                coords  = coords.to(device)
                pred_n  = model(coords, theta_n)
                val_loss += F.mse_loss(pred_n, iv_n).item() * theta_n.size(0)
        val_loss /= len(val_ds)

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"Time: {elapsed:.2f}s")

        if val_loss < best_val_loss and epoch <= swa_start:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "artifacts/models/fno_v2_best.pth")
            print(f"  → New best: {best_val_loss:.6f} (saved)")

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.6f}")
    torch.optim.swa_utils.update_bn(train_loader, swa_model,
                                    device=device)
    torch.save(swa_model.module.state_dict(), "artifacts/weights/fno_v2_final_prod.pth")
    print("SWA Model saved to artifacts/weights/fno_v2_final_prod.pth")


if __name__ == "__main__":
    train_fno(epochs=500, batch_size=1024)
