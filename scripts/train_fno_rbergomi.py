"""
train_fno_rbergomi.py — Training script for Rough Bergomi FNO surrogate.
Trains a MirrorPaddedFNO2d (param_dim=4) using Huber loss.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from src.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization


# ─── Normalizers ──────────────────────────────────────────────────────────────
class ParameterNormalizer4D:
    """
    Z-score normalizer for 4-dimensional Rough Bergomi parameter vector [v0, H, eta, rho].
    """

    PARAM_NAMES = ["v0", "H", "eta", "rho"]

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "ParameterNormalizer4D":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return X_norm * self.std + self.mean

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - mean) / std

    def inverse_transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return t * std + mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "ParameterNormalizer4D":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std = data["std"]
        return n


class IVSurfaceNormalizer:
    """
    Per-grid-point z-score normalizer for the (8, 11) implied volatility surface.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, Y: np.ndarray) -> "IVSurfaceNormalizer":
        self.mean = Y.mean(axis=0)
        self.std = Y.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / self.std

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        return Y_norm * self.std + self.mean

    def transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - mean) / std

    def inverse_transform_tensor(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=t.dtype, device=t.device)
        std = torch.tensor(self.std, dtype=t.dtype, device=t.device)
        return t * std + mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "IVSurfaceNormalizer":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std = data["std"]
        return n


# ─── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(repo_root, "data", "rBergomiDataset_v1.npz")
EPOCHS = 3 if '--smoke' in sys.argv else 500
BATCH_SIZE = 512
LR = 3e-4
SWA_START = 2 if '--smoke' in sys.argv else 400

WEIGHTS_BEST = os.path.join(repo_root, "artifacts", "weights", "fno_rbergomi_best.pth")
WEIGHTS_PROD = os.path.join(repo_root, "artifacts", "weights", "fno_rbergomi_final_prod.pth")
NORM_PARAM = os.path.join(repo_root, "artifacts", "models", "param_normalizer_rbergomi.npz")
NORM_IV = os.path.join(repo_root, "artifacts", "models", "iv_normalizer_rbergomi.npz")


def _make_spatial_input(T_grid, K_grid, device):
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing="ij")
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, nT, nK, 2)
    return torch.tensor(spatial, dtype=torch.float32, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on: {device}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run scripts/generate_dataset_rbergomi.py first."
        )

    # 1. Load data
    data = np.load(DATASET_PATH)
    params_train = data["params_train"]
    iv_train = data["iv_train"]
    params_val = data["params_val"]
    iv_val = data["iv_val"]
    T_grid = data["T_grid"]
    K_grid = data["K_grid"]

    print(f"  Train: {params_train.shape[0]:,} samples, Val: {params_val.shape[0]:,} samples")

    # 2. Fit normalizers
    param_norm = ParameterNormalizer4D().fit(params_train)
    iv_norm = IVSurfaceNormalizer().fit(iv_train)

    os.makedirs(os.path.dirname(NORM_PARAM), exist_ok=True)
    os.makedirs(os.path.dirname(WEIGHTS_BEST), exist_ok=True)
    param_norm.save(NORM_PARAM)
    iv_norm.save(NORM_IV)
    print(f"  Normalizers saved to {NORM_PARAM} and {NORM_IV}")

    # Transform data
    X_train_n = param_norm.transform(params_train)
    Y_train_n = iv_norm.transform(iv_train)
    X_val_n = param_norm.transform(params_val)
    Y_val_n = iv_norm.transform(iv_val)

    # Datasets
    tr_ds = TensorDataset(
        torch.tensor(X_train_n, dtype=torch.float32),
        torch.tensor(Y_train_n, dtype=torch.float32),
    )
    va_ds = TensorDataset(
        torch.tensor(X_val_n, dtype=torch.float32),
        torch.tensor(Y_val_n, dtype=torch.float32),
    )
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 3. Model setup (param_dim=4)
    model = MirrorPaddedFNO2d(param_dim=4).to(device)
    spatial = _make_spatial_input(T_grid, K_grid, device)
    swa_mod = AveragedModel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_sched = SWALR(optimizer, swa_lr=LR * 0.1)

    t_grid_tensor = torch.tensor(T_grid, dtype=torch.float32, device=device)
    k_grid_tensor = torch.tensor(K_grid, dtype=torch.float32, device=device)

    best_val_mae = float("inf")
    best_ep = 0
    t0 = time.time()

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val MAE (bps)':>14} | {'Time':>7}")
    print("  " + "-" * 62)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for X_b, Y_b in tr_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)

            sp = spatial.expand(B, -1, -1, -1)
            pred = model(sp, X_b)

            # Huber loss
            loss_huber = F.huber_loss(pred, Y_b, delta=1.0)

            # Arbitrage penalty on denormalized predictions
            pred_denorm = iv_norm.inverse_transform_tensor(pred)
            loss_arb = arbitrage_free_regularization(pred_denorm, t_grid_tensor, k_grid_tensor)

            loss = loss_huber + 1e-4 * loss_arb

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * B

        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        va_mae = 0.0
        with torch.no_grad():
            for X_b, Y_b in va_dl:
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                B = X_b.size(0)

                sp = spatial.expand(B, -1, -1, -1)
                pred = model(sp, X_b)

                va_loss += F.huber_loss(pred, Y_b, delta=1.0).item() * B

                # Denormalize to calculate absolute MAE in IV units
                pred_iv = iv_norm.inverse_transform_tensor(pred)
                target_iv = iv_norm.inverse_transform_tensor(Y_b)
                mae_batch = torch.mean(torch.abs(pred_iv - target_iv))
                va_mae += mae_batch.item() * B

        va_loss /= len(va_ds)
        va_mae /= len(va_ds)
        va_mae_bps = va_mae * 10000.0  # 1 bp = 0.0001 (0.01% IV)

        if ep >= SWA_START:
            swa_mod.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            best_ep = ep
            torch.save(model.state_dict(), WEIGHTS_BEST)

        if ep % 20 == 0 or ep == 1:
            elapsed = (time.time() - t0) / 60
            print(
                f"  {ep:>5} | {tr_loss:>10.6f} | {va_loss:>10.6f} | {va_mae_bps:>10.2f} bps | {elapsed:>5.1f}min"
                f"{'  ← best' if ep == best_ep else ''}"
            )

    # ── Save SWA model ────────────────────────────────────────────────────────
    torch.optim.swa_utils.update_bn(tr_dl, swa_mod, device=device)
    torch.save(swa_mod.module.state_dict(), WEIGHTS_PROD)
    print(f"\n  Best Val MAE: {best_val_mae*10000.0:.2f} bps @ epoch {best_ep}")
    print(f"  SWA model saved to {WEIGHTS_PROD}")
    print(f"  Best model saved to {WEIGHTS_BEST}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
