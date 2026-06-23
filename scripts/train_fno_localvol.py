"""
train_fno_localvol.py — Train FNO surrogate for Local Volatility mapping.

Trains MirrorPaddedFNO2d with param_dim=40 (8 slices * 5 SVI params) to map SVI parameters
directly to the (8, 11) Dupire Local Volatility surface.
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

# Ensure project root is on PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fno_model import MirrorPaddedFNO2d

# Config
DATASET_PATH = 'data/LocalVolDataset_v1.npz'
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_PARAMS = 40  # 8 slices * 5 SVI params
EPOCHS = 3 if '--smoke' in sys.argv else 150
BATCH_SIZE = 4096
LR = 8e-4
SWA_START = 2 if '--smoke' in sys.argv else 120

WEIGHTS_BEST = 'artifacts/weights/fno_localvol_best.pth'
WEIGHTS_PROD = 'artifacts/weights/fno_localvol_final_prod.pth'
NORM_PARAM = 'artifacts/models/param_normalizer_localvol.npz'
NORM_LV = 'artifacts/models/lv_normalizer_localvol.npz'


class ParameterNormalizer:
    """Z-score normalizer for the 40-dimensional SVI parameter vector."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> "ParameterNormalizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "ParameterNormalizer":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std = data["std"]
        return n


class LVSurfaceNormalizer:
    """Per-grid-point z-score normalizer for the (8, 11) local volatility surface."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, Y: np.ndarray) -> "LVSurfaceNormalizer":
        self.mean = Y.mean(axis=0)
        self.std = Y.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return (Y - self.mean) / self.std

    def inverse_transform(self, Y_norm: np.ndarray) -> np.ndarray:
        return Y_norm * self.std + self.mean

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "LVSurfaceNormalizer":
        data = np.load(path)
        n = cls()
        n.mean = data["mean"]
        n.std = data["std"]
        return n


def _make_spatial_input(T_grid, K_grid, device):
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing='ij')
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, nT, nK, 2)
    return torch.tensor(spatial, dtype=torch.float32, device=device)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training Local Vol FNO on: {device}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run generate_dataset_localvol.py first.")

    data = np.load(DATASET_PATH)
    params = data['params']  # (N, 40)
    lv = data['lv']          # (N, 8, 11)

    N = params.shape[0]
    print(f"  Loaded dataset with {N:,} samples.")

    # Train/val split (80/20)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, Y_tr = params[tr_idx], lv[tr_idx]
    X_va, Y_va = params[va_idx], lv[va_idx]

    # Normalizers
    param_norm = ParameterNormalizer().fit(X_tr)
    lv_norm = LVSurfaceNormalizer().fit(Y_tr)
    
    os.makedirs('artifacts/models', exist_ok=True)
    os.makedirs('artifacts/weights', exist_ok=True)
    param_norm.save(NORM_PARAM)
    lv_norm.save(NORM_LV)
    print(f"  Normalizers saved to {NORM_PARAM} and {NORM_LV}")

    # Transform data
    X_tr_n = param_norm.transform(X_tr)
    Y_tr_n = lv_norm.transform(Y_tr)
    X_va_n = param_norm.transform(X_va)
    Y_va_n = lv_norm.transform(Y_va)

    tr_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                          torch.tensor(Y_tr_n, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_va_n, dtype=torch.float32),
                          torch.tensor(Y_va_n, dtype=torch.float32))
                          
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Model definition with param_dim=40
    model = MirrorPaddedFNO2d(param_dim=N_PARAMS).to(device)
    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    swa_mod = AveragedModel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_sched = SWALR(optimizer, swa_lr=LR * 0.1)

    best_val = float('inf')
    best_ep = 0
    t0 = time.time()

    print(f"\n  {'Epoch':>5} | {'Train MSE':>11} | {'Val MSE':>11} | {'Time':>7}")
    print("  " + "-" * 48)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for X_b, Y_b in tr_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)
            
            sp = spatial.expand(B, -1, -1, -1)
            pred = model(sp, X_b)  # (B, 8, 11)
            
            # Basic MSE Loss in normalized space
            loss = F.mse_loss(pred, Y_b)
            
            # Optional: Add small penalty for negative local volatility in denormalized space
            # but since normalized targets are mean-zero, we enforce positivity after inference
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tr_loss += loss.item() * B
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in va_dl:
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                B = X_b.size(0)
                sp = spatial.expand(B, -1, -1, -1)
                pred = model(sp, X_b)
                va_loss += F.mse_loss(pred, Y_b).item() * B
        va_loss /= len(va_ds)

        if ep >= SWA_START:
            swa_mod.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        if va_loss < best_val:
            best_val = va_loss
            best_ep = ep
            torch.save(model.state_dict(), WEIGHTS_BEST)

        if ep % 10 == 0 or ep == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  {ep:>5} | {tr_loss:>11.6f} | {va_loss:>11.6f} | {elapsed:>5.1f}m"
                  f"{'  ← best' if ep == best_ep else ''}")

    # Save SWA model
    torch.optim.swa_utils.update_bn(tr_dl, swa_mod, device=device)
    torch.save(swa_mod.module.state_dict(), WEIGHTS_PROD)
    
    print(f"\n  Training completed. Best val MSE: {best_val:.6f} @ epoch {best_ep}")
    print(f"  Best model saved → {WEIGHTS_BEST}")
    print(f"  SWA model saved  → {WEIGHTS_PROD}")
    print(f"  Total time       : {(time.time() - t0) / 60:.2f} min")


if __name__ == '__main__':
    train()
