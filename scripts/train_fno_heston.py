"""
train_fno_heston.py — Train FiLM-conditioned FNO on Classic Heston implied volatility dataset.

Trains MirrorPaddedFNO2d with param_dim=5 (kappa, theta, sigma, rho, v0) using Huber loss.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from src.normalizers import ParameterNormalizerHeston, IVSurfaceNormalizer

# --- Config ---
DATASET_PATH = 'data/HestonDataset_v1.npz'
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_PARAMS = 5  # [kappa, theta, sigma, rho, v0]
EPOCHS = 500
BATCH_SIZE = 512
LR = 3e-4
SWA_START = 400

WEIGHTS_BEST = 'artifacts/weights/fno_heston_best.pth'
WEIGHTS_PROD = 'artifacts/weights/fno_heston_final_prod.pth'
NORM_PARAM = 'artifacts/models/param_normalizer_heston.npz'
NORM_IV = 'artifacts/models/iv_normalizer_heston.npz'


def _make_spatial_input(T_grid, K_grid, device):
    T_norm = (T_grid - T_grid.mean()) / T_grid.std()
    K_norm = (K_grid - K_grid.mean()) / K_grid.std()
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing='ij')
    spatial = np.stack([T_mesh, K_mesh], axis=-1)[None]  # (1, nT, nK, 2)
    return torch.tensor(spatial, dtype=torch.float32, device=device)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training Heston FNO on: {device}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run scripts/generate_dataset_heston.py first.")

    data = np.load(DATASET_PATH)
    params = data['params']  # (N, 5): [kappa, theta, sigma, rho, v0]
    iv = data['iv']          # (N, 8, 11)
    
    N = params.shape[0]
    print(f"  Loaded dataset with {N:,} samples.")

    # Train/val split (80/20)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    split = int(0.8 * N)
    tr_idx = idx[:split]
    va_idx = idx[split:]

    X_tr = params[tr_idx]
    Y_tr = iv[tr_idx]
    X_va = params[va_idx]
    Y_va = iv[va_idx]

    # Load normalizers if they exist, otherwise fit and save them
    if os.path.exists(NORM_PARAM) and os.path.exists(NORM_IV):
        print("  Loading pre-fit normalizers...")
        param_norm = ParameterNormalizerHeston.load(NORM_PARAM)
        iv_norm = IVSurfaceNormalizer.load(NORM_IV)
    else:
        print("  Normalizers not found. Fitting on train split...")
        param_norm = ParameterNormalizerHeston().fit(X_tr)
        iv_norm = IVSurfaceNormalizer().fit(Y_tr)
        os.makedirs(os.path.dirname(NORM_PARAM), exist_ok=True)
        param_norm.save(NORM_PARAM)
        iv_norm.save(NORM_IV)
        print(f"  Saved normalizers to {NORM_PARAM} and {NORM_IV}")

    # Transform data
    X_tr_n = param_norm.transform(X_tr)
    Y_tr_n = iv_norm.transform(Y_tr)
    X_va_n = param_norm.transform(X_va)
    Y_va_n = iv_norm.transform(Y_va)

    tr_ds = TensorDataset(torch.tensor(X_tr_n, dtype=torch.float32),
                          torch.tensor(Y_tr_n, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_va_n, dtype=torch.float32),
                          torch.tensor(Y_va_n, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize FNO Model (param_dim=5 for classic Heston)
    model = MirrorPaddedFNO2d(param_dim=N_PARAMS).to(device)
    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    swa_mod = AveragedModel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_sched = SWALR(optimizer, swa_lr=LR * 0.1)

    t_grid_tensor = torch.tensor(T_GRID, dtype=torch.float32, device=device)
    k_grid_tensor = torch.tensor(K_GRID, dtype=torch.float32, device=device)

    best_val = float('inf')
    best_ep = 0
    t0 = time.time()

    print(f"\n  {'Epoch':>5} | {'Train Huber':>12} | {'Val Huber':>10} | {'Time':>7}")
    print("  " + "-" * 48)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for X_b, Y_b in tr_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)
            
            sp = spatial.expand(B, -1, -1, -1)
            pred = model(sp, X_b)  # (B, nT, nK)
            
            # Huber Loss + Arbitrage Regularization
            loss = F.huber_loss(pred, Y_b, delta=1.0)
            loss = loss + 1e-4 * arbitrage_free_regularization(pred, t_grid_tensor, k_grid_tensor)
            
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
                va_loss += F.huber_loss(pred, Y_b, delta=1.0).item() * B
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

        if ep % 20 == 0 or ep == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  {ep:>5} | {tr_loss:>12.6f} | {va_loss:>10.6f} | {elapsed:>5.1f}min"
                  f"{'  ← best' if ep == best_ep else ''}")

    # Save SWA model weights
    torch.optim.swa_utils.update_bn(tr_dl, swa_mod, device=device)
    torch.save(swa_mod.module.state_dict(), WEIGHTS_PROD)
    print(f"\n  Best val loss : {best_val:.6f} @ epoch {best_ep}")
    print(f"  SWA model saved → {WEIGHTS_PROD}")
    print(f"  Best model saved  → {WEIGHTS_BEST}")
    print(f"  Total time        : {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    train()
