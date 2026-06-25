"""
train_fno.py — End-to-End Training of Arbitrage-Free FNO Surrogate.

Wraps the MirrorPaddedFNO2d inside the ArbitrageFreeFNO wrapper and trains
end-to-end. Since the projection layer guarantees 100% arbitrage-free surfaces,
the soft arbitrage penalty weight is set to 0.0 (optimizing only Huber loss).
Supports CUDA execution and mixed-precision boundaries.
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

from deepvol.arbitrage.projection_layer import ArbitrageFreeFNO, DifferentiableArbitrageFreeProjection
from deepvol.surrogates.fno_model import MirrorPaddedFNO2d, arbitrage_free_regularization
from deepvol.surrogates.normalizers import ParameterNormalizerHeston, IVSurfaceNormalizer

# --- Config ---
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "data/HestonDataset_v1.npz" if os.path.exists("data/HestonDataset_v1.npz")
    else "/home/execorn/programming/derivatives/data/HestonDataset_v1.npz"
)
T_GRID = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
K_GRID = np.linspace(-0.5, 0.5, 11)
N_PARAMS = 5  # [kappa, theta, sigma, rho, v0]
EPOCHS = int(os.environ.get("EPOCHS", 3 if '--smoke' in sys.argv else 150))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
LR = float(os.environ.get("LR", 8e-4))
SWA_START = int(os.environ.get("SWA_START", 2 if '--smoke' in sys.argv else 120))

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
    print(f"  Training End-to-End Arbitrage-Free Heston FNO on: {device}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Please ensure dataset path is correct.")

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

    # Initialize Base FNO Model (param_dim=5 for classic Heston)
    base_fno = MirrorPaddedFNO2d(param_dim=N_PARAMS).to(device)
    
    # Initialize projection layer
    projection_layer = DifferentiableArbitrageFreeProjection(
        T_grid=T_GRID,
        K_grid=K_GRID,
        S0=1.0,
        is_log_moneyness=True
    ).to(device)

    # Wrap the model inside ArbitrageFreeFNO
    wrapped_model = ArbitrageFreeFNO(
        base_fno=base_fno,
        projection_layer=projection_layer,
        normalizer=iv_norm
    ).to(device)

    spatial = _make_spatial_input(T_GRID, K_GRID, device)
    
    # AveragedModel for SWA
    swa_mod = AveragedModel(wrapped_model)

    optimizer = optim.AdamW(wrapped_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_sched = SWALR(optimizer, swa_lr=LR * 0.1)

    os.makedirs(os.path.dirname(WEIGHTS_BEST), exist_ok=True)

    best_val = float('inf')
    best_ep = 0
    t0 = time.time()

    print(f"\n  {'Epoch':>5} | {'Train Huber':>12} | {'Val Huber':>10} | {'Time':>7}")
    print("  " + "-" * 48)

    for ep in range(1, EPOCHS + 1):
        wrapped_model.train()
        tr_loss = 0.0
        for X_b, Y_b in tr_dl:
            X_b = X_b.to(device)
            Y_b = Y_b.to(device)
            B = X_b.size(0)
            
            sp = spatial.expand(B, -1, -1, -1)
            
            optimizer.zero_grad(set_to_none=True)
            pred = wrapped_model(sp, X_b)
            loss = F.huber_loss(pred, Y_b, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
            optimizer.step()
            
            tr_loss += loss.item() * B
        tr_loss /= len(tr_ds)

        wrapped_model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X_b, Y_b in va_dl:
                X_b = X_b.to(device)
                Y_b = Y_b.to(device)
                B = X_b.size(0)
                sp = spatial.expand(B, -1, -1, -1)
                
                pred = wrapped_model(sp, X_b)
                loss_val = F.huber_loss(pred, Y_b, delta=1.0)
                va_loss += loss_val.item() * B
        va_loss /= len(va_ds)

        if ep >= SWA_START:
            swa_mod.update_parameters(wrapped_model)
            swa_sched.step()
        else:
            scheduler.step()

        if va_loss < best_val:
            best_val = va_loss
            best_ep = ep
            # Save the base FNO weights to WEIGHTS_BEST
            torch.save(wrapped_model.base_fno.state_dict(), WEIGHTS_BEST)

        if ep % 20 == 0 or ep == 1 or ep == EPOCHS:
            elapsed = (time.time() - t0) / 60
            print(f"  {ep:>5} | {tr_loss:>12.6f} | {va_loss:>10.6f} | {elapsed:>5.1f}min"
                  f"{'  ← best' if ep == best_ep else ''}")

    # Update BN for SWA and save weights
    if EPOCHS >= SWA_START:
        torch.optim.swa_utils.update_bn(tr_dl, swa_mod, device=device)
        # Save SWA base FNO weights to WEIGHTS_PROD
        torch.save(swa_mod.module.base_fno.state_dict(), WEIGHTS_PROD)
        print(f"  SWA model saved → {WEIGHTS_PROD}")
    else:
        # If SWA did not start, copy the best model to final prod
        if os.path.exists(WEIGHTS_BEST):
            state = torch.load(WEIGHTS_BEST)
            torch.save(state, WEIGHTS_PROD)
            print(f"  Best model copied to final prod → {WEIGHTS_PROD}")

    print(f"\n  Best val loss : {best_val:.6f} @ epoch {best_ep}")
    print(f"  Best model saved  → {WEIGHTS_BEST}")
    print(f"  Total time        : {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    train()
