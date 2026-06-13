"""
train_fno_differential.py — Differential ML training for the FiLM-FNO.

Trains the FiLM-conditioned FNO with a dual loss:
  L_total = L_iv + λ · L_jacobian

where:
  L_iv       = ATM-weighted Huber loss on IV surface predictions
  L_jacobian = MSE on ∂IV/∂params predictions (Huge & Savine, 2020)
  λ           = jacobian_weight (default 1.0, tunable)

Architecture:
  The same MirrorPaddedFNO2d predicts the IV surface.
  A separate lightweight MLP head attached to the parameter encoder
  predicts ∂IV/∂params.  Both heads share the same FiLM-encoded features.

Why differential training improves results (Huge & Savine, 2020):
  The Jacobian at a point (θ, IV(θ)) defines a tangent plane that locally
  constrains the function to a hyperplane in output space.  Each sample
  effectively contributes 1 + nT*nK*5 = 441 constraints instead of 1.
  Empirically this produces ~10× lower RMSE for the same number of samples.

Usage:
    python src/train_fno_differential.py [--epochs 500] [--lambda-jac 1.0]

Saves:
    artifacts/models/fno_diff_best.pth         — best validation checkpoint
    artifacts/weights/fno_diff_final_prod.pth  — SWA-averaged final model
    artifacts/models/param_normalizer_diff.npz
    artifacts/models/iv_normalizer_diff.npz
    artifacts/models/jac_normalizer_diff.npz   — per-param Jacobian statistics
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fno_model import (MirrorPaddedFNO2d, MirrorPaddedFNO2dWithAttention,
                       arbitrage_free_regularization)
from normalizers import ParameterNormalizer, IVSurfaceNormalizer


# ---------------------------------------------------------------------------
# Jacobian normalizer
# ---------------------------------------------------------------------------

class JacobianNormalizer:
    """
    Per-parameter z-score normalisation of the Jacobian ∂IV/∂params.

    The Jacobian columns have very different scales:
      ∂IV/∂theta  ~ O(10)    (theta controls long-run variance level)
      ∂IV/∂kappa  ~ O(0.01)  (kappa is nearly ghost parameter)
    Normalising prevents the large-scale columns from dominating L_jacobian.
    """

    def __init__(self):
        self.mean = None   # (5,)
        self.std  = None   # (5,)

    def fit(self, jac: np.ndarray) -> 'JacobianNormalizer':
        """jac: (N, nT, nK, 5) float32"""
        flat = jac.reshape(-1, 5)
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std  = flat.std(axis=0).clip(min=1e-6).astype(np.float32)
        return self

    def transform(self, jac: np.ndarray) -> np.ndarray:
        return ((jac - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def summary(self) -> str:
        lines = ['JacobianNormalizer (μ ± σ per param):']
        names = ['kappa', 'theta', 'sigma', 'rho', 'v0']
        for i, name in enumerate(names):
            lines.append(f'  ∂IV/∂{name:5s}: μ={self.mean[i]:+.4f}  σ={self.std[i]:.4f}')
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def iv_loss(pred: torch.Tensor, target: torch.Tensor,
            K_grid: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    """ATM-weighted Huber loss on z-score normalised IV surfaces (B, 8, 11)."""
    atm_w  = 1.0 + (K_grid.abs() < 0.1).float().view(1, 1, 11)
    huber  = F.huber_loss(pred, target, reduction='none', delta=delta)
    return (atm_w * huber).mean()


# T-grid for masking short-maturity Jacobians (index 0 = T=0.1 is noisiest)
_T_JAC_MASK = torch.ones(8, dtype=torch.float32)   # weight per maturity
_T_JAC_MASK[0] = 0.0    # mask out T=0.1 (worst FD accuracy, highest NaN rate)


def jac_loss(pred_j: torch.Tensor, target_j: torch.Tensor,
             nan_mask: torch.Tensor) -> torch.Tensor:
    """
    MSE on normalised Jacobian (B, 8, 11, 5), masked by:
      1. nan_mask — only valid (non-interpolated) cells
      2. _T_JAC_MASK — excludes T=0.1 (noisy FD at very short maturities)
    """
    device = pred_j.device
    t_mask = _T_JAC_MASK.to(device).view(1, 8, 1, 1)  # (1, 8, 1, 1)
    valid  = nan_mask.unsqueeze(-1).float() * t_mask   # (B, 8, 11, 1)
    n_valid = valid.sum().clamp(min=1.0)
    err    = (pred_j - target_j) ** 2 * valid          # (B, 8, 11, 5)
    return err.sum() / n_valid


# ---------------------------------------------------------------------------
# Jacobian prediction head (lightweight MLP on top of FiLM features)
# ---------------------------------------------------------------------------

class JacobianHead(torch.nn.Module):
    """
    Predicts ∂IV/∂params surface from the normalised parameter vector.

    Architecture: 6 → 256 → 256 → 8×11×5 (= 440 outputs)
    Separate from the FNO: the Jacobian depends primarily on θ globally
    (not on the local (T,K) structure), making an MLP appropriate here.

    In inference: predict both IV (via FNO) and ∂IV/∂θ (via this head),
    then pass the Jacobian to the Newton-Raphson calibration loop.
    """

    def __init__(self, n_params: int = 6, n_out: int = 440,
                 hidden: int = 256, dropout: float = 0.20):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_params, hidden),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, n_out),
        )
        self.n_out = n_out

    def forward(self, theta_n: torch.Tensor) -> torch.Tensor:
        """theta_n: (B, 6) normalised → (B, 8, 11, 5) Jacobian"""
        out = self.net(theta_n)               # (B, 440)
        return out.view(-1, 8, 11, 5)


# ---------------------------------------------------------------------------
# Combined FNO + Jacobian Head model
# ---------------------------------------------------------------------------

class DifferentialFNO(torch.nn.Module):
    """
    Wraps FiLM-FNO + JacobianHead.
    forward() returns (iv_pred, jac_pred).
    """

    def __init__(self):
        super().__init__()
        self.fno = MirrorPaddedFNO2dWithAttention()  # attention model for differential training
        self.jac_head = JacobianHead()

    def forward(self, coords: torch.Tensor,
                theta_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        iv_pred  = self.fno(coords, theta_n)          # (B, 8, 11)
        jac_pred = self.jac_head(theta_n)             # (B, 8, 11, 5)
        return iv_pred, jac_pred


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_differential(
    epochs: int          = 500,
    batch_size: int      = 512,
    lr: float            = 1e-3,
    lambda_jac: float    = 1.0,
    data_path: str       = 'data/DeepRoughDataset_v3_differential.npz',
) -> None:

    if not os.path.exists(data_path):
        print(f'Dataset not found: {data_path}')
        print('Run generate_dataset_v3_differential.py first.')
        return

    print(f'Loading dataset from {data_path} ...')
    npz      = np.load(data_path)
    data     = npz['dataset']                         # (N, 94)
    jacobian = npz['jacobian']                        # (N, 8, 11, 5)
    nan_mask = npz['nan_mask'] if 'nan_mask' in npz else None

    print(f'  Dataset  shape : {data.shape}')
    print(f'  Jacobian shape : {jacobian.shape}')
    if nan_mask is not None:
        print(f'  Valid cells    : {nan_mask.mean()*100:.2f}%')

    X_raw = data[:, :6]                               # (N, 6)
    Y_raw = np.clip(data[:, 6:], 1e-4, None).reshape(-1, 8, 11)  # (N, 8, 11)
    J_raw = jacobian                                   # (N, 8, 11, 5)

    # Train / val split
    split = int(0.8 * len(X_raw))
    X_tr, X_va = X_raw[:split], X_raw[split:]
    Y_tr, Y_va = Y_raw[:split], Y_raw[split:]
    J_tr, J_va = J_raw[:split], J_raw[split:]
    if nan_mask is not None:
        M_tr = nan_mask[:split].astype(np.float32)
        M_va = nan_mask[split:].astype(np.float32)
    else:
        M_tr = np.ones((len(X_tr), 8, 11), dtype=np.float32)
        M_va = np.ones((len(X_va), 8, 11), dtype=np.float32)

    # Normalizers (fit on training split only)
    print('Fitting normalizers ...')
    param_norm = ParameterNormalizer().fit(X_tr)
    iv_norm    = IVSurfaceNormalizer().fit(Y_tr)
    jac_norm   = JacobianNormalizer().fit(J_tr)
    print(param_norm.summary())
    print(iv_norm.summary())
    print(jac_norm.summary())

    os.makedirs('artifacts/models',  exist_ok=True)
    os.makedirs('artifacts/weights', exist_ok=True)
    param_norm.save('artifacts/models/param_normalizer_diff.npz')
    iv_norm.save('artifacts/models/iv_normalizer_diff.npz')
    jac_norm.save('artifacts/models/jac_normalizer_diff.npz')

    X_tr_n = param_norm.transform(X_tr).astype(np.float32)
    X_va_n = param_norm.transform(X_va).astype(np.float32)
    Y_tr_n = iv_norm.transform(Y_tr).astype(np.float32)
    Y_va_n = iv_norm.transform(Y_va).astype(np.float32)
    J_tr_n = jac_norm.transform(J_tr).astype(np.float32)
    J_va_n = jac_norm.transform(J_va).astype(np.float32)

    # Coordinate grids
    T_raw  = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0], dtype=np.float32)
    K_raw  = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    T_norm = (T_raw - T_raw.mean()) / T_raw.std()
    K_norm = K_raw / 0.5
    T_mesh, K_mesh = np.meshgrid(T_norm, K_norm, indexing='ij')
    coord_field = np.stack([T_mesh, K_mesh], axis=-1).astype(np.float32)

    N_tr, N_va = len(X_tr_n), len(X_va_n)
    coord_tr = np.broadcast_to(coord_field, (N_tr, 8, 11, 2)).copy()
    coord_va = np.broadcast_to(coord_field, (N_va, 8, 11, 2)).copy()

    tr_ds = TensorDataset(
        torch.tensor(X_tr_n),    # (N_tr, 6)   — params normalised
        torch.tensor(Y_tr_n),    # (N_tr, 8, 11)
        torch.tensor(J_tr_n),    # (N_tr, 8, 11, 5)
        torch.tensor(coord_tr),  # (N_tr, 8, 11, 2)
        torch.tensor(M_tr),      # (N_tr, 8, 11) nan_mask
    )
    va_ds = TensorDataset(
        torch.tensor(X_va_n),
        torch.tensor(Y_va_n),
        torch.tensor(J_va_n),
        torch.tensor(coord_va),
        torch.tensor(M_va),
    )
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining on {device}')

    model     = DifferentialFNO().to(device)
    swa_model = AveragedModel(model)
    n_p       = sum(p.numel() for p in model.parameters())
    print(f'DifferentialFNO parameters: {n_p:,}')
    print(f'  FNO:         {sum(p.numel() for p in model.fno.parameters()):,}')
    print(f'  JacHead:     {sum(p.numel() for p in model.jac_head.parameters()):,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    swa_start = int(epochs * 0.75)
    swa_sched = SWALR(optimizer, swa_lr=lr * 0.1)

    T_dev   = torch.tensor(T_raw, dtype=torch.float32, device=device)
    K_dev   = torch.tensor(K_raw, dtype=torch.float32, device=device)
    iv_mean = torch.tensor(iv_norm.mean, dtype=torch.float32, device=device)
    iv_std  = torch.tensor(iv_norm.std,  dtype=torch.float32, device=device)

    best_val = float('inf')
    print(f'\nλ_jacobian = {lambda_jac}')
    print('Starting training ...\n')

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_iv = tr_loss_jac = 0.0
        t0 = time.time()

        for theta_n, iv_n, jac_n, coords, mask in tr_dl:
            theta_n = theta_n.to(device)
            iv_n    = iv_n.to(device)
            jac_n   = jac_n.to(device)
            coords  = coords.to(device)
            mask    = mask.to(device)

            optimizer.zero_grad()
            iv_pred, jac_pred = model(coords, theta_n)

            # IV loss (nan_mask weighted, ATM upweighted)
            cell_w = 0.5 + 0.5 * mask
            atm_w  = 1.0 + (K_dev.abs() < 0.1).float().view(1, 1, 11)
            huber  = F.huber_loss(iv_pred, iv_n, reduction='none', delta=0.05)
            l_iv   = (cell_w * atm_w * huber).mean()

            # Jacobian loss (valid cells only)
            l_jac  = jac_loss(jac_pred, jac_n, mask)

            # Arbitrage regularization — do NOT clamp iv_real before computing
            # the penalty. Clamping kills gradients for cells where iv_pred<0,
            # removing the signal that should push them back to valid values.
            # Clamp is applied only at inference time, not during training.
            iv_real = iv_pred * iv_std + iv_mean    # (B, 8, 11) real space, UN-clamped
            l_arb   = arbitrage_free_regularization(iv_real, T_dev, K_dev)

            loss = l_iv + lambda_jac * l_jac + 0.05 * l_arb
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss_iv  += l_iv.item()  * theta_n.size(0)
            tr_loss_jac += l_jac.item() * theta_n.size(0)

        tr_loss_iv  /= N_tr
        tr_loss_jac /= N_tr

        # Validation
        model.eval()
        va_iv_loss = va_jac_loss = 0.0
        with torch.no_grad():
            for theta_n, iv_n, jac_n, coords, mask in va_dl:
                theta_n  = theta_n.to(device)
                iv_n     = iv_n.to(device)
                jac_n    = jac_n.to(device)
                coords   = coords.to(device)
                mask     = mask.to(device)
                iv_pred, jac_pred = model(coords, theta_n)
                # BUG FIX: use the same masked jac_loss as in training
                va_iv_loss  += F.mse_loss(iv_pred, iv_n).item()         * theta_n.size(0)
                va_jac_loss += jac_loss(jac_pred, jac_n, mask).item()   * theta_n.size(0)
        va_iv_loss  /= N_va
        va_jac_loss /= N_va
        va_combined  = va_iv_loss + lambda_jac * va_jac_loss

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        print(f'Ep {epoch:03d}/{epochs} | '
              f'IV {tr_loss_iv:.5f}/{va_iv_loss:.5f} | '
              f'Jac {tr_loss_jac:.5f}/{va_jac_loss:.5f} | '
              f'{elapsed:.1f}s')

        if va_combined < best_val and epoch <= swa_start:
            best_val = va_combined
            torch.save(model.state_dict(), 'artifacts/models/fno_diff_best.pth')
            print(f'  → New best combined: {best_val:.6f}')

    print(f'\nTraining complete. Best val combined loss: {best_val:.6f}')
    torch.optim.swa_utils.update_bn(tr_dl, swa_model, device=device)
    torch.save(swa_model.module.state_dict(),
               'artifacts/weights/fno_diff_final_prod.pth')
    print('SWA model saved to artifacts/weights/fno_diff_final_prod.pth')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch-size', type=int,   default=512)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--lambda-jac', type=float, default=0.05,
                        help='Weight for Jacobian loss term (default 0.05 to prevent Jac overfitting)')
    parser.add_argument('--data-path',  type=str,
                        default='data/DeepRoughDataset_v3_differential.npz')
    args = parser.parse_args()

    train_differential(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_jac=args.lambda_jac,
        data_path=args.data_path,
    )
